"""Test Sprint 18 - Signed Snapshot Pipeline and Testnet Preflight Integration.

Bao phu:
1. build_signed_get_request spot: URL dung testnet endpoint, params co timestamp/recvWindow/signature, header dung.
2. build_signed_get_request futures: URL testnet binancefuture, method GET, header dung.
3. Chu ky GET khop vector HMAC-SHA256 chuan.
4. Thieu credentials -> build_signed_get_request raise BinanceCredentialsMissingError (fail-closed).
5. fetch_exchange_reconcile_snapshot voi mock chuan: snapshot_healthy is True, du cac truong spot/perp, equity USD.
6. fetch_exchange_reconcile_snapshot khong co credentials va khong co mock -> snapshot_healthy is False (khong raise).
7. fetch_exchange_reconcile_snapshot voi mock sai dinh dang -> snapshot_healthy is False.
8. Tich hop: snapshot mock hop le -> reconcile SYNCED; mock nguy hiem -> MARGIN_DANGER va trading_allowed is False.
9. perp_positions chuan hoa dung (BTCUSDT -> BTC/USDT:USDT).
10. dry_run=True ma khong co mock_responses -> chan goi mang that, snapshot_healthy is False.
"""

from __future__ import annotations

import hashlib
import hmac
import pytest

from src.binance_signed_client import (
    PERP_ACCOUNT_PATH,
    PERP_POSITION_RISK_PATH,
    SPOT_ACCOUNT_PATH,
    BinanceCredentialsMissingError,
    BinanceSignedTestnetClient,
    build_query_string,
)
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
from src.exchange_reconciler import reconcile_portfolio_state

_KEY = "testnet-api-key-123"
_SECRET = "testnet-api-secret-456"
BTC = "BTC/USDT:USDT"


def _client(dry_run: bool = True, **kwargs) -> BinanceSignedTestnetClient:
    return BinanceSignedTestnetClient(
        api_key=_KEY, api_secret=_SECRET, dry_run=dry_run, **kwargs
    )


# ---------------------------------------------------------------------------
# 1-4. build_signed_get_request
# ---------------------------------------------------------------------------


def test_build_signed_get_request_spot() -> None:
    client = _client()
    req = client.build_signed_get_request(SPOT_ACCOUNT_PATH, futures=False)
    assert req["method"] == "GET"
    assert req["url"] == "https://testnet.binance.vision/api/v3/account"
    assert req["headers"] == {"X-MBX-APIKEY": _KEY}
    params = req["params"]
    assert "timestamp" in params
    assert params["recvWindow"] == 5000
    assert "signature" in params
    assert len(params["signature"]) == 64


def test_build_signed_get_request_futures() -> None:
    client = _client()
    req = client.build_signed_get_request(PERP_ACCOUNT_PATH, futures=True)
    assert req["method"] == "GET"
    assert req["url"] == "https://testnet.binancefuture.com/fapi/v2/account"
    assert req["headers"] == {"X-MBX-APIKEY": _KEY}

    req_risk = client.build_signed_get_request(PERP_POSITION_RISK_PATH, futures=True)
    assert req_risk["url"] == "https://testnet.binancefuture.com/fapi/v2/positionRisk"


def test_build_signed_get_request_hmac_validity() -> None:
    client = _client()
    timestamp_ms = 1700000000000
    req = client.build_signed_get_request(
        SPOT_ACCOUNT_PATH, params={"recvWindow": 5000}, timestamp_ms=timestamp_ms
    )
    signed_params = req["params"]
    unsigned = {k: v for k, v in signed_params.items() if k != "signature"}
    expected_sig = hmac.new(
        _SECRET.encode("utf-8"),
        build_query_string(unsigned).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert signed_params["signature"] == expected_sig


def test_build_signed_get_request_missing_credentials_fail_closed() -> None:
    client = BinanceSignedTestnetClient(api_key="", api_secret="", dry_run=True)
    with pytest.raises(BinanceCredentialsMissingError):
        client.build_signed_get_request(SPOT_ACCOUNT_PATH)


# ---------------------------------------------------------------------------
# 5-7, 10. fetch_exchange_reconcile_snapshot
# ---------------------------------------------------------------------------


def _sample_mock() -> dict:
    return {
        "spot_account": {
            "balances": [
                {"asset": "USDT", "free": "1500.0", "locked": "0.0"},
                {"asset": "BTC", "free": "0.05", "locked": "0.0"},
            ]
        },
        "futures_account": {
            "totalWalletBalance": "2500.0",
            "totalUnrealizedProfit": "120.0",
            "totalMaintMargin": "80.0",
            "totalMarginBalance": "2620.0",
        },
        "position_risk": [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "-0.05",
                "entryPrice": "95000.0",
                "markPrice": "96000.0",
                "liquidationPrice": "160000.0",
                "maintMargin": "80.0",
                "marginBalance": "2620.0",
            }
        ],
    }


def test_fetch_exchange_reconcile_snapshot_standard_mock() -> None:
    client = _client()
    prices = {BTC: 96000.0}
    snap = client.fetch_exchange_reconcile_snapshot([BTC], prices, mock_responses=_sample_mock())

    assert snap["snapshot_healthy"] is True
    assert snap["snapshot_source"] == "mock"
    assert snap["spot_balances"].get("BTC") == pytest.approx(0.05)
    # Sprint 18: tien mat quote chi nam trong spot_cash_usd, khong nam trong spot_balances
    # (tranh reconciler bao ORPHAN gia tao cho USDT).
    assert "USDT" not in snap["spot_balances"]
    assert snap["perp_positions"].get(BTC) == pytest.approx(-0.05)
    assert len(snap["perp_positions_detail"]) == 1
    assert snap["perp_positions_detail"][0]["symbol"] == BTC
    assert snap["perp_positions_detail"][0]["qty"] == pytest.approx(-0.05)
    assert snap["perp_positions_detail"][0]["mark_price"] == pytest.approx(96000.0)
    assert snap["perp_wallet_balance_usd"] == pytest.approx(2500.0)
    assert snap["perp_unrealized_pnl_usd"] == pytest.approx(120.0)
    assert snap["maint_margin_usd"] == pytest.approx(80.0)
    assert snap["margin_balance_usd"] == pytest.approx(2620.0)

    # spot_cash_usd = 1500.0 (USDT) + 0.05 * 96000.0 (BTC) = 6300.0
    # exchange_equity_usd = 6300.0 + 2500.0 + 120.0 = 8920.0
    assert snap["spot_cash_usd"] == pytest.approx(6300.0)
    assert snap["exchange_equity_usd"] == pytest.approx(8920.0)


def test_fetch_exchange_reconcile_snapshot_no_credentials_no_mock() -> None:
    client = BinanceSignedTestnetClient(api_key="", api_secret="", dry_run=True)
    snap = client.fetch_exchange_reconcile_snapshot([BTC], {BTC: 96000.0})
    assert snap["snapshot_healthy"] is False
    assert snap["spot_balances"] == {}
    assert snap["perp_positions"] == {}
    assert snap["exchange_equity_usd"] == 0.0
    assert "error" in snap


def test_fetch_exchange_reconcile_snapshot_malformed_mock() -> None:
    client = _client()
    prices = {BTC: 96000.0}

    # spot_account is not a dict
    snap1 = client.fetch_exchange_reconcile_snapshot([BTC], prices, mock_responses={"spot_account": "not-a-dict"})
    assert snap1["snapshot_healthy"] is False
    assert "error" in snap1

    # futures_account is not a dict
    snap2 = client.fetch_exchange_reconcile_snapshot([BTC], prices, mock_responses={"futures_account": 999})
    assert snap2["snapshot_healthy"] is False

    # empty mock
    snap3 = client.fetch_exchange_reconcile_snapshot([BTC], prices, mock_responses={})
    assert snap3["snapshot_healthy"] is False


def test_dry_run_without_mock_blocks_network() -> None:
    client = _client(dry_run=True)
    snap = client.fetch_exchange_reconcile_snapshot([BTC], {BTC: 96000.0})
    assert snap["snapshot_healthy"] is False
    assert "error" in snap
    assert "dry run" in snap["error"].lower() or "blocked" in snap["error"].lower()


# ---------------------------------------------------------------------------
# 8-9. Tich hop Reconcile & Chuan hoa Symbol
# ---------------------------------------------------------------------------


def test_reconcile_integration_synced_and_margin_danger() -> None:
    client = _client()
    prices = {BTC: 96000.0}

    # 1. Healthy mock -> SYNCED
    snap = client.fetch_exchange_reconcile_snapshot([BTC], prices, mock_responses=_sample_mock())
    local_state = {
        "current_equity": snap["exchange_equity_usd"],
        "carry_positions": {
            BTC: {
                "qty": 0.05,
                "leg2_hedged_qty": 0.05,
                "entry_perp_price": 95000.0,
            }
        },
        "overlay_positions": {},
    }
    rep_synced = reconcile_portfolio_state(local_state, snap, prices, EXCHANGE_LOT_SPECS)
    assert rep_synced["status"] == "SYNCED"
    assert rep_synced["trading_allowed"] is True

    # 2. Mock voi maint_margin cao vuot muc 0.50 -> MARGIN_DANGER
    danger_mock = _sample_mock()
    danger_mock["futures_account"]["totalMaintMargin"] = "1500.0"
    danger_mock["futures_account"]["totalMarginBalance"] = "2000.0"
    danger_mock["position_risk"][0]["maintMargin"] = "1500.0"
    danger_mock["position_risk"][0]["marginBalance"] = "2000.0"

    snap_danger = client.fetch_exchange_reconcile_snapshot([BTC], prices, mock_responses=danger_mock)
    local_state_danger = {
        "current_equity": snap_danger["exchange_equity_usd"],
        "carry_positions": {
            BTC: {
                "qty": 0.05,
                "leg2_hedged_qty": 0.05,
                "entry_perp_price": 95000.0,
            }
        },
        "overlay_positions": {},
    }
    rep_danger = reconcile_portfolio_state(local_state_danger, snap_danger, prices, EXCHANGE_LOT_SPECS)
    assert rep_danger["status"] == "MARGIN_DANGER"
    assert rep_danger["trading_allowed"] is False
    assert len(rep_danger["repair_actions"]) > 0


def test_perp_positions_symbol_normalization() -> None:
    client = _client()
    mock_data = {
        "futures_account": {
            "totalWalletBalance": "1000.0",
            "totalMarginBalance": "1000.0",
        },
        "position_risk": [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.15",
                "markPrice": "96000.0",
                "entryPrice": "95000.0",
            }
        ],
    }
    snap = client.fetch_exchange_reconcile_snapshot([BTC], {BTC: 96000.0}, mock_responses=mock_data)
    assert snap["snapshot_healthy"] is True
    assert BTC in snap["perp_positions"]
    assert snap["perp_positions"][BTC] == pytest.approx(0.15)
    assert snap["perp_positions_detail"][0]["symbol"] == BTC
