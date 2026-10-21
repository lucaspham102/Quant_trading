"""Test Sprint 16 Chang 3 - ky lenh Binance testnet (HMAC-SHA256) va audit JSONL.

Bao phu:
1. `sign_binance_payload` khop vector HMAC-SHA256 chuan (khong phu thuoc thu vien ngoai).
2. `build_query_string` sap xep key va khong dung ky phap mu (e-05).
3. `sign_params` them timestamp/recvWindow/signature, timestamp inject duoc.
4. Fail-closed: thieu credential ma dry_run=False -> raise.
5. `build_delta_neutral_order_pair` dung URL Spot/Perp testnet + side dung.
6. Lot-size khong khop step_size -> ValueError (khong gui lenh sai).
7. Thiếu spot_price khi post_only=True -> ValueError.
8. `append_audit_record` ghi JSONL (moi dong 1 JSON hop le) va bo qua khi path=None.
9. `build_cycle_audit_record` tinh `overlay_net_delta_usd` va schema on dinh.
"""

from __future__ import annotations

import json

import pytest

from src.binance_signed_client import (
    PERP_ORDER_PATH,
    SPOT_ORDER_PATH,
    BinanceCredentialsMissingError,
    BinanceSignedTestnetClient,
    build_query_string,
    sign_binance_payload,
)
from src.carry_live_executor import append_audit_record, build_cycle_audit_record

_KEY = "testnet-key-abc"
_SECRET = "testnet-secret-xyz"


def _client(dry_run: bool = True, **kwargs) -> BinanceSignedTestnetClient:
    return BinanceSignedTestnetClient(
        api_key=_KEY, api_secret=_SECRET, dry_run=dry_run, **kwargs
    )


# ---------------------------------------------------------------------------
# 1-3. Ky va dong goi
# ---------------------------------------------------------------------------


def test_signature_matches_reference_hmac() -> None:
    import hashlib
    import hmac

    params = {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.001, "timestamp": 1700000000000}
    expected = hmac.new(
        _SECRET.encode(), build_query_string(params).encode(), hashlib.sha256
    ).hexdigest()
    assert sign_binance_payload(params, _SECRET) == expected
    assert len(sign_binance_payload(params, _SECRET)) == 64


def test_query_string_sorted_and_no_scientific_notation() -> None:
    qs = build_query_string({"z": 2, "a": 0.00001, "m": 1.5})
    assert qs.startswith("a=0.00001")
    assert "m=1.5" in qs
    assert qs.endswith("z=2")
    assert "e-05" not in qs


def test_sign_params_injects_timestamp_and_signature() -> None:
    client = _client()
    signed = client.sign_params({"symbol": "ETHUSDT", "side": "SELL"}, timestamp_ms=1700000000123)
    assert signed["timestamp"] == 1700000000123
    assert signed["recvWindow"] == 5000
    assert len(signed["signature"]) == 64
    unsigned = {k: v for k, v in signed.items() if k != "signature"}
    assert signed["signature"] == sign_binance_payload(unsigned, _SECRET)


def test_missing_credentials_fail_closed() -> None:
    with pytest.raises(BinanceCredentialsMissingError):
        BinanceSignedTestnetClient(api_key="", api_secret="", dry_run=False)
    # dry_run=True van cho phep khoi tao de test ky cuc bo
    client = BinanceSignedTestnetClient(api_key="", api_secret="", dry_run=True)
    assert client.has_credentials() is False
    with pytest.raises(BinanceCredentialsMissingError):
        client.sign_params({"symbol": "BTCUSDT"})


# ---------------------------------------------------------------------------
# 5-7. Dong goi cap lenh delta-neutral
# ---------------------------------------------------------------------------


def test_delta_neutral_pair_targets_testnet_endpoints() -> None:
    client = _client()
    pair = client.build_delta_neutral_order_pair(
        symbol="BTC/USDT:USDT",
        spot_qty=0.001,
        perp_qty=0.001,
        spot_price=96558.0,
        timestamp_ms=1700000000000,
    )
    assert pair["dry_run"] is True
    assert pair["spot"]["url"].endswith(SPOT_ORDER_PATH)
    assert "testnet.binance.vision" in pair["spot"]["url"]
    assert pair["perp"]["url"].endswith(PERP_ORDER_PATH)
    assert "testnet.binancefuture.com" in pair["perp"]["url"]
    assert pair["spot"]["params"]["side"] == "BUY"
    assert pair["spot"]["params"]["type"] == "LIMIT"
    assert pair["spot"]["params"]["timeInForce"] == "GTX"
    assert pair["perp"]["params"]["side"] == "SELL"
    assert pair["perp"]["params"]["type"] == "MARKET"
    assert pair["spot"]["headers"]["X-MBX-APIKEY"] == _KEY
    assert pair["lot"]["spot"]["step_size"] == pytest.approx(0.001)
    assert pair["lot"]["notional_spot_usd"] == pytest.approx(96.558)


def test_invalid_lot_size_rejected() -> None:
    client = _client()
    with pytest.raises(ValueError):
        client.build_delta_neutral_order_pair("BTC/USDT:USDT", spot_qty=0.0015, perp_qty=0.001)
    with pytest.raises(ValueError):
        client.build_delta_neutral_order_pair("ETH/USDT:USDT", spot_qty=0.001, perp_qty=0.001)
    with pytest.raises(ValueError):
        client.build_delta_neutral_order_pair("SOL/USDT:USDT", spot_qty=0.1, perp_qty=0.1, spot_price=0.5)


def test_post_only_requires_price() -> None:
    client = _client()
    with pytest.raises(ValueError):
        client.build_delta_neutral_order_pair("BTC/USDT:USDT", 0.001, 0.001, spot_price=None)
    pair = client.build_delta_neutral_order_pair(
        "BTC/USDT:USDT", 0.001, 0.001, post_only_spot=False
    )
    assert pair["spot"]["params"]["type"] == "MARKET"


# ---------------------------------------------------------------------------
# 8-9. Audit JSONL
# ---------------------------------------------------------------------------


def test_append_audit_record_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "logs" / "sprint16_paper_audit.jsonl"
    record = {"cycle": 1, "equity": 3000.0}
    assert append_audit_record(path, record) is True
    assert append_audit_record(path, {"cycle": 2}) is True
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cycle"] == 1
    assert json.loads(lines[1])["cycle"] == 2
    assert append_audit_record(None, record) is False


def test_build_cycle_audit_record_net_delta() -> None:
    record = build_cycle_audit_record(
        cycle=3,
        mode="paper",
        execution_style="real_taker",
        capital=3000.0,
        equity=3001.5,
        peak_equity=3002.0,
        drawdown_pct=0.0166,
        carry_spot_usd=1200.0,
        carry_perp_usd=1200.0,
        overlay_long_usd=332.85,
        overlay_short_usd=332.84,
        overlay_neutrality={"applied": True},
        kill_switch=False,
        diagnostics_summary="carryTrades=0",
        actions={"carry": [], "overlay": []},
        timestamp="2026-10-21T00:00:00+00:00",
    )
    assert record["schema"] == "sprint16.cycle.v1"
    assert record["ts"] == "2026-10-21T00:00:00+00:00"
    assert record["overlay_net_delta_usd"] == pytest.approx(0.01, abs=1e-9)
    assert record["overlay_neutrality"] == {"applied": True}
    json.dumps(record)  # phai serialize duoc


def test_build_cycle_audit_record_accepts_list_actions() -> None:
    """Regression: plan['actions'] cua executor la list[str], khong duoc crash."""
    record = build_cycle_audit_record(
        cycle=1,
        mode="paper",
        execution_style="real_taker",
        capital=3000.0,
        equity=2999.34,
        peak_equity=3000.0,
        drawdown_pct=0.02,
        carry_spot_usd=0.0,
        carry_perp_usd=0.0,
        overlay_long_usd=100.0,
        overlay_short_usd=100.03,
        actions=["HOLD BTC/USDT:USDT", "REBALANCE OVERLAY"],
        timestamp="2026-10-21T00:00:00+00:00",
    )
    assert record["actions"] == ["HOLD BTC/USDT:USDT", "REBALANCE OVERLAY"]
    assert record["overlay_net_delta_usd"] == pytest.approx(-0.03, abs=1e-9)
    json.dumps(record)
