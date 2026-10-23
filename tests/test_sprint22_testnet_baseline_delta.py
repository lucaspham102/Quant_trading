"""Sprint 22: Unit tests cho co che Baseline-Delta tren Binance Testnet.

Bao phu cac kich ban bat buoc:
1. Snapshot testnet ban dau (54,999.12 USD + 8 coin faucet vs 3000 capital) cho SYNCED, equity_gap == 0.00 USD.
2. Lech equity that +5% (+150 USD tren 3000 capital) bao EQUITY_MISMATCH.
3. Ban 0.05 BTC lay USDT (equity tong khong doi) bi phat hien qua hut spot (QTY_DRIFT).
4. Che do live giu nguyen 100% (khong co reconcile_equity_usd, dung exchange_equity_usd).
5. Restart giua chung giu nguyen baseline cu; vi the bot vua mo KHONG bi nuot vao baseline.
6. Gia thi truong BTC nhay +$100 tren coin faucet; equity gap van 0.00 USD (SYNCED).
7. Baseline JSON bi hong -> RuntimeError (fail-closed abort).
8. Truyen --reset-state KHONG lam mat baseline; chi co --reset-testnet-baseline moi tao lai.
9. Thay doi API key (credentials_sha256 khac) -> RuntimeError.
10. Test tich hop qua duong preflight _run_signed_snapshot_preflight (mock snapshot testnet that 55k faucet) -> SYNCED luc bat dau.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import run_live
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
from src.exchange_reconciler import (
    DEFAULT_TESTNET_BASELINE_PATH,
    compute_testnet_delta_snapshot,
    load_or_create_testnet_baseline,
    reconcile_portfolio_state,
)

FAUCET_SPOT_BALANCES: dict[str, float] = {
    "FDUSD": 10000.0,
    "TUSD": 10000.0,
    "USD": 10000.0,
    "USDC": 10000.0,
    "USDT": 9999.98,
    "BTC": 1.0,
    "ETH": 1.0,
    "BNB": 1.0,
    "SOL": 4.0,
    "AVAX": 44.0,
    "LINK": 36.0,
    "XRP": 341.0,
    "DOGE": 5501.0,
}

MARKET_PRICES: dict[str, float] = {
    "BTC/USDT:USDT": 60000.0,
    "ETH/USDT:USDT": 3000.0,
    "BNB/USDT:USDT": 500.0,
    "SOL/USDT:USDT": 150.0,
    "AVAX/USDT:USDT": 30.0,
    "LINK/USDT:USDT": 12.0,
    "XRP/USDT:USDT": 0.50,
    "DOGE/USDT:USDT": 0.10,
    "BTC": 60000.0,
    "ETH": 3000.0,
    "BNB": 500.0,
    "SOL": 150.0,
    "AVAX": 30.0,
    "LINK": 12.0,
    "XRP": 0.50,
    "DOGE": 0.10,
}

UNIVERSE_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
]


def _make_faucet_snapshot() -> dict[str, Any]:
    """Tao snapshot gia lap tai khoan Binance Testnet that voi 54,999.12 USD pure cash/margin."""
    return {
        "snapshot_healthy": True,
        "spot_cash_usd": 49999.98,
        "perp_wallet_balance_usd": 4999.14,
        "perp_unrealized_pnl_usd": 0.0,
        "exchange_equity_usd": 54999.12,
        "spot_balances": dict(FAUCET_SPOT_BALANCES),
        "perp_positions": {s: 0.0 for s in UNIVERSE_SYMBOLS},
        "perp_positions_detail": [
            {"symbol": s, "positionAmt": 0.0, "mark_price": MARKET_PRICES.get(s, 0.0)}
            for s in UNIVERSE_SYMBOLS
        ],
    }


def _dummy_hash(key: str = "test_key") -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def test_initial_testnet_snapshot_synced(tmp_path: Path) -> None:
    """1. Snapshot testnet ban dau (54,999.12 USD + 8 faucet coins vs 3000 capital) cho SYNCED."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()
    key_hash = _dummy_hash()

    baseline = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )

    capital = 3000.0
    local_state = {
        "current_equity": 3000.0,
        "carry_positions": {},
        "target_weights": {},
    }

    delta_snapshot = compute_testnet_delta_snapshot(
        snapshot=snapshot,
        baseline=baseline,
        prices=MARKET_PRICES,
        capital=capital,
        local_equity=local_state["current_equity"],
    )

    report = reconcile_portfolio_state(
        local_state=local_state,
        exchange_snapshot=delta_snapshot,
        prices=MARKET_PRICES,
        lot_specs=EXCHANGE_LOT_SPECS,
        environment="testnet",
    )

    assert report["status"] == "SYNCED"
    assert report["trading_allowed"] is True
    assert report["equity_gap_usd"] == pytest.approx(0.0, abs=1e-2)
    # Sprint 22 (Item 3): exchange_equity_usd giu equity THAT cua snapshot san (54,999.12 USD)
    # reconcile_equity_usd la equity tong hop dung de doi soat (3,000.00 USD)
    assert report["exchange_equity_usd"] == pytest.approx(54999.12, abs=1e-2)
    assert report["reconcile_equity_usd"] == pytest.approx(3000.0, abs=1e-2)


def test_real_equity_drift_detected(tmp_path: Path) -> None:
    """2. Lech equity that +5% (+150 USD tren 3000 capital) bao EQUITY_MISMATCH."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()
    key_hash = _dummy_hash()

    baseline = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )

    capital = 3000.0
    local_state = {
        "current_equity": 3000.0,
        "carry_positions": {},
        "target_weights": {},
    }

    # Thi truong/tai khoan tang 150 USD tren san testnet (vi du loi nhuan ngoai bot)
    drifted_snapshot = _make_faucet_snapshot()
    drifted_snapshot["perp_wallet_balance_usd"] += 150.0

    delta_snapshot = compute_testnet_delta_snapshot(
        snapshot=drifted_snapshot,
        baseline=baseline,
        prices=MARKET_PRICES,
        capital=capital,
        local_equity=local_state["current_equity"],
    )

    report = reconcile_portfolio_state(
        local_state=local_state,
        exchange_snapshot=delta_snapshot,
        prices=MARKET_PRICES,
        lot_specs=EXCHANGE_LOT_SPECS,
        equity_tolerance_pct=1.5,
        environment="testnet",
    )

    assert report["status"] == "EQUITY_MISMATCH"
    assert report["trading_allowed"] is False
    assert report["equity_gap_usd"] == pytest.approx(150.0, abs=1e-2)
    assert report["equity_gap_pct"] == pytest.approx(5.0, abs=1e-2)


def test_spot_quantity_shortfall_detected_as_qty_drift(tmp_path: Path) -> None:
    """3. Ban 0.05 BTC lay USDT (equity tong khong doi) bi phat hien qua hut spot (QTY_DRIFT)."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()
    key_hash = _dummy_hash()

    baseline = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )

    capital = 3000.0
    local_state = {
        "current_equity": 3000.0,
        "carry_positions": {},
        "target_weights": {},
    }

    # Ban 0.05 BTC @ 60,000 USD = nhan 3,000 USDT
    shortfall_snapshot = _make_faucet_snapshot()
    shortfall_snapshot["spot_balances"]["BTC"] = 0.95
    shortfall_snapshot["spot_balances"]["USDT"] += 3000.0

    delta_snapshot = compute_testnet_delta_snapshot(
        snapshot=shortfall_snapshot,
        baseline=baseline,
        prices=MARKET_PRICES,
        capital=capital,
        local_equity=local_state["current_equity"],
    )

    # Net drift = (pure_cash +3000) + (delta_spot_market_value -3000) = 0
    assert delta_snapshot["reconcile_equity_usd"] == pytest.approx(3000.0, abs=1e-2)
    assert delta_snapshot["spot_balances"]["BTC"] == pytest.approx(-0.05, abs=1e-6)

    report = reconcile_portfolio_state(
        local_state=local_state,
        exchange_snapshot=delta_snapshot,
        prices=MARKET_PRICES,
        lot_specs=EXCHANGE_LOT_SPECS,
        equity_tolerance_pct=1.5,
        environment="testnet",
    )

    assert report["status"] == "QTY_DRIFT"
    assert report["trading_allowed"] is False
    btc_item = next(it for it in report["items"] if it["symbol"] == "BTC/USDT:USDT")
    assert btc_item["status"] == "QTY_DRIFT"
    assert btc_item["repair_action"] == "RESYNC_QTY"
    assert any("Spot balance dropped below baseline" in ra["reason"] for ra in report["repair_actions"])


def test_live_mode_behavior_unchanged() -> None:
    """4. Che do live giu nguyen 100% (khong co reconcile_equity_usd, dung exchange_equity_usd)."""
    local_state = {
        "current_equity": 3000.0,
        "carry_positions": {},
        "target_weights": {},
    }
    # Live snapshot khong co truong reconcile_equity_usd
    live_snapshot = {
        "exchange_equity_usd": 3000.0,
        "spot_cash_usd": 3000.0,
        "perp_wallet_balance_usd": 0.0,
        "perp_unrealized_pnl_usd": 0.0,
        "spot_balances": {"USDT": 3000.0},
        "perp_positions": {},
    }

    report = reconcile_portfolio_state(
        local_state=local_state,
        exchange_snapshot=live_snapshot,
        prices=MARKET_PRICES,
        lot_specs=EXCHANGE_LOT_SPECS,
        environment="mainnet",
    )
    assert report["status"] == "SYNCED"
    assert report["equity_gap_usd"] == pytest.approx(0.0, abs=1e-2)

    # Neu live snapshot lech 3500 vs 3000 -> EQUITY_MISMATCH
    live_snapshot["exchange_equity_usd"] = 3500.0
    report_mismatch = reconcile_portfolio_state(
        local_state=local_state,
        exchange_snapshot=live_snapshot,
        prices=MARKET_PRICES,
        lot_specs=EXCHANGE_LOT_SPECS,
        environment="mainnet",
    )
    assert report_mismatch["status"] == "EQUITY_MISMATCH"
    assert report_mismatch["equity_gap_usd"] == pytest.approx(500.0, abs=1e-2)


def test_restart_mid_session_preserves_baseline(tmp_path: Path) -> None:
    """5. Restart giua chung giu nguyen baseline cu; vi the bot vua mo KHONG bi nuot vao baseline."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot_t0 = _make_faucet_snapshot()
    key_hash = _dummy_hash()

    # Khoi tao baseline tai t0
    b0 = load_or_create_testnet_baseline(
        snapshot=snapshot_t0,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )
    assert b0["spot_balances"]["BTC"] == pytest.approx(1.0, abs=1e-6)
    assert b0["perp_positions"]["BTC/USDT:USDT"] == pytest.approx(0.0, abs=1e-6)

    # Bot chay va mo vi the carry: mua 0.01 BTC spot, short 0.01 BTC perp
    snapshot_t1 = _make_faucet_snapshot()
    snapshot_t1["spot_balances"]["BTC"] = 1.01
    snapshot_t1["spot_balances"]["USDT"] -= 600.0  # 0.01 * 60000 = 600 USDT
    snapshot_t1["perp_positions"]["BTC/USDT:USDT"] = -0.01
    snapshot_t1["perp_positions_detail"] = [
        {"symbol": "BTC/USDT:USDT", "positionAmt": -0.01, "mark_price": 60000.0}
    ]

    local_state_t1 = {
        "current_equity": 3000.0,
        "carry_positions": {
            "BTC/USDT:USDT": {
                "qty": 0.01,
                "leg2_hedged_qty": 0.01,
                "entry_basis_bps": 5.0,
            }
        },
        "target_weights": {},
    }

    # Bot restart giua chung: load baseline ma KHONG co --reset-testnet-baseline
    b_reloaded = load_or_create_testnet_baseline(
        snapshot=snapshot_t1,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )
    # Baseline van giu moc 1.0 BTC va 0.0 perp, KHONG bi ghi de thanh 1.01 / -0.01
    assert b_reloaded["spot_balances"]["BTC"] == pytest.approx(1.0, abs=1e-6)
    assert b_reloaded["perp_positions"]["BTC/USDT:USDT"] == pytest.approx(0.0, abs=1e-6)

    delta_snapshot = compute_testnet_delta_snapshot(
        snapshot=snapshot_t1,
        baseline=b_reloaded,
        prices=MARKET_PRICES,
        capital=3000.0,
        local_equity=local_state_t1["current_equity"],
    )

    # Delta spot = 0.01, delta perp = -0.01
    assert delta_snapshot["spot_balances"]["BTC"] == pytest.approx(0.01, abs=1e-6)
    assert delta_snapshot["perp_positions"]["BTC/USDT:USDT"] == pytest.approx(-0.01, abs=1e-6)

    # Reconcile so voi local state co dung 0.01 / -0.01 -> SYNCED
    report = reconcile_portfolio_state(
        local_state=local_state_t1,
        exchange_snapshot=delta_snapshot,
        prices=MARKET_PRICES,
        lot_specs=EXCHANGE_LOT_SPECS,
        environment="testnet",
    )
    assert report["status"] == "SYNCED"
    assert report["trading_allowed"] is True


def test_faucet_coin_market_price_jump_ignored(tmp_path: Path) -> None:
    """6. Gia thi truong BTC nhay +$100 tren coin faucet; equity gap van 0.00 USD (SYNCED)."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()
    key_hash = _dummy_hash()

    baseline = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )

    # Gia BTC tang tu 60,000 len 70,000 USD (+10,000 USD)
    new_prices = dict(MARKET_PRICES)
    new_prices["BTC"] = 70000.0
    new_prices["BTC/USDT:USDT"] = 70000.0

    # Khong co giao dich nao: spot BTC van la 1.0 (delta = 0.0)
    delta_snapshot = compute_testnet_delta_snapshot(
        snapshot=snapshot,
        baseline=baseline,
        prices=new_prices,
        capital=3000.0,
        local_equity=3000.0,
    )

    local_state = {
        "current_equity": 3000.0,
        "carry_positions": {},
        "target_weights": {},
    }

    report = reconcile_portfolio_state(
        local_state=local_state,
        exchange_snapshot=delta_snapshot,
        prices=new_prices,
        lot_specs=EXCHANGE_LOT_SPECS,
        environment="testnet",
    )

    assert report["status"] == "SYNCED"
    assert report["trading_allowed"] is True
    assert report["equity_gap_usd"] == pytest.approx(0.0, abs=1e-2)


def test_corrupted_baseline_raises_runtime_error(tmp_path: Path) -> None:
    """7. Baseline JSON bi hong -> RuntimeError (fail-closed abort)."""
    baseline_path = tmp_path / "testnet_baseline.json"
    baseline_path.write_text("{invalid json corrupt content...", encoding="utf-8")

    snapshot = _make_faucet_snapshot()
    key_hash = _dummy_hash()

    with pytest.raises(RuntimeError, match="BASELINE_CORRUPTED"):
        load_or_create_testnet_baseline(
            snapshot=snapshot,
            prices=MARKET_PRICES,
            api_key_hash=key_hash,
            universe_symbols=UNIVERSE_SYMBOLS,
            path=baseline_path,
            reset_baseline=False,
        )


def test_reset_state_does_not_reset_baseline(tmp_path: Path) -> None:
    """8. reset_baseline=False giu nguyen baseline; chi reset_baseline=True moi ghi de."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()
    key_hash = _dummy_hash()

    # Khoi tao baseline
    b1 = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )
    created_at = b1["created_at"]

    # Goi lai voi reset_baseline=False (nhu khi chi co --reset-state ma khong co --reset-testnet-baseline)
    b2 = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )
    assert b2["created_at"] == created_at

    # Khi co --reset-testnet-baseline (reset_baseline=True), baseline duoc chup lai moi
    time.sleep(0.01)
    b3 = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=MARKET_PRICES,
        api_key_hash=key_hash,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=True,
    )
    assert b3["created_at"] >= created_at


def test_altered_api_keys_raises_runtime_error(tmp_path: Path) -> None:
    """9. Thay doi API key (credentials_sha256 khac) -> RuntimeError (fail-closed)."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()
    key_hash_1 = _dummy_hash("key_pair_1")
    key_hash_2 = _dummy_hash("key_pair_2")

    load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=MARKET_PRICES,
        api_key_hash=key_hash_1,
        universe_symbols=UNIVERSE_SYMBOLS,
        path=baseline_path,
        reset_baseline=False,
    )

    with pytest.raises(RuntimeError, match="BASELINE_KEY_MISMATCH"):
        load_or_create_testnet_baseline(
            snapshot=snapshot,
            prices=MARKET_PRICES,
            api_key_hash=key_hash_2,
            universe_symbols=UNIVERSE_SYMBOLS,
            path=baseline_path,
            reset_baseline=False,
        )


def test_preflight_end_to_end_synced_on_testnet_start(tmp_path: Path) -> None:
    """10. Test tich hop qua duong _run_signed_snapshot_preflight -> SYNCED luc khoi dong testnet."""
    class _MockPreflightClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def sync_server_time(self, venue: str = "both") -> dict[str, Any]:
            return {"errors": []}

        def fetch_exchange_reconcile_snapshot(
            self, symbols: list[str], prices: dict[str, float]
        ) -> dict[str, Any]:
            return _make_faucet_snapshot()

        def fetch_book_ticker(self, symbol: str, venue: str = "spot") -> dict[str, Any]:
            return {"mid": MARKET_PRICES.get(symbol, 100.0)}

    class _FakePreflightExecutor:
        def __init__(self, symbols: list[str], state: Any) -> None:
            self.symbols = symbols
            self.state = state
            self.reconcile_report: dict[str, Any] | None = None
            self.reconcile_status: str | None = None
            self.trading_allowed: bool = False

        def apply_reconcile_report(self, rep: dict[str, Any]) -> bool:
            self.reconcile_report = rep
            self.reconcile_status = rep.get("status")
            self.trading_allowed = bool(rep.get("trading_allowed"))
            return self.trading_allowed

    class _FakePreflightState:
        def __init__(self, capital: float = 3000.0) -> None:
            self.current_equity = capital
            self.carry_positions: dict[str, Any] = {}
            self.target_weights: dict[str, Any] = {}

    baseline_file = str(tmp_path / "testnet_baseline.json")
    args = run_live.parse_args([
        "--mode", "testnet",
        "--capital", "3000",
        "--testnet-baseline-file", baseline_file,
    ])
    executor = _FakePreflightExecutor(
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        state=_FakePreflightState(3000.0),
    )

    with patch("src.binance_signed_client.BinanceSignedTestnetClient", _MockPreflightClient), \
         patch.dict("os.environ", {
             "BINANCE_SPOT_TESTNET_API_KEY": "test_spot_key",
             "BINANCE_SPOT_TESTNET_API_SECRET": "test_spot_secret",
             "BINANCE_FUTURES_TESTNET_API_KEY": "test_fut_key",
             "BINANCE_FUTURES_TESTNET_API_SECRET": "test_fut_secret",
         }):
        rep = run_live._run_signed_snapshot_preflight(args, executor)

    assert rep is not None
    assert rep["status"] == "SYNCED"
    assert rep["trading_allowed"] is True
    assert rep["equity_gap_usd"] == pytest.approx(0.0, abs=1e-2)
    assert executor.reconcile_status == "SYNCED"
    assert executor.trading_allowed is True


def test_baseline_perp_not_flat_raises_runtime_error(tmp_path: Path) -> None:
    """BASELINE_PERP_NOT_FLAT: Abort khi tai khoan testnet dang co vi the perp mo luc tao baseline."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()
    snapshot["perp_positions"]["BTC/USDT:USDT"] = -0.05
    snapshot["perp_positions_detail"] = [
        {"symbol": "BTC/USDT:USDT", "positionAmt": -0.05, "mark_price": 60000.0}
    ]
    key_hash = _dummy_hash()

    with pytest.raises(RuntimeError, match="BASELINE_PERP_NOT_FLAT"):
        load_or_create_testnet_baseline(
            snapshot=snapshot,
            prices=MARKET_PRICES,
            api_key_hash=key_hash,
            universe_symbols=UNIVERSE_SYMBOLS,
            path=baseline_path,
            reset_baseline=True,
        )


def test_baseline_key_hash_empty_raises_runtime_error(tmp_path: Path) -> None:
    """BASELINE_KEY_HASH_EMPTY: Abort khi api_key_hash bi rong."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()

    with pytest.raises(RuntimeError, match="BASELINE_KEY_HASH_EMPTY"):
        load_or_create_testnet_baseline(
            snapshot=snapshot,
            prices=MARKET_PRICES,
            api_key_hash="   ",
            universe_symbols=UNIVERSE_SYMBOLS,
            path=baseline_path,
            reset_baseline=True,
        )


def test_baseline_schema_invalid_raises_runtime_error(tmp_path: Path) -> None:
    """BASELINE_SCHEMA_INVALID: Abort khi file baseline thieu cac key bat buoc hoac hash rong."""
    baseline_path = tmp_path / "testnet_baseline.json"
    snapshot = _make_faucet_snapshot()
    key_hash = _dummy_hash()

    # 1. Thieu truong bat buoc
    invalid_data = {
        "created_at": "2026-10-23T00:00:00+00:00",
        "credentials_sha256": key_hash,
        "initial_recon_equity": 54999.12,
        # "spot_balances" bi thieu
        "perp_positions": {},
        "prices": {},
    }
    baseline_path.write_text(json.dumps(invalid_data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="BASELINE_SCHEMA_INVALID"):
        load_or_create_testnet_baseline(
            snapshot=snapshot,
            prices=MARKET_PRICES,
            api_key_hash=key_hash,
            universe_symbols=UNIVERSE_SYMBOLS,
            path=baseline_path,
            reset_baseline=False,
        )

    # 2. credentials_sha256 bi rong
    invalid_data["spot_balances"] = {}
    invalid_data["credentials_sha256"] = ""
    baseline_path.write_text(json.dumps(invalid_data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="BASELINE_SCHEMA_INVALID"):
        load_or_create_testnet_baseline(
            snapshot=snapshot,
            prices=MARKET_PRICES,
            api_key_hash=key_hash,
            universe_symbols=UNIVERSE_SYMBOLS,
            path=baseline_path,
            reset_baseline=False,
        )


def test_real_keys_in_perp_positions_detail_and_missing_price_detection(tmp_path: Path) -> None:
    """Dinh dang khoa that cua Binance (/fapi/v2/account & bookTicker) va phat hien lech equity khi thieu gia."""
    raw_positions_detail = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.0000",
            "entryPrice": "0.0",
            "markPrice": "62500.25",
            "unRealizedProfit": "0.00000000",
            "liquidationPrice": "0",
            "isolated": False,
        }
    ]
    snapshot = _make_faucet_snapshot()
    snapshot["perp_positions_detail"] = raw_positions_detail
    baseline_path = tmp_path / "testnet_baseline.json"
    key_hash = _dummy_hash()

    # 1. Prices co day du -> gia markPrice duoc trich xuat hop le
    prices = {"BTC/USDT:USDT": 62500.25}
    baseline = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=prices,
        api_key_hash=key_hash,
        universe_symbols=["BTC/USDT:USDT"],
        path=baseline_path,
        reset_baseline=True,
    )
    assert baseline["prices"]["BTC/USDT:USDT"] == 62500.25

    # 2. Khi co delta spot (mua 0.1 BTC) nhung bang gia thieu gia BTC (prices={}) -> fail-closed (EQUITY_MISMATCH)
    drift_snap = dict(snapshot)
    drift_snap["spot_balances"] = dict(snapshot["spot_balances"])
    drift_snap["spot_balances"]["BTC"] += 0.1
    drift_snap["spot_balances"]["USDT"] -= 6250.025

    delta_snap = compute_testnet_delta_snapshot(
        snapshot=drift_snap,
        baseline=baseline,
        prices={},
        capital=3000.0,
        local_equity=3000.0,
    )
    report = reconcile_portfolio_state(
        local_state={"current_equity": 3000.0, "carry_positions": {}},
        exchange_snapshot=delta_snap,
        prices={},
        lot_specs=EXCHANGE_LOT_SPECS,
        environment="testnet",
    )
    assert report["trading_allowed"] is False
    assert report["status"] in ("EQUITY_MISMATCH", "QTY_DRIFT")



def test_lookup_price_mainnet_invariance() -> None:
    """Item 2: _lookup_price tren mainnet/live giu nguyen tinh bat bien qua cac dinh dang khoa."""
    from src.exchange_reconciler import _lookup_price, _lookup_price_testnet_delta

    # Cac dinh dang khoa pho bien tren CCXT va Binance
    prices_full = {
        "BTC/USDT:USDT": 64200.5,
        "ETH/USDT": 3450.0,
        "SOLUSDT": 145.2,
        "BNB": 580.0,
    }

    # 1. Tra cuu chinh xac symbol canon
    assert _lookup_price(prices_full, "BTC/USDT:USDT", "BTC") == 64200.5
    # 2. Tra cuu qua base/USDT
    assert _lookup_price(prices_full, "ETH/USDT:USDT", "ETH") == 3450.0
    # 3. Tra cuu qua concat sym (SOLUSDT)
    assert _lookup_price(prices_full, "SOL/USDT:USDT", "SOL") == 145.2
    # 4. Tra cuu qua base asset (BNB)
    assert _lookup_price(prices_full, "BNB/USDT:USDT", "BNB") == 580.0

    # _lookup_price_testnet_delta cung cho ket qua tuong thich 100%
    assert _lookup_price_testnet_delta(prices_full, "BTC/USDT:USDT", "BTC") == 64200.5
    assert _lookup_price_testnet_delta(prices_full, "ETH/USDT:USDT", "ETH") == 3450.0
    assert _lookup_price_testnet_delta(prices_full, "SOL/USDT:USDT", "SOL") == 145.2
    assert _lookup_price_testnet_delta(prices_full, "BNB/USDT:USDT", "BNB") == 580.0


def test_audit_log_records_testnet_baseline_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Item 3: Su kien TESTNET_BASELINE_CREATED duoc ghi day du vao file audit JSONL."""
    import argparse
    from types import SimpleNamespace

    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "test_spot_key")
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", "test_spot_secret")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "test_fut_key")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", "test_fut_secret")

    audit_log = tmp_path / "audit_test.jsonl"
    baseline_path = tmp_path / "new_baseline.json"

    args = SimpleNamespace(
        mode="testnet",
        live_snapshot=True,
        capital=3000.0,
        audit_log=str(audit_log),
        testnet_baseline_file=str(baseline_path),
        reset_testnet_baseline=True,
        skip_testnet_baseline=False,
    )

    executor = SimpleNamespace(
        symbols=["BTC/USDT:USDT"],
        state=SimpleNamespace(current_equity=3000.0, carry_positions={}),
        apply_reconcile_report=lambda r: None,
    )

    faucet_snap = _make_faucet_snapshot()

    with patch("src.binance_signed_client.BinanceSignedTestnetClient.sync_server_time"), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_exchange_reconcile_snapshot", return_value=faucet_snap), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_book_ticker", return_value={"bid": 60000.0, "ask": 60000.0}):

        report = run_live._run_signed_snapshot_preflight(args, executor)

    assert report is not None
    assert report["status"] == "SYNCED"

    # Kiem tra file audit_log da duoc tao va chua record TESTNET_BASELINE_CREATED
    assert audit_log.exists()
    lines = audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1

    records = [json.loads(line) for line in lines]
    created_events = [r for r in records if r.get("event") == "TESTNET_BASELINE_CREATED"]
    assert len(created_events) == 1
    event = created_events[0]
    assert event["event"] == "TESTNET_BASELINE_CREATED"
    assert event["baseline_file"] == str(baseline_path)
    assert event["perp_positions_flat"] is True
    assert event["initial_recon_equity"] == pytest.approx(54999.12, abs=1e-2)
    assert "credentials_sha256" in event
    assert len(event["credentials_sha256"]) == 64


def test_skip_testnet_baseline_emits_prominent_warning_and_blocks_faucet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Item 7: --skip-testnet-baseline in canh bao to va doi soat truc tiep tren raw equity -> EQUITY_MISMATCH."""
    import logging
    from types import SimpleNamespace
    caplog.set_level(logging.WARNING)

    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "test_spot_key")
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", "test_spot_secret")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "test_fut_key")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", "test_fut_secret")

    args = SimpleNamespace(
        mode="testnet",
        live_snapshot=True,
        capital=3000.0,
        audit_log=None,
        testnet_baseline_file=str(tmp_path / "baseline.json"),
        reset_testnet_baseline=False,
        skip_testnet_baseline=True,
    )

    executor = SimpleNamespace(
        symbols=["BTC/USDT:USDT"],
        state=SimpleNamespace(current_equity=3000.0, carry_positions={}),
        apply_reconcile_report=lambda r: None,
    )

    faucet_snap = _make_faucet_snapshot()

    with patch("src.binance_signed_client.BinanceSignedTestnetClient.sync_server_time"), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_exchange_reconcile_snapshot", return_value=faucet_snap), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_book_ticker", return_value={"bid": 60000.0, "ask": 60000.0}):

        report = run_live._run_signed_snapshot_preflight(args, executor)

    assert report is not None
    # Vi bo qua baseline-delta, snapshot 55k USD vs capital 3k USD se bi chan boi EQUITY_MISMATCH
    assert report["status"] == "EQUITY_MISMATCH"
    assert report["trading_allowed"] is False
    assert "[WARNING] TESTNET BASELINE-DELTA IS SKIPPED" in caplog.text



def test_gate1_then_gate2_no_phantom_in_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sprint 22 Phase 9 Hotfix: Gate 1 (Overlay pass) roi Gate 2 (testnet-send-orders) khong bi PHANTOM_IN_STATE."""
    import pandas as pd
    from unittest.mock import MagicMock
    from src.carry_live_executor import CarryLiveExecutor

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    key64 = "a" * 64
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", key64)
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", key64)
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", key64)
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", key64)

    state_file = tmp_path / "bot_state.json"
    baseline_file = tmp_path / "testnet_baseline.json"
    gate1_audit = tmp_path / "gate1_audit.jsonl"
    gate2_audit = tmp_path / "gate2_audit.jsonl"

    faucet_snap = _make_faucet_snapshot()

    # Mock unauthenticated CCXT exchanges for testnet market data
    class MockCCXT:
        def __init__(self, *a, **kw):
            self.apiKey = None
            self.secret = None
            self.urls = {"api": {"public": "https://testnet.binance.vision/api/v3"}}

    # Mock overlay rebalance targets (simulating OverlayGate PASS: BTC +0.006, AVAX -53.3)
    def mock_build_barbell_allocations(self, aligned_universe, scores_at_t):
        self.last_overlay_delta = {"long_usd": 360.0, "short_usd": 360.0, "net_delta_usd": 0.0}
        self.last_overlay_neutrality = {
            "neutral_qty": {"BTC/USDT:USDT": 0.006, "AVAX/USDT:USDT": -53.3},
            "long_usd": 360.0,
            "short_usd": 360.0,
        }
        return {
            "momentum_overlay": {
                "overlay_active": True,
                "allocations": [
                    {"symbol": "BTC/USDT:USDT", "side": "buy", "qty": 0.006, "weight": 0.12},
                    {"symbol": "AVAX/USDT:USDT", "side": "sell", "qty": 53.3, "weight": -0.12},
                ],
            }
        }

    # Mock order pair execution
    order_pair_executed: list[dict[str, Any]] = []
    def mock_execute_live_testnet_order_pair(*a, **kw):
        record = {
            "status": "FILLED",
            "spot": {"orderId": 6145739},
            "perp": {"orderId": 28647724834},
        }
        order_pair_executed.append(dict(kw))
        return record

    with patch("ccxt.binance", side_effect=MockCCXT), \
         patch("ccxt.binanceusdm", side_effect=MockCCXT), \
         patch("ccxt.bybit", side_effect=MockCCXT), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.sync_server_time"), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_exchange_reconcile_snapshot", return_value=faucet_snap), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_book_ticker", return_value={"bid": 60000.0, "ask": 60000.0, "mid": 60000.0}), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.execute_live_testnet_order_pair", side_effect=mock_execute_live_testnet_order_pair), \
         patch.object(run_live, "_run_testnet_signing_preflight", return_value=True), \
         patch.object(CarryLiveExecutor, "build_barbell_allocations", mock_build_barbell_allocations), \
         patch.object(CarryLiveExecutor, "fetch_latest_aligned_universe", return_value={s: pd.DataFrame({"close": [60000.0]}, index=pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")) for s in UNIVERSE_SYMBOLS}):

        # --- 1. GATE 1 RUN ---
        rc1 = run_live.main([
            "--mode", "testnet",
            "--capital", "3000",
            "--cycles", "1",
            "--reset-testnet-baseline",
            "--state-file", str(state_file),
            "--testnet-baseline-file", str(baseline_file),
            "--audit-log", str(gate1_audit),
            "--live-snapshot",
        ])
        assert rc1 == 0

        # Kiem tra file state sau Gate 1: overlay_positions phai rong (khong bi o nhiem boi paper fill)
        assert state_file.exists()
        saved_state = json.loads(state_file.read_text(encoding="utf-8"))
        assert saved_state.get("overlay_positions") == {}
        assert saved_state.get("carry_positions") == {}
        assert baseline_file.exists()

        # Kiem tra audit log Gate 1: mode phai la 'testnet' (khong bi ghi de thanh 'paper')
        assert gate1_audit.exists()
        gate1_records = [json.loads(line) for line in gate1_audit.read_text(encoding="utf-8").strip().splitlines()]
        cycle_records = [r for r in gate1_records if "cycle" in r]
        assert len(cycle_records) >= 1
        assert cycle_records[0]["mode"] == "testnet"
        # Sprint 22 Phase 10: overlay telemetry fallback in testnet_dry_state
        assert cycle_records[0]["overlay_long_usd"] == pytest.approx(360.0)
        assert cycle_records[0]["overlay_short_usd"] == pytest.approx(360.0)
        assert cycle_records[0]["overlay_net_delta_usd"] == pytest.approx(0.0)

        # --- 2. GATE 2 RUN (chay tren cung state_file va baseline_file) ---
        rc2 = run_live.main([
            "--mode", "testnet",
            "--capital", "3000",
            "--cycles", "1",
            "--testnet-send-orders",
            "--state-file", str(state_file),
            "--testnet-baseline-file", str(baseline_file),
            "--audit-log", str(gate2_audit),
            "--live-snapshot",
        ])
        assert rc2 == 0

        # Kiem tra audit log Gate 2:
        # Reconcile dat SYNCED (tuyet doi khong bi PHANTOM_IN_STATE), lenh testnet duoc gui thanh cong (SENT)
        assert gate2_audit.exists()
        gate2_records = [json.loads(line) for line in gate2_audit.read_text(encoding="utf-8").strip().splitlines()]
        order_records = [r for r in gate2_records if str(r.get("event", "")).lower() == "testnet_order_pair"]
        assert len(order_records) == 1
        assert order_records[0]["status"] == "SENT"
        assert order_records[0]["testnet_order_ids"]["spot_entry_order_id"] == 6145739
        assert order_records[0]["testnet_order_ids"]["perp_entry_order_id"] == 28647724834



def test_testnet_multi_cycles_preflight_hook_keeps_synced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sprint 22 Phase 9: Multi-cycle --cycles 2 tren testnet kich hoat cycle_preflight_hook va ca 2 cycle deu dat SYNCED."""
    import pandas as pd
    from unittest.mock import MagicMock
    from src.carry_live_executor import CarryLiveExecutor

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    key64 = "b" * 64
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", key64)
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", key64)
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", key64)
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", key64)

    state_file = tmp_path / "bot_state_2c.json"
    baseline_file = tmp_path / "testnet_baseline_2c.json"
    audit_file = tmp_path / "audit_2c.jsonl"

    faucet_snap = _make_faucet_snapshot()

    class MockCCXT:
        def __init__(self, *a, **kw):
            self.apiKey = None
            self.secret = None
            self.urls = {"api": {"public": "https://testnet.binance.vision/api/v3"}}

    with patch("ccxt.binance", side_effect=MockCCXT), \
         patch("ccxt.binanceusdm", side_effect=MockCCXT), \
         patch("ccxt.bybit", side_effect=MockCCXT), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.sync_server_time"), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_exchange_reconcile_snapshot", return_value=faucet_snap), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_book_ticker", return_value={"bid": 60000.0, "ask": 60000.0, "mid": 60000.0}), \
         patch.object(run_live, "_run_testnet_signing_preflight", return_value=True), \
         patch.object(CarryLiveExecutor, "fetch_latest_aligned_universe", return_value={s: pd.DataFrame({"close": [60000.0]}, index=pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")) for s in UNIVERSE_SYMBOLS}):

        rc = run_live.main([
            "--mode", "testnet",
            "--capital", "3000",
            "--cycles", "2",
            "--interval-sec", "0.01",
            "--reset-testnet-baseline",
            "--state-file", str(state_file),
            "--testnet-baseline-file", str(baseline_file),
            "--audit-log", str(audit_file),
            "--live-snapshot",
        ])
        assert rc == 0

    assert audit_file.exists()
    records = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").strip().splitlines()]
    cycle_records = [r for r in records if "cycle" in r]
    assert len(cycle_records) == 2
    assert cycle_records[0]["cycle"] == 1
    assert cycle_records[0]["reconcile_status"] == "SYNCED"
    assert cycle_records[0]["mode"] == "testnet"
    assert cycle_records[1]["cycle"] == 2
    assert cycle_records[1]["reconcile_status"] == "SYNCED"
    assert cycle_records[1]["mode"] == "testnet"
