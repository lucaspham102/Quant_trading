"""Sprint 23: Unit tests cho canary 400 USDT, Spot Valuation Guard va persistent positions.

Bao phu cac kich ban bat buoc:
1. Spot Valuation Guard tu dong backfill gia tu bookTicker (bid/ask) khi bang gia trong.
2. Guard raise SpotValuationPriceMissingError khi khong co bat ky nguon gia nao.
3. Tai san faucet ngoai universe (ZIL) khong lam snapshot fail.
4. resolve_effective_carry_top_k co gian theo von (400 -> 2 slot, >=1500 -> base).
5. filter_canary_lot_unviable_overlay_symbols loai BTC khi budget 60, giu nguyen khi 500.
6. _execute_atomic_carry_delta mo vi the thanh cong (Spot BUY -> Perp SELL).
7. _execute_atomic_carry_delta dong vi the thanh cong (Perp BUY reduceOnly -> Spot SELL).
8. Qty duoi lot step -> QTY_BELOW_LOT_STEP, khong gui lenh.
9. Naked-Leg Auto-Rollback: perp loi 2 lan, spot duoc ban bu lai dung so luong.
10. Persistent positions: testnet dry state xoa vi the, che do that giu qua restart.
11. CLI --persistent-positions parse dung.
12. Perp MARKET tra executedQty=0 (testnet) -> poll query_order, KHONG retry (khong short doi).
13. Delta snapshot khong double-count gia tri Spot khi co pure_cash_usd.
14. Sai so float cua phep tru baseline khong bi bao PHANTOM_IN_STATE.
15. Rollback khi dong carry phai MO vi the Short moi (reduce_only=False, tranh -2022).
16. Preflight mode live khong nap .env.testnet va fail-closed khi thieu client mainnet.
17. Perp nem Exception nhung vi the da khop -> verify qua snapshot, khong retry mu.
18. _await_fill bo qua loi poll tam thoi roi van thay FILLED.
19. Overlay perp thuc: executedQty=0 -> poll query_order va ghi state dung.

Khong co network, khong dung wall-clock, moi so sanh float dung pytest.approx.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import run_live
from src.binance_signed_client import (
    BinanceSignedTestnetClient,
    SpotValuationPriceMissingError,
)
from src.carry_live_executor import CarryLiveExecutor
from src.dynamic_carry_basis import DynamicCarryConfig
from src.quant_invariants import (
    filter_canary_lot_unviable_overlay_symbols,
    resolve_effective_carry_top_k,
)
from src.state_store import StateStore

BTC = "BTC/USDT:USDT"
POSITION: dict[str, Any] = {
    "symbol": BTC,
    "qty": 0.01,
    "entry_spot_price": 60000.0,
    "entry_perp_price": 60000.0,
    "notional_usdt": 600.0,
    "margin_allocated_usdt": 100.0,
}


def _mock_responses(spot_balances: dict[str, float]) -> dict[str, Any]:
    """Tao mock_responses cho spot_account / futures_account / position_risk."""
    return {
        "spot_account": {
            "balances": [
                {"asset": asset, "free": str(qty)} for asset, qty in spot_balances.items()
            ]
        },
        "futures_account": {
            "totalWalletBalance": "0.0",
            "totalUnrealizedProfit": "0.0",
        },
        "position_risk": [],
    }


def _live_client() -> BinanceSignedTestnetClient:
    """Client dry_run=False voi khoa gia, khong bao gio goi HTTP vi co mock_responses."""
    return BinanceSignedTestnetClient(
        api_key="k" * 64,
        api_secret="s" * 64,
        dry_run=False,
    )


def _executor(tmp_path: Path, client: Any, **kwargs: Any) -> CarryLiveExecutor:
    """Tao CarryLiveExecutor chuan cho kich ban canary 400 USDT."""
    return CarryLiveExecutor(
        carry_config=DynamicCarryConfig(initial_capital=400.0, top_k=4, eta_override=0.80),
        mode=kwargs.pop("mode", "testnet"),
        capital=400.0,
        initial_capital=400.0,
        state_store=StateStore(tmp_path / "state.json"),
        runtime_mode=kwargs.pop("runtime_mode", "testnet"),
        testnet_dry_state=kwargs.pop("testnet_dry_state", False),
        signed_client=client,
        execute_real_orders=kwargs.pop("execute_real_orders", True),
        persistent_positions=kwargs.pop("persistent_positions", False),
        **kwargs,
    )


class _FakeSignedClient:
    """Ghi lai moi lenh va luon fill day du."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        return {"executedQty": qty, "avgPrice": 81000.0, "orderId": 1}

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        return {"executedQty": qty, "avgPrice": 81000.0, "orderId": 2}


class _PerpDownClient(_FakeSignedClient):
    """Perp luon loi de kich hoat Naked-Leg Auto-Rollback."""

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        raise RuntimeError("perp down")


def test_spot_valuation_guard_backfills_price(tmp_path: Path) -> None:
    """1. Gia trong nhung co bookTicker -> tu backfill BTC va tinh lai spot_cash_usd."""
    client = _live_client()
    client.fetch_book_ticker = lambda symbol, venue="spot", timeout_sec=5.0: {
        "bid": 60000.0,
        "ask": 60002.0,
        "mid": 60001.0,
    }
    snap = client.fetch_exchange_reconcile_snapshot(
        [BTC],
        prices={},
        mock_responses=_mock_responses({"USDT": 100.0, "BTC": 0.05}),
    )
    assert snap["snapshot_healthy"] is True
    assert BTC in snap["backfilled_price_symbols"]
    assert snap["spot_cash_usd"] == pytest.approx(100.0 + 0.05 * 60001.0)


def test_spot_valuation_guard_raises_when_no_price(tmp_path: Path) -> None:
    """2. Khong co gia tu bat ky nguon nao -> SpotValuationPriceMissingError."""
    client = _live_client()
    client.fetch_book_ticker = lambda symbol, venue="spot", timeout_sec=5.0: {
        "bid": 0.0,
        "ask": 0.0,
        "mid": 0.0,
    }
    with pytest.raises(SpotValuationPriceMissingError):
        client.fetch_exchange_reconcile_snapshot(
            [BTC],
            prices={},
            mock_responses=_mock_responses({"USDT": 100.0, "BTC": 0.05}),
        )


def test_untracked_faucet_dust_does_not_raise(tmp_path: Path) -> None:
    """3. Tai san faucet ngoai universe (ZIL) chi vao untracked_spot_assets, khong raise."""
    client = _live_client()
    snap = client.fetch_exchange_reconcile_snapshot(
        [BTC],
        prices={},
        mock_responses=_mock_responses({"USDT": 100.0, "ZIL": 18446.0}),
    )
    assert snap["snapshot_healthy"] is True
    assert snap["untracked_spot_assets"]["ZIL"] == pytest.approx(18446.0)


def test_resolve_effective_carry_top_k_scales_with_capital() -> None:
    """4. 400 USDT -> 2 slot; 1500/3000 USDT -> base_top_k=4; von 0 -> 1."""
    assert resolve_effective_carry_top_k(400.0, btc_price=81000.0) == 2
    assert resolve_effective_carry_top_k(400.0, btc_price=85000.0) == 2
    assert resolve_effective_carry_top_k(3000.0) == 4
    assert resolve_effective_carry_top_k(1500.0) == 4
    assert resolve_effective_carry_top_k(0.0) == 1


def test_filter_lot_unviable_overlay_symbols() -> None:
    """5. Budget 60 loai BTC va renormalize long; budget 500 giu nguyen 100%."""
    weights = {
        BTC: 0.1,
        "ETH/USDT:USDT": 0.05,
        "SOL/USDT:USDT": -0.05,
        "DOGE/USDT:USDT": -0.05,
    }
    prices = {
        BTC: 85000.0,
        "ETH/USDT:USDT": 3000.0,
        "SOL/USDT:USDT": 150.0,
        "DOGE/USDT:USDT": 0.15,
    }
    lot_steps = {
        BTC: 0.001,
        "ETH/USDT:USDT": 0.001,
        "SOL/USDT:USDT": 0.01,
        "DOGE/USDT:USDT": 1.0,
    }

    small = filter_canary_lot_unviable_overlay_symbols(weights, 60.0, prices, lot_steps)
    assert BTC not in small
    assert small["ETH/USDT:USDT"] == pytest.approx(0.15)
    assert small["SOL/USDT:USDT"] == pytest.approx(-0.05)
    assert small["DOGE/USDT:USDT"] == pytest.approx(-0.05)

    large = filter_canary_lot_unviable_overlay_symbols(weights, 500.0, prices, lot_steps)
    assert large == pytest.approx(weights)


def test_atomic_carry_open_filled(tmp_path: Path) -> None:
    """6. Mo vi the: Spot BUY truoc, Perp SELL sau, event ATOMIC_CARRY_OPEN_FILLED."""
    fake = _FakeSignedClient()
    ex = _executor(tmp_path, fake)
    res = ex._execute_atomic_carry_delta(BTC, 0.0015, -0.0015, 81000.0)

    assert res["ok"] is True
    assert res["spot_filled_qty"] == pytest.approx(0.001)
    assert res["perp_filled_qty"] == pytest.approx(0.001)
    assert res["event"] == "ATOMIC_CARRY_OPEN_FILLED"
    assert fake.calls == [("spot", "BUY", 0.001), ("perp", "SELL", 0.001, False)]


def test_atomic_carry_close_filled(tmp_path: Path) -> None:
    """7. Dong vi the: Perp BUY reduceOnly truoc, Spot SELL sau, event CLOSE_FILLED."""
    fake = _FakeSignedClient()
    ex = _executor(tmp_path, fake)
    res = ex._execute_atomic_carry_delta(BTC, -0.0015, 0.0015, 81000.0)

    assert res["ok"] is True
    assert res["event"] == "ATOMIC_CARRY_CLOSE_FILLED"
    assert fake.calls[0] == ("perp", "BUY", 0.001, True)
    assert fake.calls[1] == ("spot", "SELL", 0.001)
    assert len(fake.calls) == 2


def test_atomic_carry_qty_below_lot_step(tmp_path: Path) -> None:
    """8. Qty 0.0009 < lot step 0.001 -> QTY_BELOW_LOT_STEP va khong gui lenh."""
    fake = _FakeSignedClient()
    ex = _executor(tmp_path, fake)
    res = ex._execute_atomic_carry_delta(BTC, 0.0009, -0.0009, 85000.0)

    assert res["ok"] is False
    assert res["reason"] == "QTY_BELOW_LOT_STEP"
    assert fake.calls == []


def test_naked_leg_auto_rollback(tmp_path: Path) -> None:
    """9. Perp loi 2 lan -> ban bu spot dung filled qty va raise RuntimeError."""
    fake = _PerpDownClient()
    ex = _executor(tmp_path, fake)

    with pytest.raises(RuntimeError, match="Naked leg rollback triggered"):
        ex._execute_atomic_carry_delta(BTC, 0.002, -0.002, 81000.0)

    assert fake.calls[-1] == ("spot", "SELL", 0.002)
    assert sum(1 for c in fake.calls if c[0] == "perp") == 2


def _seed_carry_state(path: Path) -> None:
    """Ghi 1 vi the BTC co san vao file state de test reload."""
    store = StateStore(path)
    state = store.load(default_equity=400.0)
    state.carry_positions = {BTC: dict(POSITION)}
    store.save(state)


def test_persistent_positions_reload(tmp_path: Path) -> None:
    """10. Testnet dry state xoa vi the; che do that giu vi the qua restart."""
    state_path = tmp_path / "state.json"
    carry_config = DynamicCarryConfig(initial_capital=400.0, top_k=4, eta_override=0.80)

    _seed_carry_state(state_path)
    ex_a = CarryLiveExecutor(
        carry_config=carry_config,
        mode="testnet",
        capital=400.0,
        initial_capital=400.0,
        state_store=StateStore(state_path),
        runtime_mode="testnet",
        testnet_dry_state=True,
    )
    assert ex_a.state.carry_positions == {}
    assert json.loads(state_path.read_text(encoding="utf-8"))["carry_positions"] == {}

    _seed_carry_state(state_path)
    ex_b = CarryLiveExecutor(
        carry_config=carry_config,
        mode="testnet",
        capital=400.0,
        initial_capital=400.0,
        state_store=StateStore(state_path),
        runtime_mode="testnet",
        testnet_dry_state=False,
        signed_client=_FakeSignedClient(),
        execute_real_orders=True,
        persistent_positions=True,
    )
    # Sprint 23: dinh tuyen lenh THAT -> khong bi ep dry-state, vi the duoc giu qua restart.
    assert ex_b.testnet_dry_state is False
    assert BTC in ex_b.state.carry_positions


def test_cli_persistent_positions_flag() -> None:
    """11. parse_args: mac dinh False, co --persistent-positions thi True."""
    assert run_live.parse_args([]).persistent_positions is False
    args = run_live.parse_args(["--mode", "testnet", "--persistent-positions"])
    assert args.persistent_positions is True


class _SlowFillClient(_FakeSignedClient):
    """Testnet that: Perp MARKET tra executedQty=0 luc dau, query_order moi bao FILLED."""

    def __init__(self) -> None:
        super().__init__()
        self.queried: list[tuple[Any, ...]] = []

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        return {"executedQty": 0.0, "avgPrice": 0.0, "orderId": 777, "status": "NEW"}

    def query_order(
        self, venue: str, symbol: str, order_id: Any, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.queried.append((venue, symbol, order_id))
        return {"executedQty": 0.001, "avgPrice": 81000.0, "status": "FILLED", "orderId": order_id}


def test_atomic_open_polls_zero_fill_without_retry(tmp_path: Path) -> None:
    """12. Perp executedQty=0 -> poll query_order thay FILLED, chi gui 1 lenh perp (khong short doi)."""
    fake = _SlowFillClient()
    ex = _executor(tmp_path, fake)
    res = ex._execute_atomic_carry_delta(BTC, 0.0015, -0.0015, 81000.0)

    assert res["ok"] is True
    assert res["event"] == "ATOMIC_CARRY_OPEN_FILLED"
    assert res["perp_filled_qty"] == pytest.approx(0.001)
    assert sum(1 for c in fake.calls if c[0] == "perp") == 1
    assert fake.queried and fake.queried[0][0] == "futures"


def test_delta_snapshot_no_double_count_with_pure_cash() -> None:
    """13. spot_cash_usd la tong dinh gia -> KHONG cong them delta_spot_market_value lan 2."""
    from src.exchange_reconciler import compute_testnet_delta_snapshot

    eth = "ETH/USDT:USDT"
    # Baseline: 10,000 USDT cash + 500 wallet; mua 0.002 BTC (160 USD) -> cash 9,840.
    baseline = {
        "initial_recon_equity": 10500.0,
        "spot_balances": {"BTC": 1.0, "ETH": 1.0},
        "perp_positions": {},
    }
    snapshot = {
        "snapshot_healthy": True,
        "spot_balances": {"BTC": 1.002, "ETH": 1.0},
        "pure_cash_usd": 9840.0,
        "spot_cash_usd": 9840.0 + 1.002 * 80000.0 + 1.0 * 2500.0,
        "perp_wallet_balance_usd": 500.0,
        "perp_unrealized_pnl_usd": 0.0,
    }
    prices = {BTC: 80000.0, eth: 2500.0}

    delta = compute_testnet_delta_snapshot(snapshot, baseline, prices, 400.0, 400.0)

    assert delta["spot_balances"]["BTC"] == pytest.approx(0.002)
    # Khong double count: reconcile equity == local equity (400) thay vi 400 + 160.
    assert delta["reconcile_equity_usd"] == pytest.approx(400.0)


def test_classify_leg_float_artifact_not_phantom() -> None:
    """14. 1.001 - 1.0 = 0.0009999999999998899 van la SYNCED, khong phai PHANTOM_IN_STATE."""
    from src.exchange_reconciler import SYNCED, _classify_leg

    status, action, qty, _reason = _classify_leg(
        0.001, 1.001 - 1.0, "spot", 80000.0, 0.001, 5.0, 10.0
    )
    assert status == SYNCED
    assert action is None
    assert qty == pytest.approx(0.0)


class _SpotZeroFillClient(_FakeSignedClient):
    """Spot SELL tra executedQty=0 (khong co query_order) -> kich hoat rehedge perp."""

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        if side == "SELL":
            return {"executedQty": 0.0, "avgPrice": 0.0, "orderId": 3}
        return {"executedQty": qty, "avgPrice": 81000.0, "orderId": 1}


def test_close_rollback_rehedges_without_reduce_only(tmp_path: Path) -> None:
    """15. Spot chan 2 khong khop -> rehedge perp phai la lenh MO (reduce_only=False)."""
    fake = _SpotZeroFillClient()
    ex = _executor(tmp_path, fake)
    res = ex._execute_atomic_carry_delta(BTC, -0.002, 0.002, 81000.0)

    assert res["event"] == "NAKED_LEG_CLOSE_ROLLBACK"
    assert fake.calls[0] == ("perp", "BUY", 0.002, True)
    # Sprint 23: rehedge KHONG duoc reduceOnly (vi the perp da flat sau leg 1).
    assert fake.calls[-1] == ("perp", "SELL", 0.002, False)


def test_preflight_live_mode_never_loads_testnet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """16. mode live + khong co signed_client mainnet -> BLOCKED, khong nap .env.testnet."""

    class _Ex:
        symbols = [BTC]
        state = type("S", (), {"carry_positions": {}})()

    def _boom() -> None:
        raise AssertionError("load_dotenv_testnet KHONG duoc goi o mode live")

    monkeypatch.setattr(run_live, "load_dotenv_testnet", _boom, raising=False)
    args = run_live.parse_args(["--mode", "live"])
    report = run_live._run_signed_snapshot_preflight(args, _Ex())

    assert report is not None
    assert report["status"] == "SNAPSHOT_CREDENTIALS_MISSING"
    assert report["trading_allowed"] is False


class _PerpExceptionButFilledClient(_FakeSignedClient):
    """Perp nem timeout nhung vi the that da khop (testnet hay gap)."""

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        raise TimeoutError("read timeout")

    def fetch_exchange_reconcile_snapshot(
        self, symbols: list[str], prices: dict[str, float], **kwargs: Any
    ) -> dict[str, Any]:
        return {"snapshot_healthy": True, "perp_positions": {BTC: -0.001}, "spot_balances": {}}


def test_perp_exception_verified_filled_no_retry(tmp_path: Path) -> None:
    """17. Exception nhung vi the perp da -0.001 -> coi nhu khop, khong retry, khong rollback."""
    fake = _PerpExceptionButFilledClient()
    ex = _executor(tmp_path, fake)
    res = ex._execute_atomic_carry_delta(BTC, 0.0015, -0.0015, 81000.0)

    assert res["ok"] is True
    assert res["perp_filled_qty"] == pytest.approx(0.001)
    assert sum(1 for c in fake.calls if c[0] == "perp") == 1
    assert sum(1 for c in fake.calls if c[0] == "spot") == 1  # khong ban bu spot


class _FlakyQueryClient(_FakeSignedClient):
    """query_order loi lan dau roi tra FILLED."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        return {"executedQty": 0.0, "avgPrice": 0.0, "orderId": 99, "status": "NEW"}

    def query_order(
        self, venue: str, symbol: str, order_id: Any, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.n += 1
        if self.n == 1:
            raise ConnectionError("rate limit")
        return {"executedQty": 0.001, "avgPrice": 81000.0, "status": "FILLED", "orderId": order_id}


def test_await_fill_tolerates_transient_poll_error(tmp_path: Path) -> None:
    """18. Poll loi tam thoi -> thu lai va van thay FILLED, khong rollback sai."""
    fake = _FlakyQueryClient()
    ex = _executor(tmp_path, fake)
    res = ex._execute_atomic_carry_delta(BTC, 0.0015, -0.0015, 81000.0)

    assert res["ok"] is True
    assert res["perp_filled_qty"] == pytest.approx(0.001)
    assert fake.n >= 2
    assert sum(1 for c in fake.calls if c[0] == "perp") == 1


class _OverlaySlowFillClient(_FakeSignedClient):
    """Overlay perp tra executedQty=0 luc dau, query_order bao FILLED."""

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        return {"executedQty": 0.0, "avgPrice": 0.0, "orderId": 321}

    def query_order(
        self, venue: str, symbol: str, order_id: Any, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        return {"executedQty": 0.01, "avgPrice": 80000.0, "status": "FILLED", "orderId": order_id}


def test_overlay_real_order_polls_zero_fill(tmp_path: Path) -> None:
    """19. Overlay: executedQty=0 nhung poll thay FILLED -> state overlay_positions duoc ghi."""
    fake = _OverlaySlowFillClient()
    ex = _executor(tmp_path, fake)
    # Nang nguong imbalance de co lap test rieng cho co che poll (khong kich hoat rollback).
    ex.overlay_max_imbalance_usd = 5000.0
    res = ex._execute_overlay_deltas_real({BTC: 0.01}, {}, {BTC: 80000.0})

    assert res["status"] != "ROLLED_BACK"
    assert res["orders"][0]["filled_qty"] == pytest.approx(0.01)
    assert res["orders"][0]["status"] == "filled"
    assert ex.state.overlay_positions[BTC]["side"] == "long"
    assert ex.state.overlay_positions[BTC]["qty"] == pytest.approx(0.01)
