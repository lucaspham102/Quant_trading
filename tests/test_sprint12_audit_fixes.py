"""Regression test cho 5 lỗi vận hành được sửa sau đợt kiểm toán Sprint 12.

Phạm vi
-------
1. ``place_maker_leg1`` fail-closed ``SPREAD_UNKNOWN`` ở ``live``/``testnet``
   nhưng vẫn đi qua nhánh fallback giá ở ``paper``.
2. ``StateStore.load`` không xóa lãi/lỗ thực hiện khi file đã có
   ``initial_capital`` (capital sync chỉ cập nhật vốn gốc/peak).
3. State legacy không có key ``initial_capital`` vẫn được đồng bộ vốn.
4. Strategy mismatch vẫn giữ cờ ``kill_switch_triggered`` từ file cũ.
5. Ngưỡng audit delta drift co giãn theo lot step và cờ ``drift_unrepairable``
   chỉ được xóa khi drift vượt 5 lần ngưỡng.
6. Kill-switch được kiểm tra lại giữa chu kỳ ``reconcile_once``: nếu equity
   xuống dưới ngưỡng sau bước đóng lệnh thì không có lệnh mở mới nào được đặt.
7. Bất biến kế toán khi ``reinvest_profits=False``: ``gross_usd`` neo trên vốn
   gốc tĩnh nên ``sum(net_pnl) == final_equity - initial_capital``.

Nguyên tắc offline
------------------
Không gọi mạng, không ``sleep`` thật, không ghi ra ngoài ``tmp_path``. Mọi sàn
là đối tượng Python thuần ghi lại lời gọi ``create_order``. Dữ liệu Barbell
được tổng hợp tất định trong bộ nhớ (4 symbol x 400 giờ).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.carry_live_executor import CarryLiveExecutor
from src.config import CORE_4_SYMBOLS, TradingConfig
from src.cross_exchange_carry import (
    LOT_STEP_SIZES,
    CrossExchangeCarryConfig,
)
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import StateStore

PERP_SYMBOL = "BTC/USDT:USDT"
SPOT_SYMBOL = "BTC/USDT"
SECOND_SYMBOL = "ETH/USDT:USDT"
STEP_BTC = LOT_STEP_SIZES[PERP_SYMBOL]


# ---------------------------------------------------------------------------
# Helpers dùng chung
# ---------------------------------------------------------------------------


def _price_frame(price: float) -> pd.DataFrame:
    """DataFrame 1 dòng có ``perp_close``/``spot_close`` cho audit lấy giá."""
    idx = pd.date_range("2024-01-01 01:00", periods=1, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "close": float(price),
            "spot_close": float(price),
            "perp_close": float(price),
        },
        index=idx,
    ).astype("float64")


def _write_state_file(
    tmp_path: Path,
    payload: dict[str, Any],
    name: str = "state.json",
) -> Path:
    """Ghi payload JSON vào ``tmp_path`` và trả về đường dẫn."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_executor(
    tmp_path: Path,
    *,
    mode: str = "paper",
    capital: float = 3000.0,
    perp: Any = None,
    spot: Any = None,
) -> CarryLiveExecutor:
    """Tạo executor offline với state file nằm trong ``tmp_path``."""
    state_path = tmp_path / "executor_state.json"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=tmp_path / "raw",
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=DynamicCarryConfig(),
        mode=mode,  # type: ignore[arg-type]
        symbols=[PERP_SYMBOL, SECOND_SYMBOL],
        capital=capital,
        perp_exchange=perp,
        spot_exchange=spot,
        exchange_map={"binance": perp, "binance_spot": spot},
        state_store=StateStore(state_path),
        raw_data_dir=tmp_path / "raw",
    )


class FakeExchange:
    """Sàn giả ghi nhận lệnh, order book tĩnh do test điều khiển."""

    def __init__(
        self,
        bids: Sequence[Sequence[float]] | None = None,
        asks: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.bids = [list(level) for level in (bids or [[59990.0, 10]])]
        self.asks = [list(level) for level in (asks or [[60010.0, 10]])]

    def fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        """Trả order book tĩnh (best bid/ask do test đặt)."""
        return {
            "bids": [list(level) for level in self.bids],
            "asks": [list(level) for level in self.asks],
        }

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ghi nhận lệnh và trả trạng thái đã khớp đủ cho market."""
        self.orders.append(
            {
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": float(amount),
                "price": None if price is None else float(price),
                "params": dict(params) if params is not None else {},
            }
        )
        if type == "limit":
            return {"id": "leg1-1", "status": "open", "filled": 0.0}
        return {"id": f"m{len(self.orders)}", "status": "closed", "filled": float(amount)}

    def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Ghi nhận hủy lệnh; test không cần kiểm tra sâu ở file này."""
        return {"id": order_id, "status": "canceled"}


class RaisingBookExchange(FakeExchange):
    """Sàn giả mô phỏng ``fetch_order_book`` luôn ném lỗi (mất kết nối)."""

    def fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        """Luôn ném lỗi để buộc executor rơi vào nhánh SPREAD_UNKNOWN."""
        raise RuntimeError("order book unavailable (offline simulation)")


# ---------------------------------------------------------------------------
# 1. SPREAD_UNKNOWN fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["live", "testnet"])
def test_spread_unknown_fails_closed_in_live_and_testnet(
    tmp_path: Path,
    mode: str,
) -> None:
    """``fetch_order_book`` lỗi ở live/testnet -> skipped/SPREAD_UNKNOWN, không đặt lệnh."""
    perp = RaisingBookExchange()
    spot = RaisingBookExchange()
    executor = _make_executor(tmp_path, mode=mode, perp=perp, spot=spot)

    res = executor.place_maker_leg1(PERP_SYMBOL, "sell", 0.2, 60000.0)

    assert res["status"] == "skipped"
    assert res["reason"] == "SPREAD_UNKNOWN"
    assert res["filled"] == 0.0
    assert perp.orders == []
    assert spot.orders == []


def test_spread_unknown_paper_falls_back_to_reference_price(tmp_path: Path) -> None:
    """Cùng lỗi orderbook ở paper -> không bị chặn, dùng giá tham chiếu fallback."""
    perp = RaisingBookExchange()
    spot = RaisingBookExchange()
    executor = _make_executor(tmp_path, mode="paper", perp=perp, spot=spot)

    res = executor.place_maker_leg1(PERP_SYMBOL, "sell", 0.2, 60000.0)

    assert res.get("status") != "skipped"
    assert res["leg1_status"] == "FILLED"
    # Bán fallback tại ref_price * (1 + 0.00005) khi không có best ask.
    assert res["limit_price"] == pytest.approx(60000.0 * 1.00005)
    assert perp.orders == []
    assert spot.orders == []


# ---------------------------------------------------------------------------
# 2-4. StateStore.load: capital sync và kill-switch sticky
# ---------------------------------------------------------------------------


def test_state_load_preserves_realized_loss(tmp_path: Path) -> None:
    """File có ``initial_capital=3000`` và equity 2800 -> giữ nguyên lỗ 200 USD."""
    path = _write_state_file(
        tmp_path,
        {
            "initial_capital": 3000.0,
            "current_equity": 2800.0,
            "carry_positions": {},
            "overlay_positions": {},
        },
    )

    state = StateStore(path).load(initial_capital=3000.0)

    assert state.current_equity == pytest.approx(2800.0)
    assert state.initial_capital == pytest.approx(3000.0)
    assert state.peak_equity >= 3000.0


def test_state_load_syncs_legacy_file_without_initial_capital(tmp_path: Path) -> None:
    """State legacy (không có key ``initial_capital``) vẫn được đồng bộ về 3000."""
    path = _write_state_file(
        tmp_path,
        {
            "current_equity": 400.0,
            "carry_positions": {},
            "overlay_positions": {},
        },
    )

    state = StateStore(path).load(initial_capital=3000.0)

    assert state.current_equity == pytest.approx(3000.0)
    assert state.initial_capital == pytest.approx(3000.0)


def test_state_load_strategy_mismatch_keeps_kill_switch(tmp_path: Path) -> None:
    """Đổi chiến lược nhưng file cũ đã kill-switch -> giữ cờ True, equity reset 3000."""
    path = _write_state_file(
        tmp_path,
        {
            "strategy": "dynamic_carry",
            "kill_switch_triggered": True,
            "initial_capital": 3000.0,
            "current_equity": 2500.0,
            "carry_positions": {},
            "overlay_positions": {},
        },
    )

    state = StateStore(path).load(strategy="barbell", initial_capital=3000.0)

    assert state.kill_switch_triggered is True
    assert state.current_equity == pytest.approx(3000.0)
    assert state.initial_capital == pytest.approx(3000.0)
    assert state.strategy == "barbell"


# ---------------------------------------------------------------------------
# 5-6. Delta drift: ngưỡng theo lot step và cờ drift_unrepairable
# ---------------------------------------------------------------------------


def test_drift_threshold_scales_with_lot_step(tmp_path: Path) -> None:
    """BTC step 0,001 tại giá 60k: drift 12 USD bỏ qua, drift 120 USD sinh item."""
    executor = _make_executor(tmp_path)
    aligned = {PERP_SYMBOL: _price_frame(60000.0)}

    executor.state.carry_positions = {
        PERP_SYMBOL: {"qty": 1.0, "spot_qty": 1.0 + 12.0 / 60000.0}
    }
    assert executor.audit_delta_drift(aligned) == []

    executor.state.carry_positions = {
        PERP_SYMBOL: {"qty": 1.0, "spot_qty": 1.0 + 120.0 / 60000.0}
    }
    items = executor.audit_delta_drift(aligned)

    assert len(items) == 1
    assert items[0]["symbol"] == PERP_SYMBOL
    assert items[0]["drift_usd"] == pytest.approx(120.0)
    # Ngưỡng hiệu dụng phải lớn hơn min-notional 10 USD nhờ lot step.
    assert STEP_BTC * 60000.0 * 1.02 > 10.0


def test_drift_unrepairable_skips_until_five_times_threshold(tmp_path: Path) -> None:
    """Cờ ``drift_unrepairable`` bỏ qua drift 20 USD nhưng reset khi drift 400 USD."""
    executor = _make_executor(tmp_path)
    aligned = {PERP_SYMBOL: _price_frame(60000.0)}

    executor.state.carry_positions = {
        PERP_SYMBOL: {
            "qty": 1.0,
            "spot_qty": 1.0 + 20.0 / 60000.0,
            "drift_unrepairable": True,
        }
    }
    assert executor.audit_delta_drift(aligned) == []
    assert executor.state.carry_positions[PERP_SYMBOL]["drift_unrepairable"] is True

    executor.state.carry_positions[PERP_SYMBOL]["spot_qty"] = 1.0 + 400.0 / 60000.0
    items = executor.audit_delta_drift(aligned)

    assert len(items) == 1
    assert items[0]["drift_usd"] == pytest.approx(400.0)
    # Audit đã xóa cờ để chu kỳ sau có thể sửa tiếp nếu cần.
    assert "drift_unrepairable" not in executor.state.carry_positions[PERP_SYMBOL]


# ---------------------------------------------------------------------------
# 7. Kill-switch re-check giữa chu kỳ reconcile
# ---------------------------------------------------------------------------


def _score_frame(score: float, index: pd.DatetimeIndex) -> pd.DataFrame:
    """DataFrame điểm carry tối giản để ``reconcile_once`` đọc ``iloc[-1]``."""
    return pd.DataFrame(
        {"score": [float(score)], "f_ewma": [float(score)], "basis_dev": [0.0]},
        index=index,
    )


def test_kill_switch_mid_cycle_blocks_new_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Đóng lệnh lỗ làm equity thủng ngưỡng -> không có lệnh OPEN nào sau đó.

    Kịch bản: BTC đang giữ bị buộc CLOSE (score dưới ngưỡng thoát), khoản lỗ MTM
    khi đóng đưa equity từ 3000 xuống ~2700. Kill-switch phải được kiểm tra lại
    ngay sau bước đóng và chặn toàn bộ lệnh mở ETH ở bước 5c.
    """
    perp = FakeExchange()
    spot = FakeExchange()
    executor = _make_executor(tmp_path, mode="paper", perp=perp, spot=spot)

    idx = pd.date_range("2024-01-01 01:00", periods=1, freq="h", tz="UTC").as_unit("ns")
    idx.name = "datetime"
    universe = {
        PERP_SYMBOL: pd.DataFrame(
            {
                "close": [50.0],
                "spot_close": [50.0],
                "perp_close": [300.0],
                "funding_rate_event": [0.0],
            },
            index=idx,
        ),
        SECOND_SYMBOL: pd.DataFrame(
            {
                "close": [20.0],
                "spot_close": [20.0],
                "perp_close": [20.0],
                "funding_rate_event": [0.0],
            },
            index=idx,
        ),
    }

    executor.state.current_equity = 3000.0
    executor.state.peak_equity = 3000.0
    executor.state.daily_start_equity = 3000.0
    executor.state.kill_switch_triggered = False
    executor.kill_switch_triggered = False
    executor.state.overlay_positions = {}
    executor.state.carry_positions = {
        PERP_SYMBOL: {
            "symbol": PERP_SYMBOL,
            "spot_symbol": SPOT_SYMBOL,
            "qty": 1.0,
            "spot_qty": 1.0,
            "entry_spot_price": 150.0,
            "entry_perp_price": 100.0,
            "notional_usdt": 100.0,
            "margin_allocated_usdt": 0.0,
            "periods_held": 10,
            "funding_collected_usdt": 0.0,
            "leg1_status": "NONE",
            "leg1_order_id": None,
            "leg1_filled_qty": 0.0,
            "leg2_hedged_qty": 0.0,
            "exchange_long": "binance_spot",
            "exchange_short": "binance",
            "capital": 3000.0,
            "execution_state": "IDLE",
        }
    }

    place_calls: list[dict[str, Any]] = []
    overlay_calls: list[dict[str, Any]] = []
    repair_calls: list[dict[str, Any]] = []

    def fake_place_maker_leg1(
        symbol: str,
        side: str,
        target_qty: float,
        ref_price: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Đóng vai đặt lệnh thành công, đồng thời cập nhật leg state cho bước hedge."""
        is_close = bool(kwargs.get("is_close", False))
        place_calls.append(
            {"symbol": symbol, "side": side, "is_close": is_close, "qty": float(target_qty)}
        )
        pos = executor.state.carry_positions.setdefault(symbol, {"symbol": symbol})
        pos["spot_symbol"] = kwargs.get("spot_symbol") or symbol.split(":")[0]
        pos["side"] = side
        pos["leg1_status"] = "FILLED"
        pos["leg1_order_id"] = "fake-leg1"
        pos["leg1_filled_qty"] = float(target_qty)
        pos["leg2_hedged_qty"] = 0.0
        pos["execution_state"] = "POST_ONLY_LEG_1"
        return {
            "leg1_status": "FILLED",
            "leg1_order_id": "fake-leg1",
            "leg1_filled_qty": float(target_qty),
            "target_qty": float(target_qty),
            "limit_price": float(ref_price),
        }

    def fake_execute_overlay_rebalance(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Ghi nhận lời gọi overlay, không đặt lệnh thật."""
        overlay_calls.append({"args": args, "kwargs": kwargs})
        return {"orders": [], "n_orders": 0, "skipped": 0, "turnover_notional": 0.0, "functional": False}

    def fake_repair_delta_drift(item: dict[str, Any]) -> dict[str, Any]:
        """Ghi nhận lời gọi sửa drift; kịch bản này không kỳ vọng có drift."""
        repair_calls.append(dict(item))
        return {"status": "repaired", "symbol": item.get("symbol"), "qty": 0.0, "notional": 0.0}

    def fake_scores(
        aligned_universe: dict[str, pd.DataFrame],
        config: DynamicCarryConfig,
    ) -> dict[str, pd.DataFrame]:
        """BTC dưới ngưỡng thoát (CLOSE), ETH trên ngưỡng vào (OPEN)."""
        return {
            PERP_SYMBOL: _score_frame(0.0, idx),
            SECOND_SYMBOL: _score_frame(0.001, idx),
        }

    monkeypatch.setattr(executor, "place_maker_leg1", fake_place_maker_leg1)
    monkeypatch.setattr(executor, "execute_overlay_rebalance", fake_execute_overlay_rebalance)
    monkeypatch.setattr(executor, "repair_delta_drift", fake_repair_delta_drift)
    monkeypatch.setattr("src.carry_live_executor.build_carry_basis_scores", fake_scores)

    plan = executor.reconcile_once(universe)

    open_calls = [call for call in place_calls if not call["is_close"]]
    close_calls = [call for call in place_calls if call["is_close"]]
    assert len(close_calls) == 1
    assert close_calls[0]["symbol"] == PERP_SYMBOL
    assert open_calls == [], "Kill-switch giữa chu kỳ phải chặn mọi lệnh OPEN"
    assert repair_calls == []
    assert len(overlay_calls) == 1

    assert plan.get("kill_switch") is True
    assert executor.state.kill_switch_triggered is True


# ---------------------------------------------------------------------------
# 8. Bất biến kế toán khi vốn neo tĩnh
# ---------------------------------------------------------------------------


INITIAL_CAPITAL = 3000.0
CORE4 = list(CORE_4_SYMBOLS)
TREND_START = "2024-01-01 00:00:00"
FUNDING_RATE = 0.0005


def _hourly_index(periods: int) -> pd.DatetimeIndex:
    """DatetimeIndex UTC ns tên ``datetime`` với tần suất 1 giờ."""
    index = pd.date_range(TREND_START, periods=periods, freq="h", tz="UTC").as_unit("ns")
    index.name = "datetime"
    return index


def _price_path(returns: np.ndarray, start: float) -> np.ndarray:
    """Ghép chuỗi giá từ returns 1h cho trước."""
    return start * np.cumprod(1.0 + np.asarray(returns, dtype="float64"))


def _symbol_frame(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    basis_pct: np.ndarray,
    funding_rate: float = FUNDING_RATE,
) -> pd.DataFrame:
    """Dựng DataFrame đủ schema cho cả Carry Tier 1A lẫn Momentum Overlay."""
    close_arr = np.asarray(close, dtype="float64")
    n = len(close_arr)
    frame = pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr * 1.0005,
            "low": close_arr * 0.9995,
            "close": close_arr,
            "volume": np.full(n, 1000.0, dtype="float64"),
            "perp_close": close_arr,
            "spot_close": close_arr,
            "basis_pct": np.asarray(basis_pct, dtype="float64"),
            "funding_event_binance": np.full(n, funding_rate, dtype="float64"),
            "funding_rate_event": np.full(n, funding_rate, dtype="float64"),
            "ewma_funding_binance": np.full(n, funding_rate, dtype="float64"),
        },
        index=index.copy(),
    )
    frame.index.name = "datetime"
    return frame


def _trend_data(periods: int = 400) -> dict[str, pd.DataFrame]:
    """4 symbol Core-4 trend phân hóa trong 400 giờ, nhiễu xen kẽ dùng chung."""
    index = _hourly_index(periods)
    steps = np.arange(periods, dtype="float64")
    noise = 2e-4 * np.where(steps % 2 == 0, 1.0, -1.0)
    drifts = (0.0004, 0.00015, -0.00015, -0.0004)
    starts = (65000.0, 3000.0, 150.0, 600.0)

    data: dict[str, pd.DataFrame] = {}
    for i, (sym, drift, start) in enumerate(zip(CORE4, drifts, starts)):
        close = _price_path(drift + noise, start)
        basis = 0.001 * np.sin(2.0 * np.pi * steps / 24.0 + 0.3 * i)
        data[sym] = _symbol_frame(index, close, basis)
    return data


def _static_capital_config() -> BarbellConfig:
    """Barbell có overlay luôn bật nhưng vốn gốc tĩnh (``reinvest_profits=False``)."""
    return BarbellConfig(
        initial_capital=INITIAL_CAPITAL,
        carry_config=CrossExchangeCarryConfig(initial_capital=INITIAL_CAPITAL),
        carry_variant="tier1a",
        gating_mode="hard_binary",
        overlay_gross_leverage=0.40,
        overlay_lookback_hours=48,
        overlay_rebalance_hours=48,
        overlay_top_k=1,
        overlay_buffer_rank=1,
        overlay_min_spread=0.75,
        enable_snr_gate=False,
        enable_vol_targeting=False,
        enable_drawdown_cushion=False,
        reinvest_profits=False,
    )


def test_static_capital_keeps_attribution_invariant() -> None:
    """``reinvest_profits=False``: tổng ``net_pnl`` vẫn khép kín về chênh lệch vốn."""
    res = run_barbell_portfolio(_trend_data(400), _static_capital_config())
    attr = res["attribution_series"]
    equity = res["equity_curve"]

    assert res["reinvest_profits"] is False
    assert float(equity.iloc[0]) == pytest.approx(INITIAL_CAPITAL)
    assert not attr.empty

    total_net_pnl = float(attr["net_pnl"].sum())
    final_equity = float(equity.iloc[-1])
    gap = abs(total_net_pnl - (final_equity - INITIAL_CAPITAL))

    assert gap < 1e-6, (
        "Vốn neo tĩnh làm vỡ bất biến kế toán: "
        f"sum(net_pnl)={total_net_pnl:.10f}, "
        f"final_equity-initial={final_equity - INITIAL_CAPITAL:.10f}, gap={gap:.3e}"
    )
