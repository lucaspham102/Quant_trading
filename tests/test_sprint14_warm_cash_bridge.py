"""Test Sprint 14 - Warm-cash holdout bridge (Method D) va cong gate trade_start_idx.

Bao phu:
1. Gate `trade_start_idx` tren engine carry: equity phang 100% tien mat truoc moc, khong co lenh truoc moc.
2. Gate tuong tu o tang barbell (overlay weight = 0 truoc moc).
3. `run_indicator_warm_cash_start` tra ve dung cau truc va giu von = capital tai moc holdout.
4. Bridge 4 phuong phap: dang thuc B - C == (B - D) + (D - C) voi sai so < 1e-6.
5. Phan ra 3 thanh phan (phi mo, do mu chi bao, chenh lech ton kho) cong lai dung tong.
6. Tuong thich nguoc: khong truyen warm_cash thi bridge giu nguyen hanh vi Sprint 13.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.cross_exchange_carry import CrossExchangeCarryConfig, run_binance_tier1_plus_carry
from src.sprint11_eval import (
    SPRINT14_WARMUP_HOURS,
    compute_holdout_accounting_bridge,
    run_indicator_warm_cash_start,
)

BASE_PRICES = {
    "BTC/USDT:USDT": 60_000.0,
    "ETH/USDT:USDT": 3_000.0,
    "SOL/USDT:USDT": 150.0,
    "BNB/USDT:USDT": 550.0,
}
CORE_4 = list(BASE_PRICES)


def _hourly_index(periods: int, start: str = "2024-01-01 00:00:00") -> pd.DatetimeIndex:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC").as_unit("ns")
    idx.name = "datetime"
    return idx


def _make_frame(
    idx: pd.DatetimeIndex,
    base_price: float,
    funding_rate: float = 0.0003,
) -> pd.DataFrame:
    n = len(idx)
    phase = np.arange(n, dtype=np.float64)
    perp = base_price * (1.0 + 0.01 * np.sin(phase / 7.0))
    basis_pct = 0.001 + 0.0002 * np.sin(phase / 5.0)
    spot = perp / (1.0 + basis_pct)
    is_8h = (idx.hour % 8 == 0) & (idx.minute == 0)
    frame = pd.DataFrame(
        {
            "perp_close": perp,
            "spot_close": spot,
            "close": perp,
            "high": perp * 1.001,
            "low": perp * 0.999,
            "basis_pct": basis_pct,
            "funding_event_binance": np.where(is_8h, funding_rate, 0.0),
        },
        index=idx,
    )
    frame.index.name = "datetime"
    return frame


def _make_universe(idx: pd.DatetimeIndex, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    return {sym: _make_frame(idx, BASE_PRICES[sym]) for sym in (symbols or CORE_4)}


def _carry_config(capital: float = 3_000.0) -> CrossExchangeCarryConfig:
    return CrossExchangeCarryConfig(initial_capital=capital)


# ---------------------------------------------------------------------------
# 1. Gate tren engine carry
# ---------------------------------------------------------------------------


def test_carry_trade_start_gate_keeps_equity_flat_before_boundary() -> None:
    idx = _hourly_index(24 * 30)
    data = _make_universe(idx)
    cfg = _carry_config()
    cut = 120

    gated = run_binance_tier1_plus_carry(data, cfg, variant="tier1b", trade_start_idx=cut)

    equity = gated["equity_curve"]
    assert len(equity) == len(idx)
    assert float(equity.iloc[:cut].max()) == pytest.approx(cfg.initial_capital, abs=1e-9)
    assert float(equity.iloc[:cut].min()) == pytest.approx(cfg.initial_capital, abs=1e-9)

    trades = gated["trades_df"]
    if not trades.empty:
        first_trade_time = pd.Timestamp(trades.iloc[0]["time"])
        assert first_trade_time >= idx[cut]


def test_carry_trade_start_none_matches_default_behaviour() -> None:
    idx = _hourly_index(24 * 20)
    data = _make_universe(idx)
    cfg = _carry_config()

    baseline = run_binance_tier1_plus_carry(data, cfg, variant="tier1b")
    explicit_none = run_binance_tier1_plus_carry(
        data, cfg, variant="tier1b", trade_start_idx=None
    )

    assert float(baseline["equity_curve"].iloc[-1]) == pytest.approx(
        float(explicit_none["equity_curve"].iloc[-1]), rel=1e-12, abs=1e-9
    )
    assert int(baseline["metrics"]["Num Entries"]) == int(
        explicit_none["metrics"]["Num Entries"]
    )


# ---------------------------------------------------------------------------
# 2. Gate tren tang barbell
# ---------------------------------------------------------------------------


def test_barbell_trade_start_gate_keeps_cash_and_zero_overlay_weights() -> None:
    idx = _hourly_index(24 * 40)
    data = _make_universe(idx)
    cfg = BarbellConfig(initial_capital=3_000.0, carry_variant="tier1b")
    cut = 200

    res = run_barbell_portfolio(data, cfg, trade_start_idx=cut)

    equity = res["equity_curve"]
    assert len(equity) == len(idx)
    assert float(equity.iloc[:cut].abs().sub(cfg.initial_capital).abs().max()) < 1e-9

    weights = res["overlay_weights"]
    if not weights.empty:
        before = weights.iloc[:cut].abs().to_numpy()
        assert float(before.max()) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. run_indicator_warm_cash_start
# ---------------------------------------------------------------------------


def test_warm_cash_runner_returns_capital_and_respects_warmup() -> None:
    idx = _hourly_index(24 * 60)
    data = _make_universe(idx)
    split = idx[-24 * 20].isoformat()

    res = run_indicator_warm_cash_start(
        data=data,
        capital=3_000.0,
        split_date=split,
        holdout_end=idx[-1].isoformat(),
        reinvest_profits=False,
        warmup_hours=48,
    )

    assert res["method"] == "indicator_warm_cash_start"
    assert res["warmup_hours"] == 48
    assert res["trade_start_idx"] == 48
    assert res["start_equity_usd"] == pytest.approx(3_000.0)
    assert res["end_equity_usd"] == pytest.approx(3_000.0 + res["net_pnl_usd"], abs=1e-9)
    assert res["return_on_active_capital_pct"] == pytest.approx(
        res["net_pnl_usd"] / 3_000.0 * 100.0, abs=1e-9
    )


def test_warm_cash_default_warmup_matches_module_constant() -> None:
    assert SPRINT14_WARMUP_HOURS == 504


# ---------------------------------------------------------------------------
# 4 & 5. Bridge 4 phuong phap
# ---------------------------------------------------------------------------


def _payload(
    net_a: float,
    net_resim: float,
    net_fixed: float,
    entry_fee: float = 2.02,
) -> dict:
    """Payload gia lap dung schema thuc te cua sprint11_payload.json."""
    row = {
        "tier": "tier1b",
        "gating_mode": "smooth_conviction",
        "holdout_equity_start": 4_091.29,
        "holdout_net_pnl": net_a,
        "holdout_net_pnl_resim": net_resim,
        "holdout_resim_entry_fee_usd": entry_fee,
        "holdout_days": 181.0,
    }
    return {"rows": [row], "split_date": "2026-04-21", "holdout_end": "2026-10-19"}


def _fixed_payload(net_fixed: float, net_resim: float, entry_fee: float = 2.02) -> dict:
    """Payload B: chay lien tuc voi reinvest_profits=False."""
    fixed_row = {
        "tier": "tier1b",
        "gating_mode": "smooth_conviction",
        "holdout_equity_start": 3_946.72,
        "holdout_net_pnl": net_fixed,
        "holdout_net_pnl_resim": net_resim,
        "holdout_resim_entry_fee_usd": entry_fee,
        "holdout_days": 181.0,
    }
    return {"rows": [fixed_row], "split_date": "2026-04-21", "holdout_end": "2026-10-19"}


def _warm(net_pnl: float, entry_fee: float = 2.02) -> dict:
    return {
        "net_pnl_usd": net_pnl,
        "start_equity_usd": 3_000.0,
        "end_equity_usd": 3_000.0 + net_pnl,
        "return_on_active_capital_pct": net_pnl / 3_000.0 * 100.0,
        "entry_fee_usd": entry_fee,
        "warmup_hours": 504,
        "trade_start_idx": 504,
    }


def test_bridge_identity_holds_for_four_methods() -> None:
    b_net, c_net, d_net = 118.50, 97.07, 81.73
    bridge = compute_holdout_accounting_bridge(
        _payload(159.15, c_net, b_net), _fixed_payload(b_net, c_net), warm_cash=_warm(d_net)
    )

    net_b = bridge["continuous_fixed_base"]["net_pnl_usd"]
    net_c = bridge["standalone_cold_start"]["net_pnl_usd"]
    net_d = bridge["indicator_warm_cash_start"]["net_pnl_usd"]

    assert net_b == pytest.approx(b_net, abs=1e-9)
    assert net_c == pytest.approx(c_net, abs=1e-9)
    assert net_d == pytest.approx(d_net, abs=1e-9)

    identity = (net_b - net_c) - ((net_b - net_d) + (net_d - net_c))
    assert abs(identity) < 1e-6


def test_bridge_three_way_decomposition_sums_to_total_drag() -> None:
    b_net, c_net, d_net = 118.50, 97.07, 81.73
    entry_fee = 2.02
    bridge = compute_holdout_accounting_bridge(
        _payload(159.15, c_net, b_net, entry_fee),
        _fixed_payload(b_net, c_net, entry_fee),
        warm_cash=_warm(d_net, entry_fee),
    )

    dec = bridge["warm_cash_decomposition"]
    total_drag = bridge["bridge_decomposition"]["cold_start_total_drag_usd"]

    assert dec["cold_start_initial_entry_fee_usd"] == pytest.approx(entry_fee, abs=1e-9)
    assert dec["indicator_blindness_drag_usd"] == pytest.approx(d_net - c_net, abs=1e-9)
    assert dec["inventory_carryover_diff_usd"] == pytest.approx(
        (b_net - d_net) - entry_fee, abs=1e-9
    )
    assert dec["explained_total_usd"] == pytest.approx(total_drag, abs=1e-6)
    assert abs(dec["unexplained_total_usd"]) < 1e-6


def test_bridge_without_warm_cash_stays_backward_compatible() -> None:
    bridge = compute_holdout_accounting_bridge(
        _payload(159.15, 97.07, 118.50), _fixed_payload(118.50, 97.07)
    )

    assert bridge["indicator_warm_cash_start"]["available"] is False
    assert bridge["indicator_warm_cash_start"]["net_pnl_usd"] == pytest.approx(0.0)

    dec = bridge["warm_cash_decomposition"]
    assert dec["indicator_blindness_drag_usd"] == pytest.approx(0.0)
    total_drag = bridge["bridge_decomposition"]["cold_start_total_drag_usd"]
    assert dec["explained_total_usd"] == pytest.approx(total_drag, abs=1e-6)
    assert abs(dec["unexplained_total_usd"]) < 1e-6
