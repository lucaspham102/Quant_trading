"""Sprint 26 tests: triple-confirmed trend sleeve, 2-tier drawdown circuit breaker,
profile wiring and the live-demo / tournament integration.

Fully offline and deterministic: no network, no sleeps, no filesystem writes.
Synthetic price panels are pure deterministic exponentials plus a sine wobble.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.directional_trend_sleeve import (
    DrawdownCircuitBreakerConfig,
    TrendSleeveConfig,
    TrendSleeveState,
    compute_triple_confirmed_trend_weights,
    evaluate_drawdown_circuit_breaker,
    trend_weights_to_target_qty,
)
from src.strategy_profiles import PROFILE_REGISTRY

try:
    from scripts import run_parallel_live_demo as demo
except Exception:  # pragma: no cover - fallback if scripts is not importable
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _spec = importlib.util.spec_from_file_location(
        "run_parallel_live_demo", os.path.join(_ROOT, "scripts", "run_parallel_live_demo.py")
    )
    demo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(demo)

try:
    from scripts import run_1k_profile_tournament_backtest as tournament
except Exception:  # pragma: no cover - fallback if scripts is not importable
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _spec = importlib.util.spec_from_file_location(
        "run_1k_profile_tournament_backtest",
        os.path.join(_ROOT, "scripts", "run_1k_profile_tournament_backtest.py"),
    )
    tournament = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tournament)

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
SOL = "SOL/USDT:USDT"
MONEY = dict(rel=1e-9, abs=1e-9)

CB_CFG = DrawdownCircuitBreakerConfig()

SIM_REQUIRED_KEYS = (
    "trend_pnl_usd",
    "circuit_breaker_triggers",
    "circuit_breaker_bars_soft",
    "circuit_breaker_bars_hard",
    "final_circuit_breaker_state",
    "avg_trend_gross_usd",
)


# --------------------------------------------------------------------------- helpers --
class FakeExecState:
    """Minimal stand-in for a CarryLiveExecutor ``state`` object."""

    def __init__(self, carry_positions=None, overlay_positions=None) -> None:
        self.carry_positions = dict(carry_positions or {})
        self.overlay_positions = dict(overlay_positions or {})


def _trend_frame(n: int = 2000, btc_drift: float = 0.002) -> pd.DataFrame:
    """Clean hourly uptrend frame: deterministic exponential drift + sine wobble."""
    i = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            BTC: 200.0 * np.exp(btc_drift * i) * (1.0 + 0.02 * np.sin(i / 17.0)),
            ETH: 100.0 * np.exp(0.002 * i) * (1.0 + 0.02 * np.sin(i / 17.0 + 1.0)),
        }
    )


def _hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01 00:00", periods=n, freq="h", tz="UTC")


def _aligned_universe(n: int = 2000) -> dict[str, pd.DataFrame]:
    """dict[sym -> DataFrame] with a ``perp_close`` column, as apply_trend_sleeve wants."""
    index = _hourly_index(n)
    i = np.arange(n, dtype=float)
    out: dict[str, pd.DataFrame] = {}
    for k, (sym, base) in enumerate(((BTC, 200.0), (ETH, 100.0), (SOL, 50.0))):
        px = base * np.exp(0.002 * i) * (1.0 + 0.02 * np.sin(i / 17.0 + k))
        out[sym] = pd.DataFrame(
            {"perp_close": px, "close": px, "spot_close": px * 1.0005}, index=index
        )
    return out


def _three_sleeve_positions() -> dict[str, dict]:
    """Carry + overlay + trend on the SAME symbol, each on its own isolated ledger key."""
    return {
        BTC: {
            "symbol": BTC,
            "qty": 1.0,
            "entry_price": 100.0,
            "kind": "carry",
            "signed_usd": 100.0,
            "spot_qty": 1.0,
            "perp_qty": -1.0,
            "spot_entry": 100.0,
            "perp_entry": 100.0,
        },
        f"{BTC}|overlay": {
            "symbol": BTC,
            "qty": 0.5,
            "entry_price": 100.0,
            "kind": "overlay",
            "signed_usd": 50.0,
        },
        f"{BTC}|trend": {
            "symbol": BTC,
            "qty": 1.0,
            "entry_price": 100.0,
            "kind": "trend",
            "signed_usd": 100.0,
        },
    }


def _three_sleeve_basis_snapshot() -> dict:
    """Spot 110 / Perp 112: carry basis -2 USD, overlay +5 USD, trend +10 USD."""
    return {
        "symbols": {BTC: {"mid": 110.0}},
        "spot_symbols": {BTC: {"bid": 110.0, "ask": 110.0, "mid": 110.0}},
        "perp_symbols": {BTC: {"mid": 110.0, "mark": 112.0}},
    }


def _synthetic_trend_market(n: int = 2000) -> dict:
    """Market dict shaped exactly like tournament.build_market_matrices (2,000 hourly bars)."""
    index = _hourly_index(n)
    i = np.arange(n, dtype=float)
    btc = 200.0 * np.exp(0.002 * i) * (1.0 + 0.02 * np.sin(i / 17.0))
    eth = 100.0 * np.exp(0.002 * i) * (1.0 + 0.02 * np.sin(i / 23.0 + 0.5))
    perp = pd.DataFrame({BTC: btc, ETH: eth}, index=index)
    spot = perp * 1.0004
    funding = pd.DataFrame(
        {BTC: np.full(n, 0.0001), ETH: np.full(n, 0.0002)}, index=index
    )
    score = pd.DataFrame({BTC: np.full(n, 1.0), ETH: np.full(n, 1.0)}, index=index)
    snr = pd.DataFrame(np.nan, index=index, columns=[BTC, ETH])
    return {
        "index": index,
        "perp_close": perp,
        "spot_close": spot,
        "funding_event": funding,
        "funding_bps": funding * 10000.0,
        "carry_score": score,
        "snr": snr,
        "snr_dispersion": pd.Series(np.nan, index=index),
        "spot_cost_bps": 7.5,
        "perp_cost_bps": 5.0,
        "carry_cost_bps": 12.5,
    }


# ---------------------------------------------------------------- circuit breaker --
def test_cb_normal_when_drawdown_below_soft_threshold():
    flat = evaluate_drawdown_circuit_breaker(1000.0, 1000.0, None, CB_CFG)
    assert flat["state"] == "NORMAL"
    assert flat["risk_scale"] == pytest.approx(1.0)
    assert flat["current_dd_pct"] == pytest.approx(0.0)

    mild = evaluate_drawdown_circuit_breaker(941.0, 1000.0, None, CB_CFG)
    assert mild["state"] == "NORMAL"
    assert mild["risk_scale"] == pytest.approx(1.0)
    assert mild["current_dd_pct"] == pytest.approx(0.059)


def test_cb_soft_throttle_at_exactly_six_percent():
    res = evaluate_drawdown_circuit_breaker(940.0, 1000.0, None, CB_CFG)
    assert res["state"] == "SOFT_THROTTLE"
    assert res["risk_scale"] == pytest.approx(0.50)
    assert res["current_dd_pct"] == pytest.approx(0.06)


def test_cb_soft_throttle_at_9_9_percent():
    res = evaluate_drawdown_circuit_breaker(901.0, 1000.0, None, CB_CFG)
    assert res["state"] == "SOFT_THROTTLE"
    assert res["risk_scale"] == pytest.approx(0.50)
    assert res["current_dd_pct"] == pytest.approx(0.099)


def test_cb_hard_stop_locked_at_exactly_ten_percent():
    res = evaluate_drawdown_circuit_breaker(900.0, 1000.0, None, CB_CFG)
    assert res["state"] == "HARD_STOP_LOCKED"
    assert res["risk_scale"] == pytest.approx(0.0)
    assert res["current_dd_pct"] == pytest.approx(0.10)


def test_cb_cooldown_locks_even_after_full_recovery():
    # Equity is fully recovered, but the hard-stop timer is still running (default 168 bars).
    locked = evaluate_drawdown_circuit_breaker(1000.0, 1000.0, 10, CB_CFG)
    assert locked["state"] == "HARD_STOP_LOCKED"
    assert locked["risk_scale"] == pytest.approx(0.0)

    still_locked = evaluate_drawdown_circuit_breaker(1000.0, 1000.0, 167, CB_CFG)
    assert still_locked["state"] == "HARD_STOP_LOCKED"
    assert still_locked["risk_scale"] == pytest.approx(0.0)

    released = evaluate_drawdown_circuit_breaker(1000.0, 1000.0, 168, CB_CFG)
    assert released["state"] == "NORMAL"
    assert released["risk_scale"] == pytest.approx(1.0)


def test_cb_degenerate_inputs_never_raise_or_emit_nan():
    cases = [
        (0.0, 0.0, None),
        (float("nan"), 1000.0, None),
        (1000.0, float("nan"), None),
        (float("inf"), 1000.0, None),
        (1000.0, float("inf"), None),
        (-50.0, 0.0, None),
    ]
    for cur, peak, bars in cases:
        res = evaluate_drawdown_circuit_breaker(cur, peak, bars, CB_CFG)
        assert math.isfinite(res["risk_scale"])
        assert res["risk_scale"] in (0.0, 0.5, 1.0)
        assert res["state"] in ("NORMAL", "SOFT_THROTTLE", "HARD_STOP_LOCKED")


def test_cb_config_rejects_soft_ge_hard():
    with pytest.raises(ValueError):
        DrawdownCircuitBreakerConfig(soft_dd_threshold_pct=0.10, hard_dd_threshold_pct=0.10)
    with pytest.raises(ValueError):
        DrawdownCircuitBreakerConfig(soft_dd_threshold_pct=0.12, hard_dd_threshold_pct=0.10)


# --------------------------------------------------------- trend signal (causality) --
def test_trend_signal_is_strictly_causal_at_bar_n_minus_2():
    frame_a = _trend_frame(2000)
    frame_b = frame_a.copy()
    # A monstrous last bar must not be visible at bar n-2.
    frame_b.iloc[-1, :] = frame_b.iloc[-1, :] * 5.0

    cfg = TrendSleeveConfig()
    n = len(frame_a)

    weights_a = compute_triple_confirmed_trend_weights(
        frame_a, bar_index=n - 2, config=cfg, state=TrendSleeveState()
    )
    weights_b = compute_triple_confirmed_trend_weights(
        frame_b, bar_index=n - 2, config=cfg, state=TrendSleeveState()
    )

    assert weights_a, "clean uptrend must produce a target at bar n-2"
    assert set(weights_a) == set(weights_b)
    for sym in weights_a:
        assert weights_b[sym] == pytest.approx(weights_a[sym], rel=1e-12, abs=1e-12)


def test_trend_uptrend_produces_capped_long_target():
    frame = _trend_frame(2000)
    cfg = TrendSleeveConfig()
    weights = compute_triple_confirmed_trend_weights(
        frame, bar_index=len(frame) - 1, config=cfg, state=TrendSleeveState()
    )

    assert set(weights) == {BTC, ETH}
    assert sum(weights.values()) <= cfg.max_gross_pct + 1e-9
    assert sum(weights.values()) > 0.0
    assert all(w > 0.0 for w in weights.values())


def test_trend_btc_macro_gate_blocks_altcoin_long():
    # BTC clearly down, ETH clearly up on the same index.
    frame = _trend_frame(2000, btc_drift=-0.002)
    bar = len(frame) - 1

    gated_cfg = TrendSleeveConfig(use_btc_gate=True, btc_symbol=BTC)
    gated = compute_triple_confirmed_trend_weights(
        frame, bar_index=bar, config=gated_cfg, state=TrendSleeveState()
    )
    assert ETH not in gated, "altcoin long must be blocked while BTC is below its med EMA"

    open_cfg = dataclasses.replace(gated_cfg, use_btc_gate=False)
    ungated = compute_triple_confirmed_trend_weights(
        frame, bar_index=bar, config=open_cfg, state=TrendSleeveState()
    )
    assert ETH in ungated, "same data without the gate must allow the ETH long"
    assert ungated[ETH] > 0.0


def test_trend_chandelier_stop_exits_after_crash():
    frame = _trend_frame(2000)[[BTC]]
    cfg = TrendSleeveConfig()
    state = TrendSleeveState()

    opened = compute_triple_confirmed_trend_weights(
        frame, bar_index=1900, config=cfg, state=state
    )
    assert BTC in opened
    assert opened[BTC] > 0.0
    assert state.active[BTC] is True

    # Brutal crash: the last 3 bars lose 45% each (x0.55), far below the stored peak.
    crashed = frame.copy()
    crashed.iloc[-3:, 0] = crashed.iloc[-3:, 0] * 0.55
    after = compute_triple_confirmed_trend_weights(
        crashed, bar_index=len(crashed) - 1, config=cfg, state=state
    )

    assert BTC not in after
    assert state.active[BTC] is False
    assert BTC not in state.peak_close


def test_trend_circuit_breaker_scales_and_flattens_weights():
    frame = _trend_frame(2000)
    bar = len(frame) - 1
    cfg = TrendSleeveConfig()

    base = compute_triple_confirmed_trend_weights(
        frame, bar_index=bar, config=cfg, state=TrendSleeveState()
    )
    assert base
    assert sum(base.values()) > 0.0

    half = compute_triple_confirmed_trend_weights(
        frame,
        bar_index=bar,
        config=cfg,
        state=TrendSleeveState(),
        circuit_breaker={"risk_scale": 0.5},
    )
    assert set(half) == set(base)
    for sym in base:
        assert half[sym] == pytest.approx(0.5 * base[sym], rel=1e-9)

    flat = compute_triple_confirmed_trend_weights(
        frame,
        bar_index=bar,
        config=cfg,
        state=TrendSleeveState(),
        circuit_breaker={"risk_scale": 0.0},
    )
    # risk_scale 0.0 must leave the sleeve flat. The module drops every leg whose scaled
    # weight is not > 0, so callers receive {} (or, defensively, all-zero weights); both
    # mean "no long exposure". Assert per-symbol so a non-empty dict cannot slip through.
    assert sum(abs(w) for w in flat.values()) == pytest.approx(0.0, abs=1e-15)
    for sym in base:
        assert flat.get(sym, 0.0) == 0.0


def test_trend_history_guard_returns_empty_below_min_history():
    cfg = TrendSleeveConfig()
    short = _trend_frame(1000)
    assert len(short) < cfg.min_history_hours
    assert (
        compute_triple_confirmed_trend_weights(
            short, bar_index=len(short) - 1, config=cfg, state=TrendSleeveState()
        )
        == {}
    )

    # Exactly one bar short of the guard is still rejected.
    almost = _trend_frame(cfg.min_history_hours - 1)
    assert (
        compute_triple_confirmed_trend_weights(
            almost, bar_index=len(almost) - 1, config=cfg, state=TrendSleeveState()
        )
        == {}
    )


def test_trend_weights_to_target_qty_quantizes_and_drops_dust():
    weights = {SOL: 0.1234, ETH: 0.0001, "UNKNOWN/USDT:USDT": 0.10}
    prices = {SOL: 100.0, ETH: 100.0}
    qty = trend_weights_to_target_qty(
        weights, prices, 1000.0, {SOL: 0.01, ETH: 0.001}, min_notional_usd=10.0
    )

    # 0.1234 * 1000 / 100 = 1.234 base units, floored to the 0.01 SOL lot step.
    assert qty[SOL] == pytest.approx(1.23)
    # 0.1 USD notional is below the 10 USD floor -> dropped.
    assert qty[ETH] == pytest.approx(0.0)
    # Missing price -> 0.0, never a guess.
    assert qty["UNKNOWN/USDT:USDT"] == pytest.approx(0.0)

    # Non-positive price also collapses to zero.
    bad = trend_weights_to_target_qty({SOL: 0.1}, {SOL: 0.0}, 1000.0, {SOL: 0.01})
    assert bad[SOL] == pytest.approx(0.0)

    # Empty weights stay empty.
    assert trend_weights_to_target_qty({}, {}, 1000.0) == {}


# ----------------------------------------------------------- sleeve key isolation --
def test_snapshot_key_namespaces_carry_overlay_trend():
    carry_key = demo._snapshot_key(BTC, "carry")
    overlay_key = demo._snapshot_key(BTC, "overlay")
    trend_key = demo._snapshot_key(BTC, "trend")

    assert carry_key == BTC
    assert overlay_key == f"{BTC}|overlay"
    assert trend_key == f"{BTC}|trend"
    assert len({carry_key, overlay_key, trend_key}) == 3


def test_default_ledger_carries_sprint26_trend_fields():
    profile = PROFILE_REGISTRY["baseline_1k"]
    ledger = demo._sanitize_ledger(demo._default_ledger(profile))

    assert ledger["trend_pnl_usd"] == pytest.approx(0.0)
    assert ledger["trend_gross_usd"] == pytest.approx(0.0)
    assert ledger["circuit_breaker_state"] == "DISABLED"
    assert ledger["circuit_breaker_risk_scale"] == pytest.approx(1.0)
    assert ledger["circuit_breaker_triggers"] == 0
    assert ledger["trend_hard_lock_bars"] is None
    assert ledger["last_trend_rebalance_ts"] is None
    assert isinstance(ledger["trend_sleeve_state"], dict)
    assert ledger["trend_sleeve_state"]["peak_close"] == {}
    assert ledger["trend_sleeve_state"]["active"] == {}


def test_three_sleeves_coexist_on_one_symbol():
    profile = PROFILE_REGISTRY["baseline_1k"]
    ledger = demo._sanitize_ledger(demo._default_ledger(profile))
    ledger["positions_snapshot"] = _three_sleeve_positions()
    ledger["last_prices"] = {BTC: 110.0}
    snapshot = _three_sleeve_basis_snapshot()

    unreal_carry, unreal_overlay, unreal_trend = demo._unrealized_from_snapshot(
        ledger["positions_snapshot"], snapshot["symbols"], ledger["last_prices"], snapshot
    )
    assert unreal_carry == pytest.approx(-2.0, **MONEY)
    assert unreal_overlay == pytest.approx(5.0, **MONEY)
    trend_leg = ledger["positions_snapshot"][f"{BTC}|trend"]
    assert unreal_trend == pytest.approx(
        trend_leg["signed_usd"] * (110.0 / trend_leg["entry_price"] - 1.0), **MONEY
    )
    assert unreal_trend == pytest.approx(10.0, **MONEY)

    demo._update_exposure_metrics(ledger, profile)
    assert ledger["trend_gross_usd"] == pytest.approx(100.0, **MONEY)
    assert ledger["carry_slots"] == 1
    fut_wallet = profile.futures_wallet_usd
    expected_lev = (100.0 + 50.0 + 100.0) / fut_wallet
    assert ledger["futures_leverage"] == pytest.approx(expected_lev, **MONEY)
    assert ledger["futures_leverage"] > (100.0 + 50.0) / fut_wallet

    demo.accrue_cycle_accounting(
        ledger, profile, snapshot, datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    )
    assert set(ledger["positions_snapshot"]) == {BTC, f"{BTC}|overlay", f"{BTC}|trend"}
    assert ledger["positions_snapshot"][f"{BTC}|trend"]["kind"] == "trend"
    # Carry basis -2 + overlay +5 + trend +10 on top of the 1,000 USD capital.
    assert ledger["pure_trading_equity_usd"] == pytest.approx(1013.0, **MONEY)


def test_refresh_position_ledger_preserves_unknown_trend_leg():
    profile = PROFILE_REGISTRY["baseline_1k"]
    ledger = demo._sanitize_ledger(demo._default_ledger(profile))
    trend_key = f"{BTC}|trend"
    ledger["positions_snapshot"] = {
        trend_key: {
            "symbol": BTC,
            "qty": 1.0,
            "entry_price": 100.0,
            "kind": "trend",
            "signed_usd": 100.0,
        }
    }
    ledger["last_prices"] = {BTC: 100.0}

    # The executor knows nothing about trend legs (they belong to apply_trend_sleeve).
    state = FakeExecState(carry_positions={}, overlay_positions={})
    fees, realized_carry, realized_overlay = demo.refresh_position_ledger(
        ledger, state, _three_sleeve_basis_snapshot(), charge_fees=False
    )

    assert fees == pytest.approx(0.0)
    assert realized_carry == pytest.approx(0.0)
    assert realized_overlay == pytest.approx(0.0)
    assert trend_key in ledger["positions_snapshot"]
    assert ledger["positions_snapshot"][trend_key]["qty"] == pytest.approx(1.0)
    assert ledger["positions_snapshot"][trend_key]["kind"] == "trend"


# ------------------------------------------------------------- apply_trend_sleeve --
def test_apply_trend_sleeve_disabled_profile_opens_nothing():
    profile = PROFILE_REGISTRY["baseline_1k"]
    ledger = demo._sanitize_ledger(demo._default_ledger(profile))

    res = demo.apply_trend_sleeve(ledger, profile, _aligned_universe(), None)

    assert res["state"] == "DISABLED"
    assert res["risk_scale"] == pytest.approx(1.0)
    assert res["n_orders"] == 0
    assert ledger["circuit_breaker_state"] == "DISABLED"
    assert ledger["trend_gross_usd"] == pytest.approx(0.0)
    assert ledger["positions_snapshot"] == {}


def test_apply_trend_sleeve_bullish_opens_trend_legs_and_pays_fees():
    profile = PROFILE_REGISTRY["hybrid_trend_conservative_1k"]
    ledger = demo._sanitize_ledger(demo._default_ledger(profile))

    res = demo.apply_trend_sleeve(ledger, profile, _aligned_universe(), None)

    trend_keys = [k for k in ledger["positions_snapshot"] if k.endswith("|trend")]
    assert trend_keys, "bullish aligned universe must open at least one trend leg"
    assert all(ledger["positions_snapshot"][k]["kind"] == "trend" for k in trend_keys)
    assert ledger["fees_paid_usd"] > 0.0
    assert ledger["circuit_breaker_state"] in {"NORMAL", "SOFT_THROTTLE"}
    assert res["risk_scale"] > 0.0
    assert res["n_orders"] >= 1
    assert res["turnover_notional"] > 0.0


def test_apply_trend_sleeve_hard_drawdown_locks_and_opens_nothing():
    profile = PROFILE_REGISTRY["hybrid_trend_conservative_1k"]
    ledger = demo._sanitize_ledger(demo._default_ledger(profile))
    ledger["peak_equity_usd"] = 1000.0
    ledger["pure_trading_equity_usd"] = 880.0  # -12%, beyond the 10% hard stop

    res = demo.apply_trend_sleeve(ledger, profile, _aligned_universe(), None)

    assert res["state"] == "HARD_STOP_LOCKED"
    assert res["risk_scale"] == pytest.approx(0.0)
    assert ledger["circuit_breaker_state"] == "HARD_STOP_LOCKED"
    assert ledger["circuit_breaker_triggers"] == 1
    assert ledger["trend_hard_lock_bars"] == 0
    assert res["n_orders"] == 0
    assert res["positions"] == []
    assert [k for k in ledger["positions_snapshot"] if k.endswith("|trend")] == []


# ----------------------------------------------------- profiles + tournament wiring --
def test_trend_sleeve_does_not_churn_between_rebalances():
    """Between two 24h rebalances the sleeve must be frozen: zero orders, zero fees."""
    profile = PROFILE_REGISTRY["hybrid_trend_conservative_1k"]
    ledger = demo._sanitize_ledger(demo._default_ledger(profile))
    t0 = datetime(2026, 3, 1, tzinfo=timezone.utc)

    first = demo.apply_trend_sleeve(ledger, profile, _aligned_universe(), None, now_utc=t0)
    assert first["n_orders"] >= 1
    keys_before = {k: dict(v) for k, v in ledger["positions_snapshot"].items() if k.endswith("|trend")}
    fees_before = ledger["fees_paid_usd"]

    second = demo.apply_trend_sleeve(
        ledger, profile, _aligned_universe(), None, now_utc=t0 + timedelta(hours=1)
    )

    assert second.get("held") is True
    assert second["n_orders"] == 0
    assert second["fees_usd"] == pytest.approx(0.0)
    assert ledger["fees_paid_usd"] == pytest.approx(fees_before)
    keys_after = {k: dict(v) for k, v in ledger["positions_snapshot"].items() if k.endswith("|trend")}
    assert keys_after == keys_before


def test_hard_stop_flattens_immediately_without_waiting_for_rebalance():
    """A -10% hard stop is an emergency: the book must be flat on the very next tick,
    even though the 24h rebalance clock has not fired yet."""
    profile = PROFILE_REGISTRY["hybrid_trend_conservative_1k"]
    ledger = demo._sanitize_ledger(demo._default_ledger(profile))
    t0 = datetime(2026, 3, 1, tzinfo=timezone.utc)

    demo.apply_trend_sleeve(ledger, profile, _aligned_universe(), None, now_utc=t0)
    assert [k for k in ledger["positions_snapshot"] if k.endswith("|trend")]

    ledger["peak_equity_usd"] = 1000.0
    ledger["pure_trading_equity_usd"] = 880.0  # -12% => HARD_STOP_LOCKED

    res = demo.apply_trend_sleeve(
        ledger, profile, _aligned_universe(), None, now_utc=t0 + timedelta(hours=1)
    )

    assert res["state"] == "HARD_STOP_LOCKED"
    assert res["risk_scale"] == pytest.approx(0.0)
    assert [k for k in ledger["positions_snapshot"] if k.endswith("|trend")] == []
    assert res["positions"] == []
    assert res["n_orders"] >= 1
    assert res["fees_usd"] > 0.0


def test_legacy_profiles_keep_trend_sleeve_disabled():
    for name in ("default", "canary_400", "baseline_1k", "alpha_boost_1k", "turbo_neutral_1k"):
        profile = PROFILE_REGISTRY[name]
        assert profile.enable_trend_sleeve is False, name
        assert profile.trend_target_vol_ann == pytest.approx(0.0), name


def test_hybrid_profiles_enable_trend_and_split_wallets():
    for name in ("hybrid_trend_conservative_1k", "hybrid_trend_balanced_1k"):
        profile = PROFILE_REGISTRY[name]
        assert profile.enable_trend_sleeve is True, name
        assert profile.soft_dd_threshold_pct == pytest.approx(0.06), name
        assert profile.hard_dd_threshold_pct == pytest.approx(0.10), name
        assert profile.trend_target_vol_ann > 0.0, name
        assert profile.trend_max_gross_pct > 0.0, name
        assert profile.spot_wallet_usd + profile.futures_wallet_usd == pytest.approx(
            profile.capital_usd
        ), name


def test_tournament_trend_sleeve_wiring_with_synthetic_market():
    market = _synthetic_trend_market(2000)
    n = len(market["index"])

    hybrid = tournament.simulate_profile(
        PROFILE_REGISTRY["hybrid_trend_conservative_1k"],
        market,
        start_idx=0,
        end_idx=n,
        trade_start_idx=1200,
    )
    baseline = tournament.simulate_profile(
        PROFILE_REGISTRY["baseline_1k"],
        market,
        start_idx=0,
        end_idx=n,
        trade_start_idx=1200,
    )

    for res in (hybrid, baseline):
        for key in SIM_REQUIRED_KEYS:
            assert key in res, key

    # Baseline has no sleeve at all.
    assert baseline["trend_pnl_usd"] == pytest.approx(0.0, abs=1e-15)
    assert baseline["final_circuit_breaker_state"] == "DISABLED"
    assert baseline["trend_enabled"] is False
    assert baseline["avg_trend_gross_usd"] == pytest.approx(0.0, abs=1e-15)

    # Hybrid actually traded the sleeve and ran a live circuit breaker.
    assert hybrid["final_circuit_breaker_state"] != "DISABLED"
    assert abs(hybrid["trend_pnl_usd"]) > 0.0
    assert hybrid["avg_trend_gross_usd"] > 0.0
    assert hybrid["trend_enabled"] is True
    assert hybrid["circuit_breaker_bars_soft"] >= 0
    assert hybrid["circuit_breaker_bars_hard"] >= 0
