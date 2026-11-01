"""Tests for strategy profiles registry and the parallel live demo runner.

All network access is mocked; nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.strategy_profiles import (
    PROFILE_REGISTRY,
    StrategyProfileConfig,
    compute_funding_tilted_weights,
    compute_step_passive_yield_usd,
    enforce_split_wallet_futures_cap,
    get_strategy_profile,
)

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

# Sprint 26 extends the registry with three evidence-driven 1k profiles; the Sprint 25 set
# below stays frozen so a rename/removal of the original five still fails this test.
SPRINT25_PROFILES = {"default", "canary_400", "baseline_1k", "alpha_boost_1k", "turbo_neutral_1k"}
SPRINT26_PROFILES = {"carry_max_eta_1k", "hybrid_trend_conservative_1k", "hybrid_trend_balanced_1k"}
SPRINT27_PROFILES = {"bi_trend_carry_1k", "carry_stat_arb_1k", "regime_omni_1k", "flexible_regime_1k"}
EXPECTED_PROFILES = SPRINT25_PROFILES | SPRINT26_PROFILES | SPRINT27_PROFILES
MONEY = dict(rel=1e-6, abs=1e-6)
BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
SOL = "SOL/USDT:USDT"


class FakeExecState:
    """Minimal stand-in for a CarryLiveExecutor ``state`` object."""

    def __init__(self, carry_positions=None, overlay_positions=None) -> None:
        self.carry_positions = dict(carry_positions or {})
        self.overlay_positions = dict(overlay_positions or {})


@pytest.fixture()
def prof_1k() -> StrategyProfileConfig:
    return get_strategy_profile("baseline_1k")


def _fresh_ledger(path, profile):
    return demo.load_ledger(str(path), profile)


# ------------------------------------------------------------------ profiles --
def test_registry_has_exactly_the_five_profiles():
    assert set(PROFILE_REGISTRY) == EXPECTED_PROFILES
    assert len(PROFILE_REGISTRY) == 12
    # The original Sprint 25 five must never disappear.
    assert SPRINT25_PROFILES <= set(PROFILE_REGISTRY)


@pytest.mark.parametrize("name", sorted(EXPECTED_PROFILES))
def test_registry_capital_matches_wallets(name):
    prof = PROFILE_REGISTRY[name]
    assert prof.name == name
    assert prof.capital_usd == pytest.approx(
        prof.spot_wallet_usd + prof.futures_wallet_usd, **MONEY
    )


def test_profile_config_is_frozen():
    prof = PROFILE_REGISTRY["default"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        prof.capital_usd = 1.0  # type: ignore[misc]


_VALID = {"capital_usd": 100.0, "spot_wallet_usd": 80.0, "futures_wallet_usd": 20.0}
BAD_INPUTS = [
    {"capital_usd": 100.0, "spot_wallet_usd": 10.0, "futures_wallet_usd": 10.0},
    {**_VALID, "max_futures_leverage": 0.0},
    {**_VALID, "carry_top_k": 0},
]


@pytest.mark.parametrize("kwargs", BAD_INPUTS)
def test_validation_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        StrategyProfileConfig(name="bad", **kwargs)


def test_get_strategy_profile_default_and_unknown():
    assert get_strategy_profile("default") is PROFILE_REGISTRY["default"]
    with pytest.raises(ValueError):
        get_strategy_profile("nope_not_here")


def test_funding_tilted_weights_equal_funding_and_cap():
    symbols = [BTC, ETH, SOL]
    weights = compute_funding_tilted_weights(
        symbols, {s: 1.0 for s in symbols}, mode="funding_tilted",
        tilt_alpha=0.5, max_single_weight=0.55,
    )
    assert set(weights) == set(symbols)
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-9)
    for w in weights.values():
        assert w <= 0.55 + 1e-9

    cap = 0.40
    skewed = compute_funding_tilted_weights(
        symbols, {BTC: 100.0, ETH: 1.0, SOL: 1.0}, mode="funding_tilted",
        tilt_alpha=0.5, max_single_weight=cap,
    )
    assert sum(skewed.values()) == pytest.approx(1.0, rel=1e-9)
    for w in skewed.values():
        assert w <= cap + 1e-9


def test_funding_tilted_weights_alpha_zero_and_backwardation():
    symbols = [BTC, ETH, SOL]
    equal = compute_funding_tilted_weights(
        symbols, {BTC: 50.0, ETH: 2.0, SOL: 0.0}, mode="funding_tilted", tilt_alpha=0.0
    )
    for w in equal.values():
        assert w == pytest.approx(1.0 / 3.0, rel=1e-9)

    negative = compute_funding_tilted_weights(
        symbols, {BTC: -5.0, ETH: 2.0, SOL: 1.0}, mode="funding_tilted",
        tilt_alpha=0.5, max_single_weight=0.55,
    )
    assert all(w >= 0.0 for w in negative.values())
    assert sum(negative.values()) == pytest.approx(1.0, rel=1e-9)


def test_enforce_split_wallet_futures_cap():
    clamped = enforce_split_wallet_futures_cap(
        carry_perp_notional_usd=1000.0,
        desired_overlay_gross_usd=50.0,
        futures_wallet_equity_usd=100.0,
        max_futures_leverage=3.0,
    )
    gross = clamped["carry_scale_factor"] * 1000.0 + clamped["allowed_overlay_gross_usd"]
    assert gross <= 100.0 * 3.0 + 1e-6
    assert clamped["clamped"] is True

    roomy = enforce_split_wallet_futures_cap(
        carry_perp_notional_usd=100.0,
        desired_overlay_gross_usd=50.0,
        futures_wallet_equity_usd=100.0,
        max_futures_leverage=3.0,
    )
    gross = roomy["carry_scale_factor"] * 100.0 + roomy["allowed_overlay_gross_usd"]
    assert gross == pytest.approx(150.0, **MONEY)
    assert roomy["clamped"] is False


def test_passive_yield_cases():
    base = PROFILE_REGISTRY["baseline_1k"]
    # Sprint 25.1: baseline_1k tracks yield stacking alongside its pure trading equity.
    assert base.enable_yield_stacking is True
    assert compute_step_passive_yield_usd(base, 0.0, {})["total_passive_yield_usd"] == 0.0
    assert compute_step_passive_yield_usd(base, 1.0, {})["total_passive_yield_usd"] > 0.0

    boosted = PROFILE_REGISTRY["alpha_boost_1k"]
    assert compute_step_passive_yield_usd(boosted, 0.0, {ETH: 100.0})[
        "total_passive_yield_usd"
    ] == 0.0

    res = compute_step_passive_yield_usd(boosted, 1.0, {ETH: 100.0, SOL: 50.0})
    total = res["total_passive_yield_usd"]
    assert total >= 0.0 and math.isfinite(total)
    assert total == pytest.approx(
        res["bfusd_yield_usd"] + res["idle_earn_yield_usd"] + res["staking_yield_usd"],
        **MONEY,
    )


# --------------------------------------------------------------- live demo --
def test_build_profiles():
    assert set(demo.build_profiles("baseline_1k,alpha_boost_1k")) == {
        "baseline_1k", "alpha_boost_1k"
    }
    derived = demo.build_profiles(["alpha_boost_1k_pure_equal"])["alpha_boost_1k_pure_equal"]
    assert derived.name == "alpha_boost_1k_pure_equal"
    assert derived.carry_weighting_mode == "equal"
    assert derived.carry_tilt_alpha == 0.0
    with pytest.raises(ValueError):
        demo.build_profiles("does_not_exist")


def test_demo_paths_are_distinct_and_named():
    ledger_path, bot_path = demo.demo_paths("baseline_1k", "data/state")
    assert ledger_path != bot_path
    assert "demo_baseline_1k" in os.path.basename(ledger_path)
    assert "demo_bot_baseline_1k" in os.path.basename(bot_path)
    assert {ledger_path, bot_path} & set(
        demo.demo_paths("alpha_boost_1k", "data/state")
    ) == set()


def test_load_ledger_defaults_and_round_trip(tmp_path, prof_1k):
    missing = demo.load_ledger(str(tmp_path / "missing.json"), prof_1k)
    assert missing["profile"] == "baseline_1k"
    assert missing["pure_trading_equity_usd"] == pytest.approx(prof_1k.capital_usd, **MONEY)

    path = tmp_path / "ledger.json"
    ledger = demo.load_ledger(str(path), prof_1k)
    ledger["realized_pnl_usd"] = 12.5
    ledger["status"] = "OK"
    ledger["cycle"] = 3
    ledger["positions_snapshot"] = {
        BTC: {"qty": 0.5, "entry_price": 100.0, "kind": "carry", "signed_usd": 50.0}}
    demo.save_ledger_atomic(str(path), ledger)
    loaded = demo.load_ledger(str(path), prof_1k)
    assert set(loaded) == set(ledger)
    for key, value in ledger.items():
        if key == "positions_snapshot":
            # The loader normalises legs and always stamps the symbol onto each leg.
            assert loaded[key][BTC] == {
                "symbol": BTC,
                "qty": 0.5,
                "entry_price": 100.0,
                "kind": "carry",
                "signed_usd": 50.0,
            }
        else:
            assert loaded[key] == value


def test_ledgers_are_isolated_per_profile(tmp_path, prof_1k):
    alpha = get_strategy_profile("alpha_boost_1k")
    ledger_a = demo.load_ledger(str(tmp_path / "a.json"), prof_1k)
    ledger_a["realized_pnl_usd"] = 99.0
    demo.save_ledger_atomic(str(tmp_path / "a.json"), ledger_a)
    ledger_b = demo.load_ledger(str(tmp_path / "b.json"), alpha)
    assert ledger_b["profile"] == "alpha_boost_1k"
    assert ledger_b["realized_pnl_usd"] == 0.0
    assert ledger_b["pure_trading_equity_usd"] == pytest.approx(alpha.capital_usd, **MONEY)


def _funding_ledger(tmp_path, profile):
    ledger = _fresh_ledger(tmp_path / "funding.json", profile)
    ledger["positions_snapshot"] = {
        BTC: {"qty": 0.5, "entry_price": 100.0, "kind": "carry", "signed_usd": 50.0}}
    return ledger


def test_accrue_cycle_accounting_funding_and_equity(tmp_path):
    prof = get_strategy_profile("alpha_boost_1k")
    ledger = _funding_ledger(tmp_path, prof)
    snapshot = {"symbols": {BTC: {"mid": 110.0, "funding_rate": 0.0001}}}
    t0 = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

    demo.accrue_cycle_accounting(ledger, prof, snapshot, t0)
    assert ledger["last_funding_ts"] is not None
    assert ledger["funding_pnl_usd"] == 0.0
    # Carry is delta neutral (spot long + perp short): a 10% price move creates NO PnL.
    assert ledger["pure_trading_equity_usd"] == pytest.approx(prof.capital_usd, **MONEY)

    demo.accrue_cycle_accounting(ledger, prof, snapshot, t0 + timedelta(hours=8))
    expected_funding = 0.5 * 110.0 * 0.0001
    assert ledger["funding_pnl_usd"] == pytest.approx(expected_funding, **MONEY)
    assert ledger["realized_pnl_usd"] == pytest.approx(expected_funding, **MONEY)
    assert ledger["pure_trading_equity_usd"] == pytest.approx(
        prof.capital_usd + expected_funding, **MONEY)


def test_accrue_cycle_accounting_marks_overlay_to_market(tmp_path):
    prof = get_strategy_profile("alpha_boost_1k")
    ledger = _fresh_ledger(tmp_path / "overlay.json", prof)
    ledger["positions_snapshot"] = {
        f"{ETH}|overlay": {
            "symbol": ETH,
            "qty": 0.5,
            "entry_price": 100.0,
            "kind": "overlay",
            "signed_usd": 50.0,
        }
    }
    t0 = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    demo.accrue_cycle_accounting(
        ledger, prof, {"symbols": {ETH: {"mid": 110.0, "funding_rate": 0.0}}}, t0)
    # Directional overlay leg: +10% on a 50 USD book = +5 USD.
    assert ledger["pure_trading_equity_usd"] == pytest.approx(
        prof.capital_usd + 5.0, **MONEY)


def test_accrue_cycle_accounting_max_drawdown(tmp_path, prof_1k):
    ledger = _fresh_ledger(tmp_path / "dd.json", prof_1k)
    # Directional overlay leg so equity can actually fall (carry is delta neutral).
    ledger["positions_snapshot"] = {
        f"{BTC}|overlay": {
            "symbol": BTC,
            "qty": 5.0,
            "entry_price": 100.0,
            "kind": "overlay",
            "signed_usd": 500.0,
        }
    }
    t0 = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    demo.accrue_cycle_accounting(
        ledger, prof_1k, {"symbols": {BTC: {"mid": 90.0, "funding_rate": 0.0}}}, t0)
    dd_first, peak_first = ledger["max_dd_pct"], ledger["peak_equity_usd"]
    assert dd_first > 0.0

    demo.accrue_cycle_accounting(
        ledger, prof_1k, {"symbols": {BTC: {"mid": 80.0, "funding_rate": 0.0}}}, t0)
    assert ledger["max_dd_pct"] >= dd_first > 0.0
    assert ledger["peak_equity_usd"] >= peak_first
    assert ledger["peak_equity_usd"] == pytest.approx(prof_1k.capital_usd, **MONEY)


def test_apply_shadow_fills_charges_fees_on_delta_notional(tmp_path, prof_1k):
    carry_ledger = _fresh_ledger(tmp_path / "carry.json", prof_1k)
    carry_fee = demo.apply_shadow_fills(
        carry_ledger, FakeExecState(carry_positions={BTC: {"qty": 1.0, "notional": 100.0}}),
        {"symbols": {BTC: {"mid": 100.0}}}, prof_1k,
    )
    assert carry_fee == pytest.approx(100.0 * demo.CARRY_FEE_RATE, **MONEY)

    overlay_ledger = _fresh_ledger(tmp_path / "overlay.json", prof_1k)
    overlay_fee = demo.apply_shadow_fills(
        overlay_ledger,
        FakeExecState(overlay_positions={ETH: {"qty": 1.0, "signed_usd": 100.0}}),
        {"symbols": {ETH: {"mid": 100.0}}}, prof_1k,
    )
    assert overlay_fee == pytest.approx(100.0 * demo.OVERLAY_FEE_RATE, **MONEY)


def test_apply_shadow_fills_unchanged_positions_charge_no_fee(tmp_path, prof_1k):
    ledger = _fresh_ledger(tmp_path / "idem.json", prof_1k)
    state = FakeExecState(
        carry_positions={BTC: {"qty": 1.0, "notional": 100.0}},
        overlay_positions={ETH: {"qty": 1.0, "signed_usd": 100.0}},
    )
    snapshot = {"symbols": {BTC: {"mid": 100.0}, ETH: {"mid": 100.0}}}

    first_fee = demo.apply_shadow_fills(ledger, state, snapshot, prof_1k)
    assert first_fee == pytest.approx(
        100.0 * demo.CARRY_FEE_RATE + 100.0 * demo.OVERLAY_FEE_RATE, **MONEY
    )
    second_fee = demo.apply_shadow_fills(ledger, state, snapshot, prof_1k)
    assert second_fee == pytest.approx(0.0, abs=1e-6)
    assert abs(second_fee) < 1e-9


def test_apply_shadow_fills_keeps_entry_and_realises_on_close(tmp_path, prof_1k):
    ledger = _fresh_ledger(tmp_path / "entry.json", prof_1k)
    state = FakeExecState(carry_positions={BTC: {"qty": 0.5, "notional": 50.0}})
    demo.apply_shadow_fills(ledger, state, {"symbols": {BTC: {"mid": 100.0}}}, prof_1k)
    assert ledger["positions_snapshot"][BTC]["entry_price"] == pytest.approx(100.0, **MONEY)

    demo.apply_shadow_fills(ledger, state, {"symbols": {BTC: {"mid": 110.0}}}, prof_1k)
    assert ledger["positions_snapshot"][BTC]["entry_price"] == pytest.approx(100.0, **MONEY)
    # Delta-neutral carry: a 10% move must not move equity.
    assert ledger["pure_trading_equity_usd"] == pytest.approx(
        prof_1k.capital_usd - ledger["fees_paid_usd"], **MONEY)

    realized_before = ledger["realized_pnl_usd"]
    state.carry_positions = {}
    demo.apply_shadow_fills(ledger, state, {"symbols": {BTC: {"mid": 110.0}}}, prof_1k)
    assert ledger["positions_snapshot"] == {}
    # Closing a carry leg costs the exit fee only; there is no directional price PnL.
    assert ledger["realized_pnl_usd"] - realized_before == pytest.approx(
        -0.5 * 110.0 * demo.CARRY_FEE_RATE, **MONEY
    )


def test_apply_shadow_fills_realises_overlay_pnl_on_partial_close(tmp_path, prof_1k):
    ledger = _fresh_ledger(tmp_path / "ovclose.json", prof_1k)
    state = FakeExecState(overlay_positions={ETH: {"qty": 1.0, "signed_usd": 100.0}})
    demo.apply_shadow_fills(ledger, state, {"symbols": {ETH: {"mid": 100.0}}}, prof_1k)
    demo.apply_shadow_fills(ledger, state, {"symbols": {ETH: {"mid": 110.0}}}, prof_1k)
    assert ledger["pure_trading_equity_usd"] == pytest.approx(
        prof_1k.capital_usd + 10.0 - ledger["fees_paid_usd"], **MONEY)

    # Halve the leg at 110: half the +10 USD gain must be realised, entry price untouched.
    state.overlay_positions = {ETH: {"qty": 0.5, "signed_usd": 50.0}}
    demo.apply_shadow_fills(ledger, state, {"symbols": {ETH: {"mid": 110.0}}}, prof_1k)
    assert ledger["positions_snapshot"][f"{ETH}|overlay"]["entry_price"] == pytest.approx(
        100.0, **MONEY)
    assert ledger["realized_pnl_usd"] == pytest.approx(
        5.0 - ledger["fees_paid_usd"], **MONEY)
    assert ledger["pure_trading_equity_usd"] == pytest.approx(
        prof_1k.capital_usd + 10.0 - ledger["fees_paid_usd"], **MONEY)


def test_print_scoreboard_ascii_and_contents(capsys):
    record = {
        "profile": "baseline_1k", "mode": "SHADOW_LIVE", "pure_trading_equity_usd": 1000.0,
        "yield_stacked_equity_usd": 1000.0, "net_pnl_usd": 0.0, "status": "OK",
    }
    demo.print_scoreboard([record])
    out = capsys.readouterr().out
    assert all(ord(ch) < 128 for ch in out)
    assert "baseline_1k" in out
    assert "OK" in out


def test_append_scoreboard_writes_one_jsonl_line_per_call(tmp_path):
    path = tmp_path / "scoreboard.jsonl"
    demo.append_scoreboard(str(path), {"profile": "baseline_1k", "cycle": 1})
    demo.append_scoreboard(str(path), {"profile": "baseline_1k", "cycle": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["cycle"] for line in lines] == [1, 2]


def test_fetch_market_once_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(demo, "fetch_binance_shadow_snapshot", boom)
    result = demo.fetch_market_once(timeout_sec=0.01, spot_timeout_sec=0.01)
    assert result["healthy"] is False
    assert "errors" in result
    assert result["spot_symbols"] == {}


# ------------------------------------------- sprint 25.1 regression guards ----
def test_live_default_profile_is_baseline_1k():
    """Audit fix 1: baseline_1k (beat alpha_boost on 181d holdout) is now the default."""
    assert demo.parse_args([]).live_testnet_profile == "baseline_1k"


def test_default_and_baseline_track_yield_stacking():
    assert PROFILE_REGISTRY["baseline_1k"].enable_yield_stacking is True
    assert PROFILE_REGISTRY["default"].enable_yield_stacking is True
    # Pure trading equity is untouched: yield is reported in its own column only.
    base = PROFILE_REGISTRY["baseline_1k"]
    res = compute_step_passive_yield_usd(base, 1.0, {})
    assert res["total_passive_yield_usd"] > 0.0


def test_tournament_includes_default_3000_profile():
    profiles = tournament.build_tournament_profiles()
    # Sprint 26 keeps `default` runnable in the tournament, but the DEFAULT_PROFILES roster
    # is now the 1k evidence set (the 3000 USDT mainnet profile stays opt-in via --profiles).
    assert "default" in profiles
    assert profiles["default"].capital_usd == pytest.approx(3000.0, **MONEY)
    assert "baseline_1k" in tournament.DEFAULT_PROFILES
    assert "hybrid_trend_conservative_1k" in tournament.DEFAULT_PROFILES
    assert set(profiles) == {
        "default",
        "baseline_1k",
        "carry_max_eta_1k",
        "hybrid_trend_conservative_1k",
        "hybrid_trend_balanced_1k",
        "alpha_boost_1k",
        "turbo_neutral_1k",
        "alpha_boost_1k_pure_equal", "bi_trend_carry_1k", "carry_stat_arb_1k", "regime_omni_1k", "flexible_regime_1k",
    }


def _synthetic_carry_market(funding_bars: list[int], n: int = 48) -> dict:
    index = pd.date_range("2024-01-01 00:00", periods=n, freq="h", tz="UTC")
    cols = [BTC, ETH]
    perp = pd.DataFrame({s: np.full(n, 100.0) for s in cols}, index=index)
    spot = perp.copy()
    funding = pd.DataFrame({s: np.zeros(n) for s in cols}, index=index)
    for bar in funding_bars:
        funding.loc[index[bar], BTC] = 0.001
    score = pd.DataFrame({BTC: np.full(n, 1.0), ETH: np.full(n, -1.0)}, index=index)
    snr = pd.DataFrame(np.nan, index=index, columns=cols)
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


def test_regression_no_funding_lookahead_on_entry_bar():
    """A leg opened at bar t must NOT receive bar t's funding (audit bug 1)."""
    prof = PROFILE_REGISTRY["baseline_1k"]
    open_bar_only = tournament.simulate_profile(
        prof, _synthetic_carry_market(funding_bars=[0]), start_idx=0, end_idx=48,
        trade_start_idx=0,
    )
    assert open_bar_only["diagnostics"]["n_carry_opens"] > 0
    assert open_bar_only["funding_pnl_usd"] == pytest.approx(0.0, abs=1e-12)

    later_bar_too = tournament.simulate_profile(
        prof, _synthetic_carry_market(funding_bars=[0, 8]), start_idx=0, end_idx=48,
        trade_start_idx=0,
    )
    # Only the *subsequent* funding stamp pays the position opened at bar 0.
    assert later_bar_too["funding_pnl_usd"] > 0.0
    assert later_bar_too["funding_pnl_usd"] == pytest.approx(
        0.5 * (later_bar_too["funding_pnl_usd"]) * 2.0, rel=1e-9
    )


def _carry_leg(qty: float = 1.0, spot_entry: float = 100.0, perp_entry: float = 100.0) -> dict:
    return {
        "symbol": BTC,
        "qty": qty,
        "entry_price": perp_entry,
        "kind": "carry",
        "signed_usd": qty * perp_entry,
        "spot_qty": qty,
        "perp_qty": -qty,
        "spot_entry": spot_entry,
        "perp_entry": perp_entry,
    }


def _dual_snapshot(spot_mid: float, perp_mark: float) -> dict:
    return {
        "symbols": {BTC: {"mid": perp_mark}},
        "spot_symbols": {BTC: {"bid": spot_mid, "ask": spot_mid, "mid": spot_mid}},
        "perp_symbols": {BTC: {"mid": perp_mark, "mark": perp_mark}},
    }


def test_regression_carry_delta_neutral_price_pnl_and_basis(tmp_path, prof_1k):
    """Carry must be flat when Spot/Perp move together and carry only the real basis."""
    ledger = _fresh_ledger(tmp_path / "basis.json", prof_1k)
    ledger["positions_snapshot"] = {BTC: _carry_leg()}
    t0 = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

    demo.accrue_cycle_accounting(ledger, prof_1k, _dual_snapshot(110.0, 110.0), t0)
    assert ledger["basis_pnl_usd"] == pytest.approx(0.0, abs=1e-9)
    assert ledger["pure_trading_equity_usd"] == pytest.approx(prof_1k.capital_usd, **MONEY)

    # Spot +10% / Perp +12%: basis = 1*(110-100) + (-1)*(112-100) = -2.00 USD.
    demo.accrue_cycle_accounting(ledger, prof_1k, _dual_snapshot(110.0, 112.0), t0)
    assert ledger["spot_unrealized_pnl_usd"] == pytest.approx(10.0, **MONEY)
    assert ledger["perp_unrealized_pnl_usd"] == pytest.approx(-12.0, **MONEY)
    assert ledger["basis_pnl_usd"] == pytest.approx(-2.0, **MONEY)
    assert ledger["pure_trading_equity_usd"] == pytest.approx(
        prof_1k.capital_usd - 2.0, **MONEY
    )

    # Without a Spot book the basis is not observable -> treated as neutral, never invented.
    no_spot = {"symbols": {BTC: {"mid": 112.0}}, "perp_symbols": {BTC: {"mid": 112.0, "mark": 112.0}}}
    demo.accrue_cycle_accounting(ledger, prof_1k, no_spot, t0)
    assert ledger["basis_pnl_usd"] == pytest.approx(0.0, abs=1e-9)


def test_regression_carry_and_overlay_same_symbol_key_isolation(tmp_path, prof_1k):
    """Carry and Overlay on the same symbol must not share one ledger slot (audit bug 3)."""
    ledger = _fresh_ledger(tmp_path / "isolation.json", prof_1k)
    state = FakeExecState(
        carry_positions={
            ETH: {"qty": 1.0, "entry_spot_price": 100.0, "entry_perp_price": 101.0}
        },
        overlay_positions={ETH: {"qty": 2.0, "signed_usd": 200.0}},
    )
    snapshot = {
        "symbols": {ETH: {"mid": 100.0}},
        "spot_symbols": {ETH: {"mid": 100.0}},
        "perp_symbols": {ETH: {"mid": 100.0, "mark": 100.0}},
    }
    demo.apply_shadow_fills(ledger, state, snapshot, prof_1k)
    assert set(ledger["positions_snapshot"]) == {ETH, f"{ETH}|overlay"}
    carry_leg = ledger["positions_snapshot"][ETH]
    overlay_leg = ledger["positions_snapshot"][f"{ETH}|overlay"]
    assert carry_leg["kind"] == "carry"
    assert overlay_leg["kind"] == "overlay"
    assert carry_leg["spot_entry"] == pytest.approx(100.0, **MONEY)
    assert carry_leg["perp_entry"] == pytest.approx(101.0, **MONEY)
    assert overlay_leg["signed_usd"] == pytest.approx(200.0, **MONEY)

    # Closing the Overlay must never delete or overwrite the Carry slot.
    state.overlay_positions = {}
    demo.apply_shadow_fills(ledger, state, snapshot, prof_1k)
    assert set(ledger["positions_snapshot"]) == {ETH}
    assert ledger["positions_snapshot"][ETH]["spot_entry"] == pytest.approx(100.0, **MONEY)


def test_regression_partial_downsize_vwap_realization(tmp_path, prof_1k):
    """Downsize realises pro-rata PnL, upsize averages entries (audit bug 4)."""
    ledger = _fresh_ledger(tmp_path / "vwap.json", prof_1k)
    state = FakeExecState(overlay_positions={ETH: {"qty": 1.0, "signed_usd": 100.0}})
    snap_100 = {"symbols": {ETH: {"mid": 100.0}}}
    snap_120 = {"symbols": {ETH: {"mid": 120.0}}}
    snap_130 = {"symbols": {ETH: {"mid": 130.0}}}

    demo.apply_shadow_fills(ledger, state, snap_100, prof_1k)
    assert ledger["positions_snapshot"][f"{ETH}|overlay"]["entry_price"] == pytest.approx(
        100.0, **MONEY
    )

    # Scale 100 -> 200 USD at 120: VWAP = (100*100 + 100*120) / 200 = 110.
    demo.apply_shadow_fills(ledger, state, snap_120, prof_1k)
    state.overlay_positions = {ETH: {"qty": 200.0 / 120.0, "signed_usd": 200.0}}
    demo.apply_shadow_fills(ledger, state, snap_120, prof_1k)
    leg = ledger["positions_snapshot"][f"{ETH}|overlay"]
    assert leg["entry_price"] == pytest.approx(110.0, **MONEY)
    assert leg["signed_usd"] == pytest.approx(200.0, **MONEY)

    # Scale down 200 -> 100 USD at 130: half of the +36.36 gain is realised, entry stays 110.
    realized_before = ledger["realized_pnl_usd"]
    state.overlay_positions = {ETH: {"qty": 100.0 / 130.0, "signed_usd": 100.0}}
    demo.apply_shadow_fills(ledger, state, snap_130, prof_1k)
    leg = ledger["positions_snapshot"][f"{ETH}|overlay"]
    assert leg["entry_price"] == pytest.approx(110.0, **MONEY)
    assert ledger["realized_pnl_usd"] - realized_before == pytest.approx(
        200.0 * (130.0 / 110.0 - 1.0) * 0.5 - 100.0 * demo.OVERLAY_FEE_RATE, **MONEY
    )


def test_dual_price_maps_split_spot_and_perp():
    snapshot = {
        "symbols": {BTC: {"bid": 100.0, "ask": 102.0, "mid": 101.0, "mark_price": 101.5}},
        "spot_symbols": {BTC: {"bid": 99.0, "ask": 101.0, "mid": 100.0}},
        "perp_symbols": {BTC: {"bid": 100.0, "ask": 102.0, "mid": 101.0, "mark": 101.5}},
    }
    spot, perp = demo.dual_price_maps(snapshot)
    assert spot[BTC] == {"bid": 99.0, "ask": 101.0, "mid": 100.0}
    assert perp[BTC]["mark"] == pytest.approx(101.5, **MONEY)
    assert perp[BTC]["mid"] == pytest.approx(101.0, **MONEY)

    # Legacy snapshot (Perp only) must still produce a Perp map and no invented Spot book.
    legacy_spot, legacy_perp = demo.dual_price_maps({"symbols": {BTC: {"mid": 100.0}}})
    assert legacy_spot == {}
    assert legacy_perp[BTC]["mid"] == pytest.approx(100.0, **MONEY)


def test_fetch_spot_book_tickers_fixture_and_fail_closed():
    good = demo.fetch_binance_spot_book_tickers(
        [BTC, ETH], fixture_data={BTC: {"bid": 99.0, "ask": 101.0}}
    )
    assert good == {BTC: {"bid": 99.0, "ask": 101.0, "mid": 100.0}}

    crossed = demo.fetch_binance_spot_book_tickers(
        [BTC], fixture_data={BTC: {"bid": 101.0, "ask": 100.0}}
    )
    assert crossed == {}


class _ReconClient:
    """Minimal fake signed client exposing only the reconcile snapshot call."""

    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def fetch_exchange_reconcile_snapshot(self, symbols, prices=None, **kwargs):
        self.calls += 1
        return self.snapshot


def _healthy_exchange_snapshot(spot_usdt: float = 1000.0) -> dict:
    return {
        "snapshot_healthy": True,
        "spot_balances": {"USDT": spot_usdt},
        "spot_balances_raw": {"USDT": spot_usdt},
        "perp_positions": {},
        "perp_positions_detail": [],
        "pure_cash_usd": spot_usdt,
        "spot_cash_usd": spot_usdt,
        "perp_wallet_balance_usd": 0.0,
        "perp_unrealized_pnl_usd": 0.0,
        "exchange_equity_usd": spot_usdt,
        "universe_symbols": [BTC],
    }


def test_exchange_reconcile_synced_and_drift_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "spot-key")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "futures-key")
    baseline_path = str(tmp_path / "demo_testnet_baseline.json")

    client = _ReconClient(_healthy_exchange_snapshot(1000.0))
    synced = demo.compute_exchange_reconcile(
        client, [BTC], {BTC: 100.0}, baseline_path, 1000.0, 1000.0
    )
    assert synced["reconcile_status"] == "SYNCED"
    assert synced["exchange_recon_equity_usd"] == pytest.approx(1000.0, abs=1e-6)
    assert synced["recon_drift_usd"] == pytest.approx(0.0, abs=1e-6)
    assert os.path.exists(baseline_path)

    # Local ledger claims 1,050 USD while the exchange only holds 1,000 -> 5% drift.
    drift = demo.compute_exchange_reconcile(
        client, [BTC], {BTC: 100.0}, baseline_path, 1000.0, 1050.0
    )
    assert drift["reconcile_status"] == "DRIFT_WARNING"
    assert drift["recon_drift_usd"] == pytest.approx(-50.0, abs=1e-6)

    unhealthy = demo.compute_exchange_reconcile(
        _ReconClient({"snapshot_healthy": False}), [BTC], {}, baseline_path, 1000.0, 1000.0
    )
    assert unhealthy["reconcile_status"] == "UNAVAILABLE"
    assert unhealthy["exchange_recon_equity_usd"] is None


def test_initialize_demo_testnet_baseline_idempotent_and_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "spot-key")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "futures-key")

    baseline_path = str(tmp_path / "demo_testnet_baseline.json")
    assert os.path.exists(baseline_path) is False
    # Existing file -> reused, no exchange call needed.
    with open(baseline_path, "w", encoding="utf-8") as fh:
        json.dump({"credentials_sha256": "x"}, fh)
    ok, note = demo.initialize_demo_testnet_baseline(
        _ReconClient({"snapshot_healthy": False}), baseline_path, [BTC]
    )
    assert ok is True
    assert note == "BASELINE_EXISTS"

    # Unhealthy snapshot -> refuse (fail-closed), operator is told why.
    fresh_path = str(tmp_path / "fresh.json")
    ok, note = demo.initialize_demo_testnet_baseline(
        _ReconClient({"snapshot_healthy": False}), fresh_path, [BTC]
    )
    assert ok is False
    assert note == "snapshot_unhealthy"
    assert os.path.exists(fresh_path) is False


def test_run_profile_cycle_records_reconciliation_fields(tmp_path, prof_1k):
    class FakeExecutor:
        def __init__(self) -> None:
            self.state = FakeExecState()
            self.calls = 0

        def reconcile_once(self, aligned_universe=None):
            self.calls += 1
            return None

    ledger = _fresh_ledger(tmp_path / "cycle.json", prof_1k)
    executor = FakeExecutor()
    rec = demo.run_profile_cycle(
        prof_1k,
        ledger,
        {"symbols": {}},
        None,
        mode="TESTNET_REAL",
        execute_real=True,
        executor=executor,
        reconcile_fn=lambda _ledger: {
            "exchange_recon_equity_usd": 1003.0,
            "recon_drift_usd": 3.0,
            "reconcile_status": "SYNCED",
        },
    )
    assert executor.calls == 1
    assert rec["reconcile_status"] == "SYNCED"
    assert rec["exchange_recon_equity_usd"] == pytest.approx(1003.0, **MONEY)
    assert rec["recon_drift_usd"] == pytest.approx(3.0, **MONEY)
    assert rec["basis_pnl_usd"] == pytest.approx(0.0, abs=1e-12)
    assert ledger["reconcile_status"] == "SYNCED"


# ---------------------------------------------------------------------------
# Forensic Patch Tests: (a) -1021 resync, (b) recon detail & note dedup,
# (c) baseline_1k overlay disable & auto-flatten, (d) orphan pending_close rescue
# ---------------------------------------------------------------------------


def _mock_universe(n_bars: int = 750) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2025-11-01", periods=n_bars, freq="1h", tz="UTC")
    out: dict[str, pd.DataFrame] = {}
    for sym, px in ((BTC, 100.0), (ETH, 50.0), (SOL, 20.0)):
        out[sym] = pd.DataFrame(
            {
                "close": px,
                "spot_close": px,
                "perp_close": px * 1.0005,
                "funding_rate_event": 0.0001,
                "last_known_funding_rate": 0.0001,
                "basis_pct": 0.0005,
            },
            index=idx,
        )
    return out


def test_signed_client_auto_resyncs_server_time_on_1021(monkeypatch):
    from src.binance_signed_client import (
        BinanceHttpError,
        BinanceSignedTestnetClient,
        PERP_ACCOUNT_PATH,
        PERP_POSITION_RISK_PATH,
        SPOT_ACCOUNT_PATH,
        VENUE_FUTURES,
    )

    client = BinanceSignedTestnetClient(
        api_key="test-key",
        api_secret="test-secret",
        dry_run=False,
        autoload_dotenv=False,
    )
    sync_calls: list[str] = []

    def _fake_sync(venue: str = "both") -> dict[str, int]:
        sync_calls.append(venue)
        client.futures_time_offset_ms = 2500
        client.spot_time_offset_ms = 2500
        return {"spot_offset_ms": 2500, "futures_offset_ms": 2500}

    monkeypatch.setattr(client, "sync_server_time", _fake_sync)

    # 1) submit_signed_order: first POST raises -1021, retry succeeds with new signature
    post_timestamps: list[int] = []

    def _fake_http_order(method, url, params=None, headers=None, timeout_sec=10.0):
        post_timestamps.append(int((params or {}).get("timestamp", 0)))
        if len(post_timestamps) == 1:
            raise BinanceHttpError(
                "Binance API loi -1021: Timestamp for this request is outside of the recvWindow.",
                code=-1021,
                msg="Timestamp for this request is outside of the recvWindow.",
            )
        return {
            "orderId": 999,
            "status": "FILLED",
            "executedQty": "0.01",
            "avgPrice": "50000.0",
        }

    monkeypatch.setattr(client, "_http_request", _fake_http_order)
    req = client.build_order_request(VENUE_FUTURES, BTC, "BUY", 0.01, "MARKET")
    res = client.submit_signed_order(VENUE_FUTURES, req)
    assert res["filled"] is True
    assert res["orderId"] == 999
    assert len(post_timestamps) == 2
    assert sync_calls == [VENUE_FUTURES]

    # 2) fetch_exchange_reconcile_snapshot: first attempt raises -1021, retry succeeds
    sync_calls.clear()
    get_calls = {"n": 0}

    def _fake_http_snap(method, url, params=None, headers=None, timeout_sec=5.0):
        get_calls["n"] += 1
        if get_calls["n"] == 1:
            raise BinanceHttpError(
                "Binance API loi -1021: Timestamp for this request is outside of the recvWindow.",
                code=-1021,
                msg="Timestamp for this request is outside of the recvWindow.",
            )
        if SPOT_ACCOUNT_PATH in url:
            return {"balances": [{"asset": "USDT", "free": "650.0", "locked": "0.0"}]}
        if PERP_ACCOUNT_PATH in url:
            return {"totalWalletBalance": "350.0", "totalUnrealizedProfit": "0.0"}
        if PERP_POSITION_RISK_PATH in url:
            return []
        return {}

    monkeypatch.setattr(client, "_http_request", _fake_http_snap)
    snap = client.fetch_exchange_reconcile_snapshot([BTC], {BTC: 50000.0})
    assert snap["snapshot_healthy"] is True
    assert snap["exchange_equity_usd"] == pytest.approx(1000.0, **MONEY)
    assert "both" in sync_calls


def test_recon_snapshot_unhealthy_preserves_detail_and_note_does_not_accumulate(tmp_path, prof_1k):
    baseline_path = str(tmp_path / "demo_testnet_baseline.json")
    unhealthy = demo.compute_exchange_reconcile(
        _ReconClient({"snapshot_healthy": False, "error": "Network fetch failed: HTTP 400 [-1021]"}),
        [BTC],
        {BTC: 100.0},
        baseline_path,
        1000.0,
        1000.0,
    )
    assert unhealthy["reconcile_status"] == "UNAVAILABLE"
    assert "recon_snapshot_unhealthy:Network fetch failed: HTTP 400 [-1021]" in unhealthy["note"]

    class _SteppingExecutor:
        def __init__(self) -> None:
            self.state = FakeExecState()
            self.step = 0

        def reconcile_once(self, aligned_universe=None):
            self.step += 1
            # Change position qty each cycle so refresh_position_ledger charges a fee every cycle
            self.state.carry_positions = {
                BTC: {
                    "qty": float(self.step),
                    "entry_spot_price": 100.0,
                    "entry_perp_price": 100.0,
                    "notional_usdt": 100.0 * self.step,
                    "funding_collected_usdt": 0.0,
                }
            }
            return None

    ledger = _fresh_ledger(tmp_path / "cycle_note.json", prof_1k)
    executor = _SteppingExecutor()
    snap = {"symbols": {BTC: {"spot_mid": 100.0, "perp_mark": 100.0, "mid": 100.0}}}

    for _ in range(4):
        demo.run_profile_cycle(
            prof_1k,
            ledger,
            snap,
            None,
            mode="TESTNET_REAL",
            execute_real=True,
            executor=executor,
            reconcile_fn=lambda _l: unhealthy,
        )

    note_tokens = [t for t in ledger["note"].split(" ") if t]
    assert note_tokens.count("fees_estimated") == 1
    assert "recon_snapshot_unhealthy:Network" in ledger["note"]


def test_baseline_1k_disables_overlay_and_flattens_existing_overlay_positions(tmp_path):
    from src.carry_live_executor import CarryLiveExecutor

    prof = get_strategy_profile("baseline_1k")
    assert prof.overlay_gross_pct == 0.0
    assert prof.overlay_target_vol_ann == 0.0

    state_path = str(tmp_path / "demo_bot_baseline_1k.json")
    executor = demo.build_demo_executor(
        prof,
        mode="paper",
        bot_state_path=state_path,
        raw_dir=str(tmp_path),
        execute_real=False,
        signed_client=None,
    )
    assert isinstance(executor, CarryLiveExecutor)
    assert executor.overlay_gross_pct == 0.0
    assert executor.overlay_target_vol_ann == 0.0
    assert executor.overlay_config.enable_overlay is False

    # Seed legacy overlay positions on baseline_1k (simulating Laptop live state before patch)
    executor.state.overlay_positions = {
        BTC: {"qty": 1.0, "side": "long", "entry_price": 91.72, "ts": "2026-01-01T00:00:00+00:00"},
        ETH: {"qty": 2.0, "side": "short", "entry_price": 52.85, "ts": "2026-01-01T00:00:00+00:00"},
    }
    executor.state_store.save(executor.state)

    universe = _mock_universe(n_bars=750)
    ledger = _fresh_ledger(tmp_path / "ledger_baseline_1k.json", prof)
    snap = {
        "symbols": {
            BTC: {"spot_mid": 100.0, "perp_mark": 100.0, "mid": 100.0},
            ETH: {"spot_mid": 50.0, "perp_mark": 50.0, "mid": 50.0},
            SOL: {"spot_mid": 20.0, "perp_mark": 20.0, "mid": 20.0},
        }
    }
    rec = demo.run_profile_cycle(
        prof,
        ledger,
        snap,
        universe,
        mode="SHADOW_LIVE",
        execute_real=False,
        executor=executor,
    )
    assert executor.state.overlay_positions == {}
    assert executor.state.momentum_overlay.get("allocations") == []
    assert executor.state.momentum_overlay.get("overlay_active") is False
    assert ledger["overlay_long_usd"] == pytest.approx(0.0, abs=1e-9)
    assert ledger["overlay_short_usd"] == pytest.approx(0.0, abs=1e-9)
    assert rec["overlay_long_usd"] == pytest.approx(0.0, abs=1e-9)
    assert rec["overlay_short_usd"] == pytest.approx(0.0, abs=1e-9)


def test_orphan_kill_switch_pending_close_rescued_and_real_overlay_flattens_on_empty_targets(tmp_path):
    prof = get_strategy_profile("regime_omni_1k")
    state_path = str(tmp_path / "demo_bot_regime_omni_1k.json")

    placed_orders: list[dict] = []

    class _FakeRealSignedClient:
        def place_futures_market_order(self, symbol, side, quantity, reduce_only=False, timeout_sec=10.0):
            placed_orders.append(
                {"symbol": symbol, "side": side, "qty": float(quantity), "reduce_only": bool(reduce_only)}
            )
            return {
                "orderId": len(placed_orders),
                "status": "FILLED",
                "executedQty": float(quantity),
                "avgPrice": 100.0,
                "filled": True,
            }

        def place_spot_market_order(self, symbol, side, quantity, timeout_sec=10.0):
            return {
                "orderId": 100 + len(placed_orders),
                "status": "FILLED",
                "executedQty": float(quantity),
                "avgPrice": 100.0,
                "filled": True,
            }

        def get_spot_free_balance(self, symbol):
            return 100.0

        def fetch_exchange_reconcile_snapshot(self, symbols, prices=None):
            return {"snapshot_healthy": True, "perp_positions": {}, "spot_balances": {}}

    fake_client = _FakeRealSignedClient()
    executor = demo.build_demo_executor(
        prof,
        mode="testnet",
        bot_state_path=state_path,
        raw_dir=str(tmp_path),
        execute_real=True,
        signed_client=fake_client,
    )
    # 1) Direct test: execute_overlay_rebalance with empty targets & existing overlay_positions
    # routes through _execute_overlay_deltas_real (using signed_client) and does NOT set pending_close
    executor.state.overlay_positions = {
        BTC: {"qty": 0.5, "side": "long", "entry_price": 100.0},
        ETH: {"qty": 1.0, "side": "short", "entry_price": 50.0},
    }
    universe = _mock_universe(n_bars=750)
    ov_res = executor.execute_overlay_rebalance(
        {"overlay_active": False, "allocations": []},
        universe,
    )
    assert executor.state.overlay_positions == {}
    assert executor.state.kill_switch_pending_close is False
    assert len(placed_orders) == 2
    assert ov_res["n_orders"] == 2

    # 2) Full reconcile_once test: orphan kill_switch_pending_close=True (with kill_switch_triggered=False)
    # is automatically cleared at the start of reconcile_once() so the bot is NOT deadlocked.
    executor.state.kill_switch_triggered = False
    executor.state.kill_switch_pending_close = True
    executor.state_store.save(executor.state)

    plan = executor.reconcile_once(universe)
    assert not plan.get("kill_switch", False)
    assert executor.state.kill_switch_pending_close is False
    assert executor.state.kill_switch_triggered is False

