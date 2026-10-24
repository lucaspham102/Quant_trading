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

EXPECTED_PROFILES = {"default", "canary_400", "baseline_1k", "alpha_boost_1k", "turbo_neutral_1k"}
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
    assert len(PROFILE_REGISTRY) == 5


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
    assert base.enable_yield_stacking is False
    assert compute_step_passive_yield_usd(base, 1.0, {})["total_passive_yield_usd"] == 0.0

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
    result = demo.fetch_market_once(timeout_sec=0.01)
    assert result["healthy"] is False
    assert "errors" in result
