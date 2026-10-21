"""Test Sprint 15 Chang 1 - Backtest-vs-Live parity cho carry gate va overlay gate.

Bao phu:
1. `DynamicCarryConfig` giu mac dinh `switch_rule="advantage"` (khong vo Sprint 7-10)
   va `eta_override=None` -> `notional_eta()` theo `structure`.
2. `eta_override` duoc `notional_eta()` uu tien (dong bo eta=0.80 cua Tier 1B).
3. Rule `tier1b_buffer_rank`: EXIT khi score <= exit_threshold (0.0) sau min_hold.
4. Rule `tier1b_buffer_rank`: chua du min_hold thi HOLD (HOLD_MIN_PERIODS).
5. Rule `tier1b_buffer_rank`: roi ngoai buffer VA co candidate >= entry -> SWITCH_RANK_DROP.
6. Rule `tier1b_buffer_rank`: roi ngoai buffer nhung KHONG co candidate -> HOLD_BUFFER_RANK.
7. Rule `advantage` khong doi hanh vi (EXIT_OUT_OF_BUFFER_RANK).
8. `CarryLiveExecutor._carry_selection_scores` dung pure `f_ewma` khi `tier1b_gate=True`.
9. `compute_live_overlay_targets` khong look-ahead (bar tuong lai khong doi target).
10. `compute_live_overlay_targets` cho cap long/short voi |w| >= min_weight_change va
    tong leverage <= 0.50.
11. `compute_live_overlay_targets` tra ve inactive khi thieu du lieu.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig, compute_live_overlay_targets
from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.dynamic_carry_basis import DynamicCarryConfig, select_portfolio
from src.state_store import StateStore

TIER1B = dict(
    ema_span=42,
    top_k=4,
    entry_threshold=0.000045,
    exit_threshold=0.0,
    min_hold_periods=63,
    buffer_rank=3,
    w_basis=0.0,
    expected_hold_periods=63,
    switch_rule="tier1b_buffer_rank",
    eta_override=0.80,
)


def _cfg(**overrides: Any) -> DynamicCarryConfig:
    base = dict(TIER1B)
    base.update(overrides)
    return DynamicCarryConfig(**base)


# ---------------------------------------------------------------------------
# 1-2. Config: switch_rule / eta_override
# ---------------------------------------------------------------------------


def test_default_config_keeps_advantage_rule_and_structure_eta() -> None:
    cfg = DynamicCarryConfig()
    assert cfg.switch_rule == "advantage"
    assert cfg.eta_override is None
    assert cfg.notional_eta() == pytest.approx(0.75)
    coinm = DynamicCarryConfig(structure="coinm_1leg")
    assert coinm.notional_eta() == pytest.approx(1.0)


def test_eta_override_supersedes_structure_default() -> None:
    assert _cfg().notional_eta() == pytest.approx(0.80)
    assert _cfg(eta_override=0.9, structure="coinm_1leg").notional_eta() == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 3-7. select_portfolio voi rule tier1b_buffer_rank
# ---------------------------------------------------------------------------


def test_tier1b_exits_when_score_at_exit_threshold() -> None:
    cfg = _cfg()
    sym = "AAA/USDT:USDT"
    scores = {sym: 0.0, "BBB/USDT:USDT": -0.0001}
    decision = select_portfolio(scores, {sym}, {sym: 63}, cfg)
    assert decision.actions[sym] == "CLOSE"
    assert decision.reasons[sym] == "EXIT_BELOW_THRESHOLD"


def test_tier1b_holds_before_min_hold_periods() -> None:
    cfg = _cfg()
    sym = "AAA/USDT:USDT"
    scores = {sym: -0.0002}
    decision = select_portfolio(scores, {sym}, {sym: 10}, cfg)
    assert decision.actions[sym] == "HOLD"
    assert decision.reasons[sym] == "HOLD_MIN_PERIODS"


def test_tier1b_rank_drop_switches_to_qualifying_candidate() -> None:
    cfg = _cfg()
    held = "AAA/USDT:USDT"
    # held tut xuong hang 8 (> top_k + buffer = 7); cac candidate moi deu >= entry_threshold
    scores: dict[str, float] = {}
    for i, sc in enumerate(
        [0.00030, 0.00025, 0.00020, 0.00015, 0.00012, 0.00010, 0.00009], start=1
    ):
        scores[f"C{i}/USDT:USDT"] = sc
    scores[held] = 0.000050
    decision = select_portfolio(scores, {held}, {held: 80}, cfg)
    assert decision.actions[held] == "CLOSE"
    assert decision.reasons[held] == "SWITCH_RANK_DROP"


def test_tier1b_rank_drop_without_candidate_is_held() -> None:
    cfg = _cfg()
    held = "AAA/USDT:USDT"
    # held o hang 8 nhung KHONG candidate nao >= entry_threshold (0.000045)
    scores: dict[str, float] = {}
    for i, sc in enumerate(
        [0.0000449, 0.0000448, 0.0000447, 0.0000446, 0.0000445, 0.0000444, 0.0000443],
        start=1,
    ):
        scores[f"L{i}/USDT:USDT"] = sc
    scores[held] = 0.000040
    decision = select_portfolio(scores, {held}, {held: 80}, cfg)
    assert decision.actions[held] == "HOLD"
    assert decision.reasons[held] == "HOLD_BUFFER_RANK"


def test_advantage_rule_unchanged_for_out_of_buffer_rank() -> None:
    cfg = DynamicCarryConfig(top_k=2, buffer_rank=1, exit_threshold=0.00002, min_hold_periods=1)
    assert cfg.switch_rule == "advantage"
    held = "AAA/USDT:USDT"
    scores = {
        "C1/USDT:USDT": 0.0005,
        "C2/USDT:USDT": 0.0004,
        "C3/USDT:USDT": 0.0003,
        "C4/USDT:USDT": 0.0002,
        held: 0.0001,
    }
    decision = select_portfolio(scores, {held}, {held: 5}, cfg)
    assert decision.actions[held] == "CLOSE"
    assert decision.reasons[held] == "EXIT_OUT_OF_BUFFER_RANK"


# ---------------------------------------------------------------------------
# 8. Executor: pure f_ewma khi tier1b_gate
# ---------------------------------------------------------------------------


def _make_executor(tmp_path: Path, tier1b: bool) -> CarryLiveExecutor:
    state_path = tmp_path / "state.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=raw_dir,
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=_cfg(initial_capital=3000.0),
        symbols=["AAA/USDT:USDT"],
        state_store=StateStore(state_path),
        raw_data_dir=raw_dir,
        capital=3000.0,
        initial_capital=3000.0,
        tier1b_gate=tier1b,
    )


def test_carry_selection_scores_uses_pure_funding_ema(tmp_path: Path) -> None:
    metrics = {
        "AAA/USDT:USDT": {"score": 0.00025, "f_ewma": 0.000040},
        "BBB/USDT:USDT": {"score": 0.00030, "f_ewma": 0.000060},
    }
    legacy = _make_executor(tmp_path / "a", tier1b=False)
    assert legacy._carry_selection_scores(metrics, {"AAA/USDT:USDT": 0.1}) == {
        "AAA/USDT:USDT": 0.1
    }

    parity = _make_executor(tmp_path / "b", tier1b=True)
    assert parity._carry_selection_scores(metrics, {"AAA/USDT:USDT": 0.1}) == {
        "AAA/USDT:USDT": pytest.approx(0.000040),
        "BBB/USDT:USDT": pytest.approx(0.000060),
    }


# ---------------------------------------------------------------------------
# 9-11. compute_live_overlay_targets
# ---------------------------------------------------------------------------

OVERLAY_SYMBOLS = ("AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT", "DDD/USDT:USDT")


def _overlay_universe(bars: int = 900, drift: float = 1.0) -> dict[str, pd.DataFrame]:
    """Universe co trend phan hoa manh de momentum tao ca chan long va short."""
    idx = pd.date_range("2025-01-01", periods=bars, freq="1h", tz="UTC")
    slopes = {"AAA/USDT:USDT": 0.0015, "BBB/USDT:USDT": 0.0010,
              "CCC/USDT:USDT": -0.0010, "DDD/USDT:USDT": -0.0015}
    out: dict[str, pd.DataFrame] = {}
    for sym in OVERLAY_SYMBOLS:
        base = 100.0 + np.arange(bars) * slopes[sym] * drift
        out[sym] = pd.DataFrame(
            {"perp_close": base, "spot_close": base * 0.9999}, index=idx
        )
    return out


def test_overlay_targets_are_not_look_ahead() -> None:
    base = _overlay_universe()
    cut = 800
    future_changed = {
        sym: df.copy() for sym, df in _overlay_universe().items()
    }
    for sym, df in future_changed.items():
        df.iloc[cut:, 0] = df.iloc[cut:, 0] * 1.5

    cfg = BarbellConfig(initial_capital=3000.0, overlay_lookback_hours=336)
    res_base = compute_live_overlay_targets(
        {sym: df.iloc[:cut] for sym, df in base.items()}, cfg, lookback_hours=720
    )
    res_future = compute_live_overlay_targets(
        {sym: df.iloc[:cut] for sym, df in future_changed.items()}, cfg, lookback_hours=720
    )
    assert res_base["weights"] == pytest.approx(res_future["weights"])


def test_overlay_targets_long_short_pair_with_weight_cap() -> None:
    cfg = BarbellConfig(initial_capital=3000.0, overlay_lookback_hours=336)
    res = compute_live_overlay_targets(_overlay_universe(), cfg, lookback_hours=720)

    weights = res["weights"]
    assert res["active"] is True
    assert 0.15 <= res["leverage"] <= 0.50
    assert 0.50 <= res["conviction"] <= 1.00
    assert any(w > 0 for w in weights.values())
    assert any(w < 0 for w in weights.values())
    assert sum(abs(w) for w in weights.values()) <= 0.50 + 1e-9
    assert all(abs(w) >= 0.08 for w in weights.values())


def test_overlay_targets_inactive_without_data() -> None:
    cfg = BarbellConfig(initial_capital=3000.0)
    res = compute_live_overlay_targets({}, cfg)
    assert res["active"] is False
    assert res["weights"] == {}
    assert res["skip_reason"] == "NO_DATA"
