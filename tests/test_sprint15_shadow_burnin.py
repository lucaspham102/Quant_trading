"""Test Sprint 15 Chang 3 - shadow-live burn-in (offline, khong goi mang).

Bao phu:
1. `tier1b_carry_config` khop dung bo tham so Frozen Alpha Sprint 11.
2. `check_feed` PASS voi fixture hop le va FAIL khi thieu symbol.
3. `check_feed` bat spread qua rong va book crossed.
4. `check_state` bat hedge mismatch va so du am.
5. `check_state` bat kill-switch va drawdown vuot 6%.
6. `run_shadow_burnin.main` chay 3 chu ky offline fixture -> 3/3 PASS, exit 0.
7. `--strict` tra exit 1 khi co chu ky FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_shadow_burnin as burnin

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "shadow_fixture_example.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Cau hinh Tier 1B
# ---------------------------------------------------------------------------


def test_tier1b_carry_config_matches_frozen_alpha() -> None:
    cfg = burnin.tier1b_carry_config(3000.0)
    assert cfg.ema_span == 42
    assert cfg.top_k == 4
    assert cfg.entry_threshold == pytest.approx(0.000045)
    assert cfg.exit_threshold == pytest.approx(0.0)
    assert cfg.min_hold_periods == 63
    assert cfg.buffer_rank == 3
    assert cfg.w_basis == pytest.approx(0.0)
    assert cfg.expected_hold_periods == 63
    assert cfg.switch_rule == "tier1b_buffer_rank"
    assert cfg.notional_eta() == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# 2-3. check_feed
# ---------------------------------------------------------------------------


def test_check_feed_passes_with_valid_fixture() -> None:
    fx = _fixture()
    snap = burnin.fetch_binance_shadow_snapshot(list(fx.keys()), fixture_data=fx)
    assert burnin.check_feed(snap, sorted(fx.keys())) == []


def test_check_feed_flags_missing_symbols() -> None:
    fx = _fixture()
    symbols = list(fx.keys())
    snap = burnin.fetch_binance_shadow_snapshot(symbols, fixture_data=fx)
    problems = burnin.check_feed(snap, symbols + ["XXX/USDT:USDT"])
    assert any(p.startswith("FEED_MISSING_SYMBOLS") for p in problems)


def test_check_feed_flags_wide_spread_and_crossed_book() -> None:
    bad = {
        "AAA/USDT:USDT": {"bid": 100.0, "ask": 101.0},   # ~9950 bps
        "BBB/USDT:USDT": {"bid": 100.0, "ask": 99.0},    # crossed
    }
    snap = burnin.fetch_binance_shadow_snapshot(list(bad.keys()), fixture_data=bad)
    problems = burnin.check_feed(snap, list(bad.keys()))
    assert any(p.startswith("SPREAD_TOO_WIDE") for p in problems)
    assert any(p.startswith("BOOK_CROSSED") or p.startswith("FEED_MISSING_SYMBOLS") for p in problems)


# ---------------------------------------------------------------------------
# 4-5. check_state
# ---------------------------------------------------------------------------


def _executor(tmp_path: Path):
    return burnin.build_burnin_executor(
        capital=3000.0,
        state_path=tmp_path / "state.json",
        raw_dir=tmp_path / "raw",
    )


def test_check_state_flags_hedge_mismatch_and_negative_balance(tmp_path: Path) -> None:
    ex = _executor(tmp_path)
    ex.state.carry_positions["BTC/USDT:USDT"] = {
        "qty": 0.01,
        "leg1_status": "FILLED",
        "leg1_filled_qty": 0.01,
        "leg2_hedged_qty": 0.005,
        "notional_usdt": 500.0,
        "margin_allocated_usdt": -1.0,
    }
    problems = burnin.check_state(ex, 3000.0)
    assert any(p.startswith("HEDGE_MISMATCH") for p in problems)
    assert any(p.startswith("NEGATIVE_BALANCE") for p in problems)


def test_check_state_flags_kill_switch_and_drawdown(tmp_path: Path) -> None:
    ex = _executor(tmp_path)
    ex.state.current_equity = 2800.0
    ex.state.peak_equity = 3000.0
    ex.state.kill_switch_triggered = True
    problems = burnin.check_state(ex, 3000.0)
    assert "KILL_SWITCH_TRIGGERED" in problems
    assert any(p.startswith("DRAWDOWN_EXCEEDED") for p in problems)


# ---------------------------------------------------------------------------
# 6-7. CLI end-to-end offline
# ---------------------------------------------------------------------------


def test_burnin_offline_three_cycles_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    code = burnin.main(
        [
            "--cycles", "3",
            "--interval-sec", "0",
            "--offline-fixture",
            "--strict",
            "--state-file", str(state_path),
            "--report-out", str(report_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "SUMMARY: 3/3 PASS" in out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] == 3
    assert payload["failed"] == 0
    for record in payload["records"]:
        assert record["status"] == "PASS"
        assert record["feed_source"] == "fixture"


def test_burnin_strict_returns_nonzero_when_cycle_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken_state(executor, capital):  # noqa: ANN001
        return ["KILL_SWITCH_TRIGGERED"]

    monkeypatch.setattr(burnin, "check_state", _broken_state)
    code = burnin.main(
        [
            "--cycles", "1",
            "--interval-sec", "0",
            "--offline-fixture",
            "--strict",
            "--state-file", str(tmp_path / "state.json"),
        ]
    )
    assert code == 1
