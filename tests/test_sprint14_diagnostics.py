"""Test Sprint 14 Chang 2 - decision diagnostics cho carry/overlay gate va warmup history.

Bao phu:
1. Chuoi summary dung format `Warmup=... | MaxCarryEMA=... bps (thr ...) -> ... | MaxOverlayDelta=...`.
2. Warmup gate doi trang thai dung nguong 400h.
3. Doi f_ewma sang bps va so sanh nguong carry 0,45 bps (= 0.000045).
4. Dem so symbol vuot nguong carry.
5. Tinh max_overlay_delta tu target allocations vs overlay_positions (co xu ly dau short).
6. Gan `last_decision_diagnostics` va tra ve day du key.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import StateStore


def _make_executor(tmp_path: Path, capital: float = 3_000.0) -> CarryLiveExecutor:
    """Executor paper dung state/raw dir trong tmp_path (khong mang, khong parquet)."""
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
        carry_config=DynamicCarryConfig(),
        symbols=["AAA/USDT:USDT"],
        state_store=StateStore(state_path),
        raw_data_dir=raw_dir,
        capital=capital,
        initial_capital=capital,
    )


def _universe(bars: int, symbols: tuple[str, ...] = ("AAA/USDT:USDT",)) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2025-01-01", periods=bars, freq="1h", tz="UTC")
    return {
        sym: pd.DataFrame({"perp_close": np.full(bars, 100.0)}, index=idx) for sym in symbols
    }


def _metrics(mapping: dict[str, float]) -> dict[str, dict[str, Any]]:
    return {
        sym: {
            "score": 0.0,
            "f_ewma": value,
            "basis_dev": 0.0,
            "basis_pct": 0.0,
            "valid": True,
        }
        for sym, value in mapping.items()
    }


def _overlay_alloc(entries: list[tuple[str, str, float]]) -> dict[str, Any]:
    return {
        "momentum_overlay": {
            "overlay_active": True,
            "allocations": [
                {"symbol": sym, "side": side, "notional": notional}
                for sym, side, notional in entries
            ],
        }
    }


# ---------------------------------------------------------------------------
# 1. Dinh dang summary
# ---------------------------------------------------------------------------


def test_diagnostics_summary_contains_all_gate_fields(tmp_path: Path) -> None:
    ex = _make_executor(tmp_path)
    res = ex.build_decision_diagnostics(_universe(500), _metrics({"AAA/USDT:USDT": 0.00006}), None)

    summary = res["summary"]
    assert summary.startswith("Warmup=500h")
    assert "MaxCarryEMA=" in summary and "bps" in summary
    assert "MaxOverlayDelta=" in summary
    assert "_" not in summary


# ---------------------------------------------------------------------------
# 2. Warmup gate
# ---------------------------------------------------------------------------


def test_warmup_gate_blocks_under_720_hours(tmp_path: Path) -> None:
    ex = _make_executor(tmp_path)

    cold = ex.build_decision_diagnostics(_universe(719), None, None)
    warm = ex.build_decision_diagnostics(_universe(720), None, None)

    assert cold["warmup_hours_loaded"] == 719
    assert cold["warmup_min_required"] == 720
    assert cold["warmup_ok"] is False
    assert "COLD" in cold["summary"]

    assert warm["warmup_hours_loaded"] == 720
    assert warm["warmup_ok"] is True
    assert "-> OK" in warm["summary"]


def test_warmup_uses_minimum_loaded_bars_across_symbols(tmp_path: Path) -> None:
    ex = _make_executor(tmp_path)
    universe = _universe(720, symbols=("AAA/USDT:USDT", "BBB/USDT:USDT"))
    universe["BBB/USDT:USDT"] = universe["BBB/USDT:USDT"].iloc[:380]

    res = ex.build_decision_diagnostics(universe, None, None)

    assert res["warmup_hours_loaded"] == 380
    assert res["warmup_ok"] is False


# ---------------------------------------------------------------------------
# 3 & 4. Carry EMA gate
# ---------------------------------------------------------------------------


def test_carry_ema_converted_to_bps_and_threshold_from_config(tmp_path: Path) -> None:
    ex = _make_executor(tmp_path)
    # 0.00006 * 10000 = 0.6 bps > 0.45 bps
    res = ex.build_decision_diagnostics(_universe(500), _metrics({"AAA/USDT:USDT": 0.00006}), None)

    assert res["carry_ema_threshold_bps"] == pytest.approx(0.45, abs=1e-12)
    assert res["max_carry_ema_bps"] == pytest.approx(0.6, abs=1e-12)
    assert res["carry_gate_pass"] is True
    assert "PASS" in res["summary"].split("|")[1]

    # 0.00003 * 10000 = 0.3 bps < 0.45 bps -> chan
    blocked = ex.build_decision_diagnostics(
        _universe(500), _metrics({"AAA/USDT:USDT": 0.00003}), None
    )
    assert blocked["max_carry_ema_bps"] == pytest.approx(0.3, abs=1e-12)
    assert blocked["carry_gate_pass"] is False
    assert "BLOCK" in blocked["summary"].split("|")[1]


def test_symbols_above_carry_threshold_counts_only_passing_symbols(tmp_path: Path) -> None:
    ex = _make_executor(tmp_path)
    metrics = _metrics(
        {
            "AAA/USDT:USDT": 0.00006,   # 0.6 bps pass
            "BBB/USDT:USDT": -0.00005,  # 0.5 bps pass (tri tuyet doi)
            "CCC/USDT:USDT": 0.00001,   # 0.1 bps block
        }
    )

    res = ex.build_decision_diagnostics(_universe(500), metrics, None)

    assert res["symbols_above_carry_threshold"] == 2
    assert res["max_carry_ema_bps"] == pytest.approx(0.6, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. Overlay delta gate
# ---------------------------------------------------------------------------


def test_overlay_delta_from_target_vs_current_positions(tmp_path: Path) -> None:
    ex = _make_executor(tmp_path, capital=1_000.0)
    ex.state.current_equity = 1_000.0
    ex.state.overlay_positions = {
        "AAA/USDT:USDT": {"symbol": "AAA/USDT:USDT", "side": "long", "notional": 100.0}
    }

    # Target short 200 USD -> weight -0.2 vs hien tai +0.1 => delta 0.3
    res = ex.build_decision_diagnostics(
        _universe(500), None, _overlay_alloc([("AAA/USDT:USDT", "short_perp", 200.0)])
    )

    assert res["max_overlay_delta"] == pytest.approx(0.3, abs=1e-9)
    assert res["overlay_delta_threshold"] == pytest.approx(0.08)
    assert res["overlay_gate_pass"] is True


def test_overlay_delta_blocks_when_overlay_inactive(tmp_path: Path) -> None:
    ex = _make_executor(tmp_path, capital=1_000.0)
    ex.state.current_equity = 1_000.0
    inactive = {"momentum_overlay": {"overlay_active": False, "allocations": []}}

    res = ex.build_decision_diagnostics(_universe(500), None, inactive)

    assert res["max_overlay_delta"] == pytest.approx(0.0)
    assert res["overlay_gate_pass"] is False


# ---------------------------------------------------------------------------
# 6. Gan last_decision_diagnostics + du key
# ---------------------------------------------------------------------------


def test_diagnostics_returned_keys_and_cached_on_instance(tmp_path: Path) -> None:
    ex = _make_executor(tmp_path)
    expected_keys = {
        "warmup_hours_loaded",
        "warmup_min_required",
        "warmup_ok",
        "carry_ema_span",
        "max_carry_ema_bps",
        "carry_ema_threshold_bps",
        "carry_gate_pass",
        "symbols_above_carry_threshold",
        "max_overlay_delta",
        "overlay_delta_threshold",
        "overlay_gate_pass",
        "summary",
    }

    res = ex.build_decision_diagnostics(_universe(500), _metrics({"AAA/USDT:USDT": 0.0}), None)

    assert expected_keys.issubset(set(res.keys()))
    assert ex.last_decision_diagnostics == res
    assert isinstance(res["summary"], str)
    assert isinstance(res["warmup_hours_loaded"], int)
    assert isinstance(res["max_overlay_delta"], float)
