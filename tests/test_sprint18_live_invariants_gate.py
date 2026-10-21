"""Offline, deterministic tests for Sprint 18 live gates.

Covers the 720h warmup saturation gate and the runtime LOT_NEUTRALITY
invariant gate added to :mod:`src.carry_live_executor`:

- Module constants ``WARMUP_MIN_HOURS`` / ``WARMUP_BLOCK_REASON`` /
  ``WARMUP_MIN_FUNDING_PERIODS_8H``.
- ``compute_warmup_readiness`` uses the MINIMUM history length across symbols.
- ``build_barbell_allocations`` fails closed when warmup is not ready.
- ``build_decision_diagnostics`` exposes warmup / invariant fields.
- ``reconcile_once`` blocks would-be OPENs below 720h but still opens at 720h.
- ``build_cycle_audit_record`` and ``run_loop`` audit JSONL fields.

All fixtures are synthetic parquet/DataFrames written into ``tmp_path``;
there is no network access and no wall-clock dependency.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

from src.carry_live_executor import (
    WARMUP_BLOCK_REASON,
    WARMUP_MIN_FUNDING_PERIODS_8H,
    WARMUP_MIN_HOURS,
    CarryLiveExecutor,
    build_cycle_audit_record,
)
from src.config import TradingConfig
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import StateStore

_START = "2024-01-01 00:00:00"
_HIGH_FUNDING = 0.001


def _hourly_index(n_hours: int = 60, start: str = _START) -> pd.DatetimeIndex:
    """Build a fixed hourly UTC index (independent of the test run time)."""
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    return idx.as_unit("ns")


def _ohlcv_frame(index: pd.DatetimeIndex, close: float) -> pd.DataFrame:
    """Build a synthetic, flat OHLCV frame around ``close``."""
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame.astype("float64")


def _aligned_frame(
    symbol: str,
    funding_rate: float,
    n_hours: int = 60,
    spot_price: float = 100.0,
    perp_price: float = 100.05,
) -> pd.DataFrame:
    """Build an aligned spot/perp/funding frame matching align_spot_perp_funding."""
    idx = _hourly_index(n_hours)
    frame = _ohlcv_frame(idx, perp_price)
    frame["spot_close"] = float(spot_price)
    frame["perp_close"] = float(perp_price)
    frame["basis_pct"] = (float(perp_price) - float(spot_price)) / float(spot_price)

    mask_8h = idx.hour.isin([0, 8, 16]) & (idx.minute == 0)
    frame["funding_rate_event"] = 0.0
    frame.loc[mask_8h, "funding_rate_event"] = float(funding_rate)
    frame["last_known_funding_rate"] = float(funding_rate)
    frame.attrs["symbol"] = symbol
    return frame


def _write_cache_bundle(
    raw_dir: Path,
    symbol: str,
    funding_rate: float,
    n_hours: int = 60,
    spot_price: float = 100.0,
    perp_price: float = 100.05,
) -> None:
    """Write the parquet cache bundle expected by funding_loader / run_sprint7."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    idx = _hourly_index(n_hours)
    perp = _ohlcv_frame(idx, perp_price)
    spot = _ohlcv_frame(idx, spot_price)

    clean = symbol.replace("/", "_").replace(":", "_")
    spot_clean = symbol.split(":")[0].replace("/", "").replace(":", "").upper()
    if spot_clean.endswith("USDTUSDT"):
        spot_clean = spot_clean[:-4]

    funding_idx = pd.date_range(
        _START,
        periods=max(1, n_hours // 8 + 1),
        freq="8h",
        tz="UTC",
    ).as_unit("ns")
    funding = pd.DataFrame(
        {
            "timestamp": funding_idx.asi8 // 1_000_000,
            "datetime": funding_idx,
            "funding_rate": float(funding_rate),
            "mark_price": float(spot_price),
        }
    )

    perp.to_parquet(raw_dir / f"{clean}_1h.parquet", engine="pyarrow")
    spot.to_parquet(raw_dir / f"{spot_clean}_SPOT_1h.parquet", engine="pyarrow")
    funding.to_parquet(raw_dir / f"{clean}_funding.parquet", engine="pyarrow", index=False)


def _make_executor(
    tmp_path: Path,
    raw_dir: Path | None = None,
    symbols: Sequence[str] | None = None,
    carry_config: DynamicCarryConfig | None = None,
) -> CarryLiveExecutor:
    """Create an executor whose state file and raw dir live under ``tmp_path``."""
    resolved_raw = Path(raw_dir) if raw_dir is not None else tmp_path / "raw"
    resolved_raw.mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / "state.json"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=resolved_raw,
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=carry_config if carry_config is not None else DynamicCarryConfig(),
        symbols=symbols if symbols is not None else ["AAA/USDT:USDT"],
        state_store=StateStore(state_path),
        raw_data_dir=resolved_raw,
        max_allowed_spread_bps=10_000.0,
    )


def _funding_config() -> DynamicCarryConfig:
    """Config used by the funding-driven OPEN scenarios."""
    return DynamicCarryConfig(
        top_k=2,
        entry_threshold=0.00010,
        exit_threshold=0.00002,
        min_hold_periods=9,
    )


# ---------------------------------------------------------------------------
# 1. Module constants
# ---------------------------------------------------------------------------


def test_sprint18_warmup_constants_are_locked() -> None:
    """The 720h saturation gate and its block reason are part of the contract."""
    assert WARMUP_MIN_HOURS == 720
    assert WARMUP_BLOCK_REASON == "WARMUP_INSUFFICIENT_BELOW_720H"
    assert WARMUP_MIN_FUNDING_PERIODS_8H == 90


# ---------------------------------------------------------------------------
# 2. compute_warmup_readiness
# ---------------------------------------------------------------------------


def test_compute_warmup_readiness_below_threshold_is_not_ready(tmp_path: Path) -> None:
    """504 bars per symbol must be reported as cold (threshold 720h)."""
    executor = _make_executor(tmp_path, symbols=["AAA/USDT:USDT"])
    universe = {
        "AAA/USDT:USDT": _aligned_frame(
            "AAA/USDT:USDT", funding_rate=_HIGH_FUNDING, n_hours=504
        )
    }

    readiness = executor.compute_warmup_readiness(universe)

    assert readiness["warmup_ready"] is False
    assert readiness["warmup_threshold_hours"] == 720
    assert readiness["warmup_min_required"] == 720
    assert readiness["warmup_hours_loaded"] == 504
    assert readiness["min_funding_periods_8h"] == 90
    assert executor.warmup_ready is False


def test_compute_warmup_readiness_at_threshold_is_ready(tmp_path: Path) -> None:
    """Exactly 720 bars is the saturation boundary and must pass."""
    executor = _make_executor(tmp_path, symbols=["AAA/USDT:USDT"])
    universe = {
        "AAA/USDT:USDT": _aligned_frame(
            "AAA/USDT:USDT", funding_rate=_HIGH_FUNDING, n_hours=720
        )
    }

    readiness = executor.compute_warmup_readiness(universe)

    assert readiness["warmup_ready"] is True
    assert readiness["warmup_hours_loaded"] == 720
    assert executor.warmup_ready is True


def test_compute_warmup_readiness_uses_minimum_history_across_symbols(
    tmp_path: Path,
) -> None:
    """One short symbol (719h) must drag the whole universe below 720h."""
    executor = _make_executor(
        tmp_path, symbols=["AAA/USDT:USDT", "BBB/USDT:USDT"]
    )
    universe = {
        "AAA/USDT:USDT": _aligned_frame(
            "AAA/USDT:USDT", funding_rate=_HIGH_FUNDING, n_hours=720
        ),
        "BBB/USDT:USDT": _aligned_frame(
            "BBB/USDT:USDT", funding_rate=_HIGH_FUNDING, n_hours=719
        ),
    }

    readiness = executor.compute_warmup_readiness(universe)

    assert readiness["warmup_hours_loaded"] == 719
    assert readiness["warmup_ready"] is False
    assert executor.warmup_ready is False


# ---------------------------------------------------------------------------
# 3. build_barbell_allocations warmup fail-closed
# ---------------------------------------------------------------------------


def test_build_barbell_allocations_blocks_when_warmup_not_ready(tmp_path: Path) -> None:
    """Cold warmup must return empty carry and overlay sleeves with a reason."""
    executor = _make_executor(
        tmp_path,
        symbols=["AAA/USDT:USDT", "BBB/USDT:USDT"],
        carry_config=_funding_config(),
    )
    universe = {
        "AAA/USDT:USDT": _aligned_frame(
            "AAA/USDT:USDT", funding_rate=_HIGH_FUNDING, n_hours=504
        ),
        "BBB/USDT:USDT": _aligned_frame(
            "BBB/USDT:USDT", funding_rate=_HIGH_FUNDING, n_hours=504
        ),
    }

    executor.warmup_ready = False
    allocations = executor.build_barbell_allocations(universe)

    assert allocations["warmup_blocked"] is True
    assert allocations["warmup_block_reason"] == WARMUP_BLOCK_REASON
    assert allocations["carry_sleeve"]["allocations"] == []
    assert allocations["momentum_overlay"]["allocations"] == []
    assert allocations["carry_sleeve"]["total_notional"] == pytest.approx(0.0)
    assert allocations["momentum_overlay"]["total_notional"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. build_decision_diagnostics warmup fields
# ---------------------------------------------------------------------------


def test_build_decision_diagnostics_reports_warmup_state(tmp_path: Path) -> None:
    """Diagnostics expose warmup threshold/loaded hours and the summary token."""
    executor = _make_executor(
        tmp_path,
        symbols=["AAA/USDT:USDT", "BBB/USDT:USDT"],
        carry_config=_funding_config(),
    )
    cold_universe = {
        sym: _aligned_frame(sym, funding_rate=_HIGH_FUNDING, n_hours=504)
        for sym in ("AAA/USDT:USDT", "BBB/USDT:USDT")
    }
    warm_universe = {
        sym: _aligned_frame(sym, funding_rate=_HIGH_FUNDING, n_hours=720)
        for sym in ("AAA/USDT:USDT", "BBB/USDT:USDT")
    }

    cold_diag = executor.build_decision_diagnostics(cold_universe)
    assert cold_diag["warmup_min_required"] == 720
    assert cold_diag["warmup_threshold_hours"] == 720
    assert cold_diag["warmup_ready"] is False
    assert cold_diag["min_funding_periods_8h"] == 90
    assert isinstance(cold_diag["invariants_passed"], bool)
    assert "Warmup=504h" in cold_diag["summary"]

    warm_diag = executor.build_decision_diagnostics(warm_universe)
    assert warm_diag["warmup_min_required"] == 720
    assert warm_diag["warmup_ready"] is True
    assert "Warmup=720h" in warm_diag["summary"]


# ---------------------------------------------------------------------------
# 5. reconcile_once warmup gate (504h vs 720h)
# ---------------------------------------------------------------------------


def test_reconcile_once_below_720h_skips_every_would_be_open(tmp_path: Path) -> None:
    """At 504h of history no carry position may be opened."""
    symbols = ["AAA/USDT:USDT", "BBB/USDT:USDT"]
    executor = _make_executor(
        tmp_path, symbols=symbols, carry_config=_funding_config()
    )
    cold_universe = {
        sym: _aligned_frame(sym, funding_rate=_HIGH_FUNDING, n_hours=504)
        for sym in symbols
    }

    plan = executor.reconcile_once(aligned_universe=cold_universe)

    assert executor.warmup_ready is False
    assert executor.state.carry_positions == {}

    actions = {action.symbol: action.action for action in plan["actions"]}
    reasons = {action.symbol: action.reason for action in plan["actions"]}

    assert "OPEN_DELTA_NEUTRAL" not in actions.values()
    warmup_skips = [sym for sym, act in actions.items() if act == "SKIP"]
    assert warmup_skips, "expected the high-funding symbols to be gated to SKIP"
    for sym in warmup_skips:
        assert reasons[sym] == WARMUP_BLOCK_REASON

    saved = json.loads(executor.state_store.file_path.read_text(encoding="utf-8"))
    assert saved["carry_positions"] == {}


def test_reconcile_once_at_720h_still_opens(tmp_path: Path) -> None:
    """The same data with full 720h warmup must produce real OPEN actions."""
    symbols = ["AAA/USDT:USDT", "BBB/USDT:USDT"]
    executor = _make_executor(
        tmp_path, symbols=symbols, carry_config=_funding_config()
    )
    warm_universe = {
        sym: _aligned_frame(sym, funding_rate=_HIGH_FUNDING, n_hours=720)
        for sym in symbols
    }

    plan = executor.reconcile_once(aligned_universe=warm_universe)

    assert executor.warmup_ready is True
    actions = {action.symbol: action.action for action in plan["actions"]}
    opened = [sym for sym, act in actions.items() if act == "OPEN_DELTA_NEUTRAL"]
    assert opened, f"expected at least one OPEN at 720h, got {actions}"
    assert all(reason != WARMUP_BLOCK_REASON for reason in
               (action.reason for action in plan["actions"]))


# ---------------------------------------------------------------------------
# 6. build_cycle_audit_record + run_loop JSONL audit
# ---------------------------------------------------------------------------


def test_build_cycle_audit_record_contains_sprint18_fields() -> None:
    """The audit payload must carry invariant status and warmup threshold/hours."""
    record = build_cycle_audit_record(
        cycle=1,
        mode="paper",
        execution_style="maker_taker_hedge",
        capital=3000.0,
        equity=3000.0,
        peak_equity=3000.0,
        drawdown_pct=0.0,
        carry_spot_usd=0.0,
        carry_perp_usd=0.0,
        overlay_long_usd=0.0,
        overlay_short_usd=0.0,
        invariants_passed=True,
        warmup_threshold_hours=720,
        warmup_hours_loaded=504,
    )

    assert record["invariants_passed"] is True
    assert record["warmup_threshold_hours"] == 720
    assert record["warmup_hours_loaded"] == 504


def test_run_loop_writes_one_audit_jsonl_record(tmp_path: Path) -> None:
    """One run_loop cycle with an audit path writes exactly one parseable line."""
    raw_dir = tmp_path / "raw"
    symbols = ["AAA/USDT:USDT", "BBB/USDT:USDT"]
    _write_cache_bundle(raw_dir, "AAA/USDT:USDT", _HIGH_FUNDING, n_hours=720)
    _write_cache_bundle(raw_dir, "BBB/USDT:USDT", _HIGH_FUNDING, n_hours=720)

    executor = _make_executor(
        tmp_path,
        raw_dir=raw_dir,
        symbols=symbols,
        carry_config=_funding_config(),
    )
    audit_path = tmp_path / "audit" / "cycles.jsonl"
    executor.audit_log_path = str(audit_path)

    exit_code = executor.run_loop(cycles=1, interval_sec=0.0)

    assert exit_code == 0
    assert audit_path.exists()
    lines = [
        line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 1

    payload: dict[str, Any] = json.loads(lines[0])
    assert isinstance(payload.get("invariants_passed"), bool)
    assert payload["warmup_threshold_hours"] == 720
    assert payload["warmup_hours_loaded"] == 720
