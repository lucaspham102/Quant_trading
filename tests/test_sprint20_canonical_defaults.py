"""Sprint 20 - canonical defaults cho Method D (warm cash start).

Bao phu:
1. Hang so canonical/legacy: 720h / 504h / alias 504h.
2. Chu ky ham `run_indicator_warm_cash_start` (mac dinh 720h, clock aligned, alias None).
3. Lan goi mac dinh -> warmup 720h, aligned True, trade_start_idx dung 720h.
4. Lan goi mac dinh -> rebalance_phase_offset = global_start_idx % 48.
5. Lan goi legacy tuong minh (504h, khong align) -> offset 0 va trade_start_idx khac.
6. Alias `phase_aligned=False` ghi de `align_rebalance_clock=True`.
7. Artifact canonical `data/forensics_sprint17/bridge_sprint17.json` khop so lieu.
8. Dang thuc ke toan tren lan chay mac dinh (equity - start == net pnl).

Tat ca du lieu la tong hop, khong doc `data/raw`, khong dung mang.
"""

from __future__ import annotations

import inspect
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sprint11_eval import (
    CANONICAL_SATURATED_WARMUP_HOURS,
    LEGACY_SPRINT14_WARMUP_HOURS,
    SPRINT14_WARMUP_HOURS,
    run_indicator_warm_cash_start,
)

_SPLIT_DATE = "2024-02-01"
_HOLDOUT_END = "2024-02-20"
_CAPITAL = 3000.0


# ---------------------------------------------------------------------------
# Synthetic data (giong tests/test_sprint15_clock_alignment.py)
# ---------------------------------------------------------------------------


def _synthetic_data(bars: int = 900) -> dict[str, pd.DataFrame]:
    """Du lieu tong hop 3 symbol du de chay Method D (khong mang, khong parquet)."""
    idx = pd.date_range("2024-01-01", periods=bars, freq="1h", tz="UTC")
    slopes = {"AAA/USDT:USDT": 0.0012, "BBB/USDT:USDT": -0.0008, "CCC/USDT:USDT": 0.0004}
    out: dict[str, pd.DataFrame] = {}
    for sym, slope in slopes.items():
        close = 100.0 + np.arange(bars) * slope
        out[sym] = pd.DataFrame(
            {
                "perp_close": close,
                "spot_close": close * 0.9998,
                "close": close,
                "funding_rate_event": np.full(bars, 0.00008),
            },
            index=idx,
        )
    return out


@lru_cache(maxsize=8)
def _run_default_cached() -> dict:
    """Lan chay canonical (720h + align) - cache de giu test nhanh va doc lap du lieu."""
    return run_indicator_warm_cash_start(
        data=_synthetic_data(),
        capital=_CAPITAL,
        split_date=_SPLIT_DATE,
        holdout_end=_HOLDOUT_END,
    )


# ---------------------------------------------------------------------------
# 1. Hang so canonical / legacy
# ---------------------------------------------------------------------------


def test_canonical_and_legacy_warmup_constants() -> None:
    """720h la chuan bao hoa; 504h chi la legacy va alias phai tro ve 504."""
    assert CANONICAL_SATURATED_WARMUP_HOURS == 720
    assert LEGACY_SPRINT14_WARMUP_HOURS == 504
    assert SPRINT14_WARMUP_HOURS == 504
    assert SPRINT14_WARMUP_HOURS == LEGACY_SPRINT14_WARMUP_HOURS


# ---------------------------------------------------------------------------
# 2. Chu ky ham
# ---------------------------------------------------------------------------


def test_signature_defaults_are_canonical() -> None:
    """Mac dinh moi phai la 720h, bat align clock va alias phase_aligned=None."""
    params = inspect.signature(run_indicator_warm_cash_start).parameters
    assert params["warmup_hours"].default == 720
    assert params["align_rebalance_clock"].default is True
    assert params["phase_aligned"].default is None


# ---------------------------------------------------------------------------
# 3. Lan goi mac dinh: warmup, aligned va trade_start_idx
# ---------------------------------------------------------------------------


def test_default_call_uses_720h_and_starts_trading_at_split() -> None:
    """Mac dinh phai cat 720h du lieu lich su va bat dau giao dich dung tai split."""
    data = _synthetic_data()
    idx = data["AAA/USDT:USDT"].index
    split_ts = pd.Timestamp(_SPLIT_DATE, tz="UTC")
    ext_start = split_ts - pd.Timedelta(hours=CANONICAL_SATURATED_WARMUP_HOURS)
    start_pos = int(idx.searchsorted(ext_start, side="left"))
    split_pos = int(idx.searchsorted(split_ts, side="left"))
    expected_trade_start_idx = split_pos - start_pos

    res = _run_default_cached()

    assert res["warmup_hours"] == 720
    assert res["align_rebalance_clock"] is True
    assert res["phase_aligned"] is True
    assert res["trade_start_idx"] == expected_trade_start_idx
    assert res["trade_start_idx"] == CANONICAL_SATURATED_WARMUP_HOURS


# ---------------------------------------------------------------------------
# 4. Offset dong ho 48h cua lan goi mac dinh
# ---------------------------------------------------------------------------


def test_default_call_phase_offset_matches_global_start_index() -> None:
    """Aligned offset = global_start_idx % 48 tren truc thoi gian tuyet doi."""
    data = _synthetic_data()
    idx = data["AAA/USDT:USDT"].index
    split_ts = pd.Timestamp(_SPLIT_DATE, tz="UTC")
    ext_start = split_ts - pd.Timedelta(hours=CANONICAL_SATURATED_WARMUP_HOURS)
    global_start_idx = int(idx.searchsorted(ext_start, side="left"))
    expected_offset = global_start_idx % 48

    res = _run_default_cached()

    assert expected_offset == 24
    assert res["rebalance_phase_offset"] == expected_offset


# ---------------------------------------------------------------------------
# 5. Lan goi legacy tuong minh (504h, khong align)
# ---------------------------------------------------------------------------


def test_explicit_legacy_call_is_not_aligned_and_differs() -> None:
    """504h + align=False phai giu nguyen hanh vi lich su, offset 0."""
    legacy = run_indicator_warm_cash_start(
        data=_synthetic_data(),
        capital=_CAPITAL,
        split_date=_SPLIT_DATE,
        holdout_end=_HOLDOUT_END,
        warmup_hours=LEGACY_SPRINT14_WARMUP_HOURS,
        align_rebalance_clock=False,
    )
    default = _run_default_cached()

    assert legacy["warmup_hours"] == 504
    assert legacy["align_rebalance_clock"] is False
    assert legacy["phase_aligned"] is False
    assert legacy["rebalance_phase_offset"] == 0
    assert legacy["trade_start_idx"] == 504
    assert legacy["trade_start_idx"] != default["trade_start_idx"]


# ---------------------------------------------------------------------------
# 6. Alias phase_aligned ghi de align_rebalance_clock
# ---------------------------------------------------------------------------


def test_phase_aligned_alias_overrides_align_flag() -> None:
    """phase_aligned=False phai thang align_rebalance_clock=True."""
    res = run_indicator_warm_cash_start(
        data=_synthetic_data(),
        capital=_CAPITAL,
        split_date=_SPLIT_DATE,
        holdout_end=_HOLDOUT_END,
        align_rebalance_clock=True,
        phase_aligned=False,
    )

    assert res["warmup_hours"] == 720
    assert res["align_rebalance_clock"] is False
    assert res["phase_aligned"] is False
    assert res["rebalance_phase_offset"] == 0


# ---------------------------------------------------------------------------
# 7. Artifact canonical Sprint 17
# ---------------------------------------------------------------------------


def _artifact() -> dict:
    path = Path(__file__).resolve().parents[1] / "data" / "forensics_sprint17" / "bridge_sprint17.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_canonical_artifact_matches_expected_numbers() -> None:
    """So lieu artifact phai khop canonical E (720h) va legacy D (504h)."""
    bridge = _artifact()

    e_res = bridge["e_true_fixed_warm_cash"]
    assert e_res["net_pnl_usd"] == pytest.approx(114.27, abs=0.01)
    assert e_res["warmup_hours"] == 720
    assert e_res["rebalance_phase_offset"] == 46

    d_res = bridge["indicator_warm_cash_start"]
    assert d_res["net_pnl_usd"] == pytest.approx(81.73, abs=0.01)
    assert d_res["warmup_hours"] == 504

    assert bridge["clock_phase_bridge"]["identity_gap_usd"] == pytest.approx(0.0, abs=1e-9)
    assert bridge["warm_cash_decomposition"]["unexplained_total_usd"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. Dang thuc ke toan tren lan chay mac dinh
# ---------------------------------------------------------------------------


def test_default_run_equity_identity_holds() -> None:
    """end - start phai bang dung net_pnl va start phai bang capital."""
    res = _run_default_cached()

    assert res["start_equity_usd"] == pytest.approx(_CAPITAL, abs=1e-9)
    assert res["end_equity_usd"] - res["start_equity_usd"] == pytest.approx(
        res["net_pnl_usd"], abs=1e-9
    )
    assert res["entry_fee_usd"] >= 0.0
