"""Module danh gia va kiem dinh chat luong Sprint 11 (Single-Venue Tier 1+ Barbell).

Cung cap:
- SPRINT11_SPLIT_DATE, SPRINT11_HOLDOUT_END, SPRINT11_INITIAL_CAPITAL, SPRINT11_GO_CRITERIA
- compute_beta_to_benchmark: Tinh beta cua loi nhuan so voi benchmark (ddof=1).
- split_period_returns: Chia chuoi returns thanh n doan va tinh return_pct moi doan.
- run_sprint11_tiers: Mo phong luoi 4 combo (tier1a/tier1b x hard_binary/smooth_conviction).
- smoke_test_executor: Kiem tra cau truc va runtime paper executor tren stub exchange.
- evaluate_sprint11_go_no_go: Kiem dinh 10 tieu chi GO/NO-GO theo docs/go_no_go_sprint11.md.
- build_sprint11_summary_rows: Tao bang phang cac dong tong hop chi so xuat markdown/csv.
"""
from __future__ import annotations

import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.carry_live_executor import CarryLiveExecutor
from src.cross_exchange_carry import LIQUID_8_SYMBOLS, run_binance_tier1_plus_carry
from src.state_store import BotState, StateStore

logger = logging.getLogger("sprint11_eval")

# ---------------------------------------------------------------------------
# 1. Hang so Sprint 11
# ---------------------------------------------------------------------------
SPRINT11_SPLIT_DATE: str = "2026-04-21"
SPRINT11_HOLDOUT_END: str = "2026-10-18"
SPRINT11_INITIAL_CAPITAL: float = 3000.0
SPRINT11_GO_CRITERIA: dict[str, float | int] = {
    "annualized_return_pct": 9.0,
    "min_positive_periods": 6,
    "holdout_return_pct": 3.5,
    "sharpe": 1.80,
    "maxdd_pct": 7.5,
    "abs_beta_btc": 0.15,
}

__all__ = [
    "SPRINT11_SPLIT_DATE",
    "SPRINT11_HOLDOUT_END",
    "SPRINT11_INITIAL_CAPITAL",
    "SPRINT11_GO_CRITERIA",
    "compute_beta_to_benchmark",
    "split_period_returns",
    "run_sprint11_tiers",
    "smoke_test_executor",
    "evaluate_sprint11_go_no_go",
    "build_sprint11_summary_rows",
    "StubExchange",
]


# ---------------------------------------------------------------------------
# 2. compute_beta_to_benchmark
# ---------------------------------------------------------------------------
def compute_beta_to_benchmark(returns: pd.Series, benchmark: pd.Series) -> float:
    """Tinh he so beta cua chuoi loi nhuan so voi benchmark bang OLS don (ddof=1).

    Can index, dropna. Neu var(benchmark) <= 1e-18 hoac khong hop le thi tra ve 0.0.
    """
    if returns is None or benchmark is None:
        return 0.0
    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        return 0.0
    if returns.empty or benchmark.empty:
        return 0.0

    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0

    r = aligned.iloc[:, 0].to_numpy(dtype=np.float64)
    b = aligned.iloc[:, 1].to_numpy(dtype=np.float64)

    var_b = float(np.var(b, ddof=1))
    if var_b <= 1e-18 or not np.isfinite(var_b):
        return 0.0

    cov_mat = np.cov(r, b, ddof=1)
    cov_rb = float(cov_mat[0, 1])
    if not np.isfinite(cov_rb):
        return 0.0

    return float(cov_rb / var_b)


# ---------------------------------------------------------------------------
# 3. split_period_returns
# ---------------------------------------------------------------------------
def split_period_returns(returns: pd.Series, n_periods: int = 7) -> list[dict[str, Any]]:
    """Chia chuoi loi nhuan thanh n_periods doan bang nhau va tinh return_pct moi doan.

    Tra ve list cac dict gom: period, start, end, return_pct.
    """
    if returns is None or not isinstance(returns, pd.Series) or returns.empty:
        return []

    n = max(1, int(n_periods))
    splits = np.array_split(np.arange(len(returns)), n)
    out: list[dict[str, Any]] = []

    for i, idx_arr in enumerate(splits):
        if len(idx_arr) == 0:
            continue
        sub = returns.iloc[idx_arr]
        start_ts = sub.index[0]
        end_ts = sub.index[-1]
        start_iso = start_ts.isoformat() if hasattr(start_ts, "isoformat") else str(start_ts)
        end_iso = end_ts.isoformat() if hasattr(end_ts, "isoformat") else str(end_ts)
        r_arr = sub.to_numpy(dtype=np.float64)
        ret_pct = float((np.prod(1.0 + r_arr) - 1.0) * 100.0)
        out.append({
            "period": i + 1,
            "start": start_iso,
            "end": end_iso,
            "return_pct": ret_pct,
        })

    return out


# ---------------------------------------------------------------------------
# Helper cat dataset theo range
# ---------------------------------------------------------------------------
def _slice_data_range(
    data: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Cat dict DataFrames theo khoang thoi gian [start_date, end_date] an toan timezone."""
    sliced: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        if df is None or df.empty:
            sliced[sym] = df
            continue
        if isinstance(df.index, pd.DatetimeIndex):
            tz = df.index.tz
            s_ts = pd.Timestamp(start_date)
            e_ts = pd.Timestamp(end_date)
            if tz is not None:
                if s_ts.tzinfo is None:
                    s_ts = s_ts.tz_localize(tz)
                else:
                    s_ts = s_ts.tz_convert(tz)
                if e_ts.tzinfo is None:
                    e_ts = e_ts.tz_localize(tz)
                else:
                    e_ts = e_ts.tz_convert(tz)
            e_ts_end = e_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            sub_df = df.loc[(df.index >= s_ts) & (df.index <= e_ts_end)]
            sliced[sym] = sub_df
        else:
            sliced[sym] = df
    return sliced


# ---------------------------------------------------------------------------
# 4b. run_indicator_warm_cash_start (Method D, Sprint 14)
# ---------------------------------------------------------------------------
# Sprint 20: nang chuan bao hoa warmup len 720h (~30 ngay) va neo dong ho tai can
# bang 48h theo moc UTC tuyet doi. Day la "canonical" dung cho moi lan chay moi.
CANONICAL_SATURATED_WARMUP_HOURS: int = 720
# Gia tri lich su Sprint 14 (504h, pha cat lat) chi con dung de DOI CHIEU bang
# so sanh A/B/C/D/D'/E - phai truyen TUONG MINH, khong con la mac dinh.
LEGACY_SPRINT14_WARMUP_HOURS: int = 504
# Alias tuong thich nguoc (cac script/test cu import ten nay).
SPRINT14_WARMUP_HOURS = LEGACY_SPRINT14_WARMUP_HOURS


def compute_sleeve_attribution(
    attribution_series: Any,
    holdings_log: Any,
    split_date: str = SPRINT11_SPLIT_DATE,
    holdout_end: str = SPRINT11_HOLDOUT_END,
    n_carry_slots: int = 4,
    first_window_hours: int = 720,
) -> dict[str, Any]:
    """Sprint 16: phan ra PnL holdout theo nhanh (Carry/Overlay) va cua so 30d/151d.

    Nguon so lieu: ``attribution_series`` cua ``run_barbell_portfolio`` voi cac cot
    ``funding_pnl``, ``basis_pnl``, ``overlay_pnl``, ``carry_fees``, ``overlay_fees``, ``net_pnl``.

    Dang thuc bat buoc:
    - ``carry_net_pnl_usd + overlay_net_pnl_usd == net_pnl_usd``
    - ``first_30d_net_pnl_usd + last_151d_net_pnl_usd == net_pnl_usd``
    """
    out: dict[str, Any] = {
        "available": False,
        "carry_net_pnl_usd": 0.0,
        "overlay_net_pnl_usd": 0.0,
        "carry_fees_usd": 0.0,
        "overlay_fees_usd": 0.0,
        "net_pnl_usd": 0.0,
        "avg_carry_slots_held": 0.0,
        "carry_utilization_pct": 0.0,
        "first_30d_net_pnl_usd": 0.0,
        "last_151d_net_pnl_usd": 0.0,
        "holdout_hours": 0,
        "first_window_hours": int(first_window_hours),
        "n_carry_slots": int(n_carry_slots),
    }
    if not isinstance(attribution_series, pd.DataFrame) or attribution_series.empty:
        return out
    if not isinstance(attribution_series.index, pd.DatetimeIndex):
        return out

    start_ts = pd.Timestamp(split_date, tz="UTC")
    end_ts = pd.Timestamp(holdout_end, tz="UTC")
    idx = attribution_series.index
    win = attribution_series.loc[(idx >= start_ts) & (idx <= end_ts)]
    if win.empty:
        return out

    def _col(frame: pd.DataFrame, name: str) -> pd.Series:
        if name in frame.columns:
            return frame[name].astype("float64").fillna(0.0)
        return pd.Series(0.0, index=frame.index, dtype="float64")

    funding = _col(win, "funding_pnl")
    basis = _col(win, "basis_pnl")
    overlay_net = _col(win, "overlay_pnl")
    carry_fees = _col(win, "carry_fees")
    overlay_fees = _col(win, "overlay_fees")
    if "net_pnl" in win.columns:
        net = _col(win, "net_pnl")
    else:
        net = funding + basis + overlay_net

    carry_net = funding + basis - carry_fees
    out.update(
        {
            "available": True,
            "carry_net_pnl_usd": float(carry_net.sum()),
            "overlay_net_pnl_usd": float(overlay_net.sum()),
            "carry_fees_usd": float(carry_fees.sum()),
            "overlay_fees_usd": float(overlay_fees.sum()),
            "net_pnl_usd": float(net.sum()),
            "holdout_hours": int(len(win)),
            "first_30d_net_pnl_usd": float(net.iloc[: int(first_window_hours)].sum()),
            "last_151d_net_pnl_usd": float(net.iloc[int(first_window_hours):].sum()),
        }
    )

    # Ty le giai ngan: trung binh so slot carry dang giu tren n_carry_slots.
    if isinstance(holdings_log, pd.DataFrame) and not holdings_log.empty and "n_active" in holdings_log.columns:
        log = holdings_log.copy()
        ts_col = "time" if "time" in log.columns else None
        if ts_col is not None:
            ts = pd.to_datetime(log[ts_col], utc=True, errors="coerce")
            in_win = (ts >= start_ts) & (ts <= end_ts)
            sub = log.loc[in_win.fillna(False)]
        else:
            sub = log
        if not sub.empty:
            avg_slots = float(pd.to_numeric(sub["n_active"], errors="coerce").fillna(0.0).mean())
            out["avg_carry_slots_held"] = avg_slots
            out["carry_utilization_pct"] = (
                avg_slots / float(n_carry_slots) * 100.0 if n_carry_slots > 0 else 0.0
            )
    return out


def decompose_overlay_pnl_divergence(
    weights_ref: pd.DataFrame,
    weights_alt: pd.DataFrame,
    returns: pd.DataFrame,
    notional_ref: pd.Series | None = None,
    notional_alt: pd.Series | None = None,
    fees_ref: pd.Series | None = None,
    fees_alt: pd.Series | None = None,
    funding: pd.DataFrame | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tolerance_weight: float = 1e-6,
    lag_weights: bool = True,
) -> dict[str, Any]:
    """Sprint 17: phan ra chinh xac chenh lech PnL gross cua nhanh overlay.

    Dang thuc chinh xac (per gio)::

        g_alt - g_ref = (A_alt - A_ref) * n_ref + A_alt * (n_alt - n_ref)

    voi ``A_t = sum_sym w[t, sym] * (ret[t, sym] - funding[t, sym])``.
    So hang 1 = ``weight_effect_usd`` (khac trong so muc tieu / trang thai giu vi the),
    so hang 2 = ``capital_effect_usd`` (khac von co so nhan PnL). Phan phi tach rieng
    qua ``fee_effect_usd``. Tong 3 hieu ung phai bang dung chenh lech net PnL.
    """
    empty: dict[str, Any] = {
        "available": False,
        "hours": 0,
        "gross_ref_usd": 0.0,
        "gross_alt_usd": 0.0,
        "net_ref_usd": 0.0,
        "net_alt_usd": 0.0,
        "gross_gap_usd": 0.0,
        "net_gap_usd": 0.0,
        "weight_effect_usd": 0.0,
        "capital_effect_usd": 0.0,
        "fee_effect_usd": 0.0,
        "residual_usd": 0.0,
        "weights_equal_ratio": 0.0,
        "rebal_hours_ref": 0,
        "rebal_hours_alt": 0,
        "rebal_timing_mismatch_hours": 0,
        "rebal_timing_match_ratio": 0.0,
        "notional_ref_mean": 0.0,
        "notional_alt_mean": 0.0,
        "weight_effect_by_symbol_usd": {},
    }
    if not isinstance(weights_ref, pd.DataFrame) or weights_ref.empty:
        return empty
    if not isinstance(weights_alt, pd.DataFrame) or weights_alt.empty:
        return empty
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        return empty

    idx = weights_ref.index
    if isinstance(idx, pd.DatetimeIndex):
        mask = pd.Series(True, index=idx)
        if start is not None:
            mask &= idx >= pd.Timestamp(start, tz="UTC")
        if end is not None:
            mask &= idx <= pd.Timestamp(end, tz="UTC")
        sel = idx[mask.to_numpy()]
    else:  # pragma: no cover - defensive
        sel = idx
    if len(sel) == 0:
        return empty

    cols = [c for c in weights_ref.columns if c in weights_alt.columns and c in returns.columns]
    if not cols:
        return empty

    w_ref = weights_ref.loc[sel, cols].fillna(0.0).astype(float)
    w_alt = weights_alt.reindex(sel)[cols].fillna(0.0).astype(float)
    ret = returns.reindex(sel)[cols].fillna(0.0).astype(float)
    if isinstance(funding, pd.DataFrame) and not funding.empty:
        ret_eff = ret - funding.reindex(sel)[cols].fillna(0.0).astype(float)
    else:
        ret_eff = ret

    n_ref = (
        notional_ref.reindex(sel).fillna(0.0).astype(float)
        if isinstance(notional_ref, pd.Series)
        else pd.Series(0.0, index=sel, dtype=float)
    )
    n_alt = (
        notional_alt.reindex(sel).fillna(0.0).astype(float)
        if isinstance(notional_alt, pd.Series)
        else pd.Series(0.0, index=sel, dtype=float)
    )
    f_ref = (
        fees_ref.reindex(sel).fillna(0.0).astype(float)
        if isinstance(fees_ref, pd.Series)
        else pd.Series(0.0, index=sel, dtype=float)
    )
    f_alt = (
        fees_alt.reindex(sel).fillna(0.0).astype(float)
        if isinstance(fees_alt, pd.Series)
        else pd.Series(0.0, index=sel, dtype=float)
    )

    a_ref = (w_ref * ret_eff).sum(axis=1)
    a_alt = (w_alt * ret_eff).sum(axis=1)
    gross_ref = float((a_ref * n_ref).sum())
    gross_alt = float((a_alt * n_alt).sum())
    weight_effect_series = (a_alt - a_ref) * n_ref
    capital_effect_series = a_alt * (n_alt - n_ref)
    fee_effect = float(f_ref.sum() - f_alt.sum())

    weights_equal = (w_ref - w_alt).abs().max(axis=1) <= tolerance_weight
    w_diff = (w_alt - w_ref).abs().mul(ret_eff.abs(), axis=0).mul(n_ref, axis=0)
    by_symbol = {str(c): float(w_diff[c].sum()) for c in cols}

    rebal_ref = w_ref.diff().abs().max(axis=1) > tolerance_weight
    rebal_alt = w_alt.diff().abs().max(axis=1) > tolerance_weight
    union = int((rebal_ref | rebal_alt).sum())
    mismatch = int((rebal_ref != rebal_alt).sum())

    if lag_weights and len(w_ref) > 1:
        # Trong sim, PnL cua gio t dung vi the dang giu (dat tai tai can bang truoc),
        # tuc weights[t-1]. Bo bar dau de dang thuc khep kin tuyet doi.
        w_ref = w_ref.shift(1).iloc[1:]
        w_alt = w_alt.shift(1).iloc[1:]
        ret_eff = ret_eff.iloc[1:]
        n_ref = n_ref.iloc[1:]
        n_alt = n_alt.iloc[1:]
        f_ref = f_ref.iloc[1:]
        f_alt = f_alt.iloc[1:]
        a_ref = (w_ref * ret_eff).sum(axis=1)
        a_alt = (w_alt * ret_eff).sum(axis=1)
        gross_ref = float((a_ref * n_ref).sum())
        gross_alt = float((a_alt * n_alt).sum())
        weight_effect_series = (a_alt - a_ref) * n_ref
        capital_effect_series = a_alt * (n_alt - n_ref)
        fee_effect = float(f_ref.sum() - f_alt.sum())
        weights_equal = (w_ref - w_alt).abs().max(axis=1) <= tolerance_weight
        w_diff = (w_alt - w_ref).abs().mul(ret_eff.abs(), axis=0).mul(n_ref, axis=0)
        by_symbol = {str(c): float(w_diff[c].sum()) for c in cols}

    net_ref = gross_ref - float(f_ref.sum())
    net_alt = gross_alt - float(f_alt.sum())
    return {
        "available": True,
        "hours": int(len(sel)),
        "gross_ref_usd": gross_ref,
        "gross_alt_usd": gross_alt,
        "net_ref_usd": net_ref,
        "net_alt_usd": net_alt,
        "gross_gap_usd": gross_alt - gross_ref,
        "net_gap_usd": net_alt - net_ref,
        "weight_effect_usd": float(weight_effect_series.sum()),
        "capital_effect_usd": float(capital_effect_series.sum()),
        "fee_effect_usd": fee_effect,
        "residual_usd": float(
            (net_alt - net_ref)
            - (float(weight_effect_series.sum()) + float(capital_effect_series.sum()) + fee_effect)
        ),
        "weights_equal_ratio": float(weights_equal.mean()),
        "rebal_hours_ref": int(rebal_ref.sum()),
        "rebal_hours_alt": int(rebal_alt.sum()),
        "rebal_timing_mismatch_hours": mismatch,
        "rebal_timing_match_ratio": (1.0 - mismatch / union) if union > 0 else 1.0,
        "notional_ref_mean": float(n_ref.mean()),
        "notional_alt_mean": float(n_alt.mean()),
        "weight_effect_by_symbol_usd": by_symbol,
    }


def run_indicator_warm_cash_start(
    data: dict[str, pd.DataFrame],
    capital: float = SPRINT11_INITIAL_CAPITAL,
    split_date: str = SPRINT11_SPLIT_DATE,
    holdout_end: str = SPRINT11_HOLDOUT_END,
    reinvest_profits: bool = False,
    carry_variant: str = "tier1b",
    gating_mode: str = "smooth_conviction",
    warmup_hours: int = CANONICAL_SATURATED_WARMUP_HOURS,
    align_rebalance_clock: bool = True,
    phase_aligned: bool | None = None,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Method D: chỉ báo đã ấm (nhìn được lịch sử trước holdout) nhưng tồn kho bằng tiền mặt.

    Bot thật khi bật lần đầu sẽ tải ~500 giờ nến/funding gần nhất để tính EMA-42 và
    momentum/vol 336h, nhưng tài khoản bắt đầu bằng 100% tiền mặt tại ``split_date``.
    Hàm này mô phỏng đúng trạng thái đó: dữ liệu mở rộng thêm ``warmup_hours`` giờ
    TRƯỚC ``split_date`` (để chỉ báo có đủ lịch sử) nhưng khoá mọi giao dịch trước
    ``split_date`` (``trade_start_idx``), nên equity giữ nguyên ``capital`` cho tới mốc đó
    và mọi vị thế mở sau mốc đều trả đủ phí + làm tròn lô.

    Sprint 20 (canonical defaults): ``warmup_hours=720`` (bão hoà chỉ báo) và
    ``align_rebalance_clock=True`` (neo pha đồng hồ 48h theo UTC tuyệt đối).
    Các phép đo LỊCH SỬ (D = 504h cắt lát, D' = 504h aligned) phải truyền tường minh
    ``warmup_hours=LEGACY_SPRINT14_WARMUP_HOURS``. ``phase_aligned`` (Sprint 15) được
    giữ làm alias ghi đè để tương thích ngược; nếu ``None`` thì dùng
    ``align_rebalance_clock``.
    """
    from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio

    aligned = bool(
        align_rebalance_clock if phase_aligned is None else bool(phase_aligned)
    )
    holdout_start = pd.Timestamp(split_date, tz="UTC")
    ext_start = holdout_start - pd.Timedelta(hours=int(warmup_hours))
    ext_data = _slice_data_range(data, ext_start.isoformat(), holdout_end)

    idx_ref = None
    for df in ext_data.values():
        if isinstance(df, pd.DataFrame) and not df.empty and isinstance(df.index, pd.DatetimeIndex):
            idx_ref = df.index
            break
    trade_start_idx = 0
    if idx_ref is not None and len(idx_ref) > 0:
        trade_start_idx = int(idx_ref.searchsorted(holdout_start, side="left"))

    cfg = BarbellConfig(
        initial_capital=float(capital),
        carry_variant=carry_variant,
        gating_mode=gating_mode,
        enable_snr_gate=True,
        overlay_quantize_legs=True,
        reinvest_profits=bool(reinvest_profits),
    )
    # Sprint 15: đồng hồ tái cân bằng overlay là mốc tuyệt đối 48h neo từ nến đầu chuỗi.
    # Method D cắt chuỗi tại split + warmup nên lệch pha so với backtest đầy đủ (B/C);
    # align_rebalance_clock=True neo lại offset theo vị trí tuyệt đối của split_date.
    phase_offset = 0
    if aligned:
        idx_full = None
        for df in data.values():
            if isinstance(df, pd.DataFrame) and not df.empty and isinstance(df.index, pd.DatetimeIndex):
                idx_full = df.index
                break
        if idx_full is not None and len(idx_full) > 0:
            global_start_idx = int(idx_full.searchsorted(ext_start, side="left"))
            phase_offset = int(global_start_idx % int(cfg.overlay_rebalance_hours))
    res = run_barbell_portfolio(
        ext_data,
        cfg,
        trade_start_idx=trade_start_idx,
        rebalance_phase_offset=phase_offset,
    )
    go_inputs = res.get("go_no_go_inputs", {}) if isinstance(res, dict) else {}
    final_eq = float(go_inputs.get("final_equity", res.get("final_equity", capital)))
    net_pnl = float(final_eq - capital)

    entry_fee = 0.0
    carry_pnl = 0.0
    overlay_net = 0.0
    attr = res.get("attribution_series")
    if isinstance(attr, pd.DataFrame) and not attr.empty and isinstance(attr.index, pd.DatetimeIndex):
        win = attr.loc[attr.index >= holdout_start]
        if not win.empty:
            if "fees" in win.columns:
                entry_fee = float(win["fees"].iloc[:24].sum())
            if "funding_pnl" in win.columns and "basis_pnl" in win.columns:
                carry_pnl = float(win["funding_pnl"].sum() + win["basis_pnl"].sum())
            if "overlay_pnl" in win.columns:
                overlay_net = float(win["overlay_pnl"].sum())

    return {
        "method": "indicator_warm_cash_start",
        "reinvest_profits": bool(reinvest_profits),
        "capital": float(capital),
        "warmup_hours": int(warmup_hours),
        "phase_aligned": bool(aligned),
        "align_rebalance_clock": bool(aligned),
        "rebalance_phase_offset": int(phase_offset),
        "trade_start_idx": int(trade_start_idx),
        "trade_start_bars_available": int(len(idx_ref)) if idx_ref is not None else 0,
        "start_equity_usd": float(capital),
        "end_equity_usd": float(final_eq),
        "net_pnl_usd": net_pnl,
        "return_on_active_capital_pct": (net_pnl / float(capital) * 100.0) if capital > 0 else 0.0,
        "entry_fee_usd": float(entry_fee),
        "carry_pnl_usd": float(carry_pnl),
        "overlay_net_pnl_usd": float(overlay_net),
        "sleeve_attribution": compute_sleeve_attribution(
            res.get("attribution_series"),
            res.get("carry_result", {}).get("holdings_log")
            if isinstance(res.get("carry_result"), dict)
            else None,
            split_date=split_date,
            holdout_end=holdout_end,
        ),
        # Sprint 17: chi tra ve ket qua tho khi duoc yeu cau (chan doan phap y);
        # mac dinh False de giu payload JSON nhe va on dinh.
        **({"raw_result": res} if include_raw else {}),
    }


# ---------------------------------------------------------------------------
# 4. run_sprint11_tiers
# ---------------------------------------------------------------------------
def run_sprint11_tiers(
    data: dict[str, pd.DataFrame],
    capital: float = SPRINT11_INITIAL_CAPITAL,
    split_date: str = SPRINT11_SPLIT_DATE,
    holdout_end: str = SPRINT11_HOLDOUT_END,
    benchmark_returns: pd.Series | None = None,
    reinvest_profits: bool = True,
) -> dict[str, Any]:
    """Mo phong 4 combo tham so Sprint 11 tren toan bo ky va holdout.

    Luoi 4 combo: carry_variant in ("tier1a", "tier1b") x gating_mode in ("hard_binary", "smooth_conviction").
    Moi combo duoc boc try/except de khong raise toan bo neu co 1 combo bi loi.
    """
    combos = [
        ("tier1a", "hard_binary"),
        ("tier1a", "smooth_conviction"),
        ("tier1b", "hard_binary"),
        ("tier1b", "smooth_conviction"),
    ]

    holdout_data = _slice_data_range(data, split_date, holdout_end)
    rows: list[dict[str, Any]] = []

    for variant, gating in combos:
        combo_name = f"{variant}/{gating}"
        try:
            cfg = BarbellConfig(
                initial_capital=float(capital),
                carry_variant=variant,
                gating_mode=gating,
                enable_snr_gate=True,
                overlay_quantize_legs=True,
                reinvest_profits=bool(reinvest_profits),
            )

            # 1. Chay tren toan bo ky
            res = run_barbell_portfolio(data, cfg)
            metrics = res.get("metrics", {}) if isinstance(res, dict) else {}
            go_inputs = res.get("go_no_go_inputs", {}) if isinstance(res, dict) else {}

            ann_ret = float(go_inputs.get("annualized_return_pct", metrics.get("Annualized Return %", 0.0)))
            sharpe = float(go_inputs.get("sharpe", metrics.get("Sharpe", 0.0)))
            maxdd = float(go_inputs.get("maxdd_pct", metrics.get("MaxDD %", 0.0)))
            final_eq = float(go_inputs.get("final_equity", res.get("final_equity", capital)))
            net_pnl = float(go_inputs.get("net_pnl", res.get("net_pnl", final_eq - capital)))

            c_var = str(res.get("carry_variant", variant))
            g_mode = str(res.get("gating_mode", gating))
            c_fallback = bool(res.get("carry_fallback", False))

            ov_gross = float(res.get("overlay_gross_pnl", metrics.get("overlay_gross_pnl", 0.0)))
            ov_net = float(res.get("overlay_net_pnl", metrics.get("overlay_net_pnl", 0.0)))
            ov_fees = float(res.get("overlay_fees", metrics.get("overlay_fees", 0.0)))
            ov_quant = int(res.get("overlay_quantization_events", metrics.get("overlay_quantization_events", 0)))
            ov_mean_lev = float(metrics.get("overlay_mean_leverage", 0.0))
            ov_gate_ratio = float(metrics.get("overlay_gate_on_ratio", metrics.get("overlay_active_ratio", 0.0)))

            attr_df = res.get("attribution_series")
            if isinstance(attr_df, pd.DataFrame) and not attr_df.empty:
                if "basis_pnl" in attr_df.columns and len(attr_df["basis_pnl"]) > 1:
                    basis_mtm_std = float(attr_df["basis_pnl"].std(ddof=1))
                else:
                    basis_mtm_std = 0.0
                f_sum = float(attr_df["funding_pnl"].sum()) if "funding_pnl" in attr_df.columns else 0.0
                b_sum = float(attr_df["basis_pnl"].sum()) if "basis_pnl" in attr_df.columns else 0.0
                carry_pnl = float(f_sum + b_sum)
                net_attr_sum = float(attr_df["net_pnl"].sum()) if "net_pnl" in attr_df.columns else 0.0
            else:
                basis_mtm_std = 0.0
                carry_pnl = 0.0
                net_attr_sum = 0.0

            rets = res.get("returns")
            beta_val = compute_beta_to_benchmark(rets, benchmark_returns) if benchmark_returns is not None else 0.0

            periods_list = split_period_returns(rets, 7) if isinstance(rets, pd.Series) else []
            pos_periods = sum(1 for p in periods_list if float(p.get("return_pct", 0.0)) > 0.0)

            # 2. Chay tren ky holdout
            res_h = run_barbell_portfolio(holdout_data, cfg)
            h_metrics = res_h.get("metrics", {}) if isinstance(res_h, dict) else {}
            h_final_eq = float(res_h.get("go_no_go_inputs", {}).get("final_equity", h_metrics.get("final_equity", capital)))
            h_ret_pct = float(h_metrics.get("Total Return %", ((h_final_eq - capital) / capital) * 100.0 if capital > 0 else 0.0))

            h_attr = res_h.get("attribution_series")
            if isinstance(h_attr, pd.DataFrame) and not h_attr.empty:
                h_funding = float(h_attr["funding_pnl"].sum()) if "funding_pnl" in h_attr.columns else 0.0
                h_basis = float(h_attr["basis_pnl"].sum()) if "basis_pnl" in h_attr.columns else 0.0
                h_carry_pnl = float(h_funding + h_basis)
            else:
                h_carry_pnl = 0.0

            h_ov_net = float(res_h.get("overlay_net_pnl", h_metrics.get("overlay_net_pnl", 0.0)))
            h_rets = res_h.get("returns")
            h_days = float(len(h_rets) / 24.0) if isinstance(h_rets, pd.Series) and len(h_rets) > 0 else 0.0

            # 2a-bis. Phi mo vi the ngay dau cua Standalone Cold-Start (24 bar dau)
            h_resim_entry_fee = 0.0
            h_resim_attr = res_h.get("attribution_series")
            if isinstance(h_resim_attr, pd.DataFrame) and not h_resim_attr.empty and "fees" in h_resim_attr.columns:
                h_resim_entry_fee = float(h_resim_attr["fees"].iloc[:24].sum())

            # 2b. Holdout theo phuong phap cua so lien tuc (slice tu ket qua toan ky)
            h_start_ts = pd.Timestamp(split_date, tz="UTC")
            h_end_ts = pd.Timestamp(holdout_end, tz="UTC")
            h_net_pnl = 0.0
            h_carry_slice = 0.0
            h_overlay_slice = 0.0
            h_fees_slice = 0.0
            h_equity_start = float(capital)
            h_compound_slice_pct = 0.0
            if isinstance(attr_df, pd.DataFrame) and not attr_df.empty:
                m = (attr_df.index >= h_start_ts) & (attr_df.index <= h_end_ts)
                if bool(m.any()):
                    h_net_pnl = float(attr_df.loc[m, "net_pnl"].sum())
                    if "funding_pnl" in attr_df.columns and "basis_pnl" in attr_df.columns:
                        h_carry_slice = float(attr_df.loc[m, "funding_pnl"].sum() + attr_df.loc[m, "basis_pnl"].sum())
                    if "overlay_pnl" in attr_df.columns:
                        h_overlay_slice = float(attr_df.loc[m, "overlay_pnl"].sum())
                    if "fees" in attr_df.columns:
                        h_fees_slice = float(attr_df.loc[m, "fees"].sum())
            eq_full = res.get("equity_curve")
            if isinstance(eq_full, pd.Series) and len(eq_full) > 0:
                before = eq_full[eq_full.index < h_start_ts]
                if len(before) > 0:
                    h_equity_start = float(before.iloc[-1])
            if isinstance(rets, pd.Series) and len(rets) > 0:
                mr = (rets.index >= h_start_ts) & (rets.index <= h_end_ts)
                if bool(mr.any()):
                    h_compound_slice_pct = float((np.prod(1.0 + rets[mr].to_numpy(dtype=np.float64)) - 1.0) * 100.0)
            h_ret_active_pct = (h_net_pnl / float(capital) * 100.0) if float(capital) > 0.0 else 0.0
            h_ret_compound_pct = (h_net_pnl / h_equity_start * 100.0) if h_equity_start > 0.0 else 0.0

            accounting_gap = float(abs(net_attr_sum - (final_eq - capital)))

            row = {
                "combo": combo_name,
                "carry_variant": c_var,
                "gating_mode": g_mode,
                "carry_fallback": c_fallback,
                "final_equity": final_eq,
                "net_pnl": net_pnl,
                "annualized_return_pct": ann_ret,
                "sharpe": sharpe,
                "maxdd": maxdd,
                "maxdd_pct": maxdd,
                "basis_mtm_std": basis_mtm_std,
                "carry_pnl": carry_pnl,
                "overlay_gross_pnl": ov_gross,
                "overlay_net_pnl": ov_net,
                "overlay_fees": ov_fees,
                "overlay_quantization_events": ov_quant,
                "overlay_mean_leverage": ov_mean_lev,
                "overlay_gate_on_ratio": ov_gate_ratio,
                "beta": beta_val,
                "beta_btc": beta_val,
                "positive_periods": int(pos_periods),
                "total_periods": 7,
                "periods": periods_list,
                "holdout_return_pct": h_compound_slice_pct,
                "holdout_net_pnl": h_net_pnl,
                "holdout_fees": h_fees_slice,
                "holdout_equity_start": h_equity_start,
                "holdout_return_on_active_capital_pct": h_ret_active_pct,
                "holdout_return_on_compounded_equity_pct": h_ret_compound_pct,
                "holdout_return_resim_pct": h_ret_pct,
                "holdout_net_pnl_resim": float(h_final_eq - float(capital)),
                "holdout_resim_entry_fee_usd": float(h_resim_entry_fee),
                "reinvest_profits": bool(getattr(cfg, "reinvest_profits", True)),
                "holdout_carry_pnl": (h_carry_slice if not (h_carry_slice == 0.0 and h_carry_pnl > 0.0) else h_carry_pnl),
                "holdout_overlay_net_pnl": (h_overlay_slice if not (h_overlay_slice == 0.0 and h_ov_net != 0.0) else h_ov_net),
                "holdout_days": h_days,
                "accounting_gap": accounting_gap,
                "executor_functional": False,
                "error": None,
            }
            rows.append(row)

        except Exception as e:
            logger.warning("Error running combo %s: %s", combo_name, e)
            rows.append({
                "combo": combo_name,
                "carry_variant": variant,
                "gating_mode": gating,
                "carry_fallback": False,
                "final_equity": float(capital),
                "net_pnl": 0.0,
                "annualized_return_pct": 0.0,
                "sharpe": 0.0,
                "maxdd": 0.0,
                "maxdd_pct": 0.0,
                "basis_mtm_std": 0.0,
                "carry_pnl": 0.0,
                "overlay_gross_pnl": 0.0,
                "overlay_net_pnl": 0.0,
                "overlay_fees": 0.0,
                "overlay_quantization_events": 0,
                "overlay_mean_leverage": 0.0,
                "overlay_gate_on_ratio": 0.0,
                "beta": 0.0,
                "beta_btc": 0.0,
                "positive_periods": 0,
                "total_periods": 7,
                "holdout_return_pct": 0.0,
                "holdout_net_pnl": 0.0,
                "holdout_fees": 0.0,
                "holdout_equity_start": 0.0,
                "holdout_return_on_active_capital_pct": 0.0,
                "holdout_return_on_compounded_equity_pct": 0.0,
                "holdout_return_resim_pct": 0.0,
                "holdout_net_pnl_resim": 0.0,
                "reinvest_profits": True,
                "holdout_carry_pnl": 0.0,
                "holdout_overlay_net_pnl": 0.0,
                "holdout_days": 0.0,
                "accounting_gap": 0.0,
                "executor_functional": False,
                "error": str(e),
            })

    return {
        "rows": rows,
        "capital": float(capital),
        "split_date": str(split_date),
        "holdout_end": str(holdout_end),
        "n_combos": len(rows),
    }


# ---------------------------------------------------------------------------
# 5. StubExchange & smoke_test_executor
# ---------------------------------------------------------------------------
class StubExchange:
    """Stub exchange in-memory mo phong API Binance phuc vu smoke test."""

    def __init__(self) -> None:
        self.order_counter = 0

    def milliseconds(self) -> int:
        return 1776700000000

    def fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        return {
            "bids": [[59990.0, 10.0], [59980.0, 10.0]],
            "asks": [[60010.0, 10.0], [60020.0, 10.0]],
        }

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.order_counter += 1
        return {
            "id": f"stub_order_{self.order_counter}",
            "status": "closed",
            "filled": float(amount),
            "amount": float(amount),
            "price": float(price or 60000.0),
            "symbol": symbol,
            "side": side,
            "type": order_type,
        }

    def fetch_order(self, id: str, symbol: str | None = None) -> dict[str, Any]:
        return {
            "id": str(id),
            "status": "closed",
            "filled": 0.002,
            "amount": 0.002,
            "symbol": symbol,
        }

    def cancel_order(self, id: str, symbol: str | None = None) -> bool:
        return True


def smoke_test_executor(tmp_path: Path | str | None = None) -> dict[str, Any]:
    """Kiem tra cau truc va runtime cua CarryLiveExecutor va BotState tren stub exchange."""
    methods_present = (
        hasattr(CarryLiveExecutor, "execute_maker_then_taker_pair")
        and callable(getattr(CarryLiveExecutor, "execute_maker_then_taker_pair"))
        and hasattr(CarryLiveExecutor, "execute_overlay_rebalance")
        and callable(getattr(CarryLiveExecutor, "execute_overlay_rebalance"))
    )

    state_field_names = {f.name for f in dataclasses.fields(BotState)}
    req_fields = {"carry_orders", "overlay_orders", "overlay_positions", "executor_functional"}
    state_fields_present = req_fields.issubset(state_field_names)

    carry_routed = False
    overlay_routed = False
    err_msg: str | None = None

    def _execute_harness(folder: Path) -> tuple[bool, bool]:
        stub = StubExchange()
        state_file = folder / "state.json"
        store = StateStore(state_file)

        executor = CarryLiveExecutor(
            state_store=store,
            capital=3000.0,
            symbols=["BTC/USDT:USDT"],
            overlay_config=None,
            enable_lot_quantization=True,
            perp_exchange=stub,
            spot_exchange=stub,
            mode="paper",
        )

        c_res = executor.execute_maker_then_taker_pair(
            "BTC/USDT:USDT",
            "sell",
            0.002,
            60000.0,
            sleep_fn=lambda s: None,
        )

        ov_alloc = {
            "allocations": [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "long",
                    "qty": 0.001,
                    "notional": 60.0,
                }
            ],
            "overlay_active": True,
        }
        ov_universe = {
            "BTC/USDT:USDT": pd.DataFrame({"perp_close": [60000.0], "close": [60000.0]})
        }
        ov_res = executor.execute_overlay_rebalance(ov_alloc, ov_universe)

        c_ok = bool(
            len(getattr(executor.state, "carry_orders", [])) > 0
            or float(c_res.get("filled_qty", 0.0)) > 0.0
        )
        ov_ok = bool(
            len(getattr(executor.state, "overlay_orders", [])) > 0
            or int(ov_res.get("n_orders", 0)) > 0
        )
        return c_ok, ov_ok

    try:
        if tmp_path is not None:
            carry_routed, overlay_routed = _execute_harness(Path(tmp_path))
        else:
            with tempfile.TemporaryDirectory() as td:
                carry_routed, overlay_routed = _execute_harness(Path(td))
    except Exception as e:
        err_msg = str(e)
        carry_routed = False
        overlay_routed = False

    executor_functional = bool(
        methods_present
        and state_fields_present
        and carry_routed
        and overlay_routed
        and (err_msg is None)
    )

    result = {
        "methods_present": bool(methods_present),
        "state_fields_present": bool(state_fields_present),
        "carry_routed": bool(carry_routed),
        "overlay_routed": bool(overlay_routed),
        "executor_functional": bool(executor_functional),
    }
    if err_msg is not None:
        result["error"] = err_msg

    return result


# ---------------------------------------------------------------------------
# 6. evaluate_sprint11_go_no_go
# ---------------------------------------------------------------------------
def evaluate_sprint11_go_no_go(
    payload: dict[str, Any],
    executor_functional: bool | None = None,
) -> dict[str, Any]:
    """Phan quyet GO/NO-GO Sprint 11 theo docs/go_no_go_sprint11.md.

    Chon primary row (uu tien tier1b/smooth_conviction, fallback tier1a/smooth_conviction).
    Kiem tra day du cac tieu chi bat buoc, khong raise khi thieu key.
    """
    if not isinstance(payload, dict):
        payload = {}

    rows = payload.get("rows", [])
    if isinstance(rows, list) and len(rows) > 0:
        primary: dict[str, Any] | None = None
        for r in rows:
            if isinstance(r, dict) and r.get("combo") == "tier1b/smooth_conviction" and not r.get("error"):
                primary = r
                break
        if primary is None:
            for r in rows:
                if isinstance(r, dict) and r.get("combo") == "tier1a/smooth_conviction" and not r.get("error"):
                    primary = r
                    break
        if primary is None:
            for r in rows:
                if isinstance(r, dict) and not r.get("error"):
                    primary = r
                    break
        if primary is None:
            primary = rows[0] if isinstance(rows[0], dict) else {}
    else:
        primary = payload

    primary_combo = str(primary.get("combo", "tier1b/smooth_conviction"))

    ann_ret = float(primary.get("annualized_return_pct", 0.0))
    pos_periods = int(primary.get("positive_periods", 0))
    holdout_ret = float(primary.get("holdout_return_pct", 0.0))
    holdout_carry = float(primary.get("holdout_carry_pnl", 0.0))
    holdout_overlay = float(primary.get("holdout_overlay_net_pnl", 0.0))
    holdout_resim = float(primary.get("holdout_return_resim_pct", 0.0))
    holdout_ret_active = float(primary.get("holdout_return_on_active_capital_pct", 0.0))
    holdout_ret_compound = float(primary.get("holdout_return_on_compounded_equity_pct", 0.0))
    sharpe = float(primary.get("sharpe", 0.0))
    maxdd = float(abs(primary.get("maxdd_pct", primary.get("maxdd", 0.0))))
    beta_btc = float(primary.get("beta_btc", primary.get("beta", 0.0)))
    basis_std = float(primary.get("basis_mtm_std", 0.0))
    gap = float(primary.get("accounting_gap", 0.0))
    quant_events = int(primary.get("overlay_quantization_events", 0))

    if executor_functional is not None:
        exec_ok = bool(executor_functional)
    else:
        exec_ok = bool(primary.get("executor_functional", payload.get("executor_functional", False)))

    criteria = [
        {
            "name": "annualized_return_pct",
            "value": ann_ret,
            "threshold": f">= {SPRINT11_GO_CRITERIA['annualized_return_pct']}%",
            "passed": bool(ann_ret >= float(SPRINT11_GO_CRITERIA["annualized_return_pct"])),
            "note": f"AnnRet: {ann_ret:.2f}% (threshold >= {SPRINT11_GO_CRITERIA['annualized_return_pct']}%)",
        },
        {
            "name": "min_positive_periods",
            "value": pos_periods,
            "threshold": f">= {SPRINT11_GO_CRITERIA['min_positive_periods']}/7",
            "passed": bool(pos_periods >= int(SPRINT11_GO_CRITERIA["min_positive_periods"])),
            "note": f"Positive periods: {pos_periods}/7 (threshold >= {SPRINT11_GO_CRITERIA['min_positive_periods']})",
        },
        {
            "name": "holdout_consistency",
            "value": {
                "return_pct": holdout_ret,
                "carry_pnl": holdout_carry,
                "overlay_net_pnl": holdout_overlay,
            },
            "threshold": f"ret >= {SPRINT11_GO_CRITERIA['holdout_return_pct']}% & carry_pnl > 0 & overlay_net > 0",
            "passed": bool(
                holdout_ret >= float(SPRINT11_GO_CRITERIA["holdout_return_pct"])
                and holdout_carry > 0.0
                and holdout_overlay > 0.0
            ),
            "note": (
                f"Holdout: ret={holdout_ret:.2f}% (>= {SPRINT11_GO_CRITERIA['holdout_return_pct']}%), "
                f"carry_pnl={holdout_carry:.2f} (> 0), overlay_net_pnl={holdout_overlay:.2f} (> 0)"
            ),
        },
        {
            "name": "holdout_consistency_resim",
            "value": {
                "return_pct": holdout_resim,
                "carry_pnl": holdout_carry,
                "overlay_net_pnl": holdout_overlay,
            },
            "threshold": "ret >= 3.5% (phuong phap re-sim tu 3000, khoa Sprint 11)",
            "passed": bool(holdout_resim >= 3.5 and holdout_carry > 0.0 and holdout_overlay > 0.0),
            "note": (
                f"Re-sim: ret={holdout_resim:.2f}%"
                if (holdout_resim >= 3.5 and holdout_carry > 0.0 and holdout_overlay > 0.0)
                else f"Re-sim: ret={holdout_resim:.2f}% (< 3.5%)"
            ),
        },
        {
            "name": "sharpe",
            "value": sharpe,
            "threshold": f">= {SPRINT11_GO_CRITERIA['sharpe']}",
            "passed": bool(sharpe >= float(SPRINT11_GO_CRITERIA["sharpe"])),
            "note": f"Sharpe: {sharpe:.2f} (threshold >= {SPRINT11_GO_CRITERIA['sharpe']})",
        },
        {
            "name": "maxdd_pct",
            "value": maxdd,
            "threshold": f"<= {SPRINT11_GO_CRITERIA['maxdd_pct']}%",
            "passed": bool(maxdd <= float(SPRINT11_GO_CRITERIA["maxdd_pct"])),
            "note": f"MaxDD: {maxdd:.2f}% (threshold <= {SPRINT11_GO_CRITERIA['maxdd_pct']}%)",
        },
        {
            "name": "abs_beta_btc",
            "value": abs(beta_btc),
            "threshold": f"<= {SPRINT11_GO_CRITERIA['abs_beta_btc']}",
            "passed": bool(abs(beta_btc) <= float(SPRINT11_GO_CRITERIA["abs_beta_btc"])),
            "note": f"|Beta BTC|: {abs(beta_btc):.4f} (threshold <= {SPRINT11_GO_CRITERIA['abs_beta_btc']})",
        },
        {
            "name": "basis_mtm_std",
            "value": basis_std,
            "threshold": "> 0",
            "passed": bool(basis_std > 1e-12),
            "note": f"Basis MTM std: {basis_std:.6f} (threshold > 0)",
        },
        {
            "name": "accounting_gap",
            "value": gap,
            "threshold": "< 1e-6",
            "passed": bool(gap < 1e-6),
            "note": f"Accounting gap: {gap:.2e} (threshold < 1e-6)",
        },
        {
            "name": "overlay_quantization_events",
            "value": quant_events,
            "threshold": "> 0",
            "passed": bool(quant_events > 0),
            "note": f"Overlay quantization events: {quant_events} (threshold > 0)",
        },
        {
            "name": "executor_functional",
            "value": exec_ok,
            "threshold": "== True",
            "passed": bool(exec_ok),
            "note": f"Executor functional: {exec_ok} (threshold == True)",
        },
        {
            "name": "holdout_active_capital_info",
            "value": holdout_ret_active,
            "threshold": "> 0 (chi de bao cao)",
            "passed": True,
            "note": f"active={holdout_ret_active:.2f}%, compound={holdout_ret_compound:.2f}%",
        },
    ]

    reasons = [c["note"] for c in criteria if not c["passed"]]
    go = bool(len(reasons) == 0)

    return {
        "primary_combo": primary_combo,
        "go": go,
        "criteria": criteria,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 6b. compute_holdout_accounting_bridge
# ---------------------------------------------------------------------------
def _find_primary_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Tim dong primary (tier1b/smooth_conviction) trong payload Sprint 11."""
    tiers = payload.get("tiers", payload) if isinstance(payload, dict) else {}
    rows = tiers.get("rows", []) if isinstance(tiers, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("combo")) == "tier1b/smooth_conviction":
            return row
    for row in rows:
        if isinstance(row, dict) and row.get("error") is None:
            return row
    return None


def _extract_sleeve_attribution(src: Any) -> dict[str, Any]:
    """Sprint 16: lay khoi sleeve attribution tu payload row/dict neu co."""
    if not isinstance(src, dict):
        return {}
    for key in ("sleeve_attribution", "holdout_sleeve_attribution"):
        val = src.get(key)
        if isinstance(val, dict) and val:
            return dict(val)
    return {}


def _build_sleeve_attribution_block(
    compound: dict[str, Any],
    fixed: dict[str, Any],
    cold: dict[str, Any],
    method_d: dict[str, Any],
    method_d_aligned: dict[str, Any],
) -> dict[str, Any]:
    """Sprint 16: gom attribution theo nhanh cho A/B/C/D/D' va chenh lech B - D'.

    Moi method co the thieu du lieu (payload cu) -> ``available=False`` va cac field = 0.
    Dang thuc duoc kiem tra rieng trong ``identity``.
    """
    empty: dict[str, Any] = {
        "available": False,
        "carry_net_pnl_usd": 0.0,
        "overlay_net_pnl_usd": 0.0,
        "avg_carry_slots_held": 0.0,
        "carry_utilization_pct": 0.0,
        "first_30d_net_pnl_usd": 0.0,
        "last_151d_net_pnl_usd": 0.0,
        "net_pnl_usd": 0.0,
    }

    def _norm(method: dict[str, Any]) -> dict[str, Any]:
        sleeve = method.get("sleeve_attribution") if isinstance(method, dict) else None
        if not isinstance(sleeve, dict) or not sleeve.get("available", False):
            return dict(empty)
        out = dict(empty)
        out.update(
            {
                "available": True,
                "carry_net_pnl_usd": float(sleeve.get("carry_net_pnl_usd", 0.0) or 0.0),
                "overlay_net_pnl_usd": float(sleeve.get("overlay_net_pnl_usd", 0.0) or 0.0),
                "avg_carry_slots_held": float(sleeve.get("avg_carry_slots_held", 0.0) or 0.0),
                "carry_utilization_pct": float(sleeve.get("carry_utilization_pct", 0.0) or 0.0),
                "first_30d_net_pnl_usd": float(sleeve.get("first_30d_net_pnl_usd", 0.0) or 0.0),
                "last_151d_net_pnl_usd": float(sleeve.get("last_151d_net_pnl_usd", 0.0) or 0.0),
                "net_pnl_usd": float(method.get("net_pnl_usd", 0.0) or 0.0),
            }
        )
        return out

    methods = {
        "A": _norm(compound),
        "B": _norm(fixed),
        "C": _norm(cold),
        "D": _norm(method_d),
        "D_aligned": _norm(method_d_aligned),
    }

    identity: dict[str, Any] = {}
    for name, item in methods.items():
        if not item["available"]:
            identity[name] = {"sleeve_gap_usd": 0.0, "window_gap_usd": 0.0}
            continue
        identity[name] = {
            "sleeve_gap_usd": float(
                (item["carry_net_pnl_usd"] + item["overlay_net_pnl_usd"]) - item["net_pnl_usd"]
            ),
            "window_gap_usd": float(
                (item["first_30d_net_pnl_usd"] + item["last_151d_net_pnl_usd"])
                - item["net_pnl_usd"]
            ),
        }

    b_item = methods["B"]
    d_item = methods["D_aligned"]
    delta: dict[str, Any] = {
        "available": bool(b_item["available"] and d_item["available"]),
        "total_usd": float(b_item["net_pnl_usd"] - d_item["net_pnl_usd"]),
        "carry_component_usd": float(
            b_item["carry_net_pnl_usd"] - d_item["carry_net_pnl_usd"]
        ),
        "overlay_component_usd": float(
            b_item["overlay_net_pnl_usd"] - d_item["overlay_net_pnl_usd"]
        ),
        "first_30d_component_usd": float(
            b_item["first_30d_net_pnl_usd"] - d_item["first_30d_net_pnl_usd"]
        ),
        "last_151d_component_usd": float(
            b_item["last_151d_net_pnl_usd"] - d_item["last_151d_net_pnl_usd"]
        ),
    }
    delta["residual_usd"] = float(
        delta["total_usd"]
        - (delta["carry_component_usd"] + delta["overlay_component_usd"])
    )
    return {"methods": methods, "identity": identity, "delta_b_vs_d_aligned": delta}


def compute_holdout_accounting_bridge(
    payload_compound: dict[str, Any],
    payload_fixed_base: dict[str, Any] | None = None,
    warm_cash: dict[str, Any] | None = None,
    warm_cash_aligned: dict[str, Any] | None = None,
    warm_cash_converged: dict[str, Any] | None = None,
    overlay_forensics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Doi soat 3 phuong phap do lai Holdout (warm-state vs cold-start).

    - ``continuous_compound``: chay lien tuc tu 2023-04 voi ``reinvest_profits=True``,
      do lai phan PnL trong cua so holdout (phuong phap slice lien tuc).
    - ``continuous_fixed_base``: cung chay lien tuc nhung ``reinvest_profits=False``,
      von hoat dong neo o ``initial_capital`` (co clamp khi equity giam duoi von goc).
    - ``standalone_cold_start``: chay lai rieng cua so holdout tu 100% tien mat.

    Dang thuc bao toan (sai so < 1e-6):
    ``A.net_pnl = C.net_pnl + compounding_scale_effect + cold_start_total_drag``.

    Sprint 20: ``warm_cash`` (D) va ``warm_cash_aligned`` (D') la cac phep do LICH SU
    (504h) va phai duoc tao bang cach truyen TUONG MINH
    ``warmup_hours=LEGACY_SPRINT14_WARMUP_HOURS`` (kem ``align_rebalance_clock``
    False/True). ``warm_cash_converged`` (E) la chuan moi:
    ``CANONICAL_SATURATED_WARMUP_HOURS`` (720h) + clock aligned.
    """
    row_a = _find_primary_row(payload_compound if isinstance(payload_compound, dict) else {})
    if row_a is None:
        raise ValueError("Khong tim thay dong ket qua trong payload compound")

    active_capital = 0.0
    tiers_c = payload_compound.get("tiers", payload_compound) if isinstance(payload_compound, dict) else {}
    meta = payload_compound.get("meta", {}) if isinstance(payload_compound, dict) else {}
    if isinstance(meta, dict) and meta.get("capital") is not None:
        active_capital = float(meta["capital"])
    if active_capital <= 0.0:
        active_capital = float(tiers_c.get("capital", 3000.0)) if isinstance(tiers_c, dict) else 3000.0

    start_a = float(row_a.get("holdout_equity_start", active_capital) or active_capital)
    net_a = float(row_a.get("holdout_net_pnl", 0.0))
    compound = {
        "start_equity_usd": start_a,
        "end_equity_usd": start_a + net_a,
        "net_pnl_usd": net_a,
        "return_on_start_equity_pct": (net_a / start_a * 100.0) if start_a > 0 else 0.0,
        "sleeve_attribution": _extract_sleeve_attribution(row_a),
    }

    fixed_row = _find_primary_row(payload_fixed_base) if payload_fixed_base else None
    if fixed_row is None:
        fixed = {
            "active_capital_usd": active_capital,
            "start_equity_usd": start_a,
            "end_equity_usd": start_a,
            "net_pnl_usd": 0.0,
            "return_on_active_capital_pct": 0.0,
            "available": False,
        }
    else:
        start_b = float(fixed_row.get("holdout_equity_start", active_capital) or active_capital)
        net_b = float(fixed_row.get("holdout_net_pnl", 0.0))
        fixed = {
            "active_capital_usd": active_capital,
            "start_equity_usd": start_b,
            "end_equity_usd": start_b + net_b,
            "net_pnl_usd": net_b,
            "return_on_active_capital_pct": (net_b / active_capital * 100.0) if active_capital > 0 else 0.0,
            "available": True,
            "sleeve_attribution": _extract_sleeve_attribution(fixed_row),
        }

    net_c = float(row_a.get("holdout_net_pnl_resim", 0.0))
    cold = {
        "start_equity_usd": active_capital,
        "end_equity_usd": active_capital + net_c,
        "net_pnl_usd": net_c,
        "return_on_active_capital_pct": (net_c / active_capital * 100.0) if active_capital > 0 else 0.0,
    }

    compounding_scale_effect = net_a - float(fixed["net_pnl_usd"])
    cold_start_total_drag = float(fixed["net_pnl_usd"]) - net_c
    entry_fee = float(row_a.get("holdout_resim_entry_fee_usd", 0.0) or 0.0)
    warmup_drag = cold_start_total_drag - entry_fee

    # Sprint 14 - Method D: chi bao da am + ton kho tien mat
    d_payload = warm_cash if isinstance(warm_cash, dict) else {}
    net_d = float(d_payload.get("net_pnl_usd", 0.0) or 0.0) if d_payload else 0.0
    method_d = {
        "available": bool(d_payload),
        "start_equity_usd": float(d_payload.get("start_equity_usd", active_capital) or active_capital),
        "end_equity_usd": float(d_payload.get("end_equity_usd", active_capital) or active_capital),
        "net_pnl_usd": net_d,
        "return_on_active_capital_pct": float(
            d_payload.get("return_on_active_capital_pct", 0.0) or 0.0
        ),
        "entry_fee_usd": float(d_payload.get("entry_fee_usd", 0.0) or 0.0),
        "warmup_hours": int(d_payload.get("warmup_hours", 0) or 0),
        "trade_start_idx": int(d_payload.get("trade_start_idx", 0) or 0),
        "sleeve_attribution": _extract_sleeve_attribution(d_payload),
    }

    indicator_blindness = (net_d - net_c) if method_d["available"] else 0.0

    # Sprint 15 - do lech pha dong ho tai can bang overlay 48h (Method D vs chuoi day du).
    d_aligned_payload = warm_cash_aligned if isinstance(warm_cash_aligned, dict) else {}
    net_d_aligned = (
        float(d_aligned_payload.get("net_pnl_usd", 0.0) or 0.0) if d_aligned_payload else net_d
    )
    method_d_aligned = {
        "available": bool(d_aligned_payload),
        "start_equity_usd": float(
            d_aligned_payload.get("start_equity_usd", active_capital) or active_capital
        ),
        "end_equity_usd": float(
            d_aligned_payload.get("end_equity_usd", active_capital) or active_capital
        ),
        "net_pnl_usd": net_d_aligned,
        "return_on_active_capital_pct": float(
            d_aligned_payload.get("return_on_active_capital_pct", 0.0) or 0.0
        ),
        "entry_fee_usd": float(d_aligned_payload.get("entry_fee_usd", 0.0) or 0.0),
        "warmup_hours": int(d_aligned_payload.get("warmup_hours", 0) or 0),
        "trade_start_idx": int(d_aligned_payload.get("trade_start_idx", 0) or 0),
        "rebalance_phase_offset": int(
            d_aligned_payload.get("rebalance_phase_offset", 0) or 0
        ),
        "sleeve_attribution": _extract_sleeve_attribution(d_aligned_payload),
    }
    clock_phase_drift = (
        (net_d_aligned - net_d) if (method_d["available"] and method_d_aligned["available"]) else 0.0
    )

    inventory_carryover = (
        (float(fixed["net_pnl_usd"]) - net_d - entry_fee) if method_d["available"] else warmup_drag
    )    # Dong nhat thuc: Fee + Blindness + ClockPhaseDrift + PureCarryover = B - C
    pure_inventory_carryover = (
        (float(fixed["net_pnl_usd"]) - net_d_aligned - entry_fee)
        if method_d_aligned["available"]
        else inventory_carryover
    )
    explained_total = entry_fee + indicator_blindness + clock_phase_drift + pure_inventory_carryover
    warm_cash_decomposition = {
        "cold_start_initial_entry_fee_usd": float(entry_fee),
        "indicator_blindness_drag_usd": float(indicator_blindness),
        "clock_phase_drift_usd": float(clock_phase_drift),
        "pure_inventory_carryover_usd": float(pure_inventory_carryover),
        "inventory_carryover_diff_usd": float(inventory_carryover),
        "explained_total_usd": float(explained_total),
        "unexplained_total_usd": float(cold_start_total_drag - explained_total),
    }

    # Sprint 17 - phep do chuan E: warm cash hoi tu (720h) + pha tai can bang tuyet doi.
    e_payload = warm_cash_converged if isinstance(warm_cash_converged, dict) else {}
    net_e = float(e_payload.get("net_pnl_usd", 0.0) or 0.0) if e_payload else 0.0
    e_true_fixed_warm_cash = {
        "available": bool(e_payload),
        "label": "E_true_fixed_warm_cash",
        "warmup_hours": int(e_payload.get("warmup_hours", 0) or 0),
        "rebalance_phase_offset": int(e_payload.get("rebalance_phase_offset", 0) or 0),
        "net_pnl_usd": net_e,
        "return_on_active_capital_pct": (
            (net_e / active_capital * 100.0) if (e_payload and active_capital > 0) else 0.0
        ),
        "overlay_net_pnl_usd": float(e_payload.get("overlay_net_pnl_usd", 0.0) or 0.0),
        "entry_fee_usd": float(e_payload.get("entry_fee_usd", 0.0) or 0.0),
        "gap_b_minus_e_usd": (float(fixed["net_pnl_usd"]) - net_e) if e_payload else 0.0,
        # Sprint 17: add-back dung phi vao lenh CUA CHINH E (E tra trong cua so holdout,
        # con B da tra truoc split) - dung `entry_fee` cua A/B se tinh trung hai lan
        # va lam sai lech con so (2.22 vs 3.97).
        "revised_pure_carryover_usd": (
            (float(fixed["net_pnl_usd"]) - net_e - float(e_payload.get("entry_fee_usd", 0.0) or 0.0))
            if e_payload
            else 0.0
        ),
        "sleeve_attribution": _extract_sleeve_attribution(e_payload),
    }
    overlay_divergence_forensics = (
        dict(overlay_forensics) if isinstance(overlay_forensics, dict) else {"available": False}
    )

    return {
        "holdout_days": float(row_a.get("holdout_days", 0.0) or 0.0),
        "active_capital_usd": active_capital,
        "continuous_compound": compound,
        "continuous_fixed_base": fixed,
        "standalone_cold_start": cold,
        "indicator_warm_cash_start": method_d,
        "indicator_warm_cash_start_clock_aligned": method_d_aligned,
        "warm_cash_decomposition": warm_cash_decomposition,
        "clock_phase_bridge": {
            "D_unaligned_clock_usd": float(net_d),
            "D_clock_aligned_usd": float(net_d_aligned),
            "clock_phase_drift_usd": float(clock_phase_drift),
            "pure_inventory_carryover_usd": float(pure_inventory_carryover),
            "identity_rhs_usd": float(fixed["net_pnl_usd"] - net_c),
            "identity_lhs_usd": float(explained_total),
            "identity_gap_usd": float((fixed["net_pnl_usd"] - net_c) - explained_total),
        },
        "sleeve_attribution": _build_sleeve_attribution_block(
            compound, fixed, cold, method_d, method_d_aligned
        ),
        "e_true_fixed_warm_cash": e_true_fixed_warm_cash,
        "overlay_divergence_forensics": overlay_divergence_forensics,
        "bridge_decomposition": {
            "compounding_scale_effect_usd": float(compounding_scale_effect),
            "cold_start_total_drag_usd": float(cold_start_total_drag),
            "cold_start_initial_entry_fee_usd": float(entry_fee),
            "warmup_and_min_hold_path_diff_usd": float(warmup_drag),
        },
    }


# ---------------------------------------------------------------------------
# 7. build_sprint11_summary_rows
# ---------------------------------------------------------------------------
def build_sprint11_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Tao danh sach cac dong phang phuc vu in markdown va xuat bao cao Sprint 11."""
    if not isinstance(payload, dict):
        return []

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append({
            "combo": str(r.get("combo", "")),
            "ann%": float(r.get("annualized_return_pct", 0.0)),
            "annualized_return_pct": float(r.get("annualized_return_pct", 0.0)),
            "sharpe": float(r.get("sharpe", 0.0)),
            "maxdd%": float(r.get("maxdd_pct", r.get("maxdd", 0.0))),
            "maxdd_pct": float(r.get("maxdd_pct", r.get("maxdd", 0.0))),
            "period+": int(r.get("positive_periods", 0)),
            "positive_periods": int(r.get("positive_periods", 0)),
            "holdout%": float(r.get("holdout_return_pct", 0.0)),
            "holdout_return_pct": float(r.get("holdout_return_pct", 0.0)),
            "holdout_active%": float(r.get("holdout_return_on_active_capital_pct", 0.0)),
            "holdout_compound%": float(r.get("holdout_return_on_compounded_equity_pct", 0.0)),
            "reinvest": bool(r.get("reinvest_profits", True)),
            "carry_pnl": float(r.get("carry_pnl", 0.0)),
            "overlay_net": float(r.get("overlay_net_pnl", 0.0)),
            "overlay_net_pnl": float(r.get("overlay_net_pnl", 0.0)),
            "basis_std": float(r.get("basis_mtm_std", 0.0)),
            "basis_mtm_std": float(r.get("basis_mtm_std", 0.0)),
            "beta": float(r.get("beta_btc", r.get("beta", 0.0))),
            "beta_btc": float(r.get("beta_btc", r.get("beta", 0.0))),
            "accounting_gap": float(r.get("accounting_gap", 0.0)),
            "executor": bool(r.get("executor_functional", False)),
            "executor_functional": bool(r.get("executor_functional", False)),
        })

    return out
