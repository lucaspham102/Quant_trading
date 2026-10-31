"""Backtest giai dau profile 1,000 USDT (Sprint 25): full / holdout 181 ngay / 7 cua so ban nien.

Mo phong tat dinh theo gio, tai su dung cac primitive co san (khong sua module nao):
carry (spot long + perp short) co tilt funding, overlay SNR long/short 48h, split-wallet cap,
va yield stacking thu dong. Chi doc parquet cache qua run_sprint11.load_multi_aligned_data.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_sprint11 import load_multi_aligned_data  # noqa: E402
from src.barbell_portfolio import (  # noqa: E402
    compute_adaptive_min_viable_leg_usd,
    compute_snr_dispersion,
    enforce_overlay_dollar_neutrality,
)
from src.cross_exchange_carry import LIQUID_8_SYMBOLS  # noqa: E402
from src.dynamic_carry_basis import DynamicCarryConfig, build_carry_basis_scores  # noqa: E402
from src.sprint7_eval import segment_metrics  # noqa: E402
from src.directional_trend_sleeve import (  # noqa: E402
    DrawdownCircuitBreakerConfig,
    TrendSleeveConfig,
    TrendSleeveState,
    compute_triple_confirmed_trend_weights,
    evaluate_drawdown_circuit_breaker,
    trend_weights_to_target_qty,
)
from src.stat_arb_sleeve import (  # noqa: E402
    ReversalSleeveConfig,
    compute_funding_conditioned_reversal_weights,
    reversal_weights_to_target_qty,
)
from src.strategy_profiles import (  # noqa: E402
    PROFILE_REGISTRY,
    StrategyProfileConfig,
    compute_funding_tilted_weights,
    compute_step_passive_yield_usd,
    enforce_split_wallet_futures_cap,
)

LOGGER = logging.getLogger("sprint25_tournament")

DEFAULT_PROFILES = (
    "baseline_1k,carry_max_eta_1k,hybrid_trend_conservative_1k,"
    "hybrid_trend_balanced_1k,alpha_boost_1k,turbo_neutral_1k,"
    "bi_trend_carry_1k,carry_stat_arb_1k,regime_omni_1k,flexible_regime_1k"
)
DEFAULT_SENSITIVITY_PROFILES = "baseline_1k,default"
PROFILE_ORDER: tuple[str, ...] = (
    "default",
    "baseline_1k",
    "carry_max_eta_1k",
    "hybrid_trend_conservative_1k",
    "hybrid_trend_balanced_1k",
    "alpha_boost_1k",
    "turbo_neutral_1k",
    "bi_trend_carry_1k",
    "carry_stat_arb_1k",
    "regime_omni_1k",
    "flexible_regime_1k",
)

TOURNAMENT_LOT_STEPS: dict[str, float] = {
    "BTC/USDT:USDT": 0.001,
    "ETH/USDT:USDT": 0.001,
    "SOL/USDT:USDT": 0.01,
    "BNB/USDT:USDT": 0.01,
    "LINK/USDT:USDT": 0.01,
    "AVAX/USDT:USDT": 1.0,
    "XRP/USDT:USDT": 1.0,
    "DOGE/USDT:USDT": 1.0,
}

_MIN_CARRY_NOTIONAL_USD = 10.0
_RESIZE_DEADBAND = 0.20
_SNR_GATE = 1.35
_OVERLAY_STEP_BARS = 48
_VOL_LOOKBACK = 168
# Sprint 27: EWMA funding (span 9 funding events ~ 3 ngay) cho long veto / short anti-squeeze.
_FUNDING_EWMA_ALPHA = 2.0 / (9.0 + 1.0)
_REVERSAL_WINDOW_BARS = 240
_REVERSAL_MAX_IMBALANCE_USD = 10.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sprint 25: 1k profile tournament backtest")
    p.add_argument("--profiles", type=str, default=DEFAULT_PROFILES)
    p.add_argument("--start", type=str, default="2023-04-01")
    p.add_argument("--end", type=str, default="2026-10-19")
    p.add_argument("--raw-dir", type=str, default="data/raw")
    p.add_argument("--out-json", type=str, default="logs/sprint26_high_return_evidence.json")
    p.add_argument("--holdout-days", type=int, default=181)
    p.add_argument("--warmup-hours", type=int, default=720)
    p.add_argument("--carry-cost-bps", type=float, default=12.5)
    p.add_argument("--spot-cost-bps", type=float, default=7.5)
    p.add_argument("--perp-cost-bps", type=float, default=5.0)
    # Sprint 25.1: secondary sensitivity run under BNB/maker-optimised fees (15 bps roundtrip)
    # so the operator guide can compare strict taker (25 bps) vs real-world maker execution.
    p.add_argument("--sensitivity-profiles", type=str, default=DEFAULT_SENSITIVITY_PROFILES)
    p.add_argument("--sens-spot-cost-bps", type=float, default=4.5)
    p.add_argument("--sens-perp-cost-bps", type=float, default=3.0)
    p.add_argument("--sens-carry-cost-bps", type=float, default=7.5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _floor_to_step(qty: float, step: float) -> float:
    if not (qty > 0.0) or not (step > 0.0):
        return 0.0
    return round(math.floor(qty / step + 1e-12) * step, 8)


def build_tournament_profiles() -> dict[str, StrategyProfileConfig]:
    out: dict[str, StrategyProfileConfig] = {
        n: PROFILE_REGISTRY[n] for n in PROFILE_ORDER if n in PROFILE_REGISTRY
    }
    out["alpha_boost_1k_pure_equal"] = dataclasses.replace(
        PROFILE_REGISTRY["alpha_boost_1k"],
        name="alpha_boost_1k_pure_equal",
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
    )
    return out


def build_market_matrices(aligned: dict[str, pd.DataFrame]) -> dict[str, Any]:
    symbols = sorted(s for s, df in aligned.items() if df is not None and not df.empty)
    if not symbols:
        raise ValueError("aligned data rong: khong co symbol hop le")
    index = None
    for s in symbols:
        index = aligned[s].index if index is None else index.intersection(aligned[s].index)
    index = index.sort_values()
    if len(index) < 2:
        raise ValueError("giao index chung qua ngan")

    perp, spot, fund = {}, {}, {}
    for s in symbols:
        df = aligned[s].reindex(index)
        p_col = "perp_close" if "perp_close" in df.columns else "close"
        perp[s] = df[p_col].astype("float64")
        spot[s] = df["spot_close"].astype("float64")
        if "funding_event_binance" in df.columns:
            fund[s] = df["funding_event_binance"].astype("float64").fillna(0.0)
        else:
            fund[s] = pd.Series(0.0, index=index)
    perp_df = pd.DataFrame(perp, index=index)[symbols].ffill()
    spot_df = pd.DataFrame(spot, index=index)[symbols].ffill()
    fund_df = pd.DataFrame(fund, index=index)[symbols].fillna(0.0)

    cfg = DynamicCarryConfig(
        initial_capital=3000.0, ema_span=42, top_k=4, entry_threshold=0.0,
        exit_threshold=0.0, min_hold_periods=63, buffer_rank=3, w_basis=0.0,
        basis_ema_hours=168, expected_hold_periods=63,
        switch_rule="tier1b_buffer_rank", eta_override=0.80,
    )
    # Sprint 25 forensic fix: build_carry_basis_scores chi doc schema live
    # ('funding_rate_event' / 'last_known_funding_rate'); du lieu backtest dung
    # 'funding_event_binance' / 'last_funding_binance'. Bridge tuong minh de score
    # khong bi bang 0 (neu khong carry se khong bao gio mo vi the).
    score_input: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df_src = aligned[s]
        df_bridge = df_src.copy()
        if "funding_rate_event" not in df_bridge.columns:
            if "funding_event_binance" in df_bridge.columns:
                df_bridge["funding_rate_event"] = df_bridge["funding_event_binance"].astype("float64")
            else:
                df_bridge["funding_rate_event"] = 0.0
        if "last_known_funding_rate" not in df_bridge.columns:
            if "last_funding_binance" in df_bridge.columns:
                df_bridge["last_known_funding_rate"] = df_bridge["last_funding_binance"].astype("float64")
            else:
                df_bridge["last_known_funding_rate"] = 0.0
        score_input[s] = df_bridge
    scores = build_carry_basis_scores(score_input, cfg)
    score_df = pd.DataFrame(
        {s: scores[s]["score"].reindex(index) if s in scores else np.nan for s in symbols},
        index=index,
    )[symbols].astype("float64")

    snr_full = compute_snr_dispersion(perp_df, lookback=336)
    return {
        "index": index,
        "perp_close": perp_df,
        "spot_close": spot_df,
        "funding_event": fund_df,
        "carry_score": score_df,
        "funding_bps": fund_df * 10000.0,
        "snr": snr_full[symbols],
        "snr_dispersion": snr_full["raw_snr_dispersion"],
    }


def _ok(x: float) -> bool:
    return bool(np.isfinite(x)) and x > 0.0


def _nan0(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def simulate_profile(
    profile: StrategyProfileConfig,
    market: dict[str, Any],
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    trade_start_idx: int | None = None,
    overlay_lot_specs: dict | None = None,
) -> dict[str, Any]:
    index = market["index"]
    n = len(index)
    end = n if end_idx is None else int(end_idx)
    start = int(start_idx)
    tstart = start if trade_start_idx is None else int(trade_start_idx)
    if not (0 <= start <= tstart < end <= n):
        raise ValueError(f"cua so khong hop le: start={start} trade={tstart} end={end} n={n}")

    syms = list(market["perp_close"].columns)
    sidx = {s: i for i, s in enumerate(syms)}
    perp = market["perp_close"].to_numpy(dtype=float)
    spot = market["spot_close"].to_numpy(dtype=float)
    fund = market["funding_event"].to_numpy(dtype=float)
    fbps = market["funding_bps"].to_numpy(dtype=float)
    score = market["carry_score"].to_numpy(dtype=float)
    snr = market["snr"].to_numpy(dtype=float)
    disp = market["snr_dispersion"].to_numpy(dtype=float)
    is_fund = np.isin(index.hour, [0, 8, 16]) & (index.minute == 0)

    _carry_cost_bps = market.get("carry_cost_bps")
    cost_carry = (
        float(_carry_cost_bps)
        if _carry_cost_bps is not None
        else float(market.get("spot_cost_bps", 7.5)) + float(market.get("perp_cost_bps", 5.0))
    ) / 10000.0
    cost_perp = float(market.get("perp_cost_bps", 5.0)) / 10000.0
    lot_specs = overlay_lot_specs or {
        s: {"step_size": TOURNAMENT_LOT_STEPS.get(s, 0.001), "min_notional": 10.0} for s in syms
    }
    cap_usd = float(profile.capital_usd)
    top_k = int(profile.carry_top_k)
    entry_frac = profile.carry_entry_bps * 1e-4
    exit_frac = profile.carry_exit_bps * 1e-4
    min_viable = compute_adaptive_min_viable_leg_usd(cap_usd)

    carry: dict[str, dict[str, float]] = {}
    overlay: dict[str, dict[str, float]] = {}
    carry_scale = 1.0
    # ---- Sprint 26: directional trend sleeve + 2-tier drawdown circuit breaker
    trend_enabled = bool(getattr(profile, "enable_trend_sleeve", False))
    trend_cfg: TrendSleeveConfig | None = None
    breaker_cfg = DrawdownCircuitBreakerConfig()
    if trend_enabled:
        trend_cfg = TrendSleeveConfig(
            capital_usd=cap_usd,
            target_vol_ann=float(profile.trend_target_vol_ann),
            max_gross_pct=float(profile.trend_max_gross_pct),
            rebalance_every_hours=int(profile.trend_rebalance_hours),
            z_entry=float(profile.trend_z_entry),
            allow_short_trend=bool(profile.allow_short_trend),
            z_short_entry=float(profile.trend_z_short_entry),
            require_macro_regime_filter=bool(
                getattr(profile, "require_macro_regime_filter", False)
            ),
            short_gross_scale=float(getattr(profile, "short_gross_scale", 1.0)),
        )
        breaker_cfg = DrawdownCircuitBreakerConfig(
            soft_dd_threshold_pct=float(profile.soft_dd_threshold_pct),
            hard_dd_threshold_pct=float(profile.hard_dd_threshold_pct),
        )
    trend_state = TrendSleeveState()
    trend: dict[str, dict[str, float]] = {}
    trend_pnl = 0.0
    short_trend_pnl = 0.0
    long_trend_pnl = 0.0
    # ---- Sprint 27: funding-conditioned reversal sleeve (market-neutral) + EWMA funding
    reversal_enabled = bool(getattr(profile, "enable_reversal_sleeve", False))
    reversal_cfg: ReversalSleeveConfig | None = None
    if reversal_enabled:
        reversal_cfg = ReversalSleeveConfig(
            capital_usd=cap_usd, gross_pct=float(profile.reversal_gross_pct),
        )
    reversal: dict[str, dict[str, float]] = {}
    reversal_pnl = 0.0
    reversal_rebalances = 0
    reversal_skipped_imbalance = 0
    max_reversal_imbalance = 0.0
    f_ewma = np.zeros(len(syms), dtype=float)
    trend_peak_equity = cap_usd
    trend_hard_lock_bars: int | None = None
    trend_cb_state = "DISABLED"
    trend_cb_triggers = 0
    trend_cb_soft_activations = 0
    trend_bars_soft = 0
    trend_bars_hard = 0
    trend_gross_sum = 0.0
    trend_gross_n = 0
    trend_rebalance_bars = max(1, int(getattr(profile, "trend_rebalance_hours", 24) or 24))
    # Rolling window: 3x slow EMA la du de EMA(1440) hoi tu, dong thoi giu chi phi tinh toan
    # trong tam kiem soat (moi lan tai can bang chi doc ~4,320 bar gan nhat).
    trend_window = 3 * int(TrendSleeveConfig().ema_slow_hours)
    perp_frame = market["perp_close"]
    funding_pnl = basis_pnl = overlay_pnl = fees = cum_yield = 0.0
    max_lev = 0.0
    cap_binds = 0
    slots_sum = 0
    n_open = n_close = n_resize = n_ov_rebal = 0

    m = end - start
    eq = np.full(m, cap_usd, dtype=float)
    cy = np.zeros(m, dtype=float)

    for t in range(start, end):
        traded = t >= tstart
        if traded and t > tstart:
            for s, pos in carry.items():
                i = sidx[s]
                d_sp = spot[t, i] - spot[t - 1, i]
                d_pp = perp[t, i] - perp[t - 1, i]
                if np.isfinite(d_sp) and np.isfinite(d_pp):
                    basis_pnl += pos["qty"] * (d_sp - d_pp)
            for s, pos in overlay.items():
                d_pp = perp[t, sidx[s]] - perp[t - 1, sidx[s]]
                if np.isfinite(d_pp):
                    overlay_pnl += pos["qty"] * d_pp
            for s, pos in trend.items():
                d_pp = perp[t, sidx[s]] - perp[t - 1, sidx[s]]
                if np.isfinite(d_pp):
                    step_pnl = pos["qty"] * d_pp
                    trend_pnl += step_pnl
                    if pos["qty"] < 0.0:
                        short_trend_pnl += step_pnl
                    else:
                        long_trend_pnl += step_pnl
            for key, pos in reversal.items():
                i_r = sidx[key.split("|")[0]]
                d_pp = perp[t, i_r] - perp[t - 1, i_r]
                if np.isfinite(d_pp):
                    reversal_pnl += pos["qty"] * d_pp

        # Sprint 27: EWMA funding (causal, cap nhat tai moi su kien funding, ke ca warmup).
        if is_fund[t]:
            f_now = np.where(np.isfinite(fbps[t]), fbps[t], 0.0)
            f_ewma = (1.0 - _FUNDING_EWMA_ALPHA) * f_ewma + _FUNDING_EWMA_ALPHA * f_now

        # Sprint 25 fix: harvest funding for positions held during the period ENDING at t
        # BEFORE any rebalance at t, otherwise a leg opened at close t would instantly collect
        # funding already paid, and a leg closed at t would lose the funding it had earned.
        if traded and is_fund[t]:
            for s, pos in carry.items():
                i = sidx[s]
                if np.isfinite(fund[t, i]) and np.isfinite(perp[t, i]):
                    funding_pnl += pos["qty"] * perp[t, i] * fund[t, i]
            for s, pos in overlay.items():
                f = fund[t, sidx[s]]
                if np.isfinite(f):
                    overlay_pnl += -pos["usd"] * f
            # Sprint 26 fix: a long Perp trend leg PAYS funding when the rate is positive,
            # mirroring the overlay convention (-signed_usd * rate). Omitting it would hand
            # the trend sleeve ~11%/yr of notional in fake alpha.
            for s, pos in trend.items():
                i = sidx[s]
                f = fund[t, i]
                if np.isfinite(f) and np.isfinite(perp[t, i]):
                    f_pnl = -pos["qty"] * perp[t, i] * f
                    trend_pnl += f_pnl
                    if pos["qty"] < 0.0:
                        short_trend_pnl += f_pnl
                    else:
                        long_trend_pnl += f_pnl
            for key, pos in reversal.items():
                i = sidx[key.split("|")[0]]
                f = fund[t, i]
                if np.isfinite(f) and np.isfinite(perp[t, i]):
                    reversal_pnl += -pos["qty"] * perp[t, i] * f

        if traded and is_fund[t]:
            valid = [s for s in syms if _ok(perp[t, sidx[s]]) and _ok(spot[t, sidx[s]])]
            sc = {s: _nan0(score[t, sidx[s]]) for s in syms}
            fb = {s: _nan0(fbps[t, sidx[s]]) for s in syms}
            ranked = sorted(valid, key=lambda s: (-sc[s], s))
            cand = ranked[: top_k + int(profile.carry_buffer_rank)]
            set_changed = False
            for s in sorted(carry):
                if s not in valid:
                    continue
                pos = carry[s]
                held_bars = (t - int(pos["entry_bar"])) // 8
                if sc[s] < exit_frac or (s not in cand and held_bars >= profile.carry_min_hold_bars):
                    fees += pos["qty"] * perp[t, sidx[s]] * cost_carry
                    del carry[s]
                    set_changed = True
                    n_close += 1
            new = []
            for s in cand:
                if s in carry:
                    continue
                if len(carry) + len(new) >= top_k:
                    break
                if sc[s] >= entry_frac:
                    new.append(s)
            active = sorted(list(carry) + new)
            if active:
                w = compute_funding_tilted_weights(
                    active, fb, mode=profile.carry_weighting_mode,
                    tilt_alpha=profile.carry_tilt_alpha,
                    max_single_weight=profile.carry_max_single_weight,
                )
                budget = cap_usd * profile.carry_eta * min(len(active) / top_k, 1.0) * carry_scale
                per_cap = max(cap_usd / top_k, budget * profile.carry_max_single_weight)
                set_changed = set_changed or bool(new)
                for s in active:
                    if s not in valid:
                        continue
                    i = sidx[s]
                    px = perp[t, i]
                    target = min(budget * w.get(s, 0.0), per_cap)
                    qty = _floor_to_step(target / px, TOURNAMENT_LOT_STEPS.get(s, 0.001))
                    if qty * px < _MIN_CARRY_NOTIONAL_USD:
                        if s in carry:
                            fees += carry[s]["qty"] * px * cost_carry
                            del carry[s]
                            n_close += 1
                        continue
                    if s not in carry:
                        fees += qty * px * cost_carry
                        carry[s] = {"qty": qty, "perp_entry": px, "spot_entry": spot[t, i], "entry_bar": float(t)}
                        n_open += 1
                    else:
                        old = carry[s]["qty"]
                        # Sprint 25: chi resize khi tap chon thay doi va do lech > deadband
                        # (tranh dao lot 8h/lan lam fee phinh gia tao).
                        if (
                            qty != old
                            and set_changed
                            and abs(qty - old) * px > _RESIZE_DEADBAND * old * px
                        ):
                            fees += abs(qty - old) * px * cost_carry
                            carry[s]["qty"] = qty
                            n_resize += 1

        if traded and t % _OVERLAY_STEP_BARS == 0:
            n_ov_rebal += 1
            row = snr[t]
            vo = [i for i in range(len(syms)) if np.isfinite(row[i]) and _ok(perp[t, i])]
            desired = 0.0
            li = si = -1
            if np.isfinite(disp[t]) and disp[t] >= _SNR_GATE and len(vo) >= 2:
                li = max(vo, key=lambda i: (row[i], -i))
                si = min(vo, key=lambda i: (row[i], i))
                if li != si:
                    lo = max(0, t - _VOL_LOOKBACK)
                    ratio = pd.Series(perp[lo : t + 1, li] / perp[lo : t + 1, si])
                    vol = float(ratio.pct_change().std(ddof=1)) * math.sqrt(8760.0)
                    vol = vol if np.isfinite(vol) else 0.0
                    vol_scale = min(1.0, profile.overlay_target_vol_ann / max(vol, 1e-6))
                    desired = cap_usd * profile.overlay_gross_pct * vol_scale
            carry_notional = sum(
                p["qty"] * perp[t, sidx[s]] for s, p in carry.items() if np.isfinite(perp[t, sidx[s]])
            )
            cap = enforce_split_wallet_futures_cap(
                carry_perp_notional_usd=carry_notional,
                desired_overlay_gross_usd=desired,
                futures_wallet_equity_usd=profile.futures_wallet_usd,
                max_futures_leverage=profile.max_futures_leverage,
            )
            max_lev = max(max_lev, cap["effective_futures_leverage"])
            if cap["clamped"]:
                cap_binds += 1
            factor = cap["carry_scale_factor"]
            carry_scale = factor if factor < 1.0 else 1.0
            if factor < 1.0:
                for s in sorted(carry):
                    px = perp[t, sidx[s]]
                    if not _ok(px):
                        continue
                    q_new = _floor_to_step(carry[s]["qty"] * factor, TOURNAMENT_LOT_STEPS.get(s, 0.001))
                    fees += abs(carry[s]["qty"] - q_new) * px * cost_carry
                    if q_new * px < _MIN_CARRY_NOTIONAL_USD:
                        # Sprint 25 fix: the residual lot is fully closed below, so charge
                        # the closing fee on it too (previously it vanished fee-free).
                        fees += q_new * px * cost_carry
                        del carry[s]
                        n_close += 1
                    else:
                        carry[s]["qty"] = q_new
            gross = cap["allowed_overlay_gross_usd"]
            targets: dict[str, float] = {}
            prices: dict[str, float] = {}
            if gross > 0.0 and li >= 0 and si >= 0 and li != si:
                targets = {syms[li]: gross / 2.0, syms[si]: -gross / 2.0}
                prices = {syms[li]: float(perp[t, li]), syms[si]: float(perp[t, si])}
            qmap = enforce_overlay_dollar_neutrality(
                targets, prices, lot_specs, max_imbalance_usd=10.0, min_viable_leg_usd=min_viable,
            ) if targets else {}
            new_ov = {s: q for s, q in qmap.items() if q != 0.0}
            for s in sorted(set(overlay) | set(new_ov)):
                px = perp[t, sidx[s]]
                if not _ok(px):
                    continue
                old_q = overlay[s]["qty"] if s in overlay else 0.0
                new_q = new_ov.get(s, 0.0)
                if new_q == old_q:
                    continue
                fees += abs(new_q - old_q) * px * cost_perp
                if new_q == 0.0:
                    del overlay[s]
                else:
                    overlay[s] = {"qty": new_q, "entry": px, "usd": new_q * px}

        # ---- Sprint 26: 2-tier circuit breaker tren equity mark-to-market, roi moi tai can bang trend
        if trend_enabled:
            eq_now = cap_usd + funding_pnl + basis_pnl + overlay_pnl + trend_pnl + reversal_pnl - fees
            trend_peak_equity = max(trend_peak_equity, eq_now)
            pending_bars = None if trend_hard_lock_bars is None else trend_hard_lock_bars + 1
            cb = evaluate_drawdown_circuit_breaker(
                current_equity=eq_now,
                peak_equity=trend_peak_equity,
                bars_since_hard_stop=pending_bars,
                cfg=breaker_cfg,
            )
            if cb["state"] == "HARD_STOP_LOCKED":
                if trend_hard_lock_bars is None:
                    trend_hard_lock_bars = 0
                    trend_cb_triggers += 1
                    # A hard stop is an emergency: flatten on the trigger bar itself instead
                    # of waiting up to 24h for the next rebalance. Trend PnL already accrued
                    # bar-by-bar, so only the exit fee is missing here.
                    for s in sorted(trend):
                        i = sidx[s]
                        px = float(perp[t, i]) if np.isfinite(perp[t, i]) else 0.0
                        if px > 0.0:
                            fees += abs(trend[s]["qty"]) * px * cost_perp
                        trend.pop(s, None)
                    trend_state = TrendSleeveState()
                else:
                    trend_hard_lock_bars = pending_bars
                trend_bars_hard += 1
            else:
                trend_hard_lock_bars = None
                if cb["state"] == "SOFT_THROTTLE":
                    if trend_cb_state != "SOFT_THROTTLE":
                        trend_cb_soft_activations += 1
                    trend_bars_soft += 1
            trend_cb_state = str(cb["state"])
        else:
            cb = {"risk_scale": 1.0, "state": "DISABLED", "current_dd_pct": 0.0}
            trend_cb_state = "DISABLED"

        if trend_enabled and traded and trend_cfg is not None:
            # Trend chi tai can bang moi `trend_rebalance_bars` bar (trailing stop duoc kiem tra
            # cung nhip tai can bang nay de tranh O(n^2) tren 30k bar).
            if (t - tstart) % trend_rebalance_bars == 0:
                lo = max(0, t - trend_window)
                window = perp_frame.iloc[lo : t + 1]
                weights = compute_triple_confirmed_trend_weights(
                    window,
                    bar_index=len(window) - 1,
                    config= dataclasses.replace(trend_cfg, max_gross_pct=min(1.0, float(profile.trend_max_gross_pct) + max(0.0, float(profile.carry_eta) - sum(p["qty"] * perp[t, sidx[s]] for s, p in carry.items() if np.isfinite(perp[t, sidx[s]])) / cap_usd) * 0.5)) if bool(getattr(profile, "dynamic_carry_to_alpha_margin", False)) else trend_cfg,
                    state=trend_state,
                    circuit_breaker=cb,
                    allow_new_entries=True,
                    funding_bps=dict(zip(syms, f_ewma.tolist())),
                )
                prices_now = {
                    s: float(perp[t, sidx[s]]) for s in syms if np.isfinite(perp[t, sidx[s]])
                }
                # Hard risk cap: trend notional khong duoc vuot cong suat futures con trong
                # sau khi da tru carry perp + overlay gross (ton trong max_futures_leverage).
                carry_notional_now = sum(
                    p["qty"] * perp[t, sidx[s]]
                    for s, p in carry.items()
                    if np.isfinite(perp[t, sidx[s]])
                )
                overlay_gross_now = sum(abs(float(p.get("usd", 0.0))) for p in overlay.values())
                reversal_gross_now = sum(
                    abs(p["qty"]) * perp[t, sidx[k_r.split("|")[0]]]
                    for k_r, p in reversal.items()
                    if np.isfinite(perp[t, sidx[k_r.split("|")[0]]])
                )
                fut_capacity = max(
                    0.0,
                    float(profile.futures_wallet_usd) * float(profile.max_futures_leverage)
                    - carry_notional_now
                    - overlay_gross_now
                    - reversal_gross_now,
                )
                desired_gross = float(sum(abs(w) for w in weights.values())) * cap_usd
                if desired_gross > 0.0 and desired_gross > fut_capacity:
                    scale = fut_capacity / desired_gross
                    weights = {s: float(w) * scale for s, w in weights.items()}
                targets = trend_weights_to_target_qty(
                    weights, prices_now, cap_usd, TOURNAMENT_LOT_STEPS, min_notional_usd=10.0
                )
                for s in sorted(set(trend) | set(targets)):
                    i = sidx[s]
                    px = float(perp[t, i]) if np.isfinite(perp[t, i]) else 0.0
                    if px <= 0.0:
                        continue
                    old_q = trend[s]["qty"] if s in trend else 0.0
                    new_q = float(targets.get(s, 0.0))
                    if new_q == old_q:
                        continue
                    fees += abs(new_q - old_q) * px * cost_perp
                    if new_q == 0.0:
                        trend.pop(s, None)
                    else:
                        trend[s] = {"qty": new_q, "entry": px, "bar": float(t)}
                if trend:
                    gross_now = sum(
                        abs(p["qty"]) * float(perp[t, sidx[s]])
                        for s, p in trend.items()
                        if np.isfinite(perp[t, sidx[s]])
                    )
                    trend_gross_sum += gross_now
                    trend_gross_n += 1
                    # Honest leverage telemetry: trend notional also sits on the futures
                    # wallet, so the peak-leverage report must include it.
                    fut_equity = float(profile.futures_wallet_usd)
                    if fut_equity > 0.0:
                        max_lev = max(
                            max_lev,
                            (carry_notional_now + overlay_gross_now + reversal_gross_now + gross_now)
                            / fut_equity,
                        )

        # ---- Sprint 27: funding-conditioned reversal sleeve (dollar-neutral, key SYM|reversal)
        if reversal_enabled and traded and reversal_cfg is not None:
            if (t - tstart) % trend_rebalance_bars == 0:
                lo_r = max(0, t - _REVERSAL_WINDOW_BARS)
                window_r = perp_frame.iloc[lo_r : t + 1]
                rev_w = compute_funding_conditioned_reversal_weights(
                    window_r,
                    bar_index=len(window_r) - 1,
                    funding_bps=dict(zip(syms, f_ewma.tolist())),
                    config=reversal_cfg,
                )
                prices_r = {
                    s: float(perp[t, sidx[s]]) for s in syms if np.isfinite(perp[t, sidx[s]])
                }
                carry_n = sum(
                    p["qty"] * perp[t, sidx[s]]
                    for s, p in carry.items()
                    if np.isfinite(perp[t, sidx[s]])
                )
                ov_n = sum(abs(float(p.get("usd", 0.0))) for p in overlay.values())
                tr_n = sum(
                    abs(p["qty"]) * perp[t, sidx[s]]
                    for s, p in trend.items()
                    if np.isfinite(perp[t, sidx[s]])
                )
                cap_r = max(
                    0.0,
                    float(profile.futures_wallet_usd) * float(profile.max_futures_leverage)
                    - carry_n - ov_n - tr_n,
                )
                want_r = float(sum(abs(w) for w in rev_w.values())) * cap_usd
                if want_r > 0.0 and want_r > cap_r:
                    rev_w = {s: float(w) * (cap_r / want_r) for s, w in rev_w.items()}
                rev_q = reversal_weights_to_target_qty(
                    rev_w, prices_r, cap_usd, TOURNAMENT_LOT_STEPS, min_notional_usd=10.0,
                )
                imbalance = abs(sum(q * prices_r[s] for s, q in rev_q.items() if s in prices_r))
                max_reversal_imbalance = max(max_reversal_imbalance, imbalance)
                if imbalance > _REVERSAL_MAX_IMBALANCE_USD:
                    reversal_skipped_imbalance += 1
                    rev_q = {}
                    rev_w = {}
                if rev_q or reversal:
                    new_keys = {f"{s}|reversal": q for s, q in rev_q.items() if q != 0.0}
                    changed = False
                    for key in sorted(set(reversal) | set(new_keys)):
                        s_r = key.split("|")[0]
                        px_r = float(perp[t, sidx[s_r]]) if np.isfinite(perp[t, sidx[s_r]]) else 0.0
                        if px_r <= 0.0:
                            continue
                        old_q = reversal[key]["qty"] if key in reversal else 0.0
                        new_q = float(new_keys.get(key, 0.0))
                        if new_q == old_q:
                            continue
                        fees += abs(new_q - old_q) * px_r * cost_perp
                        changed = True
                        if new_q == 0.0:
                            reversal.pop(key, None)
                        else:
                            reversal[key] = {"qty": new_q, "entry": px_r, "bar": float(t)}
                    if changed:
                        reversal_rebalances += 1
                        rev_gross = sum(
                            abs(p["qty"]) * float(perp[t, sidx[k_r.split("|")[0]]])
                            for k_r, p in reversal.items()
                            if np.isfinite(perp[t, sidx[k_r.split("|")[0]]])
                        )
                        if float(profile.futures_wallet_usd) > 0.0:
                            max_lev = max(
                                max_lev,
                                (carry_n + ov_n + tr_n + rev_gross) / float(profile.futures_wallet_usd),
                            )

        k = t - start
        if traded:
            eq[k] = cap_usd + funding_pnl + basis_pnl + overlay_pnl + trend_pnl + reversal_pnl - fees
            deployed = {
                s: p["qty"] * spot[t, sidx[s]] for s, p in carry.items() if np.isfinite(spot[t, sidx[s]])
            }
            cum_yield += compute_step_passive_yield_usd(profile, 1.0, deployed)["total_passive_yield_usd"]
            slots_sum += len(carry)
        cy[k] = cum_yield

    idx_slice = index[start:end]
    equity_pure = pd.Series(eq, index=idx_slice)
    cum_series = pd.Series(cy, index=idx_slice)
    n_traded = end - tstart
    return {
        "equity_pure": equity_pure,
        "cumulative_passive_yield_usd": cum_series,
        "equity_yield_stacked": equity_pure + cum_series,
        "funding_pnl_usd": float(funding_pnl),
        "basis_pnl_usd": float(basis_pnl),
        "overlay_pnl_usd": float(overlay_pnl),
        "trend_pnl_usd": float(trend_pnl),
        "short_trend_pnl_usd": float(short_trend_pnl),
        "long_trend_pnl_usd": float(long_trend_pnl),
        "reversal_pnl_usd": float(reversal_pnl),
        "accounting_delta_usd": float(
            round(
                (float(eq[-1]) - cap_usd)
                - (
                    funding_pnl + basis_pnl + overlay_pnl
                    + long_trend_pnl + short_trend_pnl + reversal_pnl - fees
                ),
                6,
            )
            + 0.0
        ),
        "reversal_rebalances": int(reversal_rebalances),
        "reversal_skipped_imbalance": int(reversal_skipped_imbalance),
        "max_reversal_imbalance_usd": float(max_reversal_imbalance),
        "trend_enabled": bool(trend_enabled),
        "circuit_breaker_triggers": int(trend_cb_triggers),
        "circuit_breaker_soft_activations": int(trend_cb_soft_activations),
        "circuit_breaker_bars_soft": int(trend_bars_soft),
        "circuit_breaker_bars_hard": int(trend_bars_hard),
        "final_circuit_breaker_state": str(trend_cb_state),
        "avg_trend_gross_usd": float(trend_gross_sum / trend_gross_n) if trend_gross_n else 0.0,
        "fees_usd": float(fees),
        "net_pnl_usd": float(eq[-1] - cap_usd),
        "max_observed_futures_leverage": float(max_lev),
        "cap_bind_count": int(cap_binds),
        "avg_carry_slots": float(slots_sum / n_traded) if n_traded > 0 else 0.0,
        "diagnostics": {
            "trade_start_idx": tstart,
            "n_carry_opens": n_open,
            "n_carry_closes": n_close,
            "n_carry_resizes": n_resize,
            "n_overlay_rebalances": n_ov_rebal,
            "open_carry_symbols": sorted(carry),
            "open_overlay_symbols": sorted(overlay),
            "open_trend_symbols": sorted(trend),
            "open_reversal_symbols": sorted(reversal),
        },
    }


def _metrics(equity: pd.Series, capital: float) -> dict[str, float]:
    returns = equity.pct_change().fillna(0.0)
    seg = segment_metrics(returns, periods_per_year=8760, initial_capital=capital)
    apr = float(seg["annualized_return_pct"])
    maxdd = float(seg["maxdd_pct"])
    downside = returns[returns < 0.0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    mean_ret = float(returns.mean()) if len(returns) else 0.0
    sortino = float(mean_ret / downside_std * math.sqrt(8760.0)) if downside_std > 0.0 else 0.0
    calmar = float(apr / maxdd) if maxdd > 0.0 else 0.0
    return {
        "apr": apr,
        "sharpe": float(seg["sharpe"]),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "maxdd": maxdd,
        "ret": float(seg["net_return_pct"]),
        "pnl": float(equity.iloc[-1] - capital),
    }


def evaluate_window(
    profile: StrategyProfileConfig,
    market: dict[str, Any],
    start_idx: int,
    end_idx: int,
    trade_start_idx: int,
) -> dict[str, Any]:
    sim = simulate_profile(
        profile, market, start_idx=start_idx, end_idx=end_idx, trade_start_idx=trade_start_idx,
    )
    off = int(trade_start_idx) - int(start_idx)
    # Sprint 25 fix: start one bar BEFORE the first trade so the initial entry costs are
    # inside the first return instead of being silently dropped by pct_change().
    base_off = max(0, off - 1)
    pure_eq = sim["equity_pure"].iloc[base_off:]
    stack_eq = sim["equity_yield_stacked"].iloc[base_off:]
    pm = _metrics(pure_eq, profile.capital_usd)
    ym = _metrics(stack_eq, profile.capital_usd)
    return {
        "pure_trading_net_apr_pct": pm["apr"],
        "pure_trading_sharpe": pm["sharpe"],
        "pure_trading_sortino": pm["sortino"],
        "pure_trading_calmar": pm["calmar"],
        "pure_trading_max_dd_pct": pm["maxdd"],
        "pure_trading_net_pnl_usd": pm["pnl"],
        "pure_trading_net_return_pct": pm["ret"],
        "yield_stacked_net_apr_pct": ym["apr"],
        "yield_stacked_sharpe": ym["sharpe"],
        "yield_stacked_sortino": ym["sortino"],
        "yield_stacked_calmar": ym["calmar"],
        "yield_stacked_max_dd_pct": ym["maxdd"],
        "yield_stacked_net_pnl_usd": ym["pnl"],
        "yield_stacked_net_return_pct": ym["ret"],
        "passive_yield_usd": float(sim["cumulative_passive_yield_usd"].iloc[-1]),
        "funding_pnl_usd": sim["funding_pnl_usd"],
        "basis_pnl_usd": sim["basis_pnl_usd"],
        "overlay_pnl_usd": sim["overlay_pnl_usd"],
        "trend_pnl_usd": sim["trend_pnl_usd"],
        "short_trend_pnl_usd": sim["short_trend_pnl_usd"],
        "long_trend_pnl_usd": sim["long_trend_pnl_usd"],
        "reversal_pnl_usd": sim["reversal_pnl_usd"],
        "accounting_delta_usd": sim["accounting_delta_usd"],
        "reversal_rebalances": sim["reversal_rebalances"],
        "reversal_skipped_imbalance": sim["reversal_skipped_imbalance"],
        "max_reversal_imbalance_usd": sim["max_reversal_imbalance_usd"],
        "circuit_breaker_triggers": sim["circuit_breaker_triggers"],
        "circuit_breaker_soft_activations": sim["circuit_breaker_soft_activations"],
        "circuit_breaker_bars_soft": sim["circuit_breaker_bars_soft"],
        "circuit_breaker_bars_hard": sim["circuit_breaker_bars_hard"],
        "final_circuit_breaker_state": sim["final_circuit_breaker_state"],
        "avg_trend_gross_usd": sim["avg_trend_gross_usd"],
        "fees_usd": sim["fees_usd"],
        "max_observed_futures_leverage": sim["max_observed_futures_leverage"],
        "cap_bind_count": sim["cap_bind_count"],
        "avg_carry_slots": sim["avg_carry_slots"],
        "start": pure_eq.index[0],
        "end": pure_eq.index[-1],
        "n_bars": int(len(pure_eq)),
    }


def run_tournament(
    profiles: dict[str, StrategyProfileConfig],
    market: dict[str, Any],
    *,
    holdout_days: int = 181,
    warmup_hours: int = 720,
) -> dict[str, Any]:
    index = market["index"]
    n = len(index)
    holdout_start = index[-1] - pd.Timedelta(days=holdout_days)
    h_trade = int(index.searchsorted(holdout_start, side="left"))
    h_start = max(0, int(index.searchsorted(holdout_start - pd.Timedelta(hours=warmup_hours), side="left")))
    if h_trade >= n - 1:
        raise ValueError("holdout_days qua ngan so voi du lieu")
    slices = [s for s in np.array_split(np.arange(n), 7) if len(s) > 0]

    out: dict[str, Any] = {}
    for name in sorted(profiles, key=lambda k: list(profiles).index(k)):
        prof = profiles[name]
        LOGGER.info("profile %s: full", name)
        full = evaluate_window(prof, market, 0, n, 0)
        LOGGER.info("profile %s: holdout", name)
        hold = evaluate_window(prof, market, h_start, n, h_trade)
        semi = []
        for sl in slices:
            s0, e0 = int(sl[0]), int(sl[-1]) + 1
            semi.append(evaluate_window(prof, market, max(0, s0 - warmup_hours), e0, s0))
        positive_periods = sum(1 for w in semi if w["pure_trading_net_return_pct"] > 0.0)
        n_periods = len(semi)
        worst_period = min((w["pure_trading_net_return_pct"] for w in semi), default=0.0)
        out[name] = {
            "full": full,
            "holdout": hold,
            "semi_annual": semi,
            "semi_annual_consistency": {
                "positive_periods": int(positive_periods),
                "n_periods": int(n_periods),
                "win_rate_pct": float(positive_periods / n_periods * 100.0) if n_periods else 0.0,
                "worst_period_pct": float(worst_period),
            },
        }

    names = list(out)
    best_pure = max(names, key=lambda k: out[k]["full"]["pure_trading_net_apr_pct"]) if names else None
    best_yield = max(names, key=lambda k: out[k]["full"]["yield_stacked_net_apr_pct"]) if names else None
    return {
        "profiles": out,
        "data_range": {"start": index[0], "end": index[-1], "n_bars": n},
        "holdout_days": int(holdout_days),
        "warmup_hours": int(warmup_hours),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_profile_by_pure_apr": best_pure,
        "best_profile_by_yield_stacked_apr": best_yield,
        "evidence_schema_version": "sprint26",
        "circuit_breaker_policy": {
            "soft_dd_threshold_pct": 0.06,
            "soft_dd_scale": 0.50,
            "hard_dd_threshold_pct": 0.10,
            "hard_dd_cooldown_bars": 168,
            "applies_to": "trend_sleeve_and_overlay (carry market-neutral khong bi cat)",
        },
    }


def format_tournament_table(payload: dict[str, Any]) -> str:
    cols = [
        ("Profile", 26), ("Window", 8), ("Pure APR %", 11), ("Pure Sharpe", 11),
        ("Pure MaxDD %", 12), ("Pure Calmar", 11), ("Pure Sortino", 12),
        ("Pure Net PnL $", 14), ("Yield APR %", 11),
        ("Yield Sharpe", 12), ("Yield MaxDD %", 13), ("Yield Net PnL $", 15),
        ("Funding $", 10), ("Overlay $", 10), ("Fees $", 9), ("Peak Fut Lev x", 14),
        ("Cap Binds", 9), ("Trend $", 9), ("Short $", 9), ("Long $", 9), ("Rev $", 9),
        ("Delta", 10), ("CB H/S", 7),
    ]
    sep = "+" + "+".join("-" * (w + 2) for _, w in cols) + "+"

    def line(vals: list[str]) -> str:
        return "| " + " | ".join(v.rjust(w) if i > 1 else v.ljust(w)
                                 for i, (v, (_, w)) in enumerate(zip(vals, cols))) + " |"

    rows = [sep, line([c for c, _ in cols]), sep]
    for name, res in payload["profiles"].items():
        for label, key in (("FULL", "full"), ("HOLDOUT", "holdout")):
            r = res[key]
            rows.append(line([
                name, label,
                f"{r['pure_trading_net_apr_pct']:.2f}", f"{r['pure_trading_sharpe']:.2f}",
                f"{r['pure_trading_max_dd_pct']:.2f}",
                f"{r.get('pure_trading_calmar', 0.0):.2f}", f"{r.get('pure_trading_sortino', 0.0):.2f}",
                f"{r['pure_trading_net_pnl_usd']:.2f}",
                f"{r['yield_stacked_net_apr_pct']:.2f}", f"{r['yield_stacked_sharpe']:.2f}",
                f"{r['yield_stacked_max_dd_pct']:.2f}", f"{r['yield_stacked_net_pnl_usd']:.2f}",
                f"{r['funding_pnl_usd']:.2f}", f"{r['overlay_pnl_usd']:.2f}", f"{r['fees_usd']:.2f}",
                f"{r['max_observed_futures_leverage']:.2f}", str(int(r["cap_bind_count"])),
                f"{r.get('trend_pnl_usd', 0.0):.2f}",
                f"{r.get('short_trend_pnl_usd', 0.0):.2f}",
                f"{r.get('long_trend_pnl_usd', 0.0):.2f}",
                f"{r.get('reversal_pnl_usd', 0.0):.2f}",
                f"{r.get('accounting_delta_usd', 0.0):.3e}",
                # "hard triggers / soft activations" so a throttled run can never be
                # misread as "the breaker never fired".
                f"{int(r.get('circuit_breaker_triggers', 0))}/{int(r.get('circuit_breaker_soft_activations', 0))}",
            ]))
        rows.append(sep)
    rows.append("")
    rows.append("Semi-annual pure net return % (7 windows):")
    for name, res in payload["profiles"].items():
        strip = " ".join(f"{w['pure_trading_net_return_pct']:8.2f}" for w in res["semi_annual"])
        rows.append(f"  {name:<26} {strip}")
    rows.append("")
    rows.append("Semi-annual consistency (pure):")
    for name, res in payload["profiles"].items():
        c = res.get("semi_annual_consistency", {})
        rows.append(
            f"  {name:<26} positive {int(c.get('positive_periods', 0))}/{int(c.get('n_periods', 0))}"
            f"  win_rate {c.get('win_rate_pct', 0.0):6.2f}%"
            f"  worst_6m {c.get('worst_period_pct', 0.0):8.2f}%"
        )
    rows.append("")
    rows.append(f"Best by pure APR: {payload['best_profile_by_pure_apr']}")
    rows.append(f"Best by yield-stacked APR: {payload['best_profile_by_yield_stacked_apr']}")

    fs = payload.get("fee_sensitivity")
    if isinstance(fs, dict):
        taker_costs = payload.get("costs", {}).get("taker", {})
        rows.append("")
        rows.append(
            "Fee sensitivity (roundtrip = 2 x one-way): taker "
            f"{taker_costs.get('roundtrip_bps', 0.0):.1f} bps "
            f"(one-way {taker_costs.get('one_way_bps', 0.0):.1f}) vs BNB/maker "
            f"{fs.get('roundtrip_bps', 0.0):.1f} bps "
            f"(one-way {fs.get('spot_cost_bps', 0.0) + fs.get('perp_cost_bps', 0.0):.1f})"
        )
        cols2 = [
            ("Profile", 26), ("Window", 8), ("Pure APR taker %", 16),
            ("Pure APR maker %", 16), ("Delta bps", 10), ("PnL taker $", 12),
            ("PnL maker $", 12), ("Fees taker $", 12), ("Fees maker $", 12),
        ]
        sep2 = "+" + "+".join("-" * (w + 2) for _, w in cols2) + "+"

        def line2(vals: list[str]) -> str:
            return "| " + " | ".join(
                v.rjust(w) if i > 1 else v.ljust(w)
                for i, (v, (_, w)) in enumerate(zip(vals, cols2))
            ) + " |"

        rows.extend([sep2, line2([c for c, _ in cols2]), sep2])
        base_profiles = payload["profiles"]
        for name, res in fs.get("profiles", {}).items():
            ref = base_profiles.get(name, {})
            for label, key in (("FULL", "full"), ("HOLDOUT", "holdout")):
                if key not in res or key not in ref:
                    continue
                r_maker, r_taker = res[key], ref[key]
                delta_bps = (r_maker["pure_trading_net_apr_pct"] - r_taker["pure_trading_net_apr_pct"]) * 100.0
                rows.append(line2([
                    name, label,
                    f"{r_taker['pure_trading_net_apr_pct']:.2f}",
                    f"{r_maker['pure_trading_net_apr_pct']:.2f}",
                    f"{delta_bps:+.1f}",
                    f"{r_taker['pure_trading_net_pnl_usd']:.2f}",
                    f"{r_maker['pure_trading_net_pnl_usd']:.2f}",
                    f"{r_taker['fees_usd']:.2f}",
                    f"{r_maker['fees_usd']:.2f}",
                ]))
        rows.append(sep2)
    return "\n".join(rows)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.holdout_days <= 0 or args.warmup_hours < 0:
        print("holdout-days phai > 0 va warmup-hours >= 0", file=sys.stderr)
        return 2
    available = build_tournament_profiles()
    names = [x.strip() for x in args.profiles.split(",") if x.strip()]
    unknown = [x for x in names if x not in available]
    if not names or unknown:
        print(f"Profile khong hop le: {unknown or names}. Hop le: {sorted(available)}", file=sys.stderr)
        return 2
    profiles = {x: available[x] for x in names}

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute() and not raw_dir.exists():
        raw_dir = _ROOT / raw_dir
    LOGGER.info("carry-cost-bps=%.2f (spot=%.2f perp=%.2f)", args.carry_cost_bps,
                args.spot_cost_bps, args.perp_cost_bps)
    aligned = load_multi_aligned_data(raw_dir, list(LIQUID_8_SYMBOLS), args.start, args.end)
    market = build_market_matrices(aligned)
    market["spot_cost_bps"] = float(args.spot_cost_bps)
    market["perp_cost_bps"] = float(args.perp_cost_bps)
    if args.carry_cost_bps is not None:
        market["carry_cost_bps"] = float(args.carry_cost_bps)

    payload = run_tournament(
        profiles, market, holdout_days=args.holdout_days, warmup_hours=args.warmup_hours,
    )
    taker_one_way = float(args.spot_cost_bps) + float(args.perp_cost_bps)
    maker_one_way = float(args.sens_spot_cost_bps) + float(args.sens_perp_cost_bps)
    payload["costs"] = {
        "taker": {
            "spot_cost_bps": float(args.spot_cost_bps),
            "perp_cost_bps": float(args.perp_cost_bps),
            "carry_cost_bps": float(args.carry_cost_bps),
            "one_way_bps": taker_one_way,
            "roundtrip_bps": 2.0 * taker_one_way,
        },
        "maker": {
            "spot_cost_bps": float(args.sens_spot_cost_bps),
            "perp_cost_bps": float(args.sens_perp_cost_bps),
            "carry_cost_bps": float(args.sens_carry_cost_bps),
            "one_way_bps": maker_one_way,
            "roundtrip_bps": 2.0 * maker_one_way,
        },
    }

    sens_names = [x.strip() for x in args.sensitivity_profiles.split(",") if x.strip()]
    sens_profiles = {n: available[n] for n in sens_names if n in available}
    if sens_profiles:
        market_maker = dict(market)
        market_maker["spot_cost_bps"] = float(args.sens_spot_cost_bps)
        market_maker["perp_cost_bps"] = float(args.sens_perp_cost_bps)
        market_maker["carry_cost_bps"] = float(args.sens_carry_cost_bps)
        LOGGER.info("sensitivity run (maker) cho: %s", list(sens_profiles))
        sens_payload = run_tournament(
            sens_profiles, market_maker,
            holdout_days=args.holdout_days, warmup_hours=args.warmup_hours,
        )
        payload["fee_sensitivity"] = {
            "spot_cost_bps": float(args.sens_spot_cost_bps),
            "perp_cost_bps": float(args.sens_perp_cost_bps),
            "carry_cost_bps": float(args.sens_carry_cost_bps),
            "roundtrip_bps": 2.0 * maker_one_way,
            "profiles": sens_payload["profiles"],
        }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    print(format_tournament_table(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
