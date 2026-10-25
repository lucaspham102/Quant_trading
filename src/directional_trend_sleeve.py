"""Sprint 26 triple-confirmed trend sleeve + 2-tier drawdown circuit breaker.

Strictly causal: moi tin hieu tai bar t chi dung du lieu <= bar t.
Module thuan tinh toan, khong side-effect khi import, chi dung stdlib + numpy + pandas.
"""

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

__all__ = [
    "DrawdownCircuitBreakerConfig",
    "evaluate_drawdown_circuit_breaker",
    "TrendSleeveConfig",
    "TrendSleeveState",
    "compute_triple_confirmed_trend_weights",
    "trend_weights_to_target_qty",
    "annualized_realized_vol",
    "kaufman_efficiency_ratio",
]

YEAR_HOURS = 8760.0
# Sprint 27: horizon 14 ngay (336h) cho Z_short = 0.6*Z_14d + 0.4*Z_21d.
SHORT_FAST_HORIZON_HOURS = 336


@dataclass(frozen=True)
class DrawdownCircuitBreakerConfig:
    """Configuration for the 2-tier portfolio drawdown circuit breaker."""

    soft_dd_threshold_pct: float = 0.06
    soft_dd_scale: float = 0.50
    hard_dd_threshold_pct: float = 0.10
    hard_dd_cooldown_bars: int = 168

    def __post_init__(self) -> None:
        if not (0.0 <= self.soft_dd_threshold_pct < self.hard_dd_threshold_pct):
            raise ValueError(
                "soft_dd_threshold_pct must be in [0, hard_dd_threshold_pct), "
                f"got {self.soft_dd_threshold_pct!r}"
            )
        if not (0.0 < self.hard_dd_threshold_pct <= 1.0):
            raise ValueError(
                "hard_dd_threshold_pct must be in (soft_dd_threshold_pct, 1.0], "
                f"got {self.hard_dd_threshold_pct!r}"
            )
        if not (0.0 <= self.soft_dd_scale <= 1.0):
            raise ValueError(
                f"soft_dd_scale must be in [0.0, 1.0], got {self.soft_dd_scale!r}"
            )
        if self.hard_dd_cooldown_bars < 0:
            raise ValueError(
                "hard_dd_cooldown_bars must be >= 0, "
                f"got {self.hard_dd_cooldown_bars!r}"
            )


def evaluate_drawdown_circuit_breaker(
    current_equity: float,
    peak_equity: float,
    bars_since_hard_stop: "int | None",
    cfg: "DrawdownCircuitBreakerConfig | None" = None,
) -> dict:
    """Evaluate 2-tier drawdown circuit breaker state and risk scaling factor."""
    cfg = cfg or DrawdownCircuitBreakerConfig()
    cur = float(current_equity)
    peak = float(peak_equity)
    if not (math.isfinite(cur) and math.isfinite(peak)):
        cur = 1.0
        peak = 1.0
    current_dd_pct = max(0.0, (peak - cur) / max(peak, 1e-9))
    if current_dd_pct >= cfg.hard_dd_threshold_pct or (
        bars_since_hard_stop is not None
        and bars_since_hard_stop < cfg.hard_dd_cooldown_bars
    ):
        return {
            "risk_scale": 0.0,
            "state": "HARD_STOP_LOCKED",
            "current_dd_pct": float(current_dd_pct),
        }
    elif current_dd_pct >= cfg.soft_dd_threshold_pct:
        return {
            "risk_scale": float(cfg.soft_dd_scale),
            "state": "SOFT_THROTTLE",
            "current_dd_pct": float(current_dd_pct),
        }
    else:
        return {
            "risk_scale": 1.0,
            "state": "NORMAL",
            "current_dd_pct": float(current_dd_pct),
        }


@dataclass(frozen=True)
class TrendSleeveConfig:
    """Configuration for the triple-confirmed directional trend sleeve."""

    capital_usd: float = 1000.0
    ema_fast_hours: int = 168
    ema_med_hours: int = 504
    ema_slow_hours: int = 1440
    vol_window_hours: int = 336
    short_horizon_hours: int = 504
    long_horizon_hours: int = 1440
    z_entry: float = 1.15
    z_short_weight: float = 0.6
    z_long_weight: float = 0.4
    trail_k_sigma: float = 3.0
    target_vol_ann: float = 0.12
    max_gross_pct: float = 0.55
    rebalance_every_hours: int = 24
    use_btc_gate: bool = True
    btc_symbol: str = "BTC/USDT:USDT"
    min_history_hours: int = 1440

    # ---- Sprint 27: bidirectional trend (Long bull + funding-positive Bear short).
    # Toan bo cac field duoi day CHI co tac dung khi allow_short_trend=True, nen
    # allow_short_trend=False giu nguyen hanh vi long-only cua Sprint 26 bit-for-bit.
    allow_short_trend: bool = False
    z_short_entry: float = 1.10
    trail_k_sigma_short: float = 2.2
    funding_long_veto_bps: float = 2.5
    funding_short_entry_floor_bps: float = -0.25
    funding_short_exit_bps: float = -0.50
    btc_bear_return_hours: int = 504

    # ---- Sprint 28: Macro Efficiency & Breadth Regime Filter (anti-whipsaw).
    # Mac dinh False de khong lam doi ket qua cac profile cu.
    require_macro_regime_filter: bool = False
    short_gross_scale: "float | None" = None
    er_window_hours: int = 336
    er_slow_window_hours: int = 720
    min_bull_breadth: float = 0.375
    min_bear_breadth: float = 0.75
    short_exit_bear_breadth: float = 0.50
    min_long_er: float = 0.16
    min_btc_bear_er: float = 0.20
    min_btc_bear_er_slow: float = 0.20

    def __post_init__(self) -> None:
        if not self.capital_usd > 0:
            raise ValueError(f"capital_usd must be > 0, got {self.capital_usd!r}")
        if not (0 < self.ema_fast_hours < self.ema_med_hours < self.ema_slow_hours):
            raise ValueError(
                "EMA hours must satisfy 0 < ema_fast_hours < ema_med_hours < ema_slow_hours, "
                f"got ({self.ema_fast_hours}, {self.ema_med_hours}, {self.ema_slow_hours})"
            )
        if not self.vol_window_hours > 0:
            raise ValueError(
                f"vol_window_hours must be > 0, got {self.vol_window_hours!r}"
            )
        if not self.short_horizon_hours > 0:
            raise ValueError(
                f"short_horizon_hours must be > 0, got {self.short_horizon_hours!r}"
            )
        if self.long_horizon_hours < self.short_horizon_hours:
            raise ValueError(
                "long_horizon_hours must be >= short_horizon_hours, "
                f"got {self.long_horizon_hours!r} < {self.short_horizon_hours!r}"
            )
        if not (0.0 < self.target_vol_ann <= 2.0):
            raise ValueError(
                f"target_vol_ann must be in (0, 2.0], got {self.target_vol_ann!r}"
            )
        if not (0.0 < self.max_gross_pct <= 1.0):
            raise ValueError(
                f"max_gross_pct must be in (0, 1.0], got {self.max_gross_pct!r}"
            )
        if self.rebalance_every_hours < 1:
            raise ValueError(
                "rebalance_every_hours must be >= 1, "
                f"got {self.rebalance_every_hours!r}"
            )
        if not self.trail_k_sigma > 0:
            raise ValueError(
                f"trail_k_sigma must be > 0, got {self.trail_k_sigma!r}"
            )
        if self.min_history_hours < self.ema_slow_hours:
            raise ValueError(
                "min_history_hours must be >= ema_slow_hours, "
                f"got {self.min_history_hours!r} < {self.ema_slow_hours!r}"
            )
        if self.allow_short_trend:
            if not self.z_short_entry > 0.0:
                raise ValueError(
                    f"z_short_entry must be > 0, got {self.z_short_entry!r}"
                )
            if not self.trail_k_sigma_short > 0.0:
                raise ValueError(
                    "trail_k_sigma_short must be > 0, "
                    f"got {self.trail_k_sigma_short!r}"
                )
            if not self.btc_bear_return_hours > 0:
                raise ValueError(
                    "btc_bear_return_hours must be > 0, "
                    f"got {self.btc_bear_return_hours!r}"
                )
            if not (self.funding_short_entry_floor_bps > self.funding_short_exit_bps):
                raise ValueError(
                    "funding_short_entry_floor_bps must be > funding_short_exit_bps, "
                    f"got ({self.funding_short_entry_floor_bps!r}, "
                    f"{self.funding_short_exit_bps!r})"
                )
        if self.short_gross_scale is None:
            object.__setattr__(
                self,
                "short_gross_scale",
                0.65 if self.require_macro_regime_filter else 1.0,
            )
        if not (0.0 < float(self.short_gross_scale) <= 1.0):
            raise ValueError(
                f"short_gross_scale must be in (0, 1.0], got {self.short_gross_scale!r}"
            )
        if self.er_window_hours < 1 or self.er_slow_window_hours < self.er_window_hours:
            raise ValueError(
                "er_window_hours must be >= 1 and <= er_slow_window_hours, "
                f"got ({self.er_window_hours!r}, {self.er_slow_window_hours!r})"
            )
        if not (0.0 <= self.min_bull_breadth <= 1.0):
            raise ValueError(
                f"min_bull_breadth must be in [0, 1], got {self.min_bull_breadth!r}"
            )
        if not (0.0 <= self.short_exit_bear_breadth <= self.min_bear_breadth <= 1.0):
            raise ValueError(
                "bear breadth thresholds must satisfy "
                "0 <= short_exit_bear_breadth <= min_bear_breadth <= 1, "
                f"got ({self.short_exit_bear_breadth!r}, {self.min_bear_breadth!r})"
            )


@dataclass
class TrendSleeveState:
    """Mutable state tracking active trend positions and peak close prices."""

    peak_close: dict[str, float] = field(default_factory=dict)
    active: dict[str, bool] = field(default_factory=dict)
    # Sprint 27: huong vi the (+1 long, -1 short) va day thap nhat ke tu khi short.
    side: dict[str, int] = field(default_factory=dict)
    trough_close: dict[str, float] = field(default_factory=dict)

    def snapshot(self) -> dict:
        """Return a serializable copy of the state."""
        return {
            "peak_close": {str(k): float(v) for k, v in self.peak_close.items()},
            "active": {str(k): bool(v) for k, v in self.active.items()},
            "side": {str(k): int(v) for k, v in self.side.items()},
            "trough_close": {str(k): float(v) for k, v in self.trough_close.items()},
        }

    @classmethod
    def from_snapshot(cls, payload: "dict | None") -> "TrendSleeveState":
        """Restore state from a snapshot dictionary (Sprint 26 payloads still load)."""
        if not isinstance(payload, dict):
            return cls()
        raw_peaks = payload.get("peak_close") or {}
        raw_active = payload.get("active") or {}
        raw_side = payload.get("side") or {}
        raw_trough = payload.get("trough_close") or {}
        side: dict[str, int] = {}
        for k, v in raw_side.items():
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv in (1, -1):
                side[str(k)] = iv
        return cls(
            peak_close={
                str(k): float(v)
                for k, v in raw_peaks.items()
                if v is not None and math.isfinite(float(v))
            },
            active={str(k): bool(v) for k, v in raw_active.items()},
            side=side,
            trough_close={
                str(k): float(v)
                for k, v in raw_trough.items()
                if v is not None and math.isfinite(float(v))
            },
        )


def annualized_realized_vol(
    prices: "pd.Series",
    window: int = 336,
    bar_hours: float = 1.0,
) -> float:
    """Compute annualized realized volatility from the last `window` price bars."""
    if prices is None or window <= 1 or bar_hours <= 0:
        return 0.0
    series = pd.Series(prices).astype("float64")
    tail = series.iloc[-int(window) :]
    tail = tail[np.isfinite(tail) & (tail > 0.0)]
    if len(tail) < 3:
        return 0.0
    log_rets = np.log(tail).diff().dropna()
    if len(log_rets) < 2:
        return 0.0
    std_val = float(log_rets.std(ddof=1))
    if not math.isfinite(std_val) or std_val <= 0.0:
        return 0.0
    return float(std_val * math.sqrt(YEAR_HOURS) * math.sqrt(float(bar_hours)))


def kaufman_efficiency_ratio(prices: "pd.Series", window: int = 336) -> float:
    """Compute Kaufman Efficiency Ratio ER(w) over the last `window` bars (causal)."""
    w = int(window)
    if prices is None or w <= 0:
        return 0.0
    series = pd.Series(prices).astype("float64")
    if len(series) < w + 1:
        return 0.0
    tail = series.iloc[-(w + 1) :].to_numpy(dtype=float)
    if not np.all(np.isfinite(tail)):
        return 0.0
    net_change = abs(float(tail[-1] - tail[0]))
    path_length = float(np.sum(np.abs(np.diff(tail))))
    if not math.isfinite(path_length) or path_length <= 1e-12:
        return 0.0
    return float(min(1.0, max(0.0, net_change / path_length)))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False, min_periods=int(span)).mean()


def _log_return(series: pd.Series, horizon_hours: int) -> float:
    """Log return over `horizon_hours` bars of an already-truncated series."""
    horizon = int(horizon_hours)
    if horizon <= 0 or len(series) < horizon + 1:
        return 0.0
    px_now = float(series.iloc[-1])
    px_old = float(series.iloc[-1 - horizon])
    if not (math.isfinite(px_now) and math.isfinite(px_old)):
        return 0.0
    if px_now <= 0.0 or px_old <= 0.0:
        return 0.0
    return float(math.log(px_now / px_old))


def _funding_bps_for(funding_bps, sym: str) -> float:
    """Smoothed funding rate (bps/8h) for `sym`; missing/invalid -> 0.0."""
    if funding_bps is None:
        return 0.0
    try:
        if isinstance(funding_bps, pd.Series):
            raw = funding_bps.get(sym)
        elif isinstance(funding_bps, dict):
            raw = funding_bps.get(sym)
        else:
            return 0.0
        if raw is None:
            return 0.0
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return val if math.isfinite(val) else 0.0


def _btc_bear_regime(frame: pd.DataFrame, cfg: TrendSleeveConfig) -> bool:
    """Macro bear gate: BTC duoi EMA trung han VA return 21 ngay am."""
    btc = frame.get(cfg.btc_symbol)
    if btc is None:
        return False
    macro = _btc_macro_state(frame, cfg)
    if macro is None:
        return False
    btc_now, ema_m_btc, ema_s_btc, dd_60d = macro
    return bool(
        (btc_now < ema_s_btc) and (btc_now < ema_m_btc) and (dd_60d >= 0.10)
    )


def _btc_macro_state(frame: pd.DataFrame, cfg: TrendSleeveConfig):
    """(btc_price, ema_med, ema_slow, dd_from_60d_peak) or None when unavailable."""
    btc = frame.get(cfg.btc_symbol)
    if btc is None:
        return None
    need = max(int(cfg.ema_slow_hours), int(cfg.ema_med_hours)) + 1
    if len(btc) < need:
        return None
    btc_now = float(btc.iloc[-1])
    ema_m_btc = float(_ema(btc, cfg.ema_med_hours).iloc[-1])
    ema_s_btc = float(_ema(btc, cfg.ema_slow_hours).iloc[-1])
    if not (
        math.isfinite(btc_now)
        and btc_now > 0.0
        and math.isfinite(ema_m_btc)
        and math.isfinite(ema_s_btc)
    ):
        return None
    idx = len(btc) - 1
    peak_1440 = float(btc.iloc[max(0, idx - 1440) : idx + 1].max())
    dd_60d = (peak_1440 - btc_now) / max(peak_1440, 1e-9)
    return btc_now, ema_m_btc, ema_s_btc, float(dd_60d)


def compute_triple_confirmed_trend_weights(
    prices: "dict[str, pd.Series] | pd.DataFrame",
    bar_index: int,
    config: "TrendSleeveConfig | None" = None,
    state: "TrendSleeveState | None" = None,
    circuit_breaker: "dict | None" = None,
    allow_new_entries: bool = True,
    funding_bps=None,
) -> dict[str, float]:
    """Compute strictly causal target weights at `bar_index`.

    Long-only (weights > 0) khi `config.allow_short_trend = False` - hanh vi Sprint 26
    duoc bao toan bit-for-bit. Khi `allow_short_trend = True`, tra ve ca weight AM cho
    cac vi the Short trend (funding-positive bear breakdown).
    """
    cfg = config or TrendSleeveConfig()
    if state is None:
        state = TrendSleeveState()
    if prices is None or bar_index < 0:
        return {}

    if isinstance(prices, pd.DataFrame):
        prices_dict = {str(col): prices[col] for col in prices.columns}
    elif isinstance(prices, dict):
        prices_dict = {str(k): v for k, v in prices.items()}
    else:
        return {}

    if not prices_dict:
        return {}

    frame = pd.DataFrame(
        {s: pd.Series(v).astype("float64") for s, v in prices_dict.items()}
    )
    if frame.empty:
        return {}

    # Strictly truncate to [0, bar_index] inclusive to guarantee zero lookahead.
    frame = frame.iloc[: int(bar_index) + 1]
    if len(frame) < cfg.min_history_hours:
        return {}

    # Sprint 27: macro bear gate (BTC) - tinh 1 lan cho ca cross-section.
    bear_regime = bool(cfg.allow_short_trend) and _btc_bear_regime(frame, cfg)
    long_macro_ok = True
    if cfg.allow_short_trend:
        macro = _btc_macro_state(frame, cfg)
        if macro is not None:
            long_macro_ok = not ((macro[0] < macro[2]) or (macro[3] >= 0.12))

    px_now_row = frame.iloc[-1]
    sigma_vol_by_sym: dict[str, float] = {}
    emas_by_sym: dict[str, tuple[float, float, float]] = {}
    for s in frame.columns:
        px_s = float(px_now_row[s])
        if not (math.isfinite(px_s) and px_s > 0.0):
            continue
        ser_s = frame[s]
        ef_s = float(_ema(ser_s, cfg.ema_fast_hours).iloc[-1])
        em_s = float(_ema(ser_s, cfg.ema_med_hours).iloc[-1])
        es_s = float(_ema(ser_s, cfg.ema_slow_hours).iloc[-1])
        emas_by_sym[s] = (ef_s, em_s, es_s)

    # Sprint 28: Macro Efficiency & Breadth Regime Filter (causal 100%).
    bull_breadth = 1.0
    bear_breadth = 1.0
    btc_er_336 = 1.0
    btc_smooth_bear = True
    btc_cascade_bear = True
    btc_above_fast = False
    if cfg.require_macro_regime_filter:
        valid_breadth_syms = [
            s
            for s, (_, em_s, es_s) in emas_by_sym.items()
            if math.isfinite(em_s) and math.isfinite(es_s)
        ]
        if valid_breadth_syms:
            n_b = float(len(valid_breadth_syms))
            bull_breadth = (
                sum(
                    1
                    for s in valid_breadth_syms
                    if float(px_now_row[s]) > emas_by_sym[s][1]
                )
                / n_b
            )
            bear_breadth = (
                sum(
                    1
                    for s in valid_breadth_syms
                    if (float(px_now_row[s]) < emas_by_sym[s][1])
                    and (float(px_now_row[s]) < emas_by_sym[s][2])
                )
                / n_b
            )
        else:
            bull_breadth = 0.0
            bear_breadth = 0.0

        btc_ser = frame.get(cfg.btc_symbol)
        macro_btc = _btc_macro_state(frame, cfg)
        btc_dd_ok = (macro_btc is not None) and (macro_btc[3] >= 0.12)
        if btc_ser is not None and cfg.btc_symbol in emas_by_sym:
            btc_px = float(px_now_row[cfg.btc_symbol])
            btc_ef, btc_em, btc_es = emas_by_sym[cfg.btc_symbol]
            btc_er_336 = kaufman_efficiency_ratio(btc_ser, cfg.er_window_hours)
            btc_er_504 = kaufman_efficiency_ratio(btc_ser, cfg.short_horizon_hours)
            btc_er_720 = kaufman_efficiency_ratio(btc_ser, cfg.er_slow_window_hours)
            btc_smooth_bear = bool(
                (btc_er_336 >= cfg.min_btc_bear_er)
                or (btc_er_504 >= cfg.min_btc_bear_er_slow)
                or (btc_er_720 >= cfg.min_btc_bear_er_slow)
            )
            btc_cascade_bear = bool(
                math.isfinite(btc_px)
                and math.isfinite(btc_ef)
                and math.isfinite(btc_em)
                and math.isfinite(btc_es)
                and (btc_px < btc_ef < btc_em < btc_es)
            )
            btc_above_fast = bool(
                (not math.isfinite(btc_px))
                or (not math.isfinite(btc_ef))
                or (btc_px > btc_ef)
            )
        else:
            btc_er_336 = 0.0
            btc_smooth_bear = False
            btc_cascade_bear = False
            btc_above_fast = True

        bear_regime = bool(
            cfg.allow_short_trend
            and (bear_breadth >= cfg.min_bear_breadth)
            and btc_smooth_bear
            and btc_cascade_bear
            and btc_dd_ok
        )

    for s in frame.columns:
        px_now = float(px_now_row[s])
        if not (math.isfinite(px_now) and px_now > 0.0):
            state.active[s] = False
            state.peak_close.pop(s, None)
            state.side.pop(s, None)
            state.trough_close.pop(s, None)
            continue

        series = frame[s]
        ema_f, ema_m, ema_s = emas_by_sym[s]
        er_symbol_336 = (
            kaufman_efficiency_ratio(series, cfg.er_window_hours)
            if cfg.require_macro_regime_filter
            else 1.0
        )

        if len(series) >= cfg.short_horizon_hours:
            idx_short = max(0, len(series) - 1 - int(cfg.short_horizon_hours))
            px_short_ago = float(series.iloc[idx_short])
            if math.isfinite(px_short_ago) and px_short_ago > 0.0:
                r_short = math.log(px_now / px_short_ago)
            else:
                r_short = 0.0
        else:
            r_short = 0.0

        if len(series) >= cfg.long_horizon_hours:
            idx_long = max(0, len(series) - 1 - int(cfg.long_horizon_hours))
            px_long_ago = float(series.iloc[idx_long])
            if math.isfinite(px_long_ago) and px_long_ago > 0.0:
                r_long = math.log(px_now / px_long_ago)
            else:
                r_long = 0.0
        else:
            r_long = 0.0

        sigma_short_ann = annualized_realized_vol(series, cfg.short_horizon_hours)
        sigma_long_ann = annualized_realized_vol(series, cfg.long_horizon_hours)
        sigma_vol = annualized_realized_vol(series, cfg.vol_window_hours)
        # A t-stat needs R and sigma in the SAME horizon units. R is a raw 21d/60d log
        # return, so the annualized vol must be de-annualized back to that horizon
        # (sigma_h = sigma_ann * sqrt(h / 8760)). Using the annualized figure directly
        # makes Z structurally ~0.3 and the sleeve never trades at all.
        sigma_short = sigma_short_ann * math.sqrt(cfg.short_horizon_hours / YEAR_HOURS)
        sigma_long = sigma_long_ann * math.sqrt(cfg.long_horizon_hours / YEAR_HOURS)
        sigma_vol_by_sym[s] = sigma_vol

        sigma_hourly = sigma_vol / math.sqrt(YEAR_HOURS)
        sigma_daily = sigma_hourly * math.sqrt(24.0)

        z_s = r_short / sigma_short if sigma_short > 0.0 else 0.0
        z_l = r_long / sigma_long if sigma_long > 0.0 else 0.0
        z_comp = cfg.z_short_weight * z_s + cfg.z_long_weight * z_l

        # Evidence 1: multi-horizon alignment
        emas_ok = (
            math.isfinite(ema_f) and math.isfinite(ema_m) and math.isfinite(ema_s)
        )
        aligned = bool(
            emas_ok
            and (px_now > ema_f)
            and (ema_f > ema_m)
            and (ema_m > ema_s)
            and (r_short > 0.0)
            and (r_long > 0.0)
        )

        # Evidence 2: risk-adjusted momentum conviction
        conviction = bool(math.isfinite(z_comp) and (z_comp >= cfg.z_entry))

        # Evidence 3: BTC leader gate for altcoins
        gate = True
        if cfg.use_btc_gate and s != cfg.btc_symbol:
            btc = frame.get(cfg.btc_symbol)
            if btc is None or len(btc) < cfg.ema_med_hours:
                gate = False
            else:
                ema_m_btc = float(_ema(btc, cfg.ema_med_hours).iloc[-1])
                btc_now = float(btc.iloc[-1])
                gate = bool(
                    math.isfinite(btc_now)
                    and math.isfinite(ema_m_btc)
                    and (btc_now > ema_m_btc)
                )

        was_active = bool(state.active.get(s, False))
        # Sprint 27: huong vi the. State cu (Sprint 26) khong co `side` -> mac dinh long.
        prev_side = int(state.side.get(s, 1)) if was_active else 0
        if was_active and prev_side not in (1, -1):
            prev_side = 1

        f_bps = _funding_bps_for(funding_bps, s) if bear_regime or cfg.allow_short_trend else 0.0

        # ---- Sprint 27: short-side evidence (chi tinh khi bat bidirectional mode).
        short_aligned = False
        short_conviction = False
        z_short = 0.0
        if cfg.allow_short_trend:
            r_fast = _log_return(series, SHORT_FAST_HORIZON_HOURS)
            sigma_fast_ann = annualized_realized_vol(series, SHORT_FAST_HORIZON_HOURS)
            sigma_fast = sigma_fast_ann * math.sqrt(SHORT_FAST_HORIZON_HOURS / YEAR_HOURS)
            z_fast = r_fast / sigma_fast if sigma_fast > 0.0 else 0.0
            z_short = cfg.z_short_weight * z_fast + cfg.z_long_weight * z_s
            # Pha vo cau truc: gia duoi EMA nhanh < EMA trung han VA 14d/21d deu am.
            short_aligned = bool(
                emas_ok
                and (px_now < ema_f)
                and (ema_f < ema_m)
                and (r_fast < 0.0)
                and (r_short < 0.0)
            )
            short_conviction = bool(
                math.isfinite(z_short) and (z_short <= -cfg.z_short_entry)
            )

        if was_active and prev_side == 1:
            prev_peak = state.peak_close.get(s) or px_now
            peak = max(float(prev_peak), px_now)
            state.peak_close[s] = peak
            trail_hit = (sigma_daily > 0.0) and (
                px_now < peak * (1.0 - cfg.trail_k_sigma * sigma_daily)
            )
            med_hit = (not math.isfinite(ema_m)) or (px_now < ema_m)
            macro_long_exit = bool(
                cfg.require_macro_regime_filter
                and (
                    (bull_breadth < cfg.min_bull_breadth)
                    or ((bull_breadth < 0.50) and (er_symbol_336 < 0.12))
                )
            )
            stay = not (trail_hit or med_hit or macro_long_exit)
            if not stay:
                state.active[s] = False
                state.peak_close.pop(s, None)
                state.side.pop(s, None)
                state.trough_close.pop(s, None)
        elif was_active and prev_side == -1:
            # Short: trailing stop la day thap nhat (trough) va nhanh hon (k=2.2 vs 3.0),
            # dong thoi thoat ngay khi funding am sau (crowded shorts -> squeeze risk).
            prev_trough = state.trough_close.get(s) or px_now
            trough = min(float(prev_trough), px_now)
            state.trough_close[s] = trough
            trail_up = (sigma_daily > 0.0) and (
                px_now > trough * (1.0 + cfg.trail_k_sigma_short * sigma_daily)
            )
            fast_hit = (not math.isfinite(ema_f)) or (px_now > ema_f)
            squeeze_hit = f_bps < cfg.funding_short_exit_bps
            macro_short_exit = bool(
                cfg.require_macro_regime_filter
                and (btc_above_fast or (bear_breadth < cfg.short_exit_bear_breadth))
            )
            stay = not (trail_up or fast_hit or squeeze_hit or macro_short_exit)
            if not stay:
                state.active[s] = False
                state.peak_close.pop(s, None)
                state.side.pop(s, None)
                state.trough_close.pop(s, None)
        elif (not was_active) and allow_new_entries:
            # Funding drag filter cho LONG (chi bat o Sprint 27): tranh mo long khi
            # funding 8h > +2.5 bps (~27%/nam) nghia la phe long dang qua dong.
            long_funding_ok = (not cfg.allow_short_trend) or (
                f_bps <= cfg.funding_long_veto_bps
            )
            btc_emas = emas_by_sym.get(cfg.btc_symbol)
            btc_px_now = (
                float(px_now_row[cfg.btc_symbol])
                if cfg.btc_symbol in frame.columns
                else float("nan")
            )
            btc_super_bull = bool(
                btc_emas is not None
                and math.isfinite(btc_px_now)
                and math.isfinite(btc_emas[0])
                and math.isfinite(btc_emas[1])
                and math.isfinite(btc_emas[2])
                and (btc_px_now > btc_emas[0] > btc_emas[1] > btc_emas[2])
            )
            if not cfg.require_macro_regime_filter:
                long_regime_ok = True
            elif (bull_breadth >= 0.875) and btc_super_bull:
                long_regime_ok = er_symbol_336 >= max(0.08, float(cfg.min_long_er))
            elif bull_breadth >= cfg.min_bull_breadth:
                long_regime_ok = er_symbol_336 >= max(0.16, float(cfg.min_long_er))
            else:
                long_regime_ok = False
            if (
                aligned
                and conviction
                and gate
                and long_funding_ok
                and long_macro_ok
                and long_regime_ok
            ):
                state.active[s] = True
                state.side[s] = 1
                state.peak_close[s] = px_now
                state.trough_close.pop(s, None)
            elif (
                cfg.allow_short_trend
                and bear_regime
                and short_aligned
                and short_conviction
                and (f_bps >= cfg.funding_short_entry_floor_bps)
            ):
                # Anti-squeeze guard: funding phai >= -0.25 bps/8h. Short khi funding
                # con duong/gan 0 nghia la retail van bat day -> vua an gia giam vua an carry.
                state.active[s] = True
                state.side[s] = -1
                state.trough_close[s] = px_now
                state.peak_close.pop(s, None)

    active_symbols = [
        s
        for s in frame.columns
        if bool(state.active.get(s, False)) and s in sigma_vol_by_sym
    ]
    if not active_symbols:
        return {}

    # Sprint 27: tach long/short nhung van giu nguyen thu tu `frame.columns` cho long
    # (de long-only mode ra ket qua giong het Sprint 26).
    side_of: dict[str, int] = {}
    for s in active_symbols:
        side_i = int(state.side.get(s, 1))
        side_of[s] = -1 if side_i == -1 else 1
    long_symbols = [s for s in active_symbols if side_of[s] == 1]
    short_symbols = [s for s in active_symbols if side_of[s] == -1]
    ordered = long_symbols + short_symbols

    raw_weights: dict[str, float] = {}
    for s in ordered:
        strength_i = 1.0 / max(sigma_vol_by_sym[s], 1e-6)
        raw_weights[s] = cfg.target_vol_ann * strength_i

    total_raw = sum(raw_weights.values())
    if total_raw <= 0.0:
        return {}

    gross_total = min(cfg.max_gross_pct, total_raw)
    if isinstance(circuit_breaker, dict):
        raw_scale = float(circuit_breaker.get("risk_scale", 1.0))
        if not math.isfinite(raw_scale):
            raw_scale = 0.0
    else:
        raw_scale = 1.0
    risk_scale = min(1.0, max(0.0, raw_scale))
    short_scale = float(
        cfg.short_gross_scale if cfg.short_gross_scale is not None else 1.0
    )

    result: dict[str, float] = {}
    for s in ordered:
        weight_i = gross_total * raw_weights[s] / total_raw
        if weight_i > 0.0:
            signed_i = weight_i if side_of[s] == 1 else -(weight_i * short_scale)
            result[s] = float(signed_i * risk_scale)
    return result


def trend_weights_to_target_qty(
    weights: "dict[str, float]",
    prices: "dict[str, float]",
    capital_usd: float,
    lot_steps: "dict[str, float] | None" = None,
    min_notional_usd: float = 10.0,
) -> dict[str, float]:
    """Convert target capital weights into rounded SIGNED base quantities per symbol.

    Sprint 27: weight am tao qty am (short). Weight duong giu nguyen ket qua Sprint 26.
    """
    if not weights:
        return {}
    steps = lot_steps or {}
    out: dict[str, float] = {}
    for sym, w in weights.items():
        px_raw = prices.get(sym) if isinstance(prices, dict) else None
        if px_raw is None:
            out[sym] = 0.0
            continue
        px = float(px_raw)
        w_val = float(w)
        if (
            not math.isfinite(px)
            or px <= 0.0
            or not math.isfinite(w_val)
            or capital_usd <= 0.0
        ):
            out[sym] = 0.0
            continue
        step = float(steps.get(sym, 0.001))
        if not math.isfinite(step) or step <= 0.0:
            step = 0.001
        notional = abs(w_val) * float(capital_usd)
        magnitude = round(math.floor(notional / px / step + 1e-12) * step, 8)
        if magnitude * px < float(min_notional_usd):
            out[sym] = 0.0
            continue
        # Weight 0.0 van phai ra qty 0.0 (khong duoc tu bien thanh short lot).
        if w_val == 0.0:
            out[sym] = 0.0
            continue
        out[sym] = float(magnitude if w_val > 0.0 else -magnitude)
    return out
