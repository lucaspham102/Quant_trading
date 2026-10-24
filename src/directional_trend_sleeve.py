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
]

YEAR_HOURS = 8760.0


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


@dataclass
class TrendSleeveState:
    """Mutable state tracking active trend positions and peak close prices."""

    peak_close: dict[str, float] = field(default_factory=dict)
    active: dict[str, bool] = field(default_factory=dict)

    def snapshot(self) -> dict:
        """Return a serializable copy of the state."""
        return {
            "peak_close": {str(k): float(v) for k, v in self.peak_close.items()},
            "active": {str(k): bool(v) for k, v in self.active.items()},
        }

    @classmethod
    def from_snapshot(cls, payload: "dict | None") -> "TrendSleeveState":
        """Restore state from a snapshot dictionary."""
        if not isinstance(payload, dict):
            return cls()
        raw_peaks = payload.get("peak_close") or {}
        raw_active = payload.get("active") or {}
        return cls(
            peak_close={
                str(k): float(v)
                for k, v in raw_peaks.items()
                if v is not None and math.isfinite(float(v))
            },
            active={str(k): bool(v) for k, v in raw_active.items()},
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


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False, min_periods=int(span)).mean()


def compute_triple_confirmed_trend_weights(
    prices: "dict[str, pd.Series] | pd.DataFrame",
    bar_index: int,
    config: "TrendSleeveConfig | None" = None,
    state: "TrendSleeveState | None" = None,
    circuit_breaker: "dict | None" = None,
    allow_new_entries: bool = True,
) -> dict[str, float]:
    """Compute strictly causal long-only target weights at `bar_index`."""
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

    px_now_row = frame.iloc[-1]
    sigma_vol_by_sym: dict[str, float] = {}

    for s in frame.columns:
        px_now = float(px_now_row[s])
        if not (math.isfinite(px_now) and px_now > 0.0):
            state.active[s] = False
            state.peak_close.pop(s, None)
            continue

        series = frame[s]
        ema_f = float(_ema(series, cfg.ema_fast_hours).iloc[-1])
        ema_m = float(_ema(series, cfg.ema_med_hours).iloc[-1])
        ema_s = float(_ema(series, cfg.ema_slow_hours).iloc[-1])

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
        if was_active:
            prev_peak = state.peak_close.get(s) or px_now
            peak = max(float(prev_peak), px_now)
            state.peak_close[s] = peak
            trail_hit = (sigma_daily > 0.0) and (
                px_now < peak * (1.0 - cfg.trail_k_sigma * sigma_daily)
            )
            med_hit = (not math.isfinite(ema_m)) or (px_now < ema_m)
            stay = not (trail_hit or med_hit)
        else:
            stay = False

        if was_active and not stay:
            state.active[s] = False
            state.peak_close.pop(s, None)
        elif was_active and stay:
            state.active[s] = True
        elif (not was_active) and allow_new_entries and aligned and conviction and gate:
            state.active[s] = True
            state.peak_close[s] = px_now

    active_symbols = [
        s
        for s in frame.columns
        if bool(state.active.get(s, False)) and s in sigma_vol_by_sym
    ]
    if not active_symbols:
        return {}

    raw_weights: dict[str, float] = {}
    for s in active_symbols:
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

    result: dict[str, float] = {}
    for s in active_symbols:
        weight_i = gross_total * raw_weights[s] / total_raw
        if weight_i > 0.0:
            result[s] = float(weight_i * risk_scale)
    return result


def trend_weights_to_target_qty(
    weights: "dict[str, float]",
    prices: "dict[str, float]",
    capital_usd: float,
    lot_steps: "dict[str, float] | None" = None,
    min_notional_usd: float = 10.0,
  ) -> dict[str, float]:
    """Convert target capital weights into rounded base quantities per symbol."""
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
        notional = max(0.0, w_val) * float(capital_usd)
        step = float(steps.get(sym, 0.001))
        if not math.isfinite(step) or step <= 0.0:
            step = 0.001
        qty = round(math.floor(notional / px / step + 1e-12) * step, 8)
        if qty * px < float(min_notional_usd):
            qty = 0.0
        out[sym] = float(qty)
    return out
