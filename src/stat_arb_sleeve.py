"""Sleeve stat-arb market-neutral: dao chieu qua da (overextension) co dieu kien funding.

Strictly causal: moi tin hieu tai bar t chi dung du lieu <= bar t.
Module thuan tinh toan, khong side-effect khi import, chi dung stdlib + numpy + pandas.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

__all__ = [
    "ReversalSleeveConfig",
    "compute_funding_conditioned_reversal_weights",
    "reversal_weights_to_target_qty",
]

YEAR_HOURS = 8760.0


@dataclass(frozen=True)
class ReversalSleeveConfig:
    """Configuration for the funding-conditioned overextension reversal sleeve."""

    capital_usd: float = 1000.0
    lookback_hours: int = 120
    vol_window_hours: int = 120
    z_threshold: float = 1.65
    funding_tilt: float = 0.5
    funding_clip_bps: float = 2.0
    min_spread_threshold: float = 0.60
    gross_pct: float = 0.35
    max_pairs: int = 2
    min_leg_usd: float = 10.0

    def __post_init__(self) -> None:
        if not self.capital_usd > 0:
            raise ValueError(f"capital_usd must be > 0, got {self.capital_usd!r}")
        if not self.lookback_hours >= 2:
            raise ValueError(
                f"lookback_hours must be >= 2, got {self.lookback_hours!r}"
            )
        if not self.vol_window_hours >= 2:
            raise ValueError(
                f"vol_window_hours must be >= 2, got {self.vol_window_hours!r}"
            )
        if not self.z_threshold > 0:
            raise ValueError(f"z_threshold must be > 0, got {self.z_threshold!r}")
        if not (0.0 <= self.funding_tilt <= 1.0):
            raise ValueError(
                f"funding_tilt must be in [0, 1], got {self.funding_tilt!r}"
            )
        if not self.funding_clip_bps > 0:
            raise ValueError(
                f"funding_clip_bps must be > 0, got {self.funding_clip_bps!r}"
            )
        if not self.min_spread_threshold >= 0:
            raise ValueError(
                "min_spread_threshold must be >= 0, "
                f"got {self.min_spread_threshold!r}"
            )
        if not (0.0 < self.gross_pct <= 1.0):
            raise ValueError(f"gross_pct must be in (0, 1], got {self.gross_pct!r}")
        if not self.max_pairs >= 1:
            raise ValueError(f"max_pairs must be >= 1, got {self.max_pairs!r}")
        if not self.min_leg_usd >= 0:
            raise ValueError(f"min_leg_usd must be >= 0, got {self.min_leg_usd!r}")


def _funding_lookup(funding_bps: "dict[str, float] | pd.Series | None", sym: str) -> float:
    """Return funding (bps/8h) for `sym`; missing / NaN / invalid -> 0.0."""
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


def compute_funding_conditioned_reversal_weights(
    prices: "dict[str, pd.Series] | pd.DataFrame",
    bar_index: int,
    funding_bps: "dict[str, float] | pd.Series | None" = None,
    config: "ReversalSleeveConfig | None" = None,
) -> dict[str, float]:
    """Compute strictly causal dollar-neutral signed weights (long > 0, short < 0).

    Weight = signed_notional_usd / capital_usd. Returns {} when the sleeve must be flat.
    """
    cfg = config or ReversalSleeveConfig()
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

    # Cat cung tai bar_index (inclusive): dam bao khong lookahead.
    frame = frame.iloc[: int(bar_index) + 1]
    lookback = int(cfg.lookback_hours)
    vol_win = int(cfg.vol_window_hours)
    if len(frame) < max(lookback, vol_win) + 1:
        return {}

    # Buoc b/c: return lookback va sigma tung symbol.
    rets: dict[str, float] = {}
    for s in frame.columns:
        col = frame[s]
        px_now = float(col.iloc[-1])
        px_old = float(col.iloc[-1 - lookback])
        if not (math.isfinite(px_now) and px_now > 0.0):
            continue
        if not (math.isfinite(px_old) and px_old > 0.0):
            continue
        tail = col.iloc[-(vol_win + 1):]
        tail = tail[np.isfinite(tail) & (tail > 0.0)]
        if len(tail) < 3:
            continue
        log_rets = np.log(tail).diff().dropna()
        if len(log_rets) < 2:
            continue
        sigma = float(log_rets.std(ddof=1))
        if not (math.isfinite(sigma) and sigma > 0.0):
            continue
        r_s = math.log(px_now / px_old)
        # Chuan hoa bien dong: sigma 1h phai duoc mo rong ve dung horizon lookback
        # (sigma_h = sigma_1h * sqrt(h)) truoc khi chia, neu khong mot coin vol thap
        # se luon bi coi la "overextended" va z_threshold mat y nghia thong ke.
        sigma_h = sigma * math.sqrt(float(lookback))
        if math.isfinite(r_s) and math.isfinite(sigma_h) and sigma_h > 0.0:
            rets[str(s)] = r_s / sigma_h

    # Buoc d: z cheo (cross-sectional) tren tap symbol con lai.
    if len(rets) < 3:
        return {}
    syms = sorted(rets)
    r_arr = np.array([rets[s] for s in syms], dtype="float64")
    r_std = float(r_arr.std(ddof=1))
    if not (math.isfinite(r_std) and r_std > 0.0):
        return {}
    r_mean = float(r_arr.mean())
    z = {s: (rets[s] - r_mean) / r_std for s in syms}

    # Buoc e: score = -z - tilt * funding. Funding duong (long tra short) + gia da tang
    # qua muc -> short vua duoc dao chieu vua nhan carry; funding am thi nguoc lai.
    clip = float(cfg.funding_clip_bps)
    score: dict[str, float] = {}
    for s in syms:
        f = min(clip, max(-clip, _funding_lookup(funding_bps, s)))
        score[s] = -z[s] - float(cfg.funding_tilt) * f

    # Buoc f: thi truong qua yen (spread score hep) thi dung ngoai.
    spread = max(score.values()) - min(score.values())
    if not (math.isfinite(spread) and spread >= cfg.min_spread_threshold):
        return {}

    # Buoc g: chon ung vien, tie-break theo ten symbol.
    longs = sorted(
        (s for s in syms if score[s] > 0.0 and z[s] <= -cfg.z_threshold),
        key=lambda s: (-score[s], s),
    )
    shorts = sorted(
        (s for s in syms if score[s] < 0.0 and z[s] >= cfg.z_threshold),
        key=lambda s: (score[s], s),
    )
    if not longs or not shorts:
        return {}  # khong bao gio de leg tro troi
    n_pairs = min(int(cfg.max_pairs), len(longs), len(shorts))

    # Buoc h: moi leg cung do lon -> dollar-neutral chinh xac tuyet doi.
    leg_usd = float(cfg.capital_usd) * float(cfg.gross_pct) / (2.0 * n_pairs)
    if not math.isfinite(leg_usd) or leg_usd < cfg.min_leg_usd or leg_usd <= 0.0:
        return {}
    w = leg_usd / float(cfg.capital_usd)

    out: dict[str, float] = {}
    for k in range(n_pairs):
        out[longs[k]] = float(w)
        out[shorts[k]] = float(-w)
    return out


def reversal_weights_to_target_qty(
    weights: "dict[str, float]",
    prices: "dict[str, float]",
    capital_usd: float,
    lot_steps: "dict[str, float] | None" = None,
    min_notional_usd: float = 10.0,
) -> dict[str, float]:
    """Convert signed weights into SIGNED base quantities (short -> negative qty)."""
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
            or not math.isfinite(float(capital_usd))
            or capital_usd <= 0.0
        ):
            out[sym] = 0.0
            continue
        step = float(steps.get(sym, 0.001))
        if not math.isfinite(step) or step <= 0.0:
            step = 0.001
        notional = abs(w_val) * float(capital_usd)
        # Floor tren do lon roi gan lai dau, de short ra qty am.
        mag = round(math.floor(notional / px / step + 1e-12) * step, 8)
        if mag * px < float(min_notional_usd):
            out[sym] = 0.0
            continue
        sign = 1.0 if w_val > 0.0 else -1.0
        out[sym] = float(sign * mag)
    return out
