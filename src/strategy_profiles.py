"""Bo may profile chien luoc cho giai dau toi uu hoa von 1,000 USDT.

Module gom 12 profile dang ky trong PROFILE_REGISTRY:
- default: baseline mainnet 3,000 USDT (dong bang, khong doi).
- canary_400: canary 400 USDT chia vi spot/futures.
- baseline_1k: baseline bao thu 1,000 USDT.
- alpha_boost_1k: toi uu loi nhuan can bang (ung vien chinh).
- turbo_neutral_1k: toi uu market-neutral tin hieu cao.
- carry_max_eta_1k: 100% market-neutral carry hieu suat von cao (Sprint 26).
- hybrid_trend_conservative_1k: carry + trend sleeve bao thu + circuit breaker (Sprint 26).
- hybrid_trend_balanced_1k: carry + trend sleeve can bang + circuit breaker (Sprint 26).
- bi_trend_carry_1k: carry + bidirectional trend sleeve (Sprint 27).
- carry_stat_arb_1k: carry + funding-conditioned reversal sleeve (Sprint 27).
- regime_omni_1k: carry + bidirectional trend + reversal sleeve (Sprint 27).
- flexible_regime_1k: carry + macro efficiency & breadth regime switcher (Sprint 28).

Chi dung thu vien chuan, khong co tac dung phu khi import.
"""

from dataclasses import dataclass, field

__all__ = [
    "StrategyProfileConfig",
    "PROFILE_REGISTRY",
    "get_strategy_profile",
    "compute_funding_tilted_weights",
    "enforce_split_wallet_futures_cap",
    "compute_step_passive_yield_usd",
    "profile_names",
]


@dataclass(frozen=True)
class StrategyProfileConfig:
    name: str
    capital_usd: float
    spot_wallet_usd: float
    futures_wallet_usd: float
    carry_eta: float = 0.80
    carry_top_k: int = 4
    carry_min_hold_bars: int = 63
    carry_entry_bps: float = 0.45
    carry_exit_bps: float = 0.0
    carry_buffer_rank: int = 3
    carry_weighting_mode: str = "equal"
    carry_tilt_alpha: float = 0.0
    carry_max_single_weight: float = 0.55
    overlay_target_vol_ann: float = 0.075
    overlay_gross_pct: float = 0.15
    max_futures_leverage: float = 3.0
    enable_yield_stacking: bool = False
    enable_trend_sleeve: bool = False
    trend_target_vol_ann: float = 0.0
    trend_max_gross_pct: float = 0.0
    trend_rebalance_hours: int = 24
    trend_z_entry: float = 1.15
    # ---- Sprint 27: bidirectional trend, stat-arb reversal, dynamic futures capacity
    allow_short_trend: bool = False
    trend_z_short_entry: float = 1.10
    enable_reversal_sleeve: bool = False
    reversal_gross_pct: float = 0.0
    dynamic_carry_to_alpha_margin: bool = False
    # ---- Sprint 28: Macro Efficiency & Breadth Regime Filter
    require_macro_regime_filter: bool = False
    short_gross_scale: float = 1.0
    soft_dd_threshold_pct: float = 0.06
    hard_dd_threshold_pct: float = 0.10
    bfusd_apy: float = 0.045
    idle_spot_earn_apy: float = 0.035
    liquid_staking_apy_map: dict[str, float] = field(
        default_factory=lambda: {"ETH": 0.032, "SOL": 0.062}
    )

    def __post_init__(self) -> None:
        if not self.capital_usd > 0:
            raise ValueError(f"capital_usd must be > 0, got {self.capital_usd!r}")
        if not self.spot_wallet_usd > 0:
            raise ValueError(f"spot_wallet_usd must be > 0, got {self.spot_wallet_usd!r}")
        if not self.futures_wallet_usd > 0:
            raise ValueError(
                f"futures_wallet_usd must be > 0, got {self.futures_wallet_usd!r}"
            )
        if abs((self.spot_wallet_usd + self.futures_wallet_usd) - self.capital_usd) > 1e-6:
            raise ValueError(
                "spot_wallet_usd + futures_wallet_usd must equal capital_usd: "
                f"{self.spot_wallet_usd} + {self.futures_wallet_usd} != {self.capital_usd}"
            )
        if not 0.0 < self.carry_eta <= 1.0:
            raise ValueError(f"carry_eta must be in (0, 1], got {self.carry_eta!r}")
        if self.carry_top_k < 1:
            raise ValueError(f"carry_top_k must be >= 1, got {self.carry_top_k!r}")
        if not 0.0 <= self.carry_tilt_alpha <= 1.0:
            raise ValueError(
                f"carry_tilt_alpha must be in [0, 1], got {self.carry_tilt_alpha!r}"
            )
        if not 0.0 < self.carry_max_single_weight <= 1.0:
            raise ValueError(
                "carry_max_single_weight must be in (0, 1], "
                f"got {self.carry_max_single_weight!r}"
            )
        if self.carry_weighting_mode not in ("equal", "funding_tilted"):
            raise ValueError(
                "carry_weighting_mode must be 'equal' or 'funding_tilted', "
                f"got {self.carry_weighting_mode!r}"
            )
        if not self.overlay_target_vol_ann > 0:
            raise ValueError(
                f"overlay_target_vol_ann must be > 0, got {self.overlay_target_vol_ann!r}"
            )
        if not 0.0 <= self.overlay_gross_pct <= 1.0:
            raise ValueError(
                f"overlay_gross_pct must be in [0, 1], got {self.overlay_gross_pct!r}"
            )
        if not self.max_futures_leverage > 0:
            raise ValueError(
                f"max_futures_leverage must be > 0, got {self.max_futures_leverage!r}"
            )
        if self.enable_trend_sleeve:
            if not self.trend_target_vol_ann > 0:
                raise ValueError(
                    "trend_target_vol_ann must be > 0 when enable_trend_sleeve=True, "
                    f"got {self.trend_target_vol_ann!r}"
                )
            if not 0.0 < self.trend_max_gross_pct <= 1.0:
                raise ValueError(
                    "trend_max_gross_pct must be in (0, 1], "
                    f"got {self.trend_max_gross_pct!r}"
                )
            if self.trend_rebalance_hours < 1:
                raise ValueError(
                    "trend_rebalance_hours must be >= 1, "
                    f"got {self.trend_rebalance_hours!r}"
                )
            if not 0.0 <= self.soft_dd_threshold_pct < self.hard_dd_threshold_pct <= 1.0:
                raise ValueError(
                    "soft_dd_threshold_pct and hard_dd_threshold_pct must satisfy "
                    "0.0 <= soft < hard <= 1.0, got "
                    f"({self.soft_dd_threshold_pct!r}, {self.hard_dd_threshold_pct!r})"
                )
        if self.allow_short_trend and not self.enable_trend_sleeve:
            raise ValueError(
                "allow_short_trend=True requires enable_trend_sleeve=True"
            )
        if self.allow_short_trend and not self.trend_z_short_entry > 0:
            raise ValueError(
                "trend_z_short_entry must be > 0 when allow_short_trend=True, "
                f"got {self.trend_z_short_entry!r}"
            )
        if self.require_macro_regime_filter and not self.enable_trend_sleeve:
            raise ValueError(
                "require_macro_regime_filter=True requires enable_trend_sleeve=True"
            )
        if not (0.0 < self.short_gross_scale <= 1.0):
            raise ValueError(
                f"short_gross_scale must be in (0, 1], got {self.short_gross_scale!r}"
            )
        if self.enable_reversal_sleeve:
            if not 0.0 < self.reversal_gross_pct <= 1.0:
                raise ValueError(
                    "reversal_gross_pct must be in (0, 1] when "
                    f"enable_reversal_sleeve=True, got {self.reversal_gross_pct!r}"
                )
        elif self.reversal_gross_pct != 0.0:
            raise ValueError(
                "reversal_gross_pct must be 0.0 when enable_reversal_sleeve=False, "
                f"got {self.reversal_gross_pct!r}"
            )
        object.__setattr__(self, "liquid_staking_apy_map", dict(self.liquid_staking_apy_map))


PROFILE_REGISTRY: dict[str, StrategyProfileConfig] = {
    "default": StrategyProfileConfig(
        name="default",
        capital_usd=3000.0,
        spot_wallet_usd=2400.0,
        futures_wallet_usd=600.0,
        carry_eta=0.80,
        carry_top_k=4,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.50,
        overlay_target_vol_ann=0.075,
        overlay_gross_pct=0.15,
        max_futures_leverage=3.0,
        # Sprint 25.1: yield stacking is tracked but never folded into `pure_trading_*`,
        # so reporting it costs nothing and gives the operator full visibility.
        enable_yield_stacking=True,
    ),
    "canary_400": StrategyProfileConfig(
        name="canary_400",
        capital_usd=400.0,
        spot_wallet_usd=260.0,
        futures_wallet_usd=140.0,
        carry_eta=0.65,
        carry_top_k=2,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.60,
        overlay_target_vol_ann=0.075,
        overlay_gross_pct=0.15,
        max_futures_leverage=2.8,
        enable_yield_stacking=False,
    ),
    "baseline_1k": StrategyProfileConfig(
        name="baseline_1k",
        capital_usd=1000.0,
        spot_wallet_usd=650.0,
        futures_wallet_usd=350.0,
        carry_eta=0.65,
        carry_top_k=3,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.50,
        overlay_target_vol_ann=0.075,
        overlay_gross_pct=0.15,
        max_futures_leverage=2.8,
        enable_yield_stacking=True,
    ),
    "alpha_boost_1k": StrategyProfileConfig(
        name="alpha_boost_1k",
        capital_usd=1000.0,
        spot_wallet_usd=650.0,
        futures_wallet_usd=350.0,
        carry_eta=0.65,
        carry_top_k=3,
        carry_min_hold_bars=42,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=2,
        carry_weighting_mode="funding_tilted",
        carry_tilt_alpha=0.50,
        carry_max_single_weight=0.50,
        overlay_target_vol_ann=0.140,
        overlay_gross_pct=0.30,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
    ),
    "turbo_neutral_1k": StrategyProfileConfig(
        name="turbo_neutral_1k",
        capital_usd=1000.0,
        spot_wallet_usd=600.0,
        futures_wallet_usd=400.0,
        carry_eta=0.60,
        carry_top_k=2,
        carry_min_hold_bars=36,
        carry_entry_bps=0.50,
        carry_exit_bps=0.05,
        carry_buffer_rank=2,
        carry_weighting_mode="funding_tilted",
        carry_tilt_alpha=0.65,
        carry_max_single_weight=0.60,
        overlay_target_vol_ann=0.180,
        overlay_gross_pct=0.45,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
    ),
    "carry_max_eta_1k": StrategyProfileConfig(
        name="carry_max_eta_1k",
        capital_usd=1000.0,
        spot_wallet_usd=780.0,
        futures_wallet_usd=220.0,
        carry_eta=0.78,
        carry_top_k=2,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.55,
        overlay_target_vol_ann=0.075,
        overlay_gross_pct=0.15,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
        # Sprint 26: 100% market-neutral high-utilization carry (khong co trend sleeve).
    ),
    "hybrid_trend_conservative_1k": StrategyProfileConfig(
        name="hybrid_trend_conservative_1k",
        capital_usd=1000.0,
        spot_wallet_usd=650.0,
        futures_wallet_usd=350.0,
        carry_eta=0.65,
        carry_top_k=3,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.50,
        overlay_target_vol_ann=0.075,
        overlay_gross_pct=0.15,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
        enable_trend_sleeve=True,
        trend_target_vol_ann=0.12,
        trend_max_gross_pct=0.55,
        trend_rebalance_hours=24,
        trend_z_entry=1.15,
        soft_dd_threshold_pct=0.06,
        hard_dd_threshold_pct=0.10,
    ),
    "hybrid_trend_balanced_1k": StrategyProfileConfig(
        name="hybrid_trend_balanced_1k",
        capital_usd=1000.0,
        spot_wallet_usd=650.0,
        futures_wallet_usd=350.0,
        carry_eta=0.72,
        carry_top_k=2,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=2,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.55,
        overlay_target_vol_ann=0.075,
        overlay_gross_pct=0.15,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
        enable_trend_sleeve=True,
        trend_target_vol_ann=0.16,
        trend_max_gross_pct=0.80,
        trend_rebalance_hours=24,
        trend_z_entry=1.05,
        soft_dd_threshold_pct=0.06,
        hard_dd_threshold_pct=0.10,
    ),
    # ---- Sprint 27: 550/450 split -> cong suat futures 1,350 USD (3.0x) thay vi 1,050 USD.
    "bi_trend_carry_1k": StrategyProfileConfig(
        name="bi_trend_carry_1k",
        capital_usd=1000.0,
        spot_wallet_usd=550.0,
        futures_wallet_usd=450.0,
        carry_eta=0.55,
        carry_top_k=3,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.50,
        overlay_target_vol_ann=0.075,
        overlay_gross_pct=0.15,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
        enable_trend_sleeve=True,
        allow_short_trend=True,
        trend_target_vol_ann=0.15,
        trend_max_gross_pct=0.65,
        trend_rebalance_hours=24,
        trend_z_entry=1.15,
        trend_z_short_entry=1.10,
        soft_dd_threshold_pct=0.06,
        hard_dd_threshold_pct=0.10,
    ),
    "carry_stat_arb_1k": StrategyProfileConfig(
        name="carry_stat_arb_1k",
        capital_usd=1000.0,
        spot_wallet_usd=550.0,
        futures_wallet_usd=450.0,
        carry_eta=0.55,
        carry_top_k=3,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.50,
        overlay_target_vol_ann=0.11,
        overlay_gross_pct=0.25,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
        enable_reversal_sleeve=True,
        reversal_gross_pct=0.35,
        soft_dd_threshold_pct=0.06,
        hard_dd_threshold_pct=0.10,
    ),
    "regime_omni_1k": StrategyProfileConfig(
        name="regime_omni_1k",
        capital_usd=1000.0,
        spot_wallet_usd=550.0,
        futures_wallet_usd=450.0,
        carry_eta=0.55,
        carry_top_k=3,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.50,
        overlay_target_vol_ann=0.11,
        overlay_gross_pct=0.25,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
        enable_trend_sleeve=True,
        allow_short_trend=True,
        require_macro_regime_filter=True,
        short_gross_scale=0.75,
        trend_target_vol_ann=0.15,
        trend_max_gross_pct=0.65,
        trend_rebalance_hours=24,
        trend_z_entry=1.15,
        trend_z_short_entry=1.10,
        enable_reversal_sleeve=True,
        reversal_gross_pct=0.35,
        dynamic_carry_to_alpha_margin=True,
        soft_dd_threshold_pct=0.06,
        hard_dd_threshold_pct=0.10,
    ),
    # ---- Sprint 28: Market-Structure & Macro Regime Switcher (600 Spot / 400 Futures)
    "flexible_regime_1k": StrategyProfileConfig(
        name="flexible_regime_1k",
        capital_usd=1000.0,
        spot_wallet_usd=600.0,
        futures_wallet_usd=400.0,
        carry_eta=0.60,
        carry_top_k=3,
        carry_min_hold_bars=63,
        carry_entry_bps=0.45,
        carry_exit_bps=0.0,
        carry_buffer_rank=3,
        carry_weighting_mode="equal",
        carry_tilt_alpha=0.0,
        carry_max_single_weight=0.50,
        overlay_target_vol_ann=0.075,
        overlay_gross_pct=0.15,
        max_futures_leverage=3.0,
        enable_yield_stacking=True,
        enable_trend_sleeve=True,
        allow_short_trend=True,
        require_macro_regime_filter=True,
        short_gross_scale=0.65,
        trend_target_vol_ann=0.15,
        trend_max_gross_pct=0.70,
        trend_rebalance_hours=24,
        trend_z_entry=1.15,
        trend_z_short_entry=1.10,
        enable_reversal_sleeve=True,
        reversal_gross_pct=0.25,
        dynamic_carry_to_alpha_margin=True,
        soft_dd_threshold_pct=0.06,
        hard_dd_threshold_pct=0.10,
    ),
}


def profile_names() -> list[str]:
    return sorted(PROFILE_REGISTRY)


def get_strategy_profile(
    name: "str | StrategyProfileConfig | None",
) -> StrategyProfileConfig:
    if name is None:
        return PROFILE_REGISTRY["default"]
    if isinstance(name, StrategyProfileConfig):
        return name
    if name not in PROFILE_REGISTRY:
        raise ValueError(
            f"Unknown strategy profile: {name!r}. Valid profiles: {sorted(PROFILE_REGISTRY)}"
        )
    return PROFILE_REGISTRY[name]


def compute_funding_tilted_weights(
    active_symbols: list[str],
    funding_rates_bps: dict[str, float],
    mode: str = "equal",
    tilt_alpha: float = 0.50,
    max_single_weight: float = 0.55,
) -> dict[str, float]:
    if not active_symbols:
        return {}
    if len(active_symbols) == 1:
        return {active_symbols[0]: 1.0}

    n = len(active_symbols)
    w_eq = 1.0 / n
    equal = {s: w_eq for s in active_symbols}
    if mode == "equal" or tilt_alpha <= 0.0:
        return equal

    scores = {
        s: max(0.0, float(funding_rates_bps.get(s, 0.0) or 0.0)) for s in active_symbols
    }
    total_score = sum(scores.values())
    if total_score <= 1e-12:
        return equal
    if max_single_weight < w_eq - 1e-12:
        return equal

    weights = {
        s: (1.0 - tilt_alpha) * w_eq + tilt_alpha * (scores[s] / total_score)
        for s in active_symbols
    }
    pinned: set[str] = set()
    for _ in range(100):
        over = [
            s for s in active_symbols
            if s not in pinned and weights[s] > max_single_weight + 1e-12
        ]
        if not over:
            break
        excess = sum(weights[s] for s in over) - len(over) * max_single_weight
        for s in over:
            weights[s] = max_single_weight
            pinned.add(s)
        free = [s for s in active_symbols if s not in pinned]
        if not free:
            break
        free_mass = sum(weights[s] for s in free)
        for s in free:
            if free_mass > 0.0:
                weights[s] += excess * (weights[s] / free_mass)
            else:
                weights[s] += excess / len(free)

    total = sum(weights.values())
    if total <= 0.0:
        return equal
    return {s: weights[s] / total for s in active_symbols}


def enforce_split_wallet_futures_cap(
    carry_perp_notional_usd: float,
    desired_overlay_gross_usd: float,
    futures_wallet_equity_usd: float,
    max_futures_leverage: float = 3.0,
) -> dict:
    max_total = max(0.0, futures_wallet_equity_usd * max_futures_leverage)
    available = max(0.0, max_total - carry_perp_notional_usd)
    desired = max(0.0, desired_overlay_gross_usd)
    allowed_overlay = min(desired, available)
    if carry_perp_notional_usd > max_total:
        scale = max_total / max(carry_perp_notional_usd, 1e-9)
    else:
        scale = 1.0
    scale = min(1.0, max(0.0, scale))
    if futures_wallet_equity_usd > 0:
        leverage = (
            carry_perp_notional_usd * scale + allowed_overlay
        ) / futures_wallet_equity_usd
    else:
        leverage = 0.0
    leverage = min(leverage, max_futures_leverage)
    clamped = bool(scale < 1.0 or allowed_overlay < desired - 1e-9)
    return {
        "allowed_overlay_gross_usd": float(allowed_overlay),
        "carry_scale_factor": float(scale),
        "effective_futures_leverage": float(leverage),
        "clamped": clamped,
    }


def compute_step_passive_yield_usd(
    profile: StrategyProfileConfig,
    dt_hours: float,
    deployed_spot_by_symbol_usd: dict[str, float],
) -> dict:
    if (not profile.enable_yield_stacking) or dt_hours <= 0:
        return {
            "bfusd_yield_usd": 0.0,
            "idle_earn_yield_usd": 0.0,
            "staking_yield_usd": 0.0,
            "total_passive_yield_usd": 0.0,
        }
    year_hours = 8760.0
    bfusd = profile.futures_wallet_usd * (profile.bfusd_apy / year_hours) * dt_hours
    total_deployed = sum(max(0.0, float(v)) for v in deployed_spot_by_symbol_usd.values())
    idle_spot = max(0.0, profile.spot_wallet_usd - total_deployed)
    idle_earn = idle_spot * (profile.idle_spot_earn_apy / year_hours) * dt_hours
    staking = 0.0
    for sym, usd_val in deployed_spot_by_symbol_usd.items():
        base = sym.split("/")[0].split(":")[0].upper().strip()
        staking += (
            max(0.0, float(usd_val))
            * (profile.liquid_staking_apy_map.get(base, 0.0) / year_hours)
            * dt_hours
        )
    return {
        "bfusd_yield_usd": float(bfusd),
        "idle_earn_yield_usd": float(idle_earn),
        "staking_yield_usd": float(staking),
        "total_passive_yield_usd": float(bfusd + idle_earn + staking),
    }
