"""Opt-in velocity and acceleration switches for Sprint 28 Study.

Sprint 28: High-Velocity & Tactical Breakout Ablation Study (V1 -> V5).

The module is intentionally side-effect free. Every switch defaults to a
value that reproduces the historical Champion baseline (V0 = U4 + U5 + U3).
The live tournament never imports this module, ensuring 100% Zero-Interference.

Velocity Catalog
----------------
V0  Champion Baseline (U4 Blended 9.25 bps + U5 Flat Transfer + U3 Emergency Exit)
V1  Fast-Horizon Trend Alignment (EMA 72h/168h/504h, Momentum 168h/504h)
V2  High-Frequency Rebalance Cadence (Trend 8h/12h, Overlay 24h, deadband anti-churn)
V3  Early-Wave Entry Gate Relaxation (z_entry 0.95, ER 0.12, Breadth 0.25, Trail 2.5 Sigma)
V4  Agile Carry Rotation (Carry EMA span 21, min-hold 21, buffer-rank 2)
V5  Tactical Breakout Sub-Sleeve (7d Donchian Squeeze Breakout 25% gross when macro FLAT)
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "VelocityFlags",
    "VELOCITY_IDS",
    "BASELINE_VELOCITY_ID",
    "VELOCITY_CONFIGS",
    "COMBINATORIAL_CONFIGS",
    "build_velocity_flags",
    "materialize_velocity_profile",
]

VELOCITY_IDS: Tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5")
BASELINE_VELOCITY_ID = "V0_CHAMPION_BASELINE"


@dataclass(frozen=True)
class VelocityFlags:
    """All Sprint 28 velocity switches. The zero-value instance reproduces Champion V0 exactly."""

    # V1: Fast-Horizon Trend Alignment
    v1_fast_trend: bool = False
    v1_ema_fast_hours: int = 72       # 3d instead of 168h (7d)
    v1_ema_med_hours: int = 168      # 7d instead of 504h (21d)
    v1_ema_slow_hours: int = 504     # 21d instead of 1440h (60d)
    v1_short_horizon_hours: int = 168 # 7d instead of 504h (21d)
    v1_long_horizon_hours: int = 504  # 21d instead of 1440h (60d)
    v1_vol_window_hours: int = 168    # 7d instead of 336h (14d)

    # V2: High-Frequency Rebalance Cadence
    v2_high_frequency_cadence: bool = False
    v2_trend_rebalance_hours: int = 8      # 8h matching funding instead of 24h
    v2_overlay_rebalance_hours: int = 24   # 24h instead of 48h
    v2_trend_resize_deadband: float = 0.15 # Minimum relative change to trigger trend re-hedge

    # V3: Early-Wave Entry Gate Relaxation
    v3_early_entry_relaxation: bool = False
    v3_z_entry: float = 0.95               # Lowered from 1.15 to 0.95
    v3_min_long_er: float = 0.12           # Lowered from 0.16 to 0.12
    v3_min_bull_breadth: float = 0.25      # 2/8 coins leading instead of 3/8 (0.375)
    v3_trail_k_sigma: float = 2.5          # Tightened trailing stop (2.5 vs 3.0) to lock gains

    # V4: Agile Carry Rotation
    v4_agile_carry_rotation: bool = False
    v4_carry_ema_span: int = 21            # 7d instead of 42 (14d) funding event EWMA
    v4_carry_min_hold_bars: int = 21       # 7d hold instead of 63 (21d)
    v4_carry_buffer_rank: int = 2          # Top 5 buffer instead of 3 (Top 6)

    # V5: Tactical Breakout / Impulse Sub-Sleeve
    v5_tactical_breakout: bool = False
    v5_breakout_lookback_hours: int = 168  # 7-day Donchian / Volatility Squeeze channel
    v5_breakout_max_gross_pct: float = 0.25 # Modest impulse gross allocation (max 25%)
    v5_breakout_z_threshold: float = 1.30   # High-conviction impulse trigger
    v5_breakout_trail_k_sigma: float = 2.0  # Swift 2.0 Sigma profit-lock / risk-exit

    def is_noop(self) -> bool:
        return self == VelocityFlags()

    def active_ids(self) -> List[str]:
        out: List[str] = []
        if self.v1_fast_trend:
            out.append("V1")
        if self.v2_high_frequency_cadence:
            out.append("V2")
        if self.v3_early_entry_relaxation:
            out.append("V3")
        if self.v4_agile_carry_rotation:
            out.append("V4")
        if self.v5_tactical_breakout:
            out.append("V5")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_BASE_FLAGS = VelocityFlags()


def build_velocity_flags(upgrade_ids: Iterable[str]) -> VelocityFlags:
    """Compose a velocity flag container from individual upgrade IDs."""
    ids = {str(x).strip().upper() for x in upgrade_ids if str(x).strip()}
    unknown = ids - set(VELOCITY_IDS) - {BASELINE_VELOCITY_ID, "V0", "C0"}
    if unknown:
        raise ValueError(f"Unknown velocity upgrade ids: {sorted(unknown)}")

    updates: Dict[str, Any] = {}
    if "V1" in ids:
        updates["v1_fast_trend"] = True
    if "V2" in ids:
        updates["v2_high_frequency_cadence"] = True
    if "V3" in ids:
        updates["v3_early_entry_relaxation"] = True
    if "V4" in ids:
        updates["v4_agile_carry_rotation"] = True
    if "V5" in ids:
        updates["v5_tactical_breakout"] = True

    return VelocityFlags(**updates)


VELOCITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "V0": {
        "config_id": BASELINE_VELOCITY_ID,
        "upgrade_ids": [],
        "upgrade_id": "V0",
        "code_name": BASELINE_VELOCITY_ID,
        "title": "Champion Baseline Hiện Hành (Sprint 27 U4+U5+U3, Blended Cost 9.25 bps)",
        "target_layer": "Reference / Champion Baseline",
        "mechanism": (
            "Kế thừa toàn vẹn Champion Sprint 27 (Maker-First Blended 9.25 bps, Universal Flat Transfer, "
            "Emergency Exit 12 kỳ). Tốc độ chậm truyền thống: Trend EMA 168h/504h/1440h, đánh giá 24h, "
            "Overlay 48h, Carry span=42 min_hold=63."
        ),
    },
    "V1": {
        "config_id": "V1_FAST_HORIZON_TREND",
        "upgrade_ids": ["V1"],
        "upgrade_id": "V1",
        "code_name": "V1_FAST_HORIZON_TREND",
        "title": "Fast-Horizon Trend Alignment (Rút ngắn 50% độ trễ EMA & Momentum)",
        "target_layer": "Directional Trend Sleeve / Signal Horizon",
        "mechanism": (
            "Thay bộ khung chậm (168h, 504h, 1440h) bằng bộ khung phản ứng nhanh (72h, 168h, 504h). "
            "Điểm Z_comp tính trên cửa sổ 168h + 504h (7d + 21d). Bảo toàn Macro Regime Filter và Funding Drag Veto."
        ),
    },
    "V2": {
        "config_id": "V2_HIGH_FREQUENCY_CADENCE",
        "upgrade_ids": ["V2"],
        "upgrade_id": "V2",
        "code_name": "V2_HIGH_FREQUENCY_CADENCE",
        "title": "High-Frequency Rebalance Cadence (Trend 8h, Overlay 24h + Deadband)",
        "target_layer": "Execution Cadence / Rebalancing Clock",
        "mechanism": (
            "Hạ trend_rebalance_hours từ 24h xuống 8h (khớp nhịp Funding 00:00, 08:00, 16:00 UTC) "
            "và overlay_rebalance_hours từ 48h xuống 24h. Kèm deadband 15% chống đảo lệnh lắt nhắt."
        ),
    },
    "V3": {
        "config_id": "V3_EARLY_WAVE_ENTRY_RELAXATION",
        "upgrade_ids": ["V3"],
        "upgrade_id": "V3",
        "code_name": "V3_EARLY_WAVE_ENTRY_RELAXATION",
        "title": "Early-Wave Entry Gate Relaxation (Hạ ngưỡng kích hoạt để vào chân sóng)",
        "target_layer": "Directional Trend Sleeve / Entry & Exit Gates",
        "mechanism": (
            "Hạ z_entry từ 1.15 xuống 0.95, BTC Kaufman ER_336h từ 0.16 xuống 0.12, "
            "Bull Breadth từ 0.375 xuống 0.25 (2/8 coin bứt phá trước). "
            "Thắt chặt Trailing Stop: trail_k_sigma = 2.5 (thay vì 3.0) để khóa lời nhanh."
        ),
    },
    "V4": {
        "config_id": "V4_AGILE_CARRY_ROTATION",
        "upgrade_ids": ["V4"],
        "upgrade_id": "V4",
        "code_name": "V4_AGILE_CARRY_ROTATION",
        "title": "Agile Carry Rotation (Tăng độ nhạy xoay vòng vốn Carry)",
        "target_layer": "Delta-Neutral Carry / Selection & Holding",
        "mechanism": (
            "Rút ngắn Carry f_ewma span từ 42 xuống 21 kỳ 8h (7 ngày), hạ min_hold_periods từ 63 (21 ngày) "
            "xuống 21 kỳ (7 ngày), và thu hẹp buffer_rank từ 3 xuống 2 (Top 5). Tận dụng phí rẻ U4 9.25 bps "
            "để xoay vốn nhanh hơn sang coin bùng nổ Funding."
        ),
    },
    "V5": {
        "config_id": "V5_TACTICAL_BREAKOUT",
        "upgrade_ids": ["V5"],
        "upgrade_id": "V5",
        "code_name": "V5_TACTICAL_BREAKOUT",
        "title": "Tactical Breakout / Impulse Sub-Sleeve (Nhánh Xung Kích Ngắn Hạn khi Trend FLAT)",
        "target_layer": "Tactical Impulse Sleeve / Volatility Squeeze",
        "mechanism": (
            "Khi nhánh Trend lớn đang FLAT (chưa đủ điều kiện EMA dài hạn), cho phép kích hoạt vị thế "
            "xung kích ngắn hạn tối đa 25% gross khi phát hiện cú nén bứt phá đỉnh 7 ngày (168h Donchian), "
            "Z_168h >= 1.30 và BTC > EMA_168h. Thoát nhanh bằng Trailing Stop 2.0 Sigma."
        ),
    },
}

COMBINATORIAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "C0": {
        "combo_id": BASELINE_VELOCITY_ID,
        "label": "Champion Baseline Hiện Hành (U4 + U5 + U3)",
        "included_upgrades": [],
        "is_baseline": True,
        "is_champion": False,
    },
    "COMBO_V1_FAST_HORIZON_ONLY": {
        "combo_id": "COMBO_V1_FAST_HORIZON_ONLY",
        "label": "Tổ hợp V1 Đơn Lẻ (Fast-Horizon Trend Alignment 72/168/504h)",
        "included_upgrades": ["V1"],
        "is_baseline": False,
        "is_champion": False,
    },
    "COMBO_V1_V2_ACCEPTED_SYNERGY": {
        "combo_id": "COMBO_V1_V2_ACCEPTED_SYNERGY",
        "label": "Tổ hợp Hiệp Đồng Hai Nhánh Accepted (V1 Fast Horizon + V2 Cadence 8h/24h)",
        "included_upgrades": ["V1", "V2"],
        "is_baseline": False,
        "is_champion": False,
    },
    "COMBO_V1_V3_FAST_WAVE": {
        "combo_id": "COMBO_V1_V3_FAST_WAVE",
        "label": "Tổ hợp Sóng Nhanh (V1 Fast Horizon + V3 Early Gate 0.95)",
        "included_upgrades": ["V1", "V3"],
        "is_baseline": False,
        "is_champion": False,
    },
    "COMBO_ALL_FIVE": {
        "combo_id": "COMBO_ALL_FIVE",
        "label": "Tổ hợp Nhồi Nhét Cả 5 Nhánh (V1 + V2 + V3 + V4 + V5)",
        "included_upgrades": ["V1", "V2", "V3", "V4", "V5"],
        "is_baseline": False,
        "is_champion": False,
    },
    "BEST_VELOCITY_COMBO": {
        "combo_id": "BEST_VELOCITY_COMBO",
        "label": "Tổ Hợp Vô Địch Tăng Tốc (Dynamic Champion)",
        "included_upgrades": ["V1"],
        "is_baseline": False,
        "is_champion": True,
    },
}


def materialize_velocity_profile(
    base_profile: Any,
    config_id: str,
    flag_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, str]:
    """Materialize StrategyProfileConfig and market_key ('base' or 'v4') for a velocity config."""
    up_ids: List[str] = []
    if config_id in VELOCITY_CONFIGS:
        up_ids = list(VELOCITY_CONFIGS[config_id].get("upgrade_ids", []))
    elif config_id in COMBINATORIAL_CONFIGS:
        up_ids = list(COMBINATORIAL_CONFIGS[config_id].get("included_upgrades", []))
    else:
        for c in COMBINATORIAL_CONFIGS.values():
            if c.get("combo_id") == config_id:
                up_ids = list(c.get("included_upgrades", []))
                break
        else:
            for v in VELOCITY_CONFIGS.values():
                if v.get("config_id") == config_id:
                    up_ids = list(v.get("upgrade_ids", []))
                    break

    flags = build_velocity_flags(up_ids)
    if flag_overrides:
        flags = dataclasses.replace(flags, **flag_overrides)

    params: Dict[str, Any] = {"name": f"{base_profile.name}__{config_id}"}
    market_key = "base"

    if flags.v1_fast_trend:
        params["trend_ema_fast_hours"] = flags.v1_ema_fast_hours
        params["trend_ema_med_hours"] = flags.v1_ema_med_hours
        params["trend_ema_slow_hours"] = flags.v1_ema_slow_hours
        params["trend_vol_window_hours"] = flags.v1_vol_window_hours
        params["trend_short_horizon_hours"] = flags.v1_short_horizon_hours
        params["trend_long_horizon_hours"] = flags.v1_long_horizon_hours

    if flags.v2_high_frequency_cadence:
        params["trend_rebalance_hours"] = flags.v2_trend_rebalance_hours
        params["overlay_rebalance_hours"] = flags.v2_overlay_rebalance_hours

    if flags.v3_early_entry_relaxation:
        params["trend_z_entry"] = flags.v3_z_entry
        params["trend_min_long_er"] = flags.v3_min_long_er
        params["trend_min_bull_breadth"] = flags.v3_min_bull_breadth
        params["trend_trail_k_sigma"] = flags.v3_trail_k_sigma

    if flags.v4_agile_carry_rotation:
        params["carry_min_hold_bars"] = flags.v4_carry_min_hold_bars
        params["carry_buffer_rank"] = flags.v4_carry_buffer_rank
        market_key = "v4"

    if flags.v5_tactical_breakout:
        if getattr(base_profile, "enable_trend_sleeve", False):
            params["enable_tactical_breakout"] = True
            params["tactical_breakout_lookback_hours"] = flags.v5_breakout_lookback_hours
            params["tactical_breakout_max_gross_pct"] = flags.v5_breakout_max_gross_pct
            params["tactical_breakout_trail_k_sigma"] = flags.v5_breakout_trail_k_sigma

    profile = dataclasses.replace(base_profile, **params)
    return profile, market_key
