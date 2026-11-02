"""Opt-in upgrade switches for the Sprint 27 Ablation & Combinatorial Study.

The module is intentionally side-effect free.  Every switch defaults to a
value that reproduces the historical (Baseline / U0) behaviour of
``scripts/run_1k_profile_tournament_backtest.py``.  The live tournament
(``scripts/run_parallel_live_demo.py``) never imports this module, so enabling
a research variant cannot change production behaviour.

Upgrade catalog
---------------
U1  Smart Lot Remainder Re-allocation (carry-leg dust sweep)
U2  Controlled Dynamic Basis Weight (w_basis > 0 with clipping + discount veto)
U3  Emergency Negative Funding Exit (conditional early release from min-hold)
U4  Maker-First / Blended Execution Cost (carry cost + entry threshold tuning)
U5  Regime-Adaptive Split-Wallet Rebalancing (wallet split / universal transfer)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

__all__ = [
    "UpgradeFlags",
    "UPGRADE_IDS",
    "LIQUID_12_SYMBOLS",
    "SMALL_LOT_SWEEP_SYMBOLS",
    "build_variant_flags",
    "variant_included_upgrades",
]


UPGRADE_IDS: Tuple[str, ...] = ("U1", "U2", "U3", "U4", "U5", "U6")

# Expanded liquid 12 universe (8 legacy + SUI, ADA, NEAR, DOT)
LIQUID_12_SYMBOLS: Tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "SUI/USDT:USDT",
    "ADA/USDT:USDT",
    "NEAR/USDT:USDT",
    "DOT/USDT:USDT",
)

# Symbols whose lot step is small enough to absorb residual carry budget without
# creating a new $10+ dust position.  Ordered by execution granularity.
SMALL_LOT_SWEEP_SYMBOLS: Tuple[str, ...] = (
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "SOL/USDT:USDT",
    "LINK/USDT:USDT",
    "SUI/USDT:USDT",
    "ADA/USDT:USDT",
    "NEAR/USDT:USDT",
    "DOT/USDT:USDT",
)


@dataclass(frozen=True)
class UpgradeFlags:
    """All research switches.  The zero-value instance is an exact no-op."""

    # U1: sweep unallocated carry budget into small-lot active symbols.
    u1_smart_lot_sweep: bool = False
    u1_min_sweep_usd: float = 25.0

    # U2: dynamic basis score.  0.0 keeps the historical w_basis = 0.0 path.
    u2_basis_weight: float = 0.0
    u2_basis_clip_bps: float = 10.0
    u2_discount_veto_bps: float = 0.0

    # U3: emergency exit.  0 periods disables the override.
    u3_emergency_exit_min_periods: int = 0
    u3_emergency_exit_funding_bps: float = -1.0

    # U4: blended execution cost and optional entry-threshold tuning.
    u4_blended_carry_cost_bps: float = 0.0
    u4_entry_threshold_bps: float = 0.0

    # U5: wallet re-split and/or FLAT-regime universal transfer simulation.
    u5_spot_wallet_usd: float = 0.0
    u5_futures_wallet_usd: float = 0.0
    u5_dynamic_transfer_flat: bool = False

    # U6: expand asset universe from LIQUID_8 to LIQUID_12 (SUI, ADA, NEAR, DOT)
    enable_liquid_12_universe: bool = False
    u6_liquid_12_universe: bool = False

    def is_noop(self) -> bool:
        return self == UpgradeFlags()

    def active_ids(self) -> List[str]:
        out: List[str] = []
        if self.u1_smart_lot_sweep:
            out.append("U1")
        if self.u2_basis_weight > 0.0:
            out.append("U2")
        if self.u3_emergency_exit_min_periods > 0:
            out.append("U3")
        if self.u4_blended_carry_cost_bps > 0.0 or self.u4_entry_threshold_bps > 0.0:
            out.append("U4")
        if self.u5_dynamic_transfer_flat or (
            self.u5_spot_wallet_usd > 0.0 and self.u5_futures_wallet_usd > 0.0
        ):
            out.append("U5")
        if self.enable_liquid_12_universe or self.u6_liquid_12_universe:
            out.append("U6")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_BASE_FLAGS = UpgradeFlags()


def build_variant_flags(upgrade_ids: Iterable[str]) -> UpgradeFlags:
    """Compose a combo from the canonical single-upgrade definitions.

    Parameters are the study's documented primary settings (not cherry-picked):
      * U2: w_basis = 0.15, basis_dev clipped at +/-10 bps, veto below -8 bps
      * U3: 12-period minimum hold, emergency threshold -1.0 bps / 8h
      * U4: 9.25 bps blended carry cost per side (entry threshold unchanged);
            the 0.40 bps threshold is a separate sensitivity variant (U4T).
      * U5: dynamic FLAT transfer; profile-specific wallet re-split is applied
            by ``apply_wallet_split`` in the study runner.
    """
    ids = {str(x).strip().upper() for x in upgrade_ids if str(x).strip()}
    unknown = ids - set(UPGRADE_IDS)
    if unknown:
        raise ValueError(f"Unknown upgrade ids: {sorted(unknown)}")

    updates: Dict[str, Any] = {}
    if "U1" in ids:
        updates["u1_smart_lot_sweep"] = True
    if "U2" in ids:
        updates.update(
            u2_basis_weight=0.15,
            u2_basis_clip_bps=10.0,
            u2_discount_veto_bps=8.0,
        )
    if "U3" in ids:
        updates.update(
            u3_emergency_exit_min_periods=12,
            u3_emergency_exit_funding_bps=-1.0,
        )
    if "U4" in ids:
        updates.update(
            u4_blended_carry_cost_bps=9.25,
            u4_entry_threshold_bps=0.0,
        )
    if "U5" in ids:
        updates["u5_dynamic_transfer_flat"] = True
    if "U6" in ids:
        updates["enable_liquid_12_universe"] = True
        updates["u6_liquid_12_universe"] = True

    return UpgradeFlags(**updates)


def variant_included_upgrades(flags: UpgradeFlags) -> List[str]:
    return flags.active_ids()


def apply_wallet_split(
    profile_name: str,
    total_capital_usd: float,
    upgrade_ids: Iterable[str],
) -> Tuple[float, float]:
    """Return the (spot, futures) wallet split for a U5 scenario.

    ``carry_max_eta_1k`` is the profile that is structurally cap-bound
    (78/22 -> futures capacity $660).  The study moves it to the documented
    70/30 equilibrium.  Other profiles keep their production split and rely on
    the simulated FLAT-regime universal transfer instead.
    """
    ids = {str(x).strip().upper() for x in upgrade_ids if str(x).strip()}
    if "U5" not in ids:
        return 0.0, 0.0
    if profile_name == "carry_max_eta_1k":
        futures_share = 0.30
    else:
        return 0.0, 0.0
    futures = round(float(total_capital_usd) * futures_share, 2)
    spot = round(float(total_capital_usd) - futures, 2)
    return spot, futures
