from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_quant_microscope_dashboard import (
    PROJECT_ROOT,
    QuantMicroscopeEngine,
    create_dashboard_app,
)
from scripts.run_upgrade_ablation_study import _annotate_combo_verdicts, _make_combo_defs
from src.upgrade_variants import (
    UPGRADE_IDS,
    UpgradeFlags,
    apply_wallet_split,
    build_variant_flags,
)


def test_upgrade_flags_default_is_noop() -> None:
    flags = UpgradeFlags()
    assert flags.u1_smart_lot_sweep is False
    assert flags.u2_basis_weight == pytest.approx(0.0)
    assert flags.u2_discount_veto_bps == pytest.approx(0.0)
    assert flags.u3_emergency_exit_min_periods == 0
    assert flags.u4_blended_carry_cost_bps == pytest.approx(0.0)
    assert flags.u4_entry_threshold_bps == pytest.approx(0.0)
    assert flags.u5_dynamic_transfer_flat is False
    assert flags.enable_liquid_12_universe is False
    assert flags.u6_liquid_12_universe is False


def test_upgrade_ids_and_u4_cost_only_semantics() -> None:
    assert UPGRADE_IDS == ("U1", "U2", "U3", "U4", "U5", "U6")
    u6 = build_variant_flags(["U6"])
    assert u6.enable_liquid_12_universe is True
    assert u6.active_ids() == ["U6"]
    u4 = build_variant_flags(["U4"])
    assert u4.u4_blended_carry_cost_bps == pytest.approx(9.25)
    # U4 is cost-only; the 0.40 bps threshold is a separate sensitivity run.
    assert u4.u4_entry_threshold_bps == pytest.approx(0.0)
    assert u4.u1_smart_lot_sweep is False
    assert u4.u5_dynamic_transfer_flat is False


def test_wallet_split_only_reshapes_carry_max() -> None:
    assert apply_wallet_split("carry_max_eta_1k", 1000.0, ["U5"]) == pytest.approx((700.0, 300.0))
    assert apply_wallet_split("regime_omni_1k", 1000.0, ["U5"]) == pytest.approx((0.0, 0.0))
    assert apply_wallet_split("carry_max_eta_1k", 1000.0, []) == pytest.approx((0.0, 0.0))


def test_combo_defs_map_upgrade_ids_for_real_execution() -> None:
    verdicts = {
        "U1": {"verdict": "REJECTED", "materiality": "EFFECTIVE"},
        "U2": {"verdict": "REJECTED", "materiality": "EFFECTIVE"},
        "U3": {"verdict": "ACCEPTED", "materiality": "NEGLIGIBLE"},
        "U4": {"verdict": "ACCEPTED", "materiality": "EFFECTIVE"},
        "U5": {"verdict": "ACCEPTED", "materiality": "EFFECTIVE"},
    }
    defs = _make_combo_defs(verdicts, {}, {})
    assert defs["COMBO_A"]["upgrade_ids"] == ["U1", "U2", "U3"]
    assert defs["COMBO_B"]["upgrade_ids"] == ["U4", "U5"]
    assert defs["BEST_COMBO"]["upgrade_ids"] == ["U4", "U5"]
    assert defs["BEST_COMBO"]["is_champion"] is True
    assert defs["COMBO_C"]["upgrade_ids"] == ["U4", "U5", "U6"]
    assert defs["COMBO_C"]["combo_id"] == "COMBO_C_CHAMPION_PLUS_U6"


def test_layer5_reads_real_study_artifact_without_hardcoded_c5() -> None:
    artifact = PROJECT_ROOT / "logs" / "upgrade_ablation_and_combo_results.json"
    if not artifact.exists():
        pytest.skip("Sprint-27 study artifact has not been generated in this checkout")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    engine = QuantMicroscopeEngine(project_root=PROJECT_ROOT)
    l5 = engine._build_layer5_upgrade_ablation_and_combos("regime_omni_1k")

    assert l5["available"] is True
    assert [u["upgrade_id"] for u in l5["ablation_upgrades"]] == list(UPGRADE_IDS)
    combo_ids = {c["combo_id"] for c in l5["combinatorial_matrix"]}
    assert "C0_BASELINE_U0" in combo_ids
    assert "BEST_COMBO" in combo_ids
    assert "COMBO_C_CHAMPION_PLUS_U6" in combo_ids
    assert all("included_upgrades" in c for c in l5["combinatorial_matrix"])
    assert all("verdict" in c for c in l5["combinatorial_matrix"])
    assert all("invariants_all_passed" in c for c in l5["combinatorial_matrix"])
    champions = [c for c in l5["combinatorial_matrix"] if c.get("is_champion")]
    assert len(champions) == 1
    assert champions[0]["verdict"] == "ACCEPTED"
    assert l5["champion_vs_baseline_summary"]["champion_combo_id"] in ("BEST_COMBO", "COMBO_C_CHAMPION_PLUS_U6")
    assert l5["invariants_verification"]["all_passed"] is True
    assert len(l5["invariants_verification"]["laws"]) == 5
    assert l5["parameter_sweep"]["u4_entry_threshold"]
    assert payload["recommended_production_upgrade"]["selected_combo_id"] in ("BEST_COMBO", "COMBO_C_CHAMPION_PLUS_U6")


def test_layer5_missing_artifact_is_explicit_and_does_not_fabricate(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_quant_microscope_dashboard as dash

    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    engine = QuantMicroscopeEngine(project_root=tmp_path)
    l5 = engine._build_layer5_upgrade_ablation_and_combos("regime_omni_1k")
    assert l5["available"] is False
    assert l5["source_tag"] == "MISSING_STUDY_ARTIFACT"
    assert l5["ablation_upgrades"] == []
    assert l5["champion_vs_baseline_summary"] == {}


def test_dashboard_html_exposes_layer5_tables_and_charts() -> None:
    engine = QuantMicroscopeEngine(project_root=PROJECT_ROOT)
    client = create_dashboard_app(engine).test_client()
    html = client.get("/").get_data(as_text=True)
    for token in (
        "l5-ablation-table",
        "l5-combos-table",
        "l5-cross-profile-table",
        "l5-sensitivity-table",
        "chart-l5-combos-apr",
        "chart-l5-cross-profile",
        "chart-l5-window-progression",
    ):
        assert token in html


def test_combo_verdicts_apply_strict_falsification() -> None:
    base = {
        "combo_id": "C0", "is_baseline": True,
        "metrics_by_profile": {"p": {"24M": {"pure_apr_pct": 10.0, "pure_max_dd_pct": 8.0}}},
    }
    good = {
        "combo_id": "GOOD", "is_baseline": False,
        "metrics_by_profile": {"p": {"24M": {"pure_apr_pct": 10.2, "pure_max_dd_pct": 7.8}}},
    }
    bad = {
        "combo_id": "BAD", "is_baseline": False,
        "metrics_by_profile": {"p": {"24M": {"pure_apr_pct": 9.9, "pure_max_dd_pct": 8.6}}},
    }
    _annotate_combo_verdicts([base, good, bad])
    assert base["verdict"] == "BASELINE"
    assert good["verdict"] == "ACCEPTED"
    assert bad["verdict"] == "REJECTED"


def test_smart_lot_sweep_filters_small_lot_symbols_only() -> None:
    from scripts.run_1k_profile_tournament_backtest import _apply_smart_lot_sweep
    import numpy as np

    candidates = ["BTC/USDT:USDT", "DOGE/USDT:USDT", "SOL/USDT:USDT"]
    qty_map = {"BTC/USDT:USDT": 0.01, "DOGE/USDT:USDT": 100.0, "SOL/USDT:USDT": 2.0}
    perp_row = np.array([60000.0, 0.15, 150.0])
    sidx = {"BTC/USDT:USDT": 0, "DOGE/USDT:USDT": 1, "SOL/USDT:USDT": 2}
    lot_steps = {"BTC/USDT:USDT": 0.001, "DOGE/USDT:USDT": 1.0, "SOL/USDT:USDT": 0.01}

    updated_qty, swept = _apply_smart_lot_sweep(
        qty_map=dict(qty_map),
        candidates=candidates,
        perp_row=perp_row,
        sidx=sidx,
        lot_steps=lot_steps,
        budget_usd=1500.0,
        per_cap_usd=800.0,
        min_sweep_usd=25.0,
    )

    # BTC must not be swept into!
    assert updated_qty["BTC/USDT:USDT"] == qty_map["BTC/USDT:USDT"]
    # DOGE or SOL (both in SMALL_LOT_SWEEP_SYMBOLS) should absorb the sweep
    assert updated_qty["DOGE/USDT:USDT"] > qty_map["DOGE/USDT:USDT"] or updated_qty["SOL/USDT:USDT"] > qty_map["SOL/USDT:USDT"]
    assert swept > 0.0


def test_liquid_12_symbols_and_lot_steps() -> None:
    from scripts.run_1k_profile_tournament_backtest import TOURNAMENT_LOT_STEPS
    from src.upgrade_variants import LIQUID_12_SYMBOLS

    assert len(LIQUID_12_SYMBOLS) == 12
    for s in ("SUI/USDT:USDT", "ADA/USDT:USDT", "NEAR/USDT:USDT", "DOT/USDT:USDT"):
        assert s in LIQUID_12_SYMBOLS
        assert s in TOURNAMENT_LOT_STEPS

    assert TOURNAMENT_LOT_STEPS["SUI/USDT:USDT"] == pytest.approx(0.1)
    assert TOURNAMENT_LOT_STEPS["ADA/USDT:USDT"] == pytest.approx(1.0)
    assert TOURNAMENT_LOT_STEPS["NEAR/USDT:USDT"] == pytest.approx(0.1)
    assert TOURNAMENT_LOT_STEPS["DOT/USDT:USDT"] == pytest.approx(0.1)
