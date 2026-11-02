from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_quant_microscope_dashboard import (
    PROJECT_ROOT,
    TOURNAMENT_FOUR_PROFILES,
    QuantMicroscopeEngine,
    create_dashboard_app,
    safe_read_json_bytes,
)


def test_safe_read_json_bytes_primary_and_bak_fallback(tmp_path: Path) -> None:
    """
    Verify safe_read_json_bytes reads primary file into memory and automatically
    falls back to `.json.bak` if the primary file is missing or mid-replace.
    """
    primary = tmp_path / "tournament_heartbeat.json"
    bak = tmp_path / "tournament_heartbeat.json.bak"

    # Case 1: Both missing
    payload, tag = safe_read_json_bytes(primary, retries=1)
    assert payload is None
    assert tag == "MISSING"

    # Case 2: Only .bak exists (simulating brief window during os.replace)
    bak.write_text(json.dumps({"cycle": 104, "phase": "SLEEPING_8H"}), encoding="utf-8")
    payload, tag = safe_read_json_bytes(primary, retries=1)
    assert payload is not None
    assert payload["cycle"] == 104
    assert tag == "BAK_FALLBACK"

    # Case 3: Primary is corrupted/incomplete, fallback to .bak succeeds
    primary.write_text("{corrupted_json", encoding="utf-8")
    payload, tag = safe_read_json_bytes(primary, retries=1)
    assert payload is not None
    assert payload["cycle"] == 104
    assert tag == "BAK_FALLBACK"

    # Case 4: Valid primary takes precedence
    primary.write_text(json.dumps({"cycle": 105, "phase": "COMPLETED_CYCLE"}), encoding="utf-8")
    payload, tag = safe_read_json_bytes(primary, retries=1)
    assert payload is not None
    assert payload["cycle"] == 105
    assert tag == "LIVE_PRIMARY"


def test_engine_reads_real_workspace_data_and_preserves_zero_interference() -> None:
    """
    Run QuantMicroscopeEngine against real `data/` and `logs/` in the workspace.
    Verify all 5 anatomical layers and confirm 100% read-only behavior.
    """
    state_dir = PROJECT_ROOT / "data" / "state"
    before_files = {p.name: p.stat().st_mtime for p in state_dir.glob("*") if p.is_file()}

    engine = QuantMicroscopeEngine(project_root=PROJECT_ROOT)
    snapshot = engine.build_snapshot(selected_profile="regime_omni_1k")

    after_files = {p.name: p.stat().st_mtime for p in state_dir.glob("*") if p.is_file()}
    assert before_files == after_files, "Dashboard must NEVER write or touch files in data/state/"

    # Layer 1 checks
    l1 = snapshot["layer1_tournament_and_watchdog"]
    assert "heartbeat" in l1
    assert "data_freshness" in l1
    assert l1["data_freshness"]["common_aligned_hours"] >= 1440
    assert len(l1["leaderboard"]) == 4
    lb_names = {r["profile_name"] for r in l1["leaderboard"]}
    assert lb_names == set(TOURNAMENT_FOUR_PROFILES)
    assert len(l1["benchmark_comparison"]) == 4

    # Layer 2 checks (Carry Microscope & Min-Hold 63 Tracker)
    l2 = snapshot["layer2_carry_microscope"]
    assert l2["selected_profile"] == "regime_omni_1k"
    assert l2["gate_config"]["entry_threshold_bps"] == pytest.approx(0.45)
    assert l2["gate_config"]["exit_threshold_bps"] == pytest.approx(0.0)
    assert l2["gate_config"]["buffer_ceiling_rank"] == 6
    assert l2["gate_config"]["min_hold_8h_periods"] == 63
    assert len(l2["ranking"]) == 8
    assert len(l2["open_positions_tracker"]) >= 1
    for pos in l2["open_positions_tracker"]:
        assert "periods_held" in pos
        assert pos["min_hold_target"] == 63
        assert "spot_qty" in pos and "perp_qty" in pos
        assert "delta_drift_usd" in pos
        assert "rounding_dust_usd" in pos

    # Layer 3 checks (XS Overlay & Trend Context Gate Radar)
    l3 = snapshot["layer3_context_gate_radar"]
    xs = l3["xs_overlay_radar"]
    assert xs["snr_threshold_backtest"] == pytest.approx(1.35)
    assert xs["snr_threshold_live"] == pytest.approx(0.25)
    assert xs["dollar_neutral_lock_ok"] is True
    assert xs["lot_imbalance_usd"] <= 10.0
    assert len(xs["ranking"]) == 8

    tr = l3["trend_macro_radar"]
    assert tr["z_entry_long"] == pytest.approx(1.15)
    assert tr["z_entry_short"] == pytest.approx(-1.10)
    assert tr["max_funding_drag_long_bps"] == pytest.approx(2.5)
    assert tr["max_funding_drag_short_bps"] == pytest.approx(-0.25)
    assert tr["chandelier_long_sigma"] == pytest.approx(3.0)
    assert tr["chandelier_short_sigma"] == pytest.approx(2.2)
    assert len(tr["symbols_matrix"]) == 8

    # Layer 4 checks (5 Golden Quant Invariants & 2-Tier Circuit Breaker)
    l4 = snapshot["layer4_invariants_and_breakers"]
    assert len(l4["invariants"]) == 5
    assert l4["all_invariants_green"] is True
    assert l4["circuit_breaker"]["state"] in ("NORMAL", "SOFT_THROTTLE", "HARD_STOP_LOCKED")
    assert len(l4["kill_switches"]) == 4


def test_flask_dashboard_routes() -> None:
    """
    Verify Flask web UI (`/`) and JSON telemetry API (`/api/telemetry`, `/api/health`)
    load cleanly without errors.
    """
    engine = QuantMicroscopeEngine(project_root=PROJECT_ROOT)
    app = create_dashboard_app(engine)
    client = app.test_client()

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.get_json()["mode"] == "100_PERCENT_READ_ONLY"

    res_index = client.get("/")
    assert res_index.status_code == 200
    html = res_index.get_data(as_text=True)
    assert "LOCAL QUANT MICROSCOPE & TELEMETRY DASHBOARD" in html
    assert "TẦNG 1 (L1)" in html
    assert "TẦNG 2 (L2)" in html
    assert "TẦNG 3 (L3)" in html
    assert "TẦNG 4 (L4)" in html
    assert "TẦNG 5 (L5)" in html

    for profile in TOURNAMENT_FOUR_PROFILES:
        res_api = client.get(f"/api/telemetry?profile={profile}")
        assert res_api.status_code == 200
        payload = res_api.get_json()
        assert payload["meta"]["selected_profile"] == profile
        assert payload["layer4_invariants_and_breakers"]["all_invariants_green"] is True
        assert "layer5_upgrade_ablation_and_combos" in payload
        assert "available" in payload["layer5_upgrade_ablation_and_combos"]
