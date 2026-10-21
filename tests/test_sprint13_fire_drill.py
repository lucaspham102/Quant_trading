"""Test Sprint 13 — diễn tập 5 chu kỳ end-to-end (fire drill).

Mỗi test chạy `run_five_cycle_fire_drill` trên `tmp_path` riêng: offline hoàn toàn,
không mạng, không sleep thật. Kiểm tra 5 chu kỳ theo Blueprint Chặng 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_e2e_fire_drill import run_five_cycle_fire_drill


@pytest.fixture(scope="module")
def drill(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Chạy drill một lần cho cả module để giữ thời gian test thấp."""
    tmp = tmp_path_factory.mktemp("fire_drill")
    return run_five_cycle_fire_drill(tmp / "state.json")


def test_drill_all_checks_pass(drill: dict) -> None:
    """Toàn bộ check phải PASS, không có check nào bị bỏ qua."""
    failed = {k: v for k, v in drill["checks"].items() if not v}
    assert not failed, f"check FAIL: {failed}"
    assert drill["passed"] is True


def test_drill_covers_five_cycles(drill: dict) -> None:
    """Báo cáo phải có đúng 5 chu kỳ và mọi chu kỳ đều pass."""
    cycles = drill["cycles"]
    assert [c["cycle"] for c in cycles] == [1, 2, 3, 4, 5]
    assert all(c["pass"] for c in cycles)


def test_cycle1_positions_persisted_to_disk(drill: dict) -> None:
    """Cycle 1: state ghi xuống đĩa, đọc lại được, kill-switch chưa bật."""
    payload = json.loads(Path(drill["state_path"]).read_text(encoding="utf-8"))
    assert "kill_switch_triggered" in payload
    assert drill["checks"]["c1_state_persisted"] is True
    assert drill["checks"]["c1_kill_switch_false"] is True


def test_cycle1_quantities_on_lot_step(drill: dict) -> None:
    """Cycle 1: mọi khối lượng phải là bội số lot step của sàn."""
    assert drill["checks"]["c1_lot_step_multiple"] is True


def test_cycle2_sub_lot_dust_marked_and_full_lot_healed(drill: dict) -> None:
    """Cycle 2: dust dưới 1 lô bị đánh cờ, lệch 2 lô thì tự gỡ cờ và sửa."""
    assert drill["checks"]["c2a_btc_unrepairable_flag"] is True
    assert drill["checks"]["c2a_btc_no_order"] is True
    assert drill["checks"]["c2b_btc_flag_cleared"] is True
    assert drill["checks"]["c2b_btc_repair_order"] is True


def test_cycle3_spread_and_crossed_book_blocked(drill: dict) -> None:
    """Cycle 3: spread 18,5 bps và crossed book đều bị chặn, không đặt lệnh."""
    assert drill["checks"]["c3_wide_spread_skipped"] is True
    assert drill["checks"]["c3_crossed_book_skipped"] is True
    assert drill["checks"]["c3_no_orders_placed"] is True


def test_cycle4_kill_switch_flattens_and_persists(drill: dict) -> None:
    """Cycle 4: drawdown 6,25% đóng sạch vị thế và ghi cờ xuống đĩa."""
    assert drill["checks"]["c4_kill_switch_returned_true"] is True
    assert drill["checks"]["c4_positions_flushed"] is True
    assert drill["checks"]["c4_disk_state_persisted"] is True
    assert drill["checks"]["c4_only_market_orders"] is True


def test_cycle5_sticky_then_reset(drill: dict) -> None:
    """Cycle 5: cờ dính qua restart (kể cả đổi capital), reset mới gỡ."""
    assert drill["checks"]["c5a_reconcile_kill_switch_sticky"] is True
    assert drill["checks"]["c5b_reset_state_cleared"] is True


def test_final_state_after_reset_is_clean(drill: dict) -> None:
    """Sau reset: equity về vốn gốc, ba cờ an toàn đều tắt."""
    payload = json.loads(Path(drill["state_path"]).read_text(encoding="utf-8"))
    assert payload["kill_switch_triggered"] is False
    assert payload["kill_switch_pending_close"] is False
    assert float(payload["current_equity"]) == pytest.approx(3000.0)


def test_drill_is_deterministic(tmp_path: Path) -> None:
    """Chạy lại trên đường dẫn khác phải cho cùng tập check PASS."""
    first = run_five_cycle_fire_drill(tmp_path / "a.json")
    second = run_five_cycle_fire_drill(tmp_path / "b.json")
    assert first["checks"] == second["checks"]
    assert first["passed"] is True and second["passed"] is True
