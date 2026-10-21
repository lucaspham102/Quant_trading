"""Sinh bang doi soat Holdout 3 phuong phap (Sprint 13).

Chay: python scripts/gen_holdout_bridge.py
Doc 2 payload (compound / fixed-base) va ghi docs/sprint13_holdout_bridge.md.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sprint11_eval import compute_holdout_accounting_bridge  # noqa: E402


def _load(path: Path) -> dict:
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    comp_path = ROOT / "data" / "bridge_compound" / "sprint11_payload.json"
    fixed_path = ROOT / "data" / "bridge_fixed" / "sprint11_payload.json"
    payload_a = _load(comp_path)
    payload_b = _load(fixed_path)

    bridge = compute_holdout_accounting_bridge(payload_a, payload_b)
    a = bridge["continuous_compound"]
    b = bridge["continuous_fixed_base"]
    c = bridge["standalone_cold_start"]
    d = bridge["bridge_decomposition"]

    lines = [
        "# Sprint 13 — Cầu nối kế toán Holdout (warm-state vs cold-start)",
        "",
        "Sinh tự động bởi `scripts/gen_holdout_bridge.py` từ 2 lần chạy `run_sprint11.py`",
        f"(compound: `data/bridge_compound`, fixed-base: `data/bridge_fixed`). Cửa sổ holdout: "
        f"{bridge['holdout_days']:.0f} ngày, vốn gốc {bridge['active_capital_usd']:.2f} USDT.",
        "",
        "## Ba phương pháp (tất cả đều là lãi ròng all-in sau phí carry + phí overlay)",
        "",
        "| Phương pháp | Vốn đầu cửa sổ | PnL ròng (USDT) | Tỷ suất | Cách hiểu |",
        "|---|---|---|---|---|",
        f"| A. Continuous compound (`reinvest=True`) | {a['start_equity_usd']:.2f} | {a['net_pnl_usd']:.2f} | {a['return_on_start_equity_pct']:.2f}% | Vị thế ấm, quy mô phóng theo lãi kép |",
        f"| B. Continuous fixed-base (`reinvest=False`) | {b['start_equity_usd']:.2f} (vốn hoạt động {b['active_capital_usd']:.2f}) | {b['net_pnl_usd']:.2f} | {b['return_on_active_capital_pct']:.2f}% trên vốn 3.000 | Vị thế ấm, quy mô neo 3.000 suốt 3,45 năm |",
        f"| C. Standalone cold-start re-sim | {c['start_equity_usd']:.2f} | {c['net_pnl_usd']:.2f} | {c['return_on_active_capital_pct']:.2f}% | Bắt đầu 100% tiền mặt, phải mở lại danh mục |",
        "",
        "## Phân rã chênh lệch",
        "",
        "| Thành phần | USD | Ý nghĩa |",
        "|---|---|---|",
        f"| Hiệu ứng quy mô do lãi kép (A − B) | {d['compounding_scale_effect_usd']:.2f} | Vị thế lớn hơn nhờ lãi tích lũy, không phải alpha tốt hơn |",
        f"| Tổng ma sát cold-start (B − C) | {d['cold_start_total_drag_usd']:.2f} | Giá phải trả khi bắt đầu lại từ tiền mặt |",
        f"| — trong đó phí mở danh mục ngày đầu | {d['cold_start_initial_entry_fee_usd']:.2f} | 24 bar đầu của lần chạy standalone |",
        f"| — còn lại: lệch đường đi warmup/min_hold | {d['warmup_and_min_hold_path_diff_usd']:.2f} | Do `ema_span=42` / `min_hold=63` chưa ấm ở đầu cửa sổ |",
        "",
        "## Đẳng thức bảo toàn",
        "",
        "```",
        "A.net_pnl = C.net_pnl + (A − B) + (B − C)",
        f"{a['net_pnl_usd']:.6f} = {c['net_pnl_usd']:.6f} + {d['compounding_scale_effect_usd']:.6f} "
        f"+ {d['cold_start_total_drag_usd']:.6f}",
        f"residual = {a['net_pnl_usd'] - (c['net_pnl_usd'] + d['compounding_scale_effect_usd'] + d['cold_start_total_drag_usd']):.2e}",
        "```",
        "",
        "## Kết luận",
        "",
        "- Con số **+3,75%** từng được trích dẫn là carry **gross** (funding + basis) cộng overlay net,",
        "  chưa trừ phí carry — không phải lãi ròng all-in. Không dùng để kết luận GO.",
        "- Ba phương pháp cho ba con số khác nhau vì **cùng một chiến lược, khác trạng thái vốn**.",
        "  Cửa sổ holdout ngắn (181 ngày) nên chênh lệch quy mô vài phần trăm vốn tạo khác biệt ~0,5 điểm %.",
        "- Tiêu chí giữ nguyên: cả slice lẫn re-sim phải ≥ 3,5% → hiện **NO-GO**.",
        "",
    ]
    out = ROOT / "docs" / "sprint13_holdout_bridge.md"
    io.open(out, "w", encoding="utf-8", newline="").write("\n".join(lines))
    print(f"da ghi {out}")
    print(f"A={a['net_pnl_usd']:.2f} B={b['net_pnl_usd']:.2f} C={c['net_pnl_usd']:.2f}")
    print(
        "residual=",
        a["net_pnl_usd"]
        - (c["net_pnl_usd"] + d["compounding_scale_effect_usd"] + d["cold_start_total_drag_usd"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
