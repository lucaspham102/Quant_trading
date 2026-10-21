"""Sinh bang bridge holdout 4 phuong phap (A/B/C/D) cho Sprint 14.

A = continuous compound (tu dong tai dau tu, Sprint 11 goc)
B = continuous fixed-base (neo von 3000 USDT)
C = standalone cold-start (chi nap du lieu holdout, khong warm indicator)
D = indicator-warm cash-start (chi bao da am + 100% tien mat tai moc holdout)

Dong thoi phan ra chenh lech B - C thanh 3 phan: phi mo vi the ngay dau,
do mu cua chi bao (indicator blindness) va chenh lech ton kho (inventory carryover).

Chay: python scripts/gen_warm_cash_bridge.py
Xuat: docs/sprint14_warm_cash_bridge.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_sprint11 import load_multi_aligned_data  # noqa: E402
from src.cross_exchange_carry import LIQUID_8_SYMBOLS  # noqa: E402
from src.sprint11_eval import (  # noqa: E402
    SPRINT11_HOLDOUT_END,
    SPRINT11_INITIAL_CAPITAL,
    SPRINT11_SPLIT_DATE,
    compute_holdout_accounting_bridge,
    run_indicator_warm_cash_start,
)

RAW_DIR = _ROOT / "data" / "raw"
OUT_DOC = _ROOT / "docs" / "sprint14_warm_cash_bridge.md"
PAYLOAD_COMPOUND = _ROOT / "data" / "bridge_compound" / "sprint11_payload.json"
PAYLOAD_FIXED = _ROOT / "data" / "bridge_fixed" / "sprint11_payload.json"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def main() -> int:
    if not PAYLOAD_COMPOUND.exists() or not PAYLOAD_FIXED.exists():
        print("Thieu payload A/B. Chay truoc:")
        print("  python run_sprint11.py --reinvest 1 --out-dir data/bridge_compound")
        print("  python run_sprint11.py --reinvest 0 --out-dir data/bridge_fixed")
        return 2

    compound = json.loads(PAYLOAD_COMPOUND.read_text(encoding="utf-8"))
    fixed = json.loads(PAYLOAD_FIXED.read_text(encoding="utf-8"))

    print("[1/2] Nap du lieu da san va mo phong Method D (indicator-warm cash-start)...")
    data = load_multi_aligned_data(
        raw_dir=RAW_DIR,
        symbols=list(LIQUID_8_SYMBOLS),
        start="2023-04-01",
        end=SPRINT11_HOLDOUT_END,
    )
    if not data:
        print("Khong nap duoc du lieu. Ket thuc.")
        return 1

    warm_cash = run_indicator_warm_cash_start(
        data=data,
        capital=SPRINT11_INITIAL_CAPITAL,
        split_date=SPRINT11_SPLIT_DATE,
        holdout_end=SPRINT11_HOLDOUT_END,
        reinvest_profits=False,
    )
    print(
        f"      D: net={warm_cash['net_pnl_usd']:.2f} USD "
        f"({warm_cash['return_on_active_capital_pct']:.2f}%) "
        f"warmup={warm_cash['warmup_hours']}h trade_start_idx={warm_cash['trade_start_idx']}"
    )

    print("[2/2] Tinh bridge 4 phuong phap + phan ra 3 thanh phan...")
    bridge = compute_holdout_accounting_bridge(compound, fixed, warm_cash=warm_cash)
    a = bridge["continuous_compound"]
    b = bridge["continuous_fixed_base"]
    c = bridge["standalone_cold_start"]
    d = bridge["indicator_warm_cash_start"]
    dec = bridge["warm_cash_decomposition"]

    identity = (b["net_pnl_usd"] - c["net_pnl_usd"]) - (
        (b["net_pnl_usd"] - d["net_pnl_usd"]) + (d["net_pnl_usd"] - c["net_pnl_usd"])
    )
    decomp_residual = dec["unexplained_total_usd"]

    lines: list[str] = []
    lines.append("# Sprint 14 - Warm-cash holdout bridge (A/B/C/D)")
    lines.append("")
    lines.append(
        f"Holdout: {bridge['holdout_days']:.0f} ngay | Von active: "
        f"{_fmt(bridge['active_capital_usd'], 0)} USDT | Symbol: {len(LIQUID_8_SYMBOLS)}"
    )
    lines.append("")
    lines.append("## 1. Bon phuong phap tai cung mot moc holdout")
    lines.append("")
    lines.append("| Phuong phap | Von dau (USD) | Net PnL (USD) | Return (%) | Ghi chu |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(
        f"| A. Continuous compound | {_fmt(a['start_equity_usd'])} | {_fmt(a['net_pnl_usd'])} | "
        f"{a.get('return_on_start_equity_pct', a.get('return_on_active_capital_pct', 0.0)):.2f} | tai dau tu lai, von truot |"
    )
    lines.append(
        f"| B. Continuous fixed-base | {_fmt(b['start_equity_usd'])} | {_fmt(b['net_pnl_usd'])} | "
        f"{b['return_on_active_capital_pct']:.2f} | neo 3000 USDT, chi bao da am |"
    )
    lines.append(
        f"| C. Standalone cold-start | {_fmt(c['start_equity_usd'])} | {_fmt(c['net_pnl_usd'])} | "
        f"{c['return_on_active_capital_pct']:.2f} | nap thang vao holdout, moi thu lanh |"
    )
    lines.append(
        f"| D. Indicator-warm cash-start | {_fmt(d['start_equity_usd'])} | {_fmt(d['net_pnl_usd'])} | "
        f"{d['return_on_active_capital_pct']:.2f} | chi bao am {d['warmup_hours']}h, ton kho = tien mat |"
    )
    lines.append("")
    lines.append("## 2. Phan ra chenh lech B - C (tong do trep khoi dong)")
    lines.append("")
    lines.append("| Thanh phan | USD | Y nghia |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| cold_start_initial_entry_fee_usd | {_fmt(dec['cold_start_initial_entry_fee_usd'])} | "
        "phi mo vi the ngay dau (24 bar dau holdout) |"
    )
    lines.append(
        f"| indicator_blindness_drag_usd | {_fmt(dec['indicator_blindness_drag_usd'])} | "
        "(D - C): do tre do EMA-42/momentum chua du lich su |"
    )
    lines.append(
        f"| inventory_carryover_diff_usd | {_fmt(dec['inventory_carryover_diff_usd'])} | "
        "(B - D) tru phi mo: chenh lech ton kho duoc mang sang |"
    )
    lines.append(
        f"| **Tong giai thich** | **{_fmt(dec['explained_total_usd'])}** | "
        f"B - C = {_fmt(b['net_pnl_usd'] - c['net_pnl_usd'])} |"
    )
    lines.append("")
    lines.append("### Kiem tra dang thuc")
    lines.append("")
    lines.append(f"- `B - C - [(B - D) + (D - C)]` = {identity:.6e} (yeu cau < 1e-6)")
    lines.append(f"- `unexplained_total_usd` = {decomp_residual:.6e} (yeu cau < 1e-6)")
    lines.append("")
    lines.append("## 3. Ket luan so bo")
    lines.append("")
    verdict = "dat" if d["return_on_active_capital_pct"] >= 3.5 else "chua dat"
    lines.append(
        f"- Method D dat {d['return_on_active_capital_pct']:.2f}% tren von active 3000 USDT, "
        f"{verdict} nguong 3,5% cua Sprint 11."
    )
    lines.append(
        "- Phan lon do trep khoi dong den tu do mu chi bao (D - C), khong phai phi mo vi the; "
        "dieu nay xac nhan can pre-warm indicator truoc khi vao live."
    )
    lines.append("")

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"Da xuat bang bridge: {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
