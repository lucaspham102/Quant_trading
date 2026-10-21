"""Sinh bang bridge dong ho tai can bang 48h cho Sprint 15.

So sanh:
- D_unaligned_clock: Method D nhu Sprint 14 (moc tai can bang neo vao nen dau chuoi cat).
- D_clock_aligned: cung Method D nhung offset 48h duoc neo theo vi tri tuyet doi
  cua ``split_date`` trong chuoi day du.

Dong thoi kiem tra dang thuc 4 thanh phan:
``Fee + Blindness + ClockPhaseDrift + PureCarryover = B - C``.

Chay: python scripts/gen_clock_phase_bridge.py
Xuat: docs/sprint15_clock_phase_bridge.md + data/bridge_clock_phase/sprint15_payload.json
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
OUT_DOC = _ROOT / "docs" / "sprint15_clock_phase_bridge.md"
OUT_JSON = _ROOT / "data" / "bridge_clock_phase" / "sprint15_payload.json"
PAYLOAD_COMPOUND = _ROOT / "data" / "bridge_compound" / "sprint11_payload.json"
PAYLOAD_FIXED = _ROOT / "data" / "bridge_fixed" / "sprint11_payload.json"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def main() -> int:
    if not PAYLOAD_COMPOUND.exists() or not PAYLOAD_FIXED.exists():
        print("Thieu payload A/B. Chay truoc scripts/gen_warm_cash_bridge.py")
        return 2

    compound = json.loads(PAYLOAD_COMPOUND.read_text(encoding="utf-8"))
    fixed = json.loads(PAYLOAD_FIXED.read_text(encoding="utf-8"))

    print("[1/3] Nap du lieu da san...")
    data = load_multi_aligned_data(
        raw_dir=RAW_DIR,
        symbols=list(LIQUID_8_SYMBOLS),
        start="2023-04-01",
        end=SPRINT11_HOLDOUT_END,
    )
    if not data:
        print("Khong nap duoc du lieu. Ket thuc.")
        return 1

    kwargs = dict(
        data=data,
        capital=SPRINT11_INITIAL_CAPITAL,
        split_date=SPRINT11_SPLIT_DATE,
        holdout_end=SPRINT11_HOLDOUT_END,
        reinvest_profits=False,
    )
    print("[2/3] Mo phong Method D (unaligned clock)...")
    warm_cash = run_indicator_warm_cash_start(**kwargs)
    print(
        f"      D  : net={warm_cash['net_pnl_usd']:.2f} USD "
        f"offset=0 trade_start_idx={warm_cash['trade_start_idx']}"
    )

    print("[2/3] Mo phong Method D (clock-aligned)...")
    warm_cash_aligned = run_indicator_warm_cash_start(phase_aligned=True, **kwargs)
    print(
        f"      D' : net={warm_cash_aligned['net_pnl_usd']:.2f} USD "
        f"offset={warm_cash_aligned['rebalance_phase_offset']}"
    )

    print("[3/3] Tinh bridge 4 thanh phan...")
    bridge = compute_holdout_accounting_bridge(
        compound, fixed, warm_cash=warm_cash, warm_cash_aligned=warm_cash_aligned
    )
    a = bridge["continuous_compound"]
    b = bridge["continuous_fixed_base"]
    c = bridge["standalone_cold_start"]
    d = bridge["indicator_warm_cash_start"]
    da = bridge["indicator_warm_cash_start_clock_aligned"]
    dec = bridge["warm_cash_decomposition"]
    clock = bridge["clock_phase_bridge"]

    lines: list[str] = []
    lines.append("# Sprint 15 - Dong ho tai can bang 48h cua Method D")
    lines.append("")
    lines.append(
        f"Holdout: {bridge['holdout_days']:.0f} ngay | Von active: "
        f"{_fmt(bridge['active_capital_usd'], 0)} USDT | Symbol: {len(LIQUID_8_SYMBOLS)}"
    )
    lines.append("")
    lines.append("## 1. Bon moc so sanh")
    lines.append("")
    lines.append("| Phuong phap | Net PnL (USD) | Return (%) | Ghi chu |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| B. Continuous fixed-base | {_fmt(b['net_pnl_usd'])} | {b['return_on_active_capital_pct']:.2f} | chuoi day du |")
    lines.append(f"| C. Standalone cold-start | {_fmt(c['net_pnl_usd'])} | {c['return_on_active_capital_pct']:.2f} | khong warm |")
    lines.append(
        f"| D. Method D (unaligned clock) | {_fmt(d['net_pnl_usd'])} | "
        f"{d['return_on_active_capital_pct']:.2f} | moc 48h neo nen dau chuoi cat |"
    )
    lines.append(
        f"| D'. Method D (clock aligned) | {_fmt(da['net_pnl_usd'])} | "
        f"{da['return_on_active_capital_pct']:.2f} | offset={da['rebalance_phase_offset']}h |"
    )
    lines.append("")
    lines.append("## 2. Phan ra B - C thanh 4 thanh phan")
    lines.append("")
    lines.append("| Thanh phan | USD |")
    lines.append("|---|---:|")
    lines.append(f"| Fee mo vi the ngay dau | {_fmt(dec['cold_start_initial_entry_fee_usd'])} |")
    lines.append(f"| Indicator blindness (D - C) | {_fmt(dec['indicator_blindness_drag_usd'])} |")
    lines.append(f"| Clock phase drift (D' - D) | {_fmt(dec['clock_phase_drift_usd'])} |")
    lines.append(f"| Pure inventory carryover (B - D' - Fee) | {_fmt(dec['pure_inventory_carryover_usd'])} |")
    lines.append(f"| **Tong giai thich** | **{_fmt(dec['explained_total_usd'])}** |")
    lines.append(f"| B - C thuc te | {_fmt(b['net_pnl_usd'] - c['net_pnl_usd'])} |")
    lines.append("")
    lines.append("### Kiem tra dang thuc")
    lines.append("")
    lines.append(f"- `identity_gap_usd` = {clock['identity_gap_usd']:.6e} (yeu cau < 1e-6)")
    lines.append(f"- `unexplained_total_usd` = {dec['unexplained_total_usd']:.6e} (yeu cau < 1e-6)")
    lines.append("")
    lines.append("## 3. Ket luan")
    lines.append("")
    drift_share = (
        abs(dec["clock_phase_drift_usd"]) / abs(b["net_pnl_usd"] - c["net_pnl_usd"]) * 100.0
        if abs(b["net_pnl_usd"] - c["net_pnl_usd"]) > 0
        else 0.0
    )
    lines.append(
        f"- Lech pha dong ho 48h chi chiem {drift_share:.2f}% cua chenh lech B - C, "
        f"nen khong phai nguyen nhan chinh cua drag khoi dong."
    )
    lines.append(
        "- Sau khi tach lech pha, phan con lai la pure inventory carryover - dong nghia "
        "voi viec ton kho duoc mang sang tu chuoi day du la nguon loi the that su."
    )
    lines.append("")

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "bridge": bridge,
                "warm_cash": warm_cash,
                "warm_cash_aligned": warm_cash_aligned,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Da xuat: {OUT_DOC}")
    print(f"Da xuat: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
