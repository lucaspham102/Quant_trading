"""Sinh bang phan ra PnL theo nhanh (Carry/Overlay) + ty le giai ngan + cua so 30d/151d.

Phuong phap:
- A = continuous compound (reinvest_profits=True) tren toan bo chuoi.
- B = continuous fixed-base (reinvest_profits=False) tren toan bo chuoi.
- D = indicator-warm cash-start (Method D, moc tai can bang neo nen dau chuoi cat).
- D' = nhu D nhung dong ho 48h da can theo vi tri tuyet doi cua split_date.

Voi moi phuong phap, tai cung cua so holdout, tinh:
- carry_net_pnl_usd, overlay_net_pnl_usd (tong = net_pnl_usd)
- avg_carry_slots_held / carry_utilization_pct (tren toi da 4 slot)
- first_30d_net_pnl_usd (720h dau) / last_151d_net_pnl_usd (phan con lai)

Chay: python scripts/gen_sprint16_attribution.py
Xuat: docs/sprint16_sleeve_and_utilization_audit.md + data/bridge_sprint16/attribution.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_sprint11 import load_multi_aligned_data  # noqa: E402
from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio  # noqa: E402
from src.cross_exchange_carry import LIQUID_8_SYMBOLS  # noqa: E402
from src.sprint11_eval import (  # noqa: E402
    SPRINT11_HOLDOUT_END,
    SPRINT11_INITIAL_CAPITAL,
    SPRINT11_SPLIT_DATE,
    compute_holdout_accounting_bridge,
    compute_sleeve_attribution,
    run_indicator_warm_cash_start,
)

RAW_DIR = _ROOT / "data" / "raw"
OUT_DOC = _ROOT / "docs" / "sprint16_sleeve_and_utilization_audit.md"
OUT_JSON = _ROOT / "data" / "bridge_sprint16" / "attribution.json"
PAYLOAD_COMPOUND = _ROOT / "data" / "bridge_compound" / "sprint11_payload.json"
PAYLOAD_FIXED = _ROOT / "data" / "bridge_fixed" / "sprint11_payload.json"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _barbell_sleeve(data: dict, reinvest: bool) -> tuple[dict, float]:
    """Chay 1 lan barbell tren toan bo chuoi va tra ve sleeve attribution trong holdout."""
    cfg = BarbellConfig(
        initial_capital=float(SPRINT11_INITIAL_CAPITAL),
        carry_variant="tier1b",
        gating_mode="smooth_conviction",
        enable_snr_gate=True,
        overlay_quantize_legs=True,
        reinvest_profits=bool(reinvest),
    )
    res = run_barbell_portfolio(data, cfg)
    sleeve = compute_sleeve_attribution(
        res.get("attribution_series"),
        (res.get("carry_result") or {}).get("holdings_log"),
        split_date=SPRINT11_SPLIT_DATE,
        holdout_end=SPRINT11_HOLDOUT_END,
    )
    return sleeve, float(sleeve.get("net_pnl_usd", 0.0) or 0.0)


def _inject(payload_path: Path, sleeve: dict) -> dict:
    """Doc payload A/B va gan them khoi sleeve_attribution vao row chinh."""
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    tiers = payload.get("tiers", payload)
    rows = tiers.get("rows", []) if isinstance(tiers, dict) else []
    for row in rows:
        if str(row.get("combo", "")) == "tier1b/smooth_conviction":
            row["holdout_sleeve_attribution"] = sleeve
            break
    return payload


def main() -> int:
    print("[1/3] Nap du lieu da san...")
    data = load_multi_aligned_data(
        raw_dir=RAW_DIR,
        symbols=list(LIQUID_8_SYMBOLS),
        start="2023-04-01",
        end=SPRINT11_HOLDOUT_END,
    )
    if not data:
        print("Khong nap duoc du lieu.")
        return 1

    print("[2/3] Chay A (reinvest) va B (fixed-base) tren toan bo chuoi...")
    sleeve_a, net_a = _barbell_sleeve(data, reinvest=True)
    print(f"      A: holdout sleeve net={net_a:.2f} USD")
    sleeve_b, net_b = _barbell_sleeve(data, reinvest=False)
    print(f"      B: holdout sleeve net={net_b:.2f} USD")

    print("[2/3] Chay D va D' (Method D)...")
    kwargs = dict(
        data=data,
        capital=SPRINT11_INITIAL_CAPITAL,
        split_date=SPRINT11_SPLIT_DATE,
        holdout_end=SPRINT11_HOLDOUT_END,
        reinvest_profits=False,
    )
    warm_cash = run_indicator_warm_cash_start(**kwargs)
    warm_cash_aligned = run_indicator_warm_cash_start(phase_aligned=True, **kwargs)
    print(
        f"      D : net={warm_cash['net_pnl_usd']:.2f} | "
        f"D': net={warm_cash_aligned['net_pnl_usd']:.2f}"
    )

    print("[3/3] Tinh bridge + phan ra theo nhanh...")
    compound = _inject(PAYLOAD_COMPOUND, sleeve_a)
    fixed = _inject(PAYLOAD_FIXED, sleeve_b)
    bridge = compute_holdout_accounting_bridge(
        compound, fixed, warm_cash=warm_cash, warm_cash_aligned=warm_cash_aligned
    )
    sleeve_block = bridge["sleeve_attribution"]
    methods = sleeve_block["methods"]
    identity = sleeve_block["identity"]
    delta = sleeve_block["delta_b_vs_d_aligned"]
    dec = bridge["warm_cash_decomposition"]

    lines: list[str] = []
    lines.append("# Sprint 16 - Phan ra PnL theo nhanh, ty le giai ngan va cua so 30d/151d")
    lines.append("")
    lines.append(
        f"Holdout: {bridge['holdout_days']:.0f} ngay | Von active: "
        f"{_fmt(bridge['active_capital_usd'], 0)} USDT | {len(LIQUID_8_SYMBOLS)} symbol | "
        "4 carry slot + overlay top_k=1"
    )
    lines.append("")
    lines.append("## 1. PnL theo nhanh tai cung cua so holdout")
    lines.append("")
    lines.append(
        "| Phuong phap | Net (USD) | Carry (USD) | Overlay (USD) | Khop (USD) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for name in ("A", "B", "C", "D", "D_aligned"):
        item = methods[name]
        gap = identity[name]["sleeve_gap_usd"]
        lines.append(
            f"| {name} | {_fmt(item['net_pnl_usd'])} | {_fmt(item['carry_net_pnl_usd'])} | "
            f"{_fmt(item['overlay_net_pnl_usd'])} | {gap:.2e} |"
        )
    lines.append("")
    lines.append(
        "Ghi chu: C (standalone cold-start) chi co PnL tai lap, khong co attribution theo nhanh "
        "nen cac o Carry/Overlay/utilization de 0.00."
    )
    lines.append("")
    lines.append("## 2. Ty le giai ngan carry (toi da 4 slot)")
    lines.append("")
    lines.append("| Phuong phap | Slot trung binh | Utilization (%) |")
    lines.append("|---|---:|---:|")
    for name in ("A", "B", "C", "D", "D_aligned"):
        item = methods[name]
        lines.append(
            f"| {name} | {item['avg_carry_slots_held']:.2f} | {item['carry_utilization_pct']:.1f} |"
        )
    lines.append("")
    lines.append("## 3. Cua so 30 ngay dau vs 151 ngay con lai")
    lines.append("")
    lines.append("| Phuong phap | 30d dau (USD) | 151d sau (USD) | Tong (USD) | Khop (USD) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ("A", "B", "C", "D", "D_aligned"):
        item = methods[name]
        total = item["first_30d_net_pnl_usd"] + item["last_151d_net_pnl_usd"]
        lines.append(
            f"| {name} | {_fmt(item['first_30d_net_pnl_usd'])} | "
            f"{_fmt(item['last_151d_net_pnl_usd'])} | {_fmt(total)} | "
            f"{identity[name]['window_gap_usd']:.2e} |"
        )
    lines.append("")
    lines.append("## 4. Phan ra khoang cach B - D' (pure inventory carryover)")
    lines.append("")
    lines.append(f"- Tong B - D' = **{_fmt(delta['total_usd'])} USD**")
    lines.append(f"- Nhanh Carry: {_fmt(delta['carry_component_usd'])} USD")
    lines.append(f"- Nhanh Overlay: {_fmt(delta['overlay_component_usd'])} USD")
    lines.append(f"- 30 ngay dau: {_fmt(delta['first_30d_component_usd'])} USD")
    lines.append(f"- 151 ngay con lai: {_fmt(delta['last_151d_component_usd'])} USD")
    lines.append(f"- Residual (carry+overlay - tong): {delta['residual_usd']:.2e} USD")
    lines.append(
        f"- Pure inventory carryover sau phi mo vi the (B - D' - {_fmt(dec['cold_start_initial_entry_fee_usd'])}): "
        f"{_fmt(delta['total_usd'] - dec['cold_start_initial_entry_fee_usd'])} USD"
    )
    lines.append("")
    hysteresis_pct = (
        abs(delta["first_30d_component_usd"]) / abs(delta["total_usd"]) * 100.0
        if abs(delta["total_usd"]) > 0
        else 0.0
    )
    lines.append(
        f"Ket luan: **{hysteresis_pct:.1f}%** cua khoang cach B - D' "
        f"({_fmt(delta['first_30d_component_usd'])} USD) phat sinh trong 30 ngay dau "
        "khi danh muc carry con trong (Hysteresis Cash Drag) va chi "
        f"{_fmt(delta['last_151d_component_usd'])} USD den tu 151 ngay sau."
    )
    lines.append("")
    lines.append("## 5. Doi chieu voi phan ra dong ho 48h (Sprint 15)")
    lines.append("")
    lines.append(f"- Fee mo vi the: {_fmt(dec['cold_start_initial_entry_fee_usd'])} USD")
    lines.append(f"- Indicator blindness: {_fmt(dec['indicator_blindness_drag_usd'])} USD")
    lines.append(f"- Clock phase drift: {_fmt(dec['clock_phase_drift_usd'])} USD")
    lines.append(f"- Pure inventory carryover: {_fmt(dec['pure_inventory_carryover_usd'])} USD")
    lines.append(f"- Tong giai thich: {_fmt(dec['explained_total_usd'])} USD")
    lines.append(f"- Unexplained: {dec['unexplained_total_usd']:.2e} USD")
    lines.append("")

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "bridge": bridge,
                "sleeve_a": sleeve_a,
                "sleeve_b": sleeve_b,
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
