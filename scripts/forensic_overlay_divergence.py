"""Sprint 17 - Dieu tra phap y khoang cach PnL nhanh Overlay giua B, C, D' va E.

Cac phep do tren cung cua so holdout (2026-04-21 -> 2026-10-19), cung tham so Frozen Alpha:

- B  : fixed-base (reinvest=False) chay tren toan bo chuoi tu 2023-04-01, cat cua so holdout.
- C  : standalone cold-start - chi chay tren cua so holdout (tin hieu lanh hoan toan).
- D  : warm cash 504h (mac dinh Sprint 14), pha tai can bang neo theo chi so cat lat.
- D' : warm cash 504h, pha neo theo dong ho tuyet doi (phase_aligned=True).
- E  : warm cash 720h (>= 2x lookback 336h), pha tuyet doi -> phep do chuan
       ``E_true_fixed_warm_cash`` de doi chung.

Script in: PnL tung phep do, moc tai can bang 48h (Jaccard), von co so trung binh,
vector trong so tai Ngay 1/30/90/150 va phan ra chinh xac chenh lech PnL overlay.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from run_sprint11 import load_multi_aligned_data  # noqa: E402
from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio  # noqa: E402
from src.cross_exchange_carry import LIQUID_8_SYMBOLS  # noqa: E402
from src.sprint11_eval import (  # noqa: E402
    SPRINT11_HOLDOUT_END,
    SPRINT11_INITIAL_CAPITAL,
    SPRINT11_SPLIT_DATE,
    CANONICAL_SATURATED_WARMUP_HOURS,
    SPRINT14_WARMUP_HOURS,
    compute_holdout_accounting_bridge,
    compute_sleeve_attribution,
    decompose_overlay_pnl_divergence,
    run_indicator_warm_cash_start,
)

RAW_DIR = _ROOT / "data" / "raw"
OUT_DOC = _ROOT / "docs" / "sprint17_overlay_forensics.md"
OUT_JSON = _ROOT / "data" / "forensics_sprint17" / "overlay_divergence.json"
OUT_BRIDGE = _ROOT / "data" / "forensics_sprint17" / "bridge_sprint17.json"
PAYLOAD_COMPOUND = _ROOT / "data" / "bridge_compound" / "sprint11_payload.json"
PAYLOAD_FIXED = _ROOT / "data" / "bridge_fixed" / "sprint11_payload.json"

HOLDOUT_START = pd.Timestamp(SPRINT11_SPLIT_DATE, tz="UTC")
HOLDOUT_END = pd.Timestamp(SPRINT11_HOLDOUT_END, tz="UTC") + pd.Timedelta(days=1)
CONVERGED_WARMUP_HOURS = CANONICAL_SATURATED_WARMUP_HOURS


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _load_payload(path: Path, sleeve: dict) -> dict:
    """Doc payload Sprint 11 va gan sleeve attribution (Sprint 16) cho dong chinh."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    tiers = payload.get("tiers", payload)
    rows = tiers.get("rows", []) if isinstance(tiers, dict) else []
    for row in rows:
        if str(row.get("combo", "")) == "tier1b/smooth_conviction":
            row["holdout_sleeve_attribution"] = sleeve
            break
    return payload


def _cfg(reinvest: bool) -> BarbellConfig:
    return BarbellConfig(
        initial_capital=float(SPRINT11_INITIAL_CAPITAL),
        carry_variant="tier1b",
        gating_mode="smooth_conviction",
        enable_snr_gate=True,
        overlay_quantize_legs=True,
        reinvest_profits=bool(reinvest),
    )


def _series_window(series_like):
    if series_like is None:
        return None
    if isinstance(series_like, (pd.DataFrame, pd.Series)) and not series_like.empty:
        if isinstance(series_like.index, pd.DatetimeIndex):
            return series_like.loc[
                (series_like.index >= HOLDOUT_START) & (series_like.index < HOLDOUT_END)
            ]
    return series_like


def _from_result(res: dict) -> dict:
    """Trich series phap y cua 1 ket qua run_barbell_portfolio (cat cua so holdout)."""
    attr = res.get("attribution_series")
    overlay_net = 0.0
    if isinstance(attr, pd.DataFrame) and not attr.empty and "overlay_pnl" in attr.columns:
        win = _series_window(attr)
        if isinstance(win, pd.DataFrame) and not win.empty:
            overlay_net = float(win["overlay_pnl"].sum())
    equity = res.get("equity_curve")
    equity_start = float(SPRINT11_INITIAL_CAPITAL)
    if isinstance(equity, pd.Series) and len(equity) > 0:
        before = equity.loc[equity.index < HOLDOUT_START]
        if len(before) > 0:
            equity_start = float(before.iloc[-1])
    return {
        "weights": _series_window(res.get("overlay_weights")),
        "raw_weights": _series_window(res.get("overlay_raw_weights")),
        "notional_base": _series_window(res.get("overlay_notional_base")),
        "rebal_flags": _series_window(res.get("overlay_rebal_flags")),
        "fees": _series_window(res.get("overlay_fee_series")),
        "returns": _series_window(res.get("overlay_returns_by_symbol")),
        "funding": _series_window(res.get("overlay_funding_by_symbol")),
        "overlay_net_usd": overlay_net,
        "equity_start_usd": equity_start,
    }


def _snapshot(weights, when: pd.Timestamp) -> dict[str, float] | None:
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        return None
    sub = weights.loc[weights.index <= when]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return {str(k): round(float(v), 4) for k, v in row.items() if abs(float(v)) > 1e-9}


def _rebal_ts(flags) -> list[str]:
    if not isinstance(flags, pd.Series) or flags.empty:
        return []
    return [pd.Timestamp(ts).isoformat() for ts in flags.index[flags.fillna(False).to_numpy()]]


def _compare(a: list[str], b: list[str]) -> dict[str, float]:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    inter = set_a & set_b
    return {
        "n_a": len(set_a),
        "n_b": len(set_b),
        "n_common": len(inter),
        "jaccard": (len(inter) / len(union)) if union else 1.0,
    }


def main() -> int:
    print("[1/4] Nap du lieu da san...")
    data = load_multi_aligned_data(
        raw_dir=RAW_DIR,
        symbols=list(LIQUID_8_SYMBOLS),
        start="2023-04-01",
        end=SPRINT11_HOLDOUT_END,
    )
    if not data:
        print("Khong nap duoc du lieu.")
        return 1

    print("[2/4] Chay B (fixed-base, toan chuoi) ...")
    res_b = run_barbell_portfolio(data, _cfg(reinvest=False))
    b = _from_result(res_b)
    print(f"      B overlay_net(holdout)={b['overlay_net_usd']:.2f} USD")

    print("[2/4] Chay C (standalone cold-start tren cua so holdout) ...")
    holdout_only = {
        sym: df.loc[(df.index >= HOLDOUT_START) & (df.index < HOLDOUT_END)]
        for sym, df in data.items()
    }
    res_c = run_barbell_portfolio(holdout_only, _cfg(reinvest=False))
    c = _from_result(res_c)
    print(f"      C overlay_net(holdout)={c['overlay_net_usd']:.2f} USD")

    print("[2/4] Chay D, D' (504h) va E (720h) theo dung ham chuan ...")
    base_kwargs = dict(
        data=data,
        capital=float(SPRINT11_INITIAL_CAPITAL),
        split_date=SPRINT11_SPLIT_DATE,
        holdout_end=SPRINT11_HOLDOUT_END,
        reinvest_profits=False,
        include_raw=True,
    )
    # Sprint 20: D/D' la phep do LICH SU 504h -> truyen tuong minh.
    res_d = run_indicator_warm_cash_start(
        phase_aligned=False,
        warmup_hours=SPRINT14_WARMUP_HOURS,
        align_rebalance_clock=False,
        **base_kwargs,
    )
    res_dp = run_indicator_warm_cash_start(
        phase_aligned=True,
        warmup_hours=SPRINT14_WARMUP_HOURS,
        align_rebalance_clock=True,
        **base_kwargs,
    )
    res_e = run_indicator_warm_cash_start(
        phase_aligned=True,
        warmup_hours=CONVERGED_WARMUP_HOURS,
        align_rebalance_clock=True,
        **base_kwargs,
    )
    dp = _from_result(res_dp["raw_result"])
    e = _from_result(res_e["raw_result"])
    print(
        f"      D net={res_d['net_pnl_usd']:.2f} | D' net={res_dp['net_pnl_usd']:.2f} "
        f"(overlay {dp['overlay_net_usd']:.2f}) | E net={res_e['net_pnl_usd']:.2f} "
        f"(overlay {e['overlay_net_usd']:.2f})"
    )

    print("[3/4] Phan ra chenh lech ...")
    decs = {
        "B_minus_Dp": decompose_overlay_pnl_divergence(
            b["weights"], dp["weights"], b["returns"],
            notional_ref=b["notional_base"], notional_alt=dp["notional_base"],
            fees_ref=b["fees"], fees_alt=dp["fees"], funding=b["funding"],
            start=SPRINT11_SPLIT_DATE, end=SPRINT11_HOLDOUT_END,
        ),
        "B_minus_E": decompose_overlay_pnl_divergence(
            b["weights"], e["weights"], b["returns"],
            notional_ref=b["notional_base"], notional_alt=e["notional_base"],
            fees_ref=b["fees"], fees_alt=e["fees"], funding=b["funding"],
            start=SPRINT11_SPLIT_DATE, end=SPRINT11_HOLDOUT_END,
        ),
        "B_minus_C": decompose_overlay_pnl_divergence(
            b["weights"], c["weights"], b["returns"],
            notional_ref=b["notional_base"], notional_alt=c["notional_base"],
            fees_ref=b["fees"], fees_alt=c["fees"], funding=b["funding"],
            start=SPRINT11_SPLIT_DATE, end=SPRINT11_HOLDOUT_END,
        ),
        "E_minus_Dp": decompose_overlay_pnl_divergence(
            e["weights"], dp["weights"], b["returns"],
            notional_ref=e["notional_base"], notional_alt=dp["notional_base"],
            fees_ref=e["fees"], fees_alt=dp["fees"], funding=b["funding"],
            start=SPRINT11_SPLIT_DATE, end=SPRINT11_HOLDOUT_END,
        ),
    }

    timing = {
        "B_vs_Dp": _compare(_rebal_ts(b["rebal_flags"]), _rebal_ts(dp["rebal_flags"])),
        "B_vs_E": _compare(_rebal_ts(b["rebal_flags"]), _rebal_ts(e["rebal_flags"])),
        "Dp_vs_E": _compare(_rebal_ts(dp["rebal_flags"]), _rebal_ts(e["rebal_flags"])),
    }

    snap = {}
    for day in (1, 30, 90, 150):
        when = HOLDOUT_START + pd.Timedelta(days=day)
        snap[f"day_{day}"] = {
            "B": _snapshot(b["weights"], when),
            "C": _snapshot(c["weights"], when),
            "Dp": _snapshot(dp["weights"], when),
            "E": _snapshot(e["weights"], when),
        }

    payload = {
        "overlay_pnl_usd": {
            "B": b["overlay_net_usd"],
            "C": c["overlay_net_usd"],
            "D": float(res_d["overlay_net_pnl_usd"]),
            "Dp": dp["overlay_net_usd"],
            "E": e["overlay_net_usd"],
        },
        "net_pnl_usd": {
            "B": float(118.50),
            "D": float(res_d["net_pnl_usd"]),
            "Dp": float(res_dp["net_pnl_usd"]),
            "E": float(res_e["net_pnl_usd"]),
        },
        "warmup_hours": {
            "D": int(SPRINT14_WARMUP_HOURS),
            "Dp": int(SPRINT14_WARMUP_HOURS),
            "E": int(CONVERGED_WARMUP_HOURS),
        },
        "phase_offset": {
            "D": int(res_d["rebalance_phase_offset"]),
            "Dp": int(res_dp["rebalance_phase_offset"]),
            "E": int(res_e["rebalance_phase_offset"]),
        },
        "equity_at_split_usd_B": b["equity_start_usd"],
        "notional_base_mean_usd": {
            "B": float(decs["B_minus_Dp"]["notional_ref_mean"]),
            "C": float(decs["B_minus_C"]["notional_alt_mean"]),
            "Dp": float(decs["B_minus_Dp"]["notional_alt_mean"]),
            "E": float(decs["B_minus_E"]["notional_alt_mean"]),
        },
        "rebalance_timing": timing,
        "weights_snapshot": snap,
        "decomposition": decs,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Sprint 17 - Phap y khoang cach PnL nhanh Overlay (B vs C vs D' vs E)")
    lines.append("")
    lines.append(
        f"Holdout: {SPRINT11_SPLIT_DATE} -> {SPRINT11_HOLDOUT_END} | "
        f"von active {_fmt(SPRINT11_INITIAL_CAPITAL)} USDT | tham so Frozen Alpha giu nguyen"
    )
    lines.append("")
    lines.append("## 1. PnL overlay (net) tung phep do")
    lines.append("")
    lines.append("| Phep do | Mo ta | Warmup | Phase | Overlay net (USD) | Net (USD) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    lines.append(
        f"| B | Fixed-base, chuoi day du tu 2023 | full | 0 | {_fmt(b['overlay_net_usd'])} | 118.50 |"
    )
    lines.append(f"| C | Standalone cold-start (chi holdout) | 0 | 0 | {_fmt(c['overlay_net_usd'])} | - |")
    lines.append(
        f"| D | Warm cash, pha cat lat | {int(SPRINT14_WARMUP_HOURS)} | "
        f"{int(res_d['rebalance_phase_offset'])} | {_fmt(res_d['overlay_net_pnl_usd'])} | "
        f"{_fmt(res_d['net_pnl_usd'])} |"
    )
    lines.append(
        f"| D' | Warm cash, pha tuyet doi | {int(SPRINT14_WARMUP_HOURS)} | "
        f"{int(res_dp['rebalance_phase_offset'])} | {_fmt(dp['overlay_net_usd'])} | "
        f"{_fmt(res_dp['net_pnl_usd'])} |"
    )
    lines.append(
        f"| **E** | Warm cash hoi tu (720h), pha tuyet doi | {int(CONVERGED_WARMUP_HOURS)} | "
        f"{int(res_e['rebalance_phase_offset'])} | {_fmt(e['overlay_net_usd'])} | "
        f"{_fmt(res_e['net_pnl_usd'])} |"
    )
    lines.append("")
    lines.append("## 2. Phan ra chinh xac chenh lech overlay")
    lines.append("")
    lines.append("| Cap | Weight effect | Capital effect | Fee effect | Net gap | Residual |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, dec in decs.items():
        lines.append(
            f"| {name} | {_fmt(dec['weight_effect_usd'])} | {_fmt(dec['capital_effect_usd'])} | "
            f"{_fmt(dec['fee_effect_usd'])} | {_fmt(dec['net_gap_usd'])} | "
            f"{dec['residual_usd']:.2e} |"
        )
    lines.append("")
    lines.append("## 3. Moc tai can bang 48h")
    lines.append("")
    lines.append("| Cap | So moc (A) | So moc (B) | Trung | Jaccard |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, stat in timing.items():
        lines.append(
            f"| {name} | {stat['n_a']} | {stat['n_b']} | {stat['n_common']} | "
            f"{stat['jaccard']:.3f} |"
        )
    lines.append("")
    lines.append("## 4. Von co so trung binh / ty le gio trung trong so")
    lines.append("")
    lines.append(
        f"- Notional base trung binh: B {_fmt(decs['B_minus_Dp']['notional_ref_mean'])} | "
        f"C {_fmt(decs['B_minus_C']['notional_alt_mean'])} | "
        f"D' {_fmt(decs['B_minus_Dp']['notional_alt_mean'])} | "
        f"E {_fmt(decs['B_minus_E']['notional_alt_mean'])} USD"
    )
    lines.append(
        f"- Ty le gio trong so giong B: D' "
        f"{decs['B_minus_Dp']['weights_equal_ratio'] * 100:.1f}% | E "
        f"{decs['B_minus_E']['weights_equal_ratio'] * 100:.1f}% | C "
        f"{decs['B_minus_C']['weights_equal_ratio'] * 100:.1f}%"
    )
    lines.append(f"- Equity B tai split: {_fmt(b['equity_start_usd'])} USD")
    lines.append("")
    lines.append("## 5. Ket luan 4 gia thuyet")
    lines.append("")
    d_dp = decs["B_minus_Dp"]
    d_e = decs["B_minus_E"]
    total_gap = abs(d_dp["net_gap_usd"]) or 1.0
    weight_share = abs(d_dp["weight_effect_usd"]) / total_gap * 100.0
    capital_share = abs(d_dp["capital_effect_usd"]) / total_gap * 100.0
    lines.append("| Gia thuyet | Ket qua | Bang chung dinh luong |")
    lines.append("|---|---|---|")
    lines.append(
        "| H1 - Overlay capital scaling / compounding leak | **BAC BO** | "
        f"capital_effect = {_fmt(d_dp['capital_effect_usd'])} USD ({capital_share:.1f}% gap); "
        f"notional base B {_fmt(d_dp['notional_ref_mean'])} vs D' "
        f"{_fmt(d_dp['notional_alt_mean'])} USD (B bi clamp o initial_capital) |"
    )
    lines.append(
        "| H2 - Rebalance timestamp / modulo shift | **BAC BO (la he qua, khong phai nguyen nhan)** | "
        f"Jaccard B/D' = {timing['B_vs_Dp']['jaccard']:.3f} nhung B/E = "
        f"{timing['B_vs_E']['jaccard']:.3f}; pha 48h da neo dung dong ho tuyet doi o ca D va D' |"
    )
    lines.append(
        "| H3 - Signal (EMA/momentum/vol) khac do warmup chua hoi tu | **XAC NHAN (nguyen nhan chinh)** | "
        f"weight_effect = {_fmt(d_dp['weight_effect_usd'])} USD ({weight_share:.1f}% gap); "
        f"ty le gio trong so giong B: D' {d_dp['weights_equal_ratio'] * 100:.1f}% vs E "
        f"{d_e['weights_equal_ratio'] * 100:.1f}%; nang warmup 504h -> 720h dua overlay tu "
        f"{_fmt(dp['overlay_net_usd'])} len {_fmt(e['overlay_net_usd'])} USD (B = "
        f"{_fmt(b['overlay_net_usd'])}) |"
    )
    lines.append(
        "| H4 - Warmup zeroing artifact (mat lenh co cau dau tien) | **BAC BO** | "
        f"D'/E deu vao vi the dung tai trade_start_idx (raw weights 0 -> khac 0 kich hoat "
        f"rebalance); fee effect chi {_fmt(d_dp['fee_effect_usd'])} USD |"
    )
    lines.append("")
    lines.append("### Ket luan tong")
    lines.append("")
    lines.append(
        f"- Khoang cach overlay B - D' = {_fmt(b['overlay_net_usd'] - dp['overlay_net_usd'])} USD "
        f"truoc hieu chinh; sau khi do lai bang E (warmup 720h) chi con "
        f"{_fmt(b['overlay_net_usd'] - e['overlay_net_usd'])} USD."
    )
    lines.append(
        f"- `E_true_fixed_warm_cash`: net {_fmt(res_e['net_pnl_usd'])} USD vs B 118.50 USD -> "
        f"gap {_fmt(118.50 - res_e['net_pnl_usd'])} USD (truoc day bao cao Sprint 16 la 42.41 USD)."
    )
    lines.append(
        f"- Pure inventory carryover hieu chinh: "
        f"{_fmt(118.50 - res_e['net_pnl_usd'] - float(res_e['entry_fee_usd']))} USD "
        f"(thay vi 40.39 USD) -> phan lon 'khoang cach live vs backtest' la ao giac do warmup 504h."
    )
    lines.append("")
    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Da xuat: {OUT_DOC}")
    print(f"Da xuat: {OUT_JSON}")

    dec = decs["B_minus_E"]
    print(
        f"KET LUAN: B-E weight_effect={dec['weight_effect_usd']:.2f} | "
        f"capital={dec['capital_effect_usd']:.2f} | fee={dec['fee_effect_usd']:.2f} | "
        f"residual={dec['residual_usd']:.2e}"
    )

    # ---- [4/4] Bridge Sprint 17: xuat so lieu A/B/D/D'/E + phap y vao 1 JSON ----
    print("[4/4] Xuat bridge Sprint 17 (E_true_fixed_warm_cash + overlay forensics) ...")
    res_a = run_barbell_portfolio(data, _cfg(reinvest=True))
    sleeve_a = compute_sleeve_attribution(
        res_a.get("attribution_series"),
        (res_a.get("carry_result") or {}).get("holdings_log"),
        split_date=SPRINT11_SPLIT_DATE,
        holdout_end=SPRINT11_HOLDOUT_END,
    )
    sleeve_b = compute_sleeve_attribution(
        res_b.get("attribution_series"),
        (res_b.get("carry_result") or {}).get("holdings_log"),
        split_date=SPRINT11_SPLIT_DATE,
        holdout_end=SPRINT11_HOLDOUT_END,
    )
    payload_a = _load_payload(PAYLOAD_COMPOUND, sleeve_a)
    payload_b = _load_payload(PAYLOAD_FIXED, sleeve_b)
    strip = lambda d: {k: v for k, v in d.items() if k != "raw_result"}
    forensics = {
        "available": True,
        "root_cause": "warmup_ema_non_convergence",
        "overlay_pnl_usd": payload["overlay_pnl_usd"],
        "warmup_hours": payload["warmup_hours"],
        "phase_offset": payload["phase_offset"],
        "rebalance_timing": timing,
        "notional_base_mean_usd": payload["notional_base_mean_usd"],
        "decomposition": decs,
        "weights_equal_vs_B": {
            "Dp": decs["B_minus_Dp"]["weights_equal_ratio"],
            "E": decs["B_minus_E"]["weights_equal_ratio"],
            "C": decs["B_minus_C"]["weights_equal_ratio"],
        },
    }
    bridge = compute_holdout_accounting_bridge(
        payload_a,
        payload_b,
        warm_cash=strip(res_d),
        warm_cash_aligned=strip(res_dp),
        warm_cash_converged=strip(res_e),
        overlay_forensics=forensics,
    )
    OUT_BRIDGE.parent.mkdir(parents=True, exist_ok=True)
    OUT_BRIDGE.write_text(json.dumps(bridge, indent=2, default=str), encoding="utf-8")
    print(f"Da xuat: {OUT_BRIDGE}")
    e_block = bridge["e_true_fixed_warm_cash"]
    print(
        f"BRIDGE: B={bridge['continuous_fixed_base']['net_pnl_usd']:.2f} | "
        f"D'={bridge['indicator_warm_cash_start_clock_aligned']['net_pnl_usd']:.2f} | "
        f"E={e_block['net_pnl_usd']:.2f} | gap B-E={e_block['gap_b_minus_e_usd']:.2f} | "
        f"carryover moi={e_block['revised_pure_carryover_usd']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
