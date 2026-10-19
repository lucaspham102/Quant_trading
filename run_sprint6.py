"""CLI Runner thực nghiệm và đánh giá nghiệm thu Sprint 6.

Thực hiện toàn bộ quy trình kiểm định Sprint 6:
1. Nạp và đồng bộ dữ liệu OHLCV 1h cùng Funding Rate cho rổ tài sản.
2. Kiểm tra chiến lược Funding Carry đa tài sản trên ma trận:
   Cấu trúc (coinm_1leg, spot_perp_2leg) x Kịch bản phí (zero, maker_optimistic, real_taker).
3. Kiểm tra chiến lược Cross-Sectional Market-Neutral trên ma trận:
   Mode (momentum, reversal) x Lookback (72, 168, 336)h x Rebalance (24, 48)h.
4. Tính toán Newey-West Alpha/Beta vs BTC, Sharpe năm hóa, DSR (K_carry=6, K_xs=12), Drawdown.
5. Đánh giá điều kiện nghiệm thu GO / NO-GO và xuất báo cáo CSV + Markdown.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

from src.carry_strategy import CarryConfig, run_carry_portfolio
from src.cross_sectional import CrossSectionalConfig, run_cross_sectional_portfolio
from src.funding_loader import align_ohlcv_with_funding, fetch_funding_history
from src.research import DEFAULT_SYMBOLS, load_or_fetch_symbol_data, to_markdown
from src.sprint6_eval import (
    annualized_return_pct,
    annualized_sharpe,
    compute_deflated_sharpe_ratio,
    compute_newey_west_alpha_beta,
    evaluate_go_no_go,
    max_drawdown_pct,
)


def _split_list(raw: str | Sequence[str]) -> list[str]:
    """Tách danh sách symbol từ chuỗi phân cách bởi dấu phẩy hoặc list."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Thiết lập và phân tích tham số dòng lệnh CLI cho Sprint 6."""
    parser = argparse.ArgumentParser(
        description="Sprint 6: Đánh giá Funding Carry & Cross-Sectional Market-Neutral"
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Danh sách symbol phân cách bằng dấu phẩy (mặc định DEFAULT_SYMBOLS)",
    )
    parser.add_argument(
        "--start",
        default="2023-05-07",
        help="Ngày bắt đầu (mặc định 2023-05-07)",
    )
    parser.add_argument(
        "--end",
        default="2026-04-21",
        help="Ngày kết thúc (mặc định 2026-04-21 cho DEV; 2026-10-18 cho HOLDOUT)",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Khung thời gian nến (mặc định 1h)",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Thư mục xuất file CSV kết quả (mặc định data)",
    )
    parser.add_argument(
        "--out-md",
        default="docs/sprint6_report.md",
        help="Đường dẫn file Markdown báo cáo nghiệm thu (mặc định docs/sprint6_report.md)",
    )
    parser.add_argument(
        "--skip-carry",
        action="store_true",
        help="Bỏ qua hướng kiểm định Funding Carry",
    )
    parser.add_argument(
        "--skip-xs",
        action="store_true",
        help="Bỏ qua hướng kiểm định Cross-Sectional Market-Neutral",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Điểm thực thi chính cho CLI Runner Sprint 6."""
    args = parse_args(argv)
    symbols = _split_list(args.symbols)

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")

    now_ts = pd.Timestamp.now(tz="UTC")
    days_needed = max(180, int((now_ts - start_ts).total_seconds() / 86400) + 30)

    # 1. Nạp và căn chỉnh dữ liệu OHLCV + Funding cho từng symbol
    aligned_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ohlcv_df = load_or_fetch_symbol_data(sym, args.timeframe, days=days_needed)
        if ohlcv_df is None or ohlcv_df.empty:
            continue

        # Đảm bảo DatetimeIndex UTC ns
        if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
            if "datetime" in ohlcv_df.columns:
                ohlcv_df = ohlcv_df.set_index("datetime")
        idx = pd.DatetimeIndex(ohlcv_df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        idx = idx.as_unit("ns")
        idx.name = "datetime"
        ohlcv_df.index = idx

        # Lọc dữ liệu trong khoảng [start, end]
        filtered = ohlcv_df.loc[(ohlcv_df.index >= start_ts) & (ohlcv_df.index <= end_ts)].copy()
        if filtered.empty:
            continue

        funding_df = fetch_funding_history(sym, start_date=args.start, end_date=args.end)
        aligned = align_ohlcv_with_funding(filtered, funding_df)
        aligned_data[sym] = aligned

    if not aligned_data:
        print("Lỗi: Không tải được dữ liệu cho bất kỳ symbol nào.", file=sys.stderr)
        return 1

    # 2. Xác định benchmark BTC Buy-and-Hold
    btc_sym = next(
        (s for s in aligned_data if "BTC" in s.upper()),
        list(aligned_data.keys())[0],
    )
    btc_df = aligned_data[btc_sym]
    benchmark_returns = btc_df["close"].pct_change().fillna(0.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    carry_rows: list[dict] = []
    xs_rows: list[dict] = []

    # 3. HƯỚNG CARRY: Lưới 2 cấu trúc x 3 kịch bản phí (K_carry = 6)
    if not args.skip_carry:
        structures = ["coinm_1leg", "spot_perp_2leg"]
        fee_scenarios: dict[str, dict[str, float]] = {
            "zero": {"perp_fee": 0.0, "spot_fee": 0.0, "slippage": 0.0},
            "maker_optimistic": {"perp_fee": 0.0002, "spot_fee": 0.0002, "slippage": 0.0001},
            "real_taker": {"perp_fee": 0.0005, "spot_fee": 0.00075, "slippage": 0.0002},
        }
        k_carry = len(structures) * len(fee_scenarios)

        for struct in structures:
            for fee_name, fees in fee_scenarios.items():
                cfg = CarryConfig(
                    structure=struct,
                    perp_fee=fees["perp_fee"],
                    spot_fee=fees["spot_fee"],
                    slippage=fees["slippage"],
                )
                res = run_carry_portfolio(aligned_data, cfg)
                rets = res["returns"]
                eq = res["equity_curve"]
                m = res["metrics"]

                sr = annualized_sharpe(rets, periods_per_year=8760)
                dsr = compute_deflated_sharpe_ratio(rets, num_trials=k_carry, periods_per_year=8760)
                nw = compute_newey_west_alpha_beta(rets, benchmark_returns, periods_per_year=8760)
                mdd = max_drawdown_pct(eq)
                ann_ret = annualized_return_pct(rets, periods_per_year=8760)

                carry_rows.append({
                    "structure": struct,
                    "fee_scenario": fee_name,
                    "total_return_pct": m.get("Total Return %", 0.0),
                    "annualized_return_pct": ann_ret,
                    "sharpe": sr,
                    "max_dd_pct": mdd,
                    "dsr_prob": dsr["dsr_prob"],
                    "sr_star": dsr["sr_star"],
                    "alpha": nw["alpha"],
                    "beta": nw["beta"],
                    "alpha_t": nw["alpha_t"],
                    "beta_t": nw["beta_t"],
                    "alpha_annualized_pct": nw["alpha_annualized_pct"],
                    "total_funding": m.get("Total Funding Collected", 0.0),
                    "total_fees": m.get("Total Fees Paid", 0.0),
                    "time_in_market_pct": m.get("Time in Market %", 0.0),
                    "turnover": m.get("Turnover", 0.0),
                })

        carry_df = pd.DataFrame(carry_rows)
        carry_csv = out_dir / "sprint6_carry.csv"
        carry_df.to_csv(carry_csv, index=False)

    # 4. HƯỚNG CROSS-SECTIONAL: Lưới 2 mode x 3 lookback x 2 rebalance (K_xs = 12)
    if not args.skip_xs and len(aligned_data) >= 2:
        modes = ["momentum", "reversal"]
        lookbacks = [72, 168, 336]
        rebalances = [24, 48]
        k_xs = len(modes) * len(lookbacks) * len(rebalances)

        for mode in modes:
            for lb in lookbacks:
                for rb in rebalances:
                    cfg = CrossSectionalConfig(
                        mode=mode,
                        lookback_hours=lb,
                        rebalance_hours=rb,
                    )
                    res = run_cross_sectional_portfolio(aligned_data, cfg)
                    rets = res["returns"]
                    eq = res["equity_curve"]
                    m = res["metrics"]

                    sr = annualized_sharpe(rets, periods_per_year=8760)
                    dsr = compute_deflated_sharpe_ratio(rets, num_trials=k_xs, periods_per_year=8760)
                    nw = compute_newey_west_alpha_beta(rets, benchmark_returns, periods_per_year=8760)
                    mdd = max_drawdown_pct(eq)
                    ann_ret = annualized_return_pct(rets, periods_per_year=8760)

                    xs_rows.append({
                        "mode": mode,
                        "lookback_hours": lb,
                        "rebalance_hours": rb,
                        "total_return_pct": m.get("Total Return %", 0.0),
                        "annualized_return_pct": ann_ret,
                        "sharpe": sr,
                        "max_dd_pct": mdd,
                        "dsr_prob": dsr["dsr_prob"],
                        "sr_star": dsr["sr_star"],
                        "alpha": nw["alpha"],
                        "beta": nw["beta"],
                        "alpha_t": nw["alpha_t"],
                        "beta_t": nw["beta_t"],
                        "alpha_annualized_pct": nw["alpha_annualized_pct"],
                        "total_funding": m.get("Total Funding Collected", 0.0),
                        "total_fees": m.get("Total Fees Paid", 0.0),
                        "time_in_market_pct": m.get("Time in Market %", 0.0),
                        "turnover": m.get("Turnover", 0.0),
                    })

        xs_df = pd.DataFrame(xs_rows)
        xs_csv = out_dir / "sprint6_xs.csv"
        xs_df.to_csv(xs_csv, index=False)

    # 5. Lập báo cáo Markdown tổng hợp và đánh giá GO / NO-GO
    is_holdout = str(args.end).startswith("2026-10-18")
    dataset_type = "HOLDOUT" if is_holdout else "DEV (Phát triển / Walk-forward)"

    report_lines: list[str] = [
        "# Sprint 6 — Báo cáo nghiệm thu",
        "",
        f"**Cửa sổ dữ liệu:** {dataset_type} ({args.start} → {args.end})",
        f"**Khung thời gian:** {args.timeframe} | **Danh sách symbol ({len(aligned_data)}):** {', '.join(aligned_data.keys())}",
        f"**Benchmark so sánh:** {btc_sym} Buy-and-Hold",
        "",
        "## Định nghĩa kịch bản chi phí:",
        "- `zero`: perp_fee = 0.0, spot_fee = 0.0, slippage = 0.0.",
        "- `maker_optimistic`: perp_fee = 0.0002 (2 bps), spot_fee = 0.0002 (2 bps), slippage = 0.0001 (1 bps).",
        "- `real_taker`: perp_fee = 0.0005 (5 bps), spot_fee = 0.00075 (7.5 bps), slippage = 0.0002 (2 bps).",
        "",
    ]

    # Đánh giá tiêu chí Carry
    if carry_rows:
        carry_df = pd.DataFrame(carry_rows)
        report_lines.extend([
            "## 1. Kết quả chiến lược Funding Carry (Delta-Neutral)",
            f"*Tổng số cấu hình thử nghiệm K = {len(carry_rows)}*",
            "",
            "### Bảng tổng hợp kịch bản chi phí thực tế (real_taker):",
            to_markdown(carry_df[carry_df["fee_scenario"] == "real_taker"]),
            "",
            "### Toàn bộ cấu hình (kèm kịch bản zero & maker_optimistic):",
            to_markdown(carry_df),
            "",
            "### Đánh giá nghiệm thu GO / NO-GO (Hướng Carry):",
        ])

        real_carry = [r for r in carry_rows if r["fee_scenario"] == "real_taker"]
        for row in real_carry:
            eval_res = evaluate_go_no_go(row, direction="carry")
            report_lines.extend([
                f"- **Cấu trúc `{row['structure']}` (real_taker)**:",
                f"  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): {'ĐẠT' if eval_res['cond1_market_neutral'] else 'KHÔNG ĐẠT'} (alpha_t={row['alpha_t']:.2f}, beta={row['beta']:.4f})",
                f"  - Điều kiện 2 (Sharpe >= 1.50): {'ĐẠT' if eval_res['cond2_sharpe'] else 'KHÔNG ĐẠT'} (Sharpe={row['sharpe']:.2f})",
                f"  - Điều kiện 3 (DSR >= 0.95): {'ĐẠT' if eval_res['cond3_dsr'] else 'KHÔNG ĐẠT'} (DSR={row['dsr_prob']:.4f})",
                f"  - Điều kiện 4 (Holdout 180 ngày): {'ĐẠT' if eval_res['cond4_holdout'] else 'CHƯA CHẠY / KHÔNG ĐẠT'}",
                f"  - **Kết luận:** `{eval_res['verdict']}`",
            ])
        report_lines.append("")

    # Đánh giá tiêu chí Cross-Sectional
    if xs_rows:
        xs_df = pd.DataFrame(xs_rows)
        report_lines.extend([
            "## 2. Kết quả chiến lược Cross-Sectional Market-Neutral",
            f"*Tổng số cấu hình thử nghiệm K = {len(xs_rows)}*",
            "",
            "### Bảng tổng hợp cấu hình Cross-Sectional (real_taker):",
            to_markdown(xs_df),
            "",
            "### Đánh giá nghiệm thu GO / NO-GO (Hướng Cross-Sectional):",
        ])

        for row in xs_rows:
            eval_res = evaluate_go_no_go(row, direction="xs")
            report_lines.extend([
                f"- **Cấu hình `{row['mode']}` lookback={row['lookback_hours']}h rebalance={row['rebalance_hours']}h**:",
                f"  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): {'ĐẠT' if eval_res['cond1_market_neutral'] else 'KHÔNG ĐẠT'} (alpha_t={row['alpha_t']:.2f}, beta={row['beta']:.4f})",
                f"  - Điều kiện 2 (Sharpe >= 1.00): {'ĐẠT' if eval_res['cond2_sharpe'] else 'KHÔNG ĐẠT'} (Sharpe={row['sharpe']:.2f})",
                f"  - Điều kiện 3 (DSR >= 0.95): {'ĐẠT' if eval_res['cond3_dsr'] else 'KHÔNG ĐẠT'} (DSR={row['dsr_prob']:.4f})",
                f"  - Điều kiện 4 (Holdout 180 ngày): {'ĐẠT' if eval_res['cond4_holdout'] else 'CHƯA CHẠY / KHÔNG ĐẠT'}",
                f"  - **Kết luận:** `{eval_res['verdict']}`",
            ])
        report_lines.append("")

    out_md_path = Path(args.out_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(report_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
