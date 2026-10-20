"""CLI Runner đánh giá toàn diện Sprint 9.

Mục tiêu:
So sánh trực tiếp 4 cấp độ tiến hóa trên toàn bộ 7 giai đoạn 6 tháng (2023-05-07 -> 2026-10-18,
vốn chuẩn 3000.0 USDT, chế độ phí bắt buộc maker_taker_hedge):
1. Tier 1 — Single-Exchange Binance Static Carry (Baseline Sprint 8)
2. Tier 2 — Multi-Exchange Best-Venue Carry (chọn sàn trả funding cao nhất)
3. Tier 3 — Hybrid Cross-Exchange Perp-Perp Carry (ăn chênh lệch giữa 3 sàn + eta 100%-125%)
4. Tier 4 — Full Barbell Portfolio (Hybrid Carry + Core-4 Buffered Momentum Overlay)

Xuất kết quả:
- data/sprint9_tiers_summary.csv
- data/sprint9_periods.csv
- data/sprint9_attribution.csv
- data/sprint9_report_tables.md
In bảng tổng kết và phán quyết GO / NO-GO ASCII-safe ra console.

Ví dụ:
    python run_sprint9.py
    python run_sprint9.py --download
    python run_sprint9.py --capital 3000.0 --out-dir data
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Sequence

# Đảm bảo import được các module từ thư mục gốc
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.config import CORE_4_SYMBOLS
from src.multi_exchange_loader import (
    SUPPORTED_EXCHANGES,
    align_multi_exchange_funding,
    clean_symbol_name,
    fetch_all_exchanges_funding,
    load_funding_by_exchange,
)
from src.sprint7_eval import build_benchmark_returns, to_markdown_table
from src.sprint9_eval import (
    detect_executor_state_machine,
    evaluate_sprint9_go_no_go,
    run_four_tiers_comparison,
)

DEFAULT_START = "2023-05-07"
DEFAULT_END = "2026-10-18"
DEFAULT_CAPITAL = 3000.0
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUT_DIR = "data"
DEFAULT_OUT_MD = "data/sprint9_report_tables.md"
DEFAULT_PERIODS = 7
DEFAULT_PERIOD_DAYS = 180

# Cấu hình logging ASCII-safe
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sprint9")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh cho runner Sprint 9."""
    parser = argparse.ArgumentParser(
        description="Sprint 9: Four Tiers Evolution Comparison & GO/NO-GO Evaluation"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help="Von ban dau USDT (mac dinh 3000.0)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(CORE_4_SYMBOLS),
        help="Danh sach symbol perp (mac dinh CORE_4_SYMBOLS)",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Ngay bat dau UTC (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help="Ngay ket thuc UTC (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Tai va cap nhat du lieu Funding da san tu cac san giao dich",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Thu muc xuat ket qua CSV",
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="Thu muc chua cache du lieu raw Parquet",
    )
    parser.add_argument(
        "--out-md",
        default=DEFAULT_OUT_MD,
        help="Duong dan file Markdown xuat bao cao cac bang",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Bo qua viec xuat file Markdown",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=DEFAULT_PERIODS,
        help="So giai doan cat chuoi thoi gian (mac dinh 7)",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        default=DEFAULT_PERIOD_DAYS,
        help="So ngay moi giai doan (mac dinh 180)",
    )
    return parser.parse_args(argv)


def _clean_name(symbol: str) -> str:
    """Chuyển symbol CCXT perp thành chuỗi an toàn cho tên file parquet."""
    return symbol.replace("/", "_").replace(":", "_")


def _load_parquet_file(candidates: list[Path]) -> pd.DataFrame | None:
    """Tìm và đọc file cache Parquet đầu tiên tồn tại trong danh sách ứng viên."""
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if not df.empty:
                    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                        df = df.set_index("timestamp")
                    idx = pd.DatetimeIndex(df.index)
                    if idx.tz is None:
                        idx = idx.tz_localize("UTC")
                    else:
                        idx = idx.tz_convert("UTC")
                    df.index = idx.as_unit("ns")
                    df.index.name = "datetime"
                    return df.sort_index()
            except Exception as e:
                logger.warning(f"Loi doc file {p}: {e}")
    return None


def main(argv: list[str] | None = None) -> int:
    """Điểm nhập chính cho CLI Runner Sprint 9."""
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("KHOI DONG SPRINT 9 RUNNER (VON: %.1f USDT, CHUAN PHI: maker_taker_hedge)", args.capital)
    logger.info("Chuoi thoi gian: %s den %s (7 giai doan x 180 ngay)", args.start, args.end)
    logger.info("Symbols: %s", ", ".join(args.symbols))
    logger.info("=" * 80)

    # 1. Tai du lieu da san neu duoc yeu cau
    if args.download:
        logger.info("[download] Bat dau tai funding rate da san (binance, bybit, okx)...")
        try:
            fetch_all_exchanges_funding(
                symbols=args.symbols,
                exchanges=SUPPORTED_EXCHANGES,
                start_date=args.start,
                end_date=args.end,
                data_dir=raw_dir,
            )
            logger.info("[download] Hoan tat tai funding rate da san.")
        except Exception as e:
            logger.warning("[download] Canh bao khi tai du lieu da san: %s", e)

    # 2. Doc va dong bo du lieu cho tung symbol
    multi_aligned_data: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")

    for sym in args.symbols:
        clean_raw = _clean_name(sym)
        clean_sym = clean_symbol_name(sym)

        # 2a. Tim file perp
        perp_candidates = [
            raw_dir / f"{clean_raw}_1h.parquet",
            raw_dir / f"{clean_sym}_1h.parquet",
            raw_dir / f"{clean_raw}_PERP_1h.parquet",
            raw_dir / f"{clean_sym}_PERP_1h.parquet",
        ]
        perp_df = _load_parquet_file(perp_candidates)
        if perp_df is None or perp_df.empty:
            logger.warning("Thieu cache Perp cho symbol %s. Bo qua.", sym)
            continue

        # 2b. Tim file spot
        spot_sym_raw = sym.split(":")[0].replace("/", "_")
        spot_sym_clean = clean_symbol_name(sym.split(":")[0])
        spot_candidates = [
            raw_dir / f"{spot_sym_raw}_SPOT_1h.parquet",
            raw_dir / f"{spot_sym_clean}_SPOT_1h.parquet",
            raw_dir / f"{clean_raw}_SPOT_1h.parquet",
            raw_dir / f"{clean_sym}_SPOT_1h.parquet",
        ]
        spot_df = _load_parquet_file(spot_candidates)

        if spot_df is not None and not spot_df.empty and "close" in spot_df.columns:
            spot_close_s = spot_df["close"].reindex(perp_df.index).ffill().bfill().astype("float64")
            perp_df["spot_close"] = spot_close_s
            perp_df["perp_close"] = perp_df["close"].astype("float64")
            perp_df["basis_pct"] = (perp_df["perp_close"] - spot_close_s) / spot_close_s
        else:
            perp_df["perp_close"] = perp_df["close"].astype("float64")
            perp_df["spot_close"] = perp_df["close"].astype("float64")
            perp_df["basis_pct"] = 0.0

        # Cat theo khung thoi gian [start, end]
        perp_df = perp_df[(perp_df.index >= start_ts) & (perp_df.index <= end_ts)]
        if perp_df.empty:
            logger.warning("Du lieu Perp %s sau khi cat [start, end] bi rong. Bo qua.", sym)
            continue

        # 2c. Doc funding da san
        funding_by_ex = load_funding_by_exchange(
            symbol=sym,
            exchanges=SUPPORTED_EXCHANGES,
            data_dir=raw_dir,
        )

        # 2d. Dong bo chong look-ahead
        aligned_df = align_multi_exchange_funding(
            symbol=sym,
            ohlcv_spot_perp_df=perp_df,
            funding_by_exchange=funding_by_ex,
        )
        multi_aligned_data[sym] = aligned_df
        logger.info("Da dong bo thanh cong symbol %s: %d bars 1h", sym, len(aligned_df))

    if not multi_aligned_data:
        logger.error("Khong co du lieu hop le cho bat ky symbol nao. Ket thuc.")
        return 1

    # 3. Benchmark returns tu BTC
    common_index = next(iter(multi_aligned_data.values())).index
    btc_candidates = [
        raw_dir / "BTC_USDT_USDT_1h.parquet",
        raw_dir / "BTCUSDT_1h.parquet",
        raw_dir / "BTC_USDT_USDT_PERP_1h.parquet",
        raw_dir / "BTCUSDT_PERP_1h.parquet",
    ]
    btc_df = _load_parquet_file(btc_candidates)
    if btc_df is not None and "close" in btc_df.columns:
        btc_trimmed = btc_df[(btc_df.index >= start_ts) & (btc_df.index <= end_ts)]
        btc_close_s = btc_trimmed["close"].reindex(common_index).ffill().bfill()
        btc_benchmark_returns = build_benchmark_returns(btc_close_s)
    elif "BTC/USDT:USDT" in multi_aligned_data:
        btc_close_s = multi_aligned_data["BTC/USDT:USDT"]["close"].reindex(common_index).ffill().bfill()
        btc_benchmark_returns = build_benchmark_returns(btc_close_s)
    else:
        btc_benchmark_returns = pd.Series(0.0, index=common_index, name="btc_bh")

    # 4. Chay so sanh 4 Tier
    logger.info("Dang thuc thi mo phong so sanh 4 Tier tren toan chuoi 3.45 nam...")
    tiers_results = run_four_tiers_comparison(
        multi_aligned_data=multi_aligned_data,
        btc_benchmark_returns=btc_benchmark_returns,
        initial_capital=args.capital,
        n_periods=args.n_periods,
        period_days=args.period_days,
    )

    # 5. Xuat cac file CSV va Markdown
    # 5a. data/sprint9_tiers_summary.csv
    summary_rows = [t["row"] for t in tiers_results.values()]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / "sprint9_tiers_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Da xuat: %s", summary_csv_path)

    # 5b. data/sprint9_periods.csv
    period_dfs = []
    for tier_key, t_data in tiers_results.items():
        pdf = t_data["periods"].copy()
        pdf.insert(0, "tier_name", t_data["row"]["tier"])
        pdf.insert(0, "tier_key", tier_key)
        period_dfs.append(pdf)
    all_periods_df = pd.concat(period_dfs, ignore_index=True) if period_dfs else pd.DataFrame()
    periods_csv_path = out_dir / "sprint9_periods.csv"
    all_periods_df.to_csv(periods_csv_path, index=False)
    logger.info("Da xuat: %s", periods_csv_path)

    # 5c. data/sprint9_attribution.csv
    attr_rows: list[dict[str, Any]] = []
    for tier_key, t_data in tiers_results.items():
        res = t_data["result"]
        tier_name = t_data["row"]["tier"]
        attr_s = res.get("attribution_series")

        # Dong ALL
        attr_rows.append({
            "tier_key": tier_key,
            "tier": tier_name,
            "period": "ALL",
            "funding_pnl": float(res.get("funding_pnl", 0.0)),
            "basis_pnl": float(res.get("basis_pnl", 0.0)),
            "overlay_pnl": float(res.get("overlay_pnl", 0.0)),
            "fees": float(res.get("fees", 0.0)),
            "net_pnl": float(t_data["row"].get("net_pnl", 0.0)),
        })

        # Cac dong cho tung ky P1..P7
        if isinstance(attr_s, pd.DataFrame) and not attr_s.empty:
            for p in t_data["periods"].to_dict(orient="records"):
                p_start = p["start"]
                p_end = p["end"]
                p_id = p["period"]
                sub_attr = attr_s[(attr_s.index >= p_start) & (attr_s.index < p_end)]
                f_pnl = float(sub_attr["funding_pnl"].sum()) if "funding_pnl" in sub_attr.columns else 0.0
                b_pnl = float(sub_attr["basis_pnl"].sum()) if "basis_pnl" in sub_attr.columns else 0.0
                o_pnl = float(sub_attr["overlay_pnl"].sum()) if "overlay_pnl" in sub_attr.columns else 0.0
                fees = float(sub_attr["fees"].sum()) if "fees" in sub_attr.columns else 0.0
                n_pnl = (
                    float(sub_attr["net_pnl"].sum())
                    if "net_pnl" in sub_attr.columns
                    else (f_pnl + b_pnl + o_pnl - fees)
                )
                attr_rows.append({
                    "tier_key": tier_key,
                    "tier": tier_name,
                    "period": p_id,
                    "funding_pnl": f_pnl,
                    "basis_pnl": b_pnl,
                    "overlay_pnl": o_pnl,
                    "fees": fees,
                    "net_pnl": n_pnl,
                })
    attr_df = pd.DataFrame(attr_rows)
    attr_csv_path = out_dir / "sprint9_attribution.csv"
    attr_df.to_csv(attr_csv_path, index=False)
    logger.info("Da xuat: %s", attr_csv_path)

    # 5d. data/sprint9_report_tables.md
    state_machine_ok = detect_executor_state_machine()
    gng_tier3 = evaluate_sprint9_go_no_go(
        full_metrics=tiers_results["tier3_hybrid_perp_perp"]["row"],
        period_rows=tiers_results["tier3_hybrid_perp_perp"]["periods"],
        execution_mode="maker_taker_hedge",
        margin_call_count=int(
            tiers_results["tier3_hybrid_perp_perp"]["result"].get("metrics", {}).get("Margin Calls", 0)
        ),
        executor_state_machine_verified=state_machine_ok,
        has_lookahead_bias=False,
    )
    gng_tier4 = evaluate_sprint9_go_no_go(
        full_metrics=tiers_results["tier4_full_barbell"]["row"],
        period_rows=tiers_results["tier4_full_barbell"]["periods"],
        execution_mode="maker_taker_hedge",
        margin_call_count=int(
            tiers_results["tier4_full_barbell"]["result"].get("metrics", {}).get("Margin Calls", 0)
        ),
        executor_state_machine_verified=state_machine_ok,
        has_lookahead_bias=False,
    )

    if not args.no_report:
        md_lines: list[str] = [
            "# Bao Cao Danh Gia 4 Tier Tien Hoa Sprint 9\n",
            f"**Thoi gian danh gia**: `{args.start}` den `{args.end}` (3.45 nam, 7 giai doan 180 ngay)\n",
            f"**Von khoi tao**: `{args.capital:.1f}` USDT | **Che do phi**: `maker_taker_hedge`\n\n",
            "## 1. Bang Tong Hop 4 Tier So Sanh\n",
        ]
        sum_cols = [
            "tier",
            "final_equity",
            "annualized_return_pct",
            "sharpe",
            "maxdd_pct",
            "beta_btc",
            "net_pnl",
            "positive_periods",
        ]
        md_lines.append(to_markdown_table(summary_rows, columns=sum_cols))
        md_lines.append("\n\n## 2. Chi Tiet 7 Giai Doan - Tier 3 (Hybrid Perp-Perp Carry)\n")
        p3_cols = [
            "period",
            "start",
            "end",
            "start_capital",
            "final_equity",
            "net_return_pct",
            "annualized_return_pct",
            "sharpe",
            "maxdd_pct",
            "positive",
        ]
        md_lines.append(
            to_markdown_table(
                tiers_results["tier3_hybrid_perp_perp"]["periods"].to_dict(orient="records"),
                columns=p3_cols,
            )
        )
        md_lines.append("\n\n## 3. Chi Tiet 7 Giai Doan - Tier 4 (Full Barbell Portfolio)\n")
        md_lines.append(
            to_markdown_table(
                tiers_results["tier4_full_barbell"]["periods"].to_dict(orient="records"),
                columns=p3_cols,
            )
        )
        md_lines.append("\n\n## 4. Phan Quyen GO / NO-GO Sprint 9\n")
        gng_rows = [
            {
                "Tier": "Tier 3 (Hybrid Carry)",
                "GO-1 (AnnRet >= 10%)": "PASS" if gng_tier3["cond1_return"] else "FAIL",
                "GO-2 (Consistency 6/7)": "PASS" if gng_tier3["cond2_consistency"] else "FAIL",
                "GO-3 (Risk Sharpe/DD/Beta)": "PASS" if gng_tier3["cond3_risk"] else "FAIL",
                "GO-4 (Execution Sync)": "PASS" if gng_tier3["cond4_execution_sync"] else "FAIL",
                "Verdict": gng_tier3["verdict"],
                "Failed": ", ".join(gng_tier3["failed"]) if gng_tier3["failed"] else "None",
            },
            {
                "Tier": "Tier 4 (Full Barbell)",
                "GO-1 (AnnRet >= 10%)": "PASS" if gng_tier4["cond1_return"] else "FAIL",
                "GO-2 (Consistency 6/7)": "PASS" if gng_tier4["cond2_consistency"] else "FAIL",
                "GO-3 (Risk Sharpe/DD/Beta)": "PASS" if gng_tier4["cond3_risk"] else "FAIL",
                "GO-4 (Execution Sync)": "PASS" if gng_tier4["cond4_execution_sync"] else "FAIL",
                "Verdict": gng_tier4["verdict"],
                "Failed": ", ".join(gng_tier4["failed"]) if gng_tier4["failed"] else "None",
            },
        ]
        md_lines.append(to_markdown_table(gng_rows))
        md_lines.append("\n")

        md_path = Path(args.out_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("Da xuat: %s", md_path)

    # 6. In bang ASCII-safe tong hop 4 Tier
    print("\n" + "=" * 105)
    print("FOUR TIERS EVOLUTION COMPARISON SUMMARY (CAPITAL: %.1f USDT, MAKER_TAKER_HEDGE)" % args.capital)
    print("=" * 105)
    header = (
        f"{'Tier':<42} | {'Final Eq':<9} | {'AnnRet%':<7} | {'Sharpe':<6} | "
        f"{'MaxDD%':<6} | {'Beta':<6} | {'NetPnL$':<8} | {'PosPeriods':<10}"
    )
    print(header)
    print("-" * 105)

    for r in summary_rows:
        t_name = str(r["tier"])[:42]
        print(
            f"{t_name:<42} | "
            f"{float(r['final_equity']):9.2f} | "
            f"{float(r['annualized_return_pct']):7.2f} | "
            f"{float(r['sharpe']):6.2f} | "
            f"{float(r['maxdd_pct']):6.2f} | "
            f"{float(r['beta_btc']):6.2f} | "
            f"{float(r['net_pnl']):8.2f} | "
            f"{str(r['positive_periods']):<10}"
        )
    print("=" * 105)

    # 7. In phan quyet GO / NO-GO cho Tier 3 va Tier 4
    print("\n" + "=" * 80)
    print("SPRINT 9 GO / NO-GO VERDICTS (docs/go_no_go_sprint9.md)")
    print("=" * 80)

    for label, gng in [("TIER 3 (HYBRID PERP-PERP CARRY)", gng_tier3), ("TIER 4 (FULL BARBELL PORTFOLIO)", gng_tier4)]:
        d = gng["details"]
        c1 = "[PASS]" if gng["cond1_return"] else "[FAIL]"
        c2 = "[PASS]" if gng["cond2_consistency"] else "[FAIL]"
        c3 = "[PASS]" if gng["cond3_risk"] else "[FAIL]"
        c4 = "[PASS]" if gng["cond4_execution_sync"] else "[FAIL]"

        print(f"\n--- {label} ---")
        print(f"Condition 1 (AnnRet >= 10.0%)         : {c1} (AnnRet: {d['annualized_return_pct']:.2f}%)")
        print(
            f"Condition 2 (Consistency >= 6/7 pos) : {c2} "
            f"(Pos: {gng['pos_periods']}/{gng['total_periods']}, Holdout: {gng['holdout_annualized_return_pct']:.2f}%, "
            f"Worst: {d['worst_period_return_pct']:.2f}%)"
        )
        print(
            f"Condition 3 (Risk Sharpe>=1.8, DD<=8%): {c3} "
            f"(Sharpe: {d['sharpe']:.2f}, MaxDD: {d['maxdd_pct']:.2f}%, Beta: {d['beta_btc']:.2f}, MarginCalls: {d['margin_call_count']})"
        )
        print(
            f"Condition 4 (Execution Sync & State)  : {c4} "
            f"(Mode: {d['execution_mode']}, StateMachine: {d['executor_state_machine_verified']}, "
            f"NoLookahead: {not d['has_lookahead_bias']})"
        )
        verdict_str = f"--> VERDICT: {gng['verdict']}"
        if gng["failed"]:
            verdict_str += f" (Failed: {', '.join(gng['failed'])})"
        print(verdict_str)

    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
