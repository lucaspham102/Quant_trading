"""Runner Sprint 10: 4 Tier Production Barbell với lot-size, True Basis MTM, SNR gate và vol targeting.

Mục tiêu:
Mô phỏng và kiểm định chất lượng toàn diện Sprint 10 trên 7 giai đoạn 180 ngày (2023-05-07 -> 2026-10-18):
1. Tier 1 — Single-Exchange Binance Static Carry (Long-Horizon + True Basis MTM)
2. Tier 2 — Multi-Exchange Best-Venue Carry (True Basis MTM)
3. Tier 3 — Hybrid Cross-Exchange Carry (0.5bps entry + True Basis MTM)
4. Tier 4 — Production Barbell (SNR-Gated Vol-Targeted Overlay)

Xuất 4 file kết quả:
- data/sprint10_tiers_summary.csv
- data/sprint10_periods.csv
- data/sprint10_attribution.csv
- data/sprint10_report_tables.md

Kiểm định 5 tiêu chí GO/NO-GO cho Tier 3 và Tier 4, đối soát máy trạng thái thực thi Executor Parity.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Đảm bảo import được các module từ thư mục gốc
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

try:
    from src.cross_exchange_carry import CORE_4_SYMBOLS
except ImportError:
    try:
        from src.config import CORE_4_SYMBOLS
    except ImportError:
        CORE_4_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]

from src.multi_exchange_loader import (
    SUPPORTED_EXCHANGES,
    align_multi_exchange_funding,
    clean_symbol_name,
    fetch_all_exchanges_funding,
    load_funding_by_exchange,
)
from src.sprint7_eval import build_benchmark_returns, to_markdown_table
from src.sprint10_eval import (
    build_sprint10_summary_rows,
    check_executor_parity,
    compute_period_contributions,
    evaluate_sprint10_go_no_go,
    run_sprint10_tiers,
)

DEFAULT_START = "2023-05-07"
DEFAULT_END = "2026-10-18"
DEFAULT_CAPITAL = 3000.0
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUT_DIR = "data"
DEFAULT_OUT_MD = "data/sprint10_report_tables.md"
DEFAULT_PERIODS = 7
DEFAULT_PERIOD_DAYS = 180

# Cấu hình logging ASCII-safe
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sprint10")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh cho runner Sprint 10."""
    parser = argparse.ArgumentParser(
        description="Runner Sprint 10: 4 Tier Production Barbell với lot-size, True Basis MTM, SNR gate và vol targeting."
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In thong tin log chi tiet muc DEBUG",
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
                logger.warning("Loi doc file %s: %s", p, e)
    return None


def _load_funding_file(candidates: list[Path]) -> pd.DataFrame | None:
    """Đọc file funding rate từ danh sách đường dẫn ứng viên."""
    return _load_parquet_file(candidates)


def main(argv: list[str] | None = None) -> int:
    """Điểm nhập chính cho CLI Runner Sprint 10."""
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(
        "KHOI DONG SPRINT 10 RUNNER (VON: %.1f USDT, CHUAN PHI: maker_taker_hedge)",
        args.capital,
    )
    logger.info("Chuoi thoi gian: %s den %s (%d giai doan x %d ngay)", args.start, args.end, args.n_periods, args.period_days)
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

    # 4. Chay mo phong 4 Tier Sprint 10
    logger.info("Dang thuc thi mo phong 4 Tier Production Barbell...")
    tiers_results = run_sprint10_tiers(
        multi_aligned_data=multi_aligned_data,
        btc_benchmark_returns=btc_benchmark_returns,
        initial_capital=args.capital,
        n_periods=args.n_periods,
        period_days=args.period_days,
    )

    # 5. Xuat 4 file ket qua
    # 5a. sprint10_tiers_summary.csv
    summary_rows = build_sprint10_summary_rows(tiers_results)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / "sprint10_tiers_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Da xuat: %s", summary_csv_path)

    # 5b. sprint10_periods.csv
    period_dfs: list[pd.DataFrame] = []
    for tier_key, t_data in tiers_results.items():
        pdf = t_data["periods"].copy()
        tier_name = t_data.get("tier_name", t_data.get("row", {}).get("tier_name", tier_key))
        pdf.insert(0, "tier_name", tier_name)
        pdf.insert(0, "tier_key", tier_key)
        period_dfs.append(pdf)
    all_periods_df = pd.concat(period_dfs, ignore_index=True) if period_dfs else pd.DataFrame()
    periods_csv_path = out_dir / "sprint10_periods.csv"
    all_periods_df.to_csv(periods_csv_path, index=False)
    logger.info("Da xuat: %s", periods_csv_path)

    # 5c. sprint10_attribution.csv
    attr_rows: list[dict[str, Any]] = []
    for tier_key, t_data in tiers_results.items():
        res = t_data["result"]
        tier_name = t_data.get("tier_name", t_data.get("row", {}).get("tier_name", tier_key))

        # Dong ALL
        all_c = compute_period_contributions(res, None, None)
        f_all = all_c["funding_pnl"] if all_c["funding_pnl"] != 0.0 else float(res.get("funding_pnl", 0.0))
        b_all = all_c["basis_pnl"] if all_c["basis_pnl"] != 0.0 else float(res.get("basis_pnl", 0.0))
        o_all = all_c["overlay_pnl"] if all_c["overlay_pnl"] != 0.0 else float(res.get("overlay_pnl", 0.0))
        fees_all = all_c["fees"] if all_c["fees"] != 0.0 else float(res.get("fees", 0.0))
        net_all = float(t_data.get("row", {}).get("net_pnl", 0.0))
        if net_all == 0.0 and all_c["net_pnl"] != 0.0:
            net_all = all_c["net_pnl"]
        elif net_all == 0.0:
            net_all = f_all + b_all + o_all - fees_all

        attr_rows.append({
            "tier_key": tier_key,
            "tier": tier_name,
            "period": "ALL",
            "funding_pnl": f_all,
            "basis_pnl": b_all,
            "overlay_pnl": o_all,
            "fees": fees_all,
            "net_pnl": net_all,
        })

        # Cac dong tung ky
        pdf = t_data["periods"]
        if isinstance(pdf, pd.DataFrame) and not pdf.empty:
            for p in pdf.to_dict(orient="records"):
                p_start = p.get("start")
                p_end = p.get("end")
                p_id = p.get("period")
                c = compute_period_contributions(res, p_start, p_end)
                f_pnl = c["funding_pnl"]
                b_pnl = c["basis_pnl"]
                o_pnl = c["overlay_pnl"]
                fees = c["fees"]
                n_pnl = c["net_pnl"] if c["net_pnl"] != 0.0 else (f_pnl + b_pnl + o_pnl - fees)
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
    attr_csv_path = out_dir / "sprint10_attribution.csv"
    attr_df.to_csv(attr_csv_path, index=False)
    logger.info("Da xuat: %s", attr_csv_path)

    # 5d. sprint10_report_tables.md
    tier4_key = (
        "tier4_production_barbell"
        if "tier4_production_barbell" in tiers_results
        else "tier4_barbell_production"
    )
    gng_tier3 = evaluate_sprint10_go_no_go(tiers_results["tier3_hybrid_perp_perp"], is_barbell=False)
    gng_tier4 = evaluate_sprint10_go_no_go(tiers_results[tier4_key], is_barbell=True)
    parity_res = check_executor_parity()

    if not args.no_report:
        md_lines: list[str] = [
            "# Bao Cao Danh Gia 4 Tier Production Barbell Sprint 10\n",
            f"**Thoi gian danh gia**: `{args.start}` den `{args.end}` ({args.n_periods} giai doan x {args.period_days} ngay)\n",
            f"**Von khoi tao**: `{args.capital:.1f}` USDT | **Che do phi**: `maker_taker_hedge`\n\n",
            "## 1. Bang Tong Hop 4 Tier Sprint 10\n",
        ]

        # (a) Bang 4 Tier
        summary_md_rows = []
        for r in summary_rows:
            summary_md_rows.append({
                "Tier": r.get("tier_name", r.get("tier")),
                "FinalEq": f"{float(r.get('final_equity', 0.0)):.2f}",
                "AnnRet%": f"{float(r.get('annualized_return_pct', 0.0)):.2f}%",
                "Sharpe": f"{float(r.get('sharpe', 0.0)):.2f}",
                "MaxDD%": f"{float(r.get('maxdd_pct', 0.0)):.2f}%",
                "Beta": f"{float(r.get('beta_btc', 0.0)):.4f}",
                "NetPnL$": f"{float(r.get('net_pnl', 0.0)):.2f}",
                "PosPeriods": f"{r.get('positive_periods', 0)}/{r.get('total_periods', 0)}",
                "BasisMTMstd": f"{float(r.get('basis_mtm_std', 0.0)):.6f}",
                "OverlayActive": f"{float(r.get('overlay_active_ratio', 0.0)):.2%}",
            })
        cols_a = [
            "Tier",
            "FinalEq",
            "AnnRet%",
            "Sharpe",
            "MaxDD%",
            "Beta",
            "NetPnL$",
            "PosPeriods",
            "BasisMTMstd",
            "OverlayActive",
        ]
        md_lines.append(to_markdown_table(summary_md_rows, columns=cols_a))

        # (b) Bang 7 ky cua Tier 4
        md_lines.append("\n\n## 2. Chi Tiet 7 Giai Doan - Tier 4 (Production Barbell)\n")
        t4_data = tiers_results[tier4_key]
        t4_period_rows = []
        if isinstance(t4_data.get("periods"), pd.DataFrame):
            for r in t4_data["periods"].to_dict(orient="records"):
                t4_period_rows.append({
                    "period": r.get("period"),
                    "net_return_pct": f"{float(r.get('net_return_pct', 0.0)):.2f}%",
                    "annualized_return_pct": f"{float(r.get('annualized_return_pct', 0.0)):.2f}%",
                    "sharpe": f"{float(r.get('sharpe', 0.0)):.2f}",
                    "maxdd_pct": f"{float(r.get('maxdd_pct', 0.0)):.2f}%",
                })
        cols_b = ["period", "net_return_pct", "annualized_return_pct", "sharpe", "maxdd_pct"]
        md_lines.append(to_markdown_table(t4_period_rows, columns=cols_b))

        # (c) Bang attribution ALL 4 Tier
        md_lines.append("\n\n## 3. Phan Bo Loi Nhuan Attribution ALL (4 Tier)\n")
        attr_all_rows = []
        for t_k, t_d in tiers_results.items():
            r_res = t_d["result"]
            t_nm = t_d.get("tier_name", t_k)
            c_all = compute_period_contributions(r_res)
            f_val = c_all["funding_pnl"] if c_all["funding_pnl"] != 0.0 else float(r_res.get("funding_pnl", 0.0))
            b_val = c_all["basis_pnl"] if c_all["basis_pnl"] != 0.0 else float(r_res.get("basis_pnl", 0.0))
            o_val = c_all["overlay_pnl"] if c_all["overlay_pnl"] != 0.0 else float(r_res.get("overlay_pnl", 0.0))
            fe_val = c_all["fees"] if c_all["fees"] != 0.0 else float(r_res.get("fees", 0.0))
            ne_val = float(t_d.get("row", {}).get("net_pnl", 0.0))
            if ne_val == 0.0:
                ne_val = c_all["net_pnl"] if c_all["net_pnl"] != 0.0 else (f_val + b_val + o_val - fe_val)
            attr_all_rows.append({
                "Tier": t_nm,
                "funding": f"{f_val:.2f}",
                "basis": f"{b_val:.2f}",
                "overlay": f"{o_val:.2f}",
                "fees": f"{fe_val:.2f}",
                "net": f"{ne_val:.2f}",
            })
        cols_c = ["Tier", "funding", "basis", "overlay", "fees", "net"]
        md_lines.append(to_markdown_table(attr_all_rows, columns=cols_c))

        # Phan quyet GO/NO-GO
        md_lines.append("\n\n## 4. Phan Quyen GO / NO-GO Sprint 10\n")
        gng_md_rows = [
            {
                "Tier": "Tier 3 (Hybrid Carry)",
                "GO-1 (Return)": "PASS" if gng_tier3["conditions"]["cond1_return"]["passed"] else "FAIL",
                "GO-2 (Consistency)": "PASS" if gng_tier3["conditions"]["cond2_consistency"]["passed"] else "FAIL",
                "GO-3 (Risk/Beta)": "PASS" if gng_tier3["conditions"]["cond3_risk"]["passed"] else "FAIL",
                "GO-4 (Regime/MTM)": "PASS" if gng_tier3["conditions"]["cond4_regime_and_mtm"]["passed"] else "FAIL",
                "GO-5 (Parity)": "PASS" if gng_tier3["conditions"]["cond5_executor_parity"]["passed"] else "FAIL",
                "Verdict": gng_tier3["verdict"],
                "Failed": ", ".join(gng_tier3["failed"]) if gng_tier3["failed"] else "None",
            },
            {
                "Tier": "Tier 4 (Production Barbell)",
                "GO-1 (Return)": "PASS" if gng_tier4["conditions"]["cond1_return"]["passed"] else "FAIL",
                "GO-2 (Consistency)": "PASS" if gng_tier4["conditions"]["cond2_consistency"]["passed"] else "FAIL",
                "GO-3 (Risk/Beta)": "PASS" if gng_tier4["conditions"]["cond3_risk"]["passed"] else "FAIL",
                "GO-4 (Regime/MTM)": "PASS" if gng_tier4["conditions"]["cond4_regime_and_mtm"]["passed"] else "FAIL",
                "GO-5 (Parity)": "PASS" if gng_tier4["conditions"]["cond5_executor_parity"]["passed"] else "FAIL",
                "Verdict": gng_tier4["verdict"],
                "Failed": ", ".join(gng_tier4["failed"]) if gng_tier4["failed"] else "None",
            },
        ]
        md_lines.append(to_markdown_table(gng_md_rows))
        md_lines.append("\n")

        md_path = Path(args.out_md)
        if args.out_md == DEFAULT_OUT_MD and args.out_dir != DEFAULT_OUT_DIR:
            md_path = out_dir / "sprint10_report_tables.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("Da xuat: %s", md_path)

    # 6. In bang ASCII-safe tong hop 4 Tier
    print("\n" + "=" * 120)
    print(f"SPRINT 10 FOUR TIERS PRODUCTION BARBELL SUMMARY (CAPITAL: {args.capital:.1f} USDT, MAKER_TAKER_HEDGE)")
    print("=" * 120)
    header = (
        f"{'Tier':<42} | {'Final Eq':<9} | {'AnnRet%':<7} | {'Sharpe':<6} | "
        f"{'MaxDD%':<6} | {'Beta':<6} | {'NetPnL$':<8} | {'PosPeriods':<10} | {'BasisStd':<10} | {'Overlay':<7}"
    )
    print(header)
    print("-" * 120)
    for r in summary_rows:
        t_name = str(r.get("tier_name", r.get("tier")))[:42]
        pos_str = f"{r.get('positive_periods', 0)}/{r.get('total_periods', 0)}"
        print(
            f"{t_name:<42} | "
            f"{float(r.get('final_equity', 0.0)):9.2f} | "
            f"{float(r.get('annualized_return_pct', 0.0)):7.2f} | "
            f"{float(r.get('sharpe', 0.0)):6.2f} | "
            f"{float(r.get('maxdd_pct', 0.0)):6.2f} | "
            f"{float(r.get('beta_btc', 0.0)):6.2f} | "
            f"{float(r.get('net_pnl', 0.0)):8.2f} | "
            f"{pos_str:<10} | "
            f"{float(r.get('basis_mtm_std', 0.0)):10.6f} | "
            f"{float(r.get('overlay_active_ratio', 0.0)):6.1%}"
        )
    print("=" * 120)

    # 7. In phan quyet GO / NO-GO day du 5 dieu kien cho Tier 3 va Tier 4
    print("\n" + "=" * 90)
    print("SPRINT 10 GO / NO-GO VERDICTS (docs/go_no_go_sprint10.md)")
    print("=" * 90)

    eval_targets = [
        ("TIER 3 (HYBRID CROSS-EXCHANGE PERP-PERP CARRY)", gng_tier3),
        ("TIER 4 (PRODUCTION BARBELL PORTFOLIO)", gng_tier4),
    ]

    for label, gng in eval_targets:
        print(f"\n--- {label} ---")
        for cond_key, cond_val in gng["conditions"].items():
            tag = "[PASS]" if cond_val["passed"] else "[FAIL]"
            cond_label = {
                "cond1_return": "GO-1 (AnnRet >= 10.0%)",
                "cond2_consistency": "GO-2 (Consistency 6/7)",
                "cond3_risk": "GO-3 (Sharpe/DD/Beta)",
                "cond4_regime_and_mtm": "GO-4 (Regime Gate & MTM)",
                "cond5_executor_parity": "GO-5 (Executor Parity)",
            }.get(cond_key, cond_key)
            print(f"  {cond_label:<28}: {tag} | {cond_val['detail']}")
        verdict_str = f"--> VERDICT: {gng['verdict']}"
        if gng["failed"]:
            verdict_str += f" (Failed: {', '.join(gng['failed'])})"
        print(verdict_str)

    # 8. In khoi EXECUTOR PARITY tu check_executor_parity()
    print("\n" + "=" * 90)
    print("EXECUTOR PARITY VERIFICATION (check_executor_parity)")
    print("=" * 90)
    for check_name, passed in parity_res["checks"].items():
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {check_name:<30}: {tag}")
    overall_tag = "[PASS] Tat ca thanh phan dong bo" if parity_res["passed"] else f"[FAIL] Thieu: {', '.join(parity_res['missing'])}"
    print(f"--> OVERALL EXECUTOR PARITY   : {overall_tag}")
    print("=" * 90 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
