"""Runner Sprint 11: Single-Venue Tier 1+ Barbell Backtest va Go/No-Go Verification.

Thuc thi danh gia va kiem dinh toan dien mo hinh Barbell don san Binance Unified:
1. Nap du lieu Spot, Perp va Funding da san cho 8 symbol thanh khoan nhat (LIQUID_8_SYMBOLS).
2. Chay luoi 4 combo tham so:
   - carry_variant in ("tier1a", "tier1b")
   - gating_mode in ("hard_binary", "smooth_conviction")
3. Kiem thu smoke_test_executor cho CarryLiveExecutor tren stub exchange.
4. Danh gia 10 tieu chi GO/NO-GO theo docs/go_no_go_sprint11.md.
5. Xuat sprint11_payload.json va sprint11_report_tables.md.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

# Dam bao import duoc cac module tu thu muc goc
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from run_sprint10 import _clean_name, _load_funding_file, _load_parquet_file
from src.cross_exchange_carry import LIQUID_8_SYMBOLS
from src.multi_exchange_loader import (
    SUPPORTED_EXCHANGES,
    align_multi_exchange_funding,
    clean_symbol_name,
    fetch_all_exchanges_funding,
    load_funding_by_exchange,
)
from src.sprint7_eval import build_benchmark_returns, to_markdown_table
from src.sprint11_eval import (
    build_sprint11_summary_rows,
    evaluate_sprint11_go_no_go,
    run_sprint11_tiers,
    smoke_test_executor,
)

# Cau hinh logging ASCII-safe
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sprint11")


def _get_git_branch() -> str:
    """Lay ten nhanh git hien tai bang lenh git CLI."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _jsonable(obj: Any) -> Any:
    """Chuyen doi de quy doi tuong numpy, pandas va datetime ve kieu du lieu nguyen ban JSON."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    return obj


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phan tich cac doi so dong lenh cho runner Sprint 11."""
    parser = argparse.ArgumentParser(
        description="Runner Sprint 11: Single-Venue Tier 1+ Barbell Backtest & Go/No-Go Verification."
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=3000.0,
        help="Von ban dau USDT (mac dinh 3000.0)",
    )
    parser.add_argument(
        "--start",
        default="2023-04-01",
        help="Ngay bat dau UTC (YYYY-MM-DD, mac dinh 2023-04-01)",
    )
    parser.add_argument(
        "--end",
        default="2026-10-19",
        help="Ngay ket thuc UTC (YYYY-MM-DD, mac dinh 2026-10-19)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Thu muc chua cache du lieu raw Parquet (mac dinh data/raw)",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Thu muc xuat ket qua (mac dinh data)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(LIQUID_8_SYMBOLS),
        help="Danh sach symbol perp (mac dinh LIQUID_8_SYMBOLS)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In thong tin log chi tiet muc DEBUG",
    )
    parser.add_argument(
        "--reinvest",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 = tai dau tu lai (mac dinh), 0 = neo von tinh tai --capital",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Tai va cap nhat du lieu Funding da san tu cac san giao dich",
    )
    return parser.parse_args(argv)


def load_multi_aligned_data(
    raw_dir: Path,
    symbols: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Doc cache Parquet + funding da san va dong bo chong look-ahead cho danh sach symbol."""
    multi_aligned_data: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    for sym in symbols:
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
    return multi_aligned_data


def main(argv: list[str] | None = None) -> int:
    """Diem nhap chinh cho CLI Runner Sprint 11."""
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(
        "KHOI DONG SPRINT 11 RUNNER (VON: %.1f USDT, CHUAN PHI: maker_taker_hedge)",
        args.capital,
    )
    logger.info("Khung thoi gian: %s den %s", args.start, args.end)
    logger.info("Symbols (%d): %s", len(args.symbols), ", ".join(args.symbols))
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
    multi_aligned_data = load_multi_aligned_data(
        raw_dir=raw_dir,
        symbols=list(args.symbols),
        start=args.start,
        end=args.end,
    )
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
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

    # 4. Mo phong Sprint 11 Tiers & Executor Smoke Test
    logger.info("Dang thuc thi mo phong 4 combo Sprint 11 Barbell...")
    tiers = run_sprint11_tiers(
        data=multi_aligned_data,
        capital=args.capital,
        benchmark_returns=btc_benchmark_returns,
        reinvest_profits=bool(args.reinvest),
    )

    logger.info("Dang kiem thu CarryLiveExecutor tren stub exchange...")
    smoke_tmp_path = out_dir / "_sprint11_smoke_state"
    smoke_tmp_path.mkdir(parents=True, exist_ok=True)
    execu = smoke_test_executor(tmp_path=smoke_tmp_path)
    for _row in tiers.get("rows", []):
        if isinstance(_row, dict):
            _row["executor_functional"] = bool(execu.get("executor_functional", False))

    verdict = evaluate_sprint11_go_no_go(tiers, executor_functional=execu.get("executor_functional", False))
    rows = build_sprint11_summary_rows(tiers)

    # 5. Ghi file sprint11_payload.json
    payload_data = {
        "tiers": tiers,
        "executor": execu,
        "verdict": verdict,
        "meta": {
            "capital": float(args.capital),
            "start": str(args.start),
            "end": str(args.end),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_symbols": int(len(multi_aligned_data)),
            "git_branch": _get_git_branch(),
        },
    }
    payload_json_path = out_dir / "sprint11_payload.json"
    clean_payload = _jsonable(payload_data)
    with open(payload_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_payload, f, indent=2, ensure_ascii=False)
    logger.info("Da xuat file payload: %s", payload_json_path)

    # 6. Duyen du lieu tao cac bang bao cao Markdown
    # a) Bang 1 - Ma tran 4 combo
    summary_md_rows: list[dict[str, Any]] = []
    for r in rows:
        summary_md_rows.append({
            "Combo": r.get("combo", ""),
            "AnnRet%": f"{float(r.get('annualized_return_pct', 0.0)):.2f}%",
            "Sharpe": f"{float(r.get('sharpe', 0.0)):.2f}",
            "MaxDD%": f"{float(r.get('maxdd_pct', 0.0)):.2f}%",
            "PositivePeriods": f"{r.get('positive_periods', 0)}/7",
            "HoldoutRet%": f"{float(r.get('holdout_return_pct', 0.0)):.2f}%",
            "CarryPnL$": f"{float(r.get('carry_pnl', 0.0)):.2f}",
            "OverlayNet$": f"{float(r.get('overlay_net_pnl', 0.0)):.2f}",
            "BasisStd": f"{float(r.get('basis_mtm_std', 0.0)):.6f}",
            "BetaBTC": f"{float(r.get('beta_btc', 0.0)):.4f}",
            "AccountingGap": f"{float(r.get('accounting_gap', 0.0)):.2e}",
            "Executor": "PASS" if r.get("executor_functional") else "FAIL",
        })

    # b) Bang 2 - 7 ky luoi cua primary combo
    primary_combo_name = str(verdict.get("primary_combo", "tier1b/smooth_conviction"))
    primary_row: dict[str, Any] | None = None
    for r in tiers.get("rows", []):
        if isinstance(r, dict) and r.get("combo") == primary_combo_name:
            primary_row = r
            break
    if primary_row is None and tiers.get("rows"):
        first_r = tiers["rows"][0]
        if isinstance(first_r, dict):
            primary_row = first_r

    periods_raw = primary_row.get("periods", []) if primary_row else []
    table2_rows: list[dict[str, Any]] = []
    if periods_raw and isinstance(periods_raw, list):
        for p in periods_raw:
            table2_rows.append({
                "Ky": p.get("period", ""),
                "Bat dau": str(p.get("start", "")),
                "Ket thuc": str(p.get("end", "")),
                "Loi nhuan (%)": f"{float(p.get('return_pct', 0.0)):.2f}%",
            })
    else:
        table2_rows.append({
            "Ky": "N/A",
            "Bat dau": "N/A",
            "Ket thuc": "N/A",
            "Loi nhuan (%)": "N/A",
        })

    # c) Bang 3 - Holdout
    table3_rows: list[dict[str, Any]] = []
    for r in tiers.get("rows", []):
        if not isinstance(r, dict):
            continue
        table3_rows.append({
            "combo": str(r.get("combo", "")),
            "holdout%": f"{float(r.get('holdout_return_pct', 0.0)):.2f}%",
            "holdout_carry_pnl": f"{float(r.get('holdout_carry_pnl', 0.0)):.2f}",
            "holdout_overlay_net_pnl": f"{float(r.get('holdout_overlay_net_pnl', 0.0)):.2f}",
            "holdout_days": f"{float(r.get('holdout_days', 0.0)):.1f}",
        })

    # d) Bang 4 - GO/NO-GO
    table4_rows: list[dict[str, Any]] = []
    for c in verdict.get("criteria", []):
        val = c.get("value")
        if isinstance(val, float):
            val_str = f"{val:.4f}" if abs(val) < 1.0 else f"{val:.2f}"
        elif isinstance(val, dict):
            val_str = ", ".join(
                f"{k}={v_sub:.2f}" if isinstance(v_sub, (int, float)) else f"{k}={v_sub}"
                for k, v_sub in val.items()
            )
        else:
            val_str = str(val)
        table4_rows.append({
            "name": str(c.get("name", "")),
            "value": val_str,
            "threshold": str(c.get("threshold", "")),
            "passed": "PASS" if c.get("passed") else "FAIL",
        })

    # e) Dong ket luan
    verdict_str = "GO" if verdict.get("go") else "NO-GO"
    reasons_list = verdict.get("reasons", [])

    report_md_lines: list[str] = [
        "# Bao Cao Danh Gia Sprint 11 (Single-Venue Tier 1+ Barbell)\n",
        f"- **Thoi gian kiem dinh**: `{args.start}` den `{args.end}`\n",
        f"- **Von khoi tao**: `{args.capital:.1f}` USDT\n",
        f"- **So symbols hop le**: `{len(multi_aligned_data)}`\n",
        f"- **Primary Combo**: `{primary_combo_name}`\n\n",
        "## Bang 1 - Ma tran 4 combo\n",
        to_markdown_table(summary_md_rows),
        "\n\n## Bang 2 - 7 ky luoi (Primary Combo)\n",
        to_markdown_table(table2_rows),
        "\n\n## Bang 3 - Holdout\n",
        to_markdown_table(table3_rows),
        "\n\n## Bang 4 - GO/NO-GO\n",
        to_markdown_table(table4_rows),
        "\n\n## Ket Luan Phan Quyet\n",
        f"**PHAN QUYET: {verdict_str}**\n",
    ]
    if reasons_list:
        report_md_lines.append("\n**Danh sach ly do khong dat:**\n")
        for reason in reasons_list:
            report_md_lines.append(f"- {reason}")
    report_md_lines.append("\n")

    report_md_path = out_dir / "sprint11_report_tables.md"
    report_md_path.write_text("\n".join(report_md_lines), encoding="utf-8")
    logger.info("Da xuat file bao cao markdown: %s", report_md_path)

    # 7. In console tóm tắt: bảng 1, bảng 4, phán quyết
    print("\n" + "=" * 100)
    print("BANG 1 - MA TRAN 4 COMBO SPRINT 11")
    print("=" * 100)
    print(to_markdown_table(summary_md_rows))

    print("\n" + "=" * 100)
    print("BANG 4 - TIEU CHI GO / NO-GO SPRINT 11")
    print("=" * 100)
    print(to_markdown_table(table4_rows))

    print("\n" + "=" * 100)
    print(f"PHAN QUYET: {verdict_str}")
    if reasons_list:
        print("Ly do NO-GO:")
        for rsn in reasons_list:
            print(f"  - {rsn}")
    print("=" * 100 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
