"""CLI Runner đánh giá Sprint 8 (Dynamic 12-Asset Carry Rotation + Spot-Perp Basis Convergence).

Quy trình:
1. Đọc và đồng bộ dữ liệu Spot, Perp OHLCV (1h) và Funding Rate (8h) cho 12 tài sản
   qua align_spot_perp_funding.
2. Quét lưới K = 12 cấu hình Dynamic Carry khóa cứng trên rổ 12 coin (DYNAMIC_12),
   chọn cấu hình tối ưu theo Sharpe toàn chuỗi và xuất toàn bộ lưới ra data/sprint8_grid.csv.
3. Chạy kiểm định Out-of-Universe trên rổ 8 coin mở rộng (DYNAMIC_8) với cùng bộ tham số tốt nhất.
4. Chạy đối chứng baseline Static Carry 4 coin của Sprint 7 (STATIC_4) trên CORE_4_SYMBOLS.
5. Cắt 7 giai đoạn 180 ngày liên tiếp trên một đường cong vốn nối tiếp duy nhất (Stitched Equity Curve).
6. Phân rã PnL Attribution (Funding vs Basis vs Fee Drag).
7. Kiểm định 4 điều kiện GO / NO-GO khóa cứng theo docs/go_no_go_sprint8.md.
8. Xuất kết quả CSV, bảng Markdown và in báo cáo ASCII-safe ra console.

Ví dụ:
    python run_sprint8.py
    python run_sprint8.py --download --skip-baseline
    python run_sprint8.py --out-md data/sprint8_report_tables.md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# Thêm thư mục gốc vào sys.path để chạy từ bất kỳ đâu
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.carry_strategy import CarryConfig, run_carry_portfolio
from src.config import CORE_4_SYMBOLS, EXTRA_8_SYMBOLS, UNIVERSE_12_SYMBOLS, TradingConfig
from src.data_loader import DataLoader
from src.dynamic_carry_basis import DynamicCarryConfig, run_dynamic_carry_portfolio
from src.funding_loader import (
    align_spot_perp_funding,
    fetch_funding_history,
    fetch_spot_ohlcv,
)
from src.sprint7_eval import build_benchmark_returns, segment_metrics, to_markdown_table
from src.sprint8_eval import (
    build_period_table,
    build_pnl_attribution,
    evaluate_sprint8_go_no_go,
    split_into_periods,
    summarize_periods,
)

DEFAULT_START = "2023-05-07"
DEFAULT_END = "2026-10-18"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUT_DIR = "data"
DEFAULT_OUT_MD = "data/sprint8_report_tables.md"
PERIODS_PER_YEAR = 8760
K_DYNAMIC = 12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh cho runner Sprint 8."""
    parser = argparse.ArgumentParser(
        description="Sprint 8: Dynamic 12-Asset Carry Rotation + Spot-Perp Basis Convergence"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Ngay bat dau UTC (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="Ngay ket thuc UTC (YYYY-MM-DD)")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="Khung thoi gian nen (1h)")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Thu muc chua cache du lieu raw")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Thu muc xuat ket qua CSV")
    parser.add_argument(
        "--out-md",
        nargs="?",
        const=DEFAULT_OUT_MD,
        default=DEFAULT_OUT_MD,
        help="Duong dan xuat bao cao bang Markdown (hoac bo trong de dung mac dinh)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Tai/cap nhat du lieu Perp, Funding va Spot cho ca 12 coin",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=7,
        help="So giai doan cat chuoi thoi gian (mac dinh 7)",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        default=180,
        help="So ngay moi giai doan (mac dinh 180)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Bo qua chay doi chung baseline Static Carry 4 coin cua Sprint 7",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Tiện ích đọc và tiền xử lý dữ liệu
# ---------------------------------------------------------------------------


def _clean_name(symbol: str) -> str:
    """Chuyển symbol CCXT perp thành chuỗi an toàn cho tên file parquet."""
    return symbol.replace("/", "_").replace(":", "_")


def _spot_name(spot_symbol: str) -> str:
    """Chuyển spot symbol thành tên định dạng chuẩn Binance cache."""
    clean = (
        spot_symbol.replace("/", "")
        .replace(":", "")
        .replace("_", "")
        .upper()
        .replace("USDTUSDT", "USDT")
    )
    return clean


def _load_perp(symbol: str, timeframe: str, raw_dir: Path) -> pd.DataFrame:
    """Đọc cache Perp OHLCV; báo lỗi nếu file không tồn tại."""
    path = raw_dir / f"{_clean_name(symbol)}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Thieu cache Perp: {path}. Hay chay voi tham so --download.")
    cfg = TradingConfig(symbol=symbol, timeframe=timeframe, raw_data_dir=raw_dir)
    return DataLoader(config=cfg).load_parquet(path)


def _load_funding(symbol: str, raw_dir: Path) -> pd.DataFrame:
    """Đọc cache Funding Rate 8h; báo lỗi nếu file không tồn tại."""
    path = raw_dir / f"{_clean_name(symbol)}_funding.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Thieu cache Funding: {path}. Hay chay voi tham so --download.")
    return pd.read_parquet(path)


def _load_spot(spot_symbol: str, timeframe: str, raw_dir: Path) -> pd.DataFrame:
    """Đọc cache Spot OHLCV; báo lỗi nếu file không tồn tại."""
    clean = _spot_name(spot_symbol)
    path = raw_dir / f"{clean}_SPOT_{timeframe}.parquet"
    if not path.exists():
        alt_path = raw_dir / f"{spot_symbol.replace('/', '').replace(':', '').upper()}_SPOT_{timeframe}.parquet"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Thieu cache Spot: {path}. Hay chay voi tham so --download.")
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx.as_unit("ns")
    return df


def _trim(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Cắt DataFrame trong khoảng [start, end) UTC."""
    out = df.copy()
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return out[(out.index >= start_ts) & (out.index < end_ts)]


def build_universe_frames(
    symbols: Sequence[str], timeframe: str, raw_dir: Path, start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Dựng dữ liệu đồng bộ Spot-Perp-Funding cho từng symbol trong rổ."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        perp = _load_perp(symbol, timeframe, raw_dir)
        funding = _load_funding(symbol, raw_dir)
        spot_sym = symbol.split(":")[0]
        spot = _load_spot(spot_sym, timeframe, raw_dir)
        aligned = align_spot_perp_funding(perp, spot, funding)
        frames[symbol] = _trim(aligned, start, end)
    return frames


def download_data(
    symbols: Sequence[str], timeframe: str, raw_dir: Path, start: str, end: str
) -> None:
    """Tải/cập nhật dữ liệu Perp + Funding và Spot OHLCV cho tất cả các symbol."""
    import ccxt

    perp_ex = ccxt.binanceusdm({"enableRateLimit": True})
    spot_ex = ccxt.binance({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)

    print(f"[download] Bat dau dong bo du lieu cho {len(symbols)} ma tu {start} den {end}...")
    for symbol in symbols:
        cfg = TradingConfig(symbol=symbol, timeframe=timeframe, raw_data_dir=raw_dir)
        loader = DataLoader(config=cfg, exchange=perp_ex)
        df = loader.update_incremental(default_start_ms=since_ms)
        print(f"[download] perp {symbol}: {len(df)} nen", flush=True)

        fdf = fetch_funding_history(
            symbol, start_date=start, end_date=end, data_dir=str(raw_dir), exchange=perp_ex
        )
        print(f"[download] funding {symbol}: {len(fdf)} ky", flush=True)

        spot_symbol = symbol.split(":")[0]
        sdf = fetch_spot_ohlcv(
            spot_symbol,
            timeframe=timeframe,
            start_date=start,
            end_date=end,
            data_dir=str(raw_dir),
            exchange=spot_ex,
        )
        print(f"[download] spot {spot_symbol}: {len(sdf)} nen", flush=True)


# ---------------------------------------------------------------------------
# Sinh lưới tham số khóa cứng K = 12
# ---------------------------------------------------------------------------


def iter_dynamic_grid() -> Iterable[tuple[str, DynamicCarryConfig, dict[str, Any]]]:
    """Sinh toàn bộ 12 cấu hình Dynamic Carry đã khóa cứng theo docs/go_no_go_sprint8.md.

    Lưới tham số K = 12:
    - top_k in {2, 3} (2)
    - w_basis in {0.0, 0.5, 1.0} (3)
    - execution in {'real_taker', 'passive_limit_maker'} (2)
    - Các tham số khác cố định: structure='spot_perp_3x', ema_span=6, basis_ema_hours=168,
      expected_hold_periods=21, entry_threshold=0.00010, exit_threshold=0.00002,
      min_hold_periods=9, buffer_rank=1, initial_capital=400.0.
    """
    top_k_vals = (2, 3)
    w_basis_vals = (0.0, 0.5, 1.0)
    exec_vals = ("real_taker", "passive_limit_maker")

    for top_k in top_k_vals:
        for w_basis in w_basis_vals:
            for execution in exec_vals:
                cfg = DynamicCarryConfig(
                    structure="spot_perp_3x",
                    execution=execution,
                    initial_capital=400.0,
                    ema_span=6,
                    basis_ema_hours=168,
                    expected_hold_periods=21,
                    w_basis=w_basis,
                    top_k=top_k,
                    entry_threshold=0.00010,
                    exit_threshold=0.00002,
                    min_hold_periods=9,
                    buffer_rank=1,
                    min_switch_advantage_override=None,
                )
                config_id = f"dynamic|k{top_k}|w{w_basis:g}|{execution}"
                params = {
                    "top_k": top_k,
                    "w_basis": w_basis,
                    "execution": execution,
                    "structure": "spot_perp_3x",
                }
                yield config_id, cfg, params


def _benchmark_from(frames: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.Series:
    """Tạo chuỗi lợi nhuận benchmark Buy-and-Hold BTC căn theo index chiến lược."""
    btc_sym = CORE_4_SYMBOLS[0]
    if btc_sym in frames:
        btc_close = frames[btc_sym]["close"].reindex(index).ffill()
    else:
        # Fallback neu ro khong chua BTC
        raise KeyError(f"Khong tim thay {btc_sym} trong frames de lam benchmark.")
    return build_benchmark_returns(btc_close)


# ---------------------------------------------------------------------------
# Luồng chạy chính
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Hàm chạy chính điều phối toàn bộ luồng Sprint 8."""
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_data(UNIVERSE_12_SYMBOLS, args.timeframe, raw_dir, args.start, args.end)

    print("=" * 80)
    print("SPRINT 8: DYNAMIC 12-ASSET CARRY ROTATION + BASIS CONVERGENCE")
    print(f"Data range: {args.start} -> {args.end} | Timeframe: {args.timeframe}")
    print("=" * 80)

    # 1. Tải và đồng bộ dữ liệu Spot-Perp-Funding cho toàn bộ 12 coin
    print("\n[1/5] Dong bo du lieu Spot-Perp-Funding cho 12 coin...")
    frames_12 = build_universe_frames(
        UNIVERSE_12_SYMBOLS, args.timeframe, raw_dir, args.start, args.end
    )
    for sym, df in frames_12.items():
        print(f"  - {sym}: {len(df)} nen dong bo [{df.index[0]} -> {df.index[-1]}]")

    # 2. Quét toàn bộ lưới K = 12 trên rổ 12 coin
    print(f"\n[2/5] Chay luoi K = {K_DYNAMIC} cau hinh tren ro 12 coin (DYNAMIC_12)...")
    grid_rows: list[dict[str, Any]] = []
    best_config_id: str | None = None
    best_config: DynamicCarryConfig | None = None
    best_res_12: dict[str, Any] | None = None
    best_metrics_12: dict[str, Any] | None = None
    best_sharpe = -999.0
    best_ann_ret = -999.0

    btc_sym = CORE_4_SYMBOLS[0]
    sample_index = frames_12[btc_sym].index
    benchmark_all = _benchmark_from(frames_12, sample_index)

    for cfg_id, cfg, params in iter_dynamic_grid():
        res = run_dynamic_carry_portfolio(frames_12, cfg, periods_per_year=PERIODS_PER_YEAR)
        rets = res["returns"]
        bench_sub = benchmark_all.reindex(rets.index).fillna(0.0)
        m = segment_metrics(
            rets,
            bench_sub,
            periods_per_year=PERIODS_PER_YEAR,
            num_trials=K_DYNAMIC,
            initial_capital=cfg.initial_capital,
        )

        row = {
            "config_id": cfg_id,
            "top_k": cfg.top_k,
            "w_basis": cfg.w_basis,
            "execution": cfg.execution,
            "structure": cfg.structure,
            "annualized_return_pct": float(m["annualized_return_pct"]),
            "sharpe": float(m["sharpe"]),
            "maxdd_pct": float(m["maxdd_pct"]),
            "total_return_pct": float(m["net_return_pct"]),
            "dsr_prob": float(m["dsr_prob"]),
            "volatility_ann_pct": float(m["volatility_ann_pct"]),
            "turnover": float(res["turnover"]),
            "funding_pnl": float(res["funding_pnl"]),
            "basis_pnl": float(res["basis_pnl"]),
            "fees": float(res["fees"]),
            "net_pnl": float(res["funding_pnl"] + res["basis_pnl"] - res["fees"]),
            "n_bars": int(m["n_bars"]),
        }
        grid_rows.append(row)

        curr_sharpe = float(m["sharpe"])
        curr_ann_ret = float(m["annualized_return_pct"])
        print(
            f"  {cfg_id:<32} | AnnRet: {curr_ann_ret:6.2f}% | Sharpe: {curr_sharpe:5.2f} "
            f"| MaxDD: {m['maxdd_pct']:5.2f}% | DSR: {m['dsr_prob']:4.2f} | NetPnL: ${row['net_pnl']:6.2f}"
        )

        # Tiêu chí chọn tối ưu: Sharpe cao nhất, tie-break theo Annualized Return
        is_better = (curr_sharpe > best_sharpe) or (
            math.isclose(curr_sharpe, best_sharpe, abs_tol=1e-4) and curr_ann_ret > best_ann_ret
        )
        if is_better:
            best_sharpe = curr_sharpe
            best_ann_ret = curr_ann_ret
            best_config_id = cfg_id
            best_config = cfg
            best_res_12 = res
            best_metrics_12 = m

    # Xuất data/sprint8_grid.csv
    grid_df = pd.DataFrame(grid_rows)
    grid_csv_path = out_dir / "sprint8_grid.csv"
    grid_df.to_csv(grid_csv_path, index=False)
    print(f"\n-> Da xuat toan bo luoi K={K_DYNAMIC} ra: {grid_csv_path}")
    print(f"-> Cau hinh DYNAMIC_12 tot nhat duoc chon: {best_config_id}")

    if best_config is None or best_res_12 is None or best_metrics_12 is None:
        raise RuntimeError("Khong tim thay cau hinh hop le trong luoi Dynamic Carry.")

    # 3. Chạy DYNAMIC_8 (Out-of-Universe) trên rổ 8 coin mở rộng với cùng best_config
    print("\n[3/5] Chay kiem dinh Out-of-Universe tren ro 8 coin mo rong (DYNAMIC_8)...")
    frames_8 = {s: frames_12[s] for s in EXTRA_8_SYMBOLS if s in frames_12}
    res_8 = run_dynamic_carry_portfolio(frames_8, best_config, periods_per_year=PERIODS_PER_YEAR)
    bench_8 = benchmark_all.reindex(res_8["returns"].index).fillna(0.0)
    metrics_8 = segment_metrics(
        res_8["returns"],
        bench_8,
        periods_per_year=PERIODS_PER_YEAR,
        num_trials=K_DYNAMIC,
        initial_capital=best_config.initial_capital,
    )
    print(
        f"  DYNAMIC_8 | AnnRet: {metrics_8['annualized_return_pct']:6.2f}% | "
        f"Sharpe: {metrics_8['sharpe']:5.2f} | MaxDD: {metrics_8['maxdd_pct']:5.2f}% | "
        f"DSR: {metrics_8['dsr_prob']:4.2f} | NetPnL: ${res_8['funding_pnl'] + res_8['basis_pnl'] - res_8['fees']:6.2f}"
    )

    # 4. Chạy STATIC_4 baseline Sprint 7 (nếu không skip)
    res_static: dict[str, Any] | None = None
    metrics_static: dict[str, Any] | None = None
    if not args.skip_baseline:
        print("\n[4/5] Chay doi chung baseline Static Carry 4 coin cua Sprint 7 (STATIC_4)...")
        frames_4 = {s: frames_12[s] for s in CORE_4_SYMBOLS if s in frames_12}
        static_cfg = CarryConfig(
            structure="spot_perp_2leg",
            carry_mode="hysteresis",
            perp_fee=0.0005,
            spot_fee=0.00075,
            slippage=0.0002,
            initial_capital=400.0,
            basis_noise_std=0.0,
            min_hold_periods=3,
        )
        res_static = run_carry_portfolio(frames_4, static_cfg, periods_per_year=PERIODS_PER_YEAR)
        bench_static = benchmark_all.reindex(res_static["returns"].index).fillna(0.0)
        metrics_static = segment_metrics(
            res_static["returns"],
            bench_static,
            periods_per_year=PERIODS_PER_YEAR,
            num_trials=1,
            initial_capital=static_cfg.initial_capital,
        )
        static_pnl = (
            float(res_static["metrics"].get("Total Funding Collected", 0.0))
            + float(res_static["metrics"].get("Total MTM PnL", 0.0))
            - float(res_static["metrics"].get("Total Fees Paid", 0.0))
        )
        print(
            f"  STATIC_4  | AnnRet: {metrics_static['annualized_return_pct']:6.2f}% | "
            f"Sharpe: {metrics_static['sharpe']:5.2f} | MaxDD: {metrics_static['maxdd_pct']:5.2f}% | "
            f"NetPnL: ${static_pnl:6.2f}"
        )
    else:
        print("\n[4/5] Bo qua baseline Static 4 coin (--skip-baseline).")

    # 5. Cắt 7 giai đoạn 180 ngày và xây dựng bảng chi tiết
    print(f"\n[5/5] Cat {args.n_periods} giai doan {args.period_days} ngay tren Stitched Equity Curve...")
    periods = split_into_periods(
        best_res_12["returns"].index, n_periods=args.n_periods, days=args.period_days
    )
    for i, (s, e) in enumerate(periods):
        print(f"  - Ky P{i+1}: {s.strftime('%Y-%m-%d')} den {e.strftime('%Y-%m-%d')}")

    # Bảng 7 kỳ của DYNAMIC_12
    periods_df_12 = build_period_table(
        returns=best_res_12["returns"],
        benchmark=benchmark_all.reindex(best_res_12["returns"].index),
        periods=periods,
        attribution_series=best_res_12["attribution_series"],
        initial_capital=best_config.initial_capital,
        num_trials=K_DYNAMIC,
        periods_per_year=PERIODS_PER_YEAR,
    )
    periods_df_12.insert(0, "universe", "dynamic12")
    periods_df_12.insert(0, "config_id", best_config_id)

    # Bảng 7 kỳ của DYNAMIC_8
    periods_df_8 = build_period_table(
        returns=res_8["returns"],
        benchmark=benchmark_all.reindex(res_8["returns"].index),
        periods=periods,
        attribution_series=res_8["attribution_series"],
        initial_capital=best_config.initial_capital,
        num_trials=K_DYNAMIC,
        periods_per_year=PERIODS_PER_YEAR,
    )
    periods_df_8.insert(0, "universe", "dynamic8")
    periods_df_8.insert(0, "config_id", f"dynamic8|{best_config_id}")

    # Bảng 7 kỳ của STATIC_4 (nếu có)
    periods_df_static: pd.DataFrame | None = None
    if res_static is not None and metrics_static is not None:
        periods_df_static = build_period_table(
            returns=res_static["returns"],
            benchmark=benchmark_all.reindex(res_static["returns"].index),
            periods=periods,
            attribution_series=None,
            initial_capital=400.0,
            num_trials=1,
            periods_per_year=PERIODS_PER_YEAR,
        )
        periods_df_static.insert(0, "universe", "static4")
        periods_df_static.insert(0, "config_id", "static4|spot_perp_2leg|hysteresis")

    # Gộp tất cả bảng periods và xuất ra data/sprint8_periods.csv
    all_periods_dfs = [periods_df_12, periods_df_8]
    if periods_df_static is not None:
        all_periods_dfs.append(periods_df_static)
    all_periods_df = pd.concat(all_periods_dfs, ignore_index=True)
    periods_csv_path = out_dir / "sprint8_periods.csv"
    all_periods_df.to_csv(periods_csv_path, index=False)
    print(f"-> Da xuat chi tiet cac ky ra: {periods_csv_path}")

    # PnL Attribution
    attr_12 = build_pnl_attribution(
        best_res_12["attribution_series"], periods=periods, initial_capital=best_config.initial_capital
    )
    attr_8 = build_pnl_attribution(
        res_8["attribution_series"], periods=periods, initial_capital=best_config.initial_capital
    )

    attr_rows: list[dict[str, Any]] = []
    # Thêm dòng ALL cho dynamic12
    attr_rows.append({
        "config_id": best_config_id,
        "universe": "dynamic12",
        "period": "ALL",
        "funding_pnl": attr_12["funding_pnl"],
        "basis_pnl": attr_12["basis_pnl"],
        "fees": attr_12["fees"],
        "net_pnl": attr_12["net_pnl"],
        "notional_traded": attr_12["notional_traded"],
        "funding_return_pct": attr_12["funding_return_pct"],
        "basis_return_pct": attr_12["basis_return_pct"],
        "fee_drag_pct": attr_12["fee_drag_pct"],
        "net_return_pct": attr_12["net_return_pct"],
    })
    for p in attr_12["periods"]:
        attr_rows.append({
            "config_id": best_config_id,
            "universe": "dynamic12",
            "period": p["period_id"],
            "funding_pnl": p["funding_pnl"],
            "basis_pnl": p["basis_pnl"],
            "fees": p["fees"],
            "net_pnl": p["net_pnl"],
            "notional_traded": p["notional_traded"],
            "funding_return_pct": p["funding_return_pct"],
            "basis_return_pct": p["basis_return_pct"],
            "fee_drag_pct": p["fee_drag_pct"],
            "net_return_pct": p["net_return_pct"],
        })

    # Thêm dòng ALL cho dynamic8
    attr_rows.append({
        "config_id": f"dynamic8|{best_config_id}",
        "universe": "dynamic8",
        "period": "ALL",
        "funding_pnl": attr_8["funding_pnl"],
        "basis_pnl": attr_8["basis_pnl"],
        "fees": attr_8["fees"],
        "net_pnl": attr_8["net_pnl"],
        "notional_traded": attr_8["notional_traded"],
        "funding_return_pct": attr_8["funding_return_pct"],
        "basis_return_pct": attr_8["basis_return_pct"],
        "fee_drag_pct": attr_8["fee_drag_pct"],
        "net_return_pct": attr_8["net_return_pct"],
    })
    for p in attr_8["periods"]:
        attr_rows.append({
            "config_id": f"dynamic8|{best_config_id}",
            "universe": "dynamic8",
            "period": p["period_id"],
            "funding_pnl": p["funding_pnl"],
            "basis_pnl": p["basis_pnl"],
            "fees": p["fees"],
            "net_pnl": p["net_pnl"],
            "notional_traded": p["notional_traded"],
            "funding_return_pct": p["funding_return_pct"],
            "basis_return_pct": p["basis_return_pct"],
            "fee_drag_pct": p["fee_drag_pct"],
            "net_return_pct": p["net_return_pct"],
        })

    attr_df = pd.DataFrame(attr_rows)
    attr_csv_path = out_dir / "sprint8_attribution.csv"
    attr_df.to_csv(attr_csv_path, index=False)
    print(f"-> Da xuat PnL Attribution ra: {attr_csv_path}")

    # Đánh giá 4 điều kiện GO / NO-GO
    live_executor_ready = Path("src/carry_live_executor.py").exists()
    go_no_go = evaluate_sprint8_go_no_go(
        full_12=best_metrics_12,
        full_8=metrics_8,
        period_rows_12=periods_df_12,
        live_executor_ready=live_executor_ready,
        num_trials=K_DYNAMIC,
    )

    # Xuất data/sprint8_summary.csv
    summary_rows = [
        {
            "config_id": best_config_id,
            "universe": "dynamic12",
            "n_symbols": 12,
            "annualized_return_pct": float(best_metrics_12["annualized_return_pct"]),
            "sharpe": float(best_metrics_12["sharpe"]),
            "maxdd_pct": float(best_metrics_12["maxdd_pct"]),
            "net_return_pct": float(best_metrics_12["net_return_pct"]),
            "dsr_prob": float(best_metrics_12["dsr_prob"]),
            "turnover": float(best_res_12["turnover"]),
            "funding_pnl": float(best_res_12["funding_pnl"]),
            "basis_pnl": float(best_res_12["basis_pnl"]),
            "fees": float(best_res_12["fees"]),
            "positive_periods_count": go_no_go["positive_periods_count"],
            "total_periods": go_no_go["total_periods"],
            "cond1_return": go_no_go["cond1_return"],
            "cond2_consistency": go_no_go["cond2_consistency"],
            "cond3_risk": go_no_go["cond3_risk"],
            "cond4_production": go_no_go["cond4_production"],
            "verdict": go_no_go["verdict"],
            "failed_conditions": ";".join(go_no_go["failed"]),
        },
        {
            "config_id": f"dynamic8|{best_config_id}",
            "universe": "dynamic8",
            "n_symbols": 8,
            "annualized_return_pct": float(metrics_8["annualized_return_pct"]),
            "sharpe": float(metrics_8["sharpe"]),
            "maxdd_pct": float(metrics_8["maxdd_pct"]),
            "net_return_pct": float(metrics_8["net_return_pct"]),
            "dsr_prob": float(metrics_8["dsr_prob"]),
            "turnover": float(res_8["turnover"]),
            "funding_pnl": float(res_8["funding_pnl"]),
            "basis_pnl": float(res_8["basis_pnl"]),
            "fees": float(res_8["fees"]),
            "positive_periods_count": sum(1 for p in periods_df_8["positive"] if p),
            "total_periods": len(periods_df_8),
            "cond1_return": go_no_go["cond1_return"],
            "cond2_consistency": go_no_go["cond2_consistency"],
            "cond3_risk": go_no_go["cond3_risk"],
            "cond4_production": go_no_go["cond4_production"],
            "verdict": go_no_go["verdict"],
            "failed_conditions": ";".join(go_no_go["failed"]),
        },
    ]

    if res_static is not None and metrics_static is not None:
        summary_rows.append({
            "config_id": "static4|spot_perp_2leg|hysteresis",
            "universe": "static4",
            "n_symbols": 4,
            "annualized_return_pct": float(metrics_static["annualized_return_pct"]),
            "sharpe": float(metrics_static["sharpe"]),
            "maxdd_pct": float(metrics_static["maxdd_pct"]),
            "net_return_pct": float(metrics_static["net_return_pct"]),
            "dsr_prob": float(metrics_static["dsr_prob"]),
            "turnover": float(res_static["metrics"].get("Turnover", 0.0)),
            "funding_pnl": float(res_static["metrics"].get("Total Funding Collected", 0.0)),
            "basis_pnl": float(res_static["metrics"].get("Total MTM PnL", 0.0)),
            "fees": float(res_static["metrics"].get("Total Fees Paid", 0.0)),
            "positive_periods_count": (
                sum(1 for p in periods_df_static["positive"] if p)
                if periods_df_static is not None
                else 0
            ),
            "total_periods": len(periods_df_static) if periods_df_static is not None else 0,
            "cond1_return": "n/a",
            "cond2_consistency": "n/a",
            "cond3_risk": "n/a",
            "cond4_production": "n/a",
            "verdict": "BASELINE",
            "failed_conditions": "",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / "sprint8_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"-> Da xuat Summary ra: {summary_csv_path}")

    # Xuất Markdown nếu có yêu cầu
    if args.out_md:
        md_path = Path(args.out_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        md_content = []
        md_content.append("# Bao Cao Nghiem Thu Sprint 8 (Dynamic Carry & Basis Convergence)\n")
        md_content.append(f"**Khoang thoi gian**: `{args.start}` den `{args.end}` (3.45 nam)\n")

        md_content.append("## Bang 1: Tong hop toan chuoi 3.45 nam (3 cau hinh doi chieu)\n")
        cols_sum = [
            "universe",
            "annualized_return_pct",
            "sharpe",
            "maxdd_pct",
            "dsr_prob",
            "turnover",
            "net_return_pct",
            "funding_pnl",
            "basis_pnl",
            "fees",
            "verdict",
        ]
        md_content.append(to_markdown_table(summary_rows, columns=cols_sum))
        md_content.append("\n")

        md_content.append("## Bang 2: Chi tiet 7 giai doan 180 ngay cua DYNAMIC_12\n")
        cols_periods = [
            "period_id",
            "start",
            "end",
            "net_return_pct",
            "annualized_return_pct",
            "sharpe",
            "maxdd_pct",
            "turnover",
            "funding_pnl",
            "basis_pnl",
            "fees",
            "positive",
        ]
        md_content.append(
            to_markdown_table(periods_df_12.to_dict(orient="records"), columns=cols_periods)
        )
        md_content.append("\n")

        md_content.append("## Bang 3: Phan ra PnL Attribution cua DYNAMIC_12\n")
        pnl_recs = [r for r in attr_rows if r["universe"] == "dynamic12"]
        cols_attr = [
            "period",
            "funding_pnl",
            "basis_pnl",
            "fees",
            "net_pnl",
            "funding_return_pct",
            "basis_return_pct",
            "fee_drag_pct",
            "net_return_pct",
        ]
        md_content.append(to_markdown_table(pnl_recs, columns=cols_attr))
        md_content.append("\n")

        md_content.append("## Bang 4: Phan quyet 4 dieu kien GO / NO-GO\n")
        verdict_rows = [
            {
                "Condition": "1. Annualized Return (>=8% ro 12, >=6% ro 8)",
                "Result": f"Ro 12: {go_no_go['details']['annualized_return_12']:.2f}% | Ro 8: {go_no_go['details']['annualized_return_8']:.2f}%",
                "Status": "PASS" if go_no_go["cond1_return"] else "FAIL",
            },
            {
                "Condition": "2. Consistency (>= 6/7 ky net return > 0)",
                "Result": f"{go_no_go['positive_periods_count']}/{go_no_go['total_periods']} ky duong",
                "Status": "PASS" if go_no_go["cond2_consistency"] else "FAIL",
            },
            {
                "Condition": "3. Risk (Sharpe >= 2.0, DSR >= 0.95, MaxDD <= 5.0%)",
                "Result": f"Sharpe: {go_no_go['details']['sharpe_12']:.2f} | DSR: {go_no_go['details']['dsr_prob_12']:.2f} | MaxDD: {go_no_go['details']['maxdd_pct_12']:.2f}%",
                "Status": "PASS" if go_no_go["cond3_risk"] else "FAIL",
            },
            {
                "Condition": "4. Production Readiness (CarryLiveExecutor ready)",
                "Result": f"Module ton tai: {go_no_go['details']['live_executor_ready']}",
                "Status": "PASS" if go_no_go["cond4_production"] else "FAIL",
            },
            {
                "Condition": "TONG KET PHAN QUYET",
                "Result": "Thoa ca 4 dieu kien" if go_no_go["verdict"] == "GO" else f"Truot: {'; '.join(go_no_go['failed'])}",
                "Status": go_no_go["verdict"],
            },
        ]
        md_content.append(to_markdown_table(verdict_rows))
        md_content.append("\n")

        md_path.write_text("\n".join(md_content), encoding="utf-8")
        print(f"-> Da xuat bao cao Markdown ra: {md_path}")

    # In báo cáo tổng kết ASCII-safe ra console
    print("\n" + "=" * 80)
    print("CHI TIET 7 GIAI DOAN (DYNAMIC_12 - BEST CONFIG)")
    print("=" * 80)
    print(
        f"{'Ky':<4} | {'Bat dau':<10} | {'Ket thuc':<10} | {'NetRet%':<7} | {'AnnRet%':<7} | "
        f"{'Sharpe':<6} | {'MaxDD%':<6} | {'Funding$':<8} | {'Basis$':<8} | {'Phí$':<7} | {'Duong':<5}"
    )
    print("-" * 80)
    for r in periods_df_12.to_dict(orient="records"):
        s_str = pd.Timestamp(r["start"]).strftime("%Y-%m-%d")
        e_str = pd.Timestamp(r["end"]).strftime("%Y-%m-%d")
        pos_str = "YES" if r["positive"] else "NO"
        print(
            f"{r['period_id']:<4} | {s_str:<10} | {e_str:<10} | {r['net_return_pct']:7.2f} | "
            f"{r['annualized_return_pct']:7.2f} | {r['sharpe']:6.2f} | {r['maxdd_pct']:6.2f} | "
            f"{r['funding_pnl']:8.2f} | {r['basis_pnl']:8.2f} | {r['fees']:7.2f} | {pos_str:<5}"
        )
    print("=" * 80)

    print("\n" + "=" * 80)
    print("PHAN QUYET GO / NO-GO SPRINT 8 (KHOA CUNG THEO DOCS)")
    print("=" * 80)
    c1_st = "[PASS]" if go_no_go["cond1_return"] else "[FAIL]"
    c2_st = "[PASS]" if go_no_go["cond2_consistency"] else "[FAIL]"
    c3_st = "[PASS]" if go_no_go["cond3_risk"] else "[FAIL]"
    c4_st = "[PASS]" if go_no_go["cond4_production"] else "[FAIL]"

    print(
        f"Dieu kien 1 (AnnRet >= 8% ro 12 & >= 6% ro 8)   : {c1_st} "
        f"(Ro 12: {go_no_go['details']['annualized_return_12']:.2f}%, Ro 8: {go_no_go['details']['annualized_return_8']:.2f}%)"
    )
    print(
        f"Dieu kien 2 (Consistency >= 6/7 ky duong)       : {c2_st} "
        f"({go_no_go['positive_periods_count']}/{go_no_go['total_periods']} ky duong)"
    )
    print(
        f"Dieu kien 3 (Sharpe >= 2.0, DSR >= 0.95, DD <=5%): {c3_st} "
        f"(Sharpe: {go_no_go['details']['sharpe_12']:.2f}, DSR: {go_no_go['details']['dsr_prob_12']:.2f}, MaxDD: {go_no_go['details']['maxdd_pct_12']:.2f}%)"
    )
    print(
        f"Dieu kien 4 (Production Readiness)              : {c4_st} "
        f"(CarryLiveExecutor san sang: {go_no_go['details']['live_executor_ready']})"
    )
    print("-" * 80)
    print(f"KET QUA CUOI CUNG: {go_no_go['verdict']}")
    if go_no_go["failed"]:
        print(f"Cac dieu kien chua dat: {', '.join(go_no_go['failed'])}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
