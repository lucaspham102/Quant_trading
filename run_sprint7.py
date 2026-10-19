"""CLI Runner đánh giá Sprint 7 (Cross-Sectional 12 Symbols & True MTM Carry).

Quy trình:
1. (Tùy chọn) Tải/cập nhật dữ liệu thật: Perp OHLCV 1h + Funding 8h cho 12 coin,
   Spot OHLCV 1h cho 4 coin core.
2. Dựng dữ liệu đồng bộ Spot-Perp-Funding (basis_pct) và dữ liệu Perp-Funding.
3. Chạy lưới Carry (4 cấu hình MTM + 4 cấu hình lý tưởng hóa Sprint 6).
4. Chạy lưới Cross-Sectional (rổ 8 coin mới top_k=2, rổ 12 coin top_k=3;
   lookback {336, 504}h x buffer {0, 2} x carry tilt {0.0, 0.25}).
5. Tính chỉ số DEV, và ở chặng holdout: chạy liền mạch trên đường cong vốn nối tiếp
   rồi cắt segment HOLDOUT (chỉ chạy 1 lần cho các cấu hình đã chọn từ DEV).

Ví dụ:
    python run_sprint7.py --download --stage dev
    python run_sprint7.py --stage holdout --select-csv data/sprint7_xs_dev.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.carry_strategy import CarryConfig, run_carry_portfolio
from src.config import (
    CORE_4_SYMBOLS,
    EXTRA_8_SYMBOLS,
    UNIVERSE_12_SYMBOLS,
    TradingConfig,
)
from src.cross_sectional import CrossSectionalConfig, run_cross_sectional_portfolio
from src.data_loader import DataLoader
from src.funding_loader import (
    align_ohlcv_with_funding,
    align_spot_perp_funding,
    fetch_funding_history,
    fetch_spot_ohlcv,
)
from src.sprint7_eval import (
    build_benchmark_returns,
    evaluate_carry_go_no_go,
    evaluate_cross_sectional_go_no_go,
    segment_metrics,
    summarize_verdicts,
    to_markdown_table,
)

DEFAULT_START = "2023-05-07"
DEFAULT_DEV_END = "2026-04-21"
DEFAULT_END = "2026-10-18"
PERIODS_PER_YEAR = 8760
K_CARRY = 4
K_XS = 16


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Phân tích tham số dòng lệnh cho runner Sprint 7."""
    parser = argparse.ArgumentParser(description="Sprint 7: 12-asset XS & True MTM Carry")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--dev-end", default=DEFAULT_DEV_END, help="Mốc bắt đầu Holdout")
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--stage",
        default="dev",
        choices=("dev", "holdout", "all"),
        help="dev = chỉ tập DEV; holdout/all = chạy liền mạch và cắt segment Holdout",
    )
    parser.add_argument(
        "--select-csv",
        default=None,
        help="CSV kết quả DEV dùng để chọn cấu hình trước khi mở khóa Holdout",
    )
    parser.add_argument("--download", action="store_true", help="Tải/cập nhật dữ liệu thật trước khi chạy")
    parser.add_argument("--skip-carry", action="store_true")
    parser.add_argument("--skip-xs", action="store_true")
    parser.add_argument("--out-md", default=None, help="File Markdown xuất bảng tổng hợp")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------


def _clean_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _spot_name(spot_symbol: str) -> str:
    return spot_symbol.replace("/", "").replace(":", "").upper()


def _load_perp(symbol: str, timeframe: str, raw_dir: Path) -> pd.DataFrame:
    """Đọc cache Perp OHLCV; lỗi nếu chưa có (dùng --download để tải)."""
    path = raw_dir / f"{_clean_name(symbol)}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Thiếu cache Perp: {path}. Hãy chạy với --download.")
    cfg = TradingConfig(symbol=symbol, timeframe=timeframe, raw_data_dir=raw_dir)
    return DataLoader(config=cfg).load_parquet(path)


def _load_funding(symbol: str, raw_dir: Path) -> pd.DataFrame:
    """Đọc cache Funding Rate; lỗi nếu chưa có."""
    path = raw_dir / f"{_clean_name(symbol)}_funding.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Thiếu cache Funding: {path}. Hãy chạy với --download.")
    return pd.read_parquet(path)


def _load_spot(spot_symbol: str, timeframe: str, raw_dir: Path) -> pd.DataFrame:
    """Đọc cache Spot OHLCV; lỗi nếu chưa có."""
    path = raw_dir / f"{_spot_name(spot_symbol)}_SPOT_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Thiếu cache Spot: {path}. Hãy chạy với --download.")
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx.as_unit("ns")
    return df


def download_data(timeframe: str, raw_dir: Path, start: str, end: str) -> None:
    """Tải/cập nhật Perp + Funding 12 coin và Spot OHLCV 4 coin core."""
    import ccxt

    perp_ex = ccxt.binanceusdm({"enableRateLimit": True})
    spot_ex = ccxt.binance({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)

    for symbol in UNIVERSE_12_SYMBOLS:
        cfg = TradingConfig(symbol=symbol, timeframe=timeframe, raw_data_dir=raw_dir)
        loader = DataLoader(config=cfg, exchange=perp_ex)
        df = loader.update_incremental(default_start_ms=since_ms)
        print(f"[download] perp {symbol}: {len(df)} nến", flush=True)
        fdf = fetch_funding_history(
            symbol, start_date=start, end_date=end, data_dir=str(raw_dir), exchange=perp_ex
        )
        print(f"[download] funding {symbol}: {len(fdf)} kỳ", flush=True)

    for symbol in CORE_4_SYMBOLS:
        spot_symbol = symbol.split(":")[0]
        sdf = fetch_spot_ohlcv(
            spot_symbol,
            timeframe=timeframe,
            start_date=start,
            end_date=end,
            data_dir=str(raw_dir),
            exchange=spot_ex,
        )
        print(f"[download] spot {spot_symbol}: {len(sdf)} nến", flush=True)


def _trim(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Cắt DataFrame theo khoảng thời gian [start, end) UTC."""
    out = df.copy()
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    return out[(out.index >= start_ts) & (out.index < end_ts)]


def build_core_frames(
    timeframe: str, raw_dir: Path, start: str, end: str
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Dựng dữ liệu core: (true MTM có basis_pct, legacy không có Spot).

    Trả về:
        (mtm_frames, legacy_frames): dict {symbol: DataFrame}.
    """
    mtm_frames: dict[str, pd.DataFrame] = {}
    legacy_frames: dict[str, pd.DataFrame] = {}
    for symbol in CORE_4_SYMBOLS:
        perp = _load_perp(symbol, timeframe, raw_dir)
        funding = _load_funding(symbol, raw_dir)
        spot = _load_spot(symbol.split(":")[0], timeframe, raw_dir)

        aligned = align_spot_perp_funding(perp, spot, funding)
        mtm_frames[symbol] = _trim(aligned, start, end)

        legacy = align_ohlcv_with_funding(perp, funding)
        legacy = legacy[["open", "high", "low", "close", "volume", "funding_rate_event", "last_known_funding_rate"]]
        legacy_frames[symbol] = _trim(legacy, start, end)
    return mtm_frames, legacy_frames


def build_universe_frames(
    symbols: Sequence[str], timeframe: str, raw_dir: Path, start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Dựng dữ liệu Perp + Funding cho một rổ symbol."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        perp = _load_perp(symbol, timeframe, raw_dir)
        funding = _load_funding(symbol, raw_dir)
        frames[symbol] = _trim(align_ohlcv_with_funding(perp, funding), start, end)
    return frames


# ---------------------------------------------------------------------------
# Lưới cấu hình
# ---------------------------------------------------------------------------


def iter_carry_configs(data_model: str) -> Iterable[tuple[str, CarryConfig, dict]]:
    """Sinh 4 cấu hình Carry cho một kiểu dữ liệu (true_mtm / idealized)."""
    for structure in ("coinm_1leg", "spot_perp_2leg"):
        for mode in ("hysteresis", "passive_long_term"):
            cfg = CarryConfig(
                structure=structure,
                carry_mode=mode,
                perp_fee=0.0005,
                spot_fee=0.00075,
                slippage=0.0002,
                initial_capital=400.0,
                basis_noise_std=0.0,
                min_hold_periods=3,
                passive_min_hold_periods=21,
            )
            params = {
                "structure": structure,
                "carry_mode": mode,
                "data_model": data_model,
            }
            config_id = f"{data_model}|{structure}|{mode}"
            yield config_id, cfg, params


def iter_xs_configs() -> Iterable[tuple[str, CrossSectionalConfig, dict, Sequence[str]]]:
    """Sinh 16 cấu hình Cross-Sectional (8 cho rổ xs8, 8 cho rổ xs12)."""
    universes = (
        ("xs8", EXTRA_8_SYMBOLS, 2),
        ("xs12", UNIVERSE_12_SYMBOLS, 3),
    )
    for label, symbols, top_k in universes:
        for lookback in (336, 504):
            for buffer_rank in (0, 2):
                for tilt in (0.0, 0.25):
                    cfg = CrossSectionalConfig(
                        mode="momentum",
                        lookback_hours=lookback,
                        rebalance_hours=48,
                        min_spread=0.0,
                        top_k=top_k,
                        buffer_rank=buffer_rank,
                        carry_tilt_weight=tilt,
                        gross_leverage=1.0,
                        perp_fee=0.0005,
                        slippage=0.0002,
                        initial_capital=400.0,
                    )
                    params = {
                        "universe": label,
                        "n_symbols": len(symbols),
                        "top_k": top_k,
                        "lookback_hours": lookback,
                        "rebalance_hours": 48,
                        "buffer_rank": buffer_rank,
                        "carry_tilt_weight": tilt,
                    }
                    config_id = (
                        f"{label}|lb{lookback}|buf{buffer_rank}|tilt{tilt:g}"
                    )
                    yield config_id, cfg, params, symbols


# ---------------------------------------------------------------------------
# Chạy & chấm điểm
# ---------------------------------------------------------------------------


def _benchmark_from(
    frames: dict[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    fallback_close: pd.Series | None = None,
) -> pd.Series:
    """Lợi nhuận Buy-and-Hold BTC căn theo index chiến lược.

    Nếu rổ không chứa BTC (rổ 8 coin mới), dùng `fallback_close` (BTC Perp).
    """
    btc_symbol = CORE_4_SYMBOLS[0]
    if btc_symbol in frames:
        btc = frames[btc_symbol]["close"].reindex(index).ffill()
    elif fallback_close is not None:
        btc = fallback_close.reindex(index).ffill()
    else:
        raise KeyError(f"Rổ dữ liệu không có {btc_symbol} và không có fallback_close")
    return build_benchmark_returns(btc)


def _split_segments(
    returns: pd.Series, dev_end: str
) -> tuple[pd.Series, pd.Series]:
    """Cắt chuỗi lợi nhuận nối tiếp thành (dev, holdout) theo mốc thời gian."""
    cutoff = pd.Timestamp(dev_end, tz="UTC")
    dev = returns[returns.index < cutoff]
    holdout = returns[returns.index >= cutoff]
    return dev, holdout


def _score(
    returns: pd.Series,
    benchmark: pd.Series,
    num_trials: int,
    dev_end: str | None,
) -> dict:
    """Tính chỉ số DEV (và HOLDOUT nếu dev_end được cung cấp)."""
    if dev_end is None:
        return {"dev": segment_metrics(returns, benchmark, PERIODS_PER_YEAR, num_trials)}
    dev_ret, ho_ret = _split_segments(returns, dev_end)
    return {
        "dev": segment_metrics(dev_ret, benchmark, PERIODS_PER_YEAR, num_trials),
        "holdout": segment_metrics(ho_ret, benchmark, PERIODS_PER_YEAR, num_trials),
    }


def run_carry_grid(
    mtm_frames: dict[str, pd.DataFrame],
    legacy_frames: dict[str, pd.DataFrame],
    dev_end: str | None,
    selected: set[str] | None,
) -> list[dict]:
    """Chạy lưới Carry và trả về danh sách dòng kết quả."""
    rows: list[dict] = []
    benchmarks = {
        "true_mtm": _benchmark_from(mtm_frames, next(iter(mtm_frames.values())).index),
        "idealized": _benchmark_from(legacy_frames, next(iter(legacy_frames.values())).index),
    }
    for data_model, frames in (("true_mtm", mtm_frames), ("idealized", legacy_frames)):
        for config_id, cfg, params in iter_carry_configs(data_model):
            if selected is not None and config_id not in selected:
                continue
            result = run_carry_portfolio(frames, cfg, periods_per_year=PERIODS_PER_YEAR)
            returns = result["returns"]
            bench = benchmarks[data_model].reindex(returns.index).ffill().fillna(0.0)
            scored = _score(returns, bench, K_CARRY, dev_end)

            row: dict[str, Any] = {
                "config_id": config_id,
                **params,
                "fees_paid": float(result["metrics"].get("Total Fees Paid", 0.0)),
                "funding_collected": float(result["metrics"].get("Total Funding Collected", 0.0)),
                "mtm_pnl": float(result["metrics"].get("Total MTM PnL", 0.0)),
                "num_rebalances": int(result["metrics"].get("Num Rebalances", 0)),
                "time_in_market_pct": float(result["metrics"].get("Time in Market %", 0.0)),
            }
            for seg in ("dev", "holdout"):
                if seg not in scored:
                    continue
                m = scored[seg]
                row.update({
                    f"{seg}_net_return_pct": m["net_return_pct"],
                    f"{seg}_ann_return_pct": m["annualized_return_pct"],
                    f"{seg}_sharpe": m["sharpe"],
                    f"{seg}_maxdd_pct": m["maxdd_pct"],
                    f"{seg}_dsr_prob": m["dsr_prob"],
                    f"{seg}_alpha_t": m["alpha_t"],
                    f"{seg}_beta": m["beta"],
                })
            if "holdout" in scored:
                verdict = evaluate_carry_go_no_go(scored["dev"], scored["holdout"])
                row.update({
                    "verdict": verdict["verdict"],
                    "cond1_maxdd": verdict["cond1_maxdd"],
                    "cond2_return_sharpe": verdict["cond2_return_sharpe"],
                    "failed": json.dumps(verdict["failed"]),
                })
            rows.append(row)
    return rows


def run_xs_grid(
    universe_frames: dict[str, dict[str, pd.DataFrame]],
    dev_end: str | None,
    selected: set[str] | None,
    btc_close: pd.Series,
) -> list[dict]:
    """Chạy lưới Cross-Sectional và trả về danh sách dòng kết quả."""
    rows: list[dict] = []
    for config_id, cfg, params, symbols in iter_xs_configs():
        if selected is not None and config_id not in selected:
            continue
        frames = universe_frames[params["universe"]]
        result = run_cross_sectional_portfolio(frames, cfg, periods_per_year=PERIODS_PER_YEAR)
        returns = result["returns"]
        bench = _benchmark_from(frames, returns.index, fallback_close=btc_close)
        scored = _score(returns, bench, K_XS, dev_end)

        row: dict[str, Any] = {
            "config_id": config_id,
            **params,
            "turnover": float(result["metrics"].get("Turnover", 0.0)),
            "num_rebalances": int(result["metrics"].get("Num Rebalances", 0)),
        }
        for seg in ("dev", "holdout"):
            if seg not in scored:
                continue
            m = scored[seg]
            row.update({
                f"{seg}_net_return_pct": m["net_return_pct"],
                f"{seg}_ann_return_pct": m["annualized_return_pct"],
                f"{seg}_sharpe": m["sharpe"],
                f"{seg}_maxdd_pct": m["maxdd_pct"],
                f"{seg}_dsr_prob": m["dsr_prob"],
                f"{seg}_alpha_t": m["alpha_t"],
                f"{seg}_beta": m["beta"],
            })
        if "holdout" in scored:
            verdict = evaluate_cross_sectional_go_no_go(
                scored["dev"], scored["holdout"], params["universe"], num_trials=K_XS
            )
            row.update({
                "verdict": verdict["verdict"],
                "cond1_alpha_beta": verdict["cond1_alpha_beta"],
                "cond2_sharpe_dsr": verdict["cond2_sharpe_dsr"],
                "cond3_holdout": verdict["cond3_holdout"],
                "failed": json.dumps(verdict["failed"]),
            })
        rows.append(row)

    # Bổ sung % giảm turnover của buffer_rank=2 so với buffer_rank=0 cùng cấu hình khác.
    lookup = {
        (r["universe"], r["lookback_hours"], r["carry_tilt_weight"], r["buffer_rank"]): r
        for r in rows
    }
    for r in rows:
        base = lookup.get(
            (r["universe"], r["lookback_hours"], r["carry_tilt_weight"], 0)
        )
        if r["buffer_rank"] == 2 and base is not None and base["turnover"] > 0:
            r["turnover_reduction_pct"] = (1.0 - r["turnover"] / base["turnover"]) * 100.0
    return rows


# ---------------------------------------------------------------------------
# Chọn cấu hình từ DEV
# ---------------------------------------------------------------------------


def select_best_from_csv(path: Path, kind: str) -> set[str]:
    """Chọn config tốt nhất trên DEV từ CSV (theo Sharpe, ưu tiên DSR).

    Với Carry: chọn 1 cấu hình MTM tốt nhất theo DEV Sharpe.
    Với XS: chọn 1 cấu hình tốt nhất trên rổ 12 coin, sau đó áp dụng ĐÚNG bộ tham số
    (lookback / buffer / tilt) đó cho rổ 8 coin mới để bảo toàn ý nghĩa kiểm định
    Out-of-Universe (cùng chiến lược, khác rổ tài sản).
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV rỗng: {path}")

    selected: set[str] = set()
    if kind == "carry":
        pool = df[df["data_model"] == "true_mtm"].copy()
        if pool.empty:
            pool = df
        pool = pool.sort_values(["dev_sharpe", "dev_dsr_prob"], ascending=False)
        selected.add(str(pool.iloc[0]["config_id"]))
    else:
        pool = df[df["universe"] == "xs12"].copy()
        if pool.empty:
            pool = df.copy()
        ordered = pool.sort_values(["dev_sharpe", "dev_dsr_prob"], ascending=False)
        best = ordered.iloc[0]
        config_id = str(best["config_id"])
        match = re.match(r"^(xs8|xs12)\|lb(\d+)\|buf(\d+)\|tilt([0-9.]+)$", config_id)
        if match is not None:
            lookback = int(match.group(2))
            buffer_rank = int(match.group(3))
            tilt = float(match.group(4))
        else:
            lookback = int(best["lookback_hours"])
            buffer_rank = int(best["buffer_rank"])
            tilt = float(best["carry_tilt_weight"])
        selected.add(f"xs12|lb{lookback}|buf{buffer_rank}|tilt{tilt:g}")
        selected.add(f"xs8|lb{lookback}|buf{buffer_rank}|tilt{tilt:g}")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_data(args.timeframe, raw_dir, args.start, args.end)

    if args.stage != "dev" and not args.select_csv:
        raise SystemExit(
            "Từ chối mở Holdout: bắt buộc truyền --select-csv để khóa cấu hình đã chọn "
            "trên DEV (chỉ được mở Holdout đúng 1 lần)."
        )

    dev_end = None if args.stage == "dev" else args.dev_end
    data_end = args.dev_end if args.stage == "dev" else args.end

    selected_carry: set[str] | None = None
    selected_xs: set[str] | None = None
    if args.stage != "dev":
        if args.select_csv:
            base = Path(args.select_csv)
            if "_carry_" in base.name:
                carry_csv: Path | None = base
                xs_csv: Path | None = Path(str(base).replace("_carry_", "_xs_"))
            elif "_xs_" in base.name:
                xs_csv = base
                carry_csv = Path(str(base).replace("_xs_", "_carry_"))
            else:
                carry_csv = base
                xs_csv = base
            if carry_csv is not None and carry_csv.exists() and not args.skip_carry:
                selected_carry = select_best_from_csv(carry_csv, "carry")
            if xs_csv is not None and xs_csv.exists() and not args.skip_xs:
                selected_xs = select_best_from_csv(xs_csv, "xs")

    if args.stage != "dev":
        if not args.skip_carry and not selected_carry:
            raise SystemExit("Không xác định được cấu hình Carry từ --select-csv; hủy mở Holdout.")
        if not args.skip_xs and not selected_xs:
            raise SystemExit("Không xác định được cấu hình Cross-Sectional từ --select-csv; hủy mở Holdout.")

    tables: list[str] = []
    summary: dict[str, Any] = {
        "stage": args.stage,
        "start": args.start,
        "dev_end": args.dev_end,
        "end": args.end,
        "data_end_used": data_end,
    }

    if not args.skip_carry:
        mtm_frames, legacy_frames = build_core_frames(
            args.timeframe, raw_dir, args.start, data_end
        )
        carry_rows = run_carry_grid(mtm_frames, legacy_frames, dev_end, selected_carry)
        carry_path = out_dir / f"sprint7_carry_{args.stage}.csv"
        pd.DataFrame(carry_rows).to_csv(carry_path, index=False, encoding="utf-8")
        summary["carry_csv"] = str(carry_path)
        summary["carry_verdicts"] = summarize_verdicts(carry_rows)
        cols = [
            "config_id", "dev_net_return_pct", "dev_ann_return_pct", "dev_sharpe",
            "dev_maxdd_pct", "dev_dsr_prob", "dev_alpha_t", "dev_beta",
            "holdout_net_return_pct", "holdout_ann_return_pct", "holdout_sharpe",
            "holdout_maxdd_pct", "num_rebalances", "verdict",
        ]
        tables.append("### True MTM Carry\n\n" + to_markdown_table(carry_rows, [c for c in cols if c in pd.DataFrame(carry_rows).columns]))
        print(f"[carry] {len(carry_rows)} cấu hình -> {carry_path}", flush=True)

    if not args.skip_xs:
        universe_frames: dict[str, dict[str, pd.DataFrame]] = {}
        for label, symbols, _top_k in (
            ("xs8", EXTRA_8_SYMBOLS, 2),
            ("xs12", UNIVERSE_12_SYMBOLS, 3),
        ):
            universe_frames[label] = build_universe_frames(
                symbols, args.timeframe, raw_dir, args.start, data_end
            )
        xs_rows = run_xs_grid(
            universe_frames,
            dev_end,
            selected_xs,
            btc_close=_trim(_load_perp(CORE_4_SYMBOLS[0], args.timeframe, raw_dir), args.start, data_end)["close"],
        )
        xs_path = out_dir / f"sprint7_xs_{args.stage}.csv"
        pd.DataFrame(xs_rows).to_csv(xs_path, index=False, encoding="utf-8")
        summary["xs_csv"] = str(xs_path)
        summary["xs_verdicts"] = summarize_verdicts(xs_rows)
        cols = [
            "config_id", "universe", "top_k", "lookback_hours", "buffer_rank",
            "carry_tilt_weight", "dev_net_return_pct", "dev_sharpe", "dev_alpha_t",
            "dev_beta", "dev_dsr_prob", "turnover", "turnover_reduction_pct",
            "holdout_net_return_pct", "holdout_sharpe", "holdout_maxdd_pct", "verdict",
        ]
        tables.append("### Cross-Sectional\n\n" + to_markdown_table(xs_rows, [c for c in cols if c in pd.DataFrame(xs_rows).columns]))
        print(f"[xs] {len(xs_rows)} cấu hình -> {xs_path}", flush=True)

    md = "\n\n".join(tables)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md + "\n", encoding="utf-8")

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
