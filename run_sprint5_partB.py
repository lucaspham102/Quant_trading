"""CLI Runner cho Sprint 5 Phần B: Đánh giá khung thời gian 4h & 1d resample từ 1h.

Thực hiện toàn bộ quy trình kiểm định Sprint 5 - Phần B:
1. Resample dữ liệu 1h sang 4h và 1d cho từng symbol.
2. Tách biệt tập Dev và Holdout (180 ngày cuối cùng).
3. Chạy phân tích sâu Walk-Forward trên tập Dev (alpha/beta, bootstrap, null, exposure, sharpe vs B&H).
4. Đánh giá kiểm thử Holdout 1 lần duy nhất trên 180 ngày cuối.
5. Xuất 6 file CSV kết quả và tài liệu tổng hợp định dạng Markdown.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

from src.deep_analysis import (
    analyze_timeframe,
    holdout_return,
    resample_ohlcv,
)
from src.research import DEFAULT_SYMBOLS, load_or_fetch_symbol_data, to_markdown


def _split_list(raw: str | Sequence[str]) -> list[str]:
    """Tách danh sách symbol phân cách bởi dấu phẩy hoặc khoảng trắng."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Thiết lập và phân tích tham số dòng lệnh CLI cho Phần B."""
    parser = argparse.ArgumentParser(
        description="Sprint 5 — Phần B: Thử nghiệm khung thời gian 4h & 1d kèm kiểm định Holdout"
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Danh sách symbol phân cách bằng dấu phẩy (mặc định DEFAULT_SYMBOLS)",
    )
    parser.add_argument("--days", type=int, default=1260, help="Số ngày lịch sử tải (mặc định 1260)")
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=180,
        help="Số ngày cuối dành riêng cho tập Holdout (mặc định 180)",
    )
    parser.add_argument("--n-boot", type=int, default=1000, help="Số lần lặp bootstrap (mặc định 1000)")
    parser.add_argument("--n-null", type=int, default=1000, help="Số lần lặp null test (mặc định 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (mặc định 42)")
    parser.add_argument("--out-dir", default="data", help="Thư mục xuất file CSV (mặc định data)")
    parser.add_argument(
        "--out-md",
        default="docs/sprint5_results_partB.md",
        help="Đường dẫn file Markdown tổng hợp (mặc định docs/sprint5_results_partB.md)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Điểm thực thi chính cho CLI Sprint 5 Phần B."""
    args = parse_args(argv)
    symbols = _split_list(args.symbols)

    tf_cfg: dict[str, tuple[int, int, int]] = {
        "4h": (120, 60, 60),
        "1d": (360, 120, 120),
    }

    all_alpha_beta: list[pd.DataFrame] = []
    all_bootstrap: list[pd.DataFrame] = []
    all_null: list[pd.DataFrame] = []
    all_exposure: list[pd.DataFrame] = []
    all_sharpe: list[pd.DataFrame] = []
    all_drawdown: list[pd.DataFrame] = []
    all_holdout: list[dict] = []

    for symbol in symbols:
        df1h = load_or_fetch_symbol_data(symbol, "1h", days=args.days)
        if len(df1h) == 0:
            continue

        cutoff = df1h.index[-1] - pd.Timedelta(days=args.holdout_days)
        dev = df1h.loc[df1h.index < cutoff]
        hold = df1h.loc[df1h.index >= cutoff]

        for tf in ["4h", "1d"]:
            dev_tf = resample_ohlcv(dev, tf)
            tr, te, st = tf_cfg[tf]
            res = analyze_timeframe(
                symbol,
                tf,
                dev_tf,
                train_days=tr,
                test_days=te,
                step_days=st,
                n_boot=args.n_boot,
                n_null=args.n_null,
                seed=args.seed,
            )
            all_alpha_beta.append(res["alpha_beta"])
            all_bootstrap.append(res["bootstrap"])
            all_null.append(res["null"])
            all_exposure.append(res["exposure"])
            all_sharpe.append(res["sharpe"])
            all_drawdown.append(res["drawdown"].assign(timeframe=tf))

            hold_tf = resample_ohlcv(df1h, tf)
            ho = holdout_return(symbol, tf, hold_tf, cutoff)
            all_holdout.append(ho)

    alpha_beta_df = pd.concat(all_alpha_beta, ignore_index=True) if all_alpha_beta else pd.DataFrame()
    bootstrap_df = pd.concat(all_bootstrap, ignore_index=True) if all_bootstrap else pd.DataFrame()
    null_df = pd.concat(all_null, ignore_index=True) if all_null else pd.DataFrame()
    exposure_df = pd.concat(all_exposure, ignore_index=True) if all_exposure else pd.DataFrame()
    sharpe_df = pd.concat(all_sharpe, ignore_index=True) if all_sharpe else pd.DataFrame()
    drawdown_df = pd.concat(all_drawdown, ignore_index=True) if all_drawdown else pd.DataFrame()
    holdout_df = pd.DataFrame(all_holdout) if all_holdout else pd.DataFrame()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_beta_df.to_csv(out_dir / "sprint5_partB_alpha_beta.csv", index=False)
    bootstrap_df.to_csv(out_dir / "sprint5_partB_bootstrap.csv", index=False)
    null_df.to_csv(out_dir / "sprint5_partB_null.csv", index=False)
    exposure_df.to_csv(out_dir / "sprint5_partB_exposure.csv", index=False)
    sharpe_df.to_csv(out_dir / "sprint5_partB_sharpe.csv", index=False)
    drawdown_df.to_csv(out_dir / "sprint5_partB_drawdown.csv", index=False)
    holdout_df.to_csv(out_dir / "sprint5_partB_holdout.csv", index=False)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        "# Sprint 5 — Phần B (4h & 1d)",
        "Số lần thử Phần B = 2 timeframe × 8 config × 4 symbol = 64.",
        "## 1. Alpha / Beta (Newey-West HAC)",
        to_markdown(alpha_beta_df),
        "## 2. Sharpe vs Buy-and-Hold",
        to_markdown(sharpe_df),
        "## 3. Exposure-Matched Null Test",
        to_markdown(null_df),
        "## 4. Exposure Metrics",
        to_markdown(exposure_df),
        "## 5. Holdout Return (180 ngày cuối)",
        to_markdown(holdout_df),
        "## 6. Block Bootstrap Sharpe Difference",
        to_markdown(bootstrap_df),
        "## 7. Drawdown (baseline vs B&H)",
        to_markdown(drawdown_df),
        "",
    ]
    out_md.write_text("\n\n".join(sections), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
