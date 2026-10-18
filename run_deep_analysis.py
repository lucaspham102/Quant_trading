"""CLI Runner cho Deep Analysis Engine: Chẩn đoán chuyên sâu Baseline 1h vs Buy-and-Hold.

Thực thi phân tích A1-A5 cho danh sách symbols, gom nối kết quả và xuất 5 file CSV
kèm báo cáo định dạng Markdown tổng hợp.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

from src.deep_analysis import run_deep_analysis
from src.research import DEFAULT_SYMBOLS, to_markdown


def _split_list(raw: str | Sequence[str]) -> list[str]:
    """Tách danh sách symbol phân cách bởi dấu phẩy hoặc khoảng trắng."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Thiết lập và phân tích tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Deep Analysis Engine: Chẩn đoán baseline 1h vs buy-and-hold (Sprint 5 - Phần A)"
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Danh sách symbol phân cách bằng dấu phẩy (mặc định DEFAULT_SYMBOLS)",
    )
    parser.add_argument("--timeframe", default="1h", help="Khung thời gian phân tích (mặc định 1h)")
    parser.add_argument("--days", type=int, default=1260, help="Số ngày lịch sử (mặc định 1260)")
    parser.add_argument("--train-days", type=int, default=120, help="Số ngày tập train (mặc định 120)")
    parser.add_argument("--test-days", type=int, default=60, help="Số ngày tập test (mặc định 60)")
    parser.add_argument("--step-days", type=int, default=60, help="Số ngày bước trượt (mặc định 60)")
    parser.add_argument("--buffer-days", type=int, default=15, help="Số ngày đệm warmup (mặc định 15)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Số lần lặp bootstrap (mặc định 1000)")
    parser.add_argument("--n-null", type=int, default=1000, help="Số lần lặp null test (mặc định 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên (mặc định 42)")
    parser.add_argument("--out-dir", default="data", help="Thư mục xuất CSV (mặc định data)")
    parser.add_argument(
        "--out-md",
        default="docs/sprint5_results_partA.md",
        help="Đường dẫn file Markdown tổng hợp (mặc định docs/sprint5_results_partA.md)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Điểm thực thi chính cho CLI Deep Analysis."""
    args = parse_args(argv)
    symbols = _split_list(args.symbols)

    all_exposure: list[pd.DataFrame] = []
    all_drawdown: list[pd.DataFrame] = []
    all_alpha_beta: list[pd.DataFrame] = []
    all_bootstrap: list[pd.DataFrame] = []
    all_null: list[pd.DataFrame] = []

    for symbol in symbols:
        res = run_deep_analysis(
            symbol=symbol,
            timeframe=args.timeframe,
            days=args.days,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            buffer_days=args.buffer_days,
            n_boot=args.n_boot,
            n_null=args.n_null,
            seed=args.seed,
        )
        all_exposure.append(res["exposure"])
        all_drawdown.append(res["drawdown"])
        all_alpha_beta.append(res["alpha_beta"])
        all_bootstrap.append(res["bootstrap"])
        all_null.append(res["null"])

    exposure_df = pd.concat(all_exposure, ignore_index=True) if all_exposure else pd.DataFrame()
    drawdown_df = pd.concat(all_drawdown, ignore_index=True) if all_drawdown else pd.DataFrame()
    alpha_beta_df = pd.concat(all_alpha_beta, ignore_index=True) if all_alpha_beta else pd.DataFrame()
    bootstrap_df = pd.concat(all_bootstrap, ignore_index=True) if all_bootstrap else pd.DataFrame()
    null_df = pd.concat(all_null, ignore_index=True) if all_null else pd.DataFrame()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exposure_df.to_csv(out_dir / "sprint5_exposure.csv", index=False)
    drawdown_df.to_csv(out_dir / "sprint5_drawdown.csv", index=False)
    alpha_beta_df.to_csv(out_dir / "sprint5_alpha_beta.csv", index=False)
    bootstrap_df.to_csv(out_dir / "sprint5_bootstrap.csv", index=False)
    null_df.to_csv(out_dir / "sprint5_null.csv", index=False)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md_content = "\n\n".join(
        [
            "# Sprint 5 — Phần A",
            "## 1. Exposure Metrics",
            to_markdown(exposure_df),
            "## 2. Drawdown Analysis",
            to_markdown(drawdown_df),
            "## 3. Alpha / Beta (Newey-West HAC)",
            to_markdown(alpha_beta_df),
            "## 4. Block Bootstrap Sharpe Difference",
            to_markdown(bootstrap_df),
            "## 5. Exposure-Matched Null Test",
            to_markdown(null_df),
            "",
        ]
    )
    out_md.write_text(md_content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
