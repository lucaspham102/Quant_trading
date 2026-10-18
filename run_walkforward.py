"""CLI Walk-Forward Engine: kiểm định out-of-sample việc lựa chọn chiến lược.

Ví dụ:
    python run_walkforward.py --symbols BTC/USDT:USDT,ETH/USDT:USDT --timeframe 1h --days 365
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

import pandas as pd

from src.config import TradingConfig
from src.research import DEFAULT_SYMBOLS, to_markdown
from src.walkforward import apply_fee_scenario, run_walkforward


def _split_list(raw: str) -> list[str]:
    """Tách chuỗi symbol phân cách bằng dấu phẩy hoặc khoảng trắng."""
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Thiết lập và phân tích tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Engine: kiểm định out-of-sample việc chọn cấu hình chiến lược"
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Danh sách symbol, cách nhau bởi dấu phẩy",
    )
    parser.add_argument("--timeframe", default="1h", help="Khung thời gian (ví dụ: 1h, 15m)")
    parser.add_argument("--days", type=int, default=365, help="Số ngày lịch sử")
    parser.add_argument("--train-days", type=int, default=120, help="Số ngày cho tập train")
    parser.add_argument("--test-days", type=int, default=60, help="Số ngày cho tập test")
    parser.add_argument("--step-days", type=int, default=60, help="Số ngày bước trượt (step)")
    parser.add_argument(
        "--out-csv", default="data/walkforward_1h_v2.csv", help="Đường dẫn xuất kết quả per-window CSV"
    )
    parser.add_argument(
        "--out-md", default="data/walkforward_1h_v2.md", help="Đường dẫn xuất báo cáo Markdown"
    )
    parser.add_argument("--buffer-days", type=int, default=15, help="Số ngày đệm warmup")
    parser.add_argument(
        "--fee-scenarios",
        default="zero,real,double",
        help="Danh sách kịch bản phí (cách nhau bởi dấu phẩy): zero, real, double",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Hàm thực thi chính cho CLI Walk-Forward Engine."""
    args = parse_args(argv)
    symbols = _split_list(args.symbols)
    fee_scenarios = _split_list(args.fee_scenarios)

    all_per_window: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    base = dataclasses.replace(
        TradingConfig(),
        use_fee_aware_sizing=True,
        min_stop_distance_pct=0.008,
    )

    for scenario in fee_scenarios:
        base_cfg = apply_fee_scenario(base, scenario)
        for sym in symbols:
            print(
                f"[run_walkforward] {scenario}: đang chạy {sym} "
                f"({args.timeframe}, {args.days} ngày)..."
            )
            per_window_df, summary_df = run_walkforward(
                symbol=sym,
                timeframe=args.timeframe,
                days=args.days,
                train_days=args.train_days,
                test_days=args.test_days,
                step_days=args.step_days,
                buffer_days=args.buffer_days,
                base_config=base_cfg,
            )
            per_window_df.insert(0, "fee_scenario", scenario)
            summary_df.insert(0, "fee_scenario", scenario)
            all_per_window.append(per_window_df)
            all_summaries.append(summary_df)
            print(
                f"[run_walkforward] {scenario}: hoàn thành {sym}: "
                f"{len(per_window_df)} cửa sổ."
            )

    combined_per_window = (
        pd.concat(all_per_window, ignore_index=True)
        if all_per_window
        else pd.DataFrame()
    )
    combined_summary = (
        pd.concat(all_summaries, ignore_index=True)
        if all_summaries
        else pd.DataFrame()
    )

    # Ghi kết quả chi tiết từng cửa sổ ra file CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_per_window.to_csv(out_csv, index=False)

    # Ghi báo cáo tổng hợp và chi tiết ra Markdown
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    summary_md = to_markdown(combined_summary)
    per_window_md = to_markdown(combined_per_window)
    full_md = (
        f"# Walk-Forward Summary ({args.timeframe})\n\n"
        f"{summary_md}\n\n"
        f"# Walk-Forward Per-Window Results\n\n"
        f"{per_window_md}\n"
    )
    out_md.write_text(full_md, encoding="utf-8")
    print(full_md)
    print(f"\n[run_walkforward] Đã lưu CSV: {out_csv} | Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
