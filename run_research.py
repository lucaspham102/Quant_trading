"""CLI Research Engine: quét ma trận tổ hợp chiến lược đa tài sản / đa khung thời gian.

Ví dụ:
    python run_research.py --symbols BTC/USDT:USDT,ETH/USDT:USDT --timeframes 1h,15m --days 90
    python run_research.py --synthetic   # chạy offline không cần mạng
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.research import (
    DEFAULT_SYMBOLS,
    DEFAULT_TIMEFRAMES,
    export_research_results,
    run_research_matrix,
)


def _split_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(" ", ",").split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research Engine: ma trận tổ hợp chiến lược (Binance USDT-M Futures)"
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Danh sách symbol, cách nhau bởi dấu phẩy")
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES), help="Danh sách khung thời gian")
    parser.add_argument("--days", type=int, default=180, help="Số ngày lịch sử khi tải dữ liệu thật")
    parser.add_argument("--out-csv", default="data/research_results.csv", help="Đường dẫn xuất CSV")
    parser.add_argument("--out-md", default="data/research_results.md", help="Đường dẫn xuất Markdown")
    parser.add_argument("--synthetic", action="store_true", help="Dùng dữ liệu giả lập (offline, không gọi mạng)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = run_research_matrix(
            symbols=_split_list(args.symbols),
            timeframes=_split_list(args.timeframes),
            days=args.days,
            use_synthetic=args.synthetic,
        )
        md = export_research_results(results, Path(args.out_csv), Path(args.out_md))
        print(md)
        print(
            f"\n[run_research] {len(results)} tổ hợp | CSV: {args.out_csv} | MD: {args.out_md}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Lỗi khi chạy research: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
