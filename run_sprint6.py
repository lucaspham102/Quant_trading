"""CLI Runner thực nghiệm và đánh giá nghiệm thu Sprint 6.

Thực hiện toàn bộ quy trình kiểm định Sprint 6:
1. Nạp và đồng bộ dữ liệu OHLCV 1h cùng Funding Rate cho rổ tài sản.
2. Kiểm tra chiến lược Funding Carry đa tài sản trên ma trận cấu hình (K_carry = 6).
3. Kiểm tra chiến lược Cross-Sectional Market-Neutral trên ma trận cấu hình (K_xs = 12).
4. Tính toán Newey-West Alpha/Beta vs BTC, Sharpe năm hóa, DSR dùng phương sai ngang V[{SR_k}].
5. Hỗ trợ chạy liền mạch trên toàn bộ [start, end] và tách metric thành 2 segment DEV và HOLDOUT.
6. Đánh giá điều kiện nghiệm thu GO / NO-GO và xuất báo cáo CSV + Markdown.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
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


def _get_segment_mask(
    index: pd.DatetimeIndex,
    seg: str,
    holdout_ts: pd.Timestamp | None,
) -> pd.Series:
    """Trả về mặt nạ boolean (pd.Series) cho phân đoạn DEV hoặc HOLDOUT.

    Tham số:
        index: Chuỗi DatetimeIndex cần lọc.
        seg: Tên phân đoạn ('dev', 'holdout', hoặc phân đoạn duy nhất).
        holdout_ts: Thời điểm bắt đầu tập holdout. Nếu None thì toàn bộ là True.

    Trả về:
        pd.Series boolean cùng index.
    """
    if holdout_ts is None:
        return pd.Series(True, index=index)
    if seg == "dev":
        return pd.Series(index < holdout_ts, index=index)
    if seg == "holdout":
        return pd.Series(index >= holdout_ts, index=index)
    raise ValueError(f"Phân đoạn không hợp lệ: {seg}")


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
        help="Ngày kết thúc (mặc định 2026-04-21 cho DEV; 2026-10-18 khi kèm HOLDOUT)",
    )
    parser.add_argument(
        "--holdout-start",
        default=None,
        help="Ngày bắt đầu tập Holdout (ví dụ 2026-04-21). Nếu None, chạy toàn bộ cửa sổ như segment DEV.",
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
    holdout_start_ts = (
        pd.Timestamp(args.holdout_start, tz="UTC") if args.holdout_start else None
    )

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

        # Lọc dữ liệu trong khoảng toàn bộ [start, end]
        filtered = ohlcv_df.loc[
            (ohlcv_df.index >= start_ts) & (ohlcv_df.index <= end_ts)
        ].copy()
        if filtered.empty:
            continue

        funding_df = fetch_funding_history(
            sym, start_date=args.start, end_date=args.end
        )
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

    segments = ["dev"] if holdout_start_ts is None else ["dev", "holdout"]

    carry_rows: list[dict] = []
    xs_rows: list[dict] = []

    # 3. HƯỚNG CARRY: Lưới 2 cấu trúc x 3 kịch bản phí (K_carry = 6)
    if not args.skip_carry:
        structures = ["coinm_1leg", "spot_perp_2leg"]
        fee_scenarios: dict[str, dict[str, float]] = {
            "zero": {"perp_fee": 0.0, "spot_fee": 0.0, "slippage": 0.0},
            "maker_optimistic": {
                "perp_fee": 0.0002,
                "spot_fee": 0.0002,
                "slippage": 0.0001,
            },
            "real_taker": {
                "perp_fee": 0.0005,
                "spot_fee": 0.00075,
                "slippage": 0.0002,
            },
        }
        k_carry = len(structures) * len(fee_scenarios)

        # Chạy portfolios MỘT LẦN LIỀN MẠCH trên toàn bộ [start, end]
        carry_sim_results: list[dict[str, Any]] = []
        for struct in structures:
            for fee_name, fees in fee_scenarios.items():
                cfg = CarryConfig(
                    structure=struct,
                    perp_fee=fees["perp_fee"],
                    spot_fee=fees["spot_fee"],
                    slippage=fees["slippage"],
                )
                try:
                    res = run_carry_portfolio(aligned_data, cfg, periods_per_year=8760)
                    carry_sim_results.append({
                        "struct": struct,
                        "fee_name": fee_name,
                        "cfg": cfg,
                        "res": res,
                        "error": False,
                    })
                except Exception as exc:
                    carry_sim_results.append({
                        "struct": struct,
                        "fee_name": fee_name,
                        "cfg": cfg,
                        "res": None,
                        "error": True,
                        "error_msg": str(exc),
                    })

        # Xử lý theo từng phân đoạn (DEV / HOLDOUT)
        for seg in segments:
            seg_items: list[dict[str, Any]] = []
            for sim in carry_sim_results:
                if sim["error"] or sim["res"] is None:
                    seg_items.append({
                        "sim": sim,
                        "error": True,
                        "sr_per_period": np.nan,
                    })
                    continue

                res = sim["res"]
                rets = res["returns"]
                mask = _get_segment_mask(rets.index, seg, holdout_start_ts)
                rets_seg = rets.loc[mask]
                bench_seg = benchmark_returns.loc[rets_seg.index]

                sr = annualized_sharpe(rets_seg, periods_per_year=8760)
                sr_per_period = sr / math.sqrt(8760)
                ann_ret = annualized_return_pct(rets_seg, periods_per_year=8760)

                if len(rets_seg) > 0:
                    cum_ret = float(((1.0 + rets_seg).prod() - 1.0) * 100.0)
                    cum_eq = pd.Series(
                        np.cumprod(
                            np.insert(1.0 + rets_seg.to_numpy(dtype=float), 0, 1.0)
                        )
                    )
                    mdd = max_drawdown_pct(cum_eq)
                else:
                    cum_ret = 0.0
                    mdd = 0.0

                nw = compute_newey_west_alpha_beta(
                    rets_seg, bench_seg, periods_per_year=8760
                )

                if holdout_start_ts is None:
                    m = res["metrics"]
                    tot_ret = m.get("Total Return %", cum_ret)
                    tot_funding = m.get("Total Funding Collected", 0.0)
                    tot_fees = m.get("Total Fees Paid", 0.0)
                    tim = m.get("Time in Market %", 0.0)
                    turnover = m.get("Turnover", 0.0)
                else:
                    tot_ret = cum_ret
                    trades_df = res.get("trades_df", pd.DataFrame())
                    if not trades_df.empty and "time" in trades_df.columns:
                        t_mask = _get_segment_mask(
                            pd.DatetimeIndex(trades_df["time"]), seg, holdout_start_ts
                        )
                        trades_seg = trades_df.loc[t_mask.to_numpy(dtype=bool)]
                        tot_fees = (
                            float(trades_seg["fee"].sum())
                            if "fee" in trades_seg.columns
                            else 0.0
                        )
                        turnover = (
                            float(
                                trades_seg["notional"].sum()
                                / sim["cfg"].initial_capital
                            )
                            if "notional" in trades_seg.columns
                            and sim["cfg"].initial_capital > 0
                            else 0.0
                        )
                    else:
                        tot_fees = 0.0
                        turnover = 0.0

                    tot_funding = (
                        float(
                            res["metrics"].get("Total Funding Collected", 0.0)
                            * (len(rets_seg) / len(rets))
                        )
                        if len(rets) > 0
                        else 0.0
                    )

                    in_market_series = pd.Series(False, index=rets.index)
                    if not trades_df.empty and "time" in trades_df.columns:
                        for sym in aligned_data:
                            sym_trades = trades_df[trades_df["symbol"] == sym]
                            opens = sym_trades[sym_trades["action"] == "open"][
                                "time"
                            ].tolist()
                            closes = sym_trades[sym_trades["action"] == "close"][
                                "time"
                            ].tolist()
                            for op_i, op_t in enumerate(opens):
                                cl_t = (
                                    closes[op_i]
                                    if op_i < len(closes)
                                    else rets.index[-1]
                                )
                                in_market_series.loc[
                                    (in_market_series.index >= op_t)
                                    & (in_market_series.index <= cl_t)
                                ] = True
                    tim = (
                        float(in_market_series.loc[mask].mean() * 100.0)
                        if len(rets_seg) > 0
                        else 0.0
                    )

                seg_items.append({
                    "sim": sim,
                    "error": False,
                    "rets_seg": rets_seg,
                    "sr_per_period": sr_per_period,
                    "sharpe": sr,
                    "total_return_pct": tot_ret,
                    "annualized_return_pct": ann_ret,
                    "max_dd_pct": mdd,
                    "nw": nw,
                    "total_funding": tot_funding,
                    "total_fees": tot_fees,
                    "time_in_market_pct": tim,
                    "turnover": turnover,
                })

            # Tính phương sai ngang V[{SR_k}] cho các cấu hình trong cùng segment
            valid_srs = [
                item["sr_per_period"]
                for item in seg_items
                if not item["error"] and np.isfinite(item["sr_per_period"])
            ]
            sr_var = (
                float(np.var(valid_srs, ddof=1)) if len(valid_srs) >= 2 else None
            )

            # Tính lại DSR và cập nhật dòng kết quả
            for item in seg_items:
                sim = item["sim"]
                if item["error"]:
                    carry_rows.append({
                        "segment": seg,
                        "structure": sim["struct"],
                        "fee_scenario": sim["fee_name"],
                        "total_return_pct": 0.0,
                        "annualized_return_pct": 0.0,
                        "sharpe": 0.0,
                        "max_dd_pct": 0.0,
                        "dsr_prob": 0.0,
                        "sr_star": 0.0,
                        "alpha": 0.0,
                        "beta": 0.0,
                        "alpha_t": 0.0,
                        "beta_t": 0.0,
                        "alpha_annualized_pct": 0.0,
                        "total_funding": 0.0,
                        "total_fees": 0.0,
                        "time_in_market_pct": 0.0,
                        "turnover": 0.0,
                    })
                    continue

                dsr = compute_deflated_sharpe_ratio(
                    item["rets_seg"],
                    num_trials=k_carry,
                    periods_per_year=8760,
                    sr_variance=sr_var,
                )
                carry_rows.append({
                    "segment": seg,
                    "structure": sim["struct"],
                    "fee_scenario": sim["fee_name"],
                    "total_return_pct": item["total_return_pct"],
                    "annualized_return_pct": item["annualized_return_pct"],
                    "sharpe": item["sharpe"],
                    "max_dd_pct": item["max_dd_pct"],
                    "dsr_prob": dsr["dsr_prob"],
                    "sr_star": dsr["sr_star"],
                    "alpha": item["nw"]["alpha"],
                    "beta": item["nw"]["beta"],
                    "alpha_t": item["nw"]["alpha_t"],
                    "beta_t": item["nw"]["beta_t"],
                    "alpha_annualized_pct": item["nw"]["alpha_annualized_pct"],
                    "total_funding": item["total_funding"],
                    "total_fees": item["total_fees"],
                    "time_in_market_pct": item["time_in_market_pct"],
                    "turnover": item["turnover"],
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

        xs_sim_results: list[dict[str, Any]] = []
        for mode in modes:
            for lb in lookbacks:
                for rb in rebalances:
                    cfg = CrossSectionalConfig(
                        mode=mode,
                        lookback_hours=lb,
                        rebalance_hours=rb,
                    )
                    try:
                        res = run_cross_sectional_portfolio(
                            aligned_data, cfg, periods_per_year=8760
                        )
                        xs_sim_results.append({
                            "mode": mode,
                            "lookback_hours": lb,
                            "rebalance_hours": rb,
                            "cfg": cfg,
                            "res": res,
                            "error": False,
                        })
                    except Exception as exc:
                        xs_sim_results.append({
                            "mode": mode,
                            "lookback_hours": lb,
                            "rebalance_hours": rb,
                            "cfg": cfg,
                            "res": None,
                            "error": True,
                            "error_msg": str(exc),
                        })

        for seg in segments:
            seg_items = []
            for sim in xs_sim_results:
                if sim["error"] or sim["res"] is None:
                    seg_items.append({
                        "sim": sim,
                        "error": True,
                        "sr_per_period": np.nan,
                    })
                    continue

                res = sim["res"]
                rets = res["returns"]
                mask = _get_segment_mask(rets.index, seg, holdout_start_ts)
                rets_seg = rets.loc[mask]
                bench_seg = benchmark_returns.loc[rets_seg.index]

                sr = annualized_sharpe(rets_seg, periods_per_year=8760)
                sr_per_period = sr / math.sqrt(8760)
                ann_ret = annualized_return_pct(rets_seg, periods_per_year=8760)

                if len(rets_seg) > 0:
                    cum_ret = float(((1.0 + rets_seg).prod() - 1.0) * 100.0)
                    cum_eq = pd.Series(
                        np.cumprod(
                            np.insert(1.0 + rets_seg.to_numpy(dtype=float), 0, 1.0)
                        )
                    )
                    mdd = max_drawdown_pct(cum_eq)
                else:
                    cum_ret = 0.0
                    mdd = 0.0

                nw = compute_newey_west_alpha_beta(
                    rets_seg, bench_seg, periods_per_year=8760
                )

                if holdout_start_ts is None:
                    m = res["metrics"]
                    tot_ret = m.get("Total Return %", cum_ret)
                    tot_funding = m.get("Total Funding Collected", 0.0)
                    tot_fees = m.get("Total Fees Paid", 0.0)
                    tim = m.get("Time in Market %", 0.0)
                    turnover = m.get("Turnover", 0.0)
                else:
                    tot_ret = cum_ret
                    trades_df = res.get("trades_df", pd.DataFrame())
                    weights_df = res.get("weights_df", pd.DataFrame())
                    if not trades_df.empty and "time" in trades_df.columns:
                        t_mask = _get_segment_mask(
                            pd.DatetimeIndex(trades_df["time"]), seg, holdout_start_ts
                        )
                        trades_seg = trades_df.loc[t_mask.to_numpy(dtype=bool)]
                        turnover = (
                            float(trades_seg["turnover"].sum())
                            if "turnover" in trades_seg.columns
                            else 0.0
                        )
                        all_to = float(res["metrics"].get("Turnover", 0.0))
                        all_fees = float(res["metrics"].get("Total Fees Paid", 0.0))
                        tot_fees = (
                            float(all_fees * (turnover / all_to))
                            if all_to > 0
                            else 0.0
                        )
                    else:
                        turnover = 0.0
                        tot_fees = 0.0

                    if not weights_df.empty:
                        w_seg = weights_df.loc[mask]
                        in_mkt = np.sum(np.abs(w_seg.values), axis=1) > 1e-9
                        tim = (
                            float(np.mean(in_mkt) * 100.0)
                            if len(in_mkt) > 0
                            else 0.0
                        )
                    else:
                        tim = 0.0

                    all_funding = float(
                        res["metrics"].get("Total Funding Collected", 0.0)
                    )
                    tot_funding = (
                        float(all_funding * (len(rets_seg) / len(rets)))
                        if len(rets) > 0
                        else 0.0
                    )

                seg_items.append({
                    "sim": sim,
                    "error": False,
                    "rets_seg": rets_seg,
                    "sr_per_period": sr_per_period,
                    "sharpe": sr,
                    "total_return_pct": tot_ret,
                    "annualized_return_pct": ann_ret,
                    "max_dd_pct": mdd,
                    "nw": nw,
                    "total_funding": tot_funding,
                    "total_fees": tot_fees,
                    "time_in_market_pct": tim,
                    "turnover": turnover,
                })

            valid_srs = [
                item["sr_per_period"]
                for item in seg_items
                if not item["error"] and np.isfinite(item["sr_per_period"])
            ]
            sr_var = (
                float(np.var(valid_srs, ddof=1)) if len(valid_srs) >= 2 else None
            )

            for item in seg_items:
                sim = item["sim"]
                if item["error"]:
                    xs_rows.append({
                        "segment": seg,
                        "mode": sim["mode"],
                        "lookback_hours": sim["lookback_hours"],
                        "rebalance_hours": sim["rebalance_hours"],
                        "total_return_pct": 0.0,
                        "annualized_return_pct": 0.0,
                        "sharpe": 0.0,
                        "max_dd_pct": 0.0,
                        "dsr_prob": 0.0,
                        "sr_star": 0.0,
                        "alpha": 0.0,
                        "beta": 0.0,
                        "alpha_t": 0.0,
                        "beta_t": 0.0,
                        "alpha_annualized_pct": 0.0,
                        "total_funding": 0.0,
                        "total_fees": 0.0,
                        "time_in_market_pct": 0.0,
                        "turnover": 0.0,
                    })
                    continue

                dsr = compute_deflated_sharpe_ratio(
                    item["rets_seg"],
                    num_trials=k_xs,
                    periods_per_year=8760,
                    sr_variance=sr_var,
                )
                xs_rows.append({
                    "segment": seg,
                    "mode": sim["mode"],
                    "lookback_hours": sim["lookback_hours"],
                    "rebalance_hours": sim["rebalance_hours"],
                    "total_return_pct": item["total_return_pct"],
                    "annualized_return_pct": item["annualized_return_pct"],
                    "sharpe": item["sharpe"],
                    "max_dd_pct": item["max_dd_pct"],
                    "dsr_prob": dsr["dsr_prob"],
                    "sr_star": dsr["sr_star"],
                    "alpha": item["nw"]["alpha"],
                    "beta": item["nw"]["beta"],
                    "alpha_t": item["nw"]["alpha_t"],
                    "beta_t": item["nw"]["beta_t"],
                    "alpha_annualized_pct": item["nw"]["alpha_annualized_pct"],
                    "total_funding": item["total_funding"],
                    "total_fees": item["total_fees"],
                    "time_in_market_pct": item["time_in_market_pct"],
                    "turnover": item["turnover"],
                })

        xs_df = pd.DataFrame(xs_rows)
        xs_csv = out_dir / "sprint6_xs.csv"
        xs_df.to_csv(xs_csv, index=False)

    # 5. Lập báo cáo Markdown tổng hợp và đánh giá GO / NO-GO
    has_holdout = holdout_start_ts is not None
    if has_holdout:
        dataset_desc = f"Phân đoạn liền mạch: DEV ({args.start} → {args.holdout_start}) & HOLDOUT ({args.holdout_start} → {args.end})"
    else:
        is_holdout = str(args.end).startswith("2026-10-18")
        dataset_type = "HOLDOUT" if is_holdout else "DEV (Phát triển / Walk-forward)"
        dataset_desc = f"{dataset_type} ({args.start} → {args.end})"

    report_lines: list[str] = [
        "# Sprint 6 — Báo cáo nghiệm thu",
        "",
        f"**Cửa sổ dữ liệu:** {dataset_desc}",
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
            f"*Tổng số cấu hình thử nghiệm K = {len(structures) * len(fee_scenarios)}*",
            "",
        ])

        if has_holdout:
            dev_carry = carry_df[carry_df["segment"] == "dev"]
            holdout_carry = carry_df[carry_df["segment"] == "holdout"]
            report_lines.extend([
                "### Phân đoạn DEV (Kịch bản real_taker):",
                to_markdown(dev_carry[dev_carry["fee_scenario"] == "real_taker"]),
                "",
                "### Phân đoạn DEV (Toàn bộ cấu hình):",
                to_markdown(dev_carry),
                "",
                "### Phân đoạn HOLDOUT (Kịch bản real_taker):",
                to_markdown(
                    holdout_carry[holdout_carry["fee_scenario"] == "real_taker"]
                ),
                "",
                "### Phân đoạn HOLDOUT (Toàn bộ cấu hình):",
                to_markdown(holdout_carry),
                "",
                "### Đánh giá nghiệm thu GO / NO-GO (Hướng Carry):",
            ])

            for struct in structures:
                dev_r = next(
                    (
                        r
                        for r in carry_rows
                        if r["segment"] == "dev"
                        and r["structure"] == struct
                        and r["fee_scenario"] == "real_taker"
                    ),
                    None,
                )
                holdout_r = next(
                    (
                        r
                        for r in carry_rows
                        if r["segment"] == "holdout"
                        and r["structure"] == struct
                        and r["fee_scenario"] == "real_taker"
                    ),
                    None,
                )
                if not dev_r:
                    continue

                eval_dict = {
                    "alpha_t": dev_r["alpha_t"],
                    "beta": dev_r["beta"],
                    "sharpe": dev_r["sharpe"],
                    "dsr_prob": dev_r["dsr_prob"],
                    "holdout_return_pct": (
                        holdout_r["total_return_pct"] if holdout_r else 0.0
                    ),
                    "holdout_sharpe": holdout_r["sharpe"] if holdout_r else 0.0,
                    "holdout_max_dd": (
                        holdout_r["max_dd_pct"] if holdout_r else 999.0
                    ),
                }
                eval_res = evaluate_go_no_go(eval_dict, direction="carry")
                report_lines.extend([
                    f"- **Cấu trúc `{struct}` (real_taker)**:",
                    f"  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): {'ĐẠT' if eval_res['cond1_market_neutral'] else 'KHÔNG ĐẠT'} (alpha_t={dev_r['alpha_t']:.2f}, beta={dev_r['beta']:.4f})",
                    f"  - Điều kiện 2 (Sharpe >= 1.50 [DEV]): {'ĐẠT' if eval_res['cond2_sharpe'] else 'KHÔNG ĐẠT'} (Sharpe={dev_r['sharpe']:.2f})",
                    f"  - Điều kiện 3 (DSR >= 0.95 [DEV]): {'ĐẠT' if eval_res['cond3_dsr'] else 'KHÔNG ĐẠT'} (DSR={dev_r['dsr_prob']:.4f}, SR*={dev_r['sr_star']:.4f})",
                    f"  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): {'ĐẠT' if eval_res['cond4_holdout'] else 'KHÔNG ĐẠT'} (NetRet={eval_dict['holdout_return_pct']:.2f}%, Sharpe={eval_dict['holdout_sharpe']:.2f}, MaxDD={eval_dict['holdout_max_dd']:.2f}%)",
                    f"  - **Kết luận:** `{eval_res['verdict']}`",
                ])
        else:
            report_lines.extend([
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
            f"*Tổng số cấu hình thử nghiệm K = {len(modes) * len(lookbacks) * len(rebalances)}*",
            "",
        ])

        if has_holdout:
            dev_xs = xs_df[xs_df["segment"] == "dev"]
            holdout_xs = xs_df[xs_df["segment"] == "holdout"]
            report_lines.extend([
                "### Phân đoạn DEV (Toàn bộ cấu hình):",
                to_markdown(dev_xs),
                "",
                "### Phân đoạn HOLDOUT (Toàn bộ cấu hình):",
                to_markdown(holdout_xs),
                "",
                "### Đánh giá nghiệm thu GO / NO-GO (Hướng Cross-Sectional):",
            ])

            for mode in modes:
                for lb in lookbacks:
                    for rb in rebalances:
                        dev_r = next(
                            (
                                r
                                for r in xs_rows
                                if r["segment"] == "dev"
                                and r["mode"] == mode
                                and r["lookback_hours"] == lb
                                and r["rebalance_hours"] == rb
                            ),
                            None,
                        )
                        holdout_r = next(
                            (
                                r
                                for r in xs_rows
                                if r["segment"] == "holdout"
                                and r["mode"] == mode
                                and r["lookback_hours"] == lb
                                and r["rebalance_hours"] == rb
                            ),
                            None,
                        )
                        if not dev_r:
                            continue

                        eval_dict = {
                            "alpha_t": dev_r["alpha_t"],
                            "beta": dev_r["beta"],
                            "sharpe": dev_r["sharpe"],
                            "dsr_prob": dev_r["dsr_prob"],
                            "holdout_return_pct": (
                                holdout_r["total_return_pct"]
                                if holdout_r
                                else 0.0
                            ),
                            "holdout_sharpe": (
                                holdout_r["sharpe"] if holdout_r else 0.0
                            ),
                            "holdout_max_dd": (
                                holdout_r["max_dd_pct"] if holdout_r else 999.0
                            ),
                        }
                        eval_res = evaluate_go_no_go(eval_dict, direction="xs")
                        report_lines.extend([
                            f"- **Cấu hình `{mode}` lookback={lb}h rebalance={rb}h**:",
                            f"  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): {'ĐẠT' if eval_res['cond1_market_neutral'] else 'KHÔNG ĐẠT'} (alpha_t={dev_r['alpha_t']:.2f}, beta={dev_r['beta']:.4f})",
                            f"  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): {'ĐẠT' if eval_res['cond2_sharpe'] else 'KHÔNG ĐẠT'} (Sharpe={dev_r['sharpe']:.2f})",
                            f"  - Điều kiện 3 (DSR >= 0.95 [DEV]): {'ĐẠT' if eval_res['cond3_dsr'] else 'KHÔNG ĐẠT'} (DSR={dev_r['dsr_prob']:.4f}, SR*={dev_r['sr_star']:.4f})",
                            f"  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): {'ĐẠT' if eval_res['cond4_holdout'] else 'KHÔNG ĐẠT'} (NetRet={eval_dict['holdout_return_pct']:.2f}%, Sharpe={eval_dict['holdout_sharpe']:.2f}, MaxDD={eval_dict['holdout_max_dd']:.2f}%)",
                            f"  - **Kết luận:** `{eval_res['verdict']}`",
                        ])
        else:
            report_lines.extend([
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
