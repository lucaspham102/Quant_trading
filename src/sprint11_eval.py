"""Module danh gia va kiem dinh chat luong Sprint 11 (Single-Venue Tier 1+ Barbell).

Cung cap:
- SPRINT11_SPLIT_DATE, SPRINT11_HOLDOUT_END, SPRINT11_INITIAL_CAPITAL, SPRINT11_GO_CRITERIA
- compute_beta_to_benchmark: Tinh beta cua loi nhuan so voi benchmark (ddof=1).
- split_period_returns: Chia chuoi returns thanh n doan va tinh return_pct moi doan.
- run_sprint11_tiers: Mo phong luoi 4 combo (tier1a/tier1b x hard_binary/smooth_conviction).
- smoke_test_executor: Kiem tra cau truc va runtime paper executor tren stub exchange.
- evaluate_sprint11_go_no_go: Kiem dinh 10 tieu chi GO/NO-GO theo docs/go_no_go_sprint11.md.
- build_sprint11_summary_rows: Tao bang phang cac dong tong hop chi so xuat markdown/csv.
"""
from __future__ import annotations

import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.carry_live_executor import CarryLiveExecutor
from src.cross_exchange_carry import LIQUID_8_SYMBOLS, run_binance_tier1_plus_carry
from src.state_store import BotState, StateStore

logger = logging.getLogger("sprint11_eval")

# ---------------------------------------------------------------------------
# 1. Hang so Sprint 11
# ---------------------------------------------------------------------------
SPRINT11_SPLIT_DATE: str = "2026-04-21"
SPRINT11_HOLDOUT_END: str = "2026-10-18"
SPRINT11_INITIAL_CAPITAL: float = 3000.0
SPRINT11_GO_CRITERIA: dict[str, float | int] = {
    "annualized_return_pct": 9.0,
    "min_positive_periods": 6,
    "holdout_return_pct": 3.5,
    "sharpe": 1.80,
    "maxdd_pct": 7.5,
    "abs_beta_btc": 0.15,
}

__all__ = [
    "SPRINT11_SPLIT_DATE",
    "SPRINT11_HOLDOUT_END",
    "SPRINT11_INITIAL_CAPITAL",
    "SPRINT11_GO_CRITERIA",
    "compute_beta_to_benchmark",
    "split_period_returns",
    "run_sprint11_tiers",
    "smoke_test_executor",
    "evaluate_sprint11_go_no_go",
    "build_sprint11_summary_rows",
    "StubExchange",
]


# ---------------------------------------------------------------------------
# 2. compute_beta_to_benchmark
# ---------------------------------------------------------------------------
def compute_beta_to_benchmark(returns: pd.Series, benchmark: pd.Series) -> float:
    """Tinh he so beta cua chuoi loi nhuan so voi benchmark bang OLS don (ddof=1).

    Can index, dropna. Neu var(benchmark) <= 1e-18 hoac khong hop le thi tra ve 0.0.
    """
    if returns is None or benchmark is None:
        return 0.0
    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        return 0.0
    if returns.empty or benchmark.empty:
        return 0.0

    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0

    r = aligned.iloc[:, 0].to_numpy(dtype=np.float64)
    b = aligned.iloc[:, 1].to_numpy(dtype=np.float64)

    var_b = float(np.var(b, ddof=1))
    if var_b <= 1e-18 or not np.isfinite(var_b):
        return 0.0

    cov_mat = np.cov(r, b, ddof=1)
    cov_rb = float(cov_mat[0, 1])
    if not np.isfinite(cov_rb):
        return 0.0

    return float(cov_rb / var_b)


# ---------------------------------------------------------------------------
# 3. split_period_returns
# ---------------------------------------------------------------------------
def split_period_returns(returns: pd.Series, n_periods: int = 7) -> list[dict[str, Any]]:
    """Chia chuoi loi nhuan thanh n_periods doan bang nhau va tinh return_pct moi doan.

    Tra ve list cac dict gom: period, start, end, return_pct.
    """
    if returns is None or not isinstance(returns, pd.Series) or returns.empty:
        return []

    n = max(1, int(n_periods))
    splits = np.array_split(np.arange(len(returns)), n)
    out: list[dict[str, Any]] = []

    for i, idx_arr in enumerate(splits):
        if len(idx_arr) == 0:
            continue
        sub = returns.iloc[idx_arr]
        start_ts = sub.index[0]
        end_ts = sub.index[-1]
        start_iso = start_ts.isoformat() if hasattr(start_ts, "isoformat") else str(start_ts)
        end_iso = end_ts.isoformat() if hasattr(end_ts, "isoformat") else str(end_ts)
        r_arr = sub.to_numpy(dtype=np.float64)
        ret_pct = float((np.prod(1.0 + r_arr) - 1.0) * 100.0)
        out.append({
            "period": i + 1,
            "start": start_iso,
            "end": end_iso,
            "return_pct": ret_pct,
        })

    return out


# ---------------------------------------------------------------------------
# Helper cat dataset theo range
# ---------------------------------------------------------------------------
def _slice_data_range(
    data: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Cat dict DataFrames theo khoang thoi gian [start_date, end_date] an toan timezone."""
    sliced: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        if df is None or df.empty:
            sliced[sym] = df
            continue
        if isinstance(df.index, pd.DatetimeIndex):
            tz = df.index.tz
            s_ts = pd.Timestamp(start_date)
            e_ts = pd.Timestamp(end_date)
            if tz is not None:
                if s_ts.tzinfo is None:
                    s_ts = s_ts.tz_localize(tz)
                else:
                    s_ts = s_ts.tz_convert(tz)
                if e_ts.tzinfo is None:
                    e_ts = e_ts.tz_localize(tz)
                else:
                    e_ts = e_ts.tz_convert(tz)
            e_ts_end = e_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            sub_df = df.loc[(df.index >= s_ts) & (df.index <= e_ts_end)]
            sliced[sym] = sub_df
        else:
            sliced[sym] = df
    return sliced


# ---------------------------------------------------------------------------
# 4. run_sprint11_tiers
# ---------------------------------------------------------------------------
def run_sprint11_tiers(
    data: dict[str, pd.DataFrame],
    capital: float = SPRINT11_INITIAL_CAPITAL,
    split_date: str = SPRINT11_SPLIT_DATE,
    holdout_end: str = SPRINT11_HOLDOUT_END,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    """Mo phong 4 combo tham so Sprint 11 tren toan bo ky va holdout.

    Luoi 4 combo: carry_variant in ("tier1a", "tier1b") x gating_mode in ("hard_binary", "smooth_conviction").
    Moi combo duoc boc try/except de khong raise toan bo neu co 1 combo bi loi.
    """
    combos = [
        ("tier1a", "hard_binary"),
        ("tier1a", "smooth_conviction"),
        ("tier1b", "hard_binary"),
        ("tier1b", "smooth_conviction"),
    ]

    holdout_data = _slice_data_range(data, split_date, holdout_end)
    rows: list[dict[str, Any]] = []

    for variant, gating in combos:
        combo_name = f"{variant}/{gating}"
        try:
            cfg = BarbellConfig(
                initial_capital=float(capital),
                carry_variant=variant,
                gating_mode=gating,
                enable_snr_gate=True,
                overlay_quantize_legs=True,
            )

            # 1. Chay tren toan bo ky
            res = run_barbell_portfolio(data, cfg)
            metrics = res.get("metrics", {}) if isinstance(res, dict) else {}
            go_inputs = res.get("go_no_go_inputs", {}) if isinstance(res, dict) else {}

            ann_ret = float(go_inputs.get("annualized_return_pct", metrics.get("Annualized Return %", 0.0)))
            sharpe = float(go_inputs.get("sharpe", metrics.get("Sharpe", 0.0)))
            maxdd = float(go_inputs.get("maxdd_pct", metrics.get("MaxDD %", 0.0)))
            final_eq = float(go_inputs.get("final_equity", res.get("final_equity", capital)))
            net_pnl = float(go_inputs.get("net_pnl", res.get("net_pnl", final_eq - capital)))

            c_var = str(res.get("carry_variant", variant))
            g_mode = str(res.get("gating_mode", gating))
            c_fallback = bool(res.get("carry_fallback", False))

            ov_gross = float(res.get("overlay_gross_pnl", metrics.get("overlay_gross_pnl", 0.0)))
            ov_net = float(res.get("overlay_net_pnl", metrics.get("overlay_net_pnl", 0.0)))
            ov_fees = float(res.get("overlay_fees", metrics.get("overlay_fees", 0.0)))
            ov_quant = int(res.get("overlay_quantization_events", metrics.get("overlay_quantization_events", 0)))
            ov_mean_lev = float(metrics.get("overlay_mean_leverage", 0.0))
            ov_gate_ratio = float(metrics.get("overlay_gate_on_ratio", metrics.get("overlay_active_ratio", 0.0)))

            attr_df = res.get("attribution_series")
            if isinstance(attr_df, pd.DataFrame) and not attr_df.empty:
                if "basis_pnl" in attr_df.columns and len(attr_df["basis_pnl"]) > 1:
                    basis_mtm_std = float(attr_df["basis_pnl"].std(ddof=1))
                else:
                    basis_mtm_std = 0.0
                f_sum = float(attr_df["funding_pnl"].sum()) if "funding_pnl" in attr_df.columns else 0.0
                b_sum = float(attr_df["basis_pnl"].sum()) if "basis_pnl" in attr_df.columns else 0.0
                carry_pnl = float(f_sum + b_sum)
                net_attr_sum = float(attr_df["net_pnl"].sum()) if "net_pnl" in attr_df.columns else 0.0
            else:
                basis_mtm_std = 0.0
                carry_pnl = 0.0
                net_attr_sum = 0.0

            rets = res.get("returns")
            beta_val = compute_beta_to_benchmark(rets, benchmark_returns) if benchmark_returns is not None else 0.0

            periods_list = split_period_returns(rets, 7) if isinstance(rets, pd.Series) else []
            pos_periods = sum(1 for p in periods_list if float(p.get("return_pct", 0.0)) > 0.0)

            # 2. Chay tren ky holdout
            res_h = run_barbell_portfolio(holdout_data, cfg)
            h_metrics = res_h.get("metrics", {}) if isinstance(res_h, dict) else {}
            h_final_eq = float(res_h.get("go_no_go_inputs", {}).get("final_equity", h_metrics.get("final_equity", capital)))
            h_ret_pct = float(h_metrics.get("Total Return %", ((h_final_eq - capital) / capital) * 100.0 if capital > 0 else 0.0))

            h_attr = res_h.get("attribution_series")
            if isinstance(h_attr, pd.DataFrame) and not h_attr.empty:
                h_funding = float(h_attr["funding_pnl"].sum()) if "funding_pnl" in h_attr.columns else 0.0
                h_basis = float(h_attr["basis_pnl"].sum()) if "basis_pnl" in h_attr.columns else 0.0
                h_carry_pnl = float(h_funding + h_basis)
            else:
                h_carry_pnl = 0.0

            h_ov_net = float(res_h.get("overlay_net_pnl", h_metrics.get("overlay_net_pnl", 0.0)))
            h_rets = res_h.get("returns")
            h_days = float(len(h_rets) / 24.0) if isinstance(h_rets, pd.Series) and len(h_rets) > 0 else 0.0

            # 2b. Holdout theo phuong phap cua so lien tuc (slice tu ket qua toan ky)
            h_start_ts = pd.Timestamp(split_date, tz="UTC")
            h_end_ts = pd.Timestamp(holdout_end, tz="UTC")
            h_net_pnl = 0.0
            h_carry_slice = 0.0
            h_overlay_slice = 0.0
            h_fees_slice = 0.0
            h_equity_start = float(capital)
            h_compound_slice_pct = 0.0
            if isinstance(attr_df, pd.DataFrame) and not attr_df.empty:
                m = (attr_df.index >= h_start_ts) & (attr_df.index <= h_end_ts)
                if bool(m.any()):
                    h_net_pnl = float(attr_df.loc[m, "net_pnl"].sum())
                    if "funding_pnl" in attr_df.columns and "basis_pnl" in attr_df.columns:
                        h_carry_slice = float(attr_df.loc[m, "funding_pnl"].sum() + attr_df.loc[m, "basis_pnl"].sum())
                    if "overlay_pnl" in attr_df.columns:
                        h_overlay_slice = float(attr_df.loc[m, "overlay_pnl"].sum())
                    if "fees" in attr_df.columns:
                        h_fees_slice = float(attr_df.loc[m, "fees"].sum())
            eq_full = res.get("equity_curve")
            if isinstance(eq_full, pd.Series) and len(eq_full) > 0:
                before = eq_full[eq_full.index < h_start_ts]
                if len(before) > 0:
                    h_equity_start = float(before.iloc[-1])
            if isinstance(rets, pd.Series) and len(rets) > 0:
                mr = (rets.index >= h_start_ts) & (rets.index <= h_end_ts)
                if bool(mr.any()):
                    h_compound_slice_pct = float((np.prod(1.0 + rets[mr].to_numpy(dtype=np.float64)) - 1.0) * 100.0)
            h_ret_active_pct = (h_net_pnl / float(capital) * 100.0) if float(capital) > 0.0 else 0.0
            h_ret_compound_pct = (h_net_pnl / h_equity_start * 100.0) if h_equity_start > 0.0 else 0.0

            accounting_gap = float(abs(net_attr_sum - (final_eq - capital)))

            row = {
                "combo": combo_name,
                "carry_variant": c_var,
                "gating_mode": g_mode,
                "carry_fallback": c_fallback,
                "final_equity": final_eq,
                "net_pnl": net_pnl,
                "annualized_return_pct": ann_ret,
                "sharpe": sharpe,
                "maxdd": maxdd,
                "maxdd_pct": maxdd,
                "basis_mtm_std": basis_mtm_std,
                "carry_pnl": carry_pnl,
                "overlay_gross_pnl": ov_gross,
                "overlay_net_pnl": ov_net,
                "overlay_fees": ov_fees,
                "overlay_quantization_events": ov_quant,
                "overlay_mean_leverage": ov_mean_lev,
                "overlay_gate_on_ratio": ov_gate_ratio,
                "beta": beta_val,
                "beta_btc": beta_val,
                "positive_periods": int(pos_periods),
                "total_periods": 7,
                "periods": periods_list,
                "holdout_return_pct": h_compound_slice_pct,
                "holdout_net_pnl": h_net_pnl,
                "holdout_fees": h_fees_slice,
                "holdout_equity_start": h_equity_start,
                "holdout_return_on_active_capital_pct": h_ret_active_pct,
                "holdout_return_on_compounded_equity_pct": h_ret_compound_pct,
                "holdout_return_resim_pct": h_ret_pct,
                "holdout_net_pnl_resim": float(h_final_eq - float(capital)),
                "reinvest_profits": bool(getattr(cfg, "reinvest_profits", True)),
                "holdout_carry_pnl": (h_carry_slice if not (h_carry_slice == 0.0 and h_carry_pnl > 0.0) else h_carry_pnl),
                "holdout_overlay_net_pnl": (h_overlay_slice if not (h_overlay_slice == 0.0 and h_ov_net != 0.0) else h_ov_net),
                "holdout_days": h_days,
                "accounting_gap": accounting_gap,
                "executor_functional": False,
                "error": None,
            }
            rows.append(row)

        except Exception as e:
            logger.warning("Error running combo %s: %s", combo_name, e)
            rows.append({
                "combo": combo_name,
                "carry_variant": variant,
                "gating_mode": gating,
                "carry_fallback": False,
                "final_equity": float(capital),
                "net_pnl": 0.0,
                "annualized_return_pct": 0.0,
                "sharpe": 0.0,
                "maxdd": 0.0,
                "maxdd_pct": 0.0,
                "basis_mtm_std": 0.0,
                "carry_pnl": 0.0,
                "overlay_gross_pnl": 0.0,
                "overlay_net_pnl": 0.0,
                "overlay_fees": 0.0,
                "overlay_quantization_events": 0,
                "overlay_mean_leverage": 0.0,
                "overlay_gate_on_ratio": 0.0,
                "beta": 0.0,
                "beta_btc": 0.0,
                "positive_periods": 0,
                "total_periods": 7,
                "holdout_return_pct": 0.0,
                "holdout_net_pnl": 0.0,
                "holdout_fees": 0.0,
                "holdout_equity_start": 0.0,
                "holdout_return_on_active_capital_pct": 0.0,
                "holdout_return_on_compounded_equity_pct": 0.0,
                "holdout_return_resim_pct": 0.0,
                "holdout_net_pnl_resim": 0.0,
                "reinvest_profits": True,
                "holdout_carry_pnl": 0.0,
                "holdout_overlay_net_pnl": 0.0,
                "holdout_days": 0.0,
                "accounting_gap": 0.0,
                "executor_functional": False,
                "error": str(e),
            })

    return {
        "rows": rows,
        "capital": float(capital),
        "split_date": str(split_date),
        "holdout_end": str(holdout_end),
        "n_combos": len(rows),
    }


# ---------------------------------------------------------------------------
# 5. StubExchange & smoke_test_executor
# ---------------------------------------------------------------------------
class StubExchange:
    """Stub exchange in-memory mo phong API Binance phuc vu smoke test."""

    def __init__(self) -> None:
        self.order_counter = 0

    def milliseconds(self) -> int:
        return 1776700000000

    def fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        return {
            "bids": [[59990.0, 10.0], [59980.0, 10.0]],
            "asks": [[60010.0, 10.0], [60020.0, 10.0]],
        }

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.order_counter += 1
        return {
            "id": f"stub_order_{self.order_counter}",
            "status": "closed",
            "filled": float(amount),
            "amount": float(amount),
            "price": float(price or 60000.0),
            "symbol": symbol,
            "side": side,
            "type": order_type,
        }

    def fetch_order(self, id: str, symbol: str | None = None) -> dict[str, Any]:
        return {
            "id": str(id),
            "status": "closed",
            "filled": 0.002,
            "amount": 0.002,
            "symbol": symbol,
        }

    def cancel_order(self, id: str, symbol: str | None = None) -> bool:
        return True


def smoke_test_executor(tmp_path: Path | str | None = None) -> dict[str, Any]:
    """Kiem tra cau truc va runtime cua CarryLiveExecutor va BotState tren stub exchange."""
    methods_present = (
        hasattr(CarryLiveExecutor, "execute_maker_then_taker_pair")
        and callable(getattr(CarryLiveExecutor, "execute_maker_then_taker_pair"))
        and hasattr(CarryLiveExecutor, "execute_overlay_rebalance")
        and callable(getattr(CarryLiveExecutor, "execute_overlay_rebalance"))
    )

    state_field_names = {f.name for f in dataclasses.fields(BotState)}
    req_fields = {"carry_orders", "overlay_orders", "overlay_positions", "executor_functional"}
    state_fields_present = req_fields.issubset(state_field_names)

    carry_routed = False
    overlay_routed = False
    err_msg: str | None = None

    def _execute_harness(folder: Path) -> tuple[bool, bool]:
        stub = StubExchange()
        state_file = folder / "state.json"
        store = StateStore(state_file)

        executor = CarryLiveExecutor(
            state_store=store,
            capital=3000.0,
            symbols=["BTC/USDT:USDT"],
            overlay_config=None,
            enable_lot_quantization=True,
            perp_exchange=stub,
            spot_exchange=stub,
            mode="paper",
        )

        c_res = executor.execute_maker_then_taker_pair(
            "BTC/USDT:USDT",
            "sell",
            0.002,
            60000.0,
            sleep_fn=lambda s: None,
        )

        ov_alloc = {
            "allocations": [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "long",
                    "qty": 0.001,
                    "notional": 60.0,
                }
            ],
            "overlay_active": True,
        }
        ov_universe = {
            "BTC/USDT:USDT": pd.DataFrame({"perp_close": [60000.0], "close": [60000.0]})
        }
        ov_res = executor.execute_overlay_rebalance(ov_alloc, ov_universe)

        c_ok = bool(
            len(getattr(executor.state, "carry_orders", [])) > 0
            or float(c_res.get("filled_qty", 0.0)) > 0.0
        )
        ov_ok = bool(
            len(getattr(executor.state, "overlay_orders", [])) > 0
            or int(ov_res.get("n_orders", 0)) > 0
        )
        return c_ok, ov_ok

    try:
        if tmp_path is not None:
            carry_routed, overlay_routed = _execute_harness(Path(tmp_path))
        else:
            with tempfile.TemporaryDirectory() as td:
                carry_routed, overlay_routed = _execute_harness(Path(td))
    except Exception as e:
        err_msg = str(e)
        carry_routed = False
        overlay_routed = False

    executor_functional = bool(
        methods_present
        and state_fields_present
        and carry_routed
        and overlay_routed
        and (err_msg is None)
    )

    result = {
        "methods_present": bool(methods_present),
        "state_fields_present": bool(state_fields_present),
        "carry_routed": bool(carry_routed),
        "overlay_routed": bool(overlay_routed),
        "executor_functional": bool(executor_functional),
    }
    if err_msg is not None:
        result["error"] = err_msg

    return result


# ---------------------------------------------------------------------------
# 6. evaluate_sprint11_go_no_go
# ---------------------------------------------------------------------------
def evaluate_sprint11_go_no_go(
    payload: dict[str, Any],
    executor_functional: bool | None = None,
) -> dict[str, Any]:
    """Phan quyet GO/NO-GO Sprint 11 theo docs/go_no_go_sprint11.md.

    Chon primary row (uu tien tier1b/smooth_conviction, fallback tier1a/smooth_conviction).
    Kiem tra day du cac tieu chi bat buoc, khong raise khi thieu key.
    """
    if not isinstance(payload, dict):
        payload = {}

    rows = payload.get("rows", [])
    if isinstance(rows, list) and len(rows) > 0:
        primary: dict[str, Any] | None = None
        for r in rows:
            if isinstance(r, dict) and r.get("combo") == "tier1b/smooth_conviction" and not r.get("error"):
                primary = r
                break
        if primary is None:
            for r in rows:
                if isinstance(r, dict) and r.get("combo") == "tier1a/smooth_conviction" and not r.get("error"):
                    primary = r
                    break
        if primary is None:
            for r in rows:
                if isinstance(r, dict) and not r.get("error"):
                    primary = r
                    break
        if primary is None:
            primary = rows[0] if isinstance(rows[0], dict) else {}
    else:
        primary = payload

    primary_combo = str(primary.get("combo", "tier1b/smooth_conviction"))

    ann_ret = float(primary.get("annualized_return_pct", 0.0))
    pos_periods = int(primary.get("positive_periods", 0))
    holdout_ret = float(primary.get("holdout_return_pct", 0.0))
    holdout_carry = float(primary.get("holdout_carry_pnl", 0.0))
    holdout_overlay = float(primary.get("holdout_overlay_net_pnl", 0.0))
    holdout_resim = float(primary.get("holdout_return_resim_pct", 0.0))
    holdout_ret_active = float(primary.get("holdout_return_on_active_capital_pct", 0.0))
    holdout_ret_compound = float(primary.get("holdout_return_on_compounded_equity_pct", 0.0))
    sharpe = float(primary.get("sharpe", 0.0))
    maxdd = float(abs(primary.get("maxdd_pct", primary.get("maxdd", 0.0))))
    beta_btc = float(primary.get("beta_btc", primary.get("beta", 0.0)))
    basis_std = float(primary.get("basis_mtm_std", 0.0))
    gap = float(primary.get("accounting_gap", 0.0))
    quant_events = int(primary.get("overlay_quantization_events", 0))

    if executor_functional is not None:
        exec_ok = bool(executor_functional)
    else:
        exec_ok = bool(primary.get("executor_functional", payload.get("executor_functional", False)))

    criteria = [
        {
            "name": "annualized_return_pct",
            "value": ann_ret,
            "threshold": f">= {SPRINT11_GO_CRITERIA['annualized_return_pct']}%",
            "passed": bool(ann_ret >= float(SPRINT11_GO_CRITERIA["annualized_return_pct"])),
            "note": f"AnnRet: {ann_ret:.2f}% (threshold >= {SPRINT11_GO_CRITERIA['annualized_return_pct']}%)",
        },
        {
            "name": "min_positive_periods",
            "value": pos_periods,
            "threshold": f">= {SPRINT11_GO_CRITERIA['min_positive_periods']}/7",
            "passed": bool(pos_periods >= int(SPRINT11_GO_CRITERIA["min_positive_periods"])),
            "note": f"Positive periods: {pos_periods}/7 (threshold >= {SPRINT11_GO_CRITERIA['min_positive_periods']})",
        },
        {
            "name": "holdout_consistency",
            "value": {
                "return_pct": holdout_ret,
                "carry_pnl": holdout_carry,
                "overlay_net_pnl": holdout_overlay,
            },
            "threshold": f"ret >= {SPRINT11_GO_CRITERIA['holdout_return_pct']}% & carry_pnl > 0 & overlay_net > 0",
            "passed": bool(
                holdout_ret >= float(SPRINT11_GO_CRITERIA["holdout_return_pct"])
                and holdout_carry > 0.0
                and holdout_overlay > 0.0
            ),
            "note": (
                f"Holdout: ret={holdout_ret:.2f}% (>= {SPRINT11_GO_CRITERIA['holdout_return_pct']}%), "
                f"carry_pnl={holdout_carry:.2f} (> 0), overlay_net_pnl={holdout_overlay:.2f} (> 0)"
            ),
        },
        {
            "name": "holdout_consistency_resim",
            "value": {
                "return_pct": holdout_resim,
                "carry_pnl": holdout_carry,
                "overlay_net_pnl": holdout_overlay,
            },
            "threshold": "ret >= 3.5% (phuong phap re-sim tu 3000, khoa Sprint 11)",
            "passed": bool(holdout_resim >= 3.5 and holdout_carry > 0.0 and holdout_overlay > 0.0),
            "note": (
                f"Re-sim: ret={holdout_resim:.2f}%"
                if (holdout_resim >= 3.5 and holdout_carry > 0.0 and holdout_overlay > 0.0)
                else f"Re-sim: ret={holdout_resim:.2f}% (< 3.5%)"
            ),
        },
        {
            "name": "sharpe",
            "value": sharpe,
            "threshold": f">= {SPRINT11_GO_CRITERIA['sharpe']}",
            "passed": bool(sharpe >= float(SPRINT11_GO_CRITERIA["sharpe"])),
            "note": f"Sharpe: {sharpe:.2f} (threshold >= {SPRINT11_GO_CRITERIA['sharpe']})",
        },
        {
            "name": "maxdd_pct",
            "value": maxdd,
            "threshold": f"<= {SPRINT11_GO_CRITERIA['maxdd_pct']}%",
            "passed": bool(maxdd <= float(SPRINT11_GO_CRITERIA["maxdd_pct"])),
            "note": f"MaxDD: {maxdd:.2f}% (threshold <= {SPRINT11_GO_CRITERIA['maxdd_pct']}%)",
        },
        {
            "name": "abs_beta_btc",
            "value": abs(beta_btc),
            "threshold": f"<= {SPRINT11_GO_CRITERIA['abs_beta_btc']}",
            "passed": bool(abs(beta_btc) <= float(SPRINT11_GO_CRITERIA["abs_beta_btc"])),
            "note": f"|Beta BTC|: {abs(beta_btc):.4f} (threshold <= {SPRINT11_GO_CRITERIA['abs_beta_btc']})",
        },
        {
            "name": "basis_mtm_std",
            "value": basis_std,
            "threshold": "> 0",
            "passed": bool(basis_std > 1e-12),
            "note": f"Basis MTM std: {basis_std:.6f} (threshold > 0)",
        },
        {
            "name": "accounting_gap",
            "value": gap,
            "threshold": "< 1e-6",
            "passed": bool(gap < 1e-6),
            "note": f"Accounting gap: {gap:.2e} (threshold < 1e-6)",
        },
        {
            "name": "overlay_quantization_events",
            "value": quant_events,
            "threshold": "> 0",
            "passed": bool(quant_events > 0),
            "note": f"Overlay quantization events: {quant_events} (threshold > 0)",
        },
        {
            "name": "executor_functional",
            "value": exec_ok,
            "threshold": "== True",
            "passed": bool(exec_ok),
            "note": f"Executor functional: {exec_ok} (threshold == True)",
        },
        {
            "name": "holdout_active_capital_info",
            "value": holdout_ret_active,
            "threshold": "> 0 (chi de bao cao)",
            "passed": True,
            "note": f"active={holdout_ret_active:.2f}%, compound={holdout_ret_compound:.2f}%",
        },
    ]

    reasons = [c["note"] for c in criteria if not c["passed"]]
    go = bool(len(reasons) == 0)

    return {
        "primary_combo": primary_combo,
        "go": go,
        "criteria": criteria,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 7. build_sprint11_summary_rows
# ---------------------------------------------------------------------------
def build_sprint11_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Tao danh sach cac dong phang phuc vu in markdown va xuat bao cao Sprint 11."""
    if not isinstance(payload, dict):
        return []

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append({
            "combo": str(r.get("combo", "")),
            "ann%": float(r.get("annualized_return_pct", 0.0)),
            "annualized_return_pct": float(r.get("annualized_return_pct", 0.0)),
            "sharpe": float(r.get("sharpe", 0.0)),
            "maxdd%": float(r.get("maxdd_pct", r.get("maxdd", 0.0))),
            "maxdd_pct": float(r.get("maxdd_pct", r.get("maxdd", 0.0))),
            "period+": int(r.get("positive_periods", 0)),
            "positive_periods": int(r.get("positive_periods", 0)),
            "holdout%": float(r.get("holdout_return_pct", 0.0)),
            "holdout_return_pct": float(r.get("holdout_return_pct", 0.0)),
            "holdout_active%": float(r.get("holdout_return_on_active_capital_pct", 0.0)),
            "holdout_compound%": float(r.get("holdout_return_on_compounded_equity_pct", 0.0)),
            "reinvest": bool(r.get("reinvest_profits", True)),
            "carry_pnl": float(r.get("carry_pnl", 0.0)),
            "overlay_net": float(r.get("overlay_net_pnl", 0.0)),
            "overlay_net_pnl": float(r.get("overlay_net_pnl", 0.0)),
            "basis_std": float(r.get("basis_mtm_std", 0.0)),
            "basis_mtm_std": float(r.get("basis_mtm_std", 0.0)),
            "beta": float(r.get("beta_btc", r.get("beta", 0.0))),
            "beta_btc": float(r.get("beta_btc", r.get("beta", 0.0))),
            "accounting_gap": float(r.get("accounting_gap", 0.0)),
            "executor": bool(r.get("executor_functional", False)),
            "executor_functional": bool(r.get("executor_functional", False)),
        })

    return out
