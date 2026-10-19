"""Bộ điều phối thực thi Paper/Live cho chiến lược Dynamic Carry và Hội tụ Basis Spot-Perp.

Sprint 8 - Chặng 4:
Module cung cấp:
- ReconcileAction: Dataclass biểu diễn kế hoạch điều chỉnh từng tài sản trong danh mục.
- CarryLiveExecutor: Lớp điều phối kiểm tra nến đóng 1h, đối soát vị thế Delta-Neutral,
  định giá MTM True Spot-Perp, kiểm tra an toàn ký quỹ và ghi nhận trạng thái nguyên tử.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import UNIVERSE_12_SYMBOLS, TradingConfig
from src.data_loader import DataLoader, drop_incomplete_last_candle, raw_ohlcv_to_dataframe
from src.dynamic_carry_basis import (
    DynamicCarryConfig,
    SelectionDecision,
    build_carry_basis_scores,
    select_portfolio,
)
from src.funding_loader import align_spot_perp_funding, fetch_funding_history, fetch_spot_ohlcv
from src.state_store import BotState, StateStore

logger = logging.getLogger("carry_live_executor")


@dataclass
class ReconcileAction:
    """Kế hoạch hành động cụ thể cho từng symbol tại chu kỳ đối soát."""

    symbol: str
    action: Literal["HOLD", "OPEN_DELTA_NEUTRAL", "CLOSE_DELTA_NEUTRAL", "REBALANCE_MARGIN", "SKIP"]
    reason: str
    score: float
    f_ewma: float
    basis_pct: float
    basis_dev: float
    target_notional_usdt: float
    spot_qty: float
    perp_qty: float
    margin_ratio: float
    margin_health: Literal["HEALTHY", "REBALANCE_NEEDED", "CRITICAL", "N/A"]


def _clean_symbol_name(symbol: str) -> str:
    """Chuyển symbol sang dạng tên file an toàn (ví dụ BTC/USDT:USDT -> BTC_USDT_USDT)."""
    return symbol.replace("/", "_").replace(":", "_")


def _spot_symbol_clean_name(spot_symbol: str) -> str:
    """Chuyển spot symbol sang dạng tên file SPOT (ví dụ BTC/USDT -> BTCUSDT)."""
    clean = (
        spot_symbol.replace("/", "")
        .replace(":", "")
        .replace("_", "")
        .upper()
    )
    if clean.endswith("USDTUSDT"):
        clean = clean[:-4]
    return clean


class CarryLiveExecutor:
    """Bộ điều phối vận hành thực tế / paper cho chiến lược Dynamic Carry xoay vòng 12 coin."""

    def __init__(
        self,
        config: TradingConfig = TradingConfig(timeframe="1h"),
        carry_config: DynamicCarryConfig = DynamicCarryConfig(),
        mode: Literal["paper", "live"] = "paper",
        symbols: Sequence[str] = UNIVERSE_12_SYMBOLS,
        state_store: Optional[StateStore] = None,
        perp_exchange: Any = None,
        spot_exchange: Any = None,
        raw_data_dir: Path | str = Path("data/raw"),
        lookback_hours: int = 720,
    ) -> None:
        if mode not in ("paper", "live"):
            raise ValueError(f"Mode khong hop le: {mode!r}, phai la 'paper' hoac 'live'.")

        self.config = config
        self.carry_config = carry_config
        self.mode: Literal["paper", "live"] = mode
        self.symbols: list[str] = list(symbols)
        self.raw_data_dir = Path(raw_data_dir)
        self.lookback_hours = int(lookback_hours)

        self.perp_exchange = perp_exchange
        self.spot_exchange = spot_exchange

        self.state_store: StateStore = (
            state_store if state_store is not None else StateStore(config.state_file_path)
        )
        self.state: BotState = self.state_store.load(default_equity=carry_config.initial_capital)
        if not isinstance(self.state.carry_positions, dict):
            self.state.carry_positions = {}

    def _load_symbol_from_cache(self, symbol: str) -> pd.DataFrame:
        """Đọc và ghép dữ liệu Spot, Perp và Funding từ Parquet cache cục bộ."""
        clean_sym = _clean_symbol_name(symbol)
        perp_candidates = [
            self.raw_data_dir / f"{clean_sym}_1h.parquet",
            self.raw_data_dir / f"{clean_sym}_{self.config.timeframe}.parquet",
            self.raw_data_dir / f"{symbol.replace('/', '').replace(':', '').upper()}_1h.parquet",
        ]
        perp_path = next((p for p in perp_candidates if p.exists()), None)
        if perp_path is None:
            raise FileNotFoundError(f"Missing perp cache for {symbol} in {self.raw_data_dir}")

        funding_candidates = [
            self.raw_data_dir / f"{clean_sym}_funding.parquet",
            self.raw_data_dir / f"{clean_sym}_funding_rate.parquet",
        ]
        funding_path = next((p for p in funding_candidates if p.exists()), None)
        if funding_path is None:
            raise FileNotFoundError(f"Missing funding cache for {symbol} in {self.raw_data_dir}")

        spot_sym = symbol.split(":")[0]
        spot_clean = _spot_symbol_clean_name(spot_sym)
        spot_candidates = [
            self.raw_data_dir / f"{spot_clean}_SPOT_1h.parquet",
            self.raw_data_dir / f"{spot_clean}_SPOT_{self.config.timeframe}.parquet",
            self.raw_data_dir / f"{spot_sym.replace('/', '').replace(':', '').upper()}_SPOT_1h.parquet",
            self.raw_data_dir / f"{clean_sym}_SPOT_1h.parquet",
            self.raw_data_dir / f"{spot_sym.replace('/', '_')}_SPOT_1h.parquet",
        ]
        spot_path = next((p for p in spot_candidates if p.exists()), None)
        if spot_path is None:
            raise FileNotFoundError(f"Missing spot cache for {symbol} in {self.raw_data_dir}")

        perp_df = pd.read_parquet(perp_path)
        funding_df = pd.read_parquet(funding_path)
        spot_df = pd.read_parquet(spot_path)

        for d in (perp_df, spot_df):
            if not isinstance(d.index, pd.DatetimeIndex):
                if "timestamp" in d.columns:
                    d.set_index("timestamp", inplace=True)
                elif "datetime" in d.columns:
                    d.set_index("datetime", inplace=True)
            idx = pd.DatetimeIndex(d.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            d.index = idx.as_unit("ns")

        # Áp dụng drop_incomplete_last_candle chống look-ahead
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        perp_df = drop_incomplete_last_candle(perp_df, self.config.timeframe, now_ms)
        spot_df = drop_incomplete_last_candle(spot_df, self.config.timeframe, now_ms)

        aligned = align_spot_perp_funding(perp_df, spot_df, funding_df)
        if self.lookback_hours > 0 and len(aligned) > self.lookback_hours:
            aligned = aligned.iloc[-self.lookback_hours:].copy()

        return aligned

    def _fetch_symbol_from_exchange(self, symbol: str) -> pd.DataFrame:
        """Tải dữ liệu mới nhất từ sàn Perp và Spot trực tiếp qua API."""
        server_now_ms = (
            int(self.perp_exchange.milliseconds())
            if hasattr(self.perp_exchange, "milliseconds") and callable(self.perp_exchange.milliseconds)
            else int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        )
        tf_ms = 3600 * 1000
        since_ms = server_now_ms - (self.lookback_hours + 48) * tf_ms
        limit = min(1000, self.lookback_hours + 48)

        perp_rows = self.perp_exchange.fetch_ohlcv(symbol, timeframe="1h", since=since_ms, limit=limit)
        perp_df = raw_ohlcv_to_dataframe(perp_rows, symbol, "1h")
        perp_df = drop_incomplete_last_candle(perp_df, "1h", server_now_ms)

        spot_sym = symbol.split(":")[0]
        spot_rows = self.spot_exchange.fetch_ohlcv(spot_sym, timeframe="1h", since=since_ms, limit=limit)
        spot_df = raw_ohlcv_to_dataframe(spot_rows, spot_sym, "1h")
        spot_df = drop_incomplete_last_candle(spot_df, "1h", server_now_ms)

        try:
            start_date = pd.Timestamp(since_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            funding_df = fetch_funding_history(
                symbol,
                start_date=start_date,
                data_dir=str(self.raw_data_dir),
                exchange=self.perp_exchange,
            )
        except Exception:
            clean_sym = _clean_symbol_name(symbol)
            f_path = self.raw_data_dir / f"{clean_sym}_funding.parquet"
            if f_path.exists():
                funding_df = pd.read_parquet(f_path)
            else:
                raise

        aligned = align_spot_perp_funding(perp_df, spot_df, funding_df)
        if self.lookback_hours > 0 and len(aligned) > self.lookback_hours:
            aligned = aligned.iloc[-self.lookback_hours:].copy()

        return aligned

    def fetch_latest_aligned_universe(self) -> dict[str, pd.DataFrame]:
        """Tải và đồng bộ dữ liệu cho toàn bộ rổ tài sản từ sàn hoặc cache Parquet."""
        universe: dict[str, pd.DataFrame] = {}
        has_exchanges = self.perp_exchange is not None and self.spot_exchange is not None

        for sym in self.symbols:
            if has_exchanges:
                try:
                    aligned = self._fetch_symbol_from_exchange(sym)
                    universe[sym] = aligned
                    continue
                except Exception as e:
                    logger.warning(
                        "Error fetching live data for %s: %s. Falling back to parquet cache.",
                        sym,
                        str(e),
                    )

            try:
                aligned = self._load_symbol_from_cache(sym)
                universe[sym] = aligned
            except FileNotFoundError:
                if not has_exchanges:
                    raise FileNotFoundError(
                        f"Cache file missing for {sym} in {self.raw_data_dir} and no live exchange available."
                    )
                else:
                    raise FileNotFoundError(
                        f"Failed live fetch and missing cache file for {sym} in {self.raw_data_dir}."
                    )

        return universe

    def check_margin_health(
        self,
        pos_dict: Optional[dict[str, Any]],
        spot_price: float,
        perp_price: float,
    ) -> tuple[float, Literal["HEALTHY", "REBALANCE_NEEDED", "CRITICAL", "N/A"]]:
        """Kiểm tra tỷ lệ ký quỹ và trạng thái an toàn của vị thế Delta-Neutral spot_perp_3x.

        Tham số:
            pos_dict: Dict vị thế lưu trong state.
            spot_price: Giá Spot hiện hành.
            perp_price: Giá Perp hiện hành.

        Trả về:
            Tuple (margin_ratio, health_status).
        """
        if not pos_dict or not isinstance(pos_dict, dict):
            return 0.0, "N/A"

        qty = float(pos_dict.get("qty", 0.0))
        if qty <= 0.0 or not np.isfinite(qty):
            return 0.0, "N/A"

        entry_spot = float(pos_dict.get("entry_spot_price", 0.0))
        entry_perp = float(pos_dict.get("entry_perp_price", 0.0))
        margin_allocated = float(pos_dict.get("margin_allocated_usdt", 0.0))

        if (
            spot_price <= 0.0
            or perp_price <= 0.0
            or entry_spot <= 0.0
            or entry_perp <= 0.0
            or not np.isfinite(spot_price)
            or not np.isfinite(perp_price)
        ):
            return 0.0, "N/A"

        unrealized_mtm = qty * (spot_price - entry_spot) - qty * (perp_price - entry_perp)
        effective_margin = margin_allocated + unrealized_mtm
        perp_notional = max(qty * perp_price, 1e-12)
        ratio = effective_margin / perp_notional

        if ratio < self.carry_config.maintenance_margin_ratio:
            return float(ratio), "CRITICAL"

        price_dev = abs(spot_price - entry_spot) / max(entry_spot, 1e-12)
        if (
            ratio < self.carry_config.margin_rebalance_lower_ratio
            or ratio > self.carry_config.margin_rebalance_upper_ratio
            or price_dev >= self.carry_config.rebalance_price_trigger_pct
        ):
            return float(ratio), "REBALANCE_NEEDED"

        return float(ratio), "HEALTHY"

    def reconcile_once(
        self,
        aligned_universe: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict[str, Any]:
        """Thực hiện một chu kỳ đối soát hoàn chỉnh: tính điểm, kiểm tra ký quỹ, cân bằng danh mục và lưu state."""
        # 1. Lấy dữ liệu rổ tài sản
        if aligned_universe is None:
            aligned_universe = self.fetch_latest_aligned_universe()

        if not aligned_universe:
            return {
                "timestamp": "N/A",
                "equity": float(self.state.current_equity),
                "actions": [],
                "active_positions": {},
                "margin_health_summary": "Empty universe",
                "table_ascii": "",
            }

        # Xác định mốc thời gian nến chốt mới nhất
        all_last_ts = [df.index[-1] for df in aligned_universe.values() if len(df) > 0]
        latest_ts = max(all_last_ts) if all_last_ts else pd.Timestamp.now(tz="UTC")
        timestamp_str = latest_ts.isoformat()

        # 2. Tính toán điểm Carry + Basis chống look-ahead
        scores_dict = build_carry_basis_scores(aligned_universe, self.carry_config)

        scores_at_t: dict[str, float] = {}
        metrics_at_t: dict[str, dict[str, Any]] = {}

        for sym in self.symbols:
            sdf = scores_dict.get(sym)
            udf = aligned_universe.get(sym)
            if sdf is not None and not sdf.empty and udf is not None and not udf.empty:
                sc = sdf["score"].iloc[-1]
                f_ewma = sdf["f_ewma"].iloc[-1]
                b_dev = sdf["basis_dev"].iloc[-1]

                if "basis_pct" in udf.columns:
                    b_pct = udf["basis_pct"].iloc[-1]
                elif "perp_close" in udf.columns and "spot_close" in udf.columns:
                    sp = udf["spot_close"].iloc[-1]
                    pe = udf["perp_close"].iloc[-1]
                    b_pct = (pe - sp) / sp if sp > 0 else 0.0
                else:
                    b_pct = 0.0

                sc_clean = float(sc) if pd.notna(sc) and np.isfinite(sc) else 0.0
                f_clean = float(f_ewma) if pd.notna(f_ewma) and np.isfinite(f_ewma) else 0.0
                bdev_clean = float(b_dev) if pd.notna(b_dev) and np.isfinite(b_dev) else 0.0
                bpct_clean = float(b_pct) if pd.notna(b_pct) and np.isfinite(b_pct) else 0.0
                is_valid = bool(pd.notna(sc) and np.isfinite(sc))

                scores_at_t[sym] = sc_clean
                metrics_at_t[sym] = {
                    "score": sc_clean,
                    "f_ewma": f_clean,
                    "basis_dev": bdev_clean,
                    "basis_pct": bpct_clean,
                    "valid": is_valid,
                }
            else:
                scores_at_t[sym] = 0.0
                metrics_at_t[sym] = {
                    "score": 0.0,
                    "f_ewma": 0.0,
                    "basis_dev": 0.0,
                    "basis_pct": 0.0,
                    "valid": False,
                }

        # 3. MTM + An toàn ký quỹ cho các vị thế đang nắm giữ
        is_new_funding_bar = (
            latest_ts.hour in (0, 8, 16)
            and timestamp_str != self.state.carry_last_reconcile_ts
        )

        margin_health_map: dict[str, tuple[float, str]] = {}
        for sym, pos in list(self.state.carry_positions.items()):
            if is_new_funding_bar:
                pos["periods_held"] = int(pos.get("periods_held", 0)) + 1
                udf = aligned_universe.get(sym)
                if udf is not None and not udf.empty and "funding_rate_event" in udf.columns:
                    fr_evt = float(udf["funding_rate_event"].iloc[-1])
                    qty = float(pos.get("qty", 0.0))
                    sp_curr = float(udf["spot_close"].iloc[-1] if "spot_close" in udf.columns else udf["close"].iloc[-1])
                    notional = qty * sp_curr
                    funding_amount = notional * fr_evt
                    pos["funding_collected_usdt"] = float(pos.get("funding_collected_usdt", 0.0)) + funding_amount

            udf = aligned_universe.get(sym)
            if udf is not None and not udf.empty:
                sp_curr = float(udf["spot_close"].iloc[-1] if "spot_close" in udf.columns else udf["close"].iloc[-1])
                pe_curr = float(udf["perp_close"].iloc[-1] if "perp_close" in udf.columns else udf["close"].iloc[-1])
                ratio, health = self.check_margin_health(pos, sp_curr, pe_curr)
                margin_health_map[sym] = (ratio, health)
            else:
                margin_health_map[sym] = (0.0, "N/A")

        # 4. Lựa chọn danh mục qua select_portfolio
        current_holdings = set(self.state.carry_positions.keys())
        periods_held_map = {sym: int(pos.get("periods_held", 0)) for sym, pos in self.state.carry_positions.items()}
        decision = select_portfolio(scores_at_t, current_holdings, periods_held_map, self.carry_config)

        resolved_actions: dict[str, Literal["HOLD", "OPEN_DELTA_NEUTRAL", "CLOSE_DELTA_NEUTRAL", "REBALANCE_MARGIN", "SKIP"]] = {}
        resolved_reasons: dict[str, str] = {}

        for sym in self.symbols:
            if not metrics_at_t[sym]["valid"] and sym not in current_holdings:
                resolved_actions[sym] = "SKIP"
                resolved_reasons[sym] = "MISSING_DATA_OR_NAN"
                continue

            raw_act = decision.actions.get(sym, "SKIP")
            raw_rs = decision.reasons.get(sym, "SKIP")

            if raw_act == "OPEN":
                resolved_actions[sym] = "OPEN_DELTA_NEUTRAL"
                resolved_reasons[sym] = raw_rs
            elif raw_act == "CLOSE":
                resolved_actions[sym] = "CLOSE_DELTA_NEUTRAL"
                resolved_reasons[sym] = raw_rs
            elif raw_act == "HOLD":
                ratio, health = margin_health_map.get(sym, (1.0 / 3.0, "HEALTHY"))
                if health in ("REBALANCE_NEEDED", "CRITICAL"):
                    resolved_actions[sym] = "REBALANCE_MARGIN"
                    resolved_reasons[sym] = f"HOLD_MARGIN_{health}"
                else:
                    resolved_actions[sym] = "HOLD"
                    resolved_reasons[sym] = raw_rs
            else:
                resolved_actions[sym] = "SKIP"
                resolved_reasons[sym] = raw_rs

        # 5. Thực thi và Cập nhật trạng thái
        # a) Đóng vị thế CLOSE_DELTA_NEUTRAL trước
        for sym in self.symbols:
            if resolved_actions.get(sym) == "CLOSE_DELTA_NEUTRAL" and sym in self.state.carry_positions:
                pos = self.state.carry_positions[sym]
                udf = aligned_universe.get(sym)
                sp_curr = float(udf["spot_close"].iloc[-1] if "spot_close" in udf.columns else udf["close"].iloc[-1])
                pe_curr = float(udf["perp_close"].iloc[-1] if "perp_close" in udf.columns else udf["close"].iloc[-1])
                qty = float(pos.get("qty", 0.0))

                if self.mode == "live":
                    if (
                        not self.spot_exchange
                        or not hasattr(self.spot_exchange, "create_order")
                        or not self.perp_exchange
                        or not hasattr(self.perp_exchange, "create_order")
                    ):
                        raise RuntimeError("Live exchange does not support create_order for carry close")
                    spot_sym = pos.get("spot_symbol", sym.split(":")[0])
                    self.spot_exchange.create_order(spot_sym, "market", "sell", qty)
                    self.perp_exchange.create_order(sym, "market", "buy", qty, params={"reduceOnly": True})

                entry_spot = float(pos.get("entry_spot_price", sp_curr))
                entry_perp = float(pos.get("entry_perp_price", pe_curr))
                mtm_pnl = qty * (sp_curr - entry_spot) - qty * (pe_curr - entry_perp)
                close_fee = (qty * sp_curr) * self.carry_config.one_way_cost()
                funding_collected = float(pos.get("funding_collected_usdt", 0.0))
                self.state.current_equity += mtm_pnl + funding_collected - close_fee
                del self.state.carry_positions[sym]

        # b) Cân bằng ký quỹ REBALANCE_MARGIN
        for sym in self.symbols:
            if resolved_actions.get(sym) == "REBALANCE_MARGIN" and sym in self.state.carry_positions:
                pos = self.state.carry_positions[sym]
                udf = aligned_universe.get(sym)
                sp_curr = float(udf["spot_close"].iloc[-1] if "spot_close" in udf.columns else udf["close"].iloc[-1])
                pe_curr = float(udf["perp_close"].iloc[-1] if "perp_close" in udf.columns else udf["close"].iloc[-1])
                qty = float(pos.get("qty", 0.0))
                entry_spot = float(pos.get("entry_spot_price", sp_curr))
                entry_perp = float(pos.get("entry_perp_price", pe_curr))

                unrealized_mtm = qty * (sp_curr - entry_spot) - qty * (pe_curr - entry_perp)
                effective_margin = float(pos.get("margin_allocated_usdt", 0.0)) + unrealized_mtm
                v_i = qty * sp_curr + effective_margin

                if self.mode == "live":
                    if (
                        not self.spot_exchange
                        or not hasattr(self.spot_exchange, "create_order")
                        or not self.perp_exchange
                        or not hasattr(self.perp_exchange, "create_order")
                    ):
                        raise RuntimeError("Live exchange does not support create_order for carry rebalance")

                if v_i > 0:
                    new_notional = 0.75 * v_i
                    new_qty = new_notional / sp_curr
                    delta_notional = abs(new_qty - qty) * sp_curr
                    reb_fee = (
                        self.carry_config.rebalance_fee_multiplier
                        * self.carry_config.one_way_cost()
                        * delta_notional
                    )
                    pos["qty"] = float(new_qty)
                    pos["entry_spot_price"] = float(sp_curr)
                    pos["entry_perp_price"] = float(pe_curr)
                    pos["notional_usdt"] = float(new_notional)
                    pos["margin_allocated_usdt"] = float(0.25 * v_i - reb_fee)
                    self.state.current_equity -= reb_fee
                else:
                    del self.state.carry_positions[sym]

        # c) Mở vị thế OPEN_DELTA_NEUTRAL
        for sym in self.symbols:
            if resolved_actions.get(sym) == "OPEN_DELTA_NEUTRAL":
                udf = aligned_universe.get(sym)
                sp_curr = float(udf["spot_close"].iloc[-1] if "spot_close" in udf.columns else udf["close"].iloc[-1])
                pe_curr = float(udf["perp_close"].iloc[-1] if "perp_close" in udf.columns else udf["close"].iloc[-1])

                allocated_capital = (1.0 / float(self.carry_config.top_k)) * self.state.current_equity
                target_notional = allocated_capital * self.carry_config.notional_eta()
                spot_qty = target_notional / sp_curr if sp_curr > 0 else 0.0
                perp_qty = spot_qty
                fee = target_notional * self.carry_config.one_way_cost()
                margin_alloc = (allocated_capital - target_notional) - fee

                if self.mode == "live":
                    if (
                        not self.spot_exchange
                        or not hasattr(self.spot_exchange, "create_order")
                        or not self.perp_exchange
                        or not hasattr(self.perp_exchange, "create_order")
                    ):
                        raise RuntimeError("Live exchange does not support create_order for carry open")
                    spot_sym = sym.split(":")[0]
                    self.spot_exchange.create_order(spot_sym, "market", "buy", spot_qty)
                    self.perp_exchange.create_order(sym, "market", "sell", perp_qty)

                self.state.carry_positions[sym] = {
                    "symbol": sym,
                    "spot_symbol": sym.split(":")[0],
                    "qty": float(spot_qty),
                    "entry_spot_price": float(sp_curr),
                    "entry_perp_price": float(pe_curr),
                    "notional_usdt": float(target_notional),
                    "margin_allocated_usdt": float(margin_alloc),
                    "entry_time": timestamp_str,
                    "periods_held": 0,
                    "funding_collected_usdt": 0.0,
                    "score_at_entry": float(metrics_at_t[sym]["score"]),
                }
                self.state.current_equity -= fee

        self.state.carry_last_reconcile_ts = timestamp_str
        self.state_store.save(self.state)

        # 6. Dựng danh sách ReconcileAction
        actions_list: list[ReconcileAction] = []
        for sym in self.symbols:
            act = resolved_actions.get(sym, "SKIP")
            rs = resolved_reasons.get(sym, "SKIP")
            m = metrics_at_t[sym]

            if act == "OPEN_DELTA_NEUTRAL":
                pos = self.state.carry_positions.get(sym, {})
                t_notional = float(pos.get("notional_usdt", 0.0))
                s_qty = float(pos.get("qty", 0.0))
                p_qty = s_qty
                m_ratio = float(self.carry_config.margin_rebalance_target_ratio)
                m_health: Literal["HEALTHY", "REBALANCE_NEEDED", "CRITICAL", "N/A"] = "HEALTHY"
            elif act in ("HOLD", "REBALANCE_MARGIN"):
                pos = self.state.carry_positions.get(sym, {})
                t_notional = float(pos.get("notional_usdt", 0.0))
                s_qty = float(pos.get("qty", 0.0))
                p_qty = s_qty
                r, h = margin_health_map.get(sym, (float(self.carry_config.margin_rebalance_target_ratio), "HEALTHY"))
                m_ratio = float(r)
                m_health = h if h in ("HEALTHY", "REBALANCE_NEEDED", "CRITICAL") else "HEALTHY"
            elif act == "CLOSE_DELTA_NEUTRAL":
                t_notional = 0.0
                s_qty = 0.0
                p_qty = 0.0
                r, h = margin_health_map.get(sym, (0.0, "N/A"))
                m_ratio = float(r)
                m_health = h if h in ("HEALTHY", "REBALANCE_NEEDED", "CRITICAL", "N/A") else "N/A"
            else:
                t_notional = 0.0
                s_qty = 0.0
                p_qty = 0.0
                m_ratio = 0.0
                m_health = "N/A"

            actions_list.append(
                ReconcileAction(
                    symbol=sym,
                    action=act,
                    reason=rs,
                    score=float(m["score"]),
                    f_ewma=float(m["f_ewma"]),
                    basis_pct=float(m["basis_pct"]),
                    basis_dev=float(m["basis_dev"]),
                    target_notional_usdt=float(t_notional),
                    spot_qty=float(s_qty),
                    perp_qty=float(p_qty),
                    margin_ratio=float(m_ratio),
                    margin_health=m_health,
                )
            )

        n_active = len(self.state.carry_positions)
        n_healthy = sum(
            1 for a in actions_list if a.symbol in self.state.carry_positions and a.margin_health == "HEALTHY"
        )
        n_rebal = sum(
            1 for a in actions_list if a.symbol in self.state.carry_positions and a.margin_health == "REBALANCE_NEEDED"
        )
        n_crit = sum(
            1 for a in actions_list if a.symbol in self.state.carry_positions and a.margin_health == "CRITICAL"
        )
        summary_str = (
            f"Active: {n_active} | Healthy: {n_healthy} | Rebalance: {n_rebal} | Critical: {n_crit}"
        )

        plan = {
            "timestamp": timestamp_str,
            "equity": float(self.state.current_equity),
            "actions": actions_list,
            "active_positions": dict(self.state.carry_positions),
            "margin_health_summary": summary_str,
            "table_ascii": "",
        }
        plan["table_ascii"] = self.format_status_table(plan)
        return plan

    def format_status_table(self, plan: dict[str, Any]) -> str:
        """Định dạng bảng trạng thái đối soát thuần ASCII (mọi ord(ch) < 128, không emoji)."""
        actions: list[ReconcileAction] = plan.get("actions", [])
        timestamp = plan.get("timestamp", "N/A")
        equity = plan.get("equity", 0.0)

        lines: list[str] = []
        lines.append(f"=== CARRY RECONCILE STATUS [{timestamp}] | Equity: {equity:.2f} USDT ===")
        header = (
            f"| {'Symbol':<15} | {'Score(bps)':>10} | {'FundEWMA(bps)':>13} | {'Basis(%)':>9} "
            f"| {'BasisDev(bps)':>13} | {'Action':<20} | {'Notional($)':>11} | {'MarginRatio':>11} "
            f"| {'Health':<16} | {'Reason':<25} |"
        )
        sep = (
            "+"
            + "-" * 17
            + "+"
            + "-" * 12
            + "+"
            + "-" * 15
            + "+"
            + "-" * 11
            + "+"
            + "-" * 15
            + "+"
            + "-" * 22
            + "+"
            + "-" * 13
            + "+"
            + "-" * 13
            + "+"
            + "-" * 18
            + "+"
            + "-" * 27
            + "+"
        )

        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for a in actions:
            score_bps_str = f"{a.score * 10000:+.2f}" if np.isfinite(a.score) else "N/A"
            fund_bps_str = f"{a.f_ewma * 10000:+.2f}" if np.isfinite(a.f_ewma) else "N/A"
            basis_pct_str = f"{a.basis_pct * 100:+.2f}%" if np.isfinite(a.basis_pct) else "N/A"
            dev_bps_str = f"{a.basis_dev * 10000:+.2f}" if np.isfinite(a.basis_dev) else "N/A"
            ratio_str = (
                f"{a.margin_ratio * 100:.1f}%"
                if a.margin_health != "N/A" and np.isfinite(a.margin_ratio)
                else "N/A"
            )
            notional_str = f"{a.target_notional_usdt:.2f}"

            row = (
                f"| {a.symbol:<15} | {score_bps_str:>10} | {fund_bps_str:>13} | {basis_pct_str:>9} "
                f"| {dev_bps_str:>13} | {a.action:<20} | {notional_str:>11} | {ratio_str:>11} "
                f"| {a.margin_health:<16} | {a.reason:<25} |"
            )
            lines.append(row)

        lines.append(sep)
        return "\n".join(lines)

    def run_loop(self, once: bool = False, max_iterations: Optional[int] = None) -> int:
        """Vận hành vòng lặp điều phối CarryLiveExecutor (24/7 hoặc chạy một lần --once)."""
        logger.info("Starting CarryLiveExecutor run_loop (mode=%s, once=%s)", self.mode, once)
        iterations = 0
        try:
            while True:
                iterations += 1
                try:
                    plan = self.reconcile_once()
                    logger.info("\n%s", plan["table_ascii"])
                except Exception as e:
                    logger.error("Error in reconcile_once: %s", e, exc_info=True)
                    if once:
                        return 1

                if once:
                    return 0

                if max_iterations is not None and iterations >= max_iterations:
                    logger.info("Reached max_iterations (%d), exiting loop.", max_iterations)
                    return 0

                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("CarryLiveExecutor interrupted by user.")
            return 0
        except Exception as e:
            logger.critical("Fatal error in run_loop: %s", e, exc_info=True)
            return 1
