"""CLI khởi chạy Paper/Live Executor (24/7 hoặc --once)."""
import argparse
import dataclasses
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from run_backtest import generate_synthetic_ohlcv
from src.config import TradingConfig
from src.data_loader import DataLoader, _index_to_ms, timeframe_to_ms
from src.live_executor import LiveExecutor
from src.ml_filter import MetaLabelFilter
from src.notifier import TelegramNotifier
from src.risk_manager import RiskManager
from src.state_store import StateStore
from src.strategy import Strategy


class _OfflinePaperExchange:
    """Exchange giả lập phục vụ chạy --mode paper --once khi offline hoặc trong pytest."""

    def __init__(self, symbol: str = "BTC/USDT:USDT", timeframe: str = "15m") -> None:
        df = generate_synthetic_ohlcv(
            n_candles=320, symbol=symbol, timeframe=timeframe, seed=42
        )
        ts_ms = _index_to_ms(df.index)
        self.rows: list[list[float | int]] = [
            [
                int(ts_ms[i]),
                float(df["open"].iloc[i]),
                float(df["high"].iloc[i]),
                float(df["low"].iloc[i]),
                float(df["close"].iloc[i]),
                float(df["volume"].iloc[i]),
            ]
            for i in range(len(df))
        ]
        self.now_ms: int = int(ts_ms[-1]) + timeframe_to_ms(timeframe)

    def milliseconds(self) -> int:
        return self.now_ms

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> list[list[float | int]]:
        out = [r for r in self.rows if since is None or int(r[0]) >= int(since)]
        if limit is not None:
            out = out[: int(limit)]
        return [list(r) for r in out]


logger = logging.getLogger("run_live")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vận hành Paper/Live Quant Trading Bot (Binance USDT-M Futures)"
    )
    parser.add_argument(
        "--mode",
        choices=("paper", "live", "testnet", "shadow-live"),
        default="paper",
        help="Chế độ thực thi: paper, live, testnet hoặc shadow-live (read-only public feed, không đặt lệnh thật)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Chạy đúng 1 chu kỳ kiểm tra nến đóng rồi thoát",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Xoa trang thai cu va khoi tao lai von theo --capital",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="So chu ky daemon can chay truoc khi thoat",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=60.0,
        help="So giay giua 2 chu ky",
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT:USDT",
        help="Cặp giao dịch (mặc định: BTC/USDT:USDT)",
    )
    parser.add_argument(
        "--timeframe",
        default="15m",
        help="Khung thời gian nến (mặc định: 15m)",
    )
    parser.add_argument(
        "--strategy",
        choices=("directional", "dynamic_carry", "barbell"),
        default="directional",
        help="Chiến lược vận hành: directional (mặc định), dynamic_carry, hoặc barbell (Sprint 9)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=3000.0,
        help="Mức vốn vận hành cho chiến lược carry (mặc định: 3000.0 USDT)",
    )
    parser.add_argument(
        "--execution-style",
        choices=("maker_taker_hedge", "real_taker"),
        default="maker_taker_hedge",
        help="Cơ chế khớp lệnh: maker_taker_hedge (mặc định Sprint 9) hoặc real_taker",
    )
    parser.add_argument(
        "--strategy-mode",
        choices=("breakout", "trend_pullback", "mean_reversion", "regime_adaptive"),
        default="breakout",
        help="Chế độ chiến lược (mặc định: breakout)",
    )
    parser.add_argument(
        "--entry-mode",
        choices=("market", "limit_pullback"),
        default="market",
        help="Chế độ vào lệnh: market hoặc limit_pullback (mặc định: market)",
    )
    parser.add_argument(
        "--use-trailing",
        action="store_true",
        help="Bật cơ chế ATR Trailing Stop",
    )
    parser.add_argument(
        "--fee-aware",
        action="store_true",
        help="Bật cơ chế định cỡ vị thế tính phí (Fee-Aware Position Sizing)",
    )
    parser.add_argument(
        "--use-ml",
        action="store_true",
        help="Bật bộ lọc xác suất ML Meta-Labeling",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Đường dẫn file lưu trạng thái JSON (tùy chọn)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Dùng nguồn nến giả lập offline (không gọi mạng)",
    )
    parser.add_argument(
        "--shadow-timeout-sec",
        type=float,
        default=5.0,
        help="Timeout (giây) cho mỗi request public feed ở chế độ shadow-live (mặc định 5.0)",
    )
    parser.add_argument(
        "--offline-fixture",
        default=None,
        help="Đường dẫn JSON fixture cho shadow-live (không gọi mạng, dùng để test/CI)",
    )
    return parser.parse_args(argv)


def _load_shadow_fixture(fixture_path: Any) -> Any:
    """Nạp JSON fixture cho shadow-live; trả None nếu không có đường dẫn."""
    if not fixture_path:
        return None
    import json as _json

    path = Path(fixture_path)
    with open(path, "r", encoding="utf-8") as f:
        return _json.load(f)


def _run_shadow_live_preflight(args: argparse.Namespace, symbols: list[str]) -> bool:
    """Chốt an toàn trước khi chạy shadow-live: chỉ đọc public feed, fail-closed.

    Trả False (=> dừng, exit != 0) khi feed không healthy hoặc fixture lỗi.
    """
    from src.binance_public_feed import fetch_binance_shadow_snapshot

    try:
        fixture_data = _load_shadow_fixture(getattr(args, "offline_fixture", None))
    except Exception as e:
        logger.error("Shadow-live: cannot load offline fixture: %s", e)
        return False

    snapshot = fetch_binance_shadow_snapshot(
        symbols,
        timeout_sec=float(getattr(args, "shadow_timeout_sec", 5.0)),
        fixture_data=fixture_data,
    )
    logger.info(
        "Shadow-live snapshot: source=%s | healthy=%s | symbols=%d | errors=%d",
        snapshot.get("source"),
        snapshot.get("healthy"),
        len(snapshot.get("symbols", {})),
        len(snapshot.get("errors", [])),
    )
    for err in list(snapshot.get("errors", []))[:5]:
        logger.warning("Shadow-live feed issue: %s", err)
    if not snapshot.get("healthy", False):
        logger.error("Shadow-live: public feed unhealthy -> fail-closed, no cycles executed.")
        return False
    logger.info("Shadow-live: read-only feed OK (no orders can be placed in this mode).")
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    in_pytest = "PYTEST_CURRENT_TEST" in os.environ

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    has_api_keys = bool(api_key and api_secret)

    exec_mode = args.mode
    if args.mode == "live":
        if not has_api_keys:
            logger.error("Live mode: thieu BINANCE_API_KEY hoac BINANCE_API_SECRET trong env.")
            return 2
    elif args.mode == "testnet":
        if not has_api_keys:
            logger.warning("Testnet: thieu API key, chay Public Live Feed Paper Mode")
            exec_mode = "paper"
    elif args.mode == "shadow-live":
        # Shadow-live: chỉ đọc public feed + mô phỏng paper, KHÔNG dùng API key, KHÔNG đặt lệnh thật.
        logger.info("Shadow-live: read-only public feed + paper executor (no real orders).")
        exec_mode = "paper"

    if args.state_file is not None:
        state_path = Path(args.state_file)
    elif in_pytest:
        state_path = Path(tempfile.gettempdir()) / "quant_trading_pytest_bot_state.json"
    else:
        state_path = TradingConfig().state_file_path

    strat = getattr(args, "strategy", "directional")
    if strat in ("dynamic_carry", "barbell"):
        logger.info("=" * 60)
        if strat == "barbell":
            logger.info("QUANT TRADING BOT STARTING (BARBELL CARRY SLEEVE)")
            logger.info("Mode: %s (exec_mode=%s) | Strategy: barbell (paper cross-exchange carry sleeve)", args.mode, exec_mode)
        else:
            logger.info("QUANT TRADING BOT STARTING (DYNAMIC CARRY)")
            logger.info("Mode: %s (exec_mode=%s) | Strategy: %s", args.mode, exec_mode, strat)
        logger.info(
            "Capital: %.2f USDT | Execution Style: %s",
            args.capital,
            args.execution_style,
        )
        logger.info("State file: %s", state_path)
        logger.info("=" * 60)

        from src.carry_live_executor import CarryLiveExecutor
        from src.dynamic_carry_basis import DynamicCarryConfig

        perp_exchange = None
        spot_exchange = None
        if not (args.synthetic or in_pytest):
            try:
                import ccxt

                ex_cfg: dict[str, Any] = {"enableRateLimit": True}
                if has_api_keys and args.mode in ("live", "testnet"):
                    ex_cfg["apiKey"] = api_key
                    ex_cfg["secret"] = api_secret

                perp_exchange = ccxt.binanceusdm(dict(ex_cfg))
                spot_exchange = ccxt.binance(dict(ex_cfg))

                if args.mode == "testnet" and has_api_keys:
                    for ex in (perp_exchange, spot_exchange):
                        if hasattr(ex, "set_sandbox_mode") and callable(ex.set_sandbox_mode):
                            try:
                                ex.set_sandbox_mode(True)
                            except Exception as ex_err:
                                logger.warning("Could not set sandbox mode: %s", ex_err)
            except Exception as e:
                logger.warning(
                    "Could not initialize live exchanges (%s), falling back to offline cache",
                    e,
                )
                perp_exchange = None
                spot_exchange = None

        carry_config = DynamicCarryConfig(initial_capital=float(args.capital))
        trading_config = dataclasses.replace(
            TradingConfig(),
            timeframe=args.timeframe,
            state_file_path=state_path,
        )

        if strat == "barbell":
            try:
                from src.cross_exchange_carry import CORE_4_SYMBOLS, LIQUID_8_SYMBOLS
            except (ImportError, AttributeError):
                CORE_4_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]
                LIQUID_8_SYMBOLS = list(CORE_4_SYMBOLS) + [
                    "DOGE/USDT:USDT",
                    "XRP/USDT:USDT",
                    "AVAX/USDT:USDT",
                    "LINK/USDT:USDT",
                ]
            from src.barbell_portfolio import BarbellConfig

            venues: dict[str, Any] = {
                "binance": perp_exchange,
                "binance_spot": spot_exchange,
            }
            if not (args.synthetic or in_pytest):
                try:
                    import ccxt

                    bybit_cfg: dict[str, Any] = {"enableRateLimit": True}
                    bybit_key = os.environ.get("BYBIT_API_KEY")
                    bybit_secret = os.environ.get("BYBIT_API_SECRET")
                    if bybit_key and bybit_secret and args.mode in ("live", "testnet"):
                        bybit_cfg["apiKey"] = bybit_key
                        bybit_cfg["secret"] = bybit_secret
                    bybit_ex = ccxt.bybit(bybit_cfg)
                    if args.mode == "testnet" and hasattr(bybit_ex, "set_sandbox_mode") and callable(bybit_ex.set_sandbox_mode):
                        try:
                            bybit_ex.set_sandbox_mode(True)
                        except Exception as ex_err:
                            logger.warning("Could not set sandbox mode on bybit: %s", ex_err)
                    venues["bybit"] = bybit_ex
                except Exception as e:
                    logger.warning("Could not initialize Bybit exchange for venues: %s", e)

            overlay_cfg = BarbellConfig(
                initial_capital=float(args.capital),
                carry_variant="tier1b",
                gating_mode="smooth_conviction",
            )
            # Sprint 15: carry gate phai dung dung Frozen Alpha Tier 1B (Sprint 11),
            # khong dung defaults Sprint 7 (entry 1.0 bps, score co basis blend).
            carry_config = DynamicCarryConfig(
                initial_capital=float(args.capital),
                ema_span=42,
                top_k=4,
                entry_threshold=0.000045,
                exit_threshold=0.0,
                min_hold_periods=63,
                buffer_rank=3,
                w_basis=0.0,
                basis_ema_hours=168,
                expected_hold_periods=63,
                switch_rule="tier1b_buffer_rank",
                eta_override=0.80,
            )
            barbell_symbols = list(LIQUID_8_SYMBOLS)
            logger.info(
                "Universe: LIQUID-8 | Venues: %s | Lot quantization: ON", sorted(venues.keys())
            )

            executor = CarryLiveExecutor(
                config=trading_config,
                carry_config=carry_config,
                mode=exec_mode,
                state_store=StateStore(state_path),
                perp_exchange=perp_exchange,
                spot_exchange=spot_exchange,
                capital=float(args.capital),
                execution_style=args.execution_style,
                symbols=barbell_symbols,
                venues=venues,
                overlay_config=overlay_cfg,
                enable_lot_quantization=True,
                initial_capital=float(args.capital),
                reset_state=bool(args.reset_state),
                strategy=strat,
                tier1b_gate=True,
            )
            logger.info(
                "Warmup requirement: >= 400h history (lookback_hours=%d).",
                int(getattr(executor, "lookback_hours", 720)),
            )
            if args.mode == "shadow-live":
                if not _run_shadow_live_preflight(args, barbell_symbols):
                    return 1
            return executor.run_loop(
                once=bool(args.once),
                cycles=args.cycles,
                interval_sec=float(args.interval_sec),
            )

        executor = CarryLiveExecutor(
            config=trading_config,
            carry_config=carry_config,
            mode=exec_mode,
            state_store=StateStore(state_path),
            perp_exchange=perp_exchange,
            spot_exchange=spot_exchange,
            capital=float(args.capital),
            execution_style=args.execution_style,
            initial_capital=float(args.capital),
            reset_state=bool(args.reset_state),
            strategy=strat,
        )
        logger.info(
            "Warmup requirement: >= 400h history (lookback_hours=%d).",
            int(getattr(executor, "lookback_hours", 720)),
        )
        if args.mode == "shadow-live":
            if not _run_shadow_live_preflight(args, [args.symbol]):
                return 1
        return executor.run_loop(
            once=bool(args.once),
            cycles=args.cycles,
            interval_sec=float(args.interval_sec),
        )

    config = dataclasses.replace(
        TradingConfig(),
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy_mode=args.strategy_mode,
        entry_mode=args.entry_mode,
        use_trailing_stop=bool(args.use_trailing),
        use_fee_aware_sizing=bool(args.fee_aware),
        use_ml=bool(args.use_ml),
        state_file_path=state_path,
    )

    logger.info("=" * 60)
    logger.info("QUANT TRADING BOT STARTING")
    logger.info("Mode: %s | Symbol: %s | Timeframe: %s", args.mode, config.symbol, config.timeframe)
    logger.info(
        "Strategy: %s | Entry: %s | Trailing: %s | Fee-Aware: %s | ML: %s",
        config.strategy_mode,
        config.entry_mode,
        config.use_trailing_stop,
        config.use_fee_aware_sizing,
        config.use_ml,
    )
    logger.info("State file: %s", state_path)
    logger.info("=" * 60)

    if args.synthetic or in_pytest:
        exchange: Any = _OfflinePaperExchange(
            symbol=config.symbol, timeframe=config.timeframe
        )
        data_loader = DataLoader(config, exchange=exchange)
    else:
        data_loader = DataLoader(config)

    ml_filter = MetaLabelFilter(config) if config.use_ml else None
    executor_directional = LiveExecutor(
        config=config,
        mode=exec_mode if exec_mode in ("paper", "live") else "paper",
        data_loader=data_loader,
        strategy=Strategy(config),
        risk_manager=RiskManager(config),
        state_store=StateStore(state_path),
        notifier=TelegramNotifier(env_file=None if in_pytest else Path(".env")),
        ml_filter=ml_filter,
    )
    try:
        return executor_directional.run_loop(
            once=bool(args.once),
            cycles=args.cycles,
            interval_sec=float(args.interval_sec),
        )
    except TypeError:
        return executor_directional.run_loop(once=bool(args.once))


if __name__ == "__main__":
    sys.exit(main())
