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
        choices=("paper", "live"),
        default="paper",
        help="Chế độ thực thi: paper hoặc live (mặc định: paper)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Chạy đúng 1 chu kỳ kiểm tra nến đóng rồi thoát",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    in_pytest = "PYTEST_CURRENT_TEST" in os.environ

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
            logger.info("Mode: %s | Strategy: barbell (paper cross-exchange carry sleeve)", args.mode)
        else:
            logger.info("QUANT TRADING BOT STARTING (DYNAMIC CARRY)")
            logger.info("Mode: %s | Strategy: %s", args.mode, strat)
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

                perp_exchange = ccxt.binanceusdm({"enableRateLimit": True})
                spot_exchange = ccxt.binance({"enableRateLimit": True})
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

                    venues["bybit"] = ccxt.bybit({"enableRateLimit": True})
                except Exception as e:
                    logger.warning("Could not initialize Bybit exchange for venues: %s", e)

            overlay_cfg = BarbellConfig(
                initial_capital=float(args.capital),
                carry_variant="tier1b",
                gating_mode="smooth_conviction",
            )
            barbell_symbols = list(LIQUID_8_SYMBOLS)
            logger.info(
                "Universe: LIQUID-8 | Venues: %s | Lot quantization: ON", sorted(venues.keys())
            )

            executor = CarryLiveExecutor(
                config=trading_config,
                carry_config=carry_config,
                mode=args.mode,
                state_store=StateStore(state_path),
                perp_exchange=perp_exchange,
                spot_exchange=spot_exchange,
                capital=float(args.capital),
                execution_style=args.execution_style,
                symbols=barbell_symbols,
                venues=venues,
                overlay_config=overlay_cfg,
                enable_lot_quantization=True,
            )
            return executor.run_loop(once=bool(args.once))

        executor = CarryLiveExecutor(
            config=trading_config,
            carry_config=carry_config,
            mode=args.mode,
            state_store=StateStore(state_path),
            perp_exchange=perp_exchange,
            spot_exchange=spot_exchange,
            capital=float(args.capital),
            execution_style=args.execution_style,
        )
        return executor.run_loop(once=bool(args.once))

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
    executor = LiveExecutor(
        config=config,
        mode=args.mode,
        data_loader=data_loader,
        strategy=Strategy(config),
        risk_manager=RiskManager(config),
        state_store=StateStore(state_path),
        notifier=TelegramNotifier(env_file=None if in_pytest else Path(".env")),
        ml_filter=ml_filter,
    )
    return executor.run_loop(once=bool(args.once))


if __name__ == "__main__":
    sys.exit(main())
