"""Sprint 28: Unit tests for XS Overlay Anti-Churning 48h Gate, MIN_NOTIONAL barrier,
Raw Data Refresh Retry mechanism, and Telegram Formatting / PnL Attribution Fixes.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig
from src.carry_live_executor import CarryLiveExecutor, MIN_OVERLAY_DELTA_NOTIONAL_USD
from src.config import TradingConfig
from src.dynamic_carry_basis import DynamicCarryConfig
from src.raw_data_refresh import _fetch_with_retry
from src.notifier import (
    diff_cycle_executions,
    format_tournament_telegram_report,
)
from src.strategy_profiles import get_strategy_profile
from scripts.run_parallel_live_demo import (
    refresh_position_ledger,
    _mark_to_market,
    _default_ledger,
)


class TestOverlayAntiChurning(unittest.TestCase):
    def setUp(self):
        self.trading_cfg = TradingConfig(symbol="BTC/USDT:USDT")
        self.carry_cfg = DynamicCarryConfig()
        self.overlay_cfg = BarbellConfig(
            overlay_rebalance_hours=48,
        )
        self.executor = CarryLiveExecutor(
            self.trading_cfg,
            self.carry_cfg,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            overlay_config=self.overlay_cfg,
            enable_lot_quantization=False,
            mode="paper",
        )

    def test_overlay_epoch_gate_holds_between_48h_boundaries(self):
        """Stable overlay composition between 48h boundaries must be held without churning."""
        self.executor.state.overlay_positions = {
            "BTC/USDT:USDT": {"qty": 0.01, "side": "long", "entry_price": 70000.0},
            "ETH/USDT:USDT": {"qty": 0.1, "side": "short", "entry_price": 2500.0},
        }
        # Non-boundary epoch: unix_ts // 3600 % 48 != 0
        ts_mid_cycle = "2026-11-06T05:00:00Z"
        self.executor.state.overlay_last_rebalance_ts = "2026-11-06T00:00:00Z"

        # Targets have slight micro-fluctuations (smooth conviction / vol targeting drift)
        allocations = {
            "overlay_active": True,
            "allocations": [
                {"symbol": "BTC/USDT:USDT", "qty": 0.0102, "side": "long"},  # slight increase
                {"symbol": "ETH/USDT:USDT", "qty": 0.098, "side": "short"},  # slight decrease
            ],
        }
        aligned_universe = {
            "BTC/USDT:USDT": pd.DataFrame({"close": [70000.0]}),
            "ETH/USDT:USDT": pd.DataFrame({"close": [2500.0]}),
        }

        res = self.executor.execute_overlay_rebalance(
            allocations,
            aligned_universe,
            timestamp=ts_mid_cycle,
        )

        self.assertEqual(res["status"], "HOLD_48H")
        self.assertEqual(res["reason"], "EPOCH_GATE_HOLD")
        self.assertEqual(len(res["orders"]), 0)
        self.assertEqual(res["turnover_notional"], 0.0)

    def test_overlay_epoch_gate_allows_rebalance_on_boundary_or_composition_change(self):
        """48h boundary or symbol composition change (closing/switching direction) triggers rebalance."""
        self.executor.state.overlay_positions = {
            "BTC/USDT:USDT": {"qty": 0.01, "side": "long", "entry_price": 70000.0},
        }

        # Case 1: Symbol is closing to 0 (composition change -> not stable -> rebalance allowed)
        allocations_close = {
            "overlay_active": True,
            "allocations": [
                {"symbol": "BTC/USDT:USDT", "qty": 0.0, "side": "sell"},
            ],
        }
        aligned_universe = {"BTC/USDT:USDT": pd.DataFrame({"close": [70000.0]})}
        ts_mid_cycle = "2026-11-06T05:00:00Z"

        res = self.executor.execute_overlay_rebalance(
            allocations_close,
            aligned_universe,
            timestamp=ts_mid_cycle,
        )
        self.assertNotEqual(res.get("status"), "HOLD_48H")
        self.assertGreaterEqual(len(res["orders"]), 1)
        self.assertEqual(res["orders"][0]["side"], "sell")

        # Case 2: 48h boundary timestamp (epoch_hours % 48 == 0)
        boundary_h = 498336  # divisible by 48
        boundary_ts = pd.Timestamp(boundary_h * 3600, unit="s", tz="UTC").isoformat()
        self.executor.state.overlay_positions = {
            "BTC/USDT:USDT": {"qty": 0.01, "side": "long", "entry_price": 70000.0},
        }
        allocations_resize = {
            "overlay_active": True,
            "allocations": [
                {"symbol": "BTC/USDT:USDT", "qty": 0.015, "side": "long"},
            ],
        }

        res_boundary = self.executor.execute_overlay_rebalance(
            allocations_resize,
            aligned_universe,
            timestamp=boundary_ts,
        )
        self.assertNotEqual(res_boundary.get("status"), "HOLD_48H")
        self.assertEqual(res_boundary.get("status"), "OK")
        self.assertEqual(len(res_boundary["orders"]), 1)

    def test_min_overlay_delta_notional_skips_micro_orders(self):
        """Delta notional < $10.0 must be skipped to avoid Binance -4164 MIN_NOTIONAL error."""
        client_mock = MagicMock()
        client_mock.fetch_exchange_reconcile_snapshot.return_value = {
            "snapshot_healthy": True,
            "perp_positions": {"BTC/USDT:USDT": 0.01},
        }
        self.executor.signed_client = client_mock
        self.executor.mode = "real"

        # Current = 0.01 BTC ($700). Target = 0.01005 BTC ($703.50). Delta = 0.00005 BTC = $3.50 USD (< $10.0)
        targets = {"BTC/USDT:USDT": 0.01005}
        current = {"BTC/USDT:USDT": 0.01}
        prices = {"BTC/USDT:USDT": 70000.0}

        res = self.executor._execute_overlay_deltas_real(
            targets,
            current,
            prices,
            timestamp="2026-11-06T00:00:00Z",
        )

        self.assertEqual(res["skipped"], 1)
        self.assertEqual(len(res["orders"]), 0)
        client_mock.place_futures_market_order.assert_not_called()

        # If target = 0 (full close), delta notional must NOT be skipped even if small
        targets_close = {"BTC/USDT:USDT": 0.0}
        current_small = {"BTC/USDT:USDT": 0.00005}  # $3.50 USD
        client_mock.place_futures_market_order.return_value = {
            "executedQty": "0.00005",
            "avgPrice": "70000.0",
            "orderId": "12345",
        }

        res_close = self.executor._execute_overlay_deltas_real(
            targets_close,
            current_small,
            prices,
            timestamp="2026-11-06T00:00:00Z",
        )
        self.assertEqual(res_close["skipped"], 0)
        self.assertEqual(len(res_close["orders"]), 1)
        client_mock.place_futures_market_order.assert_called_once()

    def test_refresh_position_ledger_zero_phantom_fees_on_price_changes(self):
        """When overlay qty is unchanged, price movement must NOT generate phantom trading fees."""
        ledger = {
            "positions_snapshot": {
                "BTC/USDT:USDT|overlay": {
                    "symbol": "BTC/USDT:USDT",
                    "qty": 0.001,
                    "entry_price": 70000.0,
                    "kind": "overlay",
                    "signed_usd": 70.0,
                }
            },
            "fees_paid_usd": 0.0,
            "realized_pnl_usd": 0.0,
            "overlay_pnl_usd": 0.0,
            "last_prices": {"BTC/USDT:USDT": 70000.0},
        }

        # Executor position is unchanged (0.001 BTC)
        class DummyExecutorState:
            overlay_positions = {
                "BTC/USDT:USDT": {"qty": 0.001, "side": "long", "entry_price": 70000.0}
            }
            carry_positions = {}

        # Snapshot with price jumping from 70,000 to 75,000
        snapshot = {
            "symbols": {
                "BTC/USDT:USDT": {
                    "spot_price": 75000.0,
                    "perp_price": 75000.0,
                    "mark_price": 75000.0,
                }
            }
        }

        fees, _, _ = refresh_position_ledger(
            ledger,
            DummyExecutorState(),
            snapshot,
            charge_fees=True,
        )

        self.assertAlmostEqual(fees, 0.0, places=9)
        self.assertEqual(ledger["fees_paid_usd"], 0.0)


class TestTelegramFormattingAndAccountingFixes(unittest.TestCase):
    def test_overlay_fill_price_uses_market_price_on_close_and_downsize(self):
        """When closing or downsizing, executions must record current market/mark price, not old entry."""
        prev = {
            "SOL/USDT:USDT|overlay": {
                "symbol": "SOL/USDT:USDT",
                "kind": "overlay",
                "qty": -1.30,
                "entry_price": 93.38,  # Old historical entry price
                "signed_usd": -121.39,
            },
            "BNB/USDT:USDT|overlay": {
                "symbol": "BNB/USDT:USDT",
                "kind": "overlay",
                "qty": 0.17,
                "entry_price": 702.35,  # Old historical entry price
                "signed_usd": 119.40,
            },
        }
        curr = {
            "SOL/USDT:USDT|overlay": {
                "symbol": "SOL/USDT:USDT",
                "kind": "overlay",
                "qty": 0.0,  # Closed
                "entry_price": 0.0,
                "signed_usd": 0.0,
            },
            "BNB/USDT:USDT|overlay": {
                "symbol": "BNB/USDT:USDT",
                "kind": "overlay",
                "qty": 0.14,  # Downsized from 0.17
                "entry_price": 702.35,
                "signed_usd": 97.44,
            },
        }
        mid_prices = {
            "SOL/USDT:USDT": 90.38,  # Current market price at execution
            "BNB/USDT:USDT": 696.00,  # Current market price at execution
        }

        diffs = diff_cycle_executions(prev, curr, mid_prices=mid_prices)
        sol_diff = next(d for d in diffs if d["symbol"] == "SOL")
        bnb_diff = next(d for d in diffs if d["symbol"] == "BNB")

        # Fill price must be market price (~90.38 and ~696.00), NEVER old historical entry (93.38 / 702.35)
        self.assertAlmostEqual(sol_diff["fill_price"], 90.38, places=2)
        self.assertAlmostEqual(bnb_diff["fill_price"], 696.00, places=2)

    def test_overlay_upnl_display_in_active_positions(self):
        """Active overlay positions must show exact signed uPnL based on mark price, not +0.00$."""
        curr_positions = {
            "BNB/USDT:USDT|overlay": {
                "symbol": "BNB/USDT:USDT",
                "kind": "overlay",
                "qty": 0.14,
                "entry_price": 700.0,
                "signed_usd": 98.0,
            },
            "SOL/USDT:USDT|overlay": {
                "symbol": "SOL/USDT:USDT",
                "kind": "overlay",
                "qty": -1.30,
                "entry_price": 93.38,
                "signed_usd": -121.39,
            },
        }
        live_ledger = {
            "pure_trading_equity_usd": 994.04,
            "yield_stacked_equity_usd": 994.04,
            "reconcile_status": "SYNCED",
            "positions_snapshot": curr_positions,
            "last_prices": {
                "BNB/USDT:USDT": 707.0,  # +7 USD gain per coin -> +0.98$ uPnL
                "SOL/USDT:USDT": 90.38,  # Short @ 93.38, price dropped to 90.38 -> +3.90$ uPnL
            },
        }
        records = [{
            "cycle": 149,
            "profile": "regime_omni_1k",
            "pure_trading_equity_usd": 994.04,
            "yield_stacked_equity_usd": 994.04,
        }]

        report = format_tournament_telegram_report(
            records,
            live_profile="regime_omni_1k",
            live_ledger=live_ledger,
        )

        # uPnL must NOT be +0.00$
        self.assertIn("BNB", report)
        self.assertIn("uPnL: +0.98$", report)
        self.assertIn("SOL", report)
        self.assertIn("uPnL: +3.90$", report)

    def test_telegram_bullet_dash_on_same_line(self):
        """Telegram bullets must stay on the same line with content without orphaned line breaks."""
        records = [{
            "cycle": 149,
            "profile": "regime_omni_1k",
            "pure_trading_equity_usd": 994.04,
            "yield_stacked_equity_usd": 994.04,
        }]
        live_ledger = {
            "pure_trading_equity_usd": 994.04,
            "yield_stacked_equity_usd": 994.04,
            "reconcile_status": "SYNCED",
            "positions_snapshot": {
                "BTC/USDT:USDT": {
                    "symbol": "BTC/USDT:USDT",
                    "kind": "carry",
                    "qty": 0.002,
                    "spot_qty": 0.002,
                    "perp_qty": -0.002,
                    "entry_price": 76000.0,
                    "signed_usd": 152.0,
                }
            },
        }

        report = format_tournament_telegram_report(
            records,
            live_profile="regime_omni_1k",
            live_ledger=live_ledger,
        )

        lines = report.splitlines()
        for idx, line in enumerate(lines):
            # A line should never be just a bullet by itself
            stripped = line.strip()
            self.assertFalse(
                stripped in ("-", "•", "- ", "• "),
                f"Line {idx} was orphaned bullet: {line!r}",
            )

    def test_overlay_pnl_attribution_realized_plus_unrealized_continuity(self):
        """Overlay PnL in ledger must represent Total PnL (realized + unrealized) to prevent jump on close."""
        prof = get_strategy_profile("regime_omni_1k")
        ledger = _default_ledger(prof)
        sym_map = {
            "SOL/USDT:USDT": {"spot_price": 90.38, "perp_price": 90.38, "mark_price": 90.38, "mid": 90.38}
        }
        last_prices = {"SOL/USDT:USDT": 90.38}

        # Step 1: Open Short SOL at 93.38 with historical realized PnL = -6.50
        ledger["overlay_realized_pnl_usd"] = -6.50
        ledger["overlay_pnl_usd"] = -6.50
        ledger["positions_snapshot"] = {
            "SOL/USDT:USDT|overlay": {
                "symbol": "SOL/USDT:USDT",
                "kind": "overlay",
                "qty": 1.30,
                "entry_price": 93.38,
                "signed_usd": -121.39,
            }
        }

        # Mark to market: unrealized gain = -121.39 * (90.38/93.38 - 1) = +3.90 USD
        _mark_to_market(ledger, prof, sym_map, last_prices)
        total_pnl_before = ledger["overlay_pnl_usd"]
        self.assertAlmostEqual(total_pnl_before, -6.50 + 3.90, places=2)

        # Step 2: Close SOL at 90.38
        class MockExecutorState:
            overlay_positions = {}
            carry_positions = {}

        refresh_position_ledger(ledger, MockExecutorState(), {"symbols": sym_map}, charge_fees=False)
        _mark_to_market(ledger, prof, sym_map, last_prices)
        total_pnl_after = ledger["overlay_pnl_usd"]

        # Total Overlay PnL must be continuous across the close event!
        self.assertAlmostEqual(total_pnl_before, total_pnl_after, places=2)


class TestRawDataRefreshRetry(unittest.TestCase):
    def test_fetch_with_retry_succeeds_after_transient_failures(self):
        """Retry succeeds when transient errors resolve before max_retries."""
        attempts = 0

        def flaky_api():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Transient network timeout")
            return "SUCCESS"

        with patch("time.sleep", return_value=None):
            result = _fetch_with_retry(flaky_api, max_retries=3, backoff_sec=0.1, tag="test_api")

        self.assertEqual(result, "SUCCESS")
        self.assertEqual(attempts, 3)

    def test_fetch_with_retry_raises_after_exceeding_max_retries(self):
        """Retry raises last exception when all retries are exhausted."""
        attempts = 0

        def dead_api():
            nonlocal attempts
            attempts += 1
            raise TimeoutError("Persistent timeout")

        with patch("time.sleep", return_value=None):
            with self.assertRaises(TimeoutError):
                _fetch_with_retry(dead_api, max_retries=3, backoff_sec=0.1, tag="dead_api")

        self.assertEqual(attempts, 3)

class TestCarryMaxEtaProfileLeverageCap(unittest.TestCase):
    def test_carry_max_eta_overlay_reduced_to_enforce_3x_cap(self):
        """carry_max_eta_1k must have overlay_gross_pct=0.10 to ensure Max Perp Notional ($780+$100=$880) <= 3.0x futures wallet ($300)."""
        prof = get_strategy_profile("carry_max_eta_1k")
        self.assertEqual(prof.overlay_gross_pct, 0.10)
        self.assertEqual(prof.carry_eta, 0.78)
        self.assertEqual(prof.spot_wallet_usd, 700.0)
        self.assertEqual(prof.futures_wallet_usd, 300.0)

        max_carry_perp_notional = prof.capital_usd * prof.carry_eta  # $780
        max_overlay_perp_notional = prof.capital_usd * prof.overlay_gross_pct  # $100
        max_total_perp_notional = max_carry_perp_notional + max_overlay_perp_notional  # $880
        max_leverage = max_total_perp_notional / prof.futures_wallet_usd  # 2.9333x

        self.assertLessEqual(max_leverage, prof.max_futures_leverage)
        self.assertAlmostEqual(max_leverage, 880.0 / 300.0, places=4)

