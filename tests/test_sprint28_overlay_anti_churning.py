"""Sprint 28: Unit tests for XS Overlay Anti-Churning 48h Gate, MIN_NOTIONAL barrier,
and Raw Data Refresh Retry mechanism.
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
from scripts.run_parallel_live_demo import refresh_position_ledger


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
