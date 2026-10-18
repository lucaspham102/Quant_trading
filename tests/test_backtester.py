"""Test Backtester với dữ liệu synthetic (offline, deterministic)."""
import math

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester, BacktestReport, TradeRecord
from src.config import TradingConfig

CFG = TradingConfig()
SLIP = CFG.slippage
FEE = CFG.taker_fee


class FakeStrategy:
    """Trả về DataFrame đã gắn sẵn signal/atr để kiểm soát tuyệt đối kịch bản."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.frame.copy()


def make_frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (open, high, low, close, signal, atr)."""
    idx = pd.date_range("2024-03-01", periods=len(rows), freq="15min", tz="UTC", name="timestamp")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "signal", "atr"], index=idx)
    df["volume"] = 100.0
    for c in ["open", "high", "low", "close", "atr"]:
        df[c] = df[c].astype("float64")
    return df


def run_fake(rows: list[tuple]):
    frame = make_frame(rows)
    bt = Backtester(CFG, strategy=FakeStrategy(frame))
    return bt.run(frame), frame


def test_long_take_profit_pnl_and_fee_exact():
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),    # tín hiệu LONG
        (50000, 51600, 50000, 51500, 0, 500),    # khớp tại open, chạm TP
        (51500, 51550, 51450, 51500, 0, 500),
    ]
    rep, _ = run_fake(rows)
    assert rep.total_trades == 1
    tr = rep.trades[0]
    entry = 50000 * (1 + SLIP)
    tp = entry + 3.0 * 500
    exit_px = tp * (1 - SLIP)
    qty = 0.004  # 4 USDT / (2*500)
    gross = qty * (exit_px - entry)
    fee = qty * entry * FEE + qty * exit_px * FEE
    assert tr.exit_reason == "TP"
    assert tr.side == 1
    assert tr.qty == pytest.approx(qty)
    assert tr.entry_price == pytest.approx(entry)
    assert tr.exit_price == pytest.approx(exit_px)
    assert tr.gross_pnl == pytest.approx(gross)
    assert tr.total_fee == pytest.approx(fee)
    assert tr.net_pnl == pytest.approx(gross - fee)
    assert rep.total_fees_paid == pytest.approx(fee)
    assert rep.final_equity == pytest.approx(400.0 + gross - fee)


def test_short_take_profit_pnl_and_fee_exact():
    rows = [
        (50000, 50050, 49950, 50000, -1, 500),   # tín hiệu SHORT
        (50000, 50100, 48000, 48200, 0, 500),
        (48200, 48250, 48150, 48200, 0, 500),
    ]
    rep, _ = run_fake(rows)
    tr = rep.trades[0]
    entry = 50000 * (1 - SLIP)
    tp = entry - 3.0 * 500
    exit_px = tp * (1 + SLIP)
    qty = 0.004
    gross = qty * (exit_px - entry) * -1
    fee = qty * entry * FEE + qty * exit_px * FEE
    assert tr.exit_reason == "TP"
    assert tr.side == -1
    assert tr.entry_price == pytest.approx(entry)
    assert tr.exit_price == pytest.approx(exit_px)
    assert gross > 0
    assert tr.net_pnl == pytest.approx(gross - fee)


def test_long_stop_loss_loses_about_risk_plus_costs():
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 50200, 48900, 49000, 0, 500),
        (49000, 49050, 48950, 49000, 0, 500),
    ]
    rep, _ = run_fake(rows)
    tr = rep.trades[0]
    entry = 50000 * (1 + SLIP)
    sl = entry - 1000.0
    exit_px = sl * (1 - SLIP)
    assert tr.exit_reason == "SL"
    assert tr.exit_price == pytest.approx(exit_px)
    assert tr.net_pnl == pytest.approx(0.004 * (exit_px - entry) - 0.004 * (entry + exit_px) * FEE)
    assert tr.net_pnl < -4.0  # lỗ > 1% NAV do phí + slippage
    assert rep.win_rate_pct == 0.0
    assert rep.profit_factor == 0.0


def test_candle_hitting_both_sl_and_tp_assumes_sl_first():
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 53000, 47000, 50000, 0, 500),   # quét cả SL (49010) lẫn TP (51510)
        (50000, 50050, 49950, 50000, 0, 500),
    ]
    rep, _ = run_fake(rows)
    assert rep.total_trades == 1
    assert rep.trades[0].exit_reason == "SL"
    assert rep.trades[0].net_pnl < 0

    # Short: cũng phải là SL
    rows_s = [
        (50000, 50050, 49950, 50000, -1, 500),
        (50000, 53000, 47000, 50000, 0, 500),
        (50000, 50050, 49950, 50000, 0, 500),
    ]
    rep_s, _ = run_fake(rows_s)
    assert rep_s.trades[0].exit_reason == "SL"


def test_next_bar_open_execution_with_slippage():
    rows = [
        (40000, 50050, 39950, 50000, 1, 500),    # signal[0]=1; open[0] khác hẳn open[1]
        (50000, 50100, 49900, 50000, 0, 500),
        (50000, 50100, 49900, 50050, 0, 500),
    ]
    rep, frame = run_fake(rows)
    tr = rep.trades[0]
    assert tr.entry_time == frame.index[1]
    assert tr.entry_price == pytest.approx(50000 * (1 + SLIP))
    assert tr.entry_price != pytest.approx(40000 * (1 + SLIP))
    # Không chạm SL/TP -> END_OF_DATA ở close nến cuối
    assert tr.exit_reason == "END_OF_DATA"
    assert tr.exit_price == pytest.approx(50050 * (1 - SLIP))
    assert tr.exit_time == frame.index[-1]


def test_signal_on_last_bar_is_not_executed():
    rows = [
        (50000, 50050, 49950, 50000, 0, 500),
        (50000, 50050, 49950, 50000, 1, 500),
    ]
    rep, _ = run_fake(rows)
    assert rep.total_trades == 0
    assert rep.final_equity == pytest.approx(400.0)
    assert rep.profit_factor == 0.0
    assert rep.sharpe_ratio == 0.0


def test_signal_reverse_closes_at_open():
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 50100, 49900, 50000, -1, 500),   # signal đảo chiều
        (50200, 50300, 50100, 50200, 0, 500),    # đóng long tại open, mở short
        (50200, 50300, 50100, 50200, 0, 500),
    ]
    rep, _ = run_fake(rows)
    assert rep.trades[0].exit_reason == "SIGNAL_REVERSE"
    assert rep.trades[0].exit_price == pytest.approx(50200 * (1 - SLIP))
    assert rep.trades[1].side == -1
    assert rep.trades[1].entry_price == pytest.approx(50200 * (1 - SLIP))
    assert rep.trades[1].exit_reason == "END_OF_DATA"


def _synthetic_ohlcv(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-1.0, 1.0], size=n // 300 + 1) * 0.0008, 300)[:n]
    spike = rng.random(n) < 0.06
    ret = drift + rng.normal(0, 0.002, n) * np.where(spike, 3.0, 1.0)
    close = 60000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[60000.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = rng.lognormal(4.0, 0.3, n) * np.where(spike, 5.0, 1.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="timestamp")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
    df.attrs["symbol"] = "BTC/USDT:USDT"
    df.attrs["timeframe"] = "15m"
    return df


def test_full_pipeline_real_strategy_conservation():
    df = _synthetic_ohlcv()
    rep = Backtester(CFG).run(df)
    assert isinstance(rep, BacktestReport)
    assert rep.final_equity == pytest.approx(CFG.initial_capital + sum(t.net_pnl for t in rep.trades))
    assert rep.total_fees_paid == pytest.approx(sum(t.total_fee for t in rep.trades))
    for t in rep.trades:
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.total_fee)
        assert t.qty * t.entry_price <= CFG.initial_capital * 10 * CFG.max_effective_leverage


def test_report_fields_and_sane_values():
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 51600, 50000, 51500, 0, 500),   # TP thắng
        (51500, 51550, 51450, 51500, 1, 500),
        (51500, 51600, 49900, 50000, 0, 500),   # SL thua (entry~51510, sl~50510)
        (50000, 50050, 49950, 50000, 0, 500),
    ]
    rep, frame = run_fake(rows)
    assert rep.total_trades == 2
    assert {t.exit_reason for t in rep.trades} == {"TP", "SL"}
    assert 0.0 <= rep.win_rate_pct <= 100.0
    assert rep.win_rate_pct == pytest.approx(50.0)
    assert rep.profit_factor > 0 and math.isfinite(rep.profit_factor)
    assert rep.max_drawdown_pct >= 0.0
    assert math.isfinite(rep.sharpe_ratio)
    assert isinstance(rep.equity_curve, pd.Series)
    assert str(rep.equity_curve.index.tz) == "UTC"
    assert len(rep.equity_curve) == len(frame)
    assert rep.equity_curve.iloc[-1] == pytest.approx(rep.final_equity)
    assert rep.total_return_pct == pytest.approx((rep.final_equity - 400.0) / 400.0 * 100.0)
    assert all(isinstance(t, TradeRecord) for t in rep.trades)
    assert rep.final_equity == pytest.approx(400.0 + sum(t.net_pnl for t in rep.trades))
    assert rep.total_fees_paid > 0
