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
    idx = pd.date_range("2024-03-01", periods=len(rows), freq="15min", tz="UTC", name="timestamp").as_unit("ns")
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
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="timestamp").as_unit("ns")
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
    equity_before = CFG.initial_capital
    for t in rep.trades:
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.total_fee)
        notional = t.qty * t.entry_price
        assert notional <= equity_before * CFG.max_effective_leverage + 1e-9
        assert notional >= CFG.min_notional - 1e-9
        equity_before += t.net_pnl


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


# ------------------------------------------------------------------
# [MEDIUM-01] Gap-through SL/TP: giá thoát phải nằm trong thực tế có thể khớp
# ------------------------------------------------------------------

def test_long_gap_through_stop_loss_exits_at_open():
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),    # tín hiệu LONG
        (50000, 50100, 49900, 50000, 0, 500),    # vào lệnh, entry~50010, SL~49010, không chạm
        (48000, 48500, 47500, 48200, 0, 500),    # gap xuống dưới SL: open=48000
        (48200, 48250, 48150, 48200, 0, 500),
    ]
    rep, _ = run_fake(rows)
    assert rep.total_trades == 1
    tr = rep.trades[0]
    assert tr.exit_reason == "SL"
    assert tr.exit_price == pytest.approx(48000 * (1 - SLIP))  # min(open, SL) rồi trừ slippage
    assert 47500 <= tr.exit_price <= 48500


def test_short_gap_through_stop_loss_exits_at_open():
    rows = [
        (50000, 50050, 49950, 50000, -1, 500),   # tín hiệu SHORT
        (50000, 50100, 49900, 50000, 0, 500),    # entry~49990, SL~50990
        (52000, 52500, 51500, 51800, 0, 500),    # gap lên trên SL: open=52000
        (51800, 51850, 51750, 51800, 0, 500),
    ]
    rep, _ = run_fake(rows)
    assert rep.total_trades == 1
    tr = rep.trades[0]
    assert tr.exit_reason == "SL"
    assert tr.exit_price == pytest.approx(52000 * (1 + SLIP))  # max(open, SL) rồi cộng slippage
    assert 51500 <= tr.exit_price <= 52500


def test_long_gap_through_take_profit_exits_at_open():
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 50100, 49900, 50000, 0, 500),    # entry~50010, TP~51510
        (53000, 53500, 52800, 53200, 0, 500),    # gap lên trên TP: open=53000
        (53200, 53250, 53150, 53200, 0, 500),
    ]
    rep, _ = run_fake(rows)
    tr = rep.trades[0]
    assert tr.exit_reason == "TP"
    # Conservative: thoát tại open thực tế (>= TP) trừ slippage, không dưới mức TP
    assert tr.exit_price == pytest.approx(53000 * (1 - SLIP))


def test_short_gap_through_take_profit_exits_at_open():
    rows = [
        (50000, 50050, 49950, 50000, -1, 500),
        (50000, 50100, 49900, 50000, 0, 500),    # entry~49990, TP~48490
        (47000, 47200, 46500, 46800, 0, 500),    # gap xuống dưới TP: open=47000
        (46800, 46850, 46750, 46800, 0, 500),
    ]
    rep, _ = run_fake(rows)
    tr = rep.trades[0]
    assert tr.exit_reason == "TP"
    assert tr.exit_price == pytest.approx(47000 * (1 + SLIP))


# ------------------------------------------------------------------
# [MEDIUM-03] Sharpe năm hóa theo config
# ------------------------------------------------------------------

def _sharpe_from_equity(cfg: TradingConfig, values) -> float:
    bt = Backtester(cfg, strategy=FakeStrategy(pd.DataFrame()))
    idx = pd.date_range("2024-01-01", periods=len(values), freq="15min", tz="UTC", name="timestamp").as_unit("ns")
    curve = pd.Series(np.asarray(values, dtype="float64"), index=idx, name="equity")
    return bt._build_report(curve, []).sharpe_ratio


def test_sharpe_annualization_uses_config_candles_per_year():
    from dataclasses import replace

    values = [400.0, 402.0, 401.0, 404.0, 403.0, 407.0, 406.0, 410.0, 409.0, 412.0]
    cfg_1h = replace(CFG, timeframe="1h", candles_per_year=365 * 24)
    cfg_4h = replace(CFG, timeframe="4h", candles_per_year=365 * 6)

    rets = pd.Series(values).pct_change().dropna()
    base = float(rets.mean() / rets.std(ddof=1))

    assert _sharpe_from_equity(CFG, values) == pytest.approx(base * math.sqrt(365 * 24 * 4))
    assert _sharpe_from_equity(cfg_1h, values) == pytest.approx(base * math.sqrt(365 * 24))
    assert _sharpe_from_equity(cfg_4h, values) == pytest.approx(base * math.sqrt(365 * 6))
    # Tỷ lệ 1h/15m = sqrt(1/4)
    assert _sharpe_from_equity(cfg_1h, values) / _sharpe_from_equity(CFG, values) == pytest.approx(0.5)


def test_sharpe_zero_when_returns_constant_float_noise():
    # Lợi suất hằng số 1% -> std ~ 1e-17 do sai số dấu phẩy động, phải trả 0.0
    values = [100.0 * (1.01 ** k) for k in range(12)]
    assert len(pd.Series(values).pct_change().dropna()) >= 2
    assert _sharpe_from_equity(CFG, values) == 0.0


def test_sharpe_zero_when_equity_flat():
    assert _sharpe_from_equity(CFG, [400.0] * 6) == 0.0


# ------------------------------------------------------------------
# [LOW-04] Pandas 3.0: đơn vị thời gian ns & _index_to_ms
# ------------------------------------------------------------------

def test_generate_synthetic_ohlcv_index_is_utc_ns():
    from run_backtest import generate_synthetic_ohlcv

    df = generate_synthetic_ohlcv(n_candles=50, seed=1)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.unit == "ns"
    assert str(df.index.dtype) == "datetime64[ns, UTC]"


def test_index_to_ms_returns_int64_ndarray_with_exact_values():
    from src.data_loader import _index_to_ms

    base_ms = 1_704_067_200_000  # 2024-01-01T00:00:00Z
    expected = np.array([base_ms + k * 900_000 for k in range(4)], dtype="int64")
    for unit in ("ns", "us", "ms"):
        idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC").as_unit(unit)
        got = _index_to_ms(idx)
        assert isinstance(got, np.ndarray)
        assert got.dtype == np.int64
        np.testing.assert_array_equal(got, expected)


# ------------------------------------------------------------------
# Sprint 2 - Phần A2 & A3: Breakeven Stop (t+1) & Time-based Exit
# ------------------------------------------------------------------

def test_breakeven_activates_only_from_next_bar_not_same_bar():
    from src.strategy import compute_breakeven_price

    # Bar 0: phát tín hiệu LONG (atr = 500 -> initial_risk = 2 * 500 = 1000)
    # Bar 1 (nến t): khớp LONG tại open=50000 -> entry = 50010, initial_sl = 49010, tp = 51510.
    #   Ngưỡng kích hoạt breakeven (1R) = 50010 + 1000 = 51010.
    #   Trong nến 1: high=51100 (>= 51010, kích hoạt breakeven) VÀ low=49900 (<= be_sl ~ 50080.014,
    #   nhưng > initial_sl=49010). Lệnh KHÔNG được đóng ở nến 1!
    # Bar 2 (nến t+1): open=50200, high=50300, low=50000 (<= be_sl) -> đóng tại be_sl với net_pnl >= 0.
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 51100, 49900, 50300, 0, 500),
        (50200, 50300, 50000, 50100, 0, 500),
        (50100, 50150, 50050, 50100, 0, 500),
    ]
    rep, frame = run_fake(rows)
    assert rep.total_trades == 1
    tr = rep.trades[0]
    entry = 50000.0 * (1.0 + SLIP)
    be_sl = compute_breakeven_price(entry, 1, CFG)
    expected_exit = be_sl * (1.0 - SLIP)

    assert tr.entry_time == frame.index[1]
    assert tr.exit_time == frame.index[2]  # đóng ở nến t+1, KHÔNG đóng ở nến t (index[1])
    assert tr.exit_reason == "SL"
    assert tr.exit_price == pytest.approx(expected_exit)
    assert tr.net_pnl >= 0.0


def test_breakeven_short_position_covers_round_trip_fees():
    from src.strategy import compute_breakeven_price

    # Bar 0: phát tín hiệu SHORT (atr = 500 -> initial_risk = 1000)
    # Bar 1 (nến t): khớp SHORT tại open=50000 -> entry = 49990, initial_sl = 50990, tp = 48490.
    #   Ngưỡng kích hoạt breakeven (1R) = 49990 - 1000 = 48990.
    #   Trong nến 1: low=48900 (<= 48990, kích hoạt breakeven) VÀ high=50100 (>= be_sl ~ 49920.014,
    #   nhưng < initial_sl=50990). Không đóng ở nến 1.
    # Bar 2 (nến t+1): open=49800, high=50000 (>= be_sl), low=49700 -> đóng tại be_sl với net_pnl >= 0.
    rows = [
        (50000, 50050, 49950, 50000, -1, 500),
        (50000, 50100, 48900, 49700, 0, 500),
        (49800, 50000, 49700, 49900, 0, 500),
        (49900, 49950, 49850, 49900, 0, 500),
    ]
    rep, frame = run_fake(rows)
    assert rep.total_trades == 1
    tr = rep.trades[0]
    entry = 50000.0 * (1.0 - SLIP)
    be_sl = compute_breakeven_price(entry, -1, CFG)
    expected_exit = be_sl * (1.0 + SLIP)

    assert tr.entry_time == frame.index[1]
    assert tr.exit_time == frame.index[2]
    assert tr.exit_reason == "SL"
    assert tr.exit_price == pytest.approx(expected_exit)
    assert tr.net_pnl >= 0.0


def test_time_based_exit_closes_at_close_with_slippage():
    from dataclasses import replace

    cfg_time = replace(CFG, max_holding_bars=3)
    # Bar 0: tín hiệu LONG
    # Bar 1: mở lệnh tại open=50000, đi ngang (bars_held = 1)
    # Bar 2: đi ngang (bars_held = 2)
    # Bar 3: đi ngang, cuối nến bars_held = 3 >= max_holding_bars -> đóng tại close[3]=50120 với TIME_EXIT
    # Bar 4: nến dư để chứng minh đóng vì TIME_EXIT ở bar 3 chứ không phải END_OF_DATA ở bar 4
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 50100, 49900, 50020, 0, 500),
        (50020, 50100, 49920, 50050, 0, 500),
        (50050, 50150, 49950, 50120, 0, 500),
        (50120, 50200, 50000, 50150, 0, 500),
    ]
    frame = make_frame(rows)
    bt = Backtester(cfg_time, strategy=FakeStrategy(frame))
    rep = bt.run(frame)

    assert rep.total_trades == 1
    tr = rep.trades[0]
    entry = 50000.0 * (1.0 + SLIP)
    expected_exit = 50120.0 * (1.0 - SLIP)
    qty = 0.004
    gross = qty * (expected_exit - entry)
    fee = qty * entry * FEE + qty * expected_exit * FEE

    assert tr.entry_time == frame.index[1]
    assert tr.exit_time == frame.index[3]
    assert tr.exit_reason == "TIME_EXIT"
    assert tr.exit_price == pytest.approx(expected_exit)
    assert tr.gross_pnl == pytest.approx(gross)
    assert tr.total_fee == pytest.approx(fee)
    assert tr.net_pnl == pytest.approx(gross - fee)


def test_breakeven_gap_through_stop_loss_at_next_bar_open():
    """[LOW-01(b)] Gap qua Breakeven SL ở đầu nến t+1 phải thoát tại open[t+1] (kèm slippage)."""
    from src.strategy import compute_breakeven_price

    # 1) Long: nến 1 đạt >= +1R (high=51100 >= 51010) kích hoạt breakeven (be_sl ~ 50080.014).
    #    Nến 2 (t+1) mở cửa gap xuống dưới be_sl tại open=49500 -> thoát tại 49500 * (1 - SLIP).
    rows_long = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 51100, 49900, 50500, 0, 500),
        (49500, 49600, 49400, 49550, 0, 500),
        (49550, 49600, 49500, 49550, 0, 500),
    ]
    rep_l, frame_l = run_fake(rows_long)
    assert rep_l.total_trades == 1
    tr_l = rep_l.trades[0]
    entry_l = 50000.0 * (1.0 + SLIP)
    be_sl_l = compute_breakeven_price(entry_l, 1, CFG)
    assert 49500.0 < be_sl_l
    assert tr_l.entry_time == frame_l.index[1]
    assert tr_l.exit_time == frame_l.index[2]
    assert tr_l.exit_reason == "SL"
    assert tr_l.exit_price == pytest.approx(49500.0 * (1.0 - SLIP))

    # 2) Short: nến 1 đạt >= +1R (low=48900 <= 48990) kích hoạt breakeven (be_sl ~ 49920.014).
    #    Nến 2 (t+1) mở cửa gap lên trên be_sl tại open=50500 -> thoát tại 50500 * (1 + SLIP).
    rows_short = [
        (50000, 50050, 49950, 50000, -1, 500),
        (50000, 50100, 48900, 49500, 0, 500),
        (50500, 50600, 50400, 50450, 0, 500),
        (50450, 50500, 50400, 50450, 0, 500),
    ]
    rep_s, frame_s = run_fake(rows_short)
    assert rep_s.total_trades == 1
    tr_s = rep_s.trades[0]
    entry_s = 50000.0 * (1.0 - SLIP)
    be_sl_s = compute_breakeven_price(entry_s, -1, CFG)
    assert 50500.0 > be_sl_s
    assert tr_s.entry_time == frame_s.index[1]
    assert tr_s.exit_time == frame_s.index[2]
    assert tr_s.exit_reason == "SL"
    assert tr_s.exit_price == pytest.approx(50500.0 * (1.0 + SLIP))


def test_limit_pullback_entry_maker_fee_no_slippage_and_ttl_expiry():
    from dataclasses import replace

    # A) Khớp Buy Limit Maker: không slippage chiều vào, phí maker 0.02%
    cfg_limit = replace(
        CFG,
        entry_mode="limit_pullback",
        limit_pullback_atr_mult=0.5,
        limit_order_ttl_bars=3,
    )
    # Bar 0: tín hiệu LONG, atr=500, close=50000 -> limit_price = 50000 - 0.5*500 = 49750
    # Bar 1: open=50000, low=49800 (> 49750) -> chưa chạm, chưa khớp
    # Bar 2: open=49900, low=49700 (<= 49750) -> chạm limit! fill_price = min(49900, 49750) = 49750
    #        Mở vị thế Maker tại 49750 (không slippage), tp = 49750 + 3.0*500 = 51250
    # Bar 3: high=51300 >= 51250 -> chạm TP!
    rows_buy = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 50100, 49800, 49900, 0, 500),
        (49900, 50000, 49700, 49800, 0, 500),
        (49800, 51300, 49750, 51000, 0, 500),
    ]
    frame_buy = make_frame(rows_buy)
    bt_buy = Backtester(cfg_limit, strategy=FakeStrategy(frame_buy))
    rep_buy = bt_buy.run(frame_buy)

    assert rep_buy.total_trades == 1
    tr_b = rep_buy.trades[0]
    assert tr_b.entry_price == pytest.approx(49750.0)
    assert tr_b.entry_order_type == "limit"
    assert tr_b.exit_reason == "TP"
    expected_entry_fee = tr_b.qty * 49750.0 * cfg_limit.maker_fee
    expected_exit_fee = tr_b.qty * tr_b.exit_price * cfg_limit.taker_fee
    assert tr_b.total_fee == pytest.approx(expected_entry_fee + expected_exit_fee)

    # B) Khớp Sell Limit Maker: không slippage chiều vào, fill_price = max(open, limit_price)
    # Bar 0: tín hiệu SHORT, atr=500, close=50000 -> limit_price = 50000 + 0.5*500 = 50250
    # Bar 1: open=50000, high=50300 (>= 50250) -> chạm limit! fill_price = max(50000, 50250) = 50250
    #        tp = 50250 - 3.0*500 = 48750
    # Bar 2: low=48600 <= 48750 -> chạm TP!
    rows_sell = [
        (50000, 50050, 49950, 50000, -1, 500),
        (50000, 50300, 49900, 50100, 0, 500),
        (50100, 50150, 48600, 48700, 0, 500),
    ]
    frame_sell = make_frame(rows_sell)
    bt_sell = Backtester(cfg_limit, strategy=FakeStrategy(frame_sell))
    rep_sell = bt_sell.run(frame_sell)

    assert rep_sell.total_trades == 1
    tr_s = rep_sell.trades[0]
    assert tr_s.entry_price == pytest.approx(50250.0)
    assert tr_s.entry_order_type == "limit"
    assert tr_s.exit_reason == "TP"

    # C) Hết TTL: lệnh chờ bị hủy sau limit_order_ttl_bars, giá chạm sau đó không được khớp
    cfg_ttl = replace(
        CFG,
        entry_mode="limit_pullback",
        limit_pullback_atr_mult=0.5,
        limit_order_ttl_bars=2,
    )
    # Bar 0: tín hiệu LONG (limit_price = 49750)
    # Bar 1: low=49900 > 49750 (chờ nến 1)
    # Bar 2: low=49850 > 49750 (chờ nến 2 -> hết TTL 2 nến, hủy lệnh cuối nến 2)
    # Bar 3: low=49600 <= 49750 (giá chạm nhưng lệnh chờ đã bị hủy -> không khớp!)
    # Bar 4: đi ngang
    rows_ttl = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 50100, 49900, 50000, 0, 500),
        (50000, 50100, 49850, 49950, 0, 500),
        (49950, 50000, 49600, 49800, 0, 500),
        (49800, 49900, 49700, 49800, 0, 500),
    ]
    frame_ttl = make_frame(rows_ttl)
    bt_ttl = Backtester(cfg_ttl, strategy=FakeStrategy(frame_ttl))
    rep_ttl = bt_ttl.run(frame_ttl)
    assert rep_ttl.total_trades == 0

    # D) Tín hiệu ngược chiều hủy và thay thế lệnh chờ
    # Bar 0: tín hiệu LONG -> pending Buy limit (49750)
    # Bar 1: low=49800 chưa chạm, nhưng xuất hiện tín hiệu SHORT -> hủy Buy limit, tạo Sell limit (50150)
    # Bar 2: high=50200 >= 50150 -> khớp Sell limit tại 50150!
    # Bar 3: low=48500 chạm TP
    rows_rev = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 50100, 49800, 49900, -1, 500),
        (49900, 50200, 49850, 50000, 0, 500),
        (50000, 50050, 48500, 48600, 0, 500),
    ]
    frame_rev = make_frame(rows_rev)
    bt_rev = Backtester(cfg_limit, strategy=FakeStrategy(frame_rev))
    rep_rev = bt_rev.run(frame_rev)
    assert rep_rev.total_trades == 1
    assert rep_rev.trades[0].side == -1
    assert rep_rev.trades[0].entry_price == pytest.approx(50150.0)


def test_atr_trailing_stop_activates_after_threshold_and_applies_from_next_bar():
    from dataclasses import replace

    # use_trailing_stop=True, trailing_activation_r=1.5, trailing_atr_mult=2.0
    # slippage=0 để xác định giá chính xác tuyệt đối
    cfg_trail = replace(
        CFG,
        use_trailing_stop=True,
        trailing_activation_r=1.5,
        trailing_atr_mult=2.0,
        tp_atr_multiple=5.0,
        use_breakeven=False,
        slippage=0.0,
    )

    # Bar 0: tín hiệu LONG tại close=50000, atr=500
    # Bar 1: mở vị thế tại open=50000 -> entry=50000, initial_sl=49000 (R0=1000).
    #        Ngưỡng kích hoạt 1.5R = 50000 + 1.5*1000 = 51500.
    #        Nến 1: high=51000 < 51500 -> trailing_activated vẫn False, SL vẫn 49000.
    # Bar 2: open=50500. Trong nến giá vọt lên high=52000 (>= 51500, vượt ngưỡng kích hoạt 1.5R!).
    #        Đồng thời trong cùng nến 2, giá quét sâu xuống low=50200.
    #        Nếu nâng SL ngay trong nến: peak=52000 -> SL mới = 52000 - 2.0*500 = 51000.
    #        Mức low=50200 sẽ chạm 51000 và bị khớp SL sai trái ngay nến 2!
    #        NHƯNG vì thứ tự thực thi nghiêm ngặt, SL cũ (49000) được dùng ở bước (c).
    #        Vì low=50200 > 49000, lệnh KHÔNG BỊ HIT SL ở nến 2.
    #        Sau khi nến 2 đóng tại close=51500, SL mới được nâng lên 51000 (hiệu lực từ nến 3).
    # Bar 3: open=51500, high=51600, low=50900 (<= 51000), close=51100.
    #        Tại nến 3, SL 51000 có hiệu lực -> KHỚP SL TẠI 51000!
    # Bar 4: nến dư để kiểm tra
    rows = [
        (50000, 50050, 49950, 50000, 1, 500),
        (50000, 51000, 49500, 50500, 0, 500),
        (50500, 52000, 50200, 51500, 0, 500),
        (51500, 51600, 50900, 51100, 0, 500),
        (51100, 51200, 51000, 51150, 0, 500),
    ]
    frame = make_frame(rows)
    bt = Backtester(cfg_trail, strategy=FakeStrategy(frame))
    rep = bt.run(frame)

    assert rep.total_trades == 1
    tr = rep.trades[0]
    assert tr.entry_time == frame.index[1]
    # Lệnh sống sót qua nến 2 và chỉ thoát ở nến 3 với giá SL đã nâng!
    assert tr.exit_time == frame.index[3]
    assert tr.exit_reason == "SL"
    assert tr.exit_price == pytest.approx(51000.0)
    assert tr.net_pnl > 0.0

    # Kiểm tra tính bất biến Ratchet: SL chỉ siết chặt, không bao giờ nới lỏng ngược lại
    # Khi ATR tăng đột biến từ 500 lên 1000, mức cand_sl = 52000 - 2.0*1000 = 50000 < 51000
    # Ratchet max(pos.stop_loss, cand_sl) giữ nguyên 51000.
    from src.risk_manager import Position
    from src.strategy import compute_trailing_stop

    ts = pd.Timestamp("2024-03-01", tz="UTC")
    pos_long = Position(
        symbol="BTC/USDT:USDT",
        side=1,
        entry_price=50000.0,
        stop_loss=51000.0,
        take_profit=54000.0,
        qty=0.004,
        entry_time=ts,
        entry_fee=0.1,
        notional=200.0,
        initial_sl=49000.0,
        highest_price_since_entry=52000.0,
    )
    # ATR tăng làm cand = 52000 - 2.0*1000 = 50000 < 51000 -> SL không bị kéo lùi
    ratchet_sl = compute_trailing_stop(pos_long, 51800.0, 51200.0, 1000.0, cfg_trail)
    assert ratchet_sl == pytest.approx(51000.0)



