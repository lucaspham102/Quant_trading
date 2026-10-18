"""Test RiskManager và Strategy: biên sizing, circuit breaker UTC, tín hiệu, No Look-Ahead Bias."""
import numpy as np
import pandas as pd
import pytest

from src import features
from src.config import TradingConfig
from src.risk_manager import RiskManager
from src.strategy import Strategy


# ----------------------------- RiskManager -----------------------------

def test_position_sizing_standard():
    rm = RiskManager(TradingConfig())
    qty = rm.calculate_position_size(50000.0, 49000.0)
    assert qty == pytest.approx(0.004, abs=1e-12)


def test_leverage_clamp_when_stop_too_tight():
    cfg = TradingConfig()
    rm = RiskManager(cfg)
    # raw_qty = 4 / 10 = 0.4 BTC -> notional 20000 >> 800 => phải kẹp
    qty = rm.calculate_position_size(50000.0, 49990.0)
    assert qty is not None
    assert qty * 50000.0 <= 800.0 + 1e-9
    assert qty == pytest.approx(0.016, abs=1e-12)
    # Làm tròn theo step_size
    assert abs(qty / cfg.step_size - round(qty / cfg.step_size)) < 1e-6


def test_leverage_clamp_rounds_down_not_up():
    rm = RiskManager(TradingConfig())
    entry = 30001.0  # max_qty = 800/30001 = 0.026665... -> 0.026
    qty = rm.calculate_position_size(entry, entry - 1.0)
    assert qty == pytest.approx(0.026, abs=1e-12)
    assert qty * entry <= 800.0


def test_min_notional_rejection():
    rm = RiskManager(TradingConfig())
    # raw_qty = 4/90 = 0.0444 -> notional 4.44 < 5
    assert rm.calculate_position_size(100.0, 10.0) is None
    # qty làm tròn về 0
    assert rm.calculate_position_size(50000.0, 40000.0) is None
    # Đầu vào không hợp lệ
    assert rm.calculate_position_size(50000.0, 50000.0) is None
    assert rm.calculate_position_size(-1.0, 10.0) is None


def test_build_position_fields():
    rm = RiskManager(TradingConfig())
    ts = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    pos = rm.build_position(1, 50000.0, 49000.0, 53000.0, ts)
    assert pos is not None
    assert pos.qty == pytest.approx(0.004)
    assert pos.notional == pytest.approx(200.0)
    assert pos.entry_fee == pytest.approx(200.0 * 0.0005)
    assert rm.build_position(1, 100.0, 10.0, 200.0, ts) is None


def test_circuit_breaker_utc_daily_reset():
    rm = RiskManager(TradingConfig())
    t1 = pd.Timestamp("2025-01-01 08:00", tz="UTC")
    assert rm.can_open_trade(t1) is True

    rm.update_after_trade_close(-5.0, pd.Timestamp("2025-01-01 09:00", tz="UTC"))
    assert rm.can_open_trade(pd.Timestamp("2025-01-01 09:15", tz="UTC")) is True

    rm.update_after_trade_close(-7.0, pd.Timestamp("2025-01-01 10:00", tz="UTC"))  # tổng -12
    assert rm.circuit_breaker_tripped is True
    assert rm.can_open_trade(pd.Timestamp("2025-01-01 10:15", tz="UTC")) is False
    assert rm.can_open_trade(pd.Timestamp("2025-01-01 23:45", tz="UTC")) is False
    assert rm.build_position(1, 50000.0, 49000.0, 53000.0,
                             pd.Timestamp("2025-01-01 23:45", tz="UTC")) is None

    # Sang 00:00 UTC ngày kế tiếp -> mở khóa
    assert rm.can_open_trade(pd.Timestamp("2025-01-02 00:00", tz="UTC")) is True
    assert rm.daily_realized_pnl == 0.0
    assert rm.daily_start_equity == pytest.approx(388.0)


def test_circuit_breaker_single_big_loss():
    rm = RiskManager(TradingConfig())
    rm.update_after_trade_close(-12.0, pd.Timestamp("2025-03-01 01:00", tz="UTC"))
    assert rm.can_open_trade(pd.Timestamp("2025-03-01 01:15", tz="UTC")) is False


# ------------------------------- Strategy -------------------------------

def _feature_frame(rows):
    idx = pd.date_range("2025-01-01", periods=len(rows), freq="15min", tz="UTC", name="timestamp").as_unit("ns")
    df = pd.DataFrame(rows, index=idx)
    for col in ["close", "atr", "rsi", "donchian_upper", "donchian_lower", "vol_zscore"]:
        df[col] = df[col].astype("float64")
    for col in ["open", "high", "low", "volume", "ema_fast", "ema_slow", "donchian_mid"]:
        df[col] = 1.0
    df["regime"] = df["regime"].astype(np.int8)
    return df


def test_strategy_long_short_flat_signals():
    base = dict(open=1.0, high=1.0, low=1.0, volume=1.0)
    rows = [
        # long hợp lệ
        dict(close=105.0, atr=2.0, rsi=60.0, donchian_upper=100.0, donchian_lower=90.0, vol_zscore=2.0, regime=1),
        # short hợp lệ
        dict(close=85.0, atr=2.0, rsi=40.0, donchian_upper=100.0, donchian_lower=90.0, vol_zscore=2.0, regime=-1),
        # breakout nhưng volume yếu
        dict(close=105.0, atr=2.0, rsi=60.0, donchian_upper=100.0, donchian_lower=90.0, vol_zscore=1.0, regime=1),
        # breakout nhưng regime sai
        dict(close=105.0, atr=2.0, rsi=60.0, donchian_upper=100.0, donchian_lower=90.0, vol_zscore=2.0, regime=0),
        # long nhưng RSI quá cao
        dict(close=105.0, atr=2.0, rsi=80.0, donchian_upper=100.0, donchian_lower=90.0, vol_zscore=2.0, regime=1),
        # short nhưng RSI quá thấp
        dict(close=85.0, atr=2.0, rsi=20.0, donchian_upper=100.0, donchian_lower=90.0, vol_zscore=2.0, regime=-1),
        # không breakout
        dict(close=95.0, atr=2.0, rsi=50.0, donchian_upper=100.0, donchian_lower=90.0, vol_zscore=2.0, regime=1),
    ]
    df = _feature_frame([{**base, **r} for r in rows])
    cfg = TradingConfig()
    out = Strategy(cfg).generate_signals(df)

    assert list(out["signal"]) == [1, -1, 0, 0, 0, 0, 0]
    assert out["stop_loss"].dtype == np.float64
    assert out["take_profit"].dtype == np.float64
    assert out["stop_loss"].iloc[0] == pytest.approx(105.0 - cfg.sl_atr_multiple * 2.0)
    assert out["take_profit"].iloc[0] == pytest.approx(105.0 + cfg.tp_atr_multiple * 2.0)
    assert out["stop_loss"].iloc[1] == pytest.approx(85.0 + cfg.sl_atr_multiple * 2.0)
    assert out["take_profit"].iloc[1] == pytest.approx(85.0 - cfg.tp_atr_multiple * 2.0)
    assert out["stop_loss"].iloc[2:].isna().all()
    assert out["take_profit"].iloc[2:].isna().all()


def _make_ohlcv(n, seed, extreme_tail=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.003, n)
    vol = rng.lognormal(mean=5.0, sigma=0.4, size=n)
    # Chèn các nến volume đột biến để có breakout thật
    spikes = rng.choice(n - extreme_tail, size=(n - extreme_tail) // 15, replace=False)
    vol[spikes] *= 4.0
    if extreme_tail:
        rets[-extreme_tail:] = rng.normal(0.0, 0.05, extreme_tail)
        rets[-extreme_tail] = -0.25  # flash crash
        vol[-extreme_tail:] *= 10.0
    close = 50000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[50000.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.002, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC", name="timestamp").as_unit("ns")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def test_strategy_generates_from_raw_ohlcv_columns():
    df = _make_ohlcv(800, seed=7)
    out = Strategy(TradingConfig()).generate_signals(df)
    for col in ["signal", "stop_loss", "take_profit", "atr", "regime"]:
        assert col in out.columns
    assert set(out["signal"].unique()).issubset({-1, 0, 1})
    flat = out["signal"] == 0
    assert out.loc[flat, "stop_loss"].isna().all()
    assert out.loc[~flat, "stop_loss"].notna().all()


def test_strategy_no_look_ahead_bias():
    T, extra = 600, 50
    full_raw = _make_ohlcv(T + extra, seed=42, extreme_tail=extra)
    sub_raw = full_raw.iloc[:T]
    strat = Strategy(TradingConfig())

    sig_sub = strat.generate_signals(sub_raw)
    sig_full = strat.generate_signals(full_raw)

    full_cut = sig_full.loc[sig_full.index.isin(sig_sub.index)]
    assert len(full_cut) == len(sig_sub) > 0
    assert full_cut.index.equals(sig_sub.index)
    for col in ["signal", "stop_loss", "take_profit"]:
        a = sig_sub[col].to_numpy(dtype="float64")
        b = full_cut[col].to_numpy(dtype="float64")
        assert np.allclose(a, b, equal_nan=True), f"Look-ahead bias ở cột {col}"

    # Đặc trưng cũng không được rò rỉ
    feat_sub = features.add_all_features(sub_raw, TradingConfig(), drop_warmup=False)
    feat_full = features.add_all_features(full_raw, TradingConfig(), drop_warmup=False).iloc[:T]
    assert np.allclose(
        feat_sub.to_numpy(dtype="float64"), feat_full.to_numpy(dtype="float64"), equal_nan=True
    )


# ------------------------------------------------------------------
# [MEDIUM-02] current_equity <= 0
# ------------------------------------------------------------------

@pytest.mark.parametrize("bad_equity", [0.0, -5.0])
def test_non_positive_equity_blocks_trading_same_day(bad_equity):
    rm = RiskManager(TradingConfig())
    ts = pd.Timestamp("2025-01-01 08:00", tz="UTC")
    assert rm.can_open_trade(ts) is True  # khởi tạo ngày với equity 400
    rm.current_equity = bad_equity        # daily_start_equity vẫn 400, chưa chạm giới hạn lỗ ngày
    assert rm.can_open_trade(pd.Timestamp("2025-01-01 08:15", tz="UTC")) is False
    assert rm.calculate_position_size(50000.0, 49000.0) is None
    assert rm.build_position(1, 50000.0, 49000.0, 53000.0, pd.Timestamp("2025-01-01 08:15", tz="UTC")) is None


@pytest.mark.parametrize("bad_equity", [0.0, -5.0])
def test_non_positive_equity_blocks_trading_on_new_utc_day(bad_equity):
    rm = RiskManager(TradingConfig())
    assert rm.can_open_trade(pd.Timestamp("2025-01-01 08:00", tz="UTC")) is True
    rm.current_equity = bad_equity
    new_day = pd.Timestamp("2025-01-02 00:00", tz="UTC")
    assert rm.can_open_trade(new_day) is False
    assert rm.calculate_position_size(50000.0, 49000.0) is None


# ------------------------------------------------------------------
# [LOW-03] tz-naive timestamp coi là UTC
# ------------------------------------------------------------------

def test_naive_timestamp_treated_as_utc_in_risk_manager():
    rm = RiskManager(TradingConfig())
    naive = pd.Timestamp("2025-01-01 23:45")
    rm.on_new_Timestamp(naive)
    assert rm.current_utc_date == pd.Timestamp("2025-01-01", tz="UTC").date()

    pos = rm.build_position(1, 50000.0, 49000.0, 53000.0, naive)
    assert pos is not None
    assert pos.entry_time.tzinfo is not None
    assert pos.entry_time.utcoffset() == pd.Timedelta(0)
    assert pos.entry_time == pd.Timestamp("2025-01-01 23:45", tz="UTC")


def test_aware_non_utc_entry_time_normalized_to_utc():
    rm = RiskManager(TradingConfig())
    ts = pd.Timestamp("2025-01-02 03:00", tz="Asia/Bangkok")  # = 2025-01-01 20:00 UTC
    pos = rm.build_position(1, 50000.0, 49000.0, 53000.0, ts)
    assert pos is not None
    assert str(pos.entry_time.tz) == "UTC"
    assert pos.entry_time == pd.Timestamp("2025-01-01 20:00", tz="UTC")
