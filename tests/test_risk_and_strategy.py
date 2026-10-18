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


# ------------------------- Sprint 3: Min stop & fee-aware sizing -------------------------

def test_min_stop_distance_widen_and_reject():
    from dataclasses import replace

    from src.risk_manager import enforce_min_stop_distance

    cfg = replace(TradingConfig(), min_stop_distance_pct=0.008)
    entry = 50000.0
    d_min = entry * 0.008  # 400

    # Long: SL 0.3% < 0.8% -> nới ra đúng 0.8%, giữ R:R = 2.0
    res = enforce_min_stop_distance(1, entry, 49850.0, 50300.0, cfg)
    assert res is not None
    sl, tp = res
    assert sl == pytest.approx(entry - d_min)
    assert (tp - entry) / (entry - sl) == pytest.approx(2.0)

    # Short
    res = enforce_min_stop_distance(-1, entry, 50150.0, 49700.0, cfg)
    assert res is not None
    sl, tp = res
    assert sl == pytest.approx(entry + d_min)
    assert (entry - tp) / (sl - entry) == pytest.approx(2.0)

    # Đủ khoảng cách -> giữ nguyên
    assert enforce_min_stop_distance(1, entry, 49500.0, 51000.0, cfg) == (49500.0, 51000.0)
    # Khi nới SL, TP cũng được nới theo để BẢO TOÀN R:R ban đầu (spec §2.2.B)
    res = enforce_min_stop_distance(1, entry, 49850.0, 52000.0, cfg)
    assert res is not None
    rr0 = (52000.0 - entry) / (entry - 49850.0)  # 13.333... giữ nguyên tỷ lệ
    assert res[0] == pytest.approx(entry - d_min)
    assert res[1] == pytest.approx(entry + rr0 * d_min)
    # SL sai hướng -> None
    assert enforce_min_stop_distance(1, entry, 50100.0, 51000.0, cfg) is None
    assert enforce_min_stop_distance(-1, entry, 49900.0, 49000.0, cfg) is None
    # min_stop_distance_pct = 0 -> giữ nguyên
    assert enforce_min_stop_distance(1, entry, 49990.0, 50010.0, TradingConfig()) == (49990.0, 50010.0)

    # reject
    cfg_rej = replace(cfg, min_stop_action="reject")
    assert enforce_min_stop_distance(1, entry, 49850.0, 50300.0, cfg_rej) is None
    assert enforce_min_stop_distance(-1, entry, 50150.0, 49700.0, cfg_rej) is None

    # build_position tích hợp
    ts = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    pos = RiskManager(cfg).build_position(1, entry, 49850.0, 50300.0, ts)
    assert pos is not None
    assert pos.stop_loss == pytest.approx(entry - d_min)
    assert pos.initial_sl == pytest.approx(entry - d_min)
    assert RiskManager(cfg_rej).build_position(1, entry, 49850.0, 50300.0, ts) is None


def test_fee_aware_sizing_caps_total_sl_loss_within_1pct_nav():
    import math
    from dataclasses import replace

    base = TradingConfig()
    cfg = replace(base, use_fee_aware_sizing=True)
    entry = 50000.0
    risk_amount = base.initial_capital * base.risk_per_trade_pct  # 4.0

    for sl_pct in (0.004, 0.008):
        sl = entry * (1.0 - sl_pct)
        dist = entry - sl
        eff = dist + entry * 2.0 * (cfg.taker_fee + cfg.slippage)
        expected = round(math.floor(risk_amount / eff / cfg.step_size + 1e-9) * cfg.step_size, 12)

        qty = RiskManager(cfg).calculate_position_size(entry, sl)
        assert qty == pytest.approx(expected, abs=1e-12)
        # Không bị kẹp bởi trần leverage 2.0x
        assert qty * entry < base.initial_capital * base.max_effective_leverage
        # Legacy (mặc định) sizing lớn hơn hoặc bằng fee-aware
        legacy = RiskManager(base).calculate_position_size(entry, sl)
        assert legacy is not None and legacy >= qty

        # Khi chạm SL (kèm trượt giá + phí 2 chiều) lỗ ròng <= 4.0 USDT
        exit_price = sl * (1.0 - cfg.slippage)
        gross = qty * (exit_price - entry)
        fees = qty * entry * cfg.taker_fee + qty * exit_price * cfg.taker_fee
        net = gross - fees
        assert abs(net) <= risk_amount + 1e-9

        # Short đối xứng
        sl_s = entry * (1.0 + sl_pct)
        qty_s = RiskManager(cfg).calculate_position_size(entry, sl_s)
        assert qty_s is not None
        exit_s = sl_s * (1.0 + cfg.slippage)
        net_s = qty_s * (entry - exit_s) - qty_s * entry * cfg.taker_fee - qty_s * exit_s * cfg.taker_fee
        assert abs(net_s) <= risk_amount + 1e-9

    # Maker entry: phí maker + taker + 2*slippage
    sl = entry * (1.0 - 0.004)
    qty_m = RiskManager(cfg).calculate_position_size(entry, sl, is_maker_entry=True)
    eff_m = (entry - sl) + entry * (cfg.maker_fee + cfg.taker_fee + 2.0 * cfg.slippage)
    assert qty_m is not None
    assert qty_m * eff_m <= risk_amount + 1e-9

    # build_position: entry_fee theo maker/taker
    ts = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    pos_m = RiskManager(cfg).build_position(1, entry, sl, 51000.0, ts, is_maker_entry=True)
    pos_t = RiskManager(cfg).build_position(1, entry, sl, 51000.0, ts)
    assert pos_m is not None and pos_t is not None
    assert pos_m.entry_fee == pytest.approx(pos_m.notional * cfg.maker_fee)
    assert pos_t.entry_fee == pytest.approx(pos_t.notional * cfg.taker_fee)
    assert pos_m.entry_order_type == "limit" and pos_t.entry_order_type == "market"


def test_strategy_modes_trend_pullback_mean_reversion_and_regime_adaptive():
    from dataclasses import replace
    from run_backtest import generate_synthetic_ohlcv

    # 1) Dữ liệu dựng tay kiểm tra tín hiệu Long, Short, Flat và tỷ lệ R:R
    idx = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC", name="timestamp").as_unit("ns")
    data = {
        "open": [50000.0, 50600.0, 49400.0, 49100.0, 50800.0, 50000.0],
        "high": [50100.0, 51200.0, 49600.0, 50800.0, 51200.0, 50200.0],
        "low": [49900.0, 50400.0, 48800.0, 48800.0, 50200.0, 49800.0],
        "close": [50000.0, 51000.0, 49000.0, 50600.0, 50400.0, 50000.0],
        "volume": [100.0, 150.0, 150.0, 120.0, 120.0, 100.0],
        "atr": [500.0, 500.0, 500.0, 400.0, 400.0, 500.0],
        "rsi": [50.0, 55.0, 40.0, 28.0, 72.0, 50.0],
        "ema_fast": [50000.0, 52000.0, 48000.0, 50000.0, 50000.0, 50000.0],
        "ema_slow": [50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0],
        "ema_20": [50000.0, 50500.0, 49500.0, 50000.0, 50000.0, 50000.0],
        "ema_slow_slope": [0.0, 0.01, -0.01, 0.0, 0.0, 0.0],
        "adx": [20.0, 30.0, 30.0, 18.0, 18.0, 20.0],
        "plus_di": [20.0, 35.0, 15.0, 15.0, 30.0, 20.0],
        "minus_di": [20.0, 15.0, 35.0, 30.0, 15.0, 20.0],
        "bb_mid": [50000.0, 50400.0, 49600.0, 50000.0, 50000.0, 50000.0],
        "bb_upper": [51000.0, 51500.0, 50500.0, 51000.0, 51000.0, 51000.0],
        "bb_lower": [49000.0, 49500.0, 48500.0, 49000.0, 49000.0, 49000.0],
        "bb_pct_b": [0.5, 0.75, 0.25, 0.8, 0.7, 0.5],
        "vol_zscore": [0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        "regime": [0, 1, -1, 0, 0, 0],
        "donchian_upper": [50500.0, 50500.0, 50500.0, 50500.0, 50500.0, 50500.0],
        "donchian_lower": [49500.0, 49500.0, 49500.0, 49500.0, 49500.0, 49500.0],
        "donchian_mid": [50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0],
        "ema_fast_slope": [0.0, 0.02, -0.02, 0.0, 0.0, 0.0],
    }
    hand_df = pd.DataFrame(data, index=idx)

    # A) Chế độ trend_pullback: R:R >= 2.0
    cfg_tp = replace(TradingConfig(), strategy_mode="trend_pullback", pullback_rr_ratio=2.0)
    sig_tp = Strategy(cfg_tp).generate_signals(hand_df)

    # Nến 1: Long Pullback
    assert sig_tp["signal"].iloc[1] == 1
    sl_1 = float(sig_tp["stop_loss"].iloc[1])
    tp_1 = float(sig_tp["take_profit"].iloc[1])
    c_1 = float(hand_df["close"].iloc[1])
    rr_long = (tp_1 - c_1) / (c_1 - sl_1)
    assert rr_long >= 2.0
    assert np.isfinite(sig_tp["limit_price"].iloc[1])

    # Nến 2: Short Pullback
    assert sig_tp["signal"].iloc[2] == -1
    sl_2 = float(sig_tp["stop_loss"].iloc[2])
    tp_2 = float(sig_tp["take_profit"].iloc[2])
    c_2 = float(hand_df["close"].iloc[2])
    rr_short = (c_2 - tp_2) / (sl_2 - c_2)
    assert rr_short >= 2.0
    assert np.isfinite(sig_tp["limit_price"].iloc[2])

    # Nến 5: Flat
    assert sig_tp["signal"].iloc[5] == 0
    assert np.isnan(sig_tp["limit_price"].iloc[5])

    # B) Chế độ mean_reversion: R:R >= 1.5
    cfg_mr = replace(TradingConfig(), strategy_mode="mean_reversion", mr_rr_ratio=1.5)
    sig_mr = Strategy(cfg_mr).generate_signals(hand_df)

    # Nến 3: Long Mean Reversion (oversold, chạm dưới biên, đảo chiều tăng)
    assert sig_mr["signal"].iloc[3] == 1
    sl_3 = float(sig_mr["stop_loss"].iloc[3])
    tp_3 = float(sig_mr["take_profit"].iloc[3])
    c_3 = float(hand_df["close"].iloc[3])
    rr_mr_long = (tp_3 - c_3) / (c_3 - sl_3)
    assert rr_mr_long >= 1.5

    # Nến 4: Short Mean Reversion (overbought, chạm trên biên, đảo chiều giảm)
    assert sig_mr["signal"].iloc[4] == -1
    sl_4 = float(sig_mr["stop_loss"].iloc[4])
    tp_4 = float(sig_mr["take_profit"].iloc[4])
    c_4 = float(hand_df["close"].iloc[4])
    rr_mr_short = (c_4 - tp_4) / (sl_4 - c_4)
    assert rr_mr_short >= 1.5

    # Nến 5: Flat
    assert sig_mr["signal"].iloc[5] == 0

    # C) Chế độ regime_adaptive: chuyển nhánh theo adx >= adx_threshold (25.0)
    cfg_adapt = replace(TradingConfig(), strategy_mode="regime_adaptive", adx_threshold=25.0)
    sig_adapt = Strategy(cfg_adapt).generate_signals(hand_df)

    # Nến 1 & 2 (adx=30 >= 25): lấy theo trend_pullback -> 1 và -1
    assert sig_adapt["signal"].iloc[1] == sig_tp["signal"].iloc[1] == 1
    assert sig_adapt["signal"].iloc[2] == sig_tp["signal"].iloc[2] == -1
    assert sig_adapt["stop_loss"].iloc[1] == pytest.approx(sig_tp["stop_loss"].iloc[1])

    # Nến 3 & 4 (adx=18 < 25): lấy theo mean_reversion -> 1 và -1
    assert sig_adapt["signal"].iloc[3] == sig_mr["signal"].iloc[3] == 1
    assert sig_adapt["signal"].iloc[4] == sig_mr["signal"].iloc[4] == -1
    assert sig_adapt["stop_loss"].iloc[3] == pytest.approx(sig_mr["stop_loss"].iloc[3])

    # D) Verify No Look-Ahead Bias trên cả 4 modes
    raw_df = generate_synthetic_ohlcv(n_candles=260, seed=777)
    full_feat = features.add_all_features(raw_df, TradingConfig(), drop_warmup=True)
    assert len(full_feat) > 30

    cutoff = len(full_feat) - 5
    for mode in ("breakout", "trend_pullback", "mean_reversion", "regime_adaptive"):
        cfg_m = replace(TradingConfig(), strategy_mode=mode)
        strat_m = Strategy(cfg_m)

        res_full = strat_m.generate_signals(full_feat)
        res_sub = strat_m.generate_signals(full_feat.iloc[: cutoff + 1])

        # Giá trị tại nến cutoff khi tính trên bảng đầy đủ và bảng cắt ngắn phải khớp nhau 100%
        assert res_sub["signal"].iloc[-1] == res_full["signal"].iloc[cutoff], f"Look-ahead in {mode} signal"
        assert np.isclose(
            res_sub["stop_loss"].iloc[-1], res_full["stop_loss"].iloc[cutoff], equal_nan=True
        ), f"Look-ahead in {mode} stop_loss"
        assert np.isclose(
            res_sub["take_profit"].iloc[-1], res_full["take_profit"].iloc[cutoff], equal_nan=True
        ), f"Look-ahead in {mode} take_profit"
        assert np.isclose(
            res_sub["limit_price"].iloc[-1], res_full["limit_price"].iloc[cutoff], equal_nan=True
        ), f"Look-ahead in {mode} limit_price"

