import numpy as np
import pandas as pd
import pytest

from src.config import TradingConfig
from src.features import (
    FEATURE_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    add_all_features,
    compute_atr,
    compute_bb_width_squeeze,
    compute_candle_microstructure,
    compute_donchian,
    compute_ema,
    compute_regime_filter,
    compute_return_lags,
    compute_rsi,
    compute_time_cyclical,
    compute_volume_zscore,
)


def _utc_ns_index(
    periods: int, start: str = "2024-01-01", freq: str = "15min", name: str | None = None
) -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC độ phân giải nanoseconds (pandas 3.x mặc định trả về 'us')."""
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC", name=name).as_unit("ns")


def make_ohlcv(n: int, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0.0, 0.005, n)))
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.003, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.003, n))
    volume = rng.uniform(100.0, 200.0, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="timestamp").as_unit("ns")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def wilder_reference(values: np.ndarray, period: int) -> np.ndarray:
    """Tham chiếu RMA bằng vòng lặp: giá trị đầu tiên là seed, min_periods = period."""
    out = np.full(len(values), np.nan)
    prev = values[0]
    for i in range(len(values)):
        prev = values[i] if i == 0 else (1.0 / period) * values[i] + (1 - 1.0 / period) * prev
        if i >= period - 1:
            out[i] = prev
    return out


# ---------------------------------------------------------------- formulas

def test_ema_formula():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    ema = compute_ema(s, 3)
    alpha = 2.0 / 4.0
    expected = [1.0]
    for v in s.iloc[1:]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])
    assert ema.iloc[:2].isna().all()
    np.testing.assert_allclose(ema.iloc[2:].to_numpy(), expected[2:])


def test_atr_wilder_formula():
    df = make_ohlcv(60, seed=1)
    atr = compute_atr(df, 14)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    expected = wilder_reference(tr.to_numpy(), 14)
    assert atr.iloc[:13].isna().all()
    np.testing.assert_allclose(atr.to_numpy()[13:], expected[13:], rtol=1e-10)


def test_atr_constant_range():
    n = 40
    idx = _utc_ns_index(n)
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx
    )
    atr = compute_atr(df, 14).dropna()
    np.testing.assert_allclose(atr.to_numpy(), 2.0)


def test_rsi_wilder_formula():
    df = make_ohlcv(80, seed=2)
    rsi = compute_rsi(df["close"], 14)
    delta = df["close"].diff().to_numpy()
    gain = np.where(np.isnan(delta), np.nan, np.clip(delta, 0, None))
    loss = np.where(np.isnan(delta), np.nan, np.clip(-delta, 0, None))
    # diff đầu là NaN -> bỏ qua, seed từ phần tử 1
    ag = wilder_reference(gain[1:], 14)
    al = wilder_reference(loss[1:], 14)
    expected = 100.0 - 100.0 / (1.0 + ag / al)
    got = rsi.to_numpy()[1:]
    mask = ~np.isnan(expected)
    assert np.isnan(rsi.iloc[:14]).all()
    np.testing.assert_allclose(got[mask], expected[mask], rtol=1e-9)


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 50.0))
    assert compute_rsi(up, 14).dropna().eq(100.0).all()
    down = pd.Series(np.arange(50.0, 1.0, -1.0))
    assert np.allclose(compute_rsi(down, 14).dropna(), 0.0)
    flat = pd.Series(np.full(40, 10.0))
    assert np.allclose(compute_rsi(flat, 14).dropna(), 50.0)


def test_donchian_boundary_excludes_current_bar():
    df = make_ohlcv(100, seed=3)
    window = 20
    dc = compute_donchian(df, window)
    for t in range(window, len(df)):
        assert dc["donchian_upper"].iloc[t] == df["high"].iloc[t - window : t].max()
        assert dc["donchian_lower"].iloc[t] == df["low"].iloc[t - window : t].min()
    assert dc.iloc[:window].isna().all().all()
    # Spike cực đoan ở nến t không được ảnh hưởng đến kênh tại t
    df2 = df.copy()
    df2.loc[df2.index[50], "high"] = 1e9
    dc2 = compute_donchian(df2, window)
    assert dc2["donchian_upper"].iloc[50] == dc["donchian_upper"].iloc[50]
    assert dc2["donchian_upper"].iloc[51] == 1e9
    np.testing.assert_allclose(
        dc["donchian_mid"].dropna(),
        0.5 * (dc["donchian_upper"] + dc["donchian_lower"]).dropna(),
    )


def test_volume_zscore_formula():
    df = make_ohlcv(50, seed=4)
    z = compute_volume_zscore(df, 20)
    w = df["volume"].iloc[10:30]
    expected = (df["volume"].iloc[29] - w.mean()) / w.std(ddof=1)
    assert z.iloc[29] == pytest.approx(expected)
    assert z.iloc[:19].isna().all()


def test_volume_zscore_zero_std_is_nan():
    idx = _utc_ns_index(30)
    df = pd.DataFrame({"volume": np.full(30, 5.0)}, index=idx)
    z = compute_volume_zscore(df, 20)
    assert z.isna().all()
    assert not np.isinf(z.to_numpy()).any()


def test_regime_filter_states():
    n = 400
    idx = _utc_ns_index(n)
    up = pd.DataFrame({"close": np.linspace(100.0, 500.0, n)}, index=idx)
    down = pd.DataFrame({"close": np.linspace(500.0, 100.0, n)}, index=idx)
    flat = pd.DataFrame({"close": np.full(n, 100.0)}, index=idx)
    r_up = compute_regime_filter(up)
    r_down = compute_regime_filter(down)
    r_flat = compute_regime_filter(flat)
    assert r_up.dtype == np.int8
    assert r_up.iloc[-1] == 1
    assert r_down.iloc[-1] == -1
    assert (r_flat == 0).all()
    assert set(r_up.unique()) <= {-1, 0, 1}
    assert (r_up.iloc[:199] == 0).all()


# ----------------------------------------------------------- add_all_features

def test_add_all_features_columns_and_warmup():
    df = make_ohlcv(600, seed=5)
    original = df.copy()
    cfg = TradingConfig()
    full = add_all_features(df, cfg, drop_warmup=False)
    for col in FEATURE_COLUMNS:
        assert col in full.columns
    assert len(full) == len(df)
    pd.testing.assert_frame_equal(df, original)  # không sửa df đầu vào

    dropped = add_all_features(df, cfg, drop_warmup=True)
    assert len(dropped) < len(df)
    assert not dropped.isna().any().any()
    assert not dropped.iloc[0][FEATURE_COLUMNS].isna().any()


# ------------------------------------------------------- No Look-Ahead Bias

def make_extreme_future(T: int = 500, n_future: int = 50, seed: int = 7) -> pd.DataFrame:
    base = make_ohlcv(T, seed=seed)
    rng = np.random.default_rng(seed + 1)
    last = base["close"].iloc[-1]
    # 25 nến pump x10 rồi 25 nến crash về 1/10
    path = np.concatenate(
        [np.linspace(last, last * 10.0, n_future // 2), np.linspace(last * 10.0, last * 0.1, n_future - n_future // 2)]
    )
    prev = np.concatenate([[last], path[:-1]])
    high = np.maximum(prev, path) * (1.0 + rng.uniform(0.0, 0.2, n_future))
    low = np.minimum(prev, path) * (1.0 - rng.uniform(0.0, 0.2, n_future))
    volume = rng.uniform(1000.0, 5000.0, n_future)  # ~10-30x volume
    idx = pd.date_range(base.index[-1] + pd.Timedelta(minutes=15), periods=n_future, freq="15min", tz="UTC", name="timestamp").as_unit("ns")
    fut = pd.DataFrame({"open": prev, "high": high, "low": low, "close": path, "volume": volume}, index=idx)
    return pd.concat([base, fut])


def test_no_look_ahead_bias():
    T = 500
    df_full = make_extreme_future(T=T, n_future=50)
    assert len(df_full) == T + 50
    df_sub = df_full.iloc[:T]

    feat_sub = add_all_features(df_sub, drop_warmup=False)
    feat_full = add_all_features(df_full, drop_warmup=False)

    for col in FEATURE_COLUMNS:
        a = feat_sub[col].iloc[:T].to_numpy(dtype="float64")
        b = feat_full[col].iloc[:T].to_numpy(dtype="float64")
        assert np.allclose(a, b, equal_nan=True), f"Look-ahead bias ở cột {col}"
        # NaN phải khớp vị trí tuyệt đối
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"Vị trí NaN khác nhau ở cột {col}"

    # Tổng thể trên các dòng hợp lệ
    assert np.allclose(
        feat_sub.iloc[:T][FEATURE_COLUMNS].dropna().to_numpy(dtype="float64"),
        feat_full.iloc[:T][FEATURE_COLUMNS].dropna().to_numpy(dtype="float64"),
        equal_nan=True,
    )

    # Sanity: phần tương lai thực sự cực đoan (test có tác dụng)
    assert feat_full["atr"].iloc[T:].max() > 5 * feat_full["atr"].iloc[T - 1]


# ------------------------------------------------------------------
# [LOW-01] drop_warmup không được xóa nến giữa chuỗi
# ------------------------------------------------------------------

def test_drop_warmup_keeps_contiguous_index_with_constant_volume_mid_series():
    n = 600
    df = make_ohlcv(n, seed=11)
    df.iloc[300:330, df.columns.get_loc("volume")] = 150.0  # >=20 nến volume hằng -> vol_zscore NaN
    cfg = TradingConfig()

    full = add_all_features(df, cfg, drop_warmup=False)
    assert full["vol_zscore"].iloc[325:330].isna().all()  # tiền đề của test

    out = add_all_features(df, cfg, drop_warmup=True)

    # Chỉ cắt warmup ở đầu: phần còn lại là đuôi liên tục của chuỗi gốc
    assert out.index[-1] == df.index[-1]
    assert out.index.equals(df.index[len(df) - len(out):])
    assert (out.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert out.index[0] <= df.index[250]  # warmup chỉ vài trăm nến đầu
    assert all(ts in out.index for ts in df.index[300:330])

    # Không còn NaN ở toàn bộ bảng sau warmup; đoạn volume hằng giữa chuỗi được điền vol_zscore = 0.0
    assert not out[FEATURE_COLUMNS].isna().any().any()
    assert (out.loc[df.index[319:330], "vol_zscore"] == 0.0).all()


# ------------------------------------------------------------------
# Sprint 2 - Phần A1: 9 đặc trưng vi cấu trúc & chu kỳ
# ------------------------------------------------------------------

def test_candle_microstructure_ratios_and_flat_candle():
    df = make_ohlcv(50, seed=21)
    # Chèn nến Doji phẳng (high == low) ở dòng 10 và 20
    df.iloc[10] = [100.0, 100.0, 100.0, 100.0, 150.0]
    df.iloc[20] = [250.0, 250.0, 250.0, 250.0, 120.0]

    micro = compute_candle_microstructure(df)
    cols = ["body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio"]
    assert list(micro.columns) == cols
    assert not micro.isna().any().any()
    assert not np.isinf(micro.to_numpy()).any()

    # Mọi tỷ lệ nằm trong [0, 1]
    assert (micro.to_numpy() >= 0.0).all()
    assert (micro.to_numpy() <= 1.0 + 1e-12).all()

    # Khi high == low -> cả 3 cột bằng 0.0
    for idx_pos in (10, 20):
        np.testing.assert_allclose(micro.iloc[idx_pos].to_numpy(), [0.0, 0.0, 0.0], atol=1e-15)

    # Khi high > low -> tổng 3 tỷ lệ bằng 1.0
    non_flat = (df["high"] > df["low"]).to_numpy()
    row_sums = micro.loc[non_flat].sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-12, atol=1e-12)


def test_bb_width_squeeze_formula_and_window():
    df = make_ohlcv(60, seed=22)
    sq = compute_bb_width_squeeze(df, bb_window=20, bb_std_mult=2.0)
    assert sq.name == "bb_width_squeeze"
    assert sq.iloc[:19].isna().all()
    assert sq.iloc[19:].notna().all()

    close = df["close"].astype("float64")
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=1)
    expected = ((mid + 2.0 * std) - (mid - 2.0 * std)) / mid
    np.testing.assert_allclose(sq.iloc[19:].to_numpy(), expected.iloc[19:].to_numpy(), rtol=1e-12)


def test_return_lags_1_4_16_and_cyclical_hours():
    df = make_ohlcv(100, seed=23)
    lags = compute_return_lags(df["close"], lags=(1, 4, 16))
    assert list(lags.columns) == ["return_lag_1", "return_lag_4", "return_lag_16"]

    for k in (1, 4, 16):
        col = f"return_lag_{k}"
        assert lags[col].iloc[:k].isna().all()
        expected = df["close"].iloc[k:].to_numpy() / df["close"].iloc[:-k].to_numpy() - 1.0
        np.testing.assert_allclose(lags[col].iloc[k:].to_numpy(), expected, rtol=1e-12)

    cyc = compute_time_cyclical(df.index)
    assert list(cyc.columns) == ["hour_sin", "hour_cos"]
    assert not cyc.isna().any().any()
    norm_sq = cyc["hour_sin"].to_numpy() ** 2 + cyc["hour_cos"].to_numpy() ** 2
    np.testing.assert_allclose(norm_sq, 1.0, rtol=1e-12, atol=1e-12)
    # 00:00 UTC -> sin = 0, cos = 1; 06:00 UTC (nến thứ 24 khung 15m) -> sin = 1, cos = 0
    assert cyc["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert cyc["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert cyc["hour_sin"].iloc[24] == pytest.approx(1.0, abs=1e-12)
    assert cyc["hour_cos"].iloc[24] == pytest.approx(0.0, abs=1e-12)


def test_new_features_no_look_ahead_bias():
    T = 500
    df_full = make_extreme_future(T=T, n_future=50, seed=31)
    df_sub = df_full.iloc[:T]

    feat_sub = add_all_features(df_sub, drop_warmup=False)
    feat_full = add_all_features(df_full, drop_warmup=False)

    assert len(MICROSTRUCTURE_COLUMNS) == 9
    for col in MICROSTRUCTURE_COLUMNS:
        assert col in feat_sub.columns
        a = feat_sub[col].iloc[:T].to_numpy(dtype="float64")
        b = feat_full[col].iloc[:T].to_numpy(dtype="float64")
        assert np.allclose(a, b, equal_nan=True), f"Look-ahead bias ở cột mới {col}"
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"Vị trí NaN khác nhau ở cột mới {col}"



def test_adx_bollinger_pct_b_and_ema20_formulas_and_no_lookahead():
    from src.features import (
        REGIME_STRATEGY_COLUMNS,
        compute_adx,
        compute_bollinger_bands,
        compute_ema_slope,
    )

    df = make_ohlcv(400, seed=5)
    cfg = TradingConfig()

    # ---- ADX / +DI / -DI chuẩn Wilder ----
    adx_df = compute_adx(df, 14)
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1).to_numpy()
    up = np.concatenate([[np.nan], high[1:] - high[:-1]])
    down = np.concatenate([[np.nan], low[:-1] - low[1:]])
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)
    atr_ref = wilder_reference(tr, 14)
    pdi_ref = 100.0 * wilder_reference(plus_dm, 14) / atr_ref
    mdi_ref = 100.0 * wilder_reference(minus_dm, 14) / atr_ref
    dx_ref = 100.0 * np.abs(pdi_ref - mdi_ref) / (pdi_ref + mdi_ref)
    adx_ref = np.full(len(df), np.nan)
    adx_ref[13:] = wilder_reference(dx_ref[13:], 14)
    np.testing.assert_allclose(adx_df["plus_di"].to_numpy(), pdi_ref, equal_nan=True, rtol=1e-9)
    np.testing.assert_allclose(adx_df["minus_di"].to_numpy(), mdi_ref, equal_nan=True, rtol=1e-9)
    np.testing.assert_allclose(adx_df["adx"].to_numpy(), adx_ref, equal_nan=True, rtol=1e-9)
    valid_adx = adx_df["adx"].dropna()
    assert ((valid_adx >= 0.0) & (valid_adx <= 100.0)).all()

    # ---- Bollinger %B ----
    bb = compute_bollinger_bands(df, 20, 2.0)
    close = df["close"]
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std(ddof=1)
    np.testing.assert_allclose(bb["bb_mid"].to_numpy(), mid.to_numpy(), equal_nan=True)
    np.testing.assert_allclose(bb["bb_upper"].to_numpy(), (mid + 2.0 * sd).to_numpy(), equal_nan=True)
    np.testing.assert_allclose(bb["bb_lower"].to_numpy(), (mid - 2.0 * sd).to_numpy(), equal_nan=True)
    expected_pct_b = (close - (mid - 2.0 * sd)) / (4.0 * sd)
    np.testing.assert_allclose(bb["bb_pct_b"].to_numpy(), expected_pct_b.to_numpy(), equal_nan=True)
    assert bb["bb_pct_b"].iloc[:19].isna().all()
    # Series đầu vào cho kết quả giống DataFrame
    bb_s = compute_bollinger_bands(close, 20, 2.0)
    np.testing.assert_allclose(bb_s["bb_pct_b"].to_numpy(), bb["bb_pct_b"].to_numpy(), equal_nan=True)

    # ---- EMA20 & slope ----
    feats = add_all_features(df, cfg, drop_warmup=False)
    for col in REGIME_STRATEGY_COLUMNS:
        assert col in feats.columns
    np.testing.assert_allclose(
        feats["ema_20"].to_numpy(), compute_ema(df["close"], 20).to_numpy(), equal_nan=True
    )
    slow = compute_ema(df["close"], cfg.ema_slow_period)
    expected_slope = (slow - slow.shift(5)) / slow.shift(5)
    np.testing.assert_allclose(
        feats["ema_slow_slope"].to_numpy(), expected_slope.to_numpy(), equal_nan=True
    )
    assert compute_ema_slope(pd.Series([0.0, 0.0, 0.0]), 1).iloc[1:].isna().all()

    # ---- Số nến warmup giữ nguyên + không NaN/inf ----
    trimmed = add_all_features(df, cfg, drop_warmup=True)
    first_pos = df.index.get_loc(trimmed.index[0])
    assert first_pos == cfg.ema_slow_period - 1 + cfg.regime_slope_lookback
    for col in REGIME_STRATEGY_COLUMNS:
        arr = trimmed[col].to_numpy(dtype="float64")
        assert np.isfinite(arr).all(), col

    # ---- Nến phẳng ----
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="timestamp").as_unit("ns")
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 10.0}, index=idx
    )
    adx_flat = compute_adx(flat, 14)
    bb_flat = compute_bollinger_bands(flat, 20, 2.0)
    assert (adx_flat["adx"].iloc[27:] == 0.0).all()
    assert (adx_flat["plus_di"].iloc[13:] == 0.0).all()
    assert (bb_flat["bb_pct_b"].iloc[19:] == 0.5).all()
    for frame in (adx_flat, bb_flat):
        assert not np.isinf(frame.to_numpy(dtype="float64")).any()

    # ---- No look-ahead: df.iloc[:T] vs df đầy đủ ----
    T = 300
    full = make_ohlcv(360, seed=9)
    f_sub = add_all_features(full.iloc[:T], cfg, drop_warmup=False)
    f_full = add_all_features(full, cfg, drop_warmup=False)
    for col in REGIME_STRATEGY_COLUMNS:
        a = f_sub[col].to_numpy(dtype="float64")
        b = f_full[col].iloc[:T].to_numpy(dtype="float64")
        assert np.allclose(a, b, equal_nan=True), f"Look-ahead bias ở cột {col}"
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"Vị trí NaN khác nhau ở cột {col}"
