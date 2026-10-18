"""Unit tests cho bộ lọc ML Meta-Labeling & Purged Walk-Forward (Sprint 2 - Phần B)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import run_backtest
from src.backtester import Backtester
from src.config import TradingConfig
from src.ml_filter import (
    ML_FEATURE_COLUMNS,
    MetaLabelFilter,
    PurgedWalkForwardSplit,
    build_meta_dataset,
    compute_walk_forward_probs,
    simulate_single_trade_outcome,
)

CFG = TradingConfig()


def _make_signal_frame(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="timestamp").as_unit("ns")
    close = 50000.0 + np.cumsum(rng.normal(0.0, 80.0, n))
    open_ = np.concatenate([[50000.0], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(20.0, 120.0, n)
    low = np.minimum(open_, close) - rng.uniform(20.0, 120.0, n)
    volume = rng.uniform(100.0, 500.0, n)

    df = pd.DataFrame(
        {
            "open": open_.astype("float64"),
            "high": high.astype("float64"),
            "low": low.astype("float64"),
            "close": close.astype("float64"),
            "volume": volume.astype("float64"),
            "atr": np.full(n, 200.0, dtype="float64"),
            "rsi": rng.uniform(30.0, 70.0, n).astype("float64"),
            "vol_zscore": rng.normal(1.8, 0.5, n).astype("float64"),
            "regime": rng.choice([-1, 1], size=n).astype("int8"),
            "body_to_range_ratio": rng.uniform(0.1, 0.8, n).astype("float64"),
            "upper_wick_ratio": rng.uniform(0.05, 0.4, n).astype("float64"),
            "lower_wick_ratio": rng.uniform(0.05, 0.4, n).astype("float64"),
            "bb_width_squeeze": rng.uniform(0.005, 0.05, n).astype("float64"),
            "return_lag_1": rng.normal(0.0, 0.002, n).astype("float64"),
            "return_lag_4": rng.normal(0.0, 0.004, n).astype("float64"),
            "return_lag_16": rng.normal(0.0, 0.008, n).astype("float64"),
            "hour_sin": np.sin(2.0 * np.pi * idx.hour.to_numpy(dtype="float64") / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * idx.hour.to_numpy(dtype="float64") / 24.0),
            "signal": np.zeros(n, dtype="int64"),
        },
        index=idx,
    )
    return df


def test_purged_walk_forward_split_respects_purge_and_embargo():
    # Kiểm tra validate tham số đầu vào
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(n_splits=0)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(n_splits=3, purge_bars=-1)
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(n_splits=3, purge_bars=5, embargo_bars=-2)

    n_samples = 240
    X = pd.DataFrame(np.zeros((n_samples, len(ML_FEATURE_COLUMNS))), columns=ML_FEATURE_COLUMNS)
    splitter = PurgedWalkForwardSplit(n_splits=5, purge_bars=12, embargo_bars=4)
    folds = splitter.split(X)

    assert len(folds) == 5
    prev_test_end = 0
    for train_idx, test_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert np.issubdtype(train_idx.dtype, np.integer)
        assert np.issubdtype(test_idx.dtype, np.integer)
        assert len(test_idx) > 0
        # Các tập test liên tiếp nhau theo đúng thứ tự thời gian, không giao nhau
        assert int(test_idx.min()) >= prev_test_end
        prev_test_end = int(test_idx.max()) + 1
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        if len(train_idx) > 0:
            assert int(train_idx.max()) + splitter.purge_bars < int(test_idx.min())
            assert (
                int(train_idx.max()) + splitter.purge_bars + splitter.embargo_bars
                < int(test_idx.min())
            )

    # Kiểm tra khi truyền bar_positions (vị trí nến trên chuỗi OHLCV)
    bar_positions = np.arange(n_samples, dtype=np.int64) * 5
    folds_pos = splitter.split(X, bar_positions=bar_positions)
    assert len(folds_pos) == 5
    for train_idx, test_idx in folds_pos:
        assert len(train_idx) > 0
        max_train_bar = int(bar_positions[train_idx].max())
        min_test_bar = int(bar_positions[test_idx].min())
        assert max_train_bar + splitter.purge_bars + splitter.embargo_bars < min_test_bar

    # Báo lỗi nếu số mẫu quá ít hoặc bar_positions không hợp lệ
    with pytest.raises(ValueError):
        splitter.split(X.iloc[:3])
    with pytest.raises(ValueError):
        splitter.split(X, bar_positions=bar_positions[::-1])


def test_meta_labeling_no_future_leakage_in_features():
    df = _make_signal_frame(n=40, seed=10)
    # Đặt 2 tín hiệu tại nến t=5 (Long thắng TP) và t=15 (Long thua SL)
    df.loc[df.index[5], "signal"] = 1
    df.loc[df.index[5], "rsi"] = 61.25
    # Nến 6: mở tại 50000, atr=200 -> entry ~ 50010, TP ~ 50610, SL ~ 49610 -> chạm TP
    df.loc[df.index[6], ["open", "high", "low", "close"]] = [50000.0, 50800.0, 49950.0, 50700.0]

    df.loc[df.index[15], "signal"] = 1
    df.loc[df.index[15], "rsi"] = 54.75
    # Nến 16: mở tại 50000, quét xuống SL=49500
    df.loc[df.index[16], ["open", "high", "low", "close"]] = [50000.0, 50050.0, 49400.0, 49500.0]

    # Tín hiệu ở nến cuối cùng (t=39) phải bị bỏ qua vì không có nến t+1
    df.loc[df.index[-1], "signal"] = 1

    X1, y1, exit_pos1 = build_meta_dataset(df, CFG)
    assert list(X1.columns) == ML_FEATURE_COLUMNS
    assert len(X1) == 2
    assert list(X1.index) == [df.index[5], df.index[15]]
    assert list(y1) == [1, 0]
    assert list(exit_pos1) == [6, 16]

    # Thay đổi dữ liệu tương lai sau nến phát tín hiệu t=5 không được làm đổi X[t=5]
    df_mutated = df.copy()
    df_mutated.iloc[6:, df_mutated.columns.get_loc("rsi")] = 99.9
    df_mutated.iloc[6:, df_mutated.columns.get_loc("vol_zscore")] = -99.0
    df_mutated.iloc[6:, df_mutated.columns.get_loc("return_lag_1")] = 0.5

    X2, _, _ = build_meta_dataset(df_mutated, CFG)
    np.testing.assert_allclose(
        X1.loc[df.index[5]].to_numpy(dtype="float64"),
        X2.loc[df.index[5]].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-15,
    )
    assert X2.loc[df.index[5], "rsi"] == pytest.approx(61.25)

    # Kiểm tra simulate_single_trade_outcome trực tiếp
    lbl_win, pos_win = simulate_single_trade_outcome(df, 5, CFG)
    lbl_loss, pos_loss = simulate_single_trade_outcome(df, 15, CFG)
    assert (lbl_win, pos_win) == (1, 6)
    assert (lbl_loss, pos_loss) == (0, 16)


def test_meta_label_filter_graceful_fallback_small_or_single_class():
    rng = np.random.default_rng(123)
    # 1) Tập train chỉ có 5 mẫu (< ml_min_train_samples=30) -> không crash, fallback hợp lệ
    X_small = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(5, len(ML_FEATURE_COLUMNS))),
        columns=ML_FEATURE_COLUMNS,
    )
    X_small["signal"] = 1.0
    y_small = pd.Series([0, 1, 0, 1, 1], dtype="int64")

    clf_small = MetaLabelFilter(CFG)
    clf_small.fit(X_small, y_small)
    assert clf_small.is_fitted_ is False
    probs_small = clf_small.predict_proba(X_small)
    assert probs_small.shape == (5,)
    assert np.all((probs_small >= 0.0) & (probs_small <= 1.0))
    assert isinstance(clf_small.should_take(X_small.iloc[0]), bool)

    # 2) Tập train đủ mẫu (>= 30) nhưng chỉ có 1 lớp duy nhất (toàn 0 hoặc toàn 1)
    X_single = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(40, len(ML_FEATURE_COLUMNS))),
        columns=ML_FEATURE_COLUMNS,
    )
    X_single["signal"] = 1.0

    clf_all_zero = MetaLabelFilter(CFG)
    clf_all_zero.fit(X_single, np.zeros(40, dtype=np.int64))
    assert clf_all_zero.is_fitted_ is False
    probs_zero = clf_all_zero.predict_proba(X_single.iloc[:4])
    assert probs_zero.shape == (4,)
    assert np.all((probs_zero >= 0.0) & (probs_zero <= 1.0))
    assert clf_all_zero.should_take(X_single.iloc[0]) is False

    clf_all_one = MetaLabelFilter(CFG)
    clf_all_one.fit(X_single, np.ones(40, dtype=np.int64))
    assert clf_all_one.is_fitted_ is False
    probs_one = clf_all_one.predict_proba(X_single.iloc[:4])
    assert np.all(probs_one == pytest.approx(1.0))
    assert clf_all_one.should_take(X_single.iloc[0]) is True

    # Khi signal == 0 thì should_take luôn trả về False
    flat_row = X_single.iloc[0].copy()
    flat_row["signal"] = 0.0
    assert clf_all_one.should_take(flat_row) is False


def test_backtester_with_use_ml_filters_low_probability_signals(capsys):
    # Tạo dữ liệu huấn luyện có quy luật rõ ràng: rsi > 55 -> thắng (y=1), rsi < 45 -> thua (y=0)
    rng = np.random.default_rng(99)
    n_train = 80
    X_train = pd.DataFrame(
        rng.normal(0.0, 0.1, size=(n_train, len(ML_FEATURE_COLUMNS))),
        columns=ML_FEATURE_COLUMNS,
    )
    X_train["signal"] = 1.0
    y_train = np.array([1 if i % 2 == 0 else 0 for i in range(n_train)], dtype=np.int64)
    X_train["rsi"] = np.where(y_train == 1, 65.0, 35.0)
    X_train["vol_zscore"] = np.where(y_train == 1, 3.0, 1.0)

    cfg_ml = replace(CFG, use_ml=True, ml_prob_threshold=0.55)
    ml_filter = MetaLabelFilter(cfg_ml, random_state=42).fit(X_train, y_train)
    assert ml_filter.is_fitted_ is True

    # Xây dựng chuỗi 6 nến có 2 tín hiệu LONG:
    # - Tín hiệu 1 ở bar 0: rsi=35, vol_zscore=1.0 -> prob_win thấp (< 0.55) -> bị ML từ chối!
    # - Tín hiệu 2 ở bar 3: rsi=65, vol_zscore=3.0 -> prob_win cao (>= 0.55) -> được ML duyệt!
    df_bt = _make_signal_frame(n=6, seed=7)
    for i in range(6):
        df_bt.iloc[i, df_bt.columns.get_loc("open")] = 50000.0
        df_bt.iloc[i, df_bt.columns.get_loc("high")] = 50100.0
        df_bt.iloc[i, df_bt.columns.get_loc("low")] = 49900.0
        df_bt.iloc[i, df_bt.columns.get_loc("close")] = 50000.0
        df_bt.iloc[i, df_bt.columns.get_loc("atr")] = 500.0
        df_bt.iloc[i, df_bt.columns.get_loc("signal")] = 0

    # Tín hiệu 1 (xác suất thấp) tại bar 0; nếu khớp ở bar 1 sẽ chạm TP tại high=51600
    for col in ML_FEATURE_COLUMNS:
        df_bt.iloc[0, df_bt.columns.get_loc(col)] = 0.0
    df_bt.iloc[0, df_bt.columns.get_loc("signal")] = 1
    df_bt.iloc[0, df_bt.columns.get_loc("rsi")] = 35.0
    df_bt.iloc[0, df_bt.columns.get_loc("vol_zscore")] = 1.0
    df_bt.iloc[1, df_bt.columns.get_loc("high")] = 51600.0

    # Tín hiệu 2 (xác suất cao) tại bar 3; khớp ở bar 4 và chạm TP tại high=51600
    for col in ML_FEATURE_COLUMNS:
        df_bt.iloc[3, df_bt.columns.get_loc(col)] = 0.0
    df_bt.iloc[3, df_bt.columns.get_loc("signal")] = 1
    df_bt.iloc[3, df_bt.columns.get_loc("rsi")] = 65.0
    df_bt.iloc[3, df_bt.columns.get_loc("vol_zscore")] = 3.0
    df_bt.iloc[4, df_bt.columns.get_loc("high")] = 51600.0

    probs_check = ml_filter.predict_proba(df_bt.iloc[[0, 3]])
    assert probs_check[0] < 0.55
    assert probs_check[1] >= 0.55

    class _PassthroughStrategy:
        def generate_signals(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame.copy()

    # Khi use_ml=False: cả 2 tín hiệu đều được vào lệnh -> 2 giao dịch
    rep_no_ml = Backtester(replace(CFG, use_ml=False), strategy=_PassthroughStrategy()).run(df_bt)
    assert rep_no_ml.total_trades == 2

    # Khi use_ml=True kèm ml_filter: chỉ tín hiệu 2 (prob >= 0.55) được mở tại bar 4 -> 1 giao dịch
    rep_with_ml = Backtester(cfg_ml, strategy=_PassthroughStrategy(), ml_filter=ml_filter).run(df_bt)
    assert rep_with_ml.total_trades == 1
    assert rep_with_ml.trades[0].entry_time == df_bt.index[4]

    # Kiểm tra compute_walk_forward_probs và CLI run_backtest.py --synthetic --use-ml
    wf_probs = compute_walk_forward_probs(df_bt, cfg_ml)
    assert len(wf_probs) == len(df_bt)

    exit_code = run_backtest.main(["--synthetic", "--use-ml"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ML Meta-Filter" in captured.out
    assert "ON" in captured.out
