"""Bộ lọc xác suất Meta-Labeling (HistGradientBoostingClassifier) & Purged Walk-Forward Split."""
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src import features
from src.config import TradingConfig
from src.risk_manager import Position
from src.strategy import compute_breakeven_price, is_breakeven_triggered

ML_FEATURE_COLUMNS: list[str] = [
    "rsi",
    "vol_zscore",
    "regime",
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "bb_width_squeeze",
    "return_lag_1",
    "return_lag_4",
    "return_lag_16",
    "hour_sin",
    "hour_cos",
    "signal",
]


def _ensure_ml_feature_columns(df: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
    """Đảm bảo DataFrame có đủ ML_FEATURE_COLUMNS mà không làm thay đổi các cột đã có hay độ dài bảng."""
    if all(col in df.columns for col in ML_FEATURE_COLUMNS):
        return df

    out = df.copy()
    missing = [col for col in ML_FEATURE_COLUMNS if col not in out.columns]
    if any(col != "signal" for col in missing) and all(
        c in out.columns for c in ("open", "high", "low", "close", "volume")
    ):
        computed = features.add_all_features(out, config, drop_warmup=False)
        for col in missing:
            if col in computed.columns:
                out[col] = computed[col]

    for col in ML_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out


def simulate_single_trade_outcome(
    df: pd.DataFrame,
    signal_pos: int,
    config: TradingConfig = TradingConfig(),
) -> tuple[int, int]:
    """Mô phỏng độc lập 1 lệnh phát sinh từ tín hiệu tại vị trí dòng `signal_pos` (nến t).

    Khớp tại `open[t + 1]` (kèm slippage + phí taker 2 chiều) và theo dõi cho tới khi chạm
    SL, TP, Breakeven SL (hiệu lực từ nến sau), hết `max_holding_bars` hoặc hết dữ liệu.

    Trả về:
        (label_y, exit_bar_pos):
          - label_y: 1 nếu net_pnl > 0.0, ngược lại 0.
          - exit_bar_pos: chỉ số dòng (0-based) nơi lệnh đóng thực tế.
    """
    n = len(df)
    if n == 0:
        return 0, 0
    if signal_pos < 0 or signal_pos + 1 >= n:
        clamped = min(max(0, int(signal_pos)), n - 1)
        return 0, clamped

    sig_val = df["signal"].iloc[signal_pos] if "signal" in df.columns else 0
    if pd.isna(sig_val) or int(sig_val) not in (1, -1):
        return 0, int(signal_pos)
    side: int = int(sig_val)

    entry_bar = int(signal_pos) + 1
    opens = df["open"].to_numpy(dtype="float64")
    highs = df["high"].to_numpy(dtype="float64")
    lows = df["low"].to_numpy(dtype="float64")
    closes = df["close"].to_numpy(dtype="float64")

    if "atr" in df.columns:
        atr_prev = float(df["atr"].iloc[signal_pos])
    else:
        atr_series = features.compute_atr(df, config.atr_period)
        atr_prev = float(atr_series.iloc[signal_pos])

    open_px = float(opens[entry_bar])
    if not np.isfinite(atr_prev) or atr_prev <= 0.0 or not np.isfinite(open_px) or open_px <= 0.0:
        return 0, entry_bar

    if side == 1:
        entry = open_px * (1.0 + config.slippage)
        sl = entry - config.sl_atr_multiple * atr_prev
        tp = entry + config.tp_atr_multiple * atr_prev
    else:
        entry = open_px * (1.0 - config.slippage)
        sl = entry + config.sl_atr_multiple * atr_prev
        tp = entry - config.tp_atr_multiple * atr_prev

    entry_ts = (
        pd.Timestamp(df.index[entry_bar])
        if isinstance(df.index, pd.DatetimeIndex)
        else pd.Timestamp("2024-01-01", tz="UTC")
    )
    pos = Position(
        symbol=config.symbol,
        side=1 if side == 1 else -1,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        qty=1.0,
        entry_time=entry_ts,
        entry_fee=entry * config.taker_fee,
        notional=entry,
        initial_sl=sl,
        breakeven_activated=False,
        bars_held=0,
    )

    raw_exit = float(closes[entry_bar])
    exit_bar_pos = entry_bar

    for k in range(entry_bar, n):
        opened_this_bar = k == entry_bar

        # c) Kiểm tra SL/TP trong [low[k], high[k]] (SL ưu tiên, xử lý gap qua open[k])
        if pos.side == 1:
            hit_sl = lows[k] <= pos.stop_loss
            hit_tp = highs[k] >= pos.take_profit
            raw_sl = pos.stop_loss if opened_this_bar else min(opens[k], pos.stop_loss)
            raw_tp = pos.take_profit if opened_this_bar else max(opens[k], pos.take_profit)
        else:
            hit_sl = highs[k] >= pos.stop_loss
            hit_tp = lows[k] <= pos.take_profit
            raw_sl = pos.stop_loss if opened_this_bar else max(opens[k], pos.stop_loss)
            raw_tp = pos.take_profit if opened_this_bar else min(opens[k], pos.take_profit)

        if hit_sl:
            raw_exit = float(raw_sl)
            exit_bar_pos = k
            break
        if hit_tp:
            raw_exit = float(raw_tp)
            exit_bar_pos = k
            break

        # d) Cập nhật Breakeven Stop cho nến k+1
        if config.use_breakeven and not pos.breakeven_activated:
            if is_breakeven_triggered(pos, float(highs[k]), float(lows[k]), config):
                be_sl = compute_breakeven_price(pos.entry_price, pos.side, config)
                if pos.side == 1:
                    pos.stop_loss = max(pos.stop_loss, be_sl)
                else:
                    pos.stop_loss = min(pos.stop_loss, be_sl)
                pos.breakeven_activated = True

        # e) Kiểm tra Time-based Exit tại close[k]
        pos.bars_held += 1
        if config.max_holding_bars > 0 and pos.bars_held >= config.max_holding_bars:
            raw_exit = float(closes[k])
            exit_bar_pos = k
            break

        # f) Đóng cưỡng bức ở nến cuối
        if k == n - 1:
            raw_exit = float(closes[k])
            exit_bar_pos = k
            break

    exit_price = (
        raw_exit * (1.0 - config.slippage)
        if pos.side == 1
        else raw_exit * (1.0 + config.slippage)
    )
    gross_pnl = pos.qty * (exit_price - pos.entry_price) * pos.side
    total_fee = pos.entry_fee + pos.qty * exit_price * config.taker_fee
    net_pnl = gross_pnl - total_fee
    label_y = 1 if net_pnl > 0.0 else 0
    return label_y, int(exit_bar_pos)


def build_meta_dataset(
    sig_df: pd.DataFrame,
    config: TradingConfig = TradingConfig(),
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Xây dựng tập dữ liệu Meta-Labeling (X, y, exit_positions) không rò rỉ dữ liệu tương lai.

    Lọc tất cả các dòng t có `sig_df["signal"] != 0` và `t + 1 < len(sig_df)`.
    Vector đặc trưng X[t] chỉ lấy giá trị tại ĐÚNG dòng t (`sig_df.iloc[t][ML_FEATURE_COLUMNS]`).
    """
    if len(sig_df) < 2 or "signal" not in sig_df.columns:
        empty_idx = sig_df.index[0:0]
        return (
            pd.DataFrame(columns=ML_FEATURE_COLUMNS, index=empty_idx, dtype="float64"),
            pd.Series(dtype="int64", index=empty_idx, name="label"),
            pd.Series(dtype="int64", index=empty_idx, name="exit_bar_pos"),
        )

    work_df = _ensure_ml_feature_columns(sig_df, config)
    signals = work_df["signal"].to_numpy()
    valid_positions = [
        t
        for t in range(len(work_df) - 1)
        if not pd.isna(signals[t]) and int(signals[t]) in (1, -1)
    ]

    if not valid_positions:
        empty_idx = work_df.index[0:0]
        return (
            pd.DataFrame(columns=ML_FEATURE_COLUMNS, index=empty_idx, dtype="float64"),
            pd.Series(dtype="int64", index=empty_idx, name="label"),
            pd.Series(dtype="int64", index=empty_idx, name="exit_bar_pos"),
        )

    X = work_df.iloc[valid_positions][ML_FEATURE_COLUMNS].astype("float64").copy()
    labels = np.empty(len(valid_positions), dtype=np.int64)
    exits = np.empty(len(valid_positions), dtype=np.int64)

    for idx_i, pos_t in enumerate(valid_positions):
        lbl, exit_pos = simulate_single_trade_outcome(work_df, pos_t, config)
        labels[idx_i] = lbl
        exits[idx_i] = exit_pos

    y = pd.Series(labels, index=X.index, dtype="int64", name="label")
    exit_positions = pd.Series(exits, index=X.index, dtype="int64", name="exit_bar_pos")
    return X, y, exit_positions


build_meta_labels = build_meta_dataset


class PurgedWalkForwardSplit:
    """Chia chuỗi thời gian theo Expanding Window Walk-Forward có Purge và Embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 48,
        embargo_bars: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits phải >= 1 (nhận {n_splits}).")
        if purge_bars < 0:
            raise ValueError(f"purge_bars phải >= 0 (nhận {purge_bars}).")
        if embargo_bars < 0:
            raise ValueError(f"embargo_bars phải >= 0 (nhận {embargo_bars}).")
        self.n_splits = int(n_splits)
        self.purge_bars = int(purge_bars)
        self.embargo_bars = int(embargo_bars)

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        bar_positions: Optional[np.ndarray | pd.Series | Sequence[int]] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Chia chuỗi mẫu theo thứ tự thời gian thành `n_splits` cặp (train_idx, test_idx) nguyên."""
        n_samples = len(X)
        if n_samples < self.n_splits + 1:
            raise ValueError(
                f"Số lượng mẫu ({n_samples}) phải >= n_splits + 1 ({self.n_splits + 1})."
            )

        pos_arr: Optional[np.ndarray] = None
        if bar_positions is not None:
            pos_arr = np.asarray(bar_positions, dtype=np.int64)
            if pos_arr.ndim != 1 or len(pos_arr) != n_samples:
                raise ValueError("bar_positions phải là mảng 1 chiều có cùng độ dài với X.")
            if n_samples > 1 and np.any(np.diff(pos_arr) < 0):
                raise ValueError("bar_positions phải được sắp xếp tăng dần theo thời gian.")

        block_size = n_samples // (self.n_splits + 1)
        effective_gap = self.purge_bars + self.embargo_bars
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for k in range(self.n_splits):
            s_test = (k + 1) * block_size
            e_test = (k + 2) * block_size if k < self.n_splits - 1 else n_samples
            test_idx = np.arange(s_test, e_test, dtype=np.int64)

            if pos_arr is None:
                train_end = max(0, s_test - effective_gap)
                train_idx = np.arange(0, train_end, dtype=np.int64)
            else:
                candidates = np.arange(0, s_test, dtype=np.int64)
                mask = (pos_arr[candidates] + effective_gap < pos_arr[s_test]) & (
                    candidates + effective_gap < s_test
                    if np.array_equal(pos_arr, np.arange(n_samples, dtype=np.int64))
                    else pos_arr[candidates] + effective_gap < pos_arr[s_test]
                )
                train_idx = candidates[mask]

            splits.append((train_idx, test_idx))

        return splits


class MetaLabelFilter:
    """Bộ lọc xác suất Meta-Labeling dùng HistGradientBoostingClassifier có graceful fallback."""

    def __init__(
        self,
        config: TradingConfig = TradingConfig(),
        prob_threshold: Optional[float] = None,
        random_state: int = 42,
    ) -> None:
        self.config = config
        self.prob_threshold: float = (
            float(prob_threshold)
            if prob_threshold is not None
            else float(config.ml_prob_threshold)
        )
        self.random_state = int(random_state)
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=self.random_state,
        )
        self.is_fitted_: bool = False
        self.fallback_prob_: float = 1.0

    @staticmethod
    def _extract_feature_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if all(col in X.columns for col in ML_FEATURE_COLUMNS):
                return X[ML_FEATURE_COLUMNS].to_numpy(dtype="float64")
            work = X.copy()
            for col in ML_FEATURE_COLUMNS:
                if col not in work.columns:
                    work[col] = 0.0
            return work[ML_FEATURE_COLUMNS].to_numpy(dtype="float64")
        arr = np.asarray(X, dtype="float64")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "MetaLabelFilter":
        """Huấn luyện mô hình. Tự động fallback an toàn khi ít mẫu hoặc chỉ có 1 lớp."""
        X_mat = self._extract_feature_matrix(X)
        y_arr = np.asarray(y, dtype="float64").ravel()
        valid_y = y_arr[~np.isnan(y_arr)].astype(np.int64)
        unique_classes = np.unique(valid_y) if len(valid_y) > 0 else np.array([], dtype=np.int64)

        min_samples = max(2, int(self.config.ml_min_train_samples))
        if len(X_mat) < min_samples or len(valid_y) < min_samples or len(unique_classes) < 2:
            self.is_fitted_ = False
            if len(unique_classes) == 1:
                self.fallback_prob_ = 1.0 if int(unique_classes[0]) == 1 else 0.5
            else:
                self.fallback_prob_ = 1.0
            return self

        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=min(10, max(1, len(X_mat) // 4)),
            random_state=self.random_state,
        )
        self.model.fit(X_mat, y_arr.astype(np.int64))
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Dự đoán xác suất thắng P(y=1 | X). Trả mảng 1D float64 độ dài len(X)."""
        X_mat = self._extract_feature_matrix(X)
        n = len(X_mat)
        if n == 0:
            return np.empty(0, dtype="float64")
        if not self.is_fitted_:
            return np.full(n, self.fallback_prob_, dtype="float64")
        probs = self.model.predict_proba(X_mat)[:, 1]
        return np.asarray(probs, dtype="float64")

    def should_take(self, signal_row: pd.Series | pd.DataFrame | dict[str, Any]) -> bool:
        """Kiểm tra tín hiệu đơn lẻ có xác suất thắng >= prob_threshold hay không."""
        if isinstance(signal_row, pd.DataFrame):
            if len(signal_row) == 0:
                return False
            sig_val = signal_row["signal"].iloc[0] if "signal" in signal_row.columns else 1
            if pd.isna(sig_val) or int(sig_val) == 0:
                return False
            row_df = signal_row.iloc[[0]]
        elif isinstance(signal_row, pd.Series):
            sig_val = signal_row.get("signal", 1)
            if pd.isna(sig_val) or int(sig_val) == 0:
                return False
            row_df = pd.DataFrame([signal_row])
        elif isinstance(signal_row, dict):
            sig_val = signal_row.get("signal", 1)
            if pd.isna(sig_val) or int(sig_val) == 0:
                return False
            row_df = pd.DataFrame([signal_row])
        else:
            return False

        prob = float(self.predict_proba(row_df)[0])
        return bool(prob >= self.prob_threshold)


def compute_walk_forward_probs(
    sig_df: pd.DataFrame,
    config: TradingConfig = TradingConfig(),
) -> pd.Series:
    """Tính xác suất thắng out-of-sample trên từng fold theo Purged Walk-Forward.

    Chỉ fit model trên `train_idx` của từng fold (purge_bars = max_holding_bars) rồi predict trên `test_idx`.
    Tuyệt đối không fit trên toàn chuỗi để dự đoán ngược về quá khứ.
    """
    probs = pd.Series(np.nan, index=sig_df.index, dtype="float64", name="ml_prob")
    if len(sig_df) < 2 or "signal" not in sig_df.columns:
        return probs

    work_df = _ensure_ml_feature_columns(sig_df, config)
    signals = work_df["signal"].to_numpy()
    signal_bar_positions = np.asarray(
        [
            t
            for t in range(len(work_df) - 1)
            if not pd.isna(signals[t]) and int(signals[t]) in (1, -1)
        ],
        dtype=np.int64,
    )
    if len(signal_bar_positions) == 0:
        return probs

    X, y, _ = build_meta_dataset(work_df, config)
    default_filter = MetaLabelFilter(config)
    # Khối khởi đầu (fold 0 chưa có tập train trước đó) dùng fallback_prob_ của model chưa fit
    probs.loc[X.index] = default_filter.fallback_prob_

    n_splits = max(1, int(config.ml_n_splits))
    if len(X) < n_splits + 1:
        return probs

    splitter = PurgedWalkForwardSplit(
        n_splits=n_splits,
        purge_bars=max(0, int(config.max_holding_bars)),
        embargo_bars=max(0, int(config.ml_embargo_bars)),
    )
    folds = splitter.split(X, bar_positions=signal_bar_positions)

    for train_idx, test_idx in folds:
        if len(test_idx) == 0:
            continue
        fold_filter = MetaLabelFilter(config)
        if len(train_idx) > 0:
            fold_filter.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_probs = fold_filter.predict_proba(X.iloc[test_idx])
        probs.loc[X.index[test_idx]] = fold_probs

    return probs
