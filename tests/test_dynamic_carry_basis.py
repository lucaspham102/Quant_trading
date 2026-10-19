"""Unit tests cho động cơ Dynamic Carry + Basis Convergence (Sprint 8).

Toàn bộ test dùng dữ liệu tổng hợp, tất định (seed cố định), không đọc
``data/raw`` và không gọi mạng. Mục tiêu là khóa các hành vi đặc tả tại
``docs/go_no_go_sprint8.md`` và ``.agy/architect_3a5b8955.md``:

- Chống look-ahead tuyệt đối của funding EWMA và basis EMA.
- Tập trung vốn đúng ``top_k``, bỏ qua coin funding thấp/âm.
- Buffer thứ hạng, hysteresis và ``min_switch_advantage``.
- True Spot-Perp MTM từng giờ và phí khứ hồi.
- Các biên an toàn (input rỗng, thiếu cột, equity <= 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dynamic_carry_basis import (
    DynamicCarryConfig,
    build_carry_basis_scores,
    run_dynamic_carry_portfolio,
    select_portfolio,
)

SEED = 20240601
START = "2024-01-01 00:00:00"
SCORE_COLUMNS = ["f_ewma", "basis_ema", "basis_dev", "score", "basis_pct_prev"]


# ---------------------------------------------------------------------------
# Helpers dựng dữ liệu tổng hợp
# ---------------------------------------------------------------------------
def _make_index(n: int, start: str = START) -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC ns tên ``datetime`` với tần suất 1h."""
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC").as_unit("ns")
    idx.name = "datetime"
    return idx


def _anchors(idx: pd.DatetimeIndex) -> np.ndarray:
    """Mặt nạ các mốc thanh toán funding 8h (00/08/16 UTC)."""
    return np.asarray(idx.hour.isin([0, 8, 16]) & (idx.minute == 0))


def _last_known_from_events(idx: pd.DatetimeIndex, events: np.ndarray) -> np.ndarray:
    """Mô phỏng ``last_known_funding_rate``: chỉ cập nhật tại các mốc 8h."""
    s = pd.Series(
        np.where(_anchors(idx), np.asarray(events, dtype="float64"), np.nan),
        index=idx,
        dtype="float64",
    )
    return s.ffill().fillna(0.0).to_numpy(dtype="float64")


def _frame(
    idx: pd.DatetimeIndex,
    spot: np.ndarray,
    perp: np.ndarray,
    basis: np.ndarray,
    funding_event: np.ndarray,
    last_known: np.ndarray | None = None,
) -> pd.DataFrame:
    """Dựng DataFrame đã chuẩn hóa giống đầu ra ``align_spot_perp_funding``."""
    spot = np.asarray(spot, dtype="float64")
    perp = np.asarray(perp, dtype="float64")
    basis = np.asarray(basis, dtype="float64")
    funding_event = np.asarray(funding_event, dtype="float64")
    if last_known is None:
        last_known = _last_known_from_events(idx, funding_event)

    frame = pd.DataFrame(
        {
            "open": perp,
            "high": perp,
            "low": perp,
            "close": perp,
            "volume": np.full(len(idx), 1000.0),
            "spot_close": spot,
            "perp_close": perp,
            "basis_pct": basis,
            "funding_rate_event": funding_event,
            "last_known_funding_rate": np.asarray(last_known, dtype="float64"),
        },
        index=idx.copy(),
    )
    frame.index.name = "datetime"
    return frame


def _make_universe(idx: pd.DatetimeIndex, n_syms: int = 12) -> dict[str, pd.DataFrame]:
    """Tạo 12 symbol tổng hợp với seed cố định."""
    rng = np.random.default_rng(SEED)
    t = np.arange(len(idx), dtype="float64")
    anchors = _anchors(idx)
    universe: dict[str, pd.DataFrame] = {}

    for i in range(n_syms):
        spot = 100.0 + 3.0 * i + 0.01 * t + float(rng.uniform(-0.05, 0.05))
        basis = 4e-4 * np.sin((t + 2.5 * i) / 11.0) + float(rng.uniform(-2e-5, 2e-5))
        perp = spot * (1.0 + basis)
        events = np.zeros(len(idx), dtype="float64")
        events[anchors] = 1.5e-4 + 2.0e-5 * i
        universe[f"SYM{i:02d}"] = _frame(idx, spot, perp, basis, events)

    return universe


def _single_symbol(
    idx: pd.DatetimeIndex, perp: np.ndarray, spot_value: float = 100.0
) -> dict[str, pd.DataFrame]:
    """Tạo dữ liệu 1 symbol với spot phẳng và funding bằng 0."""
    spot = np.full(len(idx), spot_value, dtype="float64")
    perp = np.asarray(perp, dtype="float64")
    basis = (perp - spot) / spot
    funding = np.zeros(len(idx), dtype="float64")
    return {"AAA": _frame(idx, spot, perp, basis, funding)}


def _single_config(**overrides: float | int | str) -> DynamicCarryConfig:
    """Config tối giản cho test MTM 1 symbol: mở ngay mốc 8h đầu tiên."""
    base: dict[str, float | int | str] = {
        "structure": "spot_perp_3x",
        "execution": "real_taker",
        "top_k": 1,
        "entry_threshold": -1.0,
        "exit_threshold": -1.0,
        "initial_capital": 400.0,
    }
    base.update(overrides)
    return DynamicCarryConfig(**base)


# ---------------------------------------------------------------------------
# A. No look-ahead
# ---------------------------------------------------------------------------
def test_build_scores_no_lookahead_when_future_data_changes() -> None:
    """Score tại t không đổi khi toàn bộ dữ liệu từ t trở đi bị thay đổi.

    Thay đổi gấp đôi giá, basis và funding từ đúng nến t trở đi. Vì score tại t
    chỉ được phép dùng dữ liệu đóng ở <= t-1, score tại t phải giống hệt bản gốc.
    """
    idx = _make_index(240)
    cfg = DynamicCarryConfig()
    data = _make_universe(idx)

    baseline = build_carry_basis_scores(data, cfg, idx)

    t_pos = 144  # 2024-01-07 00:00 UTC, đúng một mốc funding 8h
    t = idx[t_pos]
    price_cols = ["open", "high", "low", "close", "spot_close", "perp_close"]

    modified: dict[str, pd.DataFrame] = {}
    for sym, frame in data.items():
        changed = frame.copy()
        future = changed.index >= t
        changed.loc[future, price_cols] = changed.loc[future, price_cols] * 2.0
        changed.loc[future, "basis_pct"] = changed.loc[future, "basis_pct"] * 2.0
        changed.loc[future, "funding_rate_event"] = (
            changed.loc[future, "funding_rate_event"] * 2.0
        )
        changed.loc[future, "last_known_funding_rate"] = (
            changed.loc[future, "last_known_funding_rate"] * 2.0
        )
        modified[sym] = changed

    after = build_carry_basis_scores(modified, cfg, idx)

    for sym in data:
        base_score = float(baseline[sym].loc[t, "score"])
        after_score = float(after[sym].loc[t, "score"])
        assert np.isfinite(base_score)
        assert np.allclose(base_score, after_score, rtol=1e-12, atol=1e-15), sym

    # Dữ liệu tương lai thực sự có tác động từ nến t+1 trở đi.
    next_t = idx[t_pos + 1]
    base_next = np.array([float(baseline[s].loc[next_t, "score"]) for s in data])
    after_next = np.array([float(after[s].loc[next_t, "score"]) for s in data])
    assert not np.allclose(base_next, after_next, rtol=1e-12, atol=1e-15)


def test_funding_ewma_uses_shift_one() -> None:
    """Cú sốc funding tại đúng nến t không được phản ánh vào score tại t."""
    idx = _make_index(120)
    cfg = DynamicCarryConfig()
    n = len(idx)
    spot = np.full(n, 100.0, dtype="float64")
    perp = spot.copy()
    basis = np.zeros(n, dtype="float64")

    events = np.zeros(n, dtype="float64")
    events[_anchors(idx)] = 0.0003

    t_pos = 48  # 2024-01-03 00:00 UTC, mốc funding 8h
    t = idx[t_pos]
    shocked_events = events.copy()
    shocked_events[t_pos] = 0.02

    base = _frame(idx, spot, perp, basis, events)
    shocked = _frame(idx, spot, perp, basis, shocked_events)

    base_scores = build_carry_basis_scores({"AAA": base}, cfg, idx)["AAA"]
    shocked_scores = build_carry_basis_scores({"AAA": shocked}, cfg, idx)["AAA"]

    # Tại t, EWMA funding chỉ biết các kỳ <= t-1.
    assert float(base_scores.loc[t, "f_ewma"]) == pytest.approx(0.0003)
    assert float(shocked_scores.loc[t, "f_ewma"]) == pytest.approx(
        float(base_scores.loc[t, "f_ewma"])
    )
    assert float(shocked_scores.loc[t, "score"]) == pytest.approx(
        float(base_scores.loc[t, "score"])
    )

    # Từ t+1, cú sốc mới xuất hiện trong score.
    next_t = idx[t_pos + 1]
    assert float(shocked_scores.loc[next_t, "score"]) > (
        float(base_scores.loc[next_t, "score"]) + 1e-4
    )


def test_basis_ema_uses_shift_one() -> None:
    """Cú sốc basis tại đúng nến t không được phản ánh vào EMA/score tại t."""
    idx = _make_index(120)
    cfg = DynamicCarryConfig()
    n = len(idx)
    spot = np.full(n, 100.0, dtype="float64")
    basis = np.full(n, 0.0002, dtype="float64")
    perp = spot * (1.0 + basis)
    events = np.zeros(n, dtype="float64")

    t_pos = 50
    t = idx[t_pos]
    shocked_basis = basis.copy()
    shocked_basis[t_pos] = 0.02

    base = _frame(idx, spot, perp, basis, events)
    shocked = _frame(idx, spot, perp, shocked_basis, events)

    base_scores = build_carry_basis_scores({"AAA": base}, cfg, idx)["AAA"]
    shocked_scores = build_carry_basis_scores({"AAA": shocked}, cfg, idx)["AAA"]

    assert float(shocked_scores.loc[t, "basis_ema"]) == pytest.approx(
        float(base_scores.loc[t, "basis_ema"])
    )
    assert float(shocked_scores.loc[t, "score"]) == pytest.approx(
        float(base_scores.loc[t, "score"])
    )

    next_t = idx[t_pos + 1]
    assert abs(
        float(shocked_scores.loc[next_t, "score"])
        - float(base_scores.loc[next_t, "score"])
    ) > 1e-5


# ---------------------------------------------------------------------------
# B. Tập trung vốn đúng top_k
# ---------------------------------------------------------------------------
def _funding_universe() -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame], set[str]]:
    """12 symbol: chỉ 2 symbol có funding cao, còn lại 0 hoặc âm."""
    idx = _make_index(25)
    anchors = _anchors(idx)
    high = {"SYM00", "SYM01"}
    data: dict[str, pd.DataFrame] = {}

    for i in range(12):
        sym = f"SYM{i:02d}"
        events = np.zeros(len(idx), dtype="float64")
        if sym in high:
            events[anchors] = 0.001
        elif i % 2 == 0:
            events[anchors] = -0.0002

        flat = np.full(len(idx), 100.0, dtype="float64")
        data[sym] = _frame(
            idx, flat, flat.copy(), np.zeros(len(idx), dtype="float64"), events
        )

    return idx, data, high


def test_run_portfolio_concentrates_capital_on_top_k() -> None:
    """Chỉ 2 coin funding cao nhất nhận vốn; coin funding 0/âm không nhận vốn."""
    idx, data, high = _funding_universe()
    cfg = DynamicCarryConfig(
        top_k=2,
        entry_threshold=0.00010,
        initial_capital=400.0,
    )

    result = run_dynamic_carry_portfolio(data, cfg)
    trades = result["trades_df"]
    opens = trades[trades["action"] == "open"]

    assert len(opens) == 2
    assert set(opens["symbol"]) == high

    # Mỗi vị thế nhận xấp xỉ (equity / top_k) * eta, với eta = 0.75.
    for notional in opens["notional"]:
        assert float(notional) == pytest.approx(
            (cfg.initial_capital / cfg.top_k) * cfg.notional_eta(), rel=1e-2
        )

    holdings = result["holdings_log"]
    assert len(holdings) >= 1
    # Bar đầu tiên chưa thể xếp hạng (score cần dữ liệu <= t-1) nên phải xét
    # dòng log đầu tiên SAU khi danh mục đã được thiết lập.
    active_rows = holdings[holdings["n_active"] > 0]
    assert len(active_rows) >= 1
    first = active_rows.iloc[0]
    assert int(first["n_active"]) == 2
    assert set(first["symbols"]) == high
    assert float(first["total_notional"]) == pytest.approx(
        (float(first["equity"]) / cfg.top_k) * cfg.notional_eta() * 2.0,
        rel=1e-2,
    )

    for _, row in holdings.iterrows():
        assert int(row["n_active"]) <= 2
        assert set(row["symbols"]) <= high


def test_select_portfolio_target_weights_only_top_k() -> None:
    """target_weights chỉ chứa đúng top_k symbol điểm cao nhất với weight 1/top_k."""
    _, data, high = _funding_universe()
    cfg = DynamicCarryConfig(top_k=2, entry_threshold=0.00010)

    scores: dict[str, float] = {}
    for sym in data:
        if sym in high:
            scores[sym] = 0.001
        elif sym == "SYM02":
            scores[sym] = -0.0002
        else:
            scores[sym] = 0.0

    decision = select_portfolio(scores, set(), {}, cfg)

    assert set(decision.target_weights) == high
    for sym in high:
        assert decision.target_weights[sym] == pytest.approx(1.0 / cfg.top_k)

    for sym in scores:
        if sym not in high:
            assert decision.actions[sym] == "SKIP"


# ---------------------------------------------------------------------------
# C. Buffer + min_switch_advantage + min_hold_periods
# ---------------------------------------------------------------------------
def test_hysteresis_holds_when_gap_below_min_switch_advantage() -> None:
    """Chênh lệch điểm nhỏ hơn min_switch_advantage: giữ A, không đảo vị thế."""
    cfg = DynamicCarryConfig(top_k=2, min_hold_periods=9)
    min_switch = cfg.min_switch_advantage()

    b_score = 0.0010
    a_score = b_score - 0.5 * min_switch

    decision = select_portfolio(
        {"A": a_score, "B": b_score},
        current_holdings={"A"},
        periods_held={"A": 100},
        config=cfg,
    )

    assert decision.actions["A"] == "HOLD"
    assert decision.reasons["A"] == "HOLD_BUFFER_RANK"
    assert decision.actions["B"] == "OPEN"  # lấp slot còn trống, không phải swap
    assert "A" in decision.target_weights


def test_switch_when_gap_above_min_switch_advantage() -> None:
    """Chênh lệch điểm vượt min_switch_advantage: đóng A và mở B."""
    cfg = DynamicCarryConfig(top_k=2, min_hold_periods=9)
    min_switch = cfg.min_switch_advantage()

    b_score = 0.0010
    a_score = b_score - 2.0 * min_switch

    decision = select_portfolio(
        {"A": a_score, "B": b_score},
        current_holdings={"A"},
        periods_held={"A": 100},
        config=cfg,
    )

    assert decision.actions["A"] == "CLOSE"
    assert decision.reasons["A"] == "SWITCH_TO_BETTER_CANDIDATE"
    assert decision.actions["B"] == "OPEN"
    assert "A" not in decision.target_weights
    assert "B" in decision.target_weights


def test_min_hold_periods_forces_hold_below_exit_threshold() -> None:
    """Dưới min_hold_periods thì vẫn HOLD dù score đã rơi dưới exit_threshold."""
    cfg = DynamicCarryConfig(top_k=2, min_hold_periods=9)

    decision = select_portfolio(
        {"A": 1e-6, "B": 0.0010},
        current_holdings={"A"},
        periods_held={"A": 3},
        config=cfg,
    )

    assert decision.actions["A"] == "HOLD"
    assert decision.reasons["A"] == "HOLD_MIN_PERIODS"
    assert "A" in decision.target_weights


def test_buffer_rank_holds_rank_inside_top_k_plus_buffer() -> None:
    """Coin đang giữ xếp hạng 3 với top_k=2, buffer_rank=1 vẫn được HOLD."""
    cfg = DynamicCarryConfig(top_k=2, buffer_rank=1, min_hold_periods=9)
    min_switch = cfg.min_switch_advantage()

    x_score = 0.0010
    a_score = x_score - 0.5 * min_switch  # vẫn thấp hơn X nhưng chưa đủ để swap
    y_score = 0.0009

    decision = select_portfolio(
        {"X": x_score, "Y": y_score, "A": a_score},
        current_holdings={"A"},
        periods_held={"A": 100},
        config=cfg,
    )

    rank_a = next(rank for sym, _, rank in decision.ranked_symbols if sym == "A")
    assert rank_a == cfg.top_k + cfg.buffer_rank
    assert decision.actions["A"] == "HOLD"
    assert decision.reasons["A"] == "HOLD_BUFFER_RANK"
    assert "A" in decision.target_weights


def test_exit_below_threshold_after_min_hold() -> None:
    """Sau min_hold_periods, score dưới exit_threshold phải CLOSE đúng lý do."""
    cfg = DynamicCarryConfig(top_k=2, min_hold_periods=9)

    decision = select_portfolio(
        {"A": 1e-6, "B": 0.0},
        current_holdings={"A"},
        periods_held={"A": 100},
        config=cfg,
    )

    assert decision.actions["A"] == "CLOSE"
    assert decision.reasons["A"] == "EXIT_BELOW_THRESHOLD"
    assert "A" not in decision.target_weights


# ---------------------------------------------------------------------------
# D. True Spot-Perp MTM chính xác từng giờ
# ---------------------------------------------------------------------------
def test_true_spot_perp_mtm_short_perp_loses_when_perp_rises() -> None:
    """Spot phẳng, perp tăng tuyến tính: basis_pnl = -Q * (perp_T - perp_entry)."""
    n = 49
    idx = _make_index(n)
    t = np.arange(n, dtype="float64")
    perp = 100.0 + 0.2 * t
    cfg = _single_config()

    result = run_dynamic_carry_portfolio(_single_symbol(idx, perp), cfg)

    trades = result["trades_df"]
    opens = trades[trades["action"] == "open"]
    assert len(opens) == 1
    open_row = opens.iloc[0]
    assert open_row["time"] == idx[8]

    q = float(open_row["notional"]) / 100.0
    expected = -q * (float(perp[-1]) - float(perp[8]))

    assert result["basis_pnl"] == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert result["metrics"]["Total MTM PnL"] == pytest.approx(
        expected, rel=1e-9, abs=1e-9
    )
    assert result["funding_pnl"] == pytest.approx(0.0)
    assert result["metrics"]["Num Rebalances"] == 0

    # MTM từng giờ chỉ là -Q * delta_perp = -0.6 USDT sau khi đã mở.
    hourly_basis = result["attribution_series"]["basis_pnl"].to_numpy()
    assert np.allclose(hourly_basis[9:], -q * 0.2, rtol=1e-12, atol=1e-12)

    equity_diff = result["equity_curve"].diff().to_numpy()
    # Bar mở vị thế (idx[8]) chỉ chịu phí vào lệnh, MTM bắt đầu từ bar kế tiếp.
    assert equity_diff[8] == pytest.approx(-float(result["fees"]), rel=1e-12, abs=1e-9)
    assert np.allclose(equity_diff[9:], -q * 0.2, rtol=1e-10, atol=1e-10)

    final_equity = float(result["equity_curve"].iloc[-1])
    assert final_equity == pytest.approx(
        cfg.initial_capital + result["funding_pnl"] + result["basis_pnl"] - result["fees"],
        rel=1e-12,
        abs=1e-9,
    )


def test_true_spot_perp_mtm_short_perp_gains_when_perp_falls() -> None:
    """Spot phẳng, perp giảm tuyến tính: short perp sinh basis_pnl dương."""
    n = 49
    idx = _make_index(n)
    t = np.arange(n, dtype="float64")
    perp = 100.0 - 0.2 * t
    cfg = _single_config()

    result = run_dynamic_carry_portfolio(_single_symbol(idx, perp), cfg)

    opens = result["trades_df"][result["trades_df"]["action"] == "open"]
    assert len(opens) == 1
    q = float(opens.iloc[0]["notional"]) / 100.0
    expected = -q * (float(perp[-1]) - float(perp[8]))

    assert expected > 0.0
    assert result["basis_pnl"] == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert result["metrics"]["Total MTM PnL"] == pytest.approx(
        expected, rel=1e-9, abs=1e-9
    )
    assert result["funding_pnl"] == pytest.approx(0.0)
    assert result["metrics"]["Num Rebalances"] == 0

    hourly_basis = result["attribution_series"]["basis_pnl"].to_numpy()
    assert np.allclose(hourly_basis[9:], q * 0.2, rtol=1e-12, atol=1e-12)


def test_round_trip_fees_equal_cost_round_trip() -> None:
    """Phí mở + đóng đúng bằng leg_notional * cost_round_trip().

    Ghi chú kế toán: ``cost_round_trip()`` đã bao gồm cả chiều mở và chiều đóng
    (``2 * one_way_cost()``). Vì vậy tổng phí khứ hồi là
    ``leg_notional * cost_round_trip()``, không nhân thêm 2 lần nữa.
    """
    n = 48
    idx = _make_index(n)
    anchors = _anchors(idx)
    events = np.zeros(n, dtype="float64")
    events[0] = 0.001     # đủ điểm để mở tại 08:00
    events[8] = -0.001    # score tại 16:00 âm nhưng chưa đủ min_hold
    events[16] = -0.002   # score tại 24:00 âm sâu, đủ min_hold -> đóng
    events[24] = -0.002

    flat = np.full(n, 100.0, dtype="float64")
    data = {
        "AAA": _frame(
            idx,
            flat,
            flat.copy(),
            np.zeros(n, dtype="float64"),
            events,
        )
    }
    cfg = DynamicCarryConfig(
        top_k=1,
        entry_threshold=0.00010,
        exit_threshold=0.00002,
        min_hold_periods=2,
        initial_capital=400.0,
    )

    result = run_dynamic_carry_portfolio(data, cfg)
    trades = result["trades_df"]
    opens = trades[trades["action"] == "open"]
    closes = trades[trades["action"] == "close"]

    assert len(opens) == 1
    assert len(closes) == 1
    open_row = opens.iloc[0]
    close_row = closes.iloc[0]

    assert open_row["time"] == idx[8]
    assert close_row["time"] == idx[24]
    assert close_row["reason"] == "EXIT_BELOW_THRESHOLD"

    leg_notional = float(open_row["notional"])
    assert leg_notional == pytest.approx(
        cfg.initial_capital * cfg.notional_eta(), rel=1e-12
    )

    assert float(open_row["fee"]) == pytest.approx(
        leg_notional * cfg.one_way_cost(), rel=1e-12
    )
    assert float(close_row["fee"]) == pytest.approx(
        leg_notional * cfg.one_way_cost(), rel=1e-12
    )
    assert float(close_row["notional"]) == pytest.approx(leg_notional, rel=1e-12)

    assert result["fees"] == pytest.approx(
        leg_notional * cfg.cost_round_trip(), rel=1e-12, abs=1e-9
    )
    assert result["fees"] == pytest.approx(
        float(trades["fee"].sum()), rel=1e-12, abs=1e-9
    )
    assert result["metrics"]["Num Rebalances"] == 0


# ---------------------------------------------------------------------------
# E. An toàn & biên
# ---------------------------------------------------------------------------
def test_empty_input_returns_empty_outputs() -> None:
    """Dict rỗng hoặc DataFrame rỗng không crash; equity rỗng và metrics bằng 0."""
    cfg = DynamicCarryConfig()
    empty_idx = pd.DatetimeIndex([], tz="UTC", name="datetime").as_unit("ns")
    empty_frame = pd.DataFrame(
        columns=[
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spot_close",
            "perp_close",
            "basis_pct",
            "funding_rate_event",
            "last_known_funding_rate",
        ],
        index=empty_idx,
        dtype="float64",
    )

    assert build_carry_basis_scores({}, cfg) == {}

    for data in ({}, {"AAA": empty_frame}):
        result = run_dynamic_carry_portfolio(data, cfg)
        assert len(result["equity_curve"]) == 0
        assert len(result["returns"]) == 0
        assert len(result["trades_df"]) == 0
        assert len(result["holdings_log"]) == 0
        for key in ("funding_pnl", "basis_pnl", "fees", "turnover"):
            assert result[key] == 0.0
        for value in result["metrics"].values():
            assert value == 0


def test_missing_spot_and_basis_columns_fallback() -> None:
    """Thiếu spot_close/basis_pct: fallback spot=perp, basis=0, không crash."""
    idx = _make_index(24)
    close = 100.0 + np.arange(len(idx), dtype="float64") * 0.1
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(idx), 1000.0),
        },
        index=idx.copy(),
    )
    frame.index.name = "datetime"

    cfg = DynamicCarryConfig(entry_threshold=0.00010)
    scores = build_carry_basis_scores({"AAA": frame}, cfg, idx)
    out = scores["AAA"]

    assert list(out.columns) == SCORE_COLUMNS
    assert out["basis_pct_prev"].iloc[1:].eq(0.0).all()
    assert np.isfinite(out["score"].iloc[1:]).all()

    result = run_dynamic_carry_portfolio({"AAA": frame}, cfg)
    assert len(result["equity_curve"]) == len(idx)
    assert np.isfinite(result["equity_curve"]).all()
    assert result["metrics"]["Total Return %"] == pytest.approx(0.0)


def test_non_positive_equity_has_finite_returns() -> None:
    """Equity bị đẩy xuống <= 0 không sinh NaN/inf trong returns."""
    n = 48
    idx = _make_index(n)
    anchors = _anchors(idx)
    events = np.zeros(n, dtype="float64")
    events[0] = 0.001      # mở vị thế tại 08:00
    events[8] = 0.001      # score tại 16:00 vẫn dương
    events[16] = -2.0      # funding âm cực lớn, đẩy equity xuống âm

    flat = np.full(n, 100.0, dtype="float64")
    data = {
        "AAA": _frame(
            idx,
            flat,
            flat.copy(),
            np.zeros(n, dtype="float64"),
            events,
        )
    }
    cfg = DynamicCarryConfig(
        top_k=1,
        entry_threshold=0.00010,
        exit_threshold=0.0,
        min_hold_periods=9,
        initial_capital=400.0,
    )

    result = run_dynamic_carry_portfolio(data, cfg)
    equity = result["equity_curve"]
    returns = result["returns"]

    assert (equity < 0.0).any()
    assert np.isfinite(equity).all()
    assert np.isfinite(returns).all()
    assert not np.isinf(returns).any()
    assert np.isfinite(result["metrics"]["MaxDD %"])


def test_dynamic_carry_config_cost_and_eta_identities() -> None:
    """Khóa các đẳng thức cấu hình: eta, cost_round_trip, min_switch_advantage."""
    spot_perp = DynamicCarryConfig(structure="spot_perp_3x")
    assert spot_perp.notional_eta() == pytest.approx(0.75)
    assert spot_perp.one_way_cost() == pytest.approx(0.00165)
    assert spot_perp.cost_round_trip() == pytest.approx(0.00330)
    assert spot_perp.cost_round_trip() == pytest.approx(
        2.0 * spot_perp.one_way_cost()
    )
    assert spot_perp.min_switch_advantage() > 0.0
    assert spot_perp.min_switch_advantage() == pytest.approx(
        2.0 * spot_perp.cost_round_trip() / float(spot_perp.expected_hold_periods)
    )

    coinm = DynamicCarryConfig(structure="coinm_1leg")
    assert coinm.notional_eta() == pytest.approx(1.0)
    assert coinm.one_way_cost() == pytest.approx(0.00070)
    assert coinm.cost_round_trip() == pytest.approx(0.00140)
    assert coinm.min_switch_advantage() > 0.0

    overridden = DynamicCarryConfig(min_switch_advantage_override=0.0005)
    assert overridden.min_switch_advantage() == pytest.approx(0.0005)

