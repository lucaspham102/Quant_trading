"""Unit tests cho chiến lược Cross-Sectional Market-Neutral (Sprint 6).

Đảm bảo hoàn toàn offline, deterministic, thực thi nhanh (<30s), không gọi mạng,
không look-ahead bias và tuân thủ các ràng buộc Pandas 3.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cross_sectional import CrossSectionalConfig, run_cross_sectional_portfolio


def _create_synthetic_ohlcv(
    n_candles: int = 400,
    trends: dict[str, float] | None = None,
    funding_events: dict[str, dict[int, float]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Hàm bổ trợ tạo dữ liệu OHLCV 1h tổng hợp cho 4 symbol với DatetimeIndex UTC ns."""
    dates = pd.date_range(
        "2024-01-01 00:00:00", periods=n_candles, freq="1h", tz="UTC"
    ).as_unit("ns")
    default_trends = {
        "BTC/USDT": 0.0005,
        "ETH/USDT": -0.0004,
        "SOL/USDT": 0.0002,
        "BNB/USDT": -0.0001,
    }
    use_trends = trends if trends is not None else default_trends
    t_arr = np.arange(n_candles, dtype=np.float64)

    out: dict[str, pd.DataFrame] = {}
    for sym, drift in use_trends.items():
        close_prices = 100.0 * (1.0 + drift * t_arr)
        fr_events = np.zeros(n_candles, dtype=np.float64)

        if funding_events and sym in funding_events:
            for candle_idx, rate in funding_events[sym].items():
                if 0 <= candle_idx < n_candles:
                    fr_events[candle_idx] = rate

        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices * 1.002,
                "low": close_prices * 0.998,
                "close": close_prices,
                "volume": 1000.0,
                "funding_rate_event": fr_events,
            },
            index=dates,
        )
        df.index.name = "datetime"
        out[sym] = df

    return out


def test_weights_market_neutral() -> None:
    """Kiểm tra tính Market-Neutral của ma trận trọng số danh mục.

    Dữ liệu tổng hợp 4 symbol >= 400 nến 1h có xu hướng khác nhau.
    Chạy với mode 'momentum':
    - Tại mọi nến có vị thế, tổng trọng số (sum w) == 0.0 (sai số 1e-9).
    - Tổng độ lớn vị thế sum|w| <= gross_leverage + 1e-9 (và chính xác bằng gross_leverage).
    """
    n_candles = 500
    data = _create_synthetic_ohlcv(
        n_candles=n_candles,
        trends={
            "BTC/USDT": 0.0010,
            "ETH/USDT": -0.0008,
            "SOL/USDT": 0.0003,
            "BNB/USDT": 0.0,
        },
    )

    gross_lev = 1.5
    cfg = CrossSectionalConfig(
        mode="momentum",
        lookback_hours=168,
        rebalance_hours=24,
        min_spread=0.2,
        gross_leverage=gross_lev,
    )
    result = run_cross_sectional_portfolio(data, cfg)
    weights_df = result["weights_df"]

    assert len(weights_df) == n_candles
    assert set(weights_df.columns) == set(data.keys())

    has_active_position = False
    for _, row in weights_df.iterrows():
        abs_sum = float(row.abs().sum())
        row_sum = float(row.sum())
        if abs_sum > 1e-9:
            has_active_position = True
            # Market-neutral tuyệt đối: chân Long cân bằng chân Short
            assert abs(row_sum) < 1e-9
            # Giới hạn đòn bẩy gộp
            assert abs_sum <= gross_lev + 1e-9
            assert abs(abs_sum - gross_lev) < 1e-9

    assert has_active_position, "Chiến lược phải mở vị thế khi có phân kỳ điểm số"


def test_no_lookahead_scores() -> None:
    """Kiểm tra ngăn chặn rò rỉ dữ liệu tương lai (anti look-ahead bias).

    Đặt 1 symbol có cú nhảy giá lớn tại nến T:
    - Trọng số tại nến T KHÔNG được thay đổi (vì score tại T chỉ phản ánh dữ liệu <= T-1).
    - Trọng số có thể và ĐƯỢC thay đổi tại nến T+1 (vì thông tin nến T đã thành t-1 của T+1).
    - So sánh weights_df tại T giữa bộ dữ liệu có cú nhảy và bộ dữ liệu baseline không có cú nhảy.
    """
    n_candles = 100
    dates = pd.date_range(
        "2024-01-01 00:00:00", periods=n_candles, freq="1h", tz="UTC"
    ).as_unit("ns")

    t_jump = 50  # Mốc nến T
    t_arr = np.arange(n_candles, dtype=np.float64)

    # Dữ liệu cơ sở: Dao động hình sin chu kỳ 10 nến (ước số của lookback=20) để std lợi nhuận > 0 và ổn định
    noise = 0.004 * np.sin(2.0 * np.pi * t_arr / 10.0)
    close_btc = 100.0 * np.exp(np.cumsum(0.0002 + noise))
    close_eth = 100.0 * np.exp(np.cumsum(-0.0002 + noise))
    close_sol = 100.0 * np.exp(np.cumsum(0.00005 + noise))
    close_bnb_baseline = 100.0 * np.exp(np.cumsum(noise))

    # Dữ liệu nhảy giá: BNB nhảy 10x tại đúng nến T (t_jump)
    close_bnb_jump = close_bnb_baseline.copy()
    close_bnb_jump[t_jump:] *= 10.0

    data_base = {
        "BTC": pd.DataFrame({"close": close_btc, "funding_rate_event": 0.0}, index=dates),
        "ETH": pd.DataFrame({"close": close_eth, "funding_rate_event": 0.0}, index=dates),
        "SOL": pd.DataFrame({"close": close_sol, "funding_rate_event": 0.0}, index=dates),
        "BNB": pd.DataFrame({"close": close_bnb_baseline, "funding_rate_event": 0.0}, index=dates),
    }
    data_jump = {
        "BTC": pd.DataFrame({"close": close_btc, "funding_rate_event": 0.0}, index=dates),
        "ETH": pd.DataFrame({"close": close_eth, "funding_rate_event": 0.0}, index=dates),
        "SOL": pd.DataFrame({"close": close_sol, "funding_rate_event": 0.0}, index=dates),
        "BNB": pd.DataFrame({"close": close_bnb_jump, "funding_rate_event": 0.0}, index=dates),
    }

    cfg = CrossSectionalConfig(
        mode="momentum",
        lookback_hours=20,
        rebalance_hours=1,  # Rebalance mỗi nến để kiểm tra độ trễ 1 nến chính xác
        min_spread=0.1,
        gross_leverage=1.0,
    )

    res_base = run_cross_sectional_portfolio(data_base, cfg)
    res_jump = run_cross_sectional_portfolio(data_jump, cfg)

    w_base = res_base["weights_df"]
    w_jump = res_jump["weights_df"]

    ts_t_prev = dates[t_jump - 1]
    ts_t = dates[t_jump]
    ts_t_next = dates[t_jump + 1]

    # (i) Tại nến T: Cú nhảy ở nến T nhưng điểm số nến T chỉ dùng dữ liệu <= T-1.
    # Do đó trọng số tại T của BNB không đổi so với T-1 và giống hoàn toàn bộ baseline.
    assert w_jump.loc[ts_t, "BNB"] == pytest.approx(w_jump.loc[ts_t_prev, "BNB"])
    assert w_jump.loc[ts_t, "BNB"] == pytest.approx(0.0)
    assert w_jump.loc[ts_t].equals(w_base.loc[ts_t])

    # (ii) Tại nến T+1: Cú nhảy tại T đã thuộc quá khứ (t <= (T+1)-1 = T)
    # BNB vọt lên rank 1 cao nhất -> trọng số đổi sang Long (+0.5 * gross_leverage = +0.5).
    assert w_jump.loc[ts_t_next, "BNB"] == pytest.approx(0.5)
    assert w_jump.loc[ts_t_next, "BNB"] != w_jump.loc[ts_t, "BNB"]


def test_fees_only_on_turnover() -> None:
    """Kiểm tra chi phí giao dịch chỉ phát sinh khi có biến động trọng số (Turnover).

    2 kịch bản dữ liệu thiết kế sao cho chỉ có đúng 1 lần đổi trọng số trong cả backtest:
    - Kịch bản 1: gross_leverage = 1.0 (Turnover = 1.0).
    - Kịch bản 2: gross_leverage = 2.0 (Turnover = 2.0).
    Kiểm tra Total Fees Paid xấp xỉ bằng: tổng |Δw| * (perp_fee + slippage) * initial_capital.
    """
    n_candles = 60
    # Xu hướng đơn điệu ổn định: BTC tăng liên tục, ETH giảm liên tục, SOL & BNB phẳng
    data = _create_synthetic_ohlcv(
        n_candles=n_candles,
        trends={
            "BTC/USDT": 0.002,
            "ETH/USDT": -0.002,
            "SOL/USDT": 0.0,
            "BNB/USDT": 0.0,
        },
    )

    # Lookback 20h, rebalance 20h. Lần rebalance đầu tiên diễn ra tại nến 21.
    # Sau nến 21, xu hướng giữ nguyên nên xếp hạng không đổi -> không đổi trọng số nữa.
    # Kịch bản 1:
    cfg1 = CrossSectionalConfig(
        mode="momentum",
        lookback_hours=20,
        rebalance_hours=20,
        min_spread=0.1,
        gross_leverage=1.0,
        perp_fee=0.0005,
        slippage=0.0002,
        initial_capital=400.0,
    )
    res1 = run_cross_sectional_portfolio(data, cfg1)
    trades1 = res1["trades_df"]
    metrics1 = res1["metrics"]

    # Đổi trọng số từ 0.0 sang +0.5 (BTC) và -0.5 (ETH) -> Tổng |Δw| = 1.0
    expected_turnover1 = 1.0
    expected_fee1 = expected_turnover1 * (cfg1.perp_fee + cfg1.slippage) * cfg1.initial_capital
    assert metrics1["Turnover"] == pytest.approx(expected_turnover1, rel=1e-6)
    assert trades1["turnover"].sum() == pytest.approx(expected_turnover1, rel=1e-6)
    assert metrics1["Total Fees Paid"] == pytest.approx(expected_fee1, rel=1e-5)

    # Kịch bản 2: Đòn bẩy 2.0 -> Tổng |Δw| = 2.0
    cfg2 = CrossSectionalConfig(
        mode="momentum",
        lookback_hours=20,
        rebalance_hours=20,
        min_spread=0.1,
        gross_leverage=2.0,
        perp_fee=0.0010,
        slippage=0.0005,
        initial_capital=500.0,
    )
    res2 = run_cross_sectional_portfolio(data, cfg2)
    trades2 = res2["trades_df"]
    metrics2 = res2["metrics"]

    expected_turnover2 = 2.0
    expected_fee2 = expected_turnover2 * (cfg2.perp_fee + cfg2.slippage) * cfg2.initial_capital
    assert metrics2["Turnover"] == pytest.approx(expected_turnover2, rel=1e-6)
    assert trades2["turnover"].sum() == pytest.approx(expected_turnover2, rel=1e-6)
    assert metrics2["Total Fees Paid"] == pytest.approx(expected_fee2, rel=1e-5)


def test_funding_direction() -> None:
    """Kiểm tra chiều hạch toán Funding Rate:

    - Chân Long (w > 0) trả funding khi funding_rate > 0 (làm giảm lợi nhuận).
    - Chân Short (w < 0) nhận funding khi funding_rate > 0 (làm tăng lợi nhuận).
    - Kiểm tra công thức đóng góp -sum(w * funding) trên từng thành phần và trên R_port,t.
    """
    n_candles = 50
    dates = pd.date_range(
        "2024-01-01 00:00:00", periods=n_candles, freq="1h", tz="UTC"
    ).as_unit("ns")

    # Thiết lập giá sao cho BTC được Long (+0.5) và ETH được Short (-0.5) trong lookback 10h đầu
    # Sau nến 15, giá tất cả các coin hoàn toàn đi ngang (Close_t / Close_{t-1} - 1 == 0.0)
    p_btc = np.concatenate([100.0 * (1.0 + 0.005 * np.arange(15)), np.full(35, 107.0)])
    p_eth = np.concatenate([100.0 * (1.0 - 0.005 * np.arange(15)), np.full(35, 93.0)])
    p_sol = np.full(n_candles, 100.0)
    p_bnb = np.full(n_candles, 100.0)

    # Sự kiện funding tại nến 30 (nơi giá hoàn toàn phẳng và không có rebalance)
    t_event_idx = 30
    fr_event_rate = 0.0002

    fr_btc = np.zeros(n_candles)
    fr_eth = np.zeros(n_candles)
    fr_btc[t_event_idx] = fr_event_rate
    fr_eth[t_event_idx] = fr_event_rate

    data = {
        "BTC": pd.DataFrame({"close": p_btc, "funding_rate_event": fr_btc}, index=dates),
        "ETH": pd.DataFrame({"close": p_eth, "funding_rate_event": fr_eth}, index=dates),
        "SOL": pd.DataFrame({"close": p_sol, "funding_rate_event": 0.0}, index=dates),
        "BNB": pd.DataFrame({"close": p_bnb, "funding_rate_event": 0.0}, index=dates),
    }
    for df in data.values():
        df.index.name = "datetime"

    cfg = CrossSectionalConfig(
        mode="momentum",
        lookback_hours=10,
        rebalance_hours=20,
        min_spread=0.1,
        gross_leverage=1.0,
    )
    res = run_cross_sectional_portfolio(data, cfg)
    w_df = res["weights_df"]
    ret_series = res["returns"]

    ts_event = dates[t_event_idx]
    w_prev_btc = w_df.loc[dates[t_event_idx - 1], "BTC"]
    w_prev_eth = w_df.loc[dates[t_event_idx - 1], "ETH"]

    assert w_prev_btc == pytest.approx(0.5)   # Chân Long
    assert w_prev_eth == pytest.approx(-0.5)  # Chân Short

    # 1. Đóng góp funding của từng chân:
    # Long trả funding: w_btc * (-fr) = 0.5 * (-0.0002) = -0.0001 < 0
    contrib_btc = w_prev_btc * (-fr_event_rate)
    assert contrib_btc < 0.0

    # Short nhận funding: w_eth * (-fr) = -0.5 * (-0.0002) = +0.0001 > 0
    contrib_eth = w_prev_eth * (-fr_event_rate)
    assert contrib_eth > 0.0

    # 2. Khi funding hai bên bằng nhau và giá phẳng: tổng đóng góp ròng bằng 0
    assert contrib_btc + contrib_eth == pytest.approx(0.0)
    assert ret_series.loc[ts_event] == pytest.approx(0.0)

    # 3. Kịch bản bất đối xứng: Short nhận funding lớn hơn Long trả
    # Giả sử ETH (Short) có funding 0.0004 trong khi BTC (Long) chỉ 0.0001
    fr_btc[t_event_idx] = 0.0001
    fr_eth[t_event_idx] = 0.0004
    data["BTC"]["funding_rate_event"] = fr_btc
    data["ETH"]["funding_rate_event"] = fr_eth

    res_asym = run_cross_sectional_portfolio(data, cfg)
    ret_asym = res_asym["returns"]

    # Đóng góp ròng: -(0.5 * 0.0001 - 0.5 * 0.0004) = -(0.00005 - 0.0002) = +0.00015 > 0
    expected_asym_ret = - (0.5 * 0.0001 + (-0.5) * 0.0004)
    assert expected_asym_ret > 0.0
    assert ret_asym.loc[ts_event] == pytest.approx(expected_asym_ret)


def test_min_spread_filter() -> None:
    """Kiểm tra bộ lọc min_spread:

    Nếu 4 coin có điểm số gần bằng nhau (spread < min_spread), toàn bộ trọng số bằng 0,
    giữ 100% tiền mặt và Time in Market % == 0.0.
    """
    n_candles = 300
    # Cả 4 coin có quỹ đạo giá giống hệt nhau -> score giống hệt -> spread == 0.0
    data = _create_synthetic_ohlcv(
        n_candles=n_candles,
        trends={
            "BTC/USDT": 0.0001,
            "ETH/USDT": 0.0001,
            "SOL/USDT": 0.0001,
            "BNB/USDT": 0.0001,
        },
    )

    cfg = CrossSectionalConfig(
        mode="momentum",
        lookback_hours=48,
        rebalance_hours=24,
        min_spread=0.5,
        initial_capital=400.0,
    )
    result = run_cross_sectional_portfolio(data, cfg)
    w_df = result["weights_df"]
    metrics = result["metrics"]
    trades_df = result["trades_df"]
    equity_curve = result["equity_curve"]

    # Tất cả trọng số phải bằng 0.0
    assert (w_df == 0.0).all().all()
    assert metrics["Time in Market %"] == pytest.approx(0.0)
    assert metrics["Turnover"] == pytest.approx(0.0)
    assert len(trades_df) == 0
    assert metrics["Total Fees Paid"] == pytest.approx(0.0)
    assert equity_curve.iloc[-1] == pytest.approx(cfg.initial_capital)


def test_reversal_mode() -> None:
    """Kiểm tra chế độ đảo chiều (mode='reversal'):

    Rank 4 (điểm số thấp nhất) được Long (+0.5 * leverage),
    Rank 1 (điểm số cao nhất) bị Short (-0.5 * leverage).
    """
    n_candles = 100
    data = _create_synthetic_ohlcv(
        n_candles=n_candles,
        trends={
            "BTC/USDT": 0.005,   # Cao nhất -> Rank 1
            "ETH/USDT": -0.005,  # Thấp nhất -> Rank 4
            "SOL/USDT": 0.0,
            "BNB/USDT": 0.0,
        },
    )

    cfg = CrossSectionalConfig(
        mode="reversal",
        lookback_hours=20,
        rebalance_hours=10,
        min_spread=0.1,
        gross_leverage=1.2,
    )
    result = run_cross_sectional_portfolio(data, cfg)
    w_df = result["weights_df"]

    # Sau nến đủ lookback, ETH (thấp nhất) phải Long, BTC (cao nhất) phải Short
    active_rows = w_df[w_df.abs().sum(axis=1) > 0]
    assert len(active_rows) > 0

    first_active = active_rows.iloc[0]
    assert first_active["ETH/USDT"] == pytest.approx(+0.5 * cfg.gross_leverage)
    assert first_active["BTC/USDT"] == pytest.approx(-0.5 * cfg.gross_leverage)
    assert first_active["SOL/USDT"] == pytest.approx(0.0)
    assert first_active["BNB/USDT"] == pytest.approx(0.0)


def test_invalid_mode_and_empty_data() -> None:
    """Kiểm tra xử lý ngoại lệ cho mode không hợp lệ và cấu trúc dữ liệu rỗng."""
    data = _create_synthetic_ohlcv(n_candles=50)

    # 1. Mode không hợp lệ
    with pytest.raises(ValueError, match="Mode 'arbitrary' không hợp lệ"):
        cfg_invalid = CrossSectionalConfig(mode="arbitrary")
        run_cross_sectional_portfolio(data, cfg_invalid)

    # 2. Dict rỗng
    cfg = CrossSectionalConfig()
    empty_res = run_cross_sectional_portfolio({}, cfg)
    assert len(empty_res["equity_curve"]) == 0
    assert len(empty_res["trades_df"]) == 0
    assert empty_res["metrics"]["Sharpe"] == 0.0
