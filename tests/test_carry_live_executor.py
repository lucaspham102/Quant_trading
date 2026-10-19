"""Unit test offline cho CarryLiveExecutor, StateStore carry fields và CLI run_live.

Toàn bộ test đều tất định, không gọi mạng, không đọc ``data/raw`` thật và chỉ ghi
file trong ``tmp_path``. Dữ liệu parquet tổng hợp được tạo trực tiếp trong test để
khóa các hành vi sau:

- Tương thích ngược StateStore khi file JSON cũ chưa có trường carry.
- Ghi nguyên tử và khôi phục ``carry_positions`` / ``carry_last_reconcile_ts``.
- Kế hoạch reconcile OPEN -> HOLD -> CLOSE với cache parquet đúng quy ước tên.
- Kiểm tra an toàn ký quỹ và bảng trạng thái thuần ASCII.
- CLI ``run_live.main([... "--strategy", "dynamic_carry", "--once"])`` offline.
- Các nhánh an toàn: cache rỗng ném FileNotFoundError, universe rỗng/thiếu dữ liệu
  không crash.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

import run_live
from src.carry_live_executor import CarryLiveExecutor, ReconcileAction
from src.config import TradingConfig
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import BotState, StateStore

_START = "2024-01-01 00:00:00"
_N_HOURS = 60
_HIGH_FUNDING = 0.001
_LOW_FUNDING = -0.001


def _hourly_index(n_hours: int = _N_HOURS, start: str = _START) -> pd.DatetimeIndex:
    """Tạo DatetimeIndex 1h UTC cố định, không phụ thuộc thời điểm chạy test."""
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    return idx.as_unit("ns")


def _ohlcv_frame(index: pd.DatetimeIndex, close: float) -> pd.DataFrame:
    """Tạo DataFrame OHLCV tổng hợp với giá đứng yên quanh ``close``."""
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame.astype("float64")


def _aligned_frame(
    symbol: str,
    funding_rate: float,
    n_hours: int = _N_HOURS,
    spot_price: float = 100.0,
    perp_price: float = 100.05,
) -> pd.DataFrame:
    """Tạo DataFrame đã đồng bộ Spot/Perp/Funding giống đầu ra align_spot_perp_funding."""
    idx = _hourly_index(n_hours)
    frame = _ohlcv_frame(idx, perp_price)
    frame["spot_close"] = float(spot_price)
    frame["perp_close"] = float(perp_price)
    frame["basis_pct"] = (float(perp_price) - float(spot_price)) / float(spot_price)

    mask_8h = idx.hour.isin([0, 8, 16]) & (idx.minute == 0)
    frame["funding_rate_event"] = 0.0
    frame.loc[mask_8h, "funding_rate_event"] = float(funding_rate)
    frame["last_known_funding_rate"] = float(funding_rate)
    frame.attrs["symbol"] = symbol
    return frame


def _write_cache_bundle(
    raw_dir: Path,
    symbol: str,
    funding_rate: float,
    n_hours: int = _N_HOURS,
    spot_price: float = 100.0,
    perp_price: float = 100.05,
) -> None:
    """Ghi bộ cache parquet đúng quy ước tên của funding_loader / run_sprint7."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    idx = _hourly_index(n_hours)
    perp = _ohlcv_frame(idx, perp_price)
    spot = _ohlcv_frame(idx, spot_price)

    clean = symbol.replace("/", "_").replace(":", "_")
    spot_clean = symbol.split(":")[0].replace("/", "").replace(":", "").upper()
    if spot_clean.endswith("USDTUSDT"):
        spot_clean = spot_clean[:-4]

    funding_idx = pd.date_range(
        _START,
        periods=max(1, n_hours // 8 + 1),
        freq="8h",
        tz="UTC",
    ).as_unit("ns")
    funding = pd.DataFrame(
        {
            "timestamp": funding_idx.asi8 // 1_000_000,
            "datetime": funding_idx,
            "funding_rate": float(funding_rate),
            "mark_price": float(spot_price),
        }
    )

    perp.to_parquet(raw_dir / f"{clean}_1h.parquet", engine="pyarrow")
    spot.to_parquet(raw_dir / f"{spot_clean}_SPOT_1h.parquet", engine="pyarrow")
    funding.to_parquet(raw_dir / f"{clean}_funding.parquet", engine="pyarrow", index=False)


def _make_executor(
    tmp_path: Path,
    raw_dir: Path | None = None,
    symbols: Sequence[str] | None = None,
    carry_config: DynamicCarryConfig | None = None,
) -> CarryLiveExecutor:
    """Khởi tạo executor trỏ state file và raw dir vào thư mục tạm."""
    resolved_raw = Path(raw_dir) if raw_dir is not None else tmp_path / "raw"
    resolved_raw.mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / "state.json"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=resolved_raw,
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=carry_config if carry_config is not None else DynamicCarryConfig(),
        symbols=symbols if symbols is not None else ["AAA/USDT:USDT"],
        state_store=StateStore(state_path),
        raw_data_dir=resolved_raw,
    )


# ---------------------------------------------------------------------------
# 1. State store tương thích ngược + ghi nguyên tử
# ---------------------------------------------------------------------------


def test_state_store_backward_compatibility_with_carry_positions(tmp_path: Path) -> None:
    """File state kiểu cũ vẫn load được; save/load carry_positions giữ nguyên và không sót .tmp."""
    state_path = tmp_path / "state.json"
    old_payload: dict[str, Any] = {
        "open_position": None,
        "daily_realized_pnl": 1.5,
        "daily_start_equity": 500.0,
        "current_equity": 505.0,
        "current_utc_date": "2025-01-01",
        "consecutive_losses": 2,
        "circuit_breaker_tripped": False,
        "last_processed_candle_ts": "2025-01-01T00:00:00+00:00",
        "trade_history": [],
    }
    state_path.write_text(json.dumps(old_payload), encoding="utf-8")

    store = StateStore(state_path)
    loaded_old = store.load(default_equity=400.0)

    assert loaded_old.current_equity == pytest.approx(505.0)
    assert loaded_old.carry_positions == {}
    assert loaded_old.carry_last_reconcile_ts is None

    carry_positions = {
        "AAA/USDT:USDT": {
            "symbol": "AAA/USDT:USDT",
            "spot_symbol": "AAA/USDT",
            "qty": 1.5,
            "entry_spot_price": 100.0,
            "entry_perp_price": 100.05,
            "notional_usdt": 150.0,
            "margin_allocated_usdt": 49.75,
            "periods_held": 3,
            "funding_collected_usdt": 0.45,
        },
        "BBB/USDT:USDT": {
            "symbol": "BBB/USDT:USDT",
            "spot_symbol": "BBB/USDT",
            "qty": 2.0,
            "entry_spot_price": 50.0,
            "entry_perp_price": 50.02,
            "notional_usdt": 100.0,
            "margin_allocated_usdt": 99.0,
            "periods_held": 0,
            "funding_collected_usdt": 0.0,
        },
    }
    new_state = BotState(
        current_equity=420.0,
        carry_positions=carry_positions,
        carry_last_reconcile_ts="2025-01-02T00:00:00+00:00",
    )
    store.save(new_state)

    assert state_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []

    reloaded = store.load(default_equity=400.0)
    assert reloaded.carry_positions == carry_positions
    assert reloaded.carry_last_reconcile_ts == "2025-01-02T00:00:00+00:00"


# ---------------------------------------------------------------------------
# 2. Kế hoạch reconcile OPEN -> HOLD -> CLOSE
# ---------------------------------------------------------------------------


def test_reconcile_once_open_hold_close_plan(tmp_path: Path) -> None:
    """3 chu kỳ liên tiếp phải tạo OPEN cho 2 symbol funding cao, HOLD rồi CLOSE khi điểm giảm."""
    raw_dir = tmp_path / "raw"
    symbols = ["AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT"]
    _write_cache_bundle(raw_dir, "AAA/USDT:USDT", _HIGH_FUNDING)
    _write_cache_bundle(raw_dir, "BBB/USDT:USDT", _HIGH_FUNDING)
    _write_cache_bundle(raw_dir, "CCC/USDT:USDT", _LOW_FUNDING)

    carry_config = DynamicCarryConfig(
        top_k=2,
        entry_threshold=0.00010,
        exit_threshold=0.00002,
        min_hold_periods=9,
    )
    executor = _make_executor(tmp_path, raw_dir=raw_dir, symbols=symbols, carry_config=carry_config)
    expected_high = {"AAA/USDT:USDT", "BBB/USDT:USDT"}

    # Lần 1: state rỗng -> mở các symbol điểm cao nhất.
    plan_open = executor.reconcile_once()
    actions_open = {action.symbol: action.action for action in plan_open["actions"]}

    assert {
        sym for sym, act in actions_open.items() if act == "OPEN_DELTA_NEUTRAL"
    } == expected_high
    assert set(plan_open["active_positions"]) == expected_high
    assert actions_open["CCC/USDT:USDT"] == "SKIP"
    assert executor.state_store.file_path.exists()

    saved_payload = json.loads(executor.state_store.file_path.read_text(encoding="utf-8"))
    assert set(saved_payload["carry_positions"]) == expected_high

    # Lần 2: cùng dữ liệu, chưa đủ min_hold_periods -> các vị thế đang mở phải HOLD.
    plan_hold = executor.reconcile_once()
    actions_hold = {action.symbol: action.action for action in plan_hold["actions"]}
    for sym in expected_high:
        assert actions_hold[sym] == "HOLD"

    # Lần 3: điểm của vị thế đang giữ rơi dưới exit_threshold và đã đủ min_hold_periods.
    for pos in executor.state.carry_positions.values():
        pos["periods_held"] = carry_config.min_hold_periods
    low_universe = {
        sym: _aligned_frame(sym, funding_rate=_LOW_FUNDING) for sym in expected_high
    }
    plan_close = executor.reconcile_once(aligned_universe=low_universe)
    actions_close = {action.symbol: action.action for action in plan_close["actions"]}

    for sym in expected_high:
        assert actions_close[sym] == "CLOSE_DELTA_NEUTRAL"
        assert sym not in plan_close["active_positions"]
    assert executor.state.carry_positions == {}


# ---------------------------------------------------------------------------
# 3. Sức khỏe ký quỹ + bảng ASCII
# ---------------------------------------------------------------------------


def test_margin_health_check_and_ascii_console_table(tmp_path: Path) -> None:
    """Ký quỹ mỏng khi perp tăng vọt là CRITICAL/REBALANCE; vị thế chuẩn là HEALTHY; bảng thuần ASCII."""
    executor = _make_executor(tmp_path)

    thin_pos = {
        "qty": 1.0,
        "entry_spot_price": 100.0,
        "entry_perp_price": 100.0,
        "margin_allocated_usdt": 5.0,
    }
    ratio_low, health_low = executor.check_margin_health(thin_pos, spot_price=100.0, perp_price=200.0)
    assert ratio_low < executor.carry_config.maintenance_margin_ratio
    assert health_low in ("REBALANCE_NEEDED", "CRITICAL")

    healthy_pos = {
        "qty": 1.0,
        "entry_spot_price": 100.0,
        "entry_perp_price": 100.0,
        "margin_allocated_usdt": 25.0,
    }
    ratio_ok, health_ok = executor.check_margin_health(healthy_pos, spot_price=100.0, perp_price=100.0)
    assert ratio_ok == pytest.approx(0.25)
    assert health_ok == "HEALTHY"

    action = ReconcileAction(
        symbol="AAA/USDT:USDT",
        action="HOLD",
        reason="HOLD_MIN_PERIODS",
        score=0.001,
        f_ewma=0.0009,
        basis_pct=0.0005,
        basis_dev=0.0001,
        target_notional_usdt=150.0,
        spot_qty=1.5,
        perp_qty=1.5,
        margin_ratio=0.33,
        margin_health="HEALTHY",
    )
    table = executor.format_status_table(
        {
            "timestamp": "2024-01-03T11:00:00+00:00",
            "equity": 400.0,
            "actions": [action],
        }
    )
    assert isinstance(table, str)
    assert all(ord(ch) < 128 for ch in table)
    assert "HOLD" in table
    assert "AAA/USDT:USDT" in table


# ---------------------------------------------------------------------------
# 4. CLI run_live.py
# ---------------------------------------------------------------------------


def test_run_live_cli_dynamic_carry_paper_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI dynamic_carry --once không gọi mạng, trả mã 0 và ghi state JSON nguyên tử."""
    universe = {
        "BTC/USDT:USDT": _aligned_frame("BTC/USDT:USDT", funding_rate=_HIGH_FUNDING)
    }

    def fake_fetch(self: CarryLiveExecutor) -> dict[str, pd.DataFrame]:
        return {sym: frame.copy() for sym, frame in universe.items()}

    monkeypatch.setattr(
        CarryLiveExecutor,
        "fetch_latest_aligned_universe",
        fake_fetch,
    )

    state_file = tmp_path / "state.json"
    exit_code = run_live.main(
        [
            "--mode",
            "paper",
            "--strategy",
            "dynamic_carry",
            "--once",
            "--state-file",
            str(state_file),
        ]
    )

    assert exit_code == 0
    assert state_file.exists()
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert "carry_positions" in payload
    assert "carry_last_reconcile_ts" in payload


def test_parse_args_strategy_default_and_choices() -> None:
    """parse_args mặc định directional, nhận dynamic_carry và từ chối giá trị lạ."""
    assert run_live.parse_args([]).strategy == "directional"
    assert run_live.parse_args(["--strategy", "dynamic_carry"]).strategy == "dynamic_carry"

    with pytest.raises(SystemExit):
        run_live.parse_args(["--strategy", "not_a_strategy"])


# ---------------------------------------------------------------------------
# 5. Các nhánh an toàn
# ---------------------------------------------------------------------------


def test_fetch_latest_aligned_universe_empty_cache_raises_file_not_found(tmp_path: Path) -> None:
    """Cache rỗng và không có exchange phải ném đúng FileNotFoundError, không lỗi khác."""
    raw_dir = tmp_path / "empty_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    executor = _make_executor(tmp_path, raw_dir=raw_dir, symbols=["AAA/USDT:USDT"])

    with pytest.raises(FileNotFoundError):
        executor.fetch_latest_aligned_universe()


def test_reconcile_once_empty_or_partial_universe_does_not_crash(tmp_path: Path) -> None:
    """Universe rỗng hoặc thiếu symbol phải trả plan hợp lệ, không ném ngoại lệ."""
    executor = _make_executor(
        tmp_path,
        symbols=["AAA/USDT:USDT", "BBB/USDT:USDT"],
    )

    empty_plan = executor.reconcile_once(aligned_universe={})
    assert empty_plan["actions"] == []
    assert empty_plan["active_positions"] == {}

    partial_universe = {
        "AAA/USDT:USDT": _aligned_frame("AAA/USDT:USDT", funding_rate=_HIGH_FUNDING)
    }
    partial_plan = executor.reconcile_once(aligned_universe=partial_universe)

    assert isinstance(partial_plan["actions"], list)
    partial_actions = {action.symbol: action.action for action in partial_plan["actions"]}
    assert partial_actions["BBB/USDT:USDT"] == "SKIP"
