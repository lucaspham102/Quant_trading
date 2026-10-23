"""Sprint 21 - Test suite cho UTC Epoch rebalance clock + reconciler quet toan ro.

Bao phu:
1. ``build_rebalance_mask`` giu nguyen 100% hanh vi cu (backward-compat) khi
   ``use_utc_epoch_schedule=False``.
2. Lich tai can bang neo theo bucket 48h tinh tu UTC epoch 1970.
3. Chong khuyet nen: mat dung cay nen tai moc giao bucket -> nen KE TIEP kich hoat,
   khong bi troi pha o cac moc sau.
4. ``phase_offset`` dich moc bucket va ket qua deterministic.
5. OR ``weights_changed`` van hoat dong khi bat epoch schedule.
6. ``reconcile_portfolio_state`` mac dinh quet ca 8 symbol cua ro, phat hien
   PHANTOM/ORPHAN ngoai vi the dang giu.
7. Canh bao ``UNTRACKED_MAINNET_ASSET_WARNING`` + BLOCKED tren mainnet, khong chan
   giao dich tren testnet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import build_rebalance_mask
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
from src.exchange_reconciler import (
    BLOCKED,
    PHANTOM_IN_STATE,
    SYNCED,
    UNTRACKED_MAINNET_ASSET_WARNING,
    UNTRACKED_TESTNET_PERP_WARNING,
    default_universe_symbols,
    reconcile_portfolio_state,
)

# Gia tham chieu gia lap (khong bao gio goi mang).
PRICES: dict[str, float] = {
    "BTC/USDT:USDT": 96000.0,
    "ETH/USDT:USDT": 4200.0,
    "SOL/USDT:USDT": 210.0,
    "BNB/USDT:USDT": 600.0,
    "DOGE/USDT:USDT": 0.20,
    "XRP/USDT:USDT": 0.55,
    "AVAX/USDT:USDT": 35.0,
    "LINK/USDT:USDT": 20.0,
}


# ---------------------------------------------------------------------------
# 1. Backward-compat (khong truyen timestamps)
# ---------------------------------------------------------------------------


def test_legacy_mask_96_48_keeps_old_behaviour() -> None:
    """Chu ky cu: mask[0] va mask[48] bat, mask[47] tat."""
    mask = build_rebalance_mask(96, 48)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[48]) is True
    assert bool(mask.iloc[47]) is False
    assert int(mask.sum()) == 2


def test_use_utc_epoch_schedule_defaults_to_false_without_timestamps() -> None:
    """Khong truyen timestamps -> ket qua giong het goi khong tham so nao."""
    legacy = build_rebalance_mask(100, 48)
    implicit = build_rebalance_mask(100, 48, use_utc_epoch_schedule=True)
    explicit = build_rebalance_mask(100, 48, use_utc_epoch_schedule=False)
    np.testing.assert_array_equal(implicit.to_numpy(), legacy.to_numpy())
    np.testing.assert_array_equal(explicit.to_numpy(), legacy.to_numpy())
    expected = np.arange(100) % 48 == 0
    expected[0] = True
    np.testing.assert_array_equal(legacy.to_numpy(), expected)


# ---------------------------------------------------------------------------
# 2. UTC epoch bucket schedule
# ---------------------------------------------------------------------------


def _utc_hours(periods: int = 120) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01 00:00", periods=periods, freq="h", tz="UTC")


def test_epoch_schedule_fires_on_48h_utc_boundaries() -> None:
    """Chuoi 120h tu 2024-01-01 00:00 UTC: bat tai 00:00 ngay 1, 2 va 4."""
    ts = _utc_hours()
    mask = build_rebalance_mask(
        len(ts),
        48,
        index=ts,
        timestamps=ts,
        use_utc_epoch_schedule=True,
    )
    assert bool(mask.loc[pd.Timestamp("2024-01-01 00:00", tz="UTC")]) is True
    assert bool(mask.loc[pd.Timestamp("2024-01-02 00:00", tz="UTC")]) is True
    assert bool(mask.loc[pd.Timestamp("2024-01-04 00:00", tz="UTC")]) is True
    # Nen ngay truoc moc phai tat (khong bat som).
    assert bool(mask.loc[pd.Timestamp("2024-01-01 23:00", tz="UTC")]) is False
    assert bool(mask.loc[pd.Timestamp("2024-01-03 23:00", tz="UTC")]) is False
    assert int(mask.sum()) == 3


def test_epoch_schedule_survives_missing_boundary_candle() -> None:
    """CHONG KHUYET NEN: xoa cay nen tai moc giao bucket -> nen ke tiep bat, khong troi pha."""
    ts = _utc_hours()
    ts_gap = ts.delete(24)  # xoa dung 2024-01-02 00:00 UTC
    assert pd.Timestamp("2024-01-02 00:00", tz="UTC") not in ts_gap

    mask = build_rebalance_mask(
        len(ts_gap),
        48,
        index=ts_gap,
        timestamps=ts_gap,
        use_utc_epoch_schedule=True,
    )
    on_labels = [str(label) for label in ts_gap[mask.to_numpy()]]
    assert on_labels == [
        "2024-01-01 00:00:00+00:00",
        "2024-01-02 01:00:00+00:00",  # nen ngay sau moc bi mat
        "2024-01-04 00:00:00+00:00",  # moc sau van dung pha epoch
    ]
    assert int(mask.sum()) == 3


def test_epoch_schedule_phase_offset_shifts_boundaries_deterministically() -> None:
    """phase_offset=24 dich moc bucket 24 gio va ket qua lap lai y het."""
    ts = _utc_hours()
    kwargs = dict(index=ts, timestamps=ts, use_utc_epoch_schedule=True, phase_offset=24)
    shifted = build_rebalance_mask(len(ts), 48, **kwargs)
    again = build_rebalance_mask(len(ts), 48, **kwargs)

    np.testing.assert_array_equal(shifted.to_numpy(), again.to_numpy())
    on_labels = [str(label) for label in ts[shifted.to_numpy()]]
    assert on_labels == [
        "2024-01-01 00:00:00+00:00",
        "2024-01-03 00:00:00+00:00",
        "2024-01-05 00:00:00+00:00",
    ]


def test_epoch_schedule_with_short_timestamps_does_not_crash() -> None:
    """timestamps ngan hon n_bars -> khong crash, giu do dai mask va nen dau bat."""
    ts = _utc_hours(200)
    short_ts = ts[:50]
    mask = build_rebalance_mask(
        120,
        48,
        index=pd.RangeIndex(120),
        timestamps=short_ts,
        use_utc_epoch_schedule=True,
    )
    assert len(mask) == 120
    assert bool(mask.iloc[0]) is True
    # Phan duoi khong neo epoch duoc -> lui ve lich modulo local.
    assert bool(mask.iloc[96]) is True


def test_epoch_schedule_unions_weights_changed() -> None:
    """weights_changed van duoc OR vao mask khi bat epoch schedule."""
    ts = _utc_hours()
    changed = pd.Series(False, index=ts)
    changed.iloc[10] = True
    mask = build_rebalance_mask(
        len(ts),
        48,
        index=ts,
        timestamps=ts,
        use_utc_epoch_schedule=True,
        weights_changed=changed,
    )
    assert bool(mask.iloc[10]) is True
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[24]) is True


# ---------------------------------------------------------------------------
# 3. Reconciler quet toan ro + canh bao mainnet
# ---------------------------------------------------------------------------


def _local(equity: float = 3000.0, **positions: dict) -> dict:
    carry = positions.pop("carry_positions", {})
    overlay = positions.pop("overlay_positions", {})
    return {
        "carry_positions": carry,
        "overlay_positions": overlay,
        "current_equity": equity,
    }


def _snapshot(equity: float = 3000.0, **fields) -> dict:
    snap = {
        "spot_balances": {},
        "perp_positions": {},
        "exchange_equity_usd": equity,
    }
    snap.update(fields)
    return snap


def test_reconciler_default_scans_full_eight_symbol_universe() -> None:
    """Mac dinh quet toan ro LIQUID_8_SYMBOLS, environment testnet."""
    report = reconcile_portfolio_state(
        _local(), _snapshot(), PRICES, EXCHANGE_LOT_SPECS
    )
    assert len(report["items"]) == 8
    assert len(report["universe_symbols"]) == 8
    assert report["universe_symbols"] == sorted(default_universe_symbols())
    assert report["environment"] == "testnet"


def test_reconciler_empty_state_and_snapshot_are_all_synced() -> None:
    """local rong + snapshot rong -> toan bo item SYNCED, report SYNCED."""
    report = reconcile_portfolio_state({}, {}, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == SYNCED
    assert all(item["status"] == SYNCED for item in report["items"])
    assert report["orphans"] == []
    assert report["phantoms"] == []
    assert report["blocked"] is False


def test_reconciler_testnet_untracked_perp_is_counted_but_not_blocking() -> None:
    """Testnet: perp ngoai ro chi thong ke, KHONG chan giao dich."""
    snapshot = _snapshot(perp_positions={"ADA/USDT:USDT": 10.0})
    report = reconcile_portfolio_state(
        _local(), snapshot, PRICES, EXCHANGE_LOT_SPECS, environment="testnet"
    )
    assert report["untracked_perp_count"] == 1
    assert report["blocked"] is False
    # Khong chan giao dich (chinh sach faucet) nhung phai CANH BAO ro rang,
    # khong duoc im lang tra ve SYNCED.
    assert report["warnings"] == [UNTRACKED_TESTNET_PERP_WARNING]
    assert report["untracked_testnet_perp_warning"] is True
    assert report["status"] != BLOCKED


def test_reconciler_mainnet_untracked_perp_blocks_trading() -> None:
    """Mainnet: perp ngoai ro -> BLOCKED + UNTRACKED_MAINNET_ASSET_WARNING."""
    snapshot = _snapshot(perp_positions={"ADA/USDT:USDT": 10.0})
    report = reconcile_portfolio_state(
        _local(), snapshot, PRICES, EXCHANGE_LOT_SPECS, environment="mainnet"
    )
    assert report["status"] == "BLOCKED"
    assert report["blocked"] is True
    assert UNTRACKED_MAINNET_ASSET_WARNING in report["warnings"]
    assert report["trading_allowed"] is False
    assert report["untracked_perp_count"] == 1


def test_reconciler_mainnet_untracked_spot_warns_without_blocking() -> None:
    """Mainnet: chi co spot la (khong perp la) -> canh bao nhung KHONG BLOCKED."""
    snapshot = _snapshot(spot_balances={"XYZ": 5.0})
    report = reconcile_portfolio_state(
        _local(), snapshot, PRICES, EXCHANGE_LOT_SPECS, environment="mainnet"
    )
    assert UNTRACKED_MAINNET_ASSET_WARNING in report["warnings"]
    assert report["status"] != BLOCKED
    assert report["blocked"] is False
    assert report["untracked_spot_count"] == 1
    assert report["untracked_spot_assets"]["XYZ"] == pytest.approx(5.0)


def test_reconciler_mainnet_ignored_spot_asset_produces_no_warning() -> None:
    """Mainnet: spot la nam trong ignored_spot_assets -> khong canh bao."""
    snapshot = _snapshot(spot_balances={"XYZ": 5.0})
    report = reconcile_portfolio_state(
        _local(),
        snapshot,
        PRICES,
        EXCHANGE_LOT_SPECS,
        environment="mainnet",
        ignored_spot_assets=["XYZ"],
    )
    assert report["warnings"] == []
    assert report["untracked_mainnet_asset_warning"] is False
    assert report["untracked_spot_assets_flagged"] == {}


def test_reconciler_invalid_environment_raises_value_error() -> None:
    """environment la -> ValueError (fail-closed)."""
    with pytest.raises(ValueError, match="environment must be"):
        reconcile_portfolio_state(
            _local(), _snapshot(), PRICES, EXCHANGE_LOT_SPECS, environment="staging"
        )


def test_reconciler_detects_phantom_on_symbol_without_open_position() -> None:
    """Local giu LINK nhung san khong co -> PHANTOM (chung minh quet toan ro)."""
    local = _local(
        carry_positions={
            "LINK/USDT:USDT": {"qty": 5.0, "leg2_hedged_qty": 5.0},
        }
    )
    report = reconcile_portfolio_state(local, _snapshot(), PRICES, EXCHANGE_LOT_SPECS)
    link_item = next(it for it in report["items"] if it["symbol"] == "LINK/USDT:USDT")
    assert link_item["status"] == PHANTOM_IN_STATE
    assert "LINK/USDT:USDT" in report["phantoms"]
    # 7 ma con lai khong giu vi the va van duoc quet (SYNCED).
    assert len(report["items"]) == 8
    assert sum(1 for it in report["items"] if it["status"] == SYNCED) == 7
