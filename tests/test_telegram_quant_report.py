"""Unit tests cho Telegram Quant Report chuẩn Institutional Quant Desk."""
import html
from typing import Any

import pytest

from src.notifier import (
    TelegramNotifier,
    diff_cycle_executions,
    format_tournament_telegram_report,
    strip_html_tags,
)


def _build_mock_records() -> list[dict[str, Any]]:
    return [
        {
            "cycle": 121,
            "timestamp": "2026-11-05T13:56:45+00:00",
            "profile": "regime_omni_1k",
            "pure_trading_equity_usd": 989.61,
            "yield_stacked_equity_usd": 989.61,
            "max_dd_pct": 1.25,
            "reconcile_status": "SYNCED",
            "recon_drift_usd": -0.0001,
            "exchange_recon_equity_usd": 989.61,
            "funding_pnl_usd": 0.48,
            "basis_pnl_usd": 0.04,
            "overlay_pnl_usd": -6.82,
            "trend_pnl_usd": 0.0,
            "fees_paid_usd": 3.17,
            "futures_leverage": 0.27,
            "overlay_long_usd": 124.05,
            "overlay_short_usd": 131.78,
            "trend_gross_usd": 0.0,
        },
        {
            "cycle": 121,
            "timestamp": "2026-11-05T13:56:45+00:00",
            "profile": "flexible_regime_1k",
            "pure_trading_equity_usd": 995.20,
            "yield_stacked_equity_usd": 995.20,
            "max_dd_pct": 0.80,
            "reconcile_status": "NOT_APPLICABLE",
        },
        {
            "cycle": 121,
            "timestamp": "2026-11-05T13:56:45+00:00",
            "profile": "carry_max_eta_1k",
            "pure_trading_equity_usd": 991.10,
            "yield_stacked_equity_usd": 991.10,
            "max_dd_pct": 0.95,
            "reconcile_status": "NOT_APPLICABLE",
        },
        {
            "cycle": 121,
            "timestamp": "2026-11-05T13:56:45+00:00",
            "profile": "baseline_1k",
            "pure_trading_equity_usd": 1000.00,
            "yield_stacked_equity_usd": 1000.00,
            "max_dd_pct": 0.0,
            "reconcile_status": "NOT_APPLICABLE",
        },
    ]


def test_diff_cycle_executions() -> None:
    """Kiểm tra độ chính xác của hàm trích xuất lệnh giao dịch giữa 2 chu kỳ."""
    prev = {
        "BNB/USDT:USDT|overlay": {
            "symbol": "BNB/USDT:USDT",
            "kind": "overlay",
            "qty": 0.15,
            "entry_price": 650.0,
            "signed_usd": 97.5,
        },
        "SOL/USDT:USDT|overlay": {
            "symbol": "SOL/USDT:USDT",
            "kind": "overlay",
            "qty": -1.10,
            "entry_price": 95.0,
            "signed_usd": -104.5,
        },
        "BTC/USDT:USDT": {
            "symbol": "BTC/USDT:USDT",
            "kind": "carry",
            "qty": 0.002,
            "entry_price": 76000.0,
            "signed_usd": 152.0,
        },
    }
    curr = {
        "BNB/USDT:USDT|overlay": {
            "symbol": "BNB/USDT:USDT",
            "kind": "overlay",
            "qty": 0.19,
            "entry_price": 652.88,
            "signed_usd": 124.05,
        },
        "SOL/USDT:USDT|overlay": {
            "symbol": "SOL/USDT:USDT",
            "kind": "overlay",
            "qty": -1.40,
            "entry_price": 94.13,
            "signed_usd": -131.78,
        },
        "BTC/USDT:USDT": {
            "symbol": "BTC/USDT:USDT",
            "kind": "carry",
            "qty": 0.002,
            "entry_price": 76000.0,
            "signed_usd": 152.0,
        },
    }

    diffs = diff_cycle_executions(prev, curr)
    assert len(diffs) == 2

    bnb_diff = next(d for d in diffs if d["symbol"] == "BNB")
    assert bnb_diff["side"] == "BUY"
    assert bnb_diff["sleeve"] == "Overlay"
    assert bnb_diff["delta_qty"] == pytest.approx(0.04, abs=1e-6)
    assert bnb_diff["old_qty"] == pytest.approx(0.15, abs=1e-6)
    assert bnb_diff["new_qty"] == pytest.approx(0.19, abs=1e-6)
    assert bnb_diff["fill_price"] == pytest.approx(652.88, abs=1e-2)

    sol_diff = next(d for d in diffs if d["symbol"] == "SOL")
    assert sol_diff["side"] == "SELL"
    assert sol_diff["sleeve"] == "Overlay"
    assert sol_diff["delta_qty"] == pytest.approx(-0.30, abs=1e-6)
    assert sol_diff["old_qty"] == pytest.approx(-1.10, abs=1e-6)
    assert sol_diff["new_qty"] == pytest.approx(-1.40, abs=1e-6)


def test_telegram_report_with_cycle_executions() -> None:
    """Kiểm tra khối ⚡ KHỚP LỆNH TRONG CHU KỲ khi có giao dịch mới."""
    records = _build_mock_records()
    prev_positions = {
        "BNB/USDT:USDT|overlay": {
            "symbol": "BNB/USDT:USDT",
            "kind": "overlay",
            "qty": 0.15,
            "entry_price": 650.0,
            "signed_usd": 97.5,
        },
        "SOL/USDT:USDT|overlay": {
            "symbol": "SOL/USDT:USDT",
            "kind": "overlay",
            "qty": -1.10,
            "entry_price": 95.0,
            "signed_usd": -104.5,
        },
    }
    curr_positions = {
        "BTC/USDT:USDT": {
            "symbol": "BTC/USDT:USDT",
            "kind": "carry",
            "qty": 0.002,
            "spot_qty": 0.002,
            "perp_qty": -0.002,
            "entry_price": 76760.0,
            "signed_usd": 153.52,
        },
        "BNB/USDT:USDT|overlay": {
            "symbol": "BNB/USDT:USDT",
            "kind": "overlay",
            "qty": 0.19,
            "entry_price": 652.88,
            "signed_usd": 124.05,
        },
        "SOL/USDT:USDT|overlay": {
            "symbol": "SOL/USDT:USDT",
            "kind": "overlay",
            "qty": -1.40,
            "entry_price": 94.13,
            "signed_usd": -131.78,
        },
    }
    live_ledger = {
        "pure_trading_equity_usd": 989.61,
        "yield_stacked_equity_usd": 989.61,
        "reconcile_status": "SYNCED",
        "recon_drift_usd": -0.0001,
        "exchange_recon_equity_usd": 989.61,
        "funding_pnl_usd": 0.48,
        "basis_pnl_usd": 0.04,
        "overlay_pnl_usd": -6.82,
        "trend_pnl_usd": 0.0,
        "fees_paid_usd": 3.17,
        "futures_leverage": 0.27,
        "overlay_long_usd": 124.05,
        "overlay_short_usd": 131.78,
        "positions_snapshot": curr_positions,
    }

    report = format_tournament_telegram_report(
        records,
        live_profile="regime_omni_1k",
        live_ledger=live_ledger,
        prev_positions=prev_positions,
        reasons=["TRADE_OR_POSITION_CHANGED"],
    )

    # 1. Khối Executions có mặt và chứa đủ thông tin
    assert "⚡ KHỚP LỆNH TRONG CHU KỲ (CYCLE EXECUTIONS)" in report
    assert "[Overlay]" in report
    assert "BUY BNB" in report and "Δ+0.0400" in report and "(+0.1500 -> +0.1900)" in report
    assert "SELL SOL" in report and "Δ-0.3000" in report and "(-1.1000 -> -1.4000)" in report
    assert "@ $652.88" in report or "@ $94.13" in report

    # 2. Khối Vốn & Đối soát
    assert "Vốn & Đối soát:" in report
    assert "Pure Eq: $989.61" in report
    assert "Yield Eq: $989.61" in report
    assert "Exch Eq: $989.61" in report
    assert "Drift:" in report and "bps" in report

    # 3. Khối Net PnL Attribution
    assert "Net PnL Attribution:" in report
    assert "Carry Net: +0.52$" in report
    assert "Funding: +0.48$" in report
    assert "Basis: +0.04$" in report
    assert "Overlay: -6.82$" in report
    assert "Fees: -$3.17" in report

    # 4. Khối Risk & Margin
    assert "Risk & Margin Telemetry:" in report
    assert "Fut Leverage: 0.27x / 3.0x" in report
    assert "Net Delta:" in report
    assert "Gross Exp:" in report

    # 5. Khối Danh mục đang mở (Active Positions)
    assert "📌 DANH MỤC ĐANG MỞ (ACTIVE POSITIONS)" in report
    assert "BTC" in report and "[Carry] Spot +0.0020 / Perp -0.0020" in report
    assert "BNB" in report and "[Ovly ] Perp LONG  +0.1900" in report
    assert "SOL" in report and "[Ovly ] Perp SHORT -1.4000" in report


def test_telegram_report_no_executions_hold() -> None:
    """Kiểm tra khối ⚡ Executions khi chu kỳ định kỳ KHÔNG có lệnh mới (Hold positions)."""
    records = _build_mock_records()
    positions = {
        "BTC/USDT:USDT": {
            "symbol": "BTC/USDT:USDT",
            "kind": "carry",
            "qty": 0.002,
            "spot_qty": 0.002,
            "perp_qty": -0.002,
            "entry_price": 76760.0,
            "signed_usd": 153.52,
        },
    }
    live_ledger = {
        "pure_trading_equity_usd": 989.61,
        "yield_stacked_equity_usd": 989.61,
        "positions_snapshot": positions,
    }

    report = format_tournament_telegram_report(
        records,
        live_profile="regime_omni_1k",
        live_ledger=live_ledger,
        prev_positions=positions,  # Không đổi
        reasons=["PERIODIC_4C"],
    )

    assert "NONE (Hold Positions)" in report
    assert "⚡ <b>Executions:</b> <code>NONE (Hold Positions)</code>" in report
    assert "⚡ KHỚP LỆNH TRONG CHU KỲ" not in report


def test_telegram_report_html_safety() -> None:
    """Kiểm tra tính an toàn HTML, không vỡ thẻ và strip_html_tags hoạt động mượt mà."""
    records = _build_mock_records()
    live_ledger = {
        "pure_trading_equity_usd": 989.61,
        "positions_snapshot": {},
    }

    report = format_tournament_telegram_report(
        records,
        live_profile="regime_omni_1k",
        live_ledger=live_ledger,
    )

    # Thẻ mở và thẻ đóng cân bằng
    for tag in ("b", "code", "pre"):
        open_count = report.count(f"<{tag}>")
        close_count = report.count(f"</{tag}>")
        assert open_count == close_count, f"Mismatched tag: <{tag}> ({open_count} vs {close_count})"

    # Plain text fallback sạch sẽ
    plain = strip_html_tags(report)
    assert "<" not in plain and ">" not in plain
    assert "QUANT TOURNAMENT" in plain
    assert "BẢNG XẾP HẠNG TOURNAMENT" in plain
