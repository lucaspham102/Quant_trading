"""Sprint 21 - Test suite cho burn-in 3 chu ky tren Binance Testnet (offline).

Bao phu:
1. Cong 720h warmup.
2. Gioi han snapshot doi soat dung ro giao dich (tai san faucet tach rieng).
3. ``run_multi_cycle_burnin`` chay du 3 chu ky voi mock transport deterministic
   (khong cham mang) -> PASS, cycle1 toan ro SYNCED, cycle3 zero residual.
4. Ghi file evidence markdown + log JSONL (parse duoc tung dong).
5. Cycle 1 that bai -> verdict FAIL va ``error`` khong rong.
6. ``_scrub_secrets`` che gia tri credential gia.

Luu y: khong co network that; toan bo khoa la chuoi GIA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_testnet_multi_cycle_burnin import (
    UNIVERSE_SYMBOLS,
    _scrub_secrets,
    run_multi_cycle_burnin,
    scope_snapshot_to_universe,
    verify_warmup_gate,
)
from scripts.run_testnet_live_roundtrip import build_mock_transport
from src.binance_signed_client import BinanceSignedTestnetClient


# ---------------------------------------------------------------------------
# 1. warmup gate
# ---------------------------------------------------------------------------


def test_warmup_gate_720h_passes() -> None:
    gate = verify_warmup_gate(720)
    assert gate["ok"] is True
    assert gate["gate"] == "720h"
    assert gate["warmup_hours"] == pytest.approx(720.0)
    assert gate["required_hours"] == pytest.approx(720.0)


def test_warmup_gate_below_720h_fails() -> None:
    gate = verify_warmup_gate(500)
    assert gate["ok"] is False
    assert gate["gate"] == "720h"
    assert gate["warmup_hours"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 2. scope_snapshot_to_universe
# ---------------------------------------------------------------------------


def test_scope_snapshot_keeps_universe_and_reports_untracked_spot() -> None:
    snapshot = {"spot_balances": {"BTC": 1.0, "XYZ": 3.0}}
    scoped = scope_snapshot_to_universe(snapshot)
    assert scoped["spot_balances"] == {"BTC": 1.0}
    assert scoped["untracked_spot_assets"] == {"XYZ": 3.0}
    assert scoped["reconcile_symbol_scope"] == list(UNIVERSE_SYMBOLS)
    # Snapshot goc khong bi mutate.
    assert snapshot["spot_balances"] == {"BTC": 1.0, "XYZ": 3.0}


def test_scope_snapshot_treats_cash_asset_as_untracked_documented() -> None:
    """Hanh vi thuc te cua src: USDT (tien mat) cung bi xep vao untracked khi con > 0.

    Client that da tach ``spot_cash_usd`` va khong dua cash vao ``spot_balances``,
    nen tren luong chay that USDT khong xuat hien o day. Test nay ghi lai dung
    hanh vi hien tai cua helper de tranh hieu sai.
    """
    snapshot = {"spot_balances": {"BTC": 1.0, "USDT": 100.0, "XYZ": 3.0}}
    scoped = scope_snapshot_to_universe(snapshot)
    assert scoped["spot_balances"] == {"BTC": 1.0}
    assert scoped["untracked_spot_assets"] == {"USDT": 100.0, "XYZ": 3.0}


# ---------------------------------------------------------------------------
# 3-4. run_multi_cycle_burnin voi mock transport
# ---------------------------------------------------------------------------


def _mock_client() -> BinanceSignedTestnetClient:
    """Client dry_run=False nhung transport offline deterministic (khong mang)."""
    return BinanceSignedTestnetClient(
        dry_run=False,
        recv_window=10000,
        spot_api_key="mock-spot-key-0001",
        spot_api_secret="mock-spot-secret",
        futures_api_key="mock-futures-key-0001",
        futures_api_secret="mock-futures-secret",
        autoload_dotenv=False,
        transport=build_mock_transport(),
    )


def test_multi_cycle_burnin_passes_offline_with_mock_transport(tmp_path: Path) -> None:
    evidence = tmp_path / "ev.md"
    log = tmp_path / "log.jsonl"
    result = run_multi_cycle_burnin(
        client=_mock_client(),
        write_files=True,
        evidence_path=str(evidence),
        log_path=str(log),
    )

    assert result["verdict"] == "PASS"
    assert result["cycle_count"] == 3
    assert len(result["cycles"]) == 3
    assert all(bool(cycle["ok"]) for cycle in result["cycles"])
    assert result.get("error") is None

    cycle1, cycle2, cycle3 = result["cycles"]
    assert cycle1["detail"]["all_8_symbols_synced"] is True
    # Sprint 22 Phase 10: pure_cash_and_margin_usd strictly stablecoins + margin
    assert cycle1["detail"]["pure_cash_and_margin_usd"] == pytest.approx(10000.0)
    assert cycle1["detail"]["reconcile_equity_usd"] == pytest.approx(10000.0)
    assert cycle1["detail"]["pure_cash_and_margin_usd"] < 100000.0
    assert cycle3["detail"]["equity_gap_usd"] == pytest.approx(0.0)
    assert cycle3["detail"]["zero_residual"] is True

    for key in (
        "spot_entry_order_id",
        "perp_entry_order_id",
        "spot_exit_order_id",
        "perp_exit_order_id",
    ):
        cycle = cycle2 if "entry" in key else cycle3
        assert cycle["detail"][key] is not None, key


def test_multi_cycle_burnin_writes_evidence_and_parseable_jsonl(tmp_path: Path) -> None:
    evidence = tmp_path / "ev.md"
    log = tmp_path / "log.jsonl"
    result = run_multi_cycle_burnin(
        client=_mock_client(),
        write_files=True,
        evidence_path=str(evidence),
        log_path=str(log),
    )
    assert result["verdict"] == "PASS"

    assert evidence.is_file() and evidence.stat().st_size > 0
    evidence_text = evidence.read_text(encoding="utf-8")
    assert "Sprint 21" in evidence_text
    assert "PASS" in evidence_text

    assert log.is_file() and log.stat().st_size > 0
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "log JSONL phai co it nhat 1 dong"
    for line in lines:
        record = json.loads(line)  # phai parse duoc tung dong
        assert "verdict" in record
        assert record["verdict"] == "PASS"


def test_multi_cycle_burnin_fails_when_cycle1_has_no_valid_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_client()

    def _empty_book(symbol: str, venue: str = "spot", timeout_sec: float = 5.0) -> dict:
        return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "raw": {}}

    monkeypatch.setattr(client, "fetch_book_ticker", _empty_book)
    result = run_multi_cycle_burnin(
        client=client,
        write_files=False,
        evidence_path=str(tmp_path / "ev.md"),
        log_path=str(tmp_path / "log.jsonl"),
    )

    assert result["verdict"] == "FAIL"
    assert result["error"]
    assert "Cycle 1" in result["error"]
    assert not (tmp_path / "ev.md").exists()
    assert not (tmp_path / "log.jsonl").exists()


# ---------------------------------------------------------------------------
# 5. _scrub_secrets
# ---------------------------------------------------------------------------


def test_scrub_secrets_redacts_long_fake_secret(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "fake-secret-value-1234567890"
    text = f"api_secret={secret} | other=keep"
    scrubbed = _scrub_secrets(text, [secret])
    assert secret not in scrubbed
    assert "[REDACTED]" in scrubbed
    assert "other=keep" in scrubbed

    # Khong bao gio in secret ra stdout.
    print(scrubbed)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert "[REDACTED]" in captured.out


def test_scrub_secrets_ignores_too_short_values() -> None:
    short = "abc123"
    text = f"value={short}"
    assert _scrub_secrets(text, [short]) == text


def test_cycle1_pure_cash_and_margin_ignores_faucet_coin_mtm() -> None:
    """Sprint 22 Phase 10: pure_cash_and_margin_usd khong bi phong dai 486M USD boi Faucet coin."""
    from scripts.run_testnet_multi_cycle_burnin import _cycle1_bootstrap

    class FakeClient:
        def sync_server_time(self, venue="both"):
            return {"spot": 100, "futures": 100}

        def fetch_book_ticker(self, symbol, venue):
            return {"symbol": symbol, "bid": "80000.0", "ask": "80000.0", "mid": 80000.0}

        def fetch_exchange_reconcile_snapshot(self, symbols, prices, universe_symbols=None):
            return {
                "snapshot_healthy": True,
                "spot_balances_raw": {
                    "USDT": 9999.98,
                    "FDUSD": 10000.0,
                    "TUSD": 10000.0,
                    "USDC": 10000.0,
                    "USD": 10000.0,
                    "DOGE": 5501.0,
                    "XRP": 341.0,
                    "BTC": 1.0,
                },
                "spot_balances": {"BTC": 1.0, "DOGE": 5501.0, "XRP": 341.0},
                "spot_cash_usd": 486000000.0,  # Faucet coin evaluation at mid_price
                "pure_cash_usd": 49999.98,
                "perp_wallet_balance_usd": 4999.14,
                "perp_unrealized_pnl_usd": 0.0,
                "exchange_equity_usd": 486004999.14,
                "perp_positions": {},
                "perp_positions_detail": [],
                "untracked_spot_assets": {},
            }

    fake = FakeClient()
    fake.spot_api_key = "a" * 64
    fake.spot_api_secret = "a" * 64
    fake.futures_api_key = "a" * 64
    fake.futures_api_secret = "a" * 64

    res = _cycle1_bootstrap(fake, symbol="BTC/USDT:USDT", universe=list(UNIVERSE_SYMBOLS), warmup_hours=720.0)
    detail = res["detail"]
    assert detail["bootstrap_equity_usd"] == pytest.approx(486004999.14)
    # pure_cash_and_margin_usd must strictly be 49999.98 + 4999.14 = 54999.12 USD
    assert detail["pure_cash_and_margin_usd"] == pytest.approx(54999.12)
    assert detail["reconcile_equity_usd"] == pytest.approx(54999.12)
    assert detail["pure_cash_and_margin_usd"] < 100000.0
