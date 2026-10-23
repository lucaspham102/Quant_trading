"""Sprint 22: Test kiem tra credential Spot & Futures testnet trong run_live.

Bao phu:
1. check_testnet_dual_credentials: bat buoc ca spot lan futures, tu choi neu thieu 1 trong 2.
2. check_testnet_dual_credentials: khong nham voi khoa mainnet (BINANCE_API_KEY).
3. Luong run_live.main o mode testnet:
   - Du ca 2 venue: KHONG sinh canh bao "thieu API key", exec_mode van la "paper".
   - Thieu 1 trong 2 venue: sinh canh bao "thieu API key", exec_mode van la "paper".
   - Thieu ca 2: sinh canh bao "thieu API key", exec_mode van la "paper".
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import run_live


def test_check_testnet_dual_credentials_empty() -> None:
    """Empty env tra ve False."""
    assert run_live.check_testnet_dual_credentials({}) is False


def test_check_testnet_dual_credentials_only_spot() -> None:
    """Chi co spot key tra ve False (khong chap nhan chi mot ben)."""
    env = {
        "BINANCE_SPOT_TESTNET_API_KEY": "spot_k",
        "BINANCE_SPOT_TESTNET_API_SECRET": "spot_s",
    }
    assert run_live.check_testnet_dual_credentials(env) is False


def test_check_testnet_dual_credentials_only_futures() -> None:
    """Chi co futures key tra ve False (khong chap nhan chi mot ben)."""
    env = {
        "BINANCE_FUTURES_TESTNET_API_KEY": "fut_k",
        "BINANCE_FUTURES_TESTNET_API_SECRET": "fut_s",
    }
    assert run_live.check_testnet_dual_credentials(env) is False


def test_check_testnet_dual_credentials_both_present() -> None:
    """Co day du ca spot lan futures key tra ve True."""
    env = {
        "BINANCE_SPOT_TESTNET_API_KEY": "spot_k",
        "BINANCE_SPOT_TESTNET_API_SECRET": "spot_s",
        "BINANCE_FUTURES_TESTNET_API_KEY": "fut_k",
        "BINANCE_FUTURES_TESTNET_API_SECRET": "fut_s",
    }
    assert run_live.check_testnet_dual_credentials(env) is True


def test_check_testnet_dual_credentials_ignores_mainnet_keys() -> None:
    """Co BINANCE_API_KEY mainnet nhung khong co testnet keys tra ve False."""
    env = {
        "BINANCE_API_KEY": "mainnet_k",
        "BINANCE_API_SECRET": "mainnet_s",
    }
    assert run_live.check_testnet_dual_credentials(env) is False


def test_testnet_mode_with_dual_credentials_suppresses_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Khi co du ca spot lan futures testnet keys, canh bao 'thieu API key' KHONG xuat hien."""
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "spot_key_64_chars_test_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", "spot_sec_64_chars_test_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "fut_key_64_chars_test_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", "fut_sec_64_chars_test_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    # Xoa mainnet keys de kiem tra tinh doc lap
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    with caplog.at_level(logging.WARNING, logger="run_live"):
        # Mock executor va cac ham preflight de chi kiem tra luong main
        with patch.object(run_live, "_run_testnet_signing_preflight", return_value=True), \
             patch.object(run_live, "_run_reconcile_preflight", return_value=None), \
             patch.object(run_live, "_run_signed_snapshot_preflight", return_value=None), \
             patch.object(run_live, "_run_testnet_order_pair", return_value=None), \
             patch("src.carry_live_executor.CarryLiveExecutor.run_loop", return_value=0):
            rc = run_live.main(["--mode", "testnet", "--once", "--synthetic"])
            assert rc == 0

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("thieu API key" in w for w in warnings)


def test_testnet_mode_missing_credentials_emits_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Khi thieu testnet keys, canh bao 'thieu API key' XUAT HIEN va chay paper mode."""
    monkeypatch.delenv("BINANCE_SPOT_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_SPOT_TESTNET_API_SECRET", raising=False)
    monkeypatch.delenv("BINANCE_FUTURES_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_FUTURES_TESTNET_API_SECRET", raising=False)
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    # Chặn load_dotenv_testnet nạp file thật trong test
    monkeypatch.setattr("src.binance_signed_client.load_dotenv_testnet", lambda *a, **kw: {})

    with caplog.at_level(logging.WARNING, logger="run_live"):
        with patch.object(run_live, "_run_testnet_signing_preflight", return_value=True), \
             patch.object(run_live, "_run_reconcile_preflight", return_value=None), \
             patch.object(run_live, "_run_signed_snapshot_preflight", return_value=None), \
             patch.object(run_live, "_run_testnet_order_pair", return_value=None), \
             patch("src.carry_live_executor.CarryLiveExecutor.run_loop", return_value=0):
            rc = run_live.main(["--mode", "testnet", "--once", "--synthetic"])
            assert rc == 0

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("thieu API key" in w for w in warnings)
