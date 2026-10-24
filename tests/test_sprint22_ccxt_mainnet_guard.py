"""Sprint 22: Unit tests cho chot chan bao mat CCXT chong ro ri lenh/key vao Mainnet.

Bao phu 4 kich ban bat buoc:
1. assert_exchange_credential_safety: Exchange co apiKey hoac secret ma tro vao URL mainnet -> RuntimeError (CRITICAL_SECURITY_ABORT).
   Dong thoi, exchange co creds ma target_endpoints rong -> RuntimeError.
2. assert_exchange_credential_safety: Exchange co apiKey/secret va moi endpoint deu chua 'testnet' -> Vuot qua an toan.
3. assert_exchange_credential_safety: Exchange khong co apiKey/secret tro vao mainnet public URL -> Vuot qua (phuc vu 720h warmup).
4. run_live.main o che do mode == 'testnet' tuyet doi KHONG truyen key mainnet (BINANCE_API_KEY, BYBIT_API_KEY) vao CCXT.
   Nguoc lai, o mode == 'live', CCXT nhan du credential de giao dich live.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import run_live
from run_live import assert_exchange_credential_safety


class MockExchange:
    def __init__(
        self,
        api_key: str | None = None,
        secret: str | None = None,
        urls: dict[str, Any] | None = None,
    ) -> None:
        self.apiKey = api_key
        self.secret = secret
        self.urls = urls if urls is not None else {}


def test_assert_exchange_credential_safety_raises_on_mainnet_with_creds() -> None:
    """1. Exchange co apiKey hoac secret ma tro vao endpoint mainnet -> RuntimeError."""
    # Truong hop 1: Co ca key va secret, endpoint tro vao mainnet api.binance.com
    bad_exchange_1 = MockExchange(
        api_key="real_mainnet_api_key",
        secret="real_mainnet_api_secret",
        urls={"api": {"public": "https://api.binance.com/api/v3", "private": "https://api.binance.com/api/v3"}},
    )
    with pytest.raises(RuntimeError, match="CRITICAL_SECURITY_ABORT"):
        assert_exchange_credential_safety(bad_exchange_1, "binance_spot")

    # Truong hop 2: Chi co secret (thieu api_key hoac nguoc lai) van phai chan
    bad_exchange_2 = MockExchange(
        api_key=None,
        secret="orphan_secret",
        urls={"api": {"fapiPublic": "https://fapi.binance.com/fapi/v1"}},
    )
    with pytest.raises(RuntimeError, match="CRITICAL_SECURITY_ABORT"):
        assert_exchange_credential_safety(bad_exchange_2, "binance_perp")

    # Truong hop 3: Co creds nhung urls api rong (khong co endpoint de kiem tra) -> Fail-closed abort
    bad_exchange_3 = MockExchange(
        api_key="key_xyz",
        secret="sec_xyz",
        urls={"api": {}},
    )
    with pytest.raises(RuntimeError, match="khong tim thay endpoint URL nao"):
        assert_exchange_credential_safety(bad_exchange_3, "binance_empty")


def test_assert_exchange_credential_safety_passes_on_testnet_with_creds() -> None:
    """2. Exchange co apiKey/secret va cac endpoint deu chua 'testnet' -> Hoan toan an toan."""
    good_testnet_exchange = MockExchange(
        api_key="testnet_key_123",
        secret="testnet_secret_456",
        urls={
            "api": {
                "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                "spot": "https://testnet.binance.vision/api/v3",
            }
        },
    )
    # Khong raise exception
    assert_exchange_credential_safety(good_testnet_exchange, "binance_testnet")


def test_assert_exchange_credential_safety_passes_on_mainnet_public_no_creds() -> None:
    """3. Exchange khong co apiKey/secret tro vao mainnet public URL -> Vuot qua (720h warmup feed)."""
    public_feed_exchange = MockExchange(
        api_key=None,
        secret=None,
        urls={
            "api": {
                "public": "https://api.binance.com/api/v3",
                "fapiPublic": "https://fapi.binance.com/fapi/v1",
            }
        },
    )
    # Khong raise exception
    assert_exchange_credential_safety(public_feed_exchange, "binance_public_feed")

    # Truong hop exchange la None (offline/synthetic)
    assert_exchange_credential_safety(None, "none_exchange")


def test_testnet_mode_does_not_pass_mainnet_keys_to_ccxt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """4. Mode testnet tuyet doi khong truyen BINANCE_API_KEY / BYBIT_API_KEY vao CCXT.

    Nguoc lai, mode live phai truyen day du.
    """
    monkeypatch.setenv("BINANCE_API_KEY", "LEAK_DANGER_BINANCE_KEY")
    monkeypatch.setenv("BINANCE_API_SECRET", "LEAK_DANGER_BINANCE_SECRET")
    monkeypatch.setenv("BYBIT_API_KEY", "LEAK_DANGER_BYBIT_KEY")
    monkeypatch.setenv("BYBIT_API_SECRET", "LEAK_DANGER_BYBIT_SECRET")

    # Set du ca testnet keys de check_testnet_dual_credentials tra ve True
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "testnet_spot_k")
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", "testnet_spot_s")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "testnet_fut_k")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", "testnet_fut_s")

    # Xoa bien PYTEST_CURRENT_TEST de chay nhanh ccxt init cua main()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    captured_testnet_configs: list[dict[str, Any]] = []

    def fake_testnet_factory(cfg: dict[str, Any]) -> MockExchange:
        captured_testnet_configs.append(dict(cfg))
        # Tra ve mock exchange tro vao testnet de pass qua assert_exchange_credential_safety
        return MockExchange(
            api_key=cfg.get("apiKey"),
            secret=cfg.get("secret"),
            urls={"api": {"public": "https://testnet.binance.vision/api/v3"}},
        )

    state_file = str(tmp_path / "bot_state.json")

    # Chay mode testnet
    with patch("ccxt.binance", side_effect=fake_testnet_factory), \
         patch("ccxt.binanceusdm", side_effect=fake_testnet_factory), \
         patch("ccxt.bybit", side_effect=fake_testnet_factory), \
         patch.object(run_live, "_run_testnet_signing_preflight", return_value=True), \
         patch.object(run_live, "_run_reconcile_preflight", return_value=None), \
         patch.object(run_live, "_run_signed_snapshot_preflight", return_value=None), \
         patch.object(run_live, "_run_testnet_order_pair", return_value=None), \
         patch("src.carry_live_executor.CarryLiveExecutor.run_loop", return_value=0):

        rc = run_live.main(["--mode", "testnet", "--once", "--state-file", state_file])
        assert rc == 0

    assert len(captured_testnet_configs) >= 2
    for cfg in captured_testnet_configs:
        assert "apiKey" not in cfg, f"Leak detected! Found apiKey in testnet config: {cfg}"
        assert "secret" not in cfg, f"Leak detected! Found secret in testnet config: {cfg}"

    # Bay gio kiem tra nguoc lai o mode live: CCXT PHAI nhan du apiKey/secret
    live_configs: list[dict[str, Any]] = []

    def fake_live_factory(cfg: dict[str, Any]) -> MockExchange:
        live_configs.append(dict(cfg))
        # URL chua 'testnet' de qua duoc assert safety trong test moi truong gia lap
        return MockExchange(
            api_key=cfg.get("apiKey"),
            secret=cfg.get("secret"),
            urls={"api": {"public": "https://testnet.binance.vision/api/v3"}},
        )

    with patch("ccxt.binance", side_effect=fake_live_factory), \
         patch("ccxt.binanceusdm", side_effect=fake_live_factory), \
         patch("ccxt.bybit", side_effect=fake_live_factory), \
         patch.object(run_live, "_run_reconcile_preflight", return_value=None), \
         patch("src.carry_live_executor.CarryLiveExecutor.run_loop", return_value=0):

        rc_live = run_live.main(["--mode", "live", "--once", "--state-file", state_file])
        assert rc_live == 0

    assert len(live_configs) >= 2
    for cfg in live_configs:
        assert cfg.get("apiKey") in ("LEAK_DANGER_BINANCE_KEY", "LEAK_DANGER_BYBIT_KEY")
        assert cfg.get("secret") in ("LEAK_DANGER_BINANCE_SECRET", "LEAK_DANGER_BYBIT_SECRET")


def test_main_exits_nonzero_when_ccxt_has_creds_in_testnet_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Item 1: Goi main(['--mode', 'testnet', ...]) voi CCXT gia mang apiKey -> exit code != 0 va CRITICAL_SECURITY_ABORT."""
    import logging
    caplog.set_level(logging.ERROR)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "spot_k")
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", "spot_s")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "fut_k")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", "fut_s")

    state_file = str(tmp_path / "bot_state.json")

    # Gia lap CCXT spot exchange mang credential trai phep o mode testnet
    leaked_exchange = MockExchange(
        api_key="leaked_testnet_key",
        secret="leaked_testnet_sec",
        urls={"api": {"public": "https://api.binance.com/api/v3"}},
    )

    with patch("ccxt.binance", return_value=leaked_exchange), \
         patch("ccxt.binanceusdm", return_value=MockExchange()), \
         patch.object(run_live, "_run_testnet_signing_preflight", return_value=True):

        exit_code = run_live.main(["--mode", "testnet", "--once", "--state-file", state_file])
        assert exit_code != 0
        assert exit_code == 1
        assert "CRITICAL_SECURITY_ABORT" in caplog.text


def test_main_exits_nonzero_on_baseline_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Item 7: BASELINE_* loi trong preflight duoc main() bat va tra exit code != 0."""
    import logging
    caplog.set_level(logging.ERROR)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "spot_k")
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", "spot_s")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_KEY", "fut_k")
    monkeypatch.setenv("BINANCE_FUTURES_TESTNET_API_SECRET", "fut_s")

    state_file = str(tmp_path / "bot_state.json")
    baseline_file = tmp_path / "corrupted_baseline.json"
    # Ghi baseline voi hash sai
    baseline_file.write_text(
        '{"credentials_sha256": "wrong_hash", "initial_recon_equity": 3000.0, "spot_balances": {}, "perp_positions": {}, "prices": {}, "created_at": "2026-01-01T00:00:00Z"}',
        encoding="utf-8",
    )

    clean_exchange = MockExchange(urls={"api": {"public": "https://testnet.binance.vision/api/v3"}})

    with patch("ccxt.binance", return_value=clean_exchange), \
         patch("ccxt.binanceusdm", return_value=clean_exchange), \
         patch.object(run_live, "_run_testnet_signing_preflight", return_value=True), \
         patch("src.binance_signed_client.BinanceSignedTestnetClient.fetch_exchange_reconcile_snapshot", return_value={"snapshot_healthy": True, "spot_cash_usd": 3000.0, "perp_wallet_balance_usd": 0.0, "perp_unrealized_pnl_usd": 0.0, "exchange_equity_usd": 3000.0, "spot_balances": {}, "perp_positions": {}, "perp_positions_detail": []}):

        exit_code = run_live.main([
            "--mode", "testnet",
            "--once",
            "--state-file", state_file,
            "--testnet-baseline-file", str(baseline_file),
            "--live-snapshot",
        ])
        assert exit_code != 0
        assert exit_code == 1
        assert "FATAL RUNTIME ABORT" in caplog.text
        assert "BASELINE_KEY_MISMATCH" in caplog.text
