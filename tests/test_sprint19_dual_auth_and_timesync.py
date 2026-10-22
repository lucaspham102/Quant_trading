"""Test Sprint 19 - Dual (Spot/Futures) credentials, dotenv loading, time sync.

Bao phu:
1. `load_dotenv_testnet`: BOM/utf-8-sig, dong trong, comment `#`, tien to `export`,
   gia tri quote, gia tri rong, va injection qua tham so `environ={}`.
2. Khong ghi de bien da ton tai (tru khi `override=True`).
3. File khong ton tai -> tra {} (khong raise).
4. Tach biet khoa Spot/Futures + thu tu uu tien (explicit > generic > env rieng > env legacy).
5. `credential_status()` chi tra gia tri da che, KHONG BAO GIO lo raw key/secret.
6. `sync_server_time` tinh offset = serverTime - local_ms cho tung venue.
7. `sign_params` cong `*_time_offset_ms` cua dung venue, va `timestamp_ms` tuong minh thang offset.
8. Chan doan khoa Futures dung nham (ma -2014/-2015) trong check_account_access + snapshot.

Tat ca deu offline, khong doc `.env.testnet` that.
"""

from __future__ import annotations

import pytest

from src.binance_signed_client import (
    ENV_FUTURES_API_KEY,
    ENV_FUTURES_API_SECRET,
    ENV_SPOT_API_KEY,
    ENV_SPOT_API_SECRET,
    FUTURES_KEY_INVALID_CODE,
    FUTURES_KEY_INVALID_HINT,
    BinanceSignedTestnetClient,
    load_dotenv_testnet,
    sign_binance_payload,
)

_FAKE_SPOT_KEY = "fake-spot-key-0001"
_FAKE_SPOT_SECRET = "fake-spot-secret-0001"
_FAKE_FUT_KEY = "fake-futures-key-0002"
_FAKE_FUT_SECRET = "fake-futures-secret-0002"

_ENV_NAMES = (
    "BINANCE_TESTNET_API_KEY",
    "BINANCE_TESTNET_API_SECRET",
    ENV_SPOT_API_KEY,
    ENV_SPOT_API_SECRET,
    ENV_FUTURES_API_KEY,
    ENV_FUTURES_API_SECRET,
)


def make_transport(routes):
    """Boc 1 ham routes thanh transport fake (khong bao gio cham mang)."""

    def transport(method, url, params=None, headers=None, timeout_sec=10.0):
        return routes(method, url, params or {}, headers or {}, timeout_sec)

    return transport


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Co lap bien moi truong: khong bao gio doc `.env.testnet` that cua may dev."""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield


def _client(**kwargs) -> BinanceSignedTestnetClient:
    """Client offline voi cap khoa fake tuong minh (khong autoload dotenv)."""
    params = {
        "spot_api_key": _FAKE_SPOT_KEY,
        "spot_api_secret": _FAKE_SPOT_SECRET,
        "futures_api_key": _FAKE_FUT_KEY,
        "futures_api_secret": _FAKE_FUT_SECRET,
        "dry_run": True,
        "autoload_dotenv": False,
    }
    params.update(kwargs)
    return BinanceSignedTestnetClient(**params)


# ---------------------------------------------------------------------------
# 1-3. load_dotenv_testnet
# ---------------------------------------------------------------------------


def test_load_dotenv_parses_bom_comments_export_and_quotes(tmp_path) -> None:
    """Doc duoc file co BOM, bo comment/dong trong, ho tro export va quote."""
    path = tmp_path / ".env.testnet"
    path.write_text(
        "BOM_KEY=bom-value\n"
        "# dong comment\n"
        "\n"
        "PLAIN_KEY=plain-value\n"
        "export EXPORTED_KEY=exported-value\n"
        'DQ_KEY="quoted-double"\n'
        "SQ_KEY='quoted-single'\n"
        "EMPTY_KEY=\n"
        "HASH_KEY=a#b\n"
        "NO_EQUALS_LINE\n",
        encoding="utf-8-sig",
    )

    env: dict[str, str] = {}
    loaded = load_dotenv_testnet(path, extra_paths=(), environ=env)

    assert env["BOM_KEY"] == "bom-value"
    # Neu BOM khong duoc xu ly, key se dinh ky tu BOM o dau.
    assert "\ufeffBOM_KEY" not in env
    assert env["PLAIN_KEY"] == "plain-value"
    assert env["EXPORTED_KEY"] == "exported-value"
    assert env["DQ_KEY"] == "quoted-double"
    assert env["SQ_KEY"] == "quoted-single"
    assert env["EMPTY_KEY"] == ""
    # '#' chi la comment khi dau dong, khong cat gia tri.
    assert env["HASH_KEY"] == "a#b"
    assert "NO_EQUALS_LINE" not in env
    assert loaded == {
        "BOM_KEY": "bom-value",
        "PLAIN_KEY": "plain-value",
        "EXPORTED_KEY": "exported-value",
        "DQ_KEY": "quoted-double",
        "SQ_KEY": "quoted-single",
        "EMPTY_KEY": "",
        "HASH_KEY": "a#b",
    }


def test_load_dotenv_does_not_overwrite_unless_override(tmp_path) -> None:
    path = tmp_path / "dotenv"
    path.write_text("A=from-file\nB=from-file\n", encoding="utf-8")

    env = {"A": "from-env", "B": ""}
    loaded = load_dotenv_testnet(path, extra_paths=(), environ=env)
    assert env["A"] == "from-env"  # da co gia tri -> giu nguyen
    assert env["B"] == "from-file"  # rong -> duoc nap
    assert loaded == {"B": "from-file"}

    env_override = {"A": "from-env"}
    loaded_override = load_dotenv_testnet(
        path, extra_paths=(), override=True, environ=env_override
    )
    assert env_override["A"] == "from-file"
    assert env_override["B"] == "from-file"
    assert loaded_override == {"A": "from-file", "B": "from-file"}


def test_load_dotenv_missing_file_returns_empty(tmp_path) -> None:
    assert load_dotenv_testnet(tmp_path / "nope.env", extra_paths=(), environ={}) == {}
    assert (
        load_dotenv_testnet(
            None, extra_paths=(tmp_path / "also-missing.env",), environ={}
        )
        == {}
    )


# ---------------------------------------------------------------------------
# 4. Tach biet khoa Spot / Futures + thu tu uu tien
# ---------------------------------------------------------------------------


def test_spot_and_futures_keys_are_resolved_separately(monkeypatch) -> None:
    monkeypatch.setenv(ENV_SPOT_API_KEY, _FAKE_SPOT_KEY)
    monkeypatch.setenv(ENV_SPOT_API_SECRET, _FAKE_SPOT_SECRET)
    monkeypatch.setenv(ENV_FUTURES_API_KEY, _FAKE_FUT_KEY)
    monkeypatch.setenv(ENV_FUTURES_API_SECRET, _FAKE_FUT_SECRET)

    client = BinanceSignedTestnetClient(dry_run=True, autoload_dotenv=False)

    assert client.api_key_for("spot") == _FAKE_SPOT_KEY
    assert client.api_secret_for("spot") == _FAKE_SPOT_SECRET
    assert client.api_key_for("futures") == _FAKE_FUT_KEY
    assert client.api_secret_for("futures") == _FAKE_FUT_SECRET
    assert client.api_key_for("spot") != client.api_key_for("futures")
    assert client.has_spot_credentials() is True
    assert client.has_futures_credentials() is True
    assert client.has_credentials() is True
    assert client._headers("spot")["X-MBX-APIKEY"] == _FAKE_SPOT_KEY
    assert client._headers("futures")["X-MBX-APIKEY"] == _FAKE_FUT_KEY


def test_credential_precedence_and_legacy_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv(ENV_SPOT_API_KEY, "env-spot-key")
    monkeypatch.setenv(ENV_SPOT_API_SECRET, "env-spot-secret")
    monkeypatch.setenv(ENV_FUTURES_API_KEY, "env-futures-key")
    monkeypatch.setenv(ENV_FUTURES_API_SECRET, "env-futures-secret")
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "legacy-key")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "legacy-secret")

    # 1. Tham so rieng cua venue thang tham so chung.
    explicit = BinanceSignedTestnetClient(
        api_key="generic-key",
        api_secret="generic-secret",
        spot_api_key="explicit-spot-key",
        spot_api_secret="explicit-spot-secret",
        autoload_dotenv=False,
    )
    assert explicit.spot_api_key == "explicit-spot-key"
    assert explicit.spot_api_secret == "explicit-spot-secret"
    # Venue khong co tham so rieng -> roi xuong tham so chung.
    assert explicit.futures_api_key == "generic-key"
    assert explicit.futures_api_secret == "generic-secret"

    # 2. Tham so chung thang env rieng.
    generic = BinanceSignedTestnetClient(
        api_key="generic-key", api_secret="generic-secret", autoload_dotenv=False
    )
    assert generic.spot_api_key == "generic-key"
    assert generic.futures_api_key == "generic-key"

    # 3. Khong co tham so -> env rieng tung venue.
    from_env = BinanceSignedTestnetClient(autoload_dotenv=False)
    assert from_env.spot_api_key == "env-spot-key"
    assert from_env.futures_api_key == "env-futures-key"
    # Legacy alias van doc ENV cu.
    assert from_env.api_key == "legacy-key"
    assert from_env.api_secret == "legacy-secret"

    # 4. Khong co env rieng -> fallback ve env legacy.
    monkeypatch.delenv(ENV_SPOT_API_KEY)
    monkeypatch.delenv(ENV_SPOT_API_SECRET)
    monkeypatch.delenv(ENV_FUTURES_API_KEY)
    monkeypatch.delenv(ENV_FUTURES_API_SECRET)
    legacy = BinanceSignedTestnetClient(autoload_dotenv=False)
    assert legacy.spot_api_key == "legacy-key"
    assert legacy.spot_api_secret == "legacy-secret"
    assert legacy.futures_api_key == "legacy-key"
    assert legacy.futures_api_secret == "legacy-secret"


def test_credential_status_never_leaks_raw_values() -> None:
    client = _client()
    status = client.credential_status()
    blob = str(status)

    for raw in (
        _FAKE_SPOT_KEY,
        _FAKE_SPOT_SECRET,
        _FAKE_FUT_KEY,
        _FAKE_FUT_SECRET,
    ):
        assert raw not in blob, "credential_status() lam lo credential raw"

    assert set(status) == {
        "spot_key",
        "spot_secret",
        "futures_key",
        "futures_secret",
        "spot_credentials",
        "futures_credentials",
        "spot_time_offset_ms",
        "futures_time_offset_ms",
    }
    assert status["spot_key"].startswith("loaded=True (len=")
    # Dinh dang mat na: "loaded=True (len=21, prefix=fake***)" - co dau *** nhung
    # khong nhat thiet nam o cuoi chuoi.
    assert "***" in status["spot_secret"]
    assert len(_FAKE_SPOT_SECRET) > 4
    assert status["spot_credentials"] is True
    assert status["futures_credentials"] is True
    assert status["spot_time_offset_ms"] == 0
    assert status["futures_time_offset_ms"] == 0


# ---------------------------------------------------------------------------
# 5-7. Time sync va timestamp offset
# ---------------------------------------------------------------------------


def _time_routes(spot_server_ms: int, futures_server_ms: int):
    def routes(method, url, params, headers, timeout_sec):
        if url.endswith("/api/v3/time"):
            assert method.upper() == "GET"
            return {"serverTime": spot_server_ms}
        if url.endswith("/fapi/v1/time"):
            assert method.upper() == "GET"
            return {"serverTime": futures_server_ms}
        raise AssertionError(f"Unexpected URL {url}")

    return routes


def test_sync_server_time_sets_per_venue_offsets() -> None:
    local_ms = 1_700_000_000_000
    client = _client(
        transport=make_transport(_time_routes(local_ms + 123, local_ms + 456))
    )

    result = client.sync_server_time("both", now_ms=local_ms)

    assert result["errors"] == []
    assert result["synced"] is True
    assert result["spot_offset_ms"] == 123
    assert result["futures_offset_ms"] == 456
    assert result["spot_server_time_ms"] == local_ms + 123
    assert result["futures_server_time_ms"] == local_ms + 456
    assert client.spot_time_offset_ms == 123
    assert client.futures_time_offset_ms == 456
    assert client.time_offset_ms("spot") == 123
    assert client.time_offset_ms("futures") == 456


def test_sync_server_time_reports_failures_without_raising() -> None:
    def routes(method, url, params, headers, timeout_sec):
        if url.endswith("/api/v3/time"):
            return {"serverTime": 1_700_000_000_123}
        return {"code": -1001, "msg": "Internal error"}

    client = _client(transport=make_transport(routes))
    result = client.sync_server_time("both", now_ms=1_700_000_000_000)

    assert result["spot_offset_ms"] == 123
    assert result["futures_offset_ms"] == 0
    assert result["synced"] is False
    assert len(result["errors"]) == 1


def test_sign_params_applies_venue_time_offset(monkeypatch) -> None:
    client = _client()
    client.spot_time_offset_ms = 123
    client.futures_time_offset_ms = 456

    # Dong bang dong ho de test tat dinh (khong phu thuoc gio he thong).
    monkeypatch.setattr(
        "src.binance_signed_client.time.time", lambda: 1_700_000_000.0
    )

    futures_signed = client.sign_params({"symbol": "BTCUSDT"}, venue="futures")
    assert futures_signed["timestamp"] == 1_700_000_000_000 + 456
    assert futures_signed["recvWindow"] == 5000
    unsigned = {k: v for k, v in futures_signed.items() if k != "signature"}
    assert futures_signed["signature"] == sign_binance_payload(
        unsigned, _FAKE_FUT_SECRET
    )

    spot_signed = client.sign_params({"symbol": "BTCUSDT"}, venue="spot")
    assert spot_signed["timestamp"] == 1_700_000_000_000 + 123
    spot_unsigned = {k: v for k, v in spot_signed.items() if k != "signature"}
    assert spot_signed["signature"] == sign_binance_payload(
        spot_unsigned, _FAKE_SPOT_SECRET
    )


def test_sign_params_explicit_timestamp_overrides_offset() -> None:
    client = _client()
    client.spot_time_offset_ms = 999
    client.futures_time_offset_ms = 777

    signed = client.sign_params(
        {"symbol": "BTCUSDT"}, timestamp_ms=1_700_000_000_123, venue="futures"
    )
    assert signed["timestamp"] == 1_700_000_000_123


# ---------------------------------------------------------------------------
# 8. Chan doan khoa Futures dung nham
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_code", [-2014, -2015])
def test_check_account_access_flags_futures_key_misuse(api_code: int) -> None:
    def routes(method, url, params, headers, timeout_sec):
        assert "/fapi/v2/account" in url
        return {"code": api_code, "msg": "Invalid API-key, IP, or permissions"}

    client = _client(transport=make_transport(routes))
    result = client.check_account_access("futures")

    assert result["authenticated"] is False
    assert result["error"] == FUTURES_KEY_INVALID_CODE
    assert result["hint"] == FUTURES_KEY_INVALID_HINT
    assert result["api_error_code"] == api_code


def test_reconcile_snapshot_flags_futures_key_misuse() -> None:
    def routes(method, url, params, headers, timeout_sec):
        if "/api/v3/account" in url:
            return {"balances": [{"asset": "USDT", "free": "1000.0"}]}
        if "/fapi/v2/account" in url:
            return {"code": -2014, "msg": "API-key format invalid."}
        raise AssertionError(f"Unexpected URL {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    snapshot = client.fetch_exchange_reconcile_snapshot(
        ["BTC/USDT:USDT"], {"BTC/USDT:USDT": 96000.0}
    )

    assert snapshot["snapshot_healthy"] is False
    assert snapshot["error"] == FUTURES_KEY_INVALID_HINT
    assert snapshot["futures_key_status"] == FUTURES_KEY_INVALID_CODE
    assert snapshot["api_error_code"] == -2014
