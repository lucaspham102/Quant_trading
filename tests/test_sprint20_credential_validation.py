"""Test Sprint 20 - Kiem dinh dinh dang credential Binance Testnet.

Bao phu:
1. `validate_credential_format` -> ok=True cho key/secret 64 ky tu alnum sach.
2. Tu dong lam sach whitespace/newline/dau nhay boc ngoai (`_clean_credential`).
3. Secret 65 ky tu -> ma `spot_SECRET_INVALID_LENGTH_65_EXPECTED_64`.
4. Secret 64 ky tu nhung co ky tu khong phai alnum -> ma charset.
5. Secret rong/None -> ma `<venue>_SECRET_MISSING`.
6. `BinanceSignedTestnetClient` lam sach khoa truyen tuong minh + `all_ok`.
7. `credential_format_report()` phat hien secret Futures 65 ky tu.
8. `load_dotenv_testnet` parse quote / `export ` / khoang trang thua.
9. Helper `_validate_client_credentials` cua script round-trip (client stub).

Offline hoan toan: khong goi mang, khong doc `.env` that, khong in gia tri khoa.
Chi dung chuoi gia lap ("A"*64, "B"*64).
"""

from __future__ import annotations

import importlib.util
import pathlib

from src.binance_signed_client import (
    CREDENTIAL_EXPECTED_LENGTH,
    BinanceSignedTestnetClient,
    load_dotenv_testnet,
    validate_credential_format,
)

# -- hang so gia lap (ASCII, khong phai khoa that) -------------------------
FAKE_KEY_A = "A" * 64
FAKE_KEY_C = "C" * 64
FAKE_SECRET_B = "B" * 64
FAKE_SECRET_D = "D" * 64


# -- 1 ---------------------------------------------------------------------


def test_validate_credential_format_clean_values_ok() -> None:
    """Key/secret 64 ky tu alnum sach -> ok=True va message bao OK."""
    report = validate_credential_format(FAKE_KEY_A, FAKE_SECRET_B, "spot")

    assert report["ok"] is True
    assert report["key_ok"] is True
    assert report["secret_ok"] is True
    assert report["venue"] == "spot"
    assert report["expected_length"] == CREDENTIAL_EXPECTED_LENGTH == 64
    assert report["key_length"] == 64
    assert report["secret_length"] == 64
    assert report["key_present"] is True
    assert report["secret_present"] is True
    assert report["diagnostics"] == []
    assert report["normalized"] is False
    assert report["message"].endswith("CREDENTIAL_FORMAT_OK")
    # Khong duoc lo gia tri khoa trong report.
    assert FAKE_KEY_A not in report["message"]
    assert FAKE_SECRET_B not in report["message"]


# -- 2 ---------------------------------------------------------------------


def test_validate_credential_format_cleans_whitespace_and_quotes() -> None:
    """Whitespace/newline/dau nhay boc ngoai duoc tu dong lam sach."""
    key = '  "' + FAKE_KEY_A + '"  \n'
    secret = " \t'" + FAKE_SECRET_B + "'\n"

    report = validate_credential_format(key, secret, "spot")

    assert report["ok"] is True
    assert report["key_length"] == 64
    assert report["secret_length"] == 64
    assert report["normalized"] is True
    assert report["diagnostics"] == []
    assert report["message"] == "spot_CREDENTIAL_FORMAT_OK"


# -- 3 ---------------------------------------------------------------------


def test_validate_credential_format_secret_65_chars_reports_length() -> None:
    """Secret 65 ky tu -> dung ma do dai, khong phai ma charset."""
    secret_65 = FAKE_SECRET_B + "B"
    assert len(secret_65) == 65

    report = validate_credential_format(FAKE_KEY_A, secret_65, "spot")

    assert report["ok"] is False
    assert report["secret_ok"] is False
    assert report["key_ok"] is True
    assert report["secret_length"] == 65
    assert report["diagnostics"] == ["spot_SECRET_INVALID_LENGTH_65_EXPECTED_64"]
    assert "spot_SECRET_INVALID_LENGTH_65_EXPECTED_64" in report["message"]


# -- 4 ---------------------------------------------------------------------


def test_validate_credential_format_secret_bad_charset_futures() -> None:
    """Secret dai 64 nhung chua '-' -> ma charset cho venue futures."""
    bad_secret = ("C" * 63) + "-"
    assert len(bad_secret) == 64

    report = validate_credential_format(FAKE_KEY_A, bad_secret, "futures")

    assert report["ok"] is False
    assert report["secret_ok"] is False
    assert report["venue"] == "futures"
    assert "futures_SECRET_INVALID_CHARSET_ALNUM_REQUIRED" in report["diagnostics"]
    assert "futures_SECRET_INVALID_CHARSET_ALNUM_REQUIRED" in report["message"]


# -- 5 ---------------------------------------------------------------------


def test_validate_credential_format_missing_secret() -> None:
    """Secret rong hoac None -> ma `<venue>_SECRET_MISSING`, ok=False."""
    empty_report = validate_credential_format(FAKE_KEY_A, "", "spot")
    none_report = validate_credential_format(FAKE_KEY_A, None, "spot")

    for report in (empty_report, none_report):
        assert report["ok"] is False
        assert report["secret_ok"] is False
        assert report["secret_present"] is False
        assert "spot_SECRET_MISSING" in report["diagnostics"]
        assert "spot_SECRET_MISSING" in report["message"]

    missing_key_report = validate_credential_format(None, FAKE_SECRET_B, "futures")
    assert missing_key_report["ok"] is False
    assert "futures_KEY_MISSING" in missing_key_report["diagnostics"]


# -- 6 ---------------------------------------------------------------------


def test_client_cleans_explicit_credentials_and_reports_all_ok() -> None:
    """Khoa truyen tuong minh duoc lam sach; ca 2 venue dat chuan 64 ky tu."""
    client = BinanceSignedTestnetClient(
        spot_api_key='  "' + FAKE_KEY_A + '"  ',
        spot_api_secret=FAKE_SECRET_B + "\n",
        futures_api_key="'" + FAKE_KEY_C + "'",
        futures_api_secret="  " + FAKE_SECRET_D + "  ",
        autoload_dotenv=False,
    )

    assert client.spot_api_key == FAKE_KEY_A
    assert client.spot_api_secret == FAKE_SECRET_B
    assert client.futures_api_key == FAKE_KEY_C
    assert client.futures_api_secret == FAKE_SECRET_D

    report = client.credential_format_report()

    assert report["spot"]["ok"] is True
    assert report["futures"]["ok"] is True
    assert report["all_ok"] is True
    assert report["dual_venue_ok"] is True
    # Report khong duoc chua gia tri khoa.
    assert FAKE_KEY_A not in str(report["spot"]["message"])
    assert FAKE_SECRET_D not in str(report["futures"]["message"])


# -- 7 ---------------------------------------------------------------------


def test_client_credential_format_report_detects_futures_secret_65() -> None:
    """Secret Futures 65 ky tu -> all_ok=False + ma do dai dung venue."""
    client = BinanceSignedTestnetClient(
        spot_api_key=FAKE_KEY_A,
        spot_api_secret=FAKE_SECRET_B,
        futures_api_key=FAKE_KEY_C,
        futures_api_secret=FAKE_SECRET_D + "D",
        autoload_dotenv=False,
    )

    report = client.credential_format_report()

    assert report["all_ok"] is False
    assert report["spot"]["ok"] is True
    assert report["futures"]["ok"] is False
    assert "futures_SECRET_INVALID_LENGTH_65_EXPECTED_64" in report["futures"]["diagnostics"]


# -- 8 ---------------------------------------------------------------------


def test_load_dotenv_testnet_parses_quotes_export_and_spaces(tmp_path) -> None:
    """Parse quote, tien to `export ` va khoang trang thua; khong dung os.environ that."""
    env_file = tmp_path / ".env.testnet"
    env_file.write_text(
        "# comment line\n"
        "\n"
        'BINANCE_SPOT_TESTNET_API_KEY="' + FAKE_KEY_A + '"\n'
        "export BINANCE_SPOT_TESTNET_API_SECRET='" + FAKE_SECRET_B + "'\n"
        "BINANCE_FUTURES_TESTNET_API_KEY=   " + FAKE_KEY_C + "   \n"
        "export BINANCE_FUTURES_TESTNET_API_SECRET=" + FAKE_SECRET_D + "  \n",
        encoding="utf-8",
    )

    environ: dict[str, str] = {}
    loaded = load_dotenv_testnet(
        dotenv_path=env_file,
        extra_paths=(),
        environ=environ,
    )

    assert loaded["BINANCE_SPOT_TESTNET_API_KEY"] == FAKE_KEY_A
    assert loaded["BINANCE_SPOT_TESTNET_API_SECRET"] == FAKE_SECRET_B
    assert loaded["BINANCE_FUTURES_TESTNET_API_KEY"] == FAKE_KEY_C
    assert loaded["BINANCE_FUTURES_TESTNET_API_SECRET"] == FAKE_SECRET_D
    # Gia tri da duoc lam sach: khong con quote/newline/khoang trang.
    for value in loaded.values():
        assert value == value.strip()
        assert not value.startswith(("'", '"'))
        assert len(value) == 64
    assert environ["BINANCE_SPOT_TESTNET_API_KEY"] == FAKE_KEY_A


# -- 9 ---------------------------------------------------------------------

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_testnet_live_roundtrip.py"
)


def _load_roundtrip_module():
    """Nap script bang importlib (scripts/ khong co __init__.py)."""
    spec = importlib.util.spec_from_file_location("rt_sprint20", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load_roundtrip_module()


class _StubClientWithCreds:
    """Client gia lap co attribute credential 64 ky tu (khong co report method)."""

    def __init__(self, key: str, secret: str) -> None:
        self.spot_api_key = key
        self.spot_api_secret = secret


class _StubClientWithoutCreds:
    """Client gia lap khong co attribute credential -> check bi skip."""


def test_validate_client_credentials_helper_ok_and_skipped() -> None:
    """Helper tra ok=True voi stub co khoa, va skipped=True voi stub rong."""
    ok_result = rt._validate_client_credentials(
        _StubClientWithCreds(FAKE_KEY_A, FAKE_SECRET_B), "spot"
    )
    assert ok_result["ok"] is True
    assert ok_result["venue"] == "spot"
    assert ok_result.get("skipped") is not True
    assert ok_result["diagnostics"] == []

    skipped = rt._validate_client_credentials(_StubClientWithoutCreds(), "futures")
    assert skipped["skipped"] is True
    assert skipped["ok"] is True
    assert skipped["venue"] == "futures"
    assert skipped["diagnostics"] == []
