"""Sprint 16 / 18 / 19 - Adapter ky lenh Binance HMAC-SHA256 cho Testnet (fail-closed).

Nguyen tac an toan:
- KHONG hardcode khoa: chi doc tu tham so, file `.env.testnet` hoac bien moi truong.
- Sprint 19: ho tro KHOA KEP rieng biet cho Spot Testnet va USDS-M Futures Testnet.
- dry_run=True mac dinh: chi dong goi payload + tao chu ky, KHONG gui lenh.
- Thieu khoa ma dry_run=False -> raise BinanceCredentialsMissingError (fail-closed).
- Sprint 19: `sync_server_time` bu lech gio may chu de triet tieu loi -1021.

Chuan chu ky Binance: query string duoc sap xep theo key, urlencode, kem timestamp
va recvWindow; signature = HMAC_SHA256(secret, query_string).hexdigest().
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlencode

from src.cross_exchange_carry import DEFAULT_MIN_NOTIONAL, EXCHANGE_LOT_SPECS
from src.exchange_reconciler import normalize_symbol

SPOT_TESTNET_BASE_URL: str = "https://testnet.binance.vision"
PERP_TESTNET_BASE_URL: str = "https://testnet.binancefuture.com"
SPOT_ORDER_PATH: str = "/api/v3/order"
PERP_ORDER_PATH: str = "/fapi/v1/order"
SPOT_ACCOUNT_PATH: str = "/api/v3/account"
PERP_ACCOUNT_PATH: str = "/fapi/v2/account"
PERP_POSITION_RISK_PATH: str = "/fapi/v2/positionRisk"
SPOT_TIME_PATH: str = "/api/v3/time"
PERP_TIME_PATH: str = "/fapi/v1/time"
SPOT_BOOK_TICKER_PATH: str = "/api/v3/ticker/bookTicker"
PERP_BOOK_TICKER_PATH: str = "/fapi/v1/ticker/bookTicker"
DEFAULT_RECV_WINDOW_MS: int = 5000
DEFAULT_LIVE_RECV_WINDOW_MS: int = 10000
ENV_API_KEY: str = "BINANCE_TESTNET_API_KEY"
ENV_API_SECRET: str = "BINANCE_TESTNET_API_SECRET"
ENV_SPOT_API_KEY: str = "BINANCE_SPOT_TESTNET_API_KEY"
ENV_SPOT_API_SECRET: str = "BINANCE_SPOT_TESTNET_API_SECRET"
ENV_FUTURES_API_KEY: str = "BINANCE_FUTURES_TESTNET_API_KEY"
ENV_FUTURES_API_SECRET: str = "BINANCE_FUTURES_TESTNET_API_SECRET"
DEFAULT_DOTENV_TESTNET: str = ".env.testnet"
DEFAULT_DOTENV: str = ".env"

VENUE_SPOT: str = "spot"
VENUE_FUTURES: str = "futures"

FUTURES_KEY_INVALID_CODE = "FUTURES_KEY_INVALID_OR_SPOT_KEY_USED_ON_FUTURES"
FUTURES_KEY_INVALID_HINT: str = (
    "FUTURES_KEY_INVALID_OR_SPOT_KEY_USED_ON_FUTURES: "
    "Hay tao them khoa rieng tai https://testnet.binancefuture.com"
)
FUTURES_INVALID_ERROR_CODES: frozenset[int] = frozenset({-2014, -2015})
POST_ONLY_REJECT_CODES: frozenset[int] = frozenset({-2010, -2021, -1102})


class BinanceCredentialsMissingError(RuntimeError):
    """Thieu API key/secret khi can ky lenh that (fail-closed)."""


class BinanceHttpError(RuntimeError):
    """Loi HTTP/API tra ve tu Binance (kem code/msg de chan doan)."""

    def __init__(self, message: str, code: int | None = None, msg: str = "") -> None:
        super().__init__(message)
        self.code: int | None = code
        self.msg: str = msg or message


# Sprint 20: do dai chuan cua API key/secret Binance (HMAC-SHA256, base64-like 64 ky tu).
CREDENTIAL_EXPECTED_LENGTH: int = 64


class SpotValuationPriceMissingError(RuntimeError):
    """Khong the dinh gia tai san Spot dang nam giu vi thieu gia thi truong hop le."""


def _mask_secret(value: str | None) -> str:
    """Mo ta trang thai khoa an danh: KHONG BAO GIO lo gia tri that."""
    if not value:
        return "not_loaded"
    return f"loaded=True (len={len(value)}, prefix={value[:4]}***)"


def _clean_credential(value: str | None) -> str:
    """Sprint 20: chuan hoa khoa doc tu `.env.testnet`.

    - Bo khoang trang / ky tu xuong dong thua o 2 dau (va ca ben trong chuoi).
    - Bo dau nhay boc ngoai (`"` hoac `'`) do copy/paste tu dashboard.
    - KHONG bao gio log gia tri khoa.
    """
    if value is None:
        return ""
    text = str(value)
    # Giu nguyen ky tu ben trong nhung loai bo moi whitespace/newline (khoa Binance
    # khong bao gio chua whitespace hop le).
    text = "".join(text.split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    return text


def _credential_pair_is_valid(key: str | None, secret: str | None) -> bool:
    """Sprint 21: cap khoa chuyen biet dat chuan 64/64 va chi gom chu/so?"""
    clean_key = _clean_credential(key)
    clean_secret = _clean_credential(secret)
    return (
        len(clean_key) == CREDENTIAL_EXPECTED_LENGTH
        and len(clean_secret) == CREDENTIAL_EXPECTED_LENGTH
        and clean_key.isalnum()
        and clean_secret.isalnum()
    )


def prune_legacy_fallback_env(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Sprint 21: don bien fallback legacy khi khoa chuyen biet da dat chuan 64/64.

    Neu ``BINANCE_SPOT_TESTNET_API_KEY/SECRET`` hoac
    ``BINANCE_FUTURES_TESTNET_API_KEY/SECRET`` hop le (64/64), cac bien cu
    ``BINANCE_TESTNET_API_KEY/SECRET`` co do dai khac 64 bi xoa khoi moi truong
    va bao ``legacy_fallback_ignored=True`` (chi metadata, khong lo gia tri khoa).
    """
    env: dict[str, str] = environ if environ is not None else os.environ
    spot_valid = _credential_pair_is_valid(env.get(ENV_SPOT_API_KEY), env.get(ENV_SPOT_API_SECRET))
    futures_valid = _credential_pair_is_valid(
        env.get(ENV_FUTURES_API_KEY), env.get(ENV_FUTURES_API_SECRET)
    )
    ignored_keys: list[str] = []
    if spot_valid or futures_valid:
        for legacy_key in (ENV_API_KEY, ENV_API_SECRET):
            raw = env.get(legacy_key)
            if raw is None:
                continue
            if len(_clean_credential(raw)) == CREDENTIAL_EXPECTED_LENGTH:
                continue
            env.pop(legacy_key, None)
            ignored_keys.append(legacy_key)
    return {
        "spot_specific_valid": bool(spot_valid),
        "futures_specific_valid": bool(futures_valid),
        "legacy_fallback_ignored": bool(ignored_keys),
        "ignored_legacy_keys": ignored_keys,
    }


def validate_credential_format(
    key: str | None,
    secret: str | None,
    venue: str,
    legacy_key: str | None = None,
    legacy_secret: str | None = None,
) -> dict[str, Any]:
    """Sprint 20: kiem dinh dinh dang khoa Binance TRUOC khi gui mang.

    Quy tac (HMAC-SHA256 Binance): key/secret sau khi chuan hoa phai dai dung
    ``CREDENTIAL_EXPECTED_LENGTH`` (64) ky tu va chi gom chu/so (``isalnum``).

    Tra ve dict CHI chua metadata an toan (do dai + ma chan doan), KHONG chua
    gia tri khoa. Ma chan doan vi du:
    ``spot_SECRET_INVALID_LENGTH_65_EXPECTED_64``.

    Sprint 21: khi cap khoa chuyen biet da hop le 64/64 ma cap fallback legacy
    (``BINANCE_TESTNET_API_KEY/SECRET``) sai do dai, tra them
    ``legacy_fallback_ignored=True`` de caller bo qua bien cu an toan.
    """
    venue_name = str(venue or VENUE_SPOT).strip().lower() or VENUE_SPOT

    raw_key = key if key is not None else ""
    raw_secret = secret if secret is not None else ""

    clean_key = _clean_credential(raw_key)
    clean_secret = _clean_credential(raw_secret)

    errors: list[str] = []
    key_ok = True
    secret_ok = True

    if not clean_key:
        key_ok = False
        errors.append(f"{venue_name}_KEY_MISSING")
    elif len(clean_key) != CREDENTIAL_EXPECTED_LENGTH:
        key_ok = False
        errors.append(f"{venue_name}_KEY_INVALID_LENGTH_{len(clean_key)}_EXPECTED_64")
    elif not clean_key.isalnum():
        key_ok = False
        errors.append(f"{venue_name}_KEY_INVALID_CHARSET_ALNUM_REQUIRED")

    if not clean_secret:
        secret_ok = False
        errors.append(f"{venue_name}_SECRET_MISSING")
    elif len(clean_secret) != CREDENTIAL_EXPECTED_LENGTH:
        secret_ok = False
        errors.append(
            f"{venue_name}_SECRET_INVALID_LENGTH_{len(clean_secret)}_EXPECTED_64"
        )
    elif not clean_secret.isalnum():
        secret_ok = False
        errors.append(f"{venue_name}_SECRET_INVALID_CHARSET_ALNUM_REQUIRED")

    result = {
        "venue": venue_name,
        "expected_length": CREDENTIAL_EXPECTED_LENGTH,
        "key_present": bool(clean_key),
        "secret_present": bool(clean_secret),
        "key_length": len(clean_key),
        "secret_length": len(clean_secret),
        "key_ok": bool(key_ok),
        "secret_ok": bool(secret_ok),
        "ok": bool(key_ok and secret_ok),
        "normalized": bool(
            clean_key != str(raw_key) or clean_secret != str(raw_secret)
        ),
        "diagnostics": errors,
        # Sprint 20: chuoi chan doan de in ra log/bao cao (an toan, khong co khoa).
        "message": " | ".join(errors) if errors else f"{venue_name}_CREDENTIAL_FORMAT_OK",
    }

    # Sprint 21: bo qua bien fallback legacy neu khoa chuyen biet da dat chuan 64/64.
    legacy_present = legacy_key is not None or legacy_secret is not None
    legacy_invalid = False
    if legacy_present:
        for legacy_val in (legacy_key, legacy_secret):
            if legacy_val is not None:
                cleaned_legacy = _clean_credential(legacy_val)
                if cleaned_legacy and len(cleaned_legacy) != CREDENTIAL_EXPECTED_LENGTH:
                    legacy_invalid = True
                    break
    result["legacy_fallback_present"] = bool(legacy_present)
    result["legacy_fallback_ignored"] = bool(result["ok"] and legacy_invalid)
    return result


def load_dotenv_testnet(
    dotenv_path: str | Path | None = DEFAULT_DOTENV_TESTNET,
    extra_paths: tuple[str | Path, ...] = (DEFAULT_DOTENV,),
    override: bool = False,
    environ: dict[str, str] | None = None,
    report: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Nap bien moi truong tu `.env.testnet` (+ `.env`) vao os.environ (khong ghi de).

    - Doc voi ``encoding="utf-8-sig"`` de xu ly an toan BOM tren Windows.
    - Bo qua dong trong va dong comment ``#``.
    - Ho tro tien to ``export KEY=VALUE`` va gia tri duoc quote ``"..."``/``'...'``.
    - Chi nap khi bien chua ton tai hoac dang rong (tru khi ``override=True``).
    - Tra ve dict cac cap KEY->VALUE da PARSE (khong in ra console).

    Sprint 21: truyen ``report={}`` de nhan metadata ``legacy_fallback_ignored`` khi
    cap khoa chuyen biet da hop le 64/64 khien bien fallback legacy sai do dai bi bo qua.

    Luu y bao mat: KHONG BAO GIO log/in gia tri khoa tu ham nay.
    """
    target_env: dict[str, str] = environ if environ is not None else os.environ
    loaded: dict[str, str] = {}

    paths: list[Path] = []
    if dotenv_path is not None:
        paths.append(Path(dotenv_path))
    for extra in extra_paths or ():
        if extra is not None:
            paths.append(Path(extra))

    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - duong dan la
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:  # pragma: no cover - quyen doc
            continue
        for line in raw.splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.lower().startswith("export "):
                text = text[7:].strip()
            if "=" not in text:
                continue
            key, _, value = text.partition("=")
            key = key.strip()
            if not key:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if not override and target_env.get(key):
                continue
            target_env[key] = value
            loaded[key] = value

    # Sprint 21: ve sinh bien fallback legacy (chi khi khoa chuyen biet dat chuan).
    legacy_report = prune_legacy_fallback_env(target_env)
    if report is not None:
        report["loaded_keys"] = sorted(loaded)
        report["legacy_fallback_ignored"] = legacy_report["legacy_fallback_ignored"]
        report["ignored_legacy_keys"] = legacy_report["ignored_legacy_keys"]
        report["spot_specific_valid"] = legacy_report["spot_specific_valid"]
        report["futures_specific_valid"] = legacy_report["futures_specific_valid"]
    return loaded


CASH_ASSETS: frozenset[str] = frozenset(
    {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USD"}
)


def _is_cash_asset(asset: str) -> bool:
    """True neu asset la tien mat/stablecoin quote (da duoc quy doi vao spot_cash_usd)."""
    a = str(asset or "").strip().upper()
    if not a:
        return False
    if a in CASH_ASSETS:
        return True
    return a.endswith(("USDT", "USDC", "BUSD", "FDUSD"))


def _format_value(value: Any) -> str:
    """Chuan hoa so thanh chuoi thap phan on dinh (tranh ky hieu mu e-05)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e16:
            return str(int(value))
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def build_query_string(params: Mapping[str, Any]) -> str:
    """Sap xep key tang dan + urlencode de chuoi ky va body trung khop tuyet doi."""
    ordered = sorted((str(k), _format_value(v)) for k, v in params.items())
    return urlencode(ordered)


def sign_binance_payload(params: Mapping[str, Any], api_secret: str) -> str:
    """Tra ve HMAC-SHA256 hex cua (query string cua params) voi api_secret."""
    if not api_secret:
        raise BinanceCredentialsMissingError("api_secret rong, khong the ky payload")
    query = build_query_string(params)
    return hmac.new(
        api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _check_lot(symbol: str, qty: float, price: float | None = None) -> dict[str, float]:
    """Kiem tra qty khop step_size / min_notional cua cap truoc khi dung lenh."""
    spec = EXCHANGE_LOT_SPECS.get(symbol)
    if not spec:
        raise ValueError(f"Khong ho tro lot spec cho symbol {symbol!r}")
    step = float(spec["step_size"])
    min_notional = float(spec.get("min_notional", DEFAULT_MIN_NOTIONAL))
    if qty <= 0.0:
        raise ValueError(f"{symbol}: qty phai > 0, nhan {qty!r}")
    steps = qty / step
    if abs(steps - round(steps)) > 1e-6:
        raise ValueError(f"{symbol}: qty {qty!r} khong khop step_size {step!r}")
    if price is not None and price > 0.0 and qty * price < min_notional:
        raise ValueError(
            f"{symbol}: notional {qty * price:.2f} < min_notional {min_notional:.2f}"
        )
    return {"step_size": step, "min_notional": min_notional, "qty": float(qty)}


def _lookup_price(prices: Mapping[str, Any] | None, sym: str, base: str) -> float:
    """Tra cuu gia an toan tu bang prices cho symbol/base."""
    if not isinstance(prices, dict):
        return 0.0
    concat = sym.replace(":USDT", "").replace("/", "").replace("-", "")
    for k in (sym, concat, f"{base}/USDT", base, f"{base}USDT"):
        if k in prices:
            try:
                p = float(prices[k])
                if p > 0.0:
                    return p
            except (ValueError, TypeError):
                pass
    for pk, pv in prices.items():
        if normalize_symbol(str(pk)) == sym:
            try:
                p = float(pv)
                if p > 0.0:
                    return p
            except (ValueError, TypeError):
                pass
    return 0.0


class BinanceSignedTestnetClient:
    """Client ky lenh testnet (Spot + USDS-M Futures), mac dinh dry-run.

    Sprint 19: quan ly 2 cap khoa rieng biet (Spot / Futures), nap `.env.testnet`,
    dong bo gio may chu va bu `time_offset_ms` vao moi request ky HMAC.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        dry_run: bool = True,
        recv_window: int = DEFAULT_RECV_WINDOW_MS,
        spot_base_url: str = SPOT_TESTNET_BASE_URL,
        perp_base_url: str = PERP_TESTNET_BASE_URL,
        spot_api_key: str | None = None,
        spot_api_secret: str | None = None,
        futures_api_key: str | None = None,
        futures_api_secret: str | None = None,
        dotenv_path: str | Path | None = None,
        autoload_dotenv: bool = True,
        transport: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        # Sprint 19: chi tu dong nap `.env.testnet` khi caller KHONG truyen khoa
        # tuong minh -> giu hanh vi tat dinh cho unit test (api_key="" van la rong).
        if autoload_dotenv and api_key is None and api_secret is None:
            try:
                load_dotenv_testnet(
                    dotenv_path if dotenv_path is not None else DEFAULT_DOTENV_TESTNET
                )
            except Exception:  # pragma: no cover - thieu file la binh thuong
                pass

        # -- cap khoa legacy (alias tuong thich nguoc) --------------------
        self.api_key: str | None = (
            api_key if api_key is not None else os.environ.get(ENV_API_KEY)
        )
        self.api_secret: str | None = (
            api_secret if api_secret is not None else os.environ.get(ENV_API_SECRET)
        )

        # -- Sprint 19: cap khoa kep Spot / Futures -----------------------
        self.spot_api_key: str = self._resolve_credential(
            explicit=spot_api_key,
            generic=api_key,
            env_specific=ENV_SPOT_API_KEY,
            env_legacy=ENV_API_KEY,
        )
        self.spot_api_secret: str = self._resolve_credential(
            explicit=spot_api_secret,
            generic=api_secret,
            env_specific=ENV_SPOT_API_SECRET,
            env_legacy=ENV_API_SECRET,
        )
        self.futures_api_key: str = self._resolve_credential(
            explicit=futures_api_key,
            generic=api_key,
            env_specific=ENV_FUTURES_API_KEY,
            env_legacy=ENV_API_KEY,
        )
        self.futures_api_secret: str = self._resolve_credential(
            explicit=futures_api_secret,
            generic=api_secret,
            env_specific=ENV_FUTURES_API_SECRET,
            env_legacy=ENV_API_SECRET,
        )

        # Sprint 19: bu lech gio may chu (ms) cho tung venue.
        self.spot_time_offset_ms: int = 0
        self.futures_time_offset_ms: int = 0

        self.dry_run: bool = bool(dry_run)
        self.recv_window: int = int(recv_window)
        self.spot_base_url: str = spot_base_url.rstrip("/")
        self.perp_base_url: str = perp_base_url.rstrip("/")
        self.transport: Callable[..., dict[str, Any]] | None = transport
        self.last_error: dict[str, Any] | None = None
        if not self.dry_run and not self.has_credentials():
            raise BinanceCredentialsMissingError(
                "Thieu BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET: "
                "bat buoc dung --dry-run-sign hoac cau hinh bien moi truong (fail-closed)."
            )

    @staticmethod
    def _resolve_credential(
        explicit: str | None,
        generic: str | None,
        env_specific: str,
        env_legacy: str,
    ) -> str:
        """Uu tien: tham so rieng > tham so chung > env rieng > env legacy.

        Sprint 20: moi gia tri duoc chuan hoa (`_clean_credential`) de loai bo
        khoang trang/xuong dong/dau nhay thua tu `.env.testnet`.
        """
        if explicit is not None:
            return _clean_credential(explicit)
        if generic is not None:
            return _clean_credential(generic)
        return _clean_credential(
            os.environ.get(env_specific) or os.environ.get(env_legacy) or ""
        )

    # -- credentials -----------------------------------------------------
    def has_spot_credentials(self) -> bool:
        return bool(self.spot_api_key) and bool(self.spot_api_secret)

    def has_futures_credentials(self) -> bool:
        return bool(self.futures_api_key) and bool(self.futures_api_secret)

    def has_credentials(self) -> bool:
        return self.has_spot_credentials() or self.has_futures_credentials()

    def credential_status(self) -> dict[str, str]:
        """Trang thai khoa AN DANH (du an toan de in ra log/bao cao)."""
        return {
            "spot_key": _mask_secret(self.spot_api_key),
            "spot_secret": _mask_secret(self.spot_api_secret),
            "futures_key": _mask_secret(self.futures_api_key),
            "futures_secret": _mask_secret(self.futures_api_secret),
            "spot_credentials": self.has_spot_credentials(),
            "futures_credentials": self.has_futures_credentials(),
            "spot_time_offset_ms": self.spot_time_offset_ms,
            "futures_time_offset_ms": self.futures_time_offset_ms,
        }

    def credential_format_report(self) -> dict[str, Any]:
        """Sprint 20: chan doan dinh dang 64 ky tu cho ca 2 venue (an toan de log).

        Chay TRUOC moi request mang: neu khoa sai do dai (vd 65 ky tu), tra ngay
        ma ``<venue>_SECRET_INVALID_LENGTH_<len>_EXPECTED_64`` thay vi de san tra
        ve -2015/-1022 kho hieu.
        """
        spot = validate_credential_format(self.spot_api_key, self.spot_api_secret, VENUE_SPOT)
        futures = validate_credential_format(
            self.futures_api_key, self.futures_api_secret, VENUE_FUTURES
        )
        return {
            "spot": spot,
            "futures": futures,
            "all_ok": bool(spot.get("ok") and futures.get("ok")),
            "dual_venue_ok": bool(spot.get("ok") and futures.get("ok")),
        }

    def require_credentials(self) -> None:
        if not self.has_credentials():
            raise BinanceCredentialsMissingError(
                "Thieu BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET (fail-closed)."
            )

    def api_key_for(self, venue: str) -> str:
        return self.futures_api_key if venue == VENUE_FUTURES else self.spot_api_key

    def api_secret_for(self, venue: str) -> str:
        return self.futures_api_secret if venue == VENUE_FUTURES else self.spot_api_secret

    def time_offset_ms(self, venue: str) -> int:
        return (
            self.futures_time_offset_ms
            if venue == VENUE_FUTURES
            else self.spot_time_offset_ms
        )

    def _headers(self, venue: str = VENUE_SPOT) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key_for(venue) or ""}

    # -- HTTP transport --------------------------------------------------
    def _http_request(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_sec: float = 10.0,
    ) -> Any:
        """Gui 1 request HTTP toi Binance va tra JSON (dict hoac list)."""
        payload = dict(params or {})
        hdrs = dict(headers or {})
        if self.transport is not None:
            result = self.transport(
                method=method,
                url=url,
                params=payload,
                headers=hdrs,
                timeout_sec=float(timeout_sec),
            )
            if isinstance(result, Exception):
                raise result
            if not isinstance(result, (dict, list)):
                raise BinanceHttpError(f"Transport tra ve kieu khong hop le: {type(result)!r}")
            if isinstance(result, dict) and result.get("code") is not None and "msg" in result:
                code = int(result.get("code") or 0)
                raise BinanceHttpError(
                    f"Binance API loi {code}: {result.get('msg')}",
                    code=code,
                    msg=str(result.get("msg")),
                )
            return result

        query = build_query_string(payload)
        if method.upper() == "GET":
            full_url = f"{url}?{query}" if query else url
            body: bytes | None = None
        else:
            full_url = url
            body = query.encode("utf-8")
            hdrs.setdefault("Content-Type", "application/x-www-form-urlencoded")
        req = urllib.request.Request(
            full_url, data=body, headers=hdrs, method=method.upper()
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover
                detail = ""
            code, msg = _parse_binance_error(detail)
            raise BinanceHttpError(
                f"HTTP {exc.code} {url}: {detail[:300]}", code=code, msg=msg
            ) from exc
        except urllib.error.URLError as exc:
            raise BinanceHttpError(f"Network error {url}: {exc.reason}") from exc
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise BinanceHttpError(f"JSON decode loi ({url}): {raw[:200]}") from exc
        if isinstance(parsed, dict) and parsed.get("code") is not None and "msg" in parsed:
            code = int(parsed.get("code") or 0)
            raise BinanceHttpError(
                f"Binance API loi {code}: {parsed.get('msg')}", code=code, msg=str(parsed.get("msg"))
            )
        return parsed

    # -- server time sync ------------------------------------------------
    def sync_server_time(
        self,
        venue: str = "both",
        timeout_sec: float = 5.0,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Dong bo gio may chu Spot/Futures -> spot/futures_time_offset_ms.

        offset = serverTime - local_ms. Moi request ky sau do se cong offset nay,
        trieu tieu 100% loi -1021 Timestamp outside of recvWindow.
        """
        result: dict[str, Any] = {
            "spot_offset_ms": self.spot_time_offset_ms,
            "futures_offset_ms": self.futures_time_offset_ms,
            "errors": [],
        }
        venues = [VENUE_SPOT, VENUE_FUTURES] if venue == "both" else [venue]
        for v in venues:
            path = PERP_TIME_PATH if v == VENUE_FUTURES else SPOT_TIME_PATH
            base = self.perp_base_url if v == VENUE_FUTURES else self.spot_base_url
            local_ms = int(now_ms if now_ms is not None else time.time() * 1000)
            try:
                resp = self._http_request(
                    "GET", f"{base}{path}", {}, {}, timeout_sec=timeout_sec
                )
                server_ms = int(resp.get("serverTime") or 0)
                if server_ms <= 0:
                    raise BinanceHttpError(f"{v}: serverTime khong hop le")
                offset = server_ms - local_ms
            except Exception as exc:
                result["errors"].append(f"{v}: {exc}")
                continue
            if v == VENUE_FUTURES:
                self.futures_time_offset_ms = int(offset)
                result["futures_offset_ms"] = int(offset)
            else:
                self.spot_time_offset_ms = int(offset)
                result["spot_offset_ms"] = int(offset)
            result[f"{v}_server_time_ms"] = server_ms
        result["synced"] = not result["errors"]
        return result

    # -- signing ---------------------------------------------------------
    def sign_params(
        self,
        params: Mapping[str, Any],
        timestamp_ms: int | None = None,
        venue: str = VENUE_SPOT,
    ) -> dict[str, Any]:
        """Them timestamp (da bu offset) + recvWindow roi tra payload da ky."""
        self.require_credentials()
        payload: dict[str, Any] = dict(params)
        payload["timestamp"] = int(
            timestamp_ms
            if timestamp_ms is not None
            else int(time.time() * 1000) + self.time_offset_ms(venue)
        )
        payload["recvWindow"] = int(self.recv_window)
        payload["signature"] = sign_binance_payload(
            payload, self.api_secret_for(venue)
        )
        return payload

    def build_signed_get_request(
        self,
        endpoint: str,
        params: dict | None = None,
        futures: bool = False,
        timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        """Dong goi request GET da ky HMAC-SHA256."""
        venue = VENUE_FUTURES if futures else VENUE_SPOT
        if not self.has_credentials():
            raise BinanceCredentialsMissingError(
                "Thieu BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET (fail-closed)."
            )
        base_url = self.perp_base_url if futures else self.spot_base_url
        clean_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        signed_params = self.sign_params(
            params or {}, timestamp_ms=timestamp_ms, venue=venue
        )
        return {
            "method": "GET",
            "url": f"{base_url}{clean_endpoint}",
            "headers": self._headers(venue),
            "params": signed_params,
        }

    # -- order building --------------------------------------------------
    def build_delta_neutral_order_pair(
        self,
        symbol: str,
        spot_qty: float,
        perp_qty: float,
        spot_price: float | None = None,
        post_only_spot: bool = True,
        timestamp_ms: int | None = None,
        spot_use_limit_maker: bool = False,
        perp_post_only: bool = False,
        perp_price: float | None = None,
        spot_side: str = "BUY",
        perp_side: str = "SELL",
        perp_reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Dung cap lenh Spot + Perp (delta-neutral) da ky day du.

        Sprint 19 - Sua chuan Post-Only:
        - Spot (`/api/v3/order`): Post-Only BAT BUOC dung ``type="LIMIT_MAKER"`` va
          KHONG duoc gui ``timeInForce`` (gui GTX len Spot se bi loi -1102).
        - Perp (`/fapi/v1/order`): Post-Only dung ``type="LIMIT", timeInForce="GTX"``.
        """
        lot_spot = _check_lot(symbol, float(spot_qty), spot_price)
        lot_perp = _check_lot(symbol, float(perp_qty))

        spot_params: dict[str, Any] = {
            "symbol": _binance_symbol(symbol),
            "side": spot_side,
            "quantity": float(spot_qty),
        }
        if post_only_spot:
            if spot_price is None or spot_price <= 0.0:
                raise ValueError("post_only_spot=True yeu cau spot_price > 0")
            if spot_use_limit_maker:
                # Binance Spot API v3: LIMIT_MAKER la cach duy nhat dat Post-Only,
                # va KHONG duoc kem timeInForce.
                spot_params.update({"type": "LIMIT_MAKER", "price": float(spot_price)})
            else:
                spot_params.update(
                    {
                        "type": "LIMIT",
                        "timeInForce": "GTX",
                        "price": float(spot_price),
                    }
                )
        else:
            spot_params["type"] = "MARKET"

        perp_params: dict[str, Any] = {
            "symbol": _binance_symbol(symbol),
            "side": perp_side,
        }
        if perp_post_only:
            if perp_price is None or perp_price <= 0.0:
                raise ValueError("perp_post_only=True yeu cau perp_price > 0")
            perp_params.update(
                {
                    "type": "LIMIT",
                    "timeInForce": "GTX",
                    "price": float(perp_price),
                    "quantity": float(perp_qty),
                }
            )
        else:
            perp_params.update({"type": "MARKET", "quantity": float(perp_qty)})
        if perp_reduce_only:
            perp_params["reduceOnly"] = True

        return {
            "symbol": symbol,
            "dry_run": bool(self.dry_run),
            "lot": {
                "spot": lot_spot,
                "perp": lot_perp,
                "notional_spot_usd": (float(spot_qty) * float(spot_price))
                if spot_price is not None
                else None,
            },
            "spot": {
                "method": "POST",
                "venue": VENUE_SPOT,
                "url": f"{self.spot_base_url}{SPOT_ORDER_PATH}",
                "headers": self._headers(VENUE_SPOT),
                "params": self.sign_params(
                    spot_params, timestamp_ms=timestamp_ms, venue=VENUE_SPOT
                ),
            },
            "perp": {
                "method": "POST",
                "venue": VENUE_FUTURES,
                "url": f"{self.perp_base_url}{PERP_ORDER_PATH}",
                "headers": self._headers(VENUE_FUTURES),
                "params": self.sign_params(
                    perp_params, timestamp_ms=timestamp_ms, venue=VENUE_FUTURES
                ),
            },
        }

    def build_order_request(
        self,
        venue: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
        time_in_force: str | None = None,
        reduce_only: bool = False,
        price_check: float | None = None,
        timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        """Dong goi 1 request dat lenh da ky (dung cho entry/exit/rollback)."""
        params: dict[str, Any] = {
            "symbol": _binance_symbol(symbol),
            "side": str(side).upper(),
            "quantity": float(quantity),
            "type": str(order_type).upper(),
        }
        if order_type.upper() in ("LIMIT", "LIMIT_MAKER", "STOP", "TAKE_PROFIT"):
            if price is None or float(price) <= 0.0:
                raise ValueError(f"{order_type} yeu cau price > 0")
            params["price"] = float(price)
        if time_in_force and order_type.upper() not in ("LIMIT_MAKER", "MARKET"):
            params["timeInForce"] = str(time_in_force).upper()
        if reduce_only and venue == VENUE_FUTURES:
            params["reduceOnly"] = True
        if price_check is not None and price_check > 0.0:
            _check_lot(symbol, float(quantity), float(price_check))
        base = self.perp_base_url if venue == VENUE_FUTURES else self.spot_base_url
        path = PERP_ORDER_PATH if venue == VENUE_FUTURES else SPOT_ORDER_PATH
        return {
            "method": "POST",
            "venue": venue,
            "url": f"{base}{path}",
            "headers": self._headers(venue),
            "params": self.sign_params(params, timestamp_ms=timestamp_ms, venue=venue),
        }

    def fetch_symbol_filters(
        self, symbol: str, venue: str = VENUE_SPOT, timeout_sec: float = 5.0
    ) -> dict[str, float]:
        """Lay tick_size / step_size / min_notional THAT tu exchangeInfo.

        Bat buoc de tranh -1013 (PRICE_FILTER) khi gui Post-Only voi gia qua nhieu
        chu so thap phan, va de kiem tra min_notional chinh xac truoc khi dat lenh.
        """
        venue = VENUE_FUTURES if venue == VENUE_FUTURES else VENUE_SPOT
        base = self.perp_base_url if venue == VENUE_FUTURES else self.spot_base_url
        path = "/fapi/v1/exchangeInfo" if venue == VENUE_FUTURES else "/api/v3/exchangeInfo"
        raw = self._http_request(
            "GET", f"{base}{path}", {"symbol": _binance_symbol(symbol)}, {},
            timeout_sec=timeout_sec,
        )
        filters = {"tick_size": 0.0, "step_size": 0.0, "min_notional": 0.0}
        entries = raw.get("symbols") if isinstance(raw, dict) else None
        if not isinstance(entries, list) or not entries:
            return filters
        entry = entries[0] if isinstance(entries[0], dict) else {}
        for f in entry.get("filters", []) or []:
            if not isinstance(f, dict):
                continue
            ftype = str(f.get("filterType", "")).upper()
            if ftype == "PRICE_FILTER":
                filters["tick_size"] = _safe_float(f.get("tickSize"))
            elif ftype == "LOT_SIZE":
                filters["step_size"] = _safe_float(f.get("stepSize"))
            elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                val = f.get("minNotional", f.get("notional"))
                filters["min_notional"] = _safe_float(val)
        return filters

    # -- live order submission (Sprint 19) --------------------------------
    def submit_signed_order(
        self,
        venue: str,
        params: dict,
        timeout_sec: float = 10.0,
        already_signed: bool = False,
    ) -> dict[str, Any]:
        """Gui lenh that (dry_run=False) hoac tra payload mo phong (dry_run=True).

        ``params`` co the la:
        - dict tham so lenh chua ky (tu dong ky voi offset gio dung venue), hoac
        - request envelope ``{"url":..,"headers":..,"params":..}`` (tu build_order_request).

        dry_run=True: chi ky + tra payload mo phong (tuong thich 100% unit test cu).
        """
        venue = VENUE_FUTURES if venue == VENUE_FUTURES else VENUE_SPOT
        envelope: dict[str, Any] | None = None
        if (
            isinstance(params, dict)
            and "params" in params
            and "url" in params
            and isinstance(params.get("params"), dict)
        ):
            envelope = dict(params)
            raw_params: dict[str, Any] = dict(envelope.get("params") or {})
        else:
            raw_params = dict(params or {})

        base = self.perp_base_url if venue == VENUE_FUTURES else self.spot_base_url
        path = PERP_ORDER_PATH if venue == VENUE_FUTURES else SPOT_ORDER_PATH
        url = str(envelope.get("url")) if envelope else f"{base}{path}"
        headers = dict(envelope.get("headers") or {}) if envelope else self._headers(venue)
        headers.setdefault("X-MBX-APIKEY", self.api_key_for(venue) or "")

        signed = raw_params
        # Bat bien: KHONG BAO GIO ky lai payload da co signature (se tao chu ky
        # long nhau -> Binance tra -1022 Signature for this request is not valid).
        if not already_signed and "signature" not in raw_params:
            signed = self.sign_params(raw_params, venue=venue)

        if self.dry_run:
            return {
                "dry_run": True,
                "submitted": False,
                "venue": venue,
                "url": url,
                "headers": headers,
                "params": signed,
                "orderId": None,
                "status": "DRY_RUN",
                "executedQty": 0.0,
                "cummulativeQuoteQty": 0.0,
                "avgPrice": 0.0,
                "filled": False,
            }

        raw = self._http_request(
            "POST", url, signed, headers, timeout_sec=float(timeout_sec)
        )
        result = normalize_order_result(venue, raw)
        # KHONG tra ve headers: header chua X-MBX-APIKEY, rat de ro ri vao log.
        result.update({"dry_run": False, "submitted": True, "url": url})
        result["filled"] = _is_filled(result)
        return result

    def query_order(
        self,
        venue: str,
        symbol: str,
        order_id: int | str,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        """Truy van trang thai 1 lenh da dat (GET /api/v3/order hoac /fapi/v1/order)."""
        venue = VENUE_FUTURES if venue == VENUE_FUTURES else VENUE_SPOT
        params = {"symbol": _binance_symbol(symbol), "orderId": order_id}
        req = self.build_signed_get_request(
            PERP_ORDER_PATH if venue == VENUE_FUTURES else SPOT_ORDER_PATH,
            params=params,
            futures=(venue == VENUE_FUTURES),
        )
        raw = self._http_request(
            "GET", req["url"], req["params"], req["headers"], timeout_sec=timeout_sec
        )
        result = normalize_order_result(venue, raw)
        result["filled"] = _is_filled(result)
        return result

    def cancel_order(
        self,
        venue: str,
        symbol: str,
        order_id: int | str,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        """Huy 1 lenh con treo (DELETE) - dung khi Post-Only khong khop sau TTL."""
        venue = VENUE_FUTURES if venue == VENUE_FUTURES else VENUE_SPOT
        params = {"symbol": _binance_symbol(symbol), "orderId": order_id}
        if self.dry_run:
            return {"dry_run": True, "canceled": False, "orderId": order_id}
        base = self.perp_base_url if venue == VENUE_FUTURES else self.spot_base_url
        path = PERP_ORDER_PATH if venue == VENUE_FUTURES else SPOT_ORDER_PATH
        signed = self.sign_params(params, venue=venue)
        try:
            raw = self._http_request(
                "DELETE", f"{base}{path}", signed, self._headers(venue),
                timeout_sec=timeout_sec,
            )
        except BinanceHttpError as exc:
            return {"canceled": False, "orderId": order_id, "error": str(exc)}
        return {"canceled": True, "orderId": order_id, "raw": raw}

    def fetch_book_ticker(
        self, symbol: str, venue: str = VENUE_SPOT, timeout_sec: float = 5.0
    ) -> dict[str, float]:
        """Lay best bid/ask that tu bookTicker (public, khong can ky)."""
        venue = VENUE_FUTURES if venue == VENUE_FUTURES else VENUE_SPOT
        base = self.perp_base_url if venue == VENUE_FUTURES else self.spot_base_url
        path = PERP_BOOK_TICKER_PATH if venue == VENUE_FUTURES else SPOT_BOOK_TICKER_PATH
        raw = self._http_request(
            "GET", f"{base}{path}", {"symbol": _binance_symbol(symbol)}, {},
            timeout_sec=timeout_sec,
        )
        bid = _safe_float(raw.get("bidPrice"))
        ask = _safe_float(raw.get("askPrice"))
        mid = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else max(bid, ask)
        return {"bid": bid, "ask": ask, "mid": mid, "raw": raw}

    def check_account_access(
        self, venue: str, timeout_sec: float = 5.0
    ) -> dict[str, Any]:
        """Kiem tra xac thuc cua tung venue; chan doan ro loi dung nham khoa."""
        venue = VENUE_FUTURES if venue == VENUE_FUTURES else VENUE_SPOT
        if not (self.has_futures_credentials() if venue == VENUE_FUTURES
                else self.has_spot_credentials()):
            return {
                "venue": venue,
                "authenticated": False,
                "error": f"MISSING_{venue.upper()}_CREDENTIALS",
                "hint": (
                    FUTURES_KEY_INVALID_HINT if venue == VENUE_FUTURES
                    else "Hay tao khoa Spot Testnet tai https://testnet.binance.vision"
                ),
            }
        try:
            req = self.build_signed_get_request(
                PERP_ACCOUNT_PATH if venue == VENUE_FUTURES else SPOT_ACCOUNT_PATH,
                futures=(venue == VENUE_FUTURES),
            )
            raw = self._http_request(
                "GET", req["url"], req["params"], req["headers"], timeout_sec=timeout_sec
            )
        except BinanceHttpError as exc:
            if venue == VENUE_FUTURES and exc.code in FUTURES_INVALID_ERROR_CODES:
                return {
                    "venue": venue,
                    "authenticated": False,
                    "error": FUTURES_KEY_INVALID_CODE,
                    "hint": FUTURES_KEY_INVALID_HINT,
                    "api_error_code": exc.code,
                }
            return {
                "venue": venue,
                "authenticated": False,
                "error": f"API_ERROR_{exc.code}",
                "hint": str(exc),
                "api_error_code": exc.code,
            }
        except Exception as exc:
            return {
                "venue": venue,
                "authenticated": False,
                "error": "NETWORK_OR_AUTH_FAILURE",
                "hint": str(exc),
            }
        holdings: dict[str, float] = {}
        if isinstance(raw.get("balances"), list):
            for b in raw["balances"]:
                if isinstance(b, dict):
                    a = str(b.get("asset", "")).upper()
                    free = _safe_float(b.get("free"))
                    if a and free > 0.0:
                        holdings[a] = free
        elif isinstance(raw.get("assets"), list):
            for b in raw["assets"]:
                if isinstance(b, dict):
                    a = str(b.get("asset", "")).upper()
                    free = _safe_float(b.get("availableBalance", b.get("walletBalance")))
                    if a and free > 0.0:
                        holdings[a] = free
        return {
            "venue": venue,
            "authenticated": True,
            "error": None,
            "holdings": holdings,
            "server_time_offset_ms": self.time_offset_ms(venue),
        }

    def execute_live_testnet_order_pair(
        self,
        symbol: str,
        spot_qty: float,
        perp_qty: float,
        spot_price: float | None = None,
        perp_price: float | None = None,
        direction: str = "open",
        post_only_spot: bool = True,
        spot_use_limit_maker: bool = True,
        perp_post_only: bool = False,
        maker_ttl_sec: float = 2.0,
        poll_interval_sec: float = 0.2,
        timeout_sec: float = 10.0,
        require_futures: bool = False,
        perp_available: bool | None = None,
    ) -> dict[str, Any]:
        """Dat cap lenh Spot + Perp voi fallback maker->market va Atomic Leg Rollback.

        Bat bien an toan: KHONG BAO GIO de lai vi the tran (naked leg). Neu chan 1
        (Spot) da khop ma chan 2 (Perp) that bai -> lap tuc gui lenh MARKET nguoc
        chieu dong chan 1 va tra ve ``status="ROLLED_BACK_SAFE"``.
        """
        open_leg = str(direction).lower() != "close"
        spot_side = "BUY" if open_leg else "SELL"
        perp_side = "SELL" if open_leg else "BUY"
        events: list[dict[str, Any]] = []

        if self.dry_run:
            pair = self.build_delta_neutral_order_pair(
                symbol=symbol,
                spot_qty=spot_qty,
                perp_qty=perp_qty,
                spot_price=spot_price,
                post_only_spot=post_only_spot,
                spot_use_limit_maker=spot_use_limit_maker,
                perp_post_only=perp_post_only,
                perp_price=perp_price,
                spot_side=spot_side,
                perp_side=perp_side,
                perp_reduce_only=not open_leg,
            )
            return {
                "dry_run": True,
                "submitted": False,
                "status": "DRY_RUN",
                "rolled_back": False,
                "direction": "open" if open_leg else "close",
                "spot": pair["spot"],
                "perp": pair["perp"],
                "events": events,
            }

        # --- Chan 1: Spot (maker -> market fallback) --------------------
        # A.1: bat moi ngoai le cua chan Spot de con duong rollback luon duoc cham toi
        # (neu maker da khop mot phan roi loi mang -> phai dong ngay phan da khop).
        spot_tracker: dict[str, float] = {"executed_qty": 0.0}
        try:
            spot_result, events = self._execute_spot_leg(
                symbol=symbol,
                qty=float(spot_qty),
                side=spot_side,
                price=spot_price,
                post_only=post_only_spot,
                use_limit_maker=spot_use_limit_maker,
                maker_ttl_sec=maker_ttl_sec,
                poll_interval_sec=poll_interval_sec,
                timeout_sec=timeout_sec,
                events=events,
                tracker=spot_tracker,
            )
        except Exception as exc:  # noqa: BLE001 - khong duoc de lo vi the tran
            events.append({"stage": "spot", "event": "SPOT_LEG_EXCEPTION", "detail": str(exc)})
            known_qty = _safe_float(spot_tracker.get("executed_qty"))
            if known_qty > 0.0:
                rollback = self._rollback_filled_spot(
                    symbol=symbol,
                    reverse_side="SELL" if spot_side == "BUY" else "BUY",
                    qty=known_qty,
                    timeout_sec=timeout_sec,
                    events=events,
                )
                rolled_back_ok = bool(rollback.get("filled")) and _safe_float(
                    rollback.get("executedQty")
                ) > 0.0
                return {
                    "dry_run": False,
                    "status": "ROLLED_BACK_SAFE" if rolled_back_ok else "NAKED_LEG_ALERT",
                    "rolled_back": True,
                    "naked_leg": not rolled_back_ok,
                    "direction": "open" if open_leg else "close",
                    "spot": {"venue": VENUE_SPOT, "status": "EXCEPTION",
                             "executedQty": known_qty, "filled": False, "error": str(exc)},
                    "perp": None,
                    "rollback": rollback,
                    "events": events,
                }
            return {
                "dry_run": False,
                "status": "SPOT_LEG_EXCEPTION",
                "rolled_back": False,
                "naked_leg": False,
                "direction": "open" if open_leg else "close",
                "spot": {"venue": VENUE_SPOT, "status": "EXCEPTION", "executedQty": 0.0,
                         "filled": False, "error": str(exc)},
                "perp": None,
                "events": events,
            }
        spot_qty_filled = _safe_float(spot_result.get("executedQty"))

        if not spot_result.get("filled") or spot_qty_filled <= 0.0:
            return {
                "dry_run": False,
                "status": "SPOT_LEG_FAILED",
                "rolled_back": False,
                "naked_leg": False,
                "direction": "open" if open_leg else "close",
                "spot": spot_result,
                "perp": None,
                "events": events,
            }

        # --- Chan 2: Perp ----------------------------------------------
        futures_usable = self.has_futures_credentials() and (perp_available is not False)
        if not futures_usable:
            # Che do "mot khoa": mo phong CO KY tren nhanh chua co khoa hop le,
            # khong bao gio gui lenh that (tranh kich hoat rollback vo ich).
            perp_result = self._simulate_signed_perp_leg(
                symbol=symbol,
                qty=float(perp_qty),
                side=perp_side,
                reduce_only=not open_leg,
                reason=(
                    "FUTURES_KEY_UNUSABLE" if self.has_futures_credentials()
                    else "NO_FUTURES_CREDENTIALS"
                ),
            )
            events.append(
                {"stage": "perp", "event": "SIMULATED_SIGNED_ONLY",
                 "reason": perp_result.get("reason")}
            )
            perp_ok = False
        else:
            try:
                perp_result = self._execute_perp_leg(
                    symbol=symbol,
                    qty=float(perp_qty),
                    side=perp_side,
                    price=perp_price,
                    post_only=perp_post_only,
                    reduce_only=not open_leg,
                    maker_ttl_sec=maker_ttl_sec,
                    poll_interval_sec=poll_interval_sec,
                    timeout_sec=timeout_sec,
                    events=events,
                )
                perp_ok = bool(perp_result.get("filled")) and _safe_float(
                    perp_result.get("executedQty")
                ) > 0.0
            except Exception as exc:  # noqa: BLE001 - rollback bat buoc
                perp_result = {
                    "venue": VENUE_FUTURES,
                    "status": "PERP_LEG_EXCEPTION",
                    "executedQty": 0.0,
                    "filled": False,
                    "error": str(exc),
                }
                events.append({"stage": "perp", "event": "EXCEPTION", "detail": str(exc)})
                perp_ok = False

        if perp_ok or (perp_result.get("simulated") and not require_futures):
            return {
                "dry_run": False,
                "status": "FILLED",
                "rolled_back": False,
                "naked_leg": False,
                "perp_simulated": bool(perp_result.get("simulated")),
                "direction": "open" if open_leg else "close",
                "spot": spot_result,
                "perp": perp_result,
                "events": events,
            }

        # --- ATOMIC LEG ROLLBACK ---------------------------------------
        reverse_side = "SELL" if spot_side == "BUY" else "BUY"
        rollback = self._rollback_filled_spot(
            symbol=symbol,
            reverse_side=reverse_side,
            qty=spot_qty_filled,
            timeout_sec=timeout_sec,
            events=events,
        )

        rolled_back_ok = bool(rollback.get("filled")) and _safe_float(
            rollback.get("executedQty")
        ) > 0.0
        return {
            "dry_run": False,
            "status": "ROLLED_BACK_SAFE" if rolled_back_ok else "NAKED_LEG_ALERT",
            "rolled_back": True,
            "naked_leg": not rolled_back_ok,
            "direction": "open" if open_leg else "close",
            "spot": spot_result,
            "perp": perp_result,
            "rollback": rollback,
            "events": events,
        }

    def _rollback_filled_spot(
        self,
        symbol: str,
        reverse_side: str,
        qty: float,
        timeout_sec: float,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Gui lenh MARKET nguoc chieu de dong ngay khoi luong Spot da khop."""
        rollback: dict[str, Any] = {
            "venue": VENUE_SPOT,
            "side": reverse_side,
            "qty": qty,
            "orderId": None,
            "status": "ROLLBACK_FAILED",
            "filled": False,
            "executedQty": 0.0,
        }
        try:
            rb_req = self.build_order_request(
                venue=VENUE_SPOT,
                symbol=symbol,
                side=reverse_side,
                quantity=qty,
                order_type="MARKET",
            )
            rb_res = self.submit_signed_order(
                VENUE_SPOT, rb_req, timeout_sec=timeout_sec, already_signed=True
            )
            rollback.update(rb_res)
            rollback["side"] = reverse_side
            rollback["qty"] = qty
            rollback["filled"] = bool(rb_res.get("filled"))
            events.append(
                {
                    "stage": "rollback",
                    "event": "REVERSE_MARKET_SUBMITTED",
                    "orderId": rb_res.get("orderId"),
                    "filled": bool(rb_res.get("filled")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - bao dong naked leg
            rollback["error"] = str(exc)
            events.append({"stage": "rollback", "event": "ROLLBACK_EXCEPTION", "detail": str(exc)})
        return rollback

    # -- leg-level helpers ------------------------------------------------
    def _simulate_signed_perp_leg(
        self,
        symbol: str,
        qty: float,
        side: str,
        reduce_only: bool,
        reason: str,
    ) -> dict[str, Any]:
        """Mo phong CO KY cho chan Perp khi khoa Futures chua hop le (khong gui)."""
        result: dict[str, Any] = {
            "venue": VENUE_FUTURES,
            "status": "SIMULATED_SIGNED_ONLY",
            "orderId": None,
            "executedQty": 0.0,
            "filled": False,
            "simulated": True,
            "reason": reason,
            "error": FUTURES_KEY_INVALID_HINT,
        }
        try:
            req = self.build_order_request(
                VENUE_FUTURES, symbol, side, qty, "MARKET", reduce_only=reduce_only
            )
            sig = str(req["params"].get("signature", ""))
            result["signed"] = bool(sig)
            result["signature_prefix"] = sig[:8]
            result["params"] = req["params"]
            result["url"] = req["url"]
        except Exception as exc:  # noqa: BLE001 - chan doan, khong raise
            result["signed"] = False
            result["signing_error"] = str(exc)
        return result

    def _execute_spot_leg(
        self,
        symbol: str,
        qty: float,
        side: str,
        price: float | None,
        post_only: bool,
        use_limit_maker: bool,
        maker_ttl_sec: float,
        poll_interval_sec: float,
        timeout_sec: float,
        events: list[dict[str, Any]],
        tracker: dict[str, float] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def _track(result: Mapping[str, Any]) -> Mapping[str, Any]:
            """Ghi lai khoi luong da khop LON NHAT duoc biet (de rollback khi loi)."""
            if tracker is not None:
                tracker["executed_qty"] = max(
                    _safe_float(tracker.get("executed_qty")),
                    _safe_float(result.get("executedQty")),
                )
            return result

        want_maker = bool(post_only and use_limit_maker and price and price > 0.0)
        if not want_maker:
            req = self.build_order_request(
                VENUE_SPOT, symbol, side, qty, "MARKET", price_check=price
            )
            res = _track(self.submit_signed_order(VENUE_SPOT, req, timeout_sec=timeout_sec))
            events.append({"stage": "spot", "event": "MARKET_SUBMITTED", "orderId": res.get("orderId")})
            return self._await_fill(VENUE_SPOT, symbol, res, maker_ttl_sec, poll_interval_sec, events), events

        # 1. Thu Post-Only LIMIT_MAKER (Khong kem timeInForce tren Spot!)
        try:
            req = self.build_order_request(
                VENUE_SPOT, symbol, side, qty, "LIMIT_MAKER", price=price, price_check=price
            )
            res = _track(self.submit_signed_order(VENUE_SPOT, req, timeout_sec=timeout_sec))
            events.append({"stage": "spot", "event": "LIMIT_MAKER_SUBMITTED", "orderId": res.get("orderId")})
        except BinanceHttpError as exc:
            events.append(
                {
                    "stage": "spot",
                    "event": "LIMIT_MAKER_REJECTED",
                    "code": exc.code,
                    "fallback": "MARKET",
                }
            )
            req = self.build_order_request(
                VENUE_SPOT, symbol, side, qty, "MARKET", price_check=price
            )
            res = _track(self.submit_signed_order(VENUE_SPOT, req, timeout_sec=timeout_sec))
            res["fallback_from"] = "LIMIT_MAKER"
            events.append({"stage": "spot", "event": "MARKET_FALLBACK", "orderId": res.get("orderId")})
            return self._await_fill(VENUE_SPOT, symbol, res, maker_ttl_sec, poll_interval_sec, events), events

        if _is_filled(res):
            return res, events

        filled = _track(
            self._await_fill(VENUE_SPOT, symbol, res, maker_ttl_sec, poll_interval_sec, events)
        )
        if filled.get("filled"):
            return filled, events

        # 2. Khong khop sau TTL -> huy va ban MARKET phan CON LAI (tranh double fill)
        order_id = res.get("orderId")
        if order_id is not None:
            cancel_res = self.cancel_order(VENUE_SPOT, symbol, order_id, timeout_sec=timeout_sec)
            events.append(
                {"stage": "spot", "event": "LIMIT_MAKER_CANCELED", "orderId": order_id,
                 "canceled": bool(cancel_res.get("canceled"))}
            )
        # Sau khi huy, doc lai khoi luong DA KHOP de chi mua phan con thieu.
        already_filled = _safe_float(filled.get("executedQty"))
        if order_id is not None:
            try:
                final_state = self.query_order(VENUE_SPOT, symbol, order_id, timeout_sec=timeout_sec)
                already_filled = max(already_filled, _safe_float(final_state.get("executedQty")))
            except Exception as exc:  # noqa: BLE001 - giu already_filled da biet
                events.append({"stage": "spot", "event": "POST_CANCEL_QUERY_ERROR", "detail": str(exc)})
        remaining_qty = round(float(qty) - already_filled, 12)
        if remaining_qty <= 0.0:
            events.append(
                {"stage": "spot", "event": "MAKER_FILLED_AFTER_CANCEL",
                 "executedQty": already_filled}
            )
            filled["filled"] = _safe_float(filled.get("executedQty")) > 0.0
            return filled, events
        req = self.build_order_request(
            VENUE_SPOT, symbol, side, remaining_qty, "MARKET", price_check=price
        )
        mkt_raw = self.submit_signed_order(VENUE_SPOT, req, timeout_sec=timeout_sec)
        events.append({"stage": "spot", "event": "MARKET_FALLBACK", "orderId": mkt_raw.get("orderId"),
                       "remaining_qty": remaining_qty, "maker_filled_qty": already_filled})
        # Poll tren ket qua THO (chua cong don), roi moi cong don -> neu lenh MARKET
        # can poll thi ket qua poll van duoc cong dung khoi luong maker da khop.
        final = dict(self._await_fill(
            VENUE_SPOT, symbol, mkt_raw, maker_ttl_sec, poll_interval_sec, events
        ))
        final["fallback_from"] = "LIMIT_MAKER_TTL_EXPIRED"
        final["maker_filled_qty"] = already_filled
        final["requested_qty"] = float(remaining_qty)
        final["market_executed_qty"] = _safe_float(final.get("executedQty"))
        # Cong don khoi luong da khop o lenh maker + lenh market (khong bao gio > qty goc).
        total_filled = min(
            float(qty), already_filled + _safe_float(final.get("market_executed_qty"))
        )
        final["executedQty"] = total_filled
        final["filled"] = total_filled > 0.0 and (
            _is_filled(final) or _safe_float(final.get("market_executed_qty")) > 0.0
        )
        _track(final)
        return final, events

    def _execute_perp_leg(
        self,
        symbol: str,
        qty: float,
        side: str,
        price: float | None,
        post_only: bool,
        reduce_only: bool,
        maker_ttl_sec: float,
        poll_interval_sec: float,
        timeout_sec: float,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if post_only and price and price > 0.0:
            req = self.build_order_request(
                VENUE_FUTURES, symbol, side, qty, "LIMIT",
                price=price, time_in_force="GTX", reduce_only=reduce_only,
            )
        else:
            req = self.build_order_request(
                VENUE_FUTURES, symbol, side, qty, "MARKET", reduce_only=reduce_only
            )
        res = self.submit_signed_order(VENUE_FUTURES, req, timeout_sec=timeout_sec)
        events.append(
            {"stage": "perp", "event": f"{res.get('type') or 'PERP'}_SUBMITTED",
             "orderId": res.get("orderId")}
        )
        return self._await_fill(
            VENUE_FUTURES, symbol, res, maker_ttl_sec, poll_interval_sec, events
        )

    def _await_fill(
        self,
        venue: str,
        symbol: str,
        res: dict[str, Any],
        max_wait_sec: float,
        poll_interval_sec: float,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Poll trang thai lenh toi khi FILLED hoac het TTL (best-effort)."""
        if _is_filled(res) or res.get("dry_run"):
            return res
        order_id = res.get("orderId")
        if order_id is None:
            return res
        deadline = time.time() + max(0.0, float(max_wait_sec))
        interval = max(0.0, float(poll_interval_sec))
        last = res
        while time.time() < deadline:
            if interval:
                time.sleep(interval)
            try:
                last = self.query_order(venue, symbol, order_id)
            except Exception as exc:  # noqa: BLE001 - khong lam hong luong chinh
                events.append({"stage": venue, "event": "POLL_ERROR", "detail": str(exc)})
                return res
            if _is_filled(last):
                events.append(
                    {"stage": venue, "event": "FILLED", "orderId": order_id,
                     "executedQty": last.get("executedQty")}
                )
                return last
        return last

    def place_maker_with_fallback(
        self,
        symbol: str,
        qty: float,
        side: str,
        best_bid: float,
        best_ask: float,
        tick_size: float,
        venue: str = VENUE_SPOT,
        improve_inside_spread: bool = True,
        maker_ttl_sec: float = 2.0,
        poll_interval_sec: float = 0.2,
        timeout_sec: float = 10.0,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Sprint 21: dat lenh Maker Post-Only voi gia Smart Inside-Spread + fallback market.

        Khong bao gio gui GTX tai gia cat qua spread (nguyen nhan bi tu choi -2010/-2021),
        nho do giam ty le fallback market va tang toc do khop Maker tren Testnet/Mainnet.
        """
        venue_name = str(venue or VENUE_SPOT).strip().lower()
        if venue_name not in (VENUE_SPOT, VENUE_FUTURES):
            return {
                "status": "INVALID_VENUE",
                "filled": False,
                "venue": venue_name,
                "error": f"venue must be '{VENUE_SPOT}' or '{VENUE_FUTURES}'",
            }

        pricing = compute_maker_limit_price(
            side=side,
            best_bid=best_bid,
            best_ask=best_ask,
            tick_size=tick_size,
            improve_inside_spread=improve_inside_spread,
        )
        price = float(pricing.get("price", 0.0) or 0.0)
        side_u = str(side or "").strip().upper()
        if price <= 0.0:
            return {
                "status": "INVALID_PRICE",
                "filled": False,
                "venue": venue_name,
                "side": side_u,
                "qty": float(qty),
                "pricing": pricing,
                "error": "Khong the tinh gia maker tu so lenh (book rong)",
            }

        if self.dry_run:
            return {
                "dry_run": True,
                "submitted": False,
                "status": "DRY_RUN",
                "filled": False,
                "venue": venue_name,
                "symbol": symbol,
                "side": side_u,
                "qty": float(qty),
                "price": price,
                "pricing": pricing,
                "events": [],
            }

        events: list[dict[str, Any]] = []
        if venue_name == VENUE_SPOT:
            order, events = self._execute_spot_leg(
                symbol=symbol,
                qty=float(qty),
                side=side_u,
                price=price,
                post_only=True,
                use_limit_maker=True,
                maker_ttl_sec=maker_ttl_sec,
                poll_interval_sec=poll_interval_sec,
                timeout_sec=timeout_sec,
                events=events,
            )
        else:
            gtx_submitted = True
            try:
                order = self._execute_perp_leg(
                    symbol=symbol,
                    qty=float(qty),
                    side=side_u,
                    price=price,
                    post_only=True,
                    reduce_only=reduce_only,
                    maker_ttl_sec=maker_ttl_sec,
                    poll_interval_sec=poll_interval_sec,
                    timeout_sec=timeout_sec,
                    events=events,
                )
            except BinanceHttpError as exc:
                # GTX bi tu choi (post-only/crossing) -> khong duoc crash, phai ban MARKET.
                gtx_submitted = False
                events.append(
                    {"stage": "perp", "event": "GTX_REJECTED", "code": exc.code,
                     "fallback": "MARKET"}
                )
                req = self.build_order_request(
                    VENUE_FUTURES, symbol, side_u, float(qty), "MARKET",
                    reduce_only=reduce_only,
                )
                try:
                    market_raw = self.submit_signed_order(
                        VENUE_FUTURES, req, timeout_sec=timeout_sec
                    )
                except BinanceHttpError as mkt_exc:
                    return {
                        "status": "MARKET_FALLBACK_REJECTED",
                        "filled": False,
                        "venue": venue_name,
                        "symbol": symbol,
                        "side": side_u,
                        "qty": float(qty),
                        "price": price,
                        "pricing": pricing,
                        "error": f"market fallback rejected: {mkt_exc}",
                        "events": events + [
                            {"stage": "perp", "event": "MARKET_FALLBACK_REJECTED",
                             "code": mkt_exc.code}
                        ],
                    }
                order = dict(self._await_fill(
                    VENUE_FUTURES, symbol, market_raw,
                    maker_ttl_sec, poll_interval_sec, events,
                ))
                order["fallback_from"] = "GTX_REJECTED"
                order["maker_filled_qty"] = 0.0
                order["filled"] = _is_filled(order)

            # Sprint 21: GTX Futures het TTL ma chua khop -> HUY + MARKET phan con lai
            # (khong de lenh limit treo lo lung lam lech delta-neutral).
            # Chi chay nhanh nay khi lenh vua gui THAT SU la GTX: neu da fallback MARKET
            # do bi tu choi thi KHONG duoc huy/ban tiep (tranh double fill).
            if gtx_submitted and not _is_filled(order):
                order_id = order.get("orderId")
                filled_qty = _safe_float(order.get("executedQty"))
                state_known = True
                final_state: dict[str, Any] = {}
                if order_id is not None:
                    try:
                        cancel_res = self.cancel_order(
                            VENUE_FUTURES, symbol, order_id, timeout_sec=timeout_sec
                        )
                        events.append(
                            {"stage": "perp", "event": "GTX_CANCELED", "orderId": order_id,
                             "canceled": bool(cancel_res.get("canceled"))}
                        )
                    except Exception as exc:  # noqa: BLE001 - huy that bai khong chan query
                        events.append(
                            {"stage": "perp", "event": "GTX_CANCEL_ERROR", "detail": str(exc)}
                        )
                    try:
                        final_state = self.query_order(
                            VENUE_FUTURES, symbol, order_id, timeout_sec=timeout_sec
                        )
                        filled_qty = max(filled_qty, _safe_float(final_state.get("executedQty")))
                    except Exception as exc:  # noqa: BLE001 - khong chac trang thai lenh
                        state_known = False
                        events.append(
                            {"stage": "perp", "event": "POST_CANCEL_QUERY_ERROR", "detail": str(exc)}
                        )
                remaining_qty = round(float(qty) - filled_qty, 12)
                if not state_known:
                    # Khong xac dinh duoc da khop bao nhieu -> TUYET DOI khong ban MARKET
                    # tiep (tranh double fill), bao UNKNOWN de caller xu ly an toan.
                    order = dict(order)
                    order["filled"] = False
                    order["status"] = "UNKNOWN_AFTER_CANCEL"
                    order["fallback_from"] = "GTX_TTL_UNKNOWN_STATE"
                elif remaining_qty > 0.0:
                    req = self.build_order_request(
                        VENUE_FUTURES, symbol, side_u, remaining_qty, "MARKET",
                        reduce_only=reduce_only,
                    )
                    try:
                        market_raw = self.submit_signed_order(
                            VENUE_FUTURES, req, timeout_sec=timeout_sec
                        )
                    except BinanceHttpError as mkt_exc:
                        # Lenh MARKET bi tu choi (min notional/step size/ky quy) ->
                        # tra ve cau truc loi thay vi crash giua chu ky.
                        events.append(
                            {"stage": "perp", "event": "MARKET_FALLBACK_REJECTED",
                             "code": mkt_exc.code}
                        )
                        order = dict(order)
                        order["filled"] = False
                        order["status"] = "MARKET_FALLBACK_REJECTED"
                        order["fallback_from"] = "GTX_TTL_EXPIRED"
                        order["maker_filled_qty"] = filled_qty
                        order["error"] = str(mkt_exc)
                        return {
                            "status": "MARKET_FALLBACK_REJECTED",
                            "filled": False,
                            "venue": venue_name,
                            "symbol": symbol,
                            "side": side_u,
                            "qty": float(qty),
                            "price": price,
                            "pricing": pricing,
                            "order": order,
                            "error": str(mkt_exc),
                            "events": events,
                        }
                    events.append(
                        {"stage": "perp", "event": "MARKET_FALLBACK",
                         "orderId": market_raw.get("orderId"),
                         "remaining_qty": remaining_qty, "maker_filled_qty": filled_qty}
                    )
                    final = dict(self._await_fill(
                        VENUE_FUTURES, symbol, market_raw,
                        maker_ttl_sec, poll_interval_sec, events,
                    ))
                    final["fallback_from"] = "GTX_TTL_EXPIRED"
                    final["maker_filled_qty"] = filled_qty
                    market_filled = _safe_float(final.get("executedQty"))
                    final["market_executed_qty"] = market_filled
                    final["executedQty"] = min(float(qty), filled_qty + market_filled)
                    # Chi bao FILLED khi lenh MARKET thuc su FILLED (khong "fake FILLED"
                    # khi chi khop mot phan -> de caller rollback/dung an toan).
                    final["filled"] = _is_filled(final)
                    if final["filled"]:
                        final["status"] = str(final.get("status") or "FILLED")
                    order = final
                else:
                    # Maker da khop 100% ngay truoc/trong luc huy -> phai ghi nhan
                    # trang thai THAT tu san, neu khong se bao that bai sai.
                    if state_known and final_state:
                        filled_state = dict(final_state)
                        filled_state["filled"] = _is_filled(filled_state) or (
                            filled_qty + 1e-12 >= float(qty)
                        )
                        if filled_state["filled"]:
                            filled_state["status"] = "FILLED"
                        filled_state["maker_filled_qty"] = filled_qty
                        order = filled_state

        filled = (
            bool(order.get("filled"))
            and _is_filled(order)
            and _safe_float(order.get("executedQty")) > 0.0
        )
        return {
            "status": str(order.get("status") or ("FILLED" if filled else "UNFILLED")),
            "filled": filled,
            "venue": venue_name,
            "symbol": symbol,
            "side": side_u,
            "qty": float(qty),
            "price": price,
            "pricing": pricing,
            "order": order,
            "events": events,
        }

    def place_spot_market_order(self, symbol: str, side: str, quantity: float, timeout_sec: float = 10.0) -> dict[str, Any]:
        """Sprint 23: dat lenh MARKET tren Spot (ky HMAC, khong dry-run khi client dry_run=False)."""
        req = self.build_order_request(VENUE_SPOT, symbol, str(side).upper(), float(quantity), order_type="MARKET")
        result = self.submit_signed_order(VENUE_SPOT, req, timeout_sec=timeout_sec)
        result.setdefault("executedQty", 0.0)
        result.setdefault("avgPrice", 0.0)
        return result

    def place_futures_market_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False, timeout_sec: float = 10.0) -> dict[str, Any]:
        """Sprint 23: dat lenh MARKET tren USDS-M Futures (ho tro reduceOnly khi dong vi the)."""
        req = self.build_order_request(VENUE_FUTURES, symbol, str(side).upper(), float(quantity), order_type="MARKET", reduce_only=bool(reduce_only))
        result = self.submit_signed_order(VENUE_FUTURES, req, timeout_sec=timeout_sec)
        result.setdefault("executedQty", 0.0)
        result.setdefault("avgPrice", 0.0)
        return result

    def fetch_exchange_reconcile_snapshot(
        self,
        symbols: list[str],
        prices: dict[str, float] | None = None,
        mock_responses: dict | None = None,
        timeout_sec: float = 5.0,
        universe_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sprint 23: snapshot doi soat kem Self-Healing Spot Valuation Guard (backfill gia thieu)."""
        prices = dict(prices or {})
        snap = self._fetch_exchange_reconcile_snapshot_raw(
            symbols, prices, mock_responses, timeout_sec, universe_symbols
        )
        if not isinstance(snap, dict) or not snap.get("snapshot_healthy"):
            return snap

        # Sprint 23: Self-Healing Spot Valuation Guard.
        # Neu gia thieu/khong hop le cho symbol dang co so du Spot hoac vi the Perp,
        # tu dong backfill bang bookTicker (bid+ask)/2, fallback mark price cua Perp.
        spot_balances_raw: dict[str, float] = dict(snap.get("spot_balances_raw") or {})
        perp_positions_detail: list[dict[str, Any]] = list(snap.get("perp_positions_detail") or [])
        universe_bases = {
            str(s).split("/")[0].upper() for s in (snap.get("universe_symbols") or []) if "/" in str(s)
        }
        backfilled_price_symbols: list[str] = []
        mark_price_by_base: dict[str, float] = {}
        perp_qty_by_base: dict[str, float] = {}
        for _item in perp_positions_detail:
            _base = str(_item.get("symbol", "")).split("/")[0].upper()
            if not _base:
                continue
            perp_qty_by_base[_base] = perp_qty_by_base.get(_base, 0.0) + abs(float(_item.get("qty", 0.0) or 0.0))
            _mk = float(_item.get("mark_price", 0.0) or 0.0)
            if _mk > 0.0:
                mark_price_by_base[_base] = _mk

        tracked_bases: set[str] = set(universe_bases)
        for _raw in list(symbols or []):
            _canon = normalize_symbol(str(_raw))
            if "/" in _canon:
                tracked_bases.add(_canon.split("/")[0].upper())
        tracked_bases.update(perp_qty_by_base.keys())

        for _base in sorted(tracked_bases):
            _spot_qty = float(spot_balances_raw.get(_base, 0.0) or 0.0)
            _perp_qty = float(perp_qty_by_base.get(_base, 0.0) or 0.0)
            if abs(_spot_qty) <= 1e-12 and abs(_perp_qty) <= 1e-12:
                continue
            _canon_sym = f"{_base}/USDT:USDT"
            _px = _lookup_price(prices, _canon_sym, _base)
            if _px > 0.0:
                prices[_canon_sym] = _px
                continue
            _resolved = 0.0
            _ticker_fn = getattr(self, "fetch_book_ticker", None)
            if callable(_ticker_fn) and not self.dry_run:
                for _venue in (VENUE_SPOT, VENUE_FUTURES):
                    try:
                        _tk = _ticker_fn(f"{_base}USDT", venue=_venue)
                    except Exception:  # noqa: BLE001 - fallback xuong mark price
                        continue
                    if not isinstance(_tk, dict):
                        continue
                    _bid = _safe_float(_tk.get("bid"))
                    _ask = _safe_float(_tk.get("ask"))
                    if _bid > 0.0 and _ask > 0.0:
                        _resolved = (_bid + _ask) / 2.0
                    elif _safe_float(_tk.get("mid")) > 0.0:
                        _resolved = _safe_float(_tk.get("mid"))
                    if _resolved > 0.0:
                        break
            if _resolved <= 0.0:
                _resolved = float(mark_price_by_base.get(_base, 0.0) or 0.0)
            if _resolved > 0.0:
                prices[_canon_sym] = _resolved
                backfilled_price_symbols.append(_canon_sym)
            elif abs(_spot_qty) > 1e-8:
                raise SpotValuationPriceMissingError(
                    f"Missing valid market price to value non-zero Spot holding for {_canon_sym} (qty={_spot_qty})"
                )

        # Sprint 23: tinh lai spot_cash_usd SAU backfill de khong bao RECONCILE_EQUITY_MISMATCH gia tao.
        _recomputed_spot_cash_usd = 0.0
        for _asset, _free_val in spot_balances_raw.items():
            if _free_val <= 0.0:
                continue
            if _is_cash_asset(_asset):
                _recomputed_spot_cash_usd += _free_val
                continue
            _px_asset = _lookup_price(prices, normalize_symbol(_asset), _asset)
            if _px_asset > 0.0:
                _recomputed_spot_cash_usd += _free_val * _px_asset
        snap["spot_cash_usd"] = float(_recomputed_spot_cash_usd)
        snap["exchange_equity_usd"] = float(
            snap["spot_cash_usd"]
            + float(snap.get("perp_wallet_balance_usd", 0.0) or 0.0)
            + float(snap.get("perp_unrealized_pnl_usd", 0.0) or 0.0)
        )
        snap["backfilled_price_symbols"] = list(backfilled_price_symbols)
        return snap

    def _fetch_exchange_reconcile_snapshot_raw(
        self,
        symbols: list[str],
        prices: dict[str, float],
        mock_responses: dict | None = None,
        timeout_sec: float = 5.0,
        universe_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Lay snapshot so du spot va vi the perp de doi soat danh muc (fail-closed).

        Sprint 21: ``universe_symbols`` (mac dinh LIQUID_8_SYMBOLS) duoc quet **toan ro**
        thay vi chi cac ma truyen trong ``symbols``, dong thoi tra them
        ``untracked_spot_assets`` (tai san Spot ngoai danh muc, vd faucet testnet).
        """
        snapshot_source = "mock" if mock_responses is not None else "testnet_http"

        scan_symbols: list[str] = []
        for raw in list(universe_symbols or []) + list(symbols or []):
            canon = normalize_symbol(str(raw))
            if canon and canon not in scan_symbols:
                scan_symbols.append(canon)
        if not scan_symbols:
            try:
                from src.cross_exchange_carry import LIQUID_8_SYMBOLS

                scan_symbols = [normalize_symbol(str(s)) for s in LIQUID_8_SYMBOLS]
            except Exception:  # noqa: BLE001 - fallback tinh de khong vo luong chinh
                scan_symbols = [
                    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
                    "DOGE/USDT:USDT", "XRP/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
                ]
        universe_bases = {s.split("/")[0].upper() for s in scan_symbols if "/" in s}

        if mock_responses is None:
            if not self.has_credentials():
                return {
                    "snapshot_healthy": False,
                    "error": "Missing credentials for testnet snapshot fetch (fail-closed)",
                    "spot_balances": {},
                    "perp_positions": {},
                    "exchange_equity_usd": 0.0,
                }
            if self.dry_run:
                return {
                    "snapshot_healthy": False,
                    "error": "Dry run mode active, real network snapshot call blocked (fail-closed)",
                    "spot_balances": {},
                    "perp_positions": {},
                    "exchange_equity_usd": 0.0,
                }

            # Real network calls
            try:
                spot_req = self.build_signed_get_request(SPOT_ACCOUNT_PATH, futures=False)
                futures_req = self.build_signed_get_request(PERP_ACCOUNT_PATH, futures=True)
                risk_req = self.build_signed_get_request(PERP_POSITION_RISK_PATH, futures=True)

                def _do_get(req_dict: dict[str, Any]) -> Any:
                    return self._http_request(
                        "GET",
                        req_dict["url"],
                        req_dict.get("params", {}),
                        req_dict.get("headers", {}),
                        timeout_sec=timeout_sec,
                    )

                spot_account = _do_get(spot_req)
                try:
                    futures_account = _do_get(futures_req)
                    position_risk = _do_get(risk_req)
                except BinanceHttpError as exc:
                    if exc.code in FUTURES_INVALID_ERROR_CODES:
                        return {
                            "snapshot_healthy": False,
                            "error": FUTURES_KEY_INVALID_HINT,
                            "futures_key_status": FUTURES_KEY_INVALID_CODE,
                            "api_error_code": exc.code,
                            "spot_balances": {},
                            "perp_positions": {},
                            "exchange_equity_usd": 0.0,
                        }
                    raise
            except Exception as exc:
                return {
                    "snapshot_healthy": False,
                    "error": f"Network fetch failed: {exc}",
                    "spot_balances": {},
                    "perp_positions": {},
                    "exchange_equity_usd": 0.0,
                }
        else:
            if not isinstance(mock_responses, dict):
                return {
                    "snapshot_healthy": False,
                    "error": "mock_responses is not a dict",
                    "spot_balances": {},
                    "perp_positions": {},
                    "exchange_equity_usd": 0.0,
                }
            spot_account = mock_responses.get("spot_account")
            futures_account = mock_responses.get("futures_account")
            position_risk = mock_responses.get("position_risk")

        try:
            if spot_account is not None and not isinstance(spot_account, dict):
                raise TypeError("spot_account must be a dict")
            if futures_account is not None and not isinstance(futures_account, dict):
                raise TypeError("futures_account must be a dict")
            if position_risk is not None and not isinstance(position_risk, (list, dict)):
                raise TypeError("position_risk must be a list or dict")
            if spot_account is None and futures_account is None:
                raise ValueError("Both spot_account and futures_account are missing")

            spot_account = spot_account or {}
            futures_account = futures_account or {}

            # 1. spot_balances & spot_cash_usd
            raw_balances = spot_account.get("balances", [])
            if not isinstance(raw_balances, list):
                raise TypeError("spot balances must be a list")

            spot_balances_raw: dict[str, float] = {}
            for b in raw_balances:
                if isinstance(b, dict):
                    asset = str(b.get("asset", "")).strip().upper()
                    if asset:
                        try:
                            spot_balances_raw[asset] = float(b.get("free", 0.0) or 0.0)
                        except (ValueError, TypeError):
                            spot_balances_raw[asset] = 0.0

            spot_balances: dict[str, float] = {}
            for asset, free_val in spot_balances_raw.items():
                if _is_cash_asset(asset):
                    # Sprint 18: tien mat quote duoc tong hop rieng trong spot_cash_usd,
                    # KHONG dua vao spot_balances (tranh reconciler bao ORPHAN gia tao).
                    continue
                if free_val > 0.0:
                    spot_balances[asset] = free_val

            # Keep base assets present in the scanned universe even if free == 0
            for sym in scan_symbols:
                canon = normalize_symbol(str(sym))
                base = canon.split("/")[0] if "/" in canon else canon
                base = base.upper().strip()
                if base and base not in spot_balances:
                    spot_balances[base] = float(spot_balances_raw.get(base, 0.0))

            # Sprint 21: tai san Spot ngoai danh muc (vd faucet testnet) -> thong ke rieng
            untracked_spot_assets: dict[str, float] = {}
            for asset, free_val in spot_balances_raw.items():
                if free_val <= 0.0:
                    continue
                if _is_cash_asset(asset):
                    continue
                if asset.upper() in universe_bases:
                    continue
                untracked_spot_assets[asset] = float(free_val)

            pure_cash_usd = 0.0
            spot_cash_usd = 0.0
            for asset, free_val in spot_balances_raw.items():
                if free_val <= 0.0:
                    continue
                if _is_cash_asset(asset):
                    pure_cash_usd += free_val
                    spot_cash_usd += free_val
                else:
                    canon_sym = normalize_symbol(asset)
                    px = _lookup_price(prices, canon_sym, asset)
                    if px > 0.0:
                        spot_cash_usd += free_val * px

            # 2. perp futures figures
            wb_raw = futures_account.get("totalWalletBalance")
            if wb_raw is None or wb_raw == "":
                wb_raw = futures_account.get("totalMarginBalance", 0.0)
            perp_wallet_balance_usd = float(wb_raw or 0.0)

            upnl_raw = futures_account.get("totalUnrealizedProfit", 0.0)
            perp_unrealized_pnl_usd = float(upnl_raw or 0.0)

            mm_raw = futures_account.get("totalMaintMargin", 0.0)
            maint_margin_usd = float(mm_raw or 0.0)

            mb_raw = futures_account.get("totalMarginBalance", 0.0)
            margin_balance_usd = float(mb_raw or 0.0)

            exchange_equity_usd = float(spot_cash_usd + perp_wallet_balance_usd + perp_unrealized_pnl_usd)

            # 3. perp_positions & perp_positions_detail
            pos_candidates = position_risk if position_risk is not None else futures_account.get("positions", [])
            if isinstance(pos_candidates, dict):
                pos_candidates = pos_candidates.get("positions", list(pos_candidates.values()))
            if not isinstance(pos_candidates, list):
                raise TypeError("positions must be a list")

            perp_positions: dict[str, float] = {}
            perp_positions_detail: list[dict[str, Any]] = []

            for item in pos_candidates:
                if not isinstance(item, dict):
                    continue
                raw_sym = str(item.get("symbol", "")).strip()
                if not raw_sym:
                    continue
                canon_sym = normalize_symbol(raw_sym)
                base = canon_sym.split("/")[0] if "/" in canon_sym else canon_sym

                qty = float(item.get("positionAmt", item.get("qty", 0.0)) or 0.0)
                if abs(qty) == 0.0:
                    continue

                perp_positions[canon_sym] = qty

                mark_p = float(item.get("markPrice", item.get("mark_price", 0.0)) or 0.0)
                if mark_p <= 0.0:
                    mark_p = _lookup_price(prices, canon_sym, base)

                entry_p = float(item.get("entryPrice", item.get("entry_price", 0.0)) or 0.0)
                liq_p = float(item.get("liquidationPrice", item.get("liquidation_price", 0.0)) or 0.0)
                mm_pos = float(item.get("maintMargin", item.get("totalMaintMargin", item.get("maint_margin_usd", 0.0))) or 0.0)
                mb_pos = float(item.get("marginBalance", item.get("totalMarginBalance", item.get("margin_balance_usd", 0.0))) or 0.0)

                perp_positions_detail.append({
                    "symbol": canon_sym,
                    "qty": qty,
                    "mark_price": mark_p,
                    "entry_price": entry_p,
                    "liquidation_price": liq_p,
                    "maint_margin_usd": mm_pos,
                    "margin_balance_usd": mb_pos,
                })

            return {
                "snapshot_healthy": True,
                "snapshot_source": snapshot_source,
                "spot_balances": spot_balances,
                "spot_balances_raw": dict(spot_balances_raw),
                "spot_cash_usd": float(spot_cash_usd),
                "pure_cash_usd": float(pure_cash_usd),
                "perp_wallet_balance_usd": float(perp_wallet_balance_usd),
                "perp_unrealized_pnl_usd": float(perp_unrealized_pnl_usd),
                "maint_margin_usd": float(maint_margin_usd),
                "margin_balance_usd": float(margin_balance_usd),
                "exchange_equity_usd": float(exchange_equity_usd),
                "perp_positions": perp_positions,
                "perp_positions_detail": perp_positions_detail,
                "universe_symbols": list(scan_symbols),
                "untracked_spot_assets": dict(untracked_spot_assets),
            }
        except Exception as exc:
            return {
                "snapshot_healthy": False,
                "error": f"Snapshot parse error: {exc}",
                "spot_balances": {},
                "spot_balances_raw": {},
                "pure_cash_usd": 0.0,
                "perp_positions": {},
                "exchange_equity_usd": 0.0,
            }


def _parse_binance_error(raw: str) -> tuple[int | None, str]:
    """Trich (code, msg) tu body loi JSON cua Binance (neu co)."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("code") is not None:
            return int(obj.get("code")), str(obj.get("msg", ""))
    except Exception:
        pass
    return None, raw.strip()[:200]


def normalize_order_result(venue: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Chuan hoa response dat lenh Binance (Spot v3 / FAPI v1) ve 1 schema."""
    data = dict(raw or {})
    executed = _safe_float(data.get("executedQty"))
    quote_qty = _safe_float(data.get("cummulativeQuoteQty", data.get("cumQuote")))
    avg_price = _safe_float(data.get("avgPrice"))
    if avg_price <= 0.0 and executed > 0.0 and quote_qty > 0.0:
        avg_price = quote_qty / executed
    fills = data.get("fills") or []
    if avg_price <= 0.0 and isinstance(fills, list):
        notional = 0.0
        qty_sum = 0.0
        for f in fills:
            if not isinstance(f, dict):
                continue
            qf = _safe_float(f.get("qty"))
            pf = _safe_float(f.get("price"))
            notional += qf * pf
            qty_sum += qf
        if qty_sum > 0.0 and notional > 0.0:
            avg_price = notional / qty_sum
            if quote_qty <= 0.0:
                quote_qty = notional
    return {
        "venue": venue,
        "orderId": data.get("orderId"),
        "clientOrderId": data.get("clientOrderId"),
        "symbol": data.get("symbol"),
        "side": data.get("side"),
        "type": data.get("type"),
        "status": str(data.get("status", "UNKNOWN")).upper(),
        "executedQty": executed,
        "cummulativeQuoteQty": quote_qty,
        "avgPrice": avg_price,
        "transactTime": data.get("transactTime", data.get("updateTime")),
        "raw": data,
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _binance_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTCUSDT'."""
    base = symbol.split(":")[0].strip()
    return base.replace("/", "").replace("-", "").upper()

def _is_filled(result: Mapping[str, Any] | None) -> bool:
    """True CHI khi lenh da khop 100% (status == FILLED).

    PARTIALLY_FILLED KHONG duoc coi la xong: con khoi luong chua khop thi phai
    tiep tuc cho / huy / mua bu, neu khong delta-neutral se bi lech.
    """
    if not isinstance(result, Mapping):
        return False
    return str(result.get("status", "")).upper() == "FILLED"


def quantize_price(price: float, tick_size: float, side: str = "BUY") -> float:
    """Lam tron gia theo tick_size cua san (BUY -> xuong, SELL -> len).

    Dung de tranh loi -1013 PRICE_FILTER khi dat Post-Only tren Spot Testnet.
    Tinh bang Decimal de khong bi sai so IEEE-754 (vd 82872.54000000001) va
    hoat dong ca voi tick_size khong phai luy thua 10 (vd 0.25).
    """
    if tick_size is None or tick_size <= 0.0 or price <= 0.0:
        return float(price)
    from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

    d_tick = Decimal(str(tick_size))
    d_price = Decimal(str(price))
    steps = d_price / d_tick
    rounding = ROUND_FLOOR if str(side).upper() == "BUY" else ROUND_CEILING
    return float(steps.to_integral_value(rounding=rounding) * d_tick)


def compute_maker_limit_price(
    side: str,
    best_bid: float,
    best_ask: float,
    tick_size: float,
    improve_inside_spread: bool = True,
) -> dict[str, Any]:
    """Sprint 21: gia dat lenh Maker Post-Only (GTX) theo chien luoc Smart Inside-Spread.

    - Neu spread >= 2 tick va ``improve_inside_spread=True``: BUY dat tai
      ``best_bid + tick`` va SELL dat tai ``best_ask - tick``. Gia nay van nam
      nghiem ngat trong spread nen KHONG BAO GIO bi tu choi GTX, nhung duoc uu tien
      dung dau so lenh (top-of-queue) -> tang ty le khop Maker, giam fallback market.
    - Neu spread <= 1 tick: giu nguyen ``best_bid`` (BUY) / ``best_ask`` (SELL).

    Tra ve dict metadata; ``price <= 0`` nghia la khong the tinh gia (caller phai
    fallback sang MARKET).
    """
    side_u = str(side or "").strip().upper()
    bid = _safe_float(best_bid)
    ask = _safe_float(best_ask)
    tick = _safe_float(tick_size)

    info: dict[str, Any] = {
        "side": side_u,
        "best_bid": bid,
        "best_ask": ask,
        "tick_size": tick,
        "spread_ticks": 0,
        "improve_inside_spread": bool(improve_inside_spread),
        "improved": False,
        "mode": "FALLBACK_NO_BOOK",
        "price": 0.0,
    }
    if side_u not in ("BUY", "SELL"):
        info["mode"] = "INVALID_SIDE"
        return info

    # Sprint 21 (audit fix): chi coi la so lenh HOP LE khi bid < ask nghiem ngat.
    # - ask == bid (locked book) hoac ask < bid (crossed): moi gia deu co the khop
    #   ngay -> Post-Only GTX bi tu choi (-2010). Tra price=0 de caller dung MARKET.
    # - So lenh mot phia (bid hoac ask = 0): khong an toan cho GTX -> price=0.
    has_book = (tick > 0.0) and (bid > 0.0) and (ask > 0.0) and (ask > bid)
    if not has_book:
        info["mode"] = "FALLBACK_NO_BOOK"
        info["price"] = 0.0
        return info

    spread_ticks = int(round((ask - bid) / tick))
    info["spread_ticks"] = spread_ticks

    if improve_inside_spread and spread_ticks >= 2:
        if side_u == "BUY":
            candidate = quantize_price(bid + tick, tick, side="BUY")
            if candidate <= bid:
                candidate = quantize_price(bid + tick, tick, side="SELL")
        else:
            candidate = quantize_price(ask - tick, tick, side="SELL")
            if candidate >= ask:
                candidate = quantize_price(ask - tick, tick, side="BUY")
        strictly_inside = (candidate > bid and candidate < ask) if side_u == "BUY" else (
            candidate < ask and candidate > bid
        )
        if strictly_inside and candidate > 0.0:
            info["price"] = float(candidate)
            info["improved"] = True
            info["mode"] = "INSIDE_SPREAD"
            return info

    # spread <= 1 tick (hoac khong cai thien duoc): dat ngay tai mep so lenh.
    touch = bid if side_u == "BUY" else ask
    info["price"] = float(quantize_price(touch, tick, side=side_u))
    info["mode"] = "TOUCH_BEST"
    return info

