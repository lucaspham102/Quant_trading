"""Sprint 16 - Adapter ký lệnh Binance HMAC-SHA256 cho Testnet/Demo (fail-closed).

Nguyên tắc an toàn:
- KHÔNG hardcode khóa: chỉ đọc từ tham số hoặc biến môi trường
  ``BINANCE_TESTNET_API_KEY`` / ``BINANCE_TESTNET_API_SECRET``.
- ``dry_run=True`` mặc định: chỉ đóng gói payload + tạo chữ ký, KHÔNG gửi lệnh.
- Thiếu khóa mà ``dry_run=False`` -> raise ``BinanceCredentialsMissingError`` (fail-closed).

Chuẩn chữ ký Binance: query string được sắp xếp theo key, urlencode, kèm ``timestamp``
và ``recvWindow``; ``signature = HMAC_SHA256(secret, query_string).hexdigest()``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Mapping
from urllib.parse import urlencode

from src.cross_exchange_carry import EXCHANGE_LOT_SPECS, DEFAULT_MIN_NOTIONAL

SPOT_TESTNET_BASE_URL: str = "https://testnet.binance.vision"
PERP_TESTNET_BASE_URL: str = "https://testnet.binancefuture.com"
SPOT_ORDER_PATH: str = "/api/v3/order"
PERP_ORDER_PATH: str = "/fapi/v1/order"
DEFAULT_RECV_WINDOW_MS: int = 5000
ENV_API_KEY: str = "BINANCE_TESTNET_API_KEY"
ENV_API_SECRET: str = "BINANCE_TESTNET_API_SECRET"


class BinanceCredentialsMissingError(RuntimeError):
    """Thiếu API key/secret khi cần ký lệnh thật (fail-closed)."""


def _format_value(value: Any) -> str:
    """Chuẩn hóa số thành chuỗi thập phân ổn định (tránh ký hiệu mũ e-05)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e16:
            return str(int(value))
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def build_query_string(params: Mapping[str, Any]) -> str:
    """Sắp xếp key tăng dần + urlencode để chuỗi ký và body trùng khớp tuyệt đối."""
    ordered = sorted((str(k), _format_value(v)) for k, v in params.items())
    return urlencode(ordered)


def sign_binance_payload(params: Mapping[str, Any], api_secret: str) -> str:
    """Trả về HMAC-SHA256 hex của (query string của ``params``) với ``api_secret``."""
    if not api_secret:
        raise BinanceCredentialsMissingError("api_secret rong, khong the ky payload")
    query = build_query_string(params)
    return hmac.new(
        api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _check_lot(symbol: str, qty: float, price: float | None = None) -> dict[str, float]:
    """Kiểm tra qty khớp step_size / min_notional của cặp trước khi dựng lệnh."""
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


class BinanceSignedTestnetClient:
    """Client ký lệnh testnet (Spot + USDⓈ-M Futures), mặc định dry-run."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        dry_run: bool = True,
        recv_window: int = DEFAULT_RECV_WINDOW_MS,
        spot_base_url: str = SPOT_TESTNET_BASE_URL,
        perp_base_url: str = PERP_TESTNET_BASE_URL,
    ) -> None:
        self.api_key: str | None = api_key or os.environ.get(ENV_API_KEY)
        self.api_secret: str | None = api_secret or os.environ.get(ENV_API_SECRET)
        self.dry_run: bool = bool(dry_run)
        self.recv_window: int = int(recv_window)
        self.spot_base_url: str = spot_base_url.rstrip("/")
        self.perp_base_url: str = perp_base_url.rstrip("/")
        if not self.dry_run and not self.has_credentials():
            raise BinanceCredentialsMissingError(
                "Thieu BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET: "
                "bat buoc dung --dry-run-sign hoac cau hinh bien moi truong (fail-closed)."
            )

    # -- credentials -----------------------------------------------------
    def has_credentials(self) -> bool:
        return bool(self.api_key) and bool(self.api_secret)

    def require_credentials(self) -> None:
        if not self.has_credentials():
            raise BinanceCredentialsMissingError(
                "Thieu BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET (fail-closed)."
            )

    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key or ""}

    # -- signing ---------------------------------------------------------
    def sign_params(
        self, params: Mapping[str, Any], timestamp_ms: int | None = None
    ) -> dict[str, Any]:
        """Thêm ``timestamp`` + ``recvWindow`` rồi trả payload đã ký."""
        self.require_credentials()
        payload: dict[str, Any] = dict(params)
        payload["timestamp"] = int(
            timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        )
        payload["recvWindow"] = int(self.recv_window)
        payload["signature"] = sign_binance_payload(payload, self.api_secret or "")
        return payload

    # -- order building --------------------------------------------------
    def build_delta_neutral_order_pair(
        self,
        symbol: str,
        spot_qty: float,
        perp_qty: float,
        spot_price: float | None = None,
        post_only_spot: bool = True,
        timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        """Dựng cặp lệnh Spot BUY + Perp SELL (delta-neutral) đã ký đầy đủ."""
        lot_spot = _check_lot(symbol, float(spot_qty), spot_price)
        lot_perp = _check_lot(symbol, float(perp_qty))

        spot_params: dict[str, Any] = {
            "symbol": _binance_symbol(symbol),
            "side": "BUY",
            "quantity": float(spot_qty),
        }
        if post_only_spot:
            if spot_price is None or spot_price <= 0.0:
                raise ValueError("post_only_spot=True yeu cau spot_price > 0")
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
            "side": "SELL",
            "type": "MARKET",
            "quantity": float(perp_qty),
        }

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
                "url": f"{self.spot_base_url}{SPOT_ORDER_PATH}",
                "headers": self._headers(),
                "params": self.sign_params(spot_params, timestamp_ms=timestamp_ms),
            },
            "perp": {
                "method": "POST",
                "url": f"{self.perp_base_url}{PERP_ORDER_PATH}",
                "headers": self._headers(),
                "params": self.sign_params(perp_params, timestamp_ms=timestamp_ms),
            },
        }


def _binance_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTCUSDT'."""
    base = symbol.split(":")[0].strip()
    return base.replace("/", "").replace("-", "").upper()
