"""Sprint 16 & Sprint 18 - Adapter ky lenh Binance HMAC-SHA256 cho Testnet/Demo (fail-closed).

Nguyen tac an toan:
- KHONG hardcode khoa: chi doc tu tham so hoac bien moi truong
  BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET.
- dry_run=True mac dinh: chi dong goi payload + tao chu ky, KHONG gui lenh.
- Thieu khoa ma dry_run=False -> raise BinanceCredentialsMissingError (fail-closed).

Chuan chu ky Binance: query string duoc sap xep theo key, urlencode, kem timestamp
va recvWindow; signature = HMAC_SHA256(secret, query_string).hexdigest().
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping
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
DEFAULT_RECV_WINDOW_MS: int = 5000
ENV_API_KEY: str = "BINANCE_TESTNET_API_KEY"
ENV_API_SECRET: str = "BINANCE_TESTNET_API_SECRET"


class BinanceCredentialsMissingError(RuntimeError):
    """Thieu API key/secret khi can ky lenh that (fail-closed)."""


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
    """Client ky lenh testnet (Spot + USDS-M Futures), mac dinh dry-run."""

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
        """Them timestamp + recvWindow roi tra payload da ky."""
        self.require_credentials()
        payload: dict[str, Any] = dict(params)
        payload["timestamp"] = int(
            timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        )
        payload["recvWindow"] = int(self.recv_window)
        payload["signature"] = sign_binance_payload(payload, self.api_secret or "")
        return payload

    def build_signed_get_request(
        self,
        endpoint: str,
        params: dict | None = None,
        futures: bool = False,
        timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        """Dong goi request GET da ky HMAC-SHA256."""
        self.require_credentials()
        base_url = self.perp_base_url if futures else self.spot_base_url
        clean_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        signed_params = self.sign_params(params or {}, timestamp_ms=timestamp_ms)
        return {
            "method": "GET",
            "url": f"{base_url}{clean_endpoint}",
            "headers": self._headers(),
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
    ) -> dict[str, Any]:
        """Dung cap lenh Spot BUY + Perp SELL (delta-neutral) da ky day du."""
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

    # -- snapshot fetching -----------------------------------------------
    def fetch_exchange_reconcile_snapshot(
        self,
        symbols: list[str],
        prices: dict[str, float],
        mock_responses: dict | None = None,
        timeout_sec: float = 5.0,
    ) -> dict[str, Any]:
        """Lay snapshot so du spot va vi the perp de doi soat danh muc (fail-closed)."""
        snapshot_source = "mock" if mock_responses is not None else "testnet_http"

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
                    url = req_dict["url"]
                    params = req_dict.get("params", {})
                    qs = build_query_string(params)
                    full_url = f"{url}?{qs}" if qs else url
                    headers = req_dict.get("headers", {})
                    req = urllib.request.Request(full_url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        return json.loads(resp.read().decode("utf-8"))

                spot_account = _do_get(spot_req)
                futures_account = _do_get(futures_req)
                position_risk = _do_get(risk_req)
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

            # Keep base assets present in symbols even if free == 0
            for sym in (symbols or []):
                canon = normalize_symbol(str(sym))
                base = canon.split("/")[0] if "/" in canon else canon
                base = base.upper().strip()
                if base and base not in spot_balances:
                    spot_balances[base] = float(spot_balances_raw.get(base, 0.0))

            spot_cash_usd = 0.0
            for asset, free_val in spot_balances_raw.items():
                if free_val <= 0.0:
                    continue
                if _is_cash_asset(asset):
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
                "spot_cash_usd": float(spot_cash_usd),
                "perp_wallet_balance_usd": float(perp_wallet_balance_usd),
                "perp_unrealized_pnl_usd": float(perp_unrealized_pnl_usd),
                "maint_margin_usd": float(maint_margin_usd),
                "margin_balance_usd": float(margin_balance_usd),
                "exchange_equity_usd": float(exchange_equity_usd),
                "perp_positions": perp_positions,
                "perp_positions_detail": perp_positions_detail,
            }
        except Exception as exc:
            return {
                "snapshot_healthy": False,
                "error": f"Snapshot parse error: {exc}",
                "spot_balances": {},
                "perp_positions": {},
                "exchange_equity_usd": 0.0,
            }


def _binance_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTCUSDT'."""
    base = symbol.split(":")[0].strip()
    return base.replace("/", "").replace("-", "").upper()
