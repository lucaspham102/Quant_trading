"""Binance USDT-M public feed adapter for shadow-live mode.

Pulls read-only snapshots (bookTicker + premiumIndex) with fail-closed semantics.
Never places orders.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

BINANCE_FAPI_BASE = "https://fapi.binance.com"
BOOK_TICKER_PATH = "/fapi/v1/ticker/bookTicker"
PREMIUM_INDEX_PATH = "/fapi/v1/premiumIndex"


def to_binance_symbol(symbol: str) -> str:
    """Normalize ccxt or raw symbol string into Binance futures symbol format.

    Examples:
        'BTC/USDT:USDT' -> 'BTCUSDT'
        'BTC/USDT' -> 'BTCUSDT'
        'BTCUSDT' -> 'BTCUSDT'
    """
    base = symbol.split(":")[0].strip()
    return base.replace("/", "").replace("-", "").upper()


def fetch_binance_shadow_snapshot(
    symbols: Sequence[str],
    timeout_sec: float = 5.0,
    fixture_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch read-only bookTicker and premiumIndex snapshot for symbols.

    Fail-closed behavior:
    - Empty symbols raises ValueError.
    - Fixture mode makes zero network calls.
    - Network/parsing failures return healthy=False and empty symbols dict.
    - Crossed or invalid books are omitted per-symbol with error recorded.
    """
    if not symbols:
        raise ValueError("symbols must be non-empty")

    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    errors: list[str] = []
    symbols_out: dict[str, dict[str, float]] = {}

    if fixture_data is not None:
        fully_valid: list[str] = []
        for sym in symbols:
            if sym not in fixture_data or fixture_data[sym] is None:
                errors.append(f"MISSING_FIXTURE_DATA:{sym}")
                continue

            entry = fixture_data[sym]
            if not isinstance(entry, Mapping):
                errors.append(f"SPREAD_INVALID_BOOK:{sym}")
                continue

            if "bid" not in entry or "ask" not in entry:
                errors.append(f"SPREAD_INVALID_BOOK:{sym}")
                continue

            try:
                bid = float(entry["bid"])
                ask = float(entry["ask"])
            except (ValueError, TypeError):
                errors.append(f"SPREAD_INVALID_BOOK:{sym}")
                continue

            if bid <= 0.0 or ask <= 0.0:
                errors.append(f"SPREAD_INVALID_BOOK:{sym}")
                continue

            if ask < bid:
                errors.append(f"SPREAD_CROSSED_BOOK:{sym}")
                continue

            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / mid) * 10000.0 if mid > 0.0 else 0.0

            try:
                mark_price = float(entry.get("mark_price", 0.0))
            except (ValueError, TypeError):
                mark_price = 0.0

            try:
                funding_rate = float(entry.get("funding_rate", 0.0))
            except (ValueError, TypeError):
                funding_rate = 0.0

            funding_rate_bps = funding_rate * 10000.0

            symbols_out[sym] = {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_bps": spread_bps,
                "mark_price": mark_price,
                "funding_rate_bps": funding_rate_bps,
            }
            fully_valid.append(sym)

        healthy = len(fully_valid) > 0
        return {
            "source": "fixture",
            "healthy": healthy,
            "fetched_at_utc": fetched_at_utc,
            "symbols": symbols_out,
            "errors": errors,
        }

    # Network mode
    safe_timeout = float(timeout_sec) if timeout_sec and float(timeout_sec) > 0 else 5.0
    book_url = f"{BINANCE_FAPI_BASE}{BOOK_TICKER_PATH}"
    prem_url = f"{BINANCE_FAPI_BASE}{PREMIUM_INDEX_PATH}"

    book_map: dict[str, dict[str, Any]] = {}
    prem_map: dict[str, dict[str, Any]] = {}

    try:
        with urllib.request.urlopen(book_url, timeout=safe_timeout) as resp:
            raw_book = resp.read()
        book_list = json.loads(raw_book)
        if not isinstance(book_list, list):
            raise ValueError("bookTicker response is not a list")

        with urllib.request.urlopen(prem_url, timeout=safe_timeout) as resp:
            raw_prem = resp.read()
        prem_list = json.loads(raw_prem)
        if not isinstance(prem_list, list):
            raise ValueError("premiumIndex response is not a list")

        for item in book_list:
            if isinstance(item, dict) and "symbol" in item:
                book_map[str(item["symbol"])] = item

        for item in prem_list:
            if isinstance(item, dict) and "symbol" in item:
                prem_map[str(item["symbol"])] = item

    except Exception as exc:
        errors.append(f"NETWORK_FATAL_ERROR:{type(exc).__name__}:{exc}")
        return {
            "source": "network",
            "healthy": False,
            "fetched_at_utc": fetched_at_utc,
            "symbols": {},
            "errors": errors,
        }

    fully_valid_symbols: list[str] = []

    for sym in symbols:
        b_sym = to_binance_symbol(sym)
        if b_sym not in book_map:
            errors.append(f"MISSING_BOOK_TICKER:{sym}")
            continue

        book_item = book_map[b_sym]
        try:
            bid = float(book_item["bidPrice"])
            ask = float(book_item["askPrice"])
        except (KeyError, ValueError, TypeError):
            errors.append(f"SPREAD_INVALID_BOOK:{sym}")
            continue

        if bid <= 0.0 or ask <= 0.0:
            errors.append(f"SPREAD_INVALID_BOOK:{sym}")
            continue

        if ask < bid:
            errors.append(f"SPREAD_CROSSED_BOOK:{sym}")
            continue

        mid = (bid + ask) / 2.0
        spread_bps = ((ask - bid) / mid) * 10000.0 if mid > 0.0 else 0.0

        has_premium_leg = False
        mark_price = 0.0
        funding_rate_bps = 0.0

        if b_sym in prem_map:
            prem_item = prem_map[b_sym]
            try:
                mark_price = float(prem_item.get("markPrice", 0.0))
                funding_rate = float(prem_item.get("lastFundingRate", 0.0))
                funding_rate_bps = funding_rate * 10000.0
                has_premium_leg = True
            except (ValueError, TypeError):
                errors.append(f"INVALID_PREMIUM_INDEX:{sym}")
                mark_price = 0.0
                funding_rate_bps = 0.0
        else:
            errors.append(f"MISSING_PREMIUM_INDEX:{sym}")

        symbols_out[sym] = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_bps": spread_bps,
            "mark_price": mark_price,
            "funding_rate_bps": funding_rate_bps,
        }

        if has_premium_leg:
            fully_valid_symbols.append(sym)

    healthy = len(fully_valid_symbols) > 0

    return {
        "source": "network",
        "healthy": healthy,
        "fetched_at_utc": fetched_at_utc,
        "symbols": symbols_out,
        "errors": errors,
    }
