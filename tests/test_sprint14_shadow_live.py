"""Unit tests for Binance shadow-live public feed adapter."""

from __future__ import annotations

import json
import urllib.error
import pytest

import src.binance_public_feed
from src.binance_public_feed import (
    fetch_binance_shadow_snapshot,
    to_binance_symbol,
)


class DummyResponse:
    """Mock context manager for urllib.request.urlopen responses."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> DummyResponse:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def test_to_binance_symbol_mapping() -> None:
    """1. to_binance_symbol mapping for ccxt and raw format variations."""
    assert to_binance_symbol("BTC/USDT:USDT") == "BTCUSDT"
    assert to_binance_symbol("BTC/USDT") == "BTCUSDT"
    assert to_binance_symbol("BTCUSDT") == "BTCUSDT"
    assert to_binance_symbol("btc/usdt:usdt") == "BTCUSDT"
    assert to_binance_symbol("ETH-USDT:USDT") == "ETHUSDT"


def test_empty_symbols_raises() -> None:
    """2. empty symbols -> pytest.raises(ValueError)."""
    with pytest.raises(ValueError, match="symbols must be non-empty"):
        fetch_binance_shadow_snapshot([])

    with pytest.raises(ValueError, match="symbols must be non-empty"):
        fetch_binance_shadow_snapshot("")  # type: ignore[arg-type]


def test_fixture_happy_path() -> None:
    """3. fixture path: valid fixture -> source=='fixture', healthy True, exact math."""
    fixture = {
        "BTC/USDT:USDT": {
            "bid": 100.0,
            "ask": 102.0,
            "mark_price": 101.5,
            "funding_rate": 0.0001,
        }
    }
    result = fetch_binance_shadow_snapshot(["BTC/USDT:USDT"], fixture_data=fixture)

    assert result["source"] == "fixture"
    assert result["healthy"] is True
    assert result["errors"] == []
    assert "BTC/USDT:USDT" in result["symbols"]

    sym_info = result["symbols"]["BTC/USDT:USDT"]
    assert sym_info["bid"] == 100.0
    assert sym_info["ask"] == 102.0
    assert sym_info["mid"] == 101.0
    assert pytest.approx(sym_info["spread_bps"], rel=1e-6) == 198.01980198
    assert sym_info["mark_price"] == 101.5
    assert pytest.approx(sym_info["funding_rate_bps"], rel=1e-6) == 1.0


def test_fixture_makes_zero_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """4. fixture path makes zero network calls."""
    def fail_urlopen(*args, **kwargs):
        raise AssertionError("Network call attempted during fixture mode")

    monkeypatch.setattr(src.binance_public_feed.urllib.request, "urlopen", fail_urlopen)

    fixture = {
        "BTC/USDT:USDT": {
            "bid": 50000.0,
            "ask": 50010.0,
            "mark_price": 50005.0,
            "funding_rate": 0.0001,
        }
    }
    result = fetch_binance_shadow_snapshot(["BTC/USDT:USDT"], fixture_data=fixture)
    assert result["source"] == "fixture"
    assert result["healthy"] is True


def test_fixture_crossed_book() -> None:
    """5. fixture with crossed book -> symbol omitted, SPREAD_CROSSED_BOOK, healthy False."""
    fixture = {
        "BTC/USDT:USDT": {
            "bid": 50010.0,
            "ask": 50000.0,
            "mark_price": 50005.0,
            "funding_rate": 0.0001,
        }
    }
    result = fetch_binance_shadow_snapshot(["BTC/USDT:USDT"], fixture_data=fixture)
    assert "BTC/USDT:USDT" not in result["symbols"]
    assert any("SPREAD_CROSSED_BOOK" in e for e in result["errors"])
    assert result["healthy"] is False


def test_fixture_missing_symbol() -> None:
    """6. fixture with missing symbol -> symbol omitted, error recorded, healthy False."""
    result = fetch_binance_shadow_snapshot(["BTC/USDT:USDT"], fixture_data={})
    assert "BTC/USDT:USDT" not in result["symbols"]
    assert len(result["errors"]) > 0
    assert any("MISSING_FIXTURE_DATA" in e for e in result["errors"])
    assert result["healthy"] is False


def test_network_path_happy(monkeypatch: pytest.MonkeyPatch) -> None:
    """7. network path happy: fake payloads -> healthy True, timeout captured."""
    recorded_calls: list[dict[str, object]] = []

    book_bytes = json.dumps([
        {"symbol": "BTCUSDT", "bidPrice": "50000.00", "askPrice": "50010.00"}
    ]).encode("utf-8")
    prem_bytes = json.dumps([
        {"symbol": "BTCUSDT", "markPrice": "50005.00", "lastFundingRate": "0.000200"}
    ]).encode("utf-8")

    def mock_urlopen(url: str, timeout: float = 5.0):
        recorded_calls.append({"url": url, "timeout": timeout})
        if "bookTicker" in url:
            return DummyResponse(book_bytes)
        if "premiumIndex" in url:
            return DummyResponse(prem_bytes)
        raise ValueError(f"Unexpected url {url}")

    monkeypatch.setattr(src.binance_public_feed.urllib.request, "urlopen", mock_urlopen)

    result = fetch_binance_shadow_snapshot(["BTC/USDT:USDT"], timeout_sec=7.5)

    assert result["source"] == "network"
    assert result["healthy"] is True
    assert len(recorded_calls) == 2
    assert any("bookTicker" in str(c["url"]) for c in recorded_calls)
    assert any("premiumIndex" in str(c["url"]) for c in recorded_calls)
    for call in recorded_calls:
        assert call["timeout"] == 7.5

    data = result["symbols"]["BTC/USDT:USDT"]
    assert data["bid"] == 50000.00
    assert data["ask"] == 50010.00
    assert data["mid"] == 50005.00
    assert data["mark_price"] == 50005.00
    assert pytest.approx(data["funding_rate_bps"], rel=1e-6) == 2.0


def test_network_path_failure_urlerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """8. network path failure: urlopen raises URLError -> healthy False, symbols empty."""
    def fail_urlopen(url: str, timeout: float = 5.0):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(src.binance_public_feed.urllib.request, "urlopen", fail_urlopen)

    result = fetch_binance_shadow_snapshot(["BTC/USDT:USDT"])
    assert result["source"] == "network"
    assert result["healthy"] is False
    assert result["symbols"] == {}
    assert len(result["errors"]) > 0
    assert any("URLError" in e for e in result["errors"])


def test_negative_zero_bid_rejected_in_fixture() -> None:
    """9. negative/zero bid rejected in fixture path."""
    zero_bid_fixture = {
        "BTC/USDT:USDT": {"bid": 0.0, "ask": 50000.0, "mark_price": 50000.0}
    }
    res_zero = fetch_binance_shadow_snapshot(["BTC/USDT:USDT"], fixture_data=zero_bid_fixture)
    assert "BTC/USDT:USDT" not in res_zero["symbols"]
    assert any("SPREAD_INVALID_BOOK" in e for e in res_zero["errors"])
    assert res_zero["healthy"] is False

    neg_bid_fixture = {
        "ETH/USDT:USDT": {"bid": -10.0, "ask": 3000.0, "mark_price": 3000.0}
    }
    res_neg = fetch_binance_shadow_snapshot(["ETH/USDT:USDT"], fixture_data=neg_bid_fixture)
    assert "ETH/USDT:USDT" not in res_neg["symbols"]
    assert any("SPREAD_INVALID_BOOK" in e for e in res_neg["errors"])
    assert res_neg["healthy"] is False


def test_symbol_in_bookticker_absent_from_premiumindex(monkeypatch: pytest.MonkeyPatch) -> None:
    """10. symbol in bookTicker but absent from premiumIndex -> mark_price 0.0 and error recorded."""
    book_bytes = json.dumps([
        {"symbol": "BTCUSDT", "bidPrice": "50000.0", "askPrice": "50010.0"},
        {"symbol": "ETHUSDT", "bidPrice": "3000.0", "askPrice": "3002.0"},
    ]).encode("utf-8")
    prem_bytes = json.dumps([
        {"symbol": "BTCUSDT", "markPrice": "50005.0", "lastFundingRate": "0.0001"}
    ]).encode("utf-8")

    def mock_urlopen(url: str, timeout: float = 5.0):
        if "bookTicker" in url:
            return DummyResponse(book_bytes)
        return DummyResponse(prem_bytes)

    monkeypatch.setattr(src.binance_public_feed.urllib.request, "urlopen", mock_urlopen)

    result = fetch_binance_shadow_snapshot(["BTC/USDT:USDT", "ETH/USDT:USDT"])

    assert "ETH/USDT:USDT" in result["symbols"]
    eth_data = result["symbols"]["ETH/USDT:USDT"]
    assert eth_data["bid"] == 3000.0
    assert eth_data["ask"] == 3002.0
    assert eth_data["mark_price"] == 0.0
    assert eth_data["funding_rate_bps"] == 0.0
    assert any("MISSING_PREMIUM_INDEX" in e for e in result["errors"])

    # BTC is fully valid with both legs, so healthy stays True
    assert result["healthy"] is True
    assert result["symbols"]["BTC/USDT:USDT"]["mark_price"] == 50005.0
