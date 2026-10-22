"""Sprint 21 - Test suite cho Smart Inside-Spread maker pricing + nap env testnet.

Bao phu:
1. ``compute_maker_limit_price``: BUY/SELL dat nghiem ngat trong spread, spread 1
   tick thi touch best, ``improve_inside_spread=False`` thi touch best, book rong
   tra price 0.0 (caller fallback market), side la -> INVALID_SIDE.
2. Bat bien gia inside luon strictly inside ``(best_bid, best_ask)`` voi nhieu
   tick_size khac nhau (ke ca 0.25).
3. ``prune_legacy_fallback_env``: don bien legacy sai do dai khi khoa chuyen biet
   da dat chuan 64/64, giu bien legacy dung 64 ky tu.
4. ``validate_credential_format``: metadata ``legacy_fallback_ignored``.
5. ``load_dotenv_testnet``: nap .env.testnet gia lap, scrub bien legacy nhung van
   giu khoa Spot/Futures hop le.

Luu y: toan bo khoa trong file nay la chuoi GIA, khong bao gio la khoa that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.binance_signed_client import (
    ENV_API_KEY,
    ENV_API_SECRET,
    ENV_FUTURES_API_KEY,
    ENV_FUTURES_API_SECRET,
    ENV_SPOT_API_KEY,
    ENV_SPOT_API_SECRET,
    compute_maker_limit_price,
    load_dotenv_testnet,
    prune_legacy_fallback_env,
    validate_credential_format,
)

FAKE_SPOT_KEY = "S" * 64
FAKE_SPOT_SECRET = "s" * 64
FAKE_FUTURES_KEY = "F" * 64
FAKE_FUTURES_SECRET = "f" * 64


# ---------------------------------------------------------------------------
# 1. compute_maker_limit_price
# ---------------------------------------------------------------------------


def test_buy_inside_spread_one_tick_above_bid() -> None:
    info = compute_maker_limit_price("BUY", best_bid=100.0, best_ask=100.10, tick_size=0.01)
    assert info["price"] == pytest.approx(100.01)
    assert info["improved"] is True
    assert info["mode"] == "INSIDE_SPREAD"
    assert info["spread_ticks"] == 10


def test_sell_inside_spread_one_tick_below_ask() -> None:
    info = compute_maker_limit_price("SELL", best_bid=100.0, best_ask=100.10, tick_size=0.01)
    assert info["price"] == pytest.approx(100.09)
    assert info["improved"] is True
    assert info["mode"] == "INSIDE_SPREAD"
    assert info["spread_ticks"] == 10


def test_one_tick_spread_touches_best_price() -> None:
    buy = compute_maker_limit_price("BUY", best_bid=100.00, best_ask=100.01, tick_size=0.01)
    sell = compute_maker_limit_price("SELL", best_bid=100.00, best_ask=100.01, tick_size=0.01)
    assert buy["price"] == pytest.approx(100.00)
    assert buy["improved"] is False
    assert buy["mode"] == "TOUCH_BEST"
    assert sell["price"] == pytest.approx(100.01)
    assert sell["improved"] is False
    assert sell["mode"] == "TOUCH_BEST"


def test_improve_inside_spread_disabled_touches_best() -> None:
    buy = compute_maker_limit_price(
        "BUY", best_bid=100.0, best_ask=100.10, tick_size=0.01, improve_inside_spread=False
    )
    sell = compute_maker_limit_price(
        "SELL", best_bid=100.0, best_ask=100.10, tick_size=0.01, improve_inside_spread=False
    )
    assert buy["price"] == pytest.approx(100.0)
    assert buy["improved"] is False
    assert sell["price"] == pytest.approx(100.10)
    assert sell["improved"] is False


def test_empty_or_invalid_book_returns_zero_price_without_raising() -> None:
    """price == 0.0 bao hieu caller phai fallback sang MARKET."""
    info = compute_maker_limit_price("BUY", best_bid=0.0, best_ask=0.0, tick_size=0.01)
    assert info["price"] == 0.0
    assert info["mode"] == "FALLBACK_NO_BOOK"
    assert info["improved"] is False

    crossed = compute_maker_limit_price("SELL", best_bid=100.10, best_ask=100.0, tick_size=0.01)
    # So lenh crossed/locked: moi gia deu co the khop ngay -> KHONG gui GTX, tra 0.
    assert crossed["price"] == 0.0
    assert crossed["mode"] == "FALLBACK_NO_BOOK"

    locked = compute_maker_limit_price("BUY", best_bid=100.0, best_ask=100.0, tick_size=0.01)
    assert locked["price"] == 0.0
    assert locked["mode"] == "FALLBACK_NO_BOOK"


def test_invalid_side_returns_invalid_side_mode() -> None:
    info = compute_maker_limit_price("HOLD", best_bid=100.0, best_ask=100.10, tick_size=0.01)
    assert info["mode"] == "INVALID_SIDE"
    assert info["price"] == 0.0
    assert info["improved"] is False


@pytest.mark.parametrize(
    ("best_bid", "best_ask", "tick_size"),
    [
        (100.0, 100.10, 0.01),
        (25000.0, 25002.0, 0.5),
        (100.0, 100.75, 0.25),
        (1.0, 1.05, 0.01),
    ],
)
def test_inside_price_is_strictly_within_spread(
    best_bid: float, best_ask: float, tick_size: float
) -> None:
    buy = compute_maker_limit_price("BUY", best_bid, best_ask, tick_size)
    sell = compute_maker_limit_price("SELL", best_bid, best_ask, tick_size)
    assert buy["improved"] is True
    assert sell["improved"] is True
    assert best_bid < buy["price"] < best_ask
    assert best_bid < sell["price"] < best_ask


# ---------------------------------------------------------------------------
# 2. prune_legacy_fallback_env
# ---------------------------------------------------------------------------


def test_prune_removes_invalid_legacy_env_when_specific_keys_valid() -> None:
    env = {
        ENV_SPOT_API_KEY: FAKE_SPOT_KEY,
        ENV_SPOT_API_SECRET: FAKE_SPOT_SECRET,
        ENV_API_KEY: "short-legacy-key",
        ENV_API_SECRET: "short-legacy-secret",
    }
    result = prune_legacy_fallback_env(env)
    assert result["legacy_fallback_ignored"] is True
    assert result["spot_specific_valid"] is True
    assert ENV_API_KEY in result["ignored_legacy_keys"]
    assert ENV_API_SECRET in result["ignored_legacy_keys"]
    assert ENV_API_KEY not in env
    assert ENV_API_SECRET not in env
    # Khoa chuyen biet van con nguyen.
    assert env[ENV_SPOT_API_KEY] == FAKE_SPOT_KEY


def test_prune_keeps_legacy_env_when_no_valid_specific_pair() -> None:
    env = {
        ENV_SPOT_API_KEY: "too-short",
        ENV_SPOT_API_SECRET: "too-short",
        ENV_API_KEY: "short-legacy-key",
    }
    result = prune_legacy_fallback_env(env)
    assert result["legacy_fallback_ignored"] is False
    assert result["ignored_legacy_keys"] == []
    assert env[ENV_API_KEY] == "short-legacy-key"


def test_prune_keeps_legacy_key_with_exact_64_chars() -> None:
    env = {
        ENV_SPOT_API_KEY: FAKE_SPOT_KEY,
        ENV_SPOT_API_SECRET: FAKE_SPOT_SECRET,
        ENV_API_KEY: "L" * 64,
    }
    result = prune_legacy_fallback_env(env)
    assert ENV_API_KEY in env
    assert env[ENV_API_KEY] == "L" * 64
    assert result["ignored_legacy_keys"] == []
    assert result["legacy_fallback_ignored"] is False


# ---------------------------------------------------------------------------
# 3. validate_credential_format
# ---------------------------------------------------------------------------


def test_validate_flags_short_legacy_fallback_as_ignored() -> None:
    report = validate_credential_format(
        FAKE_SPOT_KEY,
        FAKE_SPOT_SECRET,
        "spot",
        legacy_key="short-legacy-key",
        legacy_secret="short-legacy-secret",
    )
    assert report["ok"] is True
    assert report["legacy_fallback_present"] is True
    assert report["legacy_fallback_ignored"] is True
    assert report["key_length"] == 64
    assert report["secret_length"] == 64
    # Khong bao gio lo gia tri khoa trong message/metadata.
    assert FAKE_SPOT_KEY not in report["message"]
    assert FAKE_SPOT_SECRET not in report["message"]


def test_validate_does_not_ignore_legacy_when_it_is_also_64_chars() -> None:
    report = validate_credential_format(
        FAKE_SPOT_KEY,
        FAKE_SPOT_SECRET,
        "spot",
        legacy_key="L" * 64,
        legacy_secret="M" * 64,
    )
    assert report["ok"] is True
    assert report["legacy_fallback_present"] is True
    assert report["legacy_fallback_ignored"] is False


def test_validate_invalid_specific_key_never_ignores_legacy() -> None:
    report = validate_credential_format(
        "short",
        "short",
        "futures",
        legacy_key="short-legacy-key",
        legacy_secret="short-legacy-secret",
    )
    assert report["ok"] is False
    assert report["legacy_fallback_ignored"] is False
    assert any("EXPECTED_64" in diag for diag in report["diagnostics"])


# ---------------------------------------------------------------------------
# 4. load_dotenv_testnet
# ---------------------------------------------------------------------------


def test_load_dotenv_testnet_keeps_specific_keys_and_prunes_legacy(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env.testnet"
    dotenv.write_text(
        "\n".join(
            [
                f"{ENV_SPOT_API_KEY}={FAKE_SPOT_KEY}",
                f"{ENV_SPOT_API_SECRET}={FAKE_SPOT_SECRET}",
                f"{ENV_FUTURES_API_KEY}={FAKE_FUTURES_KEY}",
                f"{ENV_FUTURES_API_SECRET}={FAKE_FUTURES_SECRET}",
                f"{ENV_API_KEY}=short-legacy-key",
                f"{ENV_API_SECRET}=short-legacy-secret",
            ]
        ),
        encoding="utf-8",
    )
    env: dict[str, str] = {}
    report: dict[str, object] = {}

    loaded = load_dotenv_testnet(
        dotenv_path=dotenv,
        extra_paths=(),
        override=True,
        environ=env,
        report=report,
    )

    assert report["legacy_fallback_ignored"] is True
    assert ENV_API_KEY not in env
    assert ENV_API_SECRET not in env
    assert env[ENV_SPOT_API_KEY] == FAKE_SPOT_KEY
    assert env[ENV_SPOT_API_SECRET] == FAKE_SPOT_SECRET
    assert env[ENV_FUTURES_API_KEY] == FAKE_FUTURES_KEY
    assert env[ENV_FUTURES_API_SECRET] == FAKE_FUTURES_SECRET
    assert loaded[ENV_SPOT_API_KEY] == FAKE_SPOT_KEY
    assert report["spot_specific_valid"] is True
    assert report["futures_specific_valid"] is True

# ---------------------------------------------------------------------------
# 6. place_maker_with_fallback (audit fix: GTX reject -> market, khong fake FILLED)
# ---------------------------------------------------------------------------


def _client():  # type: ignore[no-untyped-def]
    from src.binance_signed_client import BinanceSignedTestnetClient

    return BinanceSignedTestnetClient(
        spot_api_key=FAKE_SPOT_KEY,
        spot_api_secret=FAKE_SPOT_SECRET,
        futures_api_key=FAKE_FUTURES_KEY,
        futures_api_secret=FAKE_FUTURES_SECRET,
        dry_run=False,
        autoload_dotenv=False,
    )


def test_maker_spot_uses_inside_price_and_reports_filled() -> None:
    client = _client()
    captured: dict[str, float] = {}

    def _spot_leg(**kwargs):  # type: ignore[no-untyped-def]
        captured["price"] = kwargs["price"]
        return (
            {"status": "FILLED", "filled": True, "executedQty": kwargs["qty"], "orderId": 7},
            [],
        )

    client._execute_spot_leg = _spot_leg  # type: ignore[method-assign]
    res = client.place_maker_with_fallback(
        symbol="BTC/USDT", qty=1.0, side="BUY",
        best_bid=100.0, best_ask=100.10, tick_size=0.01,
    )
    assert res["filled"] is True
    assert res["status"] == "FILLED"
    assert 100.0 < captured["price"] < 100.10


def test_maker_empty_book_returns_invalid_price_without_submitting() -> None:
    client = _client()
    submitted: list[str] = []

    def _spot_leg(**kwargs):  # type: ignore[no-untyped-def]
        submitted.append("spot")
        return ({"status": "FILLED", "filled": True, "executedQty": kwargs["qty"]}, [])

    client._execute_spot_leg = _spot_leg  # type: ignore[method-assign]
    res = client.place_maker_with_fallback(
        symbol="BTC/USDT", qty=1.0, side="BUY", best_bid=0.0, best_ask=0.0, tick_size=0.01,
    )
    assert res["status"] == "INVALID_PRICE"
    assert res["filled"] is False
    assert submitted == []


def _stub_futures_client(query_order, market_status="FILLED", market_qty=0.0):  # type: ignore[no-untyped-def]
    client = _client()
    client._execute_perp_leg = lambda **kwargs: {  # type: ignore[method-assign]
        "orderId": 11, "status": "NEW", "filled": False, "executedQty": 0.0,
    }
    client.cancel_order = lambda *a, **k: {"canceled": True}  # type: ignore[method-assign]
    client.query_order = query_order  # type: ignore[method-assign]
    client.build_order_request = lambda *a, **k: {"req": True}  # type: ignore[method-assign]
    calls: list[dict] = []

    def _submit(venue, req, timeout_sec=0.0):  # type: ignore[no-untyped-def]
        calls.append({"venue": venue, "req": req})
        return {"orderId": 22, "status": market_status, "executedQty": market_qty}

    client.submit_signed_order = _submit  # type: ignore[method-assign]
    client._await_fill = lambda venue, s, res, ttl, poll, events: dict(res)  # type: ignore[method-assign]
    return client, calls


def test_maker_futures_partial_market_fill_is_not_reported_as_filled() -> None:
    client, calls = _stub_futures_client(
        lambda *a, **k: {"status": "CANCELED", "executedQty": 0.0},
        market_status="PARTIALLY_FILLED",
        market_qty=0.4,
    )
    res = client.place_maker_with_fallback(
        symbol="BTC/USDT:USDT", qty=1.0, side="BUY",
        best_bid=100.0, best_ask=100.10, tick_size=0.01, venue="futures",
    )
    assert len(calls) == 1
    assert res["filled"] is False
    assert res["status"] == "PARTIALLY_FILLED"
    assert res["order"]["executedQty"] == pytest.approx(0.4)


def test_maker_futures_gtx_rejection_falls_back_to_market() -> None:
    from src.binance_signed_client import BinanceHttpError

    client, calls = _stub_futures_client(
        lambda *a, **k: {"status": "CANCELED", "executedQty": 0.0},
        market_status="FILLED",
        market_qty=1.0,
    )

    def _reject(**kwargs):  # type: ignore[no-untyped-def]
        raise BinanceHttpError("Order would immediately match", code=-2010, msg="post only")

    client._execute_perp_leg = _reject  # type: ignore[method-assign]
    res = client.place_maker_with_fallback(
        symbol="BTC/USDT:USDT", qty=1.0, side="SELL",
        best_bid=100.0, best_ask=100.10, tick_size=0.01, venue="futures",
    )
    assert len(calls) == 1, "GTX reject phai ban MARKET ngay, khong duoc crash"
    assert res["filled"] is True
    assert res["order"]["fallback_from"] == "GTX_REJECTED"


def test_maker_futures_unknown_state_after_cancel_does_not_send_market() -> None:
    def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("query_order timeout")

    client, calls = _stub_futures_client(_boom)
    res = client.place_maker_with_fallback(
        symbol="BTC/USDT:USDT", qty=1.0, side="BUY",
        best_bid=100.0, best_ask=100.10, tick_size=0.01, venue="futures",
    )
    assert calls == [], "Khong chac trang thai lenh -> cam ban MARKET (tranh double fill)"
    assert res["status"] == "UNKNOWN_AFTER_CANCEL"
    assert res["filled"] is False

def test_maker_futures_gtx_rejection_partial_market_does_not_double_submit() -> None:
    """Audit 1.1: sau khi fallback MARKET (chua khop het) KHONG duoc huy + ban tiep."""
    from src.binance_signed_client import BinanceHttpError

    client, calls = _stub_futures_client(
        lambda *a, **k: {"status": "CANCELED", "executedQty": 0.0},
        market_status="PARTIALLY_FILLED",
        market_qty=0.4,
    )

    def _reject(**kwargs):  # type: ignore[no-untyped-def]
        raise BinanceHttpError("post only reject", code=-2010, msg="post only")

    client._execute_perp_leg = _reject  # type: ignore[method-assign]
    res = client.place_maker_with_fallback(
        symbol="BTC/USDT:USDT", qty=1.0, side="BUY",
        best_bid=100.0, best_ask=100.10, tick_size=0.01, venue="futures",
    )
    assert len(calls) == 1, "Khong duoc gui MARKET lan 2 sau khi da fallback"
    assert res["filled"] is False


def test_maker_futures_maker_filled_during_cancel_reports_filled() -> None:
    """Audit 1.2: maker khop 100% ngay truoc/trong luc huy -> phai bao FILLED."""
    client, calls = _stub_futures_client(
        lambda *a, **k: {"status": "FILLED", "executedQty": 1.0},
        market_status="FILLED",
        market_qty=1.0,
    )
    res = client.place_maker_with_fallback(
        symbol="BTC/USDT:USDT", qty=1.0, side="SELL",
        best_bid=100.0, best_ask=100.10, tick_size=0.01, venue="futures",
    )
    assert calls == [], "Da khop du 100% -> khong duoc ban MARKET them"
    assert res["filled"] is True
    assert res["status"] == "FILLED"


def test_maker_futures_market_fallback_rejected_returns_structured_error() -> None:
    """Audit 1.3: MARKET bi tu choi (min notional) -> tra cau truc, khong crash."""
    from src.binance_signed_client import BinanceHttpError

    client, _calls = _stub_futures_client(
        lambda *a, **k: {"status": "CANCELED", "executedQty": 0.0},
        market_status="FILLED",
        market_qty=1.0,
    )

    def _submit_reject(venue, req, timeout_sec=0.0):  # type: ignore[no-untyped-def]
        raise BinanceHttpError("Filter failure: MIN_NOTIONAL", code=-4164, msg="min notional")

    client.submit_signed_order = _submit_reject  # type: ignore[method-assign]
    res = client.place_maker_with_fallback(
        symbol="BTC/USDT:USDT", qty=1.0, side="BUY",
        best_bid=100.0, best_ask=100.10, tick_size=0.01, venue="futures",
    )
    assert res["status"] == "MARKET_FALLBACK_REJECTED"
    assert res["filled"] is False
    assert "MIN_NOTIONAL" in str(res.get("error", ""))
