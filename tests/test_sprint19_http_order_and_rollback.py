"""Test Sprint 19 - HTTP order submission, maker fallback, Atomic Leg Rollback.

Bao phu:
1. dry_run=True chi ky + tra payload mo phong, KHONG goi transport.
2. Gui lenh that (dry_run=False) parse dung orderId/status/executedQty/quote/avgPrice.
3. Bat bien 1 chu ky: KHONG BAO GIO ky lai payload da co `signature`.
4. LIMIT_MAKER bi tu choi (-2010/-1013) -> tu dong fallback MARKET.
5. Maker qua TTL -> cancel + MARKET fallback (instant, TTL=0).
6. Atomic Leg Rollback: spot khop, perp loi -> gui MARKET nguoc chieu -> ROLLED_BACK_SAFE.
7. Rollback that bai -> NAKED_LEG_ALERT (naked_leg=True).
8. Perp raise exception -> van rollback chan Spot.
9. Spot khong khop -> SPOT_LEG_FAILED, naked_leg=False.
10. `perp_available=False` -> chan Perp chi ky mo phong (SIMULATED_SIGNED_ONLY).
11. Chuan Post-Only: Spot LIMIT_MAKER khong kem timeInForce (default cu van LIMIT+GTX).
12. `build_order_request` chan LIMIT thieu price va bo timeInForce cho LIMIT_MAKER.

Tat ca offline qua transport fake, khong doc `.env.testnet`.
"""

from __future__ import annotations

import pytest

from src.binance_signed_client import (
    PERP_ORDER_PATH,
    SPOT_ORDER_PATH,
    BinanceSignedTestnetClient,
)

_SPOT_KEY = "fake-spot-key-0001"
_SPOT_SECRET = "fake-spot-secret-0001"
_FUT_KEY = "fake-futures-key-0002"
_FUT_SECRET = "fake-futures-secret-0002"
_SYMBOL = "BTC/USDT:USDT"

_ENV_NAMES = (
    "BINANCE_TESTNET_API_KEY",
    "BINANCE_TESTNET_API_SECRET",
    "BINANCE_SPOT_TESTNET_API_KEY",
    "BINANCE_SPOT_TESTNET_API_SECRET",
    "BINANCE_FUTURES_TESTNET_API_KEY",
    "BINANCE_FUTURES_TESTNET_API_SECRET",
)


def make_transport(routes):
    """Boc 1 ham routes thanh transport fake (khong bao gio cham mang)."""

    def transport(method, url, params=None, headers=None, timeout_sec=10.0):
        return routes(method, url, params or {}, headers or {}, timeout_sec)

    return transport


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield


def _client(dry_run: bool = True, transport=None) -> BinanceSignedTestnetClient:
    return BinanceSignedTestnetClient(
        spot_api_key=_SPOT_KEY,
        spot_api_secret=_SPOT_SECRET,
        futures_api_key=_FUT_KEY,
        futures_api_secret=_FUT_SECRET,
        dry_run=dry_run,
        autoload_dotenv=False,
        transport=transport,
    )


def _filled_payload(
    order_id: int = 1,
    qty: float = 0.001,
    price: float = 96000.0,
    side: str = "BUY",
    order_type: str = "MARKET",
) -> dict:
    """Response Binance da khop hoan toan (dung cho transport fake)."""
    return {
        "orderId": order_id,
        "symbol": "BTCUSDT",
        "side": side,
        "type": order_type,
        "status": "FILLED",
        "executedQty": str(qty),
        "cummulativeQuoteQty": str(qty * price),
        "avgPrice": str(price),
        "transactTime": 1_700_000_000_000,
    }


# ---------------------------------------------------------------------------
# 1-3. submit_signed_order
# ---------------------------------------------------------------------------


def test_dry_run_submit_returns_payload_without_network() -> None:
    calls: list[tuple[str, str]] = []

    def routes(method, url, params, headers, timeout_sec):
        calls.append((method, url))
        raise AssertionError("dry_run khong duoc goi transport")

    client = _client(dry_run=True, transport=make_transport(routes))
    result = client.submit_signed_order(
        "spot",
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001,
            "type": "MARKET",
        },
        timeout_sec=1.0,
    )

    assert result["dry_run"] is True
    assert result["submitted"] is False
    assert result["status"] == "DRY_RUN"
    assert result["orderId"] is None
    assert result["executedQty"] == 0.0
    assert result["filled"] is False
    assert len(result["params"]["signature"]) == 64
    assert calls == []


def test_live_submit_parses_filled_order() -> None:
    captured: dict = {}

    def routes(method, url, params, headers, timeout_sec):
        captured.update(
            {
                "method": method.upper(),
                "url": url,
                "params": dict(params),
                "headers": dict(headers),
            }
        )
        return _filled_payload(order_id=4242, qty=0.002, price=96000.0)

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.submit_signed_order(
        "spot",
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.002,
            "type": "MARKET",
        },
    )

    assert result["orderId"] == 4242
    assert result["status"] == "FILLED"
    assert result["executedQty"] == pytest.approx(0.002)
    assert result["cummulativeQuoteQty"] == pytest.approx(192.0)
    assert result["avgPrice"] == pytest.approx(96000.0)
    assert result["submitted"] is True
    assert result["dry_run"] is False
    assert result["filled"] is True

    assert captured["method"] == "POST"
    assert captured["url"].endswith(SPOT_ORDER_PATH)
    assert captured["headers"]["X-MBX-APIKEY"] == _SPOT_KEY


def test_submit_never_re_signs_an_already_signed_envelope() -> None:
    """Regression -1022: envelope da ky phai duoc gui nguyen ven, dung 1 chu ky."""
    received: list[dict] = []

    def routes(method, url, params, headers, timeout_sec):
        received.append(dict(params))
        return _filled_payload(order_id=77)

    client = _client(dry_run=False, transport=make_transport(routes))
    envelope = client.build_order_request(
        "spot", _SYMBOL, "BUY", 0.001, "MARKET", timestamp_ms=1_700_000_000_000
    )
    original_signature = envelope["params"]["signature"]

    result = client.submit_signed_order("spot", envelope)

    assert result["orderId"] == 77
    assert len(received) == 1
    sent = received[0]
    assert sent["signature"] == original_signature
    assert sent["timestamp"] == 1_700_000_000_000
    # Chi co dung 1 signature (khong bi ky long nhau).
    assert sum(1 for key in sent if key == "signature") == 1
    assert sent["recvWindow"] == 5000


# ---------------------------------------------------------------------------
# 4-5. Maker -> Market fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reject_code", [-2010, -1013])
def test_limit_maker_rejection_falls_back_to_market(reject_code: int) -> None:
    spot_posts: list[dict] = []

    def routes(method, url, params, headers, timeout_sec):
        if method.upper() == "POST" and url.endswith(SPOT_ORDER_PATH):
            spot_posts.append(dict(params))
            if params.get("type") == "LIMIT_MAKER":
                return {"code": reject_code, "msg": "rejected by exchange"}
            return _filled_payload(
                order_id=555,
                qty=float(params.get("quantity", 0.001)),
                price=float(params.get("price") or 96000.0),
                side=str(params.get("side", "BUY")),
            )
        raise AssertionError(f"Unexpected request {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        0.001,
        0.001,
        spot_price=96000.0,
        spot_use_limit_maker=True,
        perp_available=False,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert result["status"] == "FILLED"
    assert result["spot"]["fallback_from"] == "LIMIT_MAKER"
    assert len(spot_posts) == 2
    # Maker phai la LIMIT_MAKER va KHONG kem timeInForce (tranh -1102).
    assert spot_posts[0]["type"] == "LIMIT_MAKER"
    assert "timeInForce" not in spot_posts[0]
    assert spot_posts[1]["type"] == "MARKET"
    assert any(
        e.get("event") == "LIMIT_MAKER_REJECTED" for e in result["events"]
    )
    assert any(e.get("event") == "MARKET_FALLBACK" for e in result["events"])


def test_maker_ttl_expiry_cancels_then_falls_back_to_market() -> None:
    calls: list[tuple[str, dict]] = []

    def routes(method, url, params, headers, timeout_sec):
        calls.append((method.upper(), dict(params)))
        if method.upper() == "POST" and params.get("type") == "LIMIT_MAKER":
            return {
                "orderId": 9001,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT_MAKER",
                "status": "NEW",
                "executedQty": "0",
                "cummulativeQuoteQty": "0",
                "avgPrice": "0",
            }
        if method.upper() == "DELETE":
            return {"orderId": params.get("orderId"), "status": "CANCELED"}
        if method.upper() == "POST" and params.get("type") == "MARKET":
            return _filled_payload(
                order_id=9002,
                qty=float(params.get("quantity", 0.001)),
                side=str(params.get("side", "BUY")),
            )
        # TTL=0 nen khong duoc poll GET.
        raise AssertionError(f"Unexpected request {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        0.001,
        0.001,
        spot_price=96000.0,
        spot_use_limit_maker=True,
        perp_available=False,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    methods = [m for m, _ in calls]
    assert "DELETE" in methods
    assert result["status"] == "FILLED"
    assert result["spot"]["fallback_from"] == "LIMIT_MAKER_TTL_EXPIRED"
    assert any(
        e.get("event") == "LIMIT_MAKER_CANCELED" for e in result["events"]
    )
    # Lenh MARKET gui sau khi huy maker.
    assert methods.index("DELETE") < len(methods) - 1
    assert calls[-1][1]["type"] == "MARKET"


# ---------------------------------------------------------------------------
# 6-8. Atomic Leg Rollback
# ---------------------------------------------------------------------------


def test_rollback_reverses_filled_spot_leg_when_perp_fails() -> None:
    spot_posts: list[dict] = []

    def routes(method, url, params, headers, timeout_sec):
        if method.upper() == "POST" and url.endswith(SPOT_ORDER_PATH):
            spot_posts.append(dict(params))
            return _filled_payload(
                order_id=1000 + len(spot_posts),
                qty=float(params.get("quantity", 0.001)),
                side=str(params.get("side", "BUY")),
            )
        if method.upper() == "POST" and url.endswith(PERP_ORDER_PATH):
            return {"code": -2019, "msg": "Margin is insufficient."}
        raise AssertionError(f"Unexpected request {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        0.001,
        0.001,
        spot_price=96000.0,
        post_only_spot=False,
        spot_use_limit_maker=False,
        perp_available=True,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert result["status"] == "ROLLED_BACK_SAFE"
    assert result["rolled_back"] is True
    assert result["naked_leg"] is False
    assert result["perp"]["status"] == "PERP_LEG_EXCEPTION"

    # Chan 1 BUY, rollback phai la SELL cung khoi luong da khop.
    assert len(spot_posts) == 2
    assert spot_posts[0]["side"] == "BUY"
    assert spot_posts[1]["side"] == "SELL"
    assert spot_posts[1]["type"] == "MARKET"
    assert float(spot_posts[1]["quantity"]) == pytest.approx(0.001)

    assert result["rollback"]["side"] == "SELL"
    assert result["rollback"]["qty"] == pytest.approx(0.001)
    assert result["rollback"]["filled"] is True
    assert any(
        e.get("event") == "REVERSE_MARKET_SUBMITTED" for e in result["events"]
    )


def test_rollback_failure_raises_naked_leg_alert() -> None:
    spot_posts: list[dict] = []

    def routes(method, url, params, headers, timeout_sec):
        if method.upper() == "POST" and url.endswith(SPOT_ORDER_PATH):
            spot_posts.append(dict(params))
            if len(spot_posts) == 1:
                return _filled_payload(order_id=2001, side="BUY")
            return {"code": -2010, "msg": "Rollback rejected"}
        if method.upper() == "POST" and url.endswith(PERP_ORDER_PATH):
            return {"code": -2019, "msg": "Margin is insufficient."}
        raise AssertionError(f"Unexpected request {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        0.001,
        0.001,
        spot_price=96000.0,
        post_only_spot=False,
        spot_use_limit_maker=False,
        perp_available=True,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert result["status"] == "NAKED_LEG_ALERT"
    assert result["naked_leg"] is True
    assert result["rolled_back"] is True
    assert result["rollback"]["filled"] is False
    assert "error" in result["rollback"]
    assert any(
        e.get("event") == "ROLLBACK_EXCEPTION" for e in result["events"]
    )


def test_perp_exception_still_triggers_rollback() -> None:
    spot_posts: list[dict] = []

    def routes(method, url, params, headers, timeout_sec):
        if method.upper() == "POST" and url.endswith(SPOT_ORDER_PATH):
            spot_posts.append(dict(params))
            return _filled_payload(
                order_id=3000 + len(spot_posts), side=str(params.get("side", "BUY"))
            )
        if method.upper() == "POST" and url.endswith(PERP_ORDER_PATH):
            raise RuntimeError("socket hang up")
        raise AssertionError(f"Unexpected request {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        0.001,
        0.001,
        spot_price=96000.0,
        post_only_spot=False,
        spot_use_limit_maker=False,
        perp_available=True,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert result["status"] == "ROLLED_BACK_SAFE"
    assert result["perp"]["status"] == "PERP_LEG_EXCEPTION"
    assert "socket hang up" in result["perp"]["error"]
    assert any(
        e.get("event") == "EXCEPTION" for e in result["events"]
    )
    assert spot_posts[1]["side"] == "SELL"


# ---------------------------------------------------------------------------
# 9-10. Spot leg fail + perp simulation
# ---------------------------------------------------------------------------


def test_spot_leg_not_filled_returns_spot_leg_failed() -> None:
    def routes(method, url, params, headers, timeout_sec):
        if method.upper() == "POST" and url.endswith(SPOT_ORDER_PATH):
            return {
                "orderId": 4001,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "status": "NEW",
                "executedQty": "0",
                "cummulativeQuoteQty": "0",
                "avgPrice": "0",
            }
        raise AssertionError(f"Unexpected request {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        0.001,
        0.001,
        spot_price=96000.0,
        post_only_spot=False,
        spot_use_limit_maker=False,
        perp_available=True,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert result["status"] == "SPOT_LEG_FAILED"
    assert result["naked_leg"] is False
    assert result["rolled_back"] is False
    assert result["perp"] is None


def test_perp_unavailable_is_signed_simulation_only() -> None:
    spot_posts: list[dict] = []
    futures_calls: list[str] = []

    def routes(method, url, params, headers, timeout_sec):
        if "/fapi/" in url:
            futures_calls.append(url)
            raise AssertionError("Khong duoc gui lenh Futures khi perp_available=False")
        if method.upper() == "POST" and url.endswith(SPOT_ORDER_PATH):
            spot_posts.append(dict(params))
            return _filled_payload(
                order_id=5001, qty=0.001, side=str(params.get("side", "BUY"))
            )
        raise AssertionError(f"Unexpected request {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        0.001,
        0.001,
        spot_price=96000.0,
        post_only_spot=False,
        spot_use_limit_maker=False,
        perp_available=False,
        require_futures=False,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert futures_calls == []
    assert result["status"] == "FILLED"
    assert result["perp_simulated"] is True
    assert result["rolled_back"] is False

    perp = result["perp"]
    assert perp["status"] == "SIMULATED_SIGNED_ONLY"
    assert perp["simulated"] is True
    assert perp["signed"] is True
    assert perp["executedQty"] == 0.0
    assert len(perp["params"]["signature"]) == 64
    assert perp["url"].endswith(PERP_ORDER_PATH)


# ---------------------------------------------------------------------------
# 11-12. Chuan Post-Only va validation build_order_request
# ---------------------------------------------------------------------------


def test_delta_neutral_pair_post_only_semantics() -> None:
    client = _client()

    maker = client.build_delta_neutral_order_pair(
        _SYMBOL, 0.001, 0.001, spot_price=96000.0, spot_use_limit_maker=True
    )
    spot_params = maker["spot"]["params"]
    assert spot_params["type"] == "LIMIT_MAKER"
    assert "timeInForce" not in spot_params
    assert spot_params["price"] == pytest.approx(96000.0)

    # Tuong thich nguoc: default van la LIMIT + GTX.
    legacy = client.build_delta_neutral_order_pair(
        _SYMBOL, 0.001, 0.001, spot_price=96000.0
    )
    assert legacy["spot"]["params"]["type"] == "LIMIT"
    assert legacy["spot"]["params"]["timeInForce"] == "GTX"

    # Perp post-only dung LIMIT + GTX; mac dinh van MARKET.
    perp_maker = client.build_delta_neutral_order_pair(
        _SYMBOL,
        0.001,
        0.001,
        spot_price=96000.0,
        perp_post_only=True,
        perp_price=96000.0,
    )
    perp_params = perp_maker["perp"]["params"]
    assert perp_params["type"] == "LIMIT"
    assert perp_params["timeInForce"] == "GTX"
    assert perp_params["price"] == pytest.approx(96000.0)
    assert maker["perp"]["params"]["type"] == "MARKET"


def test_build_order_request_validates_price_and_drops_tif_for_maker() -> None:
    client = _client()

    with pytest.raises(ValueError):
        client.build_order_request("spot", _SYMBOL, "BUY", 0.001, "LIMIT")

    maker_req = client.build_order_request(
        "spot", _SYMBOL, "BUY", 0.001, "LIMIT_MAKER", price=96000.0,
        time_in_force="GTX",
    )
    assert maker_req["params"]["type"] == "LIMIT_MAKER"
    assert maker_req["params"]["price"] == pytest.approx(96000.0)
    assert "timeInForce" not in maker_req["params"]

    limit_req = client.build_order_request(
        "spot", _SYMBOL, "BUY", 0.001, "LIMIT", price=96000.0,
        time_in_force="gtx",
    )
    assert limit_req["params"]["timeInForce"] == "GTX"
    assert limit_req["url"].endswith(SPOT_ORDER_PATH)

    futures_req = client.build_order_request(
        "futures", _SYMBOL, "SELL", 0.001, "MARKET", reduce_only=True
    )
    assert futures_req["params"]["reduceOnly"] is True
    assert futures_req["url"].endswith(PERP_ORDER_PATH)
    assert futures_req["headers"]["X-MBX-APIKEY"] == _FUT_KEY

# ---------------------------------------------------------------------------
# 13-16. Audit hardening (quant_auditor Sprint 19: D.1, D.2, A.1)
# ---------------------------------------------------------------------------


def test_partial_maker_fill_only_buys_remaining_qty() -> None:
    """D.1 BLOCKER: maker khop 1 phan -> lenh MARKET chi mua phan CON THIEU.

    Neu mua lai du `qty` goc thi tong khoi luong khop se vuot (double fill) va
    pha vo delta-neutral.
    """
    market_params: list[dict] = []
    state = {"maker_queried": 0}

    def routes(method, url, params, headers, timeout_sec):
        method = method.upper()
        if method == "POST" and url.endswith(SPOT_ORDER_PATH):
            otype = str(params.get("type", ""))
            if otype == "LIMIT_MAKER":
                # Maker khop 0.001 / 0.002 roi treo.
                return {
                    "orderId": 11,
                    "symbol": "BTCUSDT",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "0.001",
                    "cummulativeQuoteQty": "96.0",
                    "avgPrice": "96000.0",
                }
            market_params.append(dict(params))
            qty = float(params["quantity"])
            return _filled_payload(order_id=12, qty=qty)
        if method == "GET" and url.endswith(SPOT_ORDER_PATH):
            state["maker_queried"] += 1
            return {
                "orderId": params.get("orderId"),
                "symbol": "BTCUSDT",
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.001",
                "avgPrice": "96000.0",
            }
        if method == "DELETE":
            return {"orderId": params.get("orderId"), "status": "CANCELED"}
        raise AssertionError(f"route khong ho tro: {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        spot_qty=0.002,
        perp_qty=0.002,
        spot_price=96000.0,
        post_only_spot=True,
        spot_use_limit_maker=True,
        perp_available=False,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert len(market_params) == 1, "phai co dung 1 lenh MARKET bu"
    assert float(market_params[0]["quantity"]) == pytest.approx(0.001)
    assert result["spot"]["executedQty"] == pytest.approx(0.002)
    assert result["spot"]["filled"] is True
    assert result["spot"]["maker_filled_qty"] == pytest.approx(0.001)


def test_partially_filled_is_not_treated_as_filled() -> None:
    """D.2 BLOCKER: PARTIALLY_FILLED KHONG duoc coi la da khop xong."""
    from src.binance_signed_client import _is_filled

    assert _is_filled({"status": "FILLED", "executedQty": 1.0}) is True
    assert _is_filled({"status": "PARTIALLY_FILLED", "executedQty": 0.5}) is False
    assert _is_filled({"status": "NEW", "executedQty": 0.0}) is False
    assert _is_filled(None) is False


def test_partial_then_never_filled_triggers_market_fallback() -> None:
    """D.2: partial roi khong len FILLED -> van phai cancel + MARKET bu phan thieu."""
    market_qtys: list[float] = []

    def routes(method, url, params, headers, timeout_sec):
        method = method.upper()
        if method == "POST" and url.endswith(SPOT_ORDER_PATH):
            if str(params.get("type", "")) == "LIMIT_MAKER":
                return {
                    "orderId": 21,
                    "symbol": "BTCUSDT",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "0.0005",
                    "avgPrice": "96000.0",
                }
            market_qtys.append(float(params["quantity"]))
            return _filled_payload(order_id=22, qty=float(params["quantity"]))
        if method == "GET" and url.endswith(SPOT_ORDER_PATH):
            return {
                "orderId": params.get("orderId"),
                "symbol": "BTCUSDT",
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.001",
                "avgPrice": "96000.0",
            }
        if method == "DELETE":
            return {"orderId": params.get("orderId"), "status": "CANCELED"}
        raise AssertionError(f"route khong ho tro: {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        spot_qty=0.002,
        perp_qty=0.002,
        spot_price=96000.0,
        perp_available=False,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    # Da khop 0.001 (con 0.001) -> MARKET chi mua 0.001, tong khop = 0.002.
    assert market_qtys == [pytest.approx(0.001)]
    assert result["spot"]["executedQty"] == pytest.approx(0.002)


def test_spot_leg_exception_after_partial_fill_triggers_rollback() -> None:
    """A.1 MAJOR: loi giua chan Spot sau khi da khop 1 phan -> van phai rollback."""
    seen_rollback: list[dict] = []

    def routes(method, url, params, headers, timeout_sec):
        method = method.upper()
        if method == "POST" and url.endswith(SPOT_ORDER_PATH):
            side = str(params.get("side", "BUY")).upper()
            if str(params.get("type", "")) == "LIMIT_MAKER":
                return {
                    "orderId": 31,
                    "symbol": "BTCUSDT",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "0.001",
                    "avgPrice": "96000.0",
                }
            seen_rollback.append(dict(params))
            assert side == "SELL"
            return _filled_payload(order_id=32, qty=float(params["quantity"]), side="SELL")
        if method == "GET" and url.endswith(SPOT_ORDER_PATH):
            return {
                "orderId": params.get("orderId"),
                "symbol": "BTCUSDT",
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.001",
                "avgPrice": "96000.0",
            }
        if method == "DELETE":
            raise RuntimeError("network down khi huy lenh")
        raise AssertionError(f"route khong ho tro: {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        spot_qty=0.002,
        perp_qty=0.002,
        spot_price=96000.0,
        perp_available=False,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert result["rolled_back"] is True
    assert result["status"] == "ROLLED_BACK_SAFE"
    assert result["naked_leg"] is False
    assert len(seen_rollback) == 1
    assert float(seen_rollback[0]["quantity"]) == pytest.approx(0.001)


def test_untradable_remainder_triggers_rollback_not_naked_leg() -> None:
    """A.1 + step_size: phan con lai KHONG chia het step_size -> phai rollback.

    Vi du that: BTCUSDT step_size=0.001, maker khop 0.0004/0.001 -> con 0.0006
    khong dat duoc -> KHONG duoc gui lenh sai (se bi -1013) va KHONG duoc bo mac
    phan da khop: phai ban MARKET 0.0004 de ve trang thai an toan.
    """
    market_params: list[dict] = []

    def routes(method, url, params, headers, timeout_sec):
        method = method.upper()
        if method == "POST" and url.endswith(SPOT_ORDER_PATH):
            if str(params.get("type", "")) == "LIMIT_MAKER":
                return {
                    "orderId": 41,
                    "symbol": "BTCUSDT",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "0.0004",
                    "avgPrice": "96000.0",
                }
            market_params.append(dict(params))
            return _filled_payload(
                order_id=42, qty=float(params["quantity"]), side=params.get("side", "SELL")
            )
        if method == "GET" and url.endswith(SPOT_ORDER_PATH):
            return {
                "orderId": params.get("orderId"),
                "symbol": "BTCUSDT",
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.0004",
                "avgPrice": "96000.0",
            }
        if method == "DELETE":
            return {"orderId": params.get("orderId"), "status": "CANCELED"}
        raise AssertionError(f"route khong ho tro: {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        spot_qty=0.001,
        perp_qty=0.001,
        spot_price=96000.0,
        perp_available=False,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )

    assert len(market_params) == 1
    assert str(market_params[0]["side"]).upper() == "SELL"
    assert float(market_params[0]["quantity"]) == pytest.approx(0.0004)
    assert result["status"] == "ROLLED_BACK_SAFE"
    assert result["naked_leg"] is False
    events = {e.get("event") for e in result["events"]}
    assert "SPOT_LEG_EXCEPTION" in events
    assert "REVERSE_MARKET_SUBMITTED" in events


# ---------------------------------------------------------------------------
# 19-22. Re-audit hardening (quant_auditor 11e1331b: NEW MAJOR 1-2, NEW MINOR 4)
# ---------------------------------------------------------------------------


def test_market_fallback_poll_keeps_maker_fill_in_executed_qty() -> None:
    """NEW MAJOR #1: lenh MARKET bu phai poll -> KHONG duoc mat khoi luong maker.

    Truoc khi va: `mkt["executedQty"]` duoc cong don TRUOC khi poll, va neu lenh
    MARKET chua FILLED ngay thi ket qua poll ghi de -> mat phan maker da khop.
    """
    polls = {"n": 0}

    def routes(method, url, params, headers, timeout_sec):
        method = method.upper()
        if method == "POST" and url.endswith(SPOT_ORDER_PATH):
            if str(params.get("type", "")) == "LIMIT_MAKER":
                return {
                    "orderId": 51,
                    "symbol": "BTCUSDT",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "0.001",
                    "avgPrice": "96000.0",
                }
            # Lenh MARKET bu: chua khop ngay -> status NEW.
            return {
                "orderId": 52,
                "symbol": "BTCUSDT",
                "status": "NEW",
                "executedQty": "0.0",
                "avgPrice": "0.0",
            }
        if method == "GET" and url.endswith(SPOT_ORDER_PATH):
            oid = params.get("orderId")
            if oid == 51:
                return {
                    "orderId": 51,
                    "symbol": "BTCUSDT",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "0.001",
                    "avgPrice": "96000.0",
                }
            polls["n"] += 1
            return {
                "orderId": 52,
                "symbol": "BTCUSDT",
                "status": "FILLED",
                "executedQty": "0.001",
                "cummulativeQuoteQty": "96.0",
                "avgPrice": "96000.0",
            }
        if method == "DELETE":
            return {"orderId": params.get("orderId"), "status": "CANCELED"}
        raise AssertionError(f"route khong ho tro: {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        spot_qty=0.002,
        perp_qty=0.002,
        spot_price=96000.0,
        perp_available=False,
        maker_ttl_sec=1.0,
        poll_interval_sec=0.0,
    )

    assert result["spot"]["maker_filled_qty"] == pytest.approx(0.001)
    assert result["spot"]["market_executed_qty"] == pytest.approx(0.001)
    # Tong phai la 0.002, khong duoc tut ve 0.001.
    assert result["spot"]["executedQty"] == pytest.approx(0.002)
    assert result["spot"]["filled"] is True


def test_aggregated_executed_qty_never_exceeds_requested() -> None:
    """NEW MAJOR #1b: tong khop khong bao gio vuot khoi luong dat (chong double fill)."""
    def routes(method, url, params, headers, timeout_sec):
        method = method.upper()
        if method == "POST" and url.endswith(SPOT_ORDER_PATH):
            if str(params.get("type", "")) == "LIMIT_MAKER":
                return {
                    "orderId": 61,
                    "symbol": "BTCUSDT",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "0.0015",
                    "avgPrice": "96000.0",
                }
            return _filled_payload(order_id=62, qty=float(params["quantity"]))
        if method == "GET" and url.endswith(SPOT_ORDER_PATH):
            return {
                "orderId": params.get("orderId"),
                "symbol": "BTCUSDT",
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.0015",
                "avgPrice": "96000.0",
            }
        if method == "DELETE":
            return {"orderId": params.get("orderId"), "status": "CANCELED"}
        raise AssertionError(f"route khong ho tro: {method} {url}")

    client = _client(dry_run=False, transport=make_transport(routes))
    result = client.execute_live_testnet_order_pair(
        _SYMBOL,
        spot_qty=0.002,
        perp_qty=0.002,
        spot_price=96000.0,
        perp_available=False,
        maker_ttl_sec=0.0,
        poll_interval_sec=0.0,
    )
    assert result["spot"]["executedQty"] <= 0.002 + 1e-12


def test_quantize_price_handles_non_power_of_ten_tick() -> None:
    """NEW MINOR #4: tick_size 0.25 (khong phai luy thua 10) van phai dung."""
    from src.binance_signed_client import quantize_price

    # BTCUSDT: tick 0.01 - khong duoc sinh sai so nhi phan.
    assert quantize_price(82790.019, 0.01, "BUY") == 82790.01
    assert quantize_price(82790.011, 0.01, "SELL") == 82790.02
    assert repr(quantize_price(82872.54, 0.01, "BUY")) == "82872.54"
    # Tick khong phai luy thua 10.
    assert quantize_price(100.75, 0.25, "BUY") == 100.75
    assert quantize_price(100.8, 0.25, "BUY") == 100.75
    assert quantize_price(100.76, 0.25, "SELL") == 101.0
    # Tick 1 va 0.00001 van giu nguyen hanh vi.
    assert quantize_price(100.9, 1.0, "BUY") == 100.0
    assert quantize_price(0.000023, 0.00001, "BUY") == pytest.approx(0.00002)
