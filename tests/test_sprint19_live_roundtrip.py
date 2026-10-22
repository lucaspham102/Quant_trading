"""Test Sprint 19 - Script round-trip Testnet live (`scripts/run_testnet_live_roundtrip.py`).

Bao phu:
1. `build_mock_transport()` tra loi account/ticker/order/exchangeInfo hoan toan offline.
2. `run_roundtrip(client=mock)` -> verdict PASS/PASS_SPOT_ONLY, 5 buoc ok,
   entry/exit order id + reconcile SYNCED + executed_qty > 0.
3. Ghi evidence markdown + log JSONL vao tmp_path, khong lo secret.
4. Log JSONL KHONG duoc chua raw API key (bat bien bao mat).
5. `main(["--mock-transport", "--no-write", "--json"])` tra 0.
6. Nhanh FAIL (ca 2 venue chua xac thuc) -> verdict FAIL, khong raise.
7. `fetch_symbol_filters` raise -> script van chay (fallback an toan).

Offline hoan toan: khong doc `.env.testnet`, khong goi mang.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_testnet_live_roundtrip.py"
)


def _load_roundtrip_module():
    """Nap script bang importlib (scripts/ khong co __init__.py)."""
    spec = importlib.util.spec_from_file_location("rt_sprint19", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = _load_roundtrip_module()

from src.binance_signed_client import (  # noqa: E402
    BinanceHttpError,
    BinanceSignedTestnetClient,
)

_SPOT_KEY = "mock-spot-key"
_SPOT_SECRET = "mock-spot-secret"
_FUT_KEY = "mock-futures-key"
_FUT_SECRET = "mock-futures-secret"

_ENV_NAMES = (
    "BINANCE_TESTNET_API_KEY",
    "BINANCE_TESTNET_API_SECRET",
    "BINANCE_SPOT_TESTNET_API_KEY",
    "BINANCE_SPOT_TESTNET_API_SECRET",
    "BINANCE_FUTURES_TESTNET_API_KEY",
    "BINANCE_FUTURES_TESTNET_API_SECRET",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Khong de test nao doc `.env.testnet` that hoac ro ri bien moi truong."""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield


def _mock_client() -> BinanceSignedTestnetClient:
    return BinanceSignedTestnetClient(
        dry_run=False,
        recv_window=10000,
        spot_api_key=_SPOT_KEY,
        spot_api_secret=_SPOT_SECRET,
        futures_api_key=_FUT_KEY,
        futures_api_secret=_FUT_SECRET,
        autoload_dotenv=False,
        transport=rt.build_mock_transport(),
    )


# ---------------------------------------------------------------------------
# 1. Mock transport offline
# ---------------------------------------------------------------------------


def test_mock_transport_answers_all_offline_routes() -> None:
    client = _mock_client()

    spot_access = client.check_account_access("spot")
    assert spot_access["authenticated"] is True
    assert spot_access["holdings"]["USDT"] == pytest.approx(5000.0)

    futures_access = client.check_account_access("futures")
    assert futures_access["authenticated"] is True
    assert futures_access["error"] is None

    ticker = client.fetch_book_ticker("BTC/USDT:USDT", "spot")
    assert ticker["bid"] > 0.0
    assert ticker["ask"] >= ticker["bid"]
    assert ticker["mid"] == pytest.approx((ticker["bid"] + ticker["ask"]) / 2.0)

    filters = client.fetch_symbol_filters("BTC/USDT:USDT", "spot")
    assert filters["tick_size"] == pytest.approx(0.01)
    assert filters["step_size"] == pytest.approx(0.001)
    assert filters["min_notional"] == pytest.approx(10.0)

    envelope = client.build_order_request(
        "spot", "BTC/USDT:USDT", "BUY", 0.001, "MARKET"
    )
    order = client.submit_signed_order("spot", envelope)
    assert order["status"] == "FILLED"
    assert order["orderId"] is not None
    assert order["executedQty"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# 2. Round-trip thanh cong
# ---------------------------------------------------------------------------


def test_run_roundtrip_with_mock_client_passes() -> None:
    result = rt.run_roundtrip(client=_mock_client(), write_files=False)

    assert result["verdict"] in ("PASS", "PASS_SPOT_ONLY")
    steps = result["steps"]
    assert len(steps) == 5
    assert all(step["ok"] for step in steps), [
        (s["step"], s["detail"]) for s in steps if not s["ok"]
    ]

    assert result["entry_order_id"] is not None
    assert result["exit_order_id"] is not None
    assert result["entry_reconcile_status"] == "SYNCED"
    assert result["exit_reconcile_status"] == "SYNCED"
    assert result["executed_qty"] > 0.0


# ---------------------------------------------------------------------------
# 3-4. Evidence + log, khong lo credential
# ---------------------------------------------------------------------------


def test_evidence_and_log_written_without_secret_values(tmp_path) -> None:
    evidence = tmp_path / "docs" / "sprint19_evidence.md"
    log = tmp_path / "logs" / "sprint19_roundtrip.jsonl"

    result = rt.run_roundtrip(
        client=_mock_client(),
        write_files=True,
        evidence_path=str(evidence),
        log_path=str(log),
    )

    assert evidence.is_file()
    assert log.is_file()

    evidence_text = evidence.read_text(encoding="utf-8")
    log_text = log.read_text(encoding="utf-8")

    # Secret raw khong duoc xuat hien o BAT KY dau.
    for secret in (_SPOT_SECRET, _FUT_SECRET):
        assert secret not in evidence_text
        assert secret not in log_text

    # Evidence markdown chi duoc dung dang che.
    assert "loaded=True (len=" in evidence_text
    assert result["verdict"] in evidence_text
    assert _SPOT_KEY not in evidence_text
    assert _FUT_KEY not in evidence_text

    log_record = json.loads(log_text.strip().splitlines()[0])
    assert log_record["verdict"] == result["verdict"]
    assert log_record["entry_order_id"] == result["entry_order_id"]


def test_jsonl_log_never_writes_raw_api_key(tmp_path) -> None:
    """Bao mat: log JSONL khong duoc chua raw key (chi duoc dang che)."""
    log = tmp_path / "logs" / "roundtrip.jsonl"
    evidence = tmp_path / "docs" / "roundtrip.md"

    rt.run_roundtrip(
        client=_mock_client(),
        write_files=True,
        evidence_path=str(evidence),
        log_path=str(log),
    )

    log_text = log.read_text(encoding="utf-8")
    leaked = [key for key in (_SPOT_KEY, _FUT_KEY) if key in log_text]
    assert not leaked, (
        "SECURITY BUG: scripts/run_testnet_live_roundtrip.py ghi raw API key "
        "vao JSONL evidence log qua order-result 'headers' (X-MBX-APIKEY). "
        f"Key bi lo: {leaked}. Can redact headers truoc khi json.dumps(result)."
    )


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def test_main_mock_transport_no_write_returns_zero(capsys) -> None:
    rc = rt.main(["--mock-transport", "--no-write", "--json"])
    out = capsys.readouterr().out.strip().splitlines()

    assert rc == 0
    payload = json.loads(out[-1])
    assert payload["verdict"] in ("PASS", "PASS_SPOT_ONLY")


# ---------------------------------------------------------------------------
# 6-7. Duong loi (khong raise) va fallback
# ---------------------------------------------------------------------------


class _UnauthenticatedClient:
    """Client gia: ca 2 venue deu khong xac thuc."""

    def credential_status(self) -> dict:
        return {"spot_key": "not_loaded", "futures_key": "not_loaded"}

    def sync_server_time(self, venue: str = "both", timeout_sec: float = 5.0) -> dict:
        return {
            "spot_offset_ms": 0,
            "futures_offset_ms": 0,
            "errors": ["offline"],
            "synced": False,
        }

    def check_account_access(
        self, venue: str, timeout_sec: float = 5.0
    ) -> dict:
        return {
            "venue": venue,
            "authenticated": False,
            "error": f"MISSING_{venue.upper()}_CREDENTIALS",
        }


def test_run_roundtrip_returns_fail_without_raising() -> None:
    result = rt.run_roundtrip(client=_UnauthenticatedClient(), write_files=False)

    assert result["verdict"] == "FAIL"
    assert result["spot_auth"] is False
    assert result["futures_auth"] is False
    assert len(result["steps"]) == 1
    assert result["steps"][0]["ok"] is False
    assert result["entry_order_id"] is None


def test_run_roundtrip_survives_symbol_filters_error(monkeypatch) -> None:
    client = _mock_client()

    def boom(*args, **kwargs):
        raise BinanceHttpError("exchangeInfo unavailable", code=-1000, msg="boom")

    monkeypatch.setattr(client, "fetch_symbol_filters", boom)

    result = rt.run_roundtrip(client=client, write_files=False)

    # Buoc 2 van ok nho fallback ve EXCHANGE_LOT_SPECS noi bo.
    step2 = result["steps"][1]
    assert step2["ok"] is True
    assert result["verdict"] in ("PASS", "PASS_SPOT_ONLY")
    assert result["entry_order_id"] is not None


# ---------------------------------------------------------------------------
# 8-10. Audit hardening: F.1 (scrub JSONL) + E.1 (khong gia SYNCED)
# ---------------------------------------------------------------------------


def test_jsonl_line_is_scrubbed_even_if_record_has_raw_secret(tmp_path) -> None:
    """F.1 MINOR: duong ghi JSONL phai ap _scrub_secrets, khong chi _sanitize.

    Giai lap tinh huong xau: mot truong long (vd `error`/`headers` tu thu vien
    ben ngoai) vo tinh chua nguyen API key -> file log KHONG duoc phep chua no.
    """
    real_key = "AKIA" + "Z" * 60
    log_path = tmp_path / "leak.jsonl"
    ev_path = tmp_path / "evidence.md"
    record = {
        "run_id": "sprint19-f1",
        "symbol": "BTC/USDT:USDT",
        "verdict": "PASS",
        "entry_order_id": 1,
        "exit_order_id": 2,
        "detail": {"headers": {"X-MBX-APIKEY": real_key}, "note": "ok"},
    }

    rt._append_evidence_and_log(result=record, evidence_path=str(ev_path), log_path=str(log_path))

    text = log_path.read_text(encoding="utf-8")
    assert real_key not in text
    assert "Z" * 60 not in text
    assert ("[REDACTED]" in text) or ("***" in text)
    assert ev_path.exists()


def test_spot_only_snapshot_is_never_reported_as_plain_synced(monkeypatch) -> None:
    """E.1 MAJOR: khi khong doc duoc Futures, status phai mang tien to SPOT_ONLY_.

    Neu van bao "SYNCED" thi nguoi van hanh tuong perp da doi soat -> che giau
    vi the mo coi (orphan position) ben Futures.
    """
    symbol = "BTC/USDT:USDT"

    class _SpotOnlyClient:
        def check_account_access(self, venue="spot"):
            return {"authenticated": True, "holdings": {"BTC": 0.001, "USDT": 5000.0}}

        def fetch_exchange_reconcile_snapshot(self, *a, **kw):  # pragma: no cover
            return {"snapshot_healthy": False}

    snap = rt._build_snapshot(
        _SpotOnlyClient(), symbol=symbol, mid_price=96000.0, futures_live=False
    )

    assert snap["snapshot_scope"] == "spot_only"
    assert snap["perp_positions_verified"] is False
    assert rt._scoped_status("SYNCED", snap["perp_positions_verified"]) == "SPOT_ONLY_SYNCED"
    assert rt._scoped_status("SYNCED", True) == "SYNCED"


def test_evidence_row_records_spot_only_scope(tmp_path) -> None:
    """E.1: dong evidence phai ghi ro status SPOT_ONLY_... de khong hieu nham."""
    ev_path = tmp_path / "evidence.md"
    log_path = tmp_path / "rt.jsonl"
    record = {
        "run_id": "sprint19-e1",
        "symbol": "BTC/USDT:USDT",
        "verdict": "PASS_SPOT_ONLY",
        "entry_order_id": 11,
        "exit_order_id": 12,
        "executed_qty": 0.001,
        "fill_price": 96000.0,
        "entry_reconcile_status": "SPOT_ONLY_SYNCED",
        "exit_reconcile_status": "SPOT_ONLY_SYNCED",
        "server_offsets": {"spot": 1200, "futures": 900},
    }

    rt._append_evidence_and_log(result=record, evidence_path=str(ev_path), log_path=str(log_path))

    md = ev_path.read_text(encoding="utf-8")
    assert "SPOT_ONLY_SYNCED" in md
    assert "PASS_SPOT_ONLY" in md


def test_verdict_is_spot_only_when_futures_positions_unverified() -> None:
    """NEW MINOR #3: co khoa Futures that nhung snapshot spot-only -> KHONG duoc PASS.

    Neu cap PASS day du thi nguoi van hanh se tuong vi the Futures da doi soat,
    trong khi thuc te co the dang co vi the mo coi (orphan).
    """
    def routes(method, url, params, headers, timeout_sec):
        if "ticker/bookTicker" in url:
            return {"symbol": "BTCUSDT", "bidPrice": "95990.0", "askPrice": "96010.0"}
        if "exchangeInfo" in url:
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.00001"},
                            {"filterType": "NOTIONAL", "minNotional": "5.0"},
                        ],
                    }
                ]
            }
        if "api/v3/account" in url:
            return {
                "balances": [
                    {"asset": "USDT", "free": "5000.0", "locked": "0.0"},
                    {"asset": "BTC", "free": "1.0", "locked": "0.0"},
                ]
            }
        if "fapi/v1/order" in url:
            qty = float(params.get("quantity", 0.0))
            return {
                "orderId": 91,
                "symbol": "BTCUSDT",
                "status": "FILLED",
                "executedQty": str(qty),
                "cumQuote": str(qty * 96000.0),
                "avgPrice": "96000.0",
                "updateTime": 1,
            }
        if "api/v3/order" in url:
            qty = float(params.get("quantity", 0.0))
            return {
                "orderId": 1,
                "symbol": "BTCUSDT",
                "status": "FILLED",
                "executedQty": str(qty),
                "cummulativeQuoteQty": str(qty * 95990.0),
                "avgPrice": "95990.0",
                "transactTime": 1,
            }
        raise AssertionError(f"route khong ho tro: {method} {url}")

    client = BinanceSignedTestnetClient(
        spot_api_key="k" * 20,
        spot_api_secret="s" * 20,
        futures_api_key="f" * 20,
        futures_api_secret="f2" * 10,
        dry_run=False,
        autoload_dotenv=False,
        transport=routes,
    )
    # Chan Futures "co khoa" nhung KHONG doc duoc vi the -> fallback spot-only.
    class _FuturesBrokenSnapshot:
        """Futures xac thuc OK nhung snapshot vi the loi (mo phong su co mang)."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def check_account_access(self, venue="spot", **kw):
            if venue == "futures":
                return {"venue": "futures", "authenticated": True, "error": None,
                        "holdings": {}, "server_time_offset_ms": 0}
            return self._inner.check_account_access(venue, **kw)

        def fetch_exchange_reconcile_snapshot(self, *a, **kw):
            return {"snapshot_healthy": False, "error": "simulated futures outage",
                    "spot_balances": {}, "perp_positions": {},
                    "exchange_equity_usd": 0.0}

    result = rt.run_roundtrip(
        client=_FuturesBrokenSnapshot(client), write_files=False
    )

    assert result["futures_auth"] is True
    assert result["futures_positions_verified"] is False
    assert result["verdict_scope"] == "spot_only"
    assert result["verdict"] != "PASS"
    assert result["verdict"] == "PASS_SPOT_ONLY"
    assert result["futures_verification_warning"]
