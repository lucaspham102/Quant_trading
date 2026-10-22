"""Sprint 19 - Binance Testnet live order round-trip automation script.

Thuc hien round-trip tren Binance Testnet:
1. Credential & Server Time Sync
2. Pre-Trade Snapshot & Lot/Price Validation
3. Live Order Entry (Spot + Perp Delta-Neutral)
4. Mid-Trip Portfolio Reconciliation
5. Live Flatten & Post-Flatten Reconciliation
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.binance_signed_client import (  # noqa: E402
    BinanceHttpError,
    BinanceSignedTestnetClient,
    FUTURES_KEY_INVALID_CODE,
    FUTURES_KEY_INVALID_HINT,
    load_dotenv_testnet,
    quantize_price,
    validate_credential_format,
)
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS  # noqa: E402
from src.exchange_reconciler import reconcile_portfolio_state  # noqa: E402


def build_mock_transport() -> Callable[..., dict[str, Any]]:
    """Tra ve closure stateful cho mock offline transport deterministic."""
    state = {
        "spot_order_seq": 1000,
        "futures_order_seq": 9000,
        "spot_balances": {"USDT": 5000.0, "BTC": 0.0},
        "perp_positions": {"BTCUSDT": 0.0},
        "orders": {},
    }

    def transport(
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        params = dict(params or {})
        method_upper = method.upper()
        now_ms = int(time.time() * 1000)

        # 1. Server time
        if "/api/v3/time" in url:
            return {"serverTime": now_ms + 123}
        if "/fapi/v1/time" in url:
            return {"serverTime": now_ms + 456}

        # 2. Book ticker
        if "/ticker/bookTicker" in url:
            return {
                "symbol": params.get("symbol", "BTCUSDT"),
                "bidPrice": "95990.0",
                "askPrice": "96010.0",
            }

        # 2b. Exchange info (tick/step/min notional)
        if "/exchangeInfo" in url:
            return {
                "symbols": [
                    {
                        "symbol": params.get("symbol", "BTCUSDT"),
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "10.0"},
                            {"filterType": "NOTIONAL", "minNotional": "10.0"},
                        ],
                    }
                ]
            }

        # 3. Account & Balance
        if "/api/v3/account" in url and method_upper == "GET":
            balances = [
                {"asset": k, "free": f"{v:.8f}", "locked": "0.0"}
                for k, v in state["spot_balances"].items()
            ]
            return {"balances": balances}

        if "/fapi/v2/account" in url and method_upper == "GET":
            return {
                "totalWalletBalance": "5000.0",
                "totalUnrealizedProfit": "0.0",
                "totalMaintMargin": "0.0",
                "totalMarginBalance": "5000.0",
                "positions": [],
            }

        if "/fapi/v2/positionRisk" in url and method_upper == "GET":
            pos_amt = state["perp_positions"].get("BTCUSDT", 0.0)
            return [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": f"{pos_amt:.6f}",
                    "entryPrice": "96000.0",
                    "markPrice": "96000.0",
                    "liquidationPrice": "0.0",
                }
            ]

        # 4. Spot order placement (POST /api/v3/order)
        if "/api/v3/order" in url and method_upper == "POST":
            state["spot_order_seq"] += 1
            order_id = state["spot_order_seq"]
            qty = float(params.get("quantity", 0.001))
            price = float(params.get("price", 96000.0))
            side = str(params.get("side", "BUY")).upper()
            if side == "BUY":
                state["spot_balances"]["BTC"] += qty
                state["spot_balances"]["USDT"] = max(
                    0.0, state["spot_balances"]["USDT"] - qty * price
                )
            else:
                state["spot_balances"]["BTC"] = max(
                    0.0, state["spot_balances"]["BTC"] - qty
                )
                state["spot_balances"]["USDT"] += qty * price

            order_data = {
                "orderId": order_id,
                "symbol": params.get("symbol", "BTCUSDT"),
                "status": "FILLED",
                "executedQty": str(qty),
                "cummulativeQuoteQty": str(qty * price),
                "avgPrice": str(price),
                "transactTime": now_ms,
            }
            state["orders"][f"spot_{order_id}"] = order_data
            return order_data

        # 5. Perp order placement (POST /fapi/v1/order)
        if "/fapi/v1/order" in url and method_upper == "POST":
            state["futures_order_seq"] += 1
            order_id = state["futures_order_seq"]
            qty = float(params.get("quantity", 0.001))
            price = float(params.get("price", 96000.0))
            side = str(params.get("side", "SELL")).upper()
            if side == "SELL":
                state["perp_positions"]["BTCUSDT"] -= qty
            else:
                state["perp_positions"]["BTCUSDT"] += qty

            order_data = {
                "orderId": order_id,
                "symbol": params.get("symbol", "BTCUSDT"),
                "status": "FILLED",
                "executedQty": str(qty),
                "cummulativeQuoteQty": str(qty * price),
                "avgPrice": str(price),
                "transactTime": now_ms,
            }
            state["orders"][f"perp_{order_id}"] = order_data
            return order_data

        # 6. Order query (GET)
        if "/api/v3/order" in url and method_upper == "GET":
            order_id = params.get("orderId")
            existing = state["orders"].get(f"spot_{order_id}")
            if existing:
                return existing
            return {
                "orderId": order_id,
                "symbol": params.get("symbol", "BTCUSDT"),
                "status": "FILLED",
                "executedQty": str(params.get("quantity", 0.001)),
                "avgPrice": "96000.0",
            }

        if "/fapi/v1/order" in url and method_upper == "GET":
            order_id = params.get("orderId")
            existing = state["orders"].get(f"perp_{order_id}")
            if existing:
                return existing
            return {
                "orderId": order_id,
                "symbol": params.get("symbol", "BTCUSDT"),
                "status": "FILLED",
                "executedQty": str(params.get("quantity", 0.001)),
                "avgPrice": "96000.0",
            }

        # 7. Cancel order (DELETE)
        if method_upper == "DELETE":
            order_id = params.get("orderId", 0)
            return {"orderId": order_id, "status": "CANCELED"}

        return {"status": "OK"}

    return transport


_REDACT_KEYS: frozenset[str] = frozenset(
    {
        "headers",
        "params",
        "signature",
        "x-mbx-apikey",
        "api_key",
        "api_secret",
        "spot_api_key",
        "spot_api_secret",
        "futures_api_key",
        "futures_api_secret",
    }
)


def _sanitize(obj: Any) -> Any:
    """Loai bo moi truong co the chua khoa API truoc khi ghi log/evidence."""
    if isinstance(obj, dict):
        clean: dict[Any, Any] = {}
        for k, v in obj.items():
            if str(k).strip().lower() in _REDACT_KEYS:
                clean[k] = "[REDACTED]"
            else:
                clean[k] = _sanitize(v)
        return clean
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _scrub_secrets(text: str) -> str:
    """Thay the moi gia tri credential da nap bang [REDACTED] (an toan 2 lop)."""
    try:
        loaded = load_dotenv_testnet(environ={}, extra_paths=())
    except Exception:  # pragma: no cover - thieu file
        return text
    for value in (loaded or {}).values():
        if value and len(value) >= 12 and value in text:
            text = text.replace(value, "[REDACTED]")
    return text


def _append_evidence_and_log(
    result: dict[str, Any],
    evidence_path: str,
    log_path: str,
) -> None:
    """Ghi log JSONL va cap nhat file markdown evidence."""
    ts_iso = datetime.now(timezone.utc).isoformat()
    ev_path = Path(evidence_path)
    ev_path.parent.mkdir(parents=True, exist_ok=True)

    header_needed = not ev_path.exists() or ev_path.stat().st_size == 0
    table_header = (
        "# Binance Testnet Live Roundtrip Evidence\n\n"
        "| Timestamp (UTC) | Symbol | Verdict | Entry Order ID | Exit Order ID | "
        "Exec Qty | Fill Price | Entry Rec | Exit Rec | Spot Offset (ms) | Perp Offset (ms) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    table_row = (
        f"| {ts_iso} | {result.get('symbol')} | {result.get('verdict')} | "
        f"{result.get('entry_order_id')} | {result.get('exit_order_id')} | "
        f"{result.get('executed_qty')} | {result.get('fill_price')} | "
        f"{result.get('entry_reconcile_status')} | {result.get('exit_reconcile_status')} | "
        f"{result.get('server_offsets', {}).get('spot')} | "
        f"{result.get('server_offsets', {}).get('futures')} |\n"
    )

    detail_section = (
        f"\n### Run Detail ({ts_iso})\n"
        f"- **Symbol**: `{result.get('symbol')}`\n"
        f"- **Verdict**: `{result.get('verdict')}`\n"
        f"- **Spot Auth**: `{result.get('spot_auth')}` ({result.get('spot_key_status')})\n"
        f"- **Futures Auth**: `{result.get('futures_auth')}` ({result.get('futures_key_status')})\n"
        f"- **Entry Order**: `{result.get('entry_order_id')}` (Qty: {result.get('executed_qty')}, Price: {result.get('fill_price')})\n"
        f"- **Exit Order**: `{result.get('exit_order_id')}` (Qty: {result.get('exit_executed_qty')})\n"
        f"- **Spot Entry Order ID**: `{result.get('spot_entry_order_id')}`\n"
        f"- **Perp Entry Order ID**: `{result.get('perp_entry_order_id')}`\n"
        f"- **Spot Exit Order ID**: `{result.get('spot_exit_order_id')}`\n"
        f"- **Perp Exit Order ID**: `{result.get('perp_exit_order_id')}`\n"
        f"- **Dual-Venue SYNCED**: `{result.get('dual_venue_synced')}`\n"
        f"- **Entry Reconcile Status**: `{result.get('entry_reconcile_status')}`\n"
        f"- **Exit Reconcile Status**: `{result.get('exit_reconcile_status')}`\n\n"
    )

    content = (table_header if header_needed else "") + table_row + detail_section
    if header_needed:
        with ev_path.open("a", encoding="utf-8") as f:
            f.write(_scrub_secrets(content))
    else:
        # File da co: chen dong bang NGAY SAU dong bang cuoi cung, de bang khong
        # bi cat vun boi cac khoi "Run Detail" cua lan chay truoc.
        existing = ev_path.read_text(encoding="utf-8")
        lines = existing.splitlines(keepends=True)
        last_row = 0
        for idx, line in enumerate(lines):
            if line.lstrip().startswith("|") and not line.lstrip().startswith("|---"):
                last_row = idx
        insert_at = last_row + 1 if last_row else len(lines)
        new_lines = lines[:insert_at] + [table_row, "\n" + detail_section.lstrip("\n")] + lines[insert_at:]
        text = "".join(new_lines)
        if not text.endswith("\n"):
            text += "\n"
        ev_path.write_text(_scrub_secrets(text), encoding="utf-8")

    l_path = Path(log_path)
    l_path.parent.mkdir(parents=True, exist_ok=True)
    log_record = dict(result)
    log_record["timestamp"] = ts_iso
    with l_path.open("a", encoding="utf-8") as f:
        blob = json.dumps(_sanitize(log_record), ensure_ascii=False, default=str)
        f.write(_scrub_secrets(blob) + "\n")


def _build_snapshot(
    client: BinanceSignedTestnetClient,
    symbol: str,
    mid_price: float,
    futures_live: bool,
) -> dict[str, Any]:
    """Lay snapshot doi soat: dual-venue neu Futures OK, else spot-only fallback.

    Futures key khong hop le KHONG duoc phep lam hong viec doi soat Spot - ta van
    doi soat That tren so du Spot va danh dau `snapshot_source` ro rang.
    """
    if futures_live:
        snap = client.fetch_exchange_reconcile_snapshot([symbol], {symbol: mid_price})
        if snap.get("snapshot_healthy"):
            snap["perp_positions_verified"] = True
            snap["snapshot_scope"] = "spot_and_futures"
            # Sprint 20: Binance Testnet cap san ~500 tai san "dust" cho moi tai khoan
            # demo. Doi soat TOAN TAI KHOAN se luon bao ORPHAN_ON_EXCHANGE cho cac tai
            # san khong giao dich -> vo nghia voi muc tieu "dual-venue SYNCED".
            # Vi vay pham vi doi soat o day la SYMBOL DANG GIAO DICH (base + cash),
            # va so tai san ngoai pham vi duoc ghi lai de kiem toan minh bach.
            full_balances = snap.get("spot_balances") or {}
            scoped_balances = _filter_holdings(full_balances, symbol)
            snap["spot_balances"] = scoped_balances
            snap["reconcile_symbol_scope"] = [symbol]
            snap["untracked_spot_assets"] = max(
                0, len(full_balances) - len(scoped_balances)
            )
            return snap
    access = client.check_account_access("spot")
    holdings = access.get("holdings") or {}
    base = str(symbol.split("/")[0]).upper()
    base_qty = float(holdings.get(base, 0.0) or 0.0)
    cash = float(holdings.get("USDT", 0.0) or 0.0)
    return {
        "snapshot_healthy": bool(access.get("authenticated")),
        "snapshot_source": "spot_only_fallback",
        # QUAN TRONG: khong doc duoc Futures -> KHONG duoc coi la da doi soat du.
        # Vi the moi status reconcile tren snapshot nay mang tien to SPOT_ONLY_.
        "snapshot_scope": "spot_only",
        "perp_positions_verified": False,
        "spot_balances": {base: base_qty},
        "spot_cash_usd": cash,
        "perp_wallet_balance_usd": 0.0,
        "perp_unrealized_pnl_usd": 0.0,
        "maint_margin_usd": 0.0,
        "margin_balance_usd": 0.0,
        "exchange_equity_usd": cash + base_qty * float(mid_price),
        "perp_positions": {},
        "perp_positions_detail": [],
    }


def _filter_holdings(holdings: dict[str, Any], symbol: str) -> dict[str, float]:
    """Chi giu lai tai san lien quan (base/quote) de log khong phinh to."""
    base = str(symbol.split("/")[0]).upper()
    quote = "USDT"
    keep: dict[str, float] = {}
    for asset in (base, quote):
        if asset in holdings:
            keep[asset] = float(holdings.get(asset) or 0.0)
    return keep


def _scoped_status(raw_status: str, verified: bool) -> str:
    """Danh dau ro status chi doi soat Spot khi khong doc duoc Futures."""
    if verified:
        return raw_status
    return f"SPOT_ONLY_{raw_status}"


def _validate_client_credentials(
    client: Any, venue: str
) -> dict[str, Any]:
    """Sprint 20: chan doan dinh dang khoa cho 1 venue, an toan voi client gia lap.

    - Uu tien `client.credential_format_report()` (client that).
    - Fallback sang thuoc tinh `<venue>_api_key` / `<venue>_api_secret`.
    - Client stub khong co credential attribute -> tra trang thai `skipped`.
    """
    report_fn = getattr(client, "credential_format_report", None)
    if callable(report_fn):
        try:
            full = report_fn()
            if isinstance(full, dict) and isinstance(full.get(venue), dict):
                return dict(full[venue])
        except Exception:  # pragma: no cover - client la
            pass
    key = getattr(client, f"{venue}_api_key", None)
    secret = getattr(client, f"{venue}_api_secret", None)
    if key is None and secret is None:
        return {
            "venue": venue,
            "ok": True,
            "skipped": True,
            "diagnostics": [],
            "message": f"{venue}_CREDENTIAL_FORMAT_CHECK_SKIPPED_NO_CREDENTIAL_ATTRIBUTE",
        }
    return validate_credential_format(key, secret, venue)


def run_roundtrip(
    *,
    symbol: str = "BTC/USDT:USDT",
    spot_qty: float = 0.001,
    perp_qty: float = 0.001,
    maker_ttl_sec: float = 3.0,
    client: BinanceSignedTestnetClient | None = None,
    write_files: bool = True,
    evidence_path: str = "docs/sprint20_dual_leg_live_evidence.md",
    log_path: str = "logs/sprint20_testnet_roundtrip.jsonl",
) -> dict:
    """Thuc hien tron ven 5 buoc round-trip testnet."""
    steps: list[dict[str, Any]] = []
    events: list[Any] = []

    result: dict[str, Any] = {
        "verdict": "FAIL",
        "symbol": symbol,
        "steps": steps,
        "spot_auth": False,
        "futures_auth": False,
        "spot_key_status": "not_loaded",
        "futures_key_status": "not_loaded",
        "entry_order_id": None,
        "exit_order_id": None,
        # Sprint 20: 4 order id that tren 2 chan (Spot entry/exit, Perp entry/exit).
        "spot_entry_order_id": None,
        "perp_entry_order_id": None,
        "spot_exit_order_id": None,
        "perp_exit_order_id": None,
        "dual_venue_synced": False,
        "credential_validation": {},
        "executed_qty": 0.0,
        "exit_executed_qty": 0.0,
        "fill_price": 0.0,
        "entry_reconcile_status": "NOT_RUN",
        "exit_reconcile_status": "NOT_RUN",
        "server_offsets": {"spot": 0, "futures": 0},
        "events": events,
        "evidence_path": evidence_path,
        "log_path": log_path,
    }

    # Khoi tao client neu chua co
    if client is None:
        try:
            load_dotenv_testnet()
        except Exception:
            pass
        client = BinanceSignedTestnetClient(dry_run=False, recv_window=10000)

    # -------------------------------------------------------------------------
    # Buoc 1: Credential & Time Sync
    # -------------------------------------------------------------------------
    cred_status = client.credential_status()
    result["spot_key_status"] = str(cred_status.get("spot_key", "not_loaded"))
    result["futures_key_status"] = str(cred_status.get("futures_key", "not_loaded"))

    # Sprint 20: kiem dinh DINH DANG khoa 64 ky tu TRUOC khi gui bat ky request mang nao.
    spot_format = _validate_client_credentials(client, "spot")
    futures_format = _validate_client_credentials(client, "futures")
    result["credential_validation"] = {
        "spot": spot_format,
        "futures": futures_format,
        "all_ok": bool(spot_format.get("ok") and futures_format.get("ok")),
    }
    for _venue, _report in (("spot", spot_format), ("futures", futures_format)):
        if not _report.get("ok"):
            # Chi in ma chan doan (do dai + ten venue), KHONG in gia tri khoa.
            print(
                f"CREDENTIAL_FORMAT_BLOCKED: {_report.get('message')}",
                file=sys.stderr,
            )

    time_sync = client.sync_server_time("both")
    result["server_offsets"]["spot"] = int(time_sync.get("spot_offset_ms", 0))
    result["server_offsets"]["futures"] = int(time_sync.get("futures_offset_ms", 0))

    spot_access = client.check_account_access("spot")
    futures_access = client.check_account_access("futures")

    spot_auth = bool(spot_access.get("authenticated"))
    futures_auth = bool(futures_access.get("authenticated"))
    result["spot_auth"] = spot_auth
    result["futures_auth"] = futures_auth

    if futures_access.get("error") == FUTURES_KEY_INVALID_CODE:
        print(FUTURES_KEY_INVALID_HINT, file=sys.stderr)

    step1_ok = spot_auth or futures_auth
    steps.append(
        {
            "step": 1,
            "name": "Credential & Time Sync",
            "ok": step1_ok,
            "detail": {
                "spot_auth": spot_auth,
                "futures_auth": futures_auth,
                "server_sync": time_sync,
                "spot_access": spot_access,
                "futures_access": futures_access,
                "credential_validation": result["credential_validation"],
            },
        }
    )

    if not step1_ok:
        result["verdict"] = "FAIL"
        if write_files:
            _append_evidence_and_log(result, evidence_path, log_path)
        return result

    # -------------------------------------------------------------------------
    # Buoc 2: Pre-Trade Snapshot & Lot/Price
    # -------------------------------------------------------------------------
    step2_ok = False
    mid_price = 0.0
    base_before = 0.0
    perp_before = 0.0
    pre_snapshot: dict[str, Any] = {}
    try:
        ticker = client.fetch_book_ticker(symbol, "spot")
        mid_price = float(ticker.get("mid") or 0.0)
        bid_price = float(ticker.get("bid") or mid_price)
        ask_price = float(ticker.get("ask") or mid_price)
        spec = EXCHANGE_LOT_SPECS.get(symbol, {"step_size": 0.001, "min_notional": 10.0})
        step_size = float(spec.get("step_size", 0.001))
        min_notional = float(spec.get("min_notional", 10.0))

        # Sprint 19: lay tick_size/step_size/min_notional THAT tu exchangeInfo de
        # khong bi -1013 PRICE_FILTER khi dat Post-Only.
        live_filters: dict[str, float] = {}
        try:
            live_filters = client.fetch_symbol_filters(symbol, "spot")
        except Exception as exc:
            live_filters = {"error": str(exc)}  # type: ignore[assignment]
        tick_size = float(live_filters.get("tick_size") or 0.0)
        if tick_size > 0.0:
            step_size = float(live_filters.get("step_size") or step_size)
            min_notional = float(live_filters.get("min_notional") or min_notional)
        maker_buy_price = quantize_price(bid_price, tick_size, "BUY")
        maker_sell_price = quantize_price(ask_price, tick_size, "SELL")
        if maker_buy_price <= 0.0:
            maker_buy_price = mid_price
        if maker_sell_price <= 0.0:
            maker_sell_price = mid_price

        notional = spot_qty * mid_price
        step_count = spot_qty / step_size
        valid_step = abs(step_count - round(step_count)) < 1e-6
        valid_notional = notional >= min_notional
        valid_price = mid_price > 0.0

        # Snapshot TRUOC khi dat lenh: lam baseline cho doi soat (vi tai khoan
        # testnet co san so du, khong the gia dinh vi the ban dau = 0).
        pre_snapshot = (
            _build_snapshot(client, symbol, mid_price, futures_auth) if mid_price > 0.0 else {}
        )
        base_asset = str(symbol.split("/")[0]).upper()
        base_before = float(
            (pre_snapshot.get("spot_balances") or {}).get(base_asset, 0.0) or 0.0
        )
        perp_before = float(
            (pre_snapshot.get("perp_positions") or {}).get(symbol, 0.0) or 0.0
        )

        step2_ok = valid_price and valid_notional and valid_step
        steps.append(
            {
                "step": 2,
                "name": "Pre-Trade Snapshot & Lot/Price",
                "ok": step2_ok,
                "detail": {
                    "mid_price": mid_price,
                    "bid": bid_price,
                    "ask": ask_price,
                    "maker_buy_price": maker_buy_price,
                    "maker_sell_price": maker_sell_price,
                    "tick_size": tick_size,
                    "notional": notional,
                    "min_notional": min_notional,
                    "step_size": step_size,
                    "valid_step": valid_step,
                    "holdings": _filter_holdings(spot_access.get("holdings", {}), symbol),
                    "baseline_spot_qty": base_before,
                    "baseline_perp_qty": perp_before,
                    "snapshot_source": pre_snapshot.get("snapshot_source", "live_dual_venue"),
                    "snapshot_healthy": pre_snapshot.get("snapshot_healthy", False),
                },
            }
        )
    except Exception as exc:
        steps.append(
            {
                "step": 2,
                "name": "Pre-Trade Snapshot & Lot/Price",
                "ok": False,
                "detail": {"error": str(exc)},
            }
        )

    if not step2_ok:
        result["verdict"] = "FAIL"
        if write_files:
            _append_evidence_and_log(result, evidence_path, log_path)
        return result

    # -------------------------------------------------------------------------
    # Buoc 3: Live Order Entry
    # -------------------------------------------------------------------------
    try:
        entry_res = client.execute_live_testnet_order_pair(
            symbol=symbol,
            spot_qty=spot_qty,
            perp_qty=perp_qty,
            spot_price=maker_buy_price,
            direction="open",
            post_only_spot=True,
            spot_use_limit_maker=True,
            perp_post_only=False,
            maker_ttl_sec=maker_ttl_sec,
            require_futures=False,
            perp_available=futures_auth,
        )
    except Exception as exc:
        steps.append(
            {
                "step": 3,
                "name": "Live Order Entry",
                "ok": False,
                "detail": {"error": str(exc)},
            }
        )
        result["verdict"] = "FAIL"
        if write_files:
            _append_evidence_and_log(result, evidence_path, log_path)
        return result

    entry_events = entry_res.get("events", [])
    events.extend(entry_events)

    spot_entry = entry_res.get("spot") or {}
    perp_entry = entry_res.get("perp") or {}
    entry_order_id = spot_entry.get("orderId")
    executed_qty = float(spot_entry.get("executedQty") or 0.0)
    fill_price = float(spot_entry.get("avgPrice") or mid_price)

    result["entry_order_id"] = entry_order_id
    result["spot_entry_order_id"] = entry_order_id
    result["perp_entry_order_id"] = (perp_entry or {}).get("orderId")
    result["executed_qty"] = executed_qty
    result["fill_price"] = fill_price

    entry_status = entry_res.get("status")
    step3_ok = entry_status in ("FILLED", "ROLLED_BACK_SAFE") and executed_qty > 0

    detail3: dict[str, Any] = {
        "status": entry_status,
        "entry_res": entry_res,
        "spot": spot_entry,
        "perp": perp_entry,
    }

    if entry_status in ("ROLLED_BACK_SAFE", "NAKED_LEG_ALERT"):
        detail3["naked_leg_warning"] = True
        steps.append(
            {
                "step": 3,
                "name": "Live Order Entry",
                "ok": step3_ok,
                "detail": detail3,
            }
        )
        result["verdict"] = "PARTIAL"
        if write_files:
            _append_evidence_and_log(result, evidence_path, log_path)
        return result

    steps.append(
        {
            "step": 3,
            "name": "Live Order Entry",
            "ok": step3_ok,
            "detail": detail3,
        }
    )

    if not step3_ok:
        result["verdict"] = "FAIL"
        if write_files:
            _append_evidence_and_log(result, evidence_path, log_path)
        return result

    # -------------------------------------------------------------------------
    # Buoc 4: Mid-Trip Reconciliation
    # -------------------------------------------------------------------------
    try:
        snapshot_mid = _build_snapshot(client, symbol, mid_price, futures_auth)
        local_perp_after_open = perp_before + (-perp_qty if futures_auth else 0.0)
        local_state_mid = {
            "carry_positions": {
                symbol: {
                    "qty": base_before + executed_qty,
                    "leg2_hedged_qty": abs(local_perp_after_open),
                }
            },
            "current_equity": float(snapshot_mid.get("exchange_equity_usd", 0.0)),
            "overlay_positions": {},
        }
        reconcile_mid = reconcile_portfolio_state(
            local_state_mid, snapshot_mid, {symbol: mid_price}, EXCHANGE_LOT_SPECS
        )
        entry_verified = bool(snapshot_mid.get("perp_positions_verified", False))
        entry_rec_status = _scoped_status(
            str(reconcile_mid.get("status") or ""), entry_verified
        )
        result["entry_reconcile_status"] = entry_rec_status

        snapshot_healthy = bool(snapshot_mid.get("snapshot_healthy", False))
        step4_ok = snapshot_healthy
        steps.append(
            {
                "step": 4,
                "name": "Mid-Trip Reconciliation",
                "ok": step4_ok,
                "detail": {
                    "snapshot_healthy": snapshot_healthy,
                    "futures_positions_verified": entry_verified,
                    "snapshot_scope": snapshot_mid.get("snapshot_scope", "unknown"),
                    "reconcile_status": entry_rec_status,
                    "summary": reconcile_mid.get("summary", ""),
                },
            }
        )
    except Exception as exc:
        steps.append(
            {
                "step": 4,
                "name": "Mid-Trip Reconciliation",
                "ok": False,
                "detail": {"error": str(exc)},
            }
        )

    # -------------------------------------------------------------------------
    # Buoc 5: Live Flatten & Post-Flatten Reconciliation
    # -------------------------------------------------------------------------
    exit_ok = False
    try:
        exit_res = client.execute_live_testnet_order_pair(
            symbol=symbol,
            spot_qty=executed_qty,
            perp_qty=executed_qty,
            spot_price=maker_sell_price,
            direction="close",
            post_only_spot=True,
            spot_use_limit_maker=True,
            perp_post_only=False,
            maker_ttl_sec=maker_ttl_sec,
            require_futures=False,
            perp_available=futures_auth,
        )
        exit_events = exit_res.get("events", [])
        events.extend(exit_events)

        spot_exit = exit_res.get("spot") or {}
        exit_order_id = spot_exit.get("orderId")
        exit_executed_qty = float(spot_exit.get("executedQty") or 0.0)

        result["exit_order_id"] = exit_order_id
        result["spot_exit_order_id"] = exit_order_id
        result["perp_exit_order_id"] = (exit_res.get("perp") or {}).get("orderId")
        result["exit_executed_qty"] = exit_executed_qty

        exit_status = exit_res.get("status")
        exit_filled = exit_status == "FILLED" or exit_executed_qty > 0

        snapshot_post = _build_snapshot(client, symbol, mid_price, futures_auth)
        local_state_post = {
            "carry_positions": {
                symbol: {
                    "qty": base_before,
                    "leg2_hedged_qty": abs(perp_before),
                }
            },
            "current_equity": float(snapshot_post.get("exchange_equity_usd", 0.0)),
            "overlay_positions": {},
        }
        reconcile_post = reconcile_portfolio_state(
            local_state_post, snapshot_post, {symbol: mid_price}, EXCHANGE_LOT_SPECS
        )
        exit_verified = bool(snapshot_post.get("perp_positions_verified", False))
        result["exit_futures_positions_verified"] = exit_verified
        exit_rec_status = _scoped_status(
            str(reconcile_post.get("status") or ""), exit_verified
        )
        result["exit_reconcile_status"] = exit_rec_status

        exit_ok = bool(exit_filled)
        steps.append(
            {
                "step": 5,
                "name": "Live Flatten & Post-Flatten Reconciliation",
                "ok": exit_ok,
                "detail": {
                    "exit_status": exit_status,
                    "exit_order_id": exit_order_id,
                    "exit_executed_qty": exit_executed_qty,
                    "futures_positions_verified": exit_verified,
                    "snapshot_scope": snapshot_post.get("snapshot_scope", "unknown"),
                    "reconcile_status": exit_rec_status,
                    "summary": reconcile_post.get("summary", ""),
                },
            }
        )
    except Exception as exc:
        steps.append(
            {
                "step": 5,
                "name": "Live Flatten & Post-Flatten Reconciliation",
                "ok": False,
                "detail": {"error": str(exc)},
            }
        )

    # -------------------------------------------------------------------------
    # Tinh toan Verdict cuoi cung
    # -------------------------------------------------------------------------
    all_steps_ok = len(steps) == 5 and all(s.get("ok", False) for s in steps)
    has_orders = result["entry_order_id"] is not None and result["exit_order_id"] is not None
    # Sprint 20: doi hoi DU 4 order id that (Spot entry/exit + Perp entry/exit).
    dual_leg_orders = all(
        result.get(key) is not None
        for key in (
            "spot_entry_order_id",
            "perp_entry_order_id",
            "spot_exit_order_id",
            "perp_exit_order_id",
        )
    )
    result["dual_leg_order_ids_complete"] = bool(dual_leg_orders)
    futures_verified = bool(result.get("exit_futures_positions_verified", False))
    synced = result["exit_reconcile_status"] == "SYNCED" and futures_verified
    result["dual_venue_synced"] = bool(synced)
    spot_only_synced = result["exit_reconcile_status"] == "SPOT_ONLY_SYNCED"
    simulated_futures = (
        not futures_auth
        or perp_entry.get("status") in ("SIMULATED_NO_FUTURES_KEY", "SIMULATED_SIGNED_ONLY")
    )
    result["futures_positions_verified"] = futures_verified
    if not futures_verified:
        # E.1: snapshot chi doc duoc Spot -> KHONG duoc cap verdict PASS day du,
        # du khoa Futures co that (co the dang co vi the mo coi chua doi soat).
        result["verdict_scope"] = "spot_only"
        result["futures_verification_warning"] = (
            "Futures positions were NOT verified - spot-only reconciliation"
        )

    if all_steps_ok and has_orders and synced and not simulated_futures and dual_leg_orders:
        result["verdict"] = "PASS"
        result["verdict_scope"] = "dual_leg_live_verified"
    elif all_steps_ok and has_orders and (synced or spot_only_synced) and (
        simulated_futures or not futures_verified
    ):
        # Spot live round-trip tron ven, chan Futures chi mo phong co ky
        # hoac chua doi soat duoc vi the Futures.
        result["verdict"] = "PASS_SPOT_ONLY"
    elif has_orders and (executed_qty > 0 and exit_ok):
        result["verdict"] = "PARTIAL"
    else:
        result["verdict"] = "FAIL"

    if write_files:
        _append_evidence_and_log(result, evidence_path, log_path)

    return result


def main(argv: list[str] | None = None) -> int:
    """Entrypoint CLI cho script round-trip live testnet."""
    parser = argparse.ArgumentParser(
        description="Sprint 19 - Binance Testnet live order round-trip"
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading symbol")
    parser.add_argument("--spot-qty", type=float, default=0.001, help="Spot quantity")
    parser.add_argument("--perp-qty", type=float, default=0.001, help="Perp quantity")
    parser.add_argument("--maker-ttl-sec", type=float, default=3.0, help="Maker TTL in sec")
    parser.add_argument(
        "--mock-transport",
        action="store_true",
        help="Run fully offline with deterministic mock transport",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run sign only")
    parser.add_argument(
        "--evidence-doc",
        default="docs/sprint20_dual_leg_live_evidence.md",
        help="Path to evidence markdown document",
    )
    parser.add_argument(
        "--log-jsonl",
        default="logs/sprint20_testnet_roundtrip.jsonl",
        help="Path to jsonl log file",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write evidence or log files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON summary line",
    )

    args = parser.parse_args(argv)

    try:
        client: BinanceSignedTestnetClient | None = None
        if args.mock_transport:
            client = BinanceSignedTestnetClient(
                dry_run=False,
                recv_window=10000,
                spot_api_key="mock-spot-key-0001",
                spot_api_secret="mock-spot-secret",
                futures_api_key="mock-futures-key-0001",
                futures_api_secret="mock-futures-secret",
                autoload_dotenv=False,
                transport=build_mock_transport(),
            )
        elif args.dry_run:
            client = BinanceSignedTestnetClient(dry_run=True, recv_window=10000)

        result = run_roundtrip(
            symbol=args.symbol,
            spot_qty=args.spot_qty,
            perp_qty=args.perp_qty,
            maker_ttl_sec=args.maker_ttl_sec,
            client=client,
            write_files=not args.no_write,
            evidence_path=args.evidence_doc,
            log_path=args.log_jsonl,
        )

        if args.json:
            print(json.dumps(result, ensure_ascii=False, default=str))
        else:
            print(f"Verdict: {result.get('verdict')}")
            print(f"Spot Auth: {result.get('spot_auth')}, Futures Auth: {result.get('futures_auth')}")
            print(f"Entry Order: {result.get('entry_order_id')}, Exit Order: {result.get('exit_order_id')}")
            print(f"Entry Rec: {result.get('entry_reconcile_status')}, Exit Rec: {result.get('exit_reconcile_status')}")

        verdict = result.get("verdict")
        if verdict in ("PASS", "PASS_SPOT_ONLY"):
            return 0
        if verdict == "PARTIAL":
            return 1
        return 2

    except Exception as exc:
        err_dict = {"verdict": "FAIL", "error": str(exc)}
        print(json.dumps(err_dict, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
