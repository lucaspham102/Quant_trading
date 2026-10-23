"""Sprint 21 - Binance Testnet multi-cycle burn-in supervisor (3 cycles).

Chay lien tiep 3 chu ky tren Binance Spot Testnet + USDⓈ-M Futures Testnet:

- Chu ky 1 (Bootstrap & Reconcile): nap `.env.testnet`, kiem dinh khoa 64/64,
  dong bo gio may chu 2 san, kiem tra cong 720h warmup va xac nhan toan bo
  ``LIQUID_8_SYMBOLS`` dat ``SYNCED``.
- Chu ky 2 (Live Dual-Leg Execution & Mid-Cycle Audit): mo cap vi the Delta-Neutral
  that bang ``Smart Inside-Spread Pegging``, kiem chung ``verify_lot_neutrality_invariant``
  va doi soat trang thai san dat ``Dual-Venue SYNCED``.
- Chu ky 3 (Clean Flatten & Zero-Residual Verification): dong phang ca 2 chan ve
  ``0.0``, doi soat lai toan bo 8 symbol dat ``SYNCED`` (``equity_gap == 0.0``).

Ket qua: ``logs/sprint21_multi_cycle_burnin.jsonl`` (da scrub secret) va
``docs/sprint21_burnin_evidence.md``.

Bao mat: KHONG BAO GIO in/log gia tri khoa API. Moi output deu qua ``_scrub_secrets``.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.binance_signed_client import (  # noqa: E402
    ENV_API_KEY,
    ENV_API_SECRET,
    ENV_FUTURES_API_KEY,
    ENV_FUTURES_API_SECRET,
    ENV_SPOT_API_KEY,
    ENV_SPOT_API_SECRET,
    BinanceSignedTestnetClient,
    compute_maker_limit_price,
    load_dotenv_testnet,
    validate_credential_format,
)
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS, LIQUID_8_SYMBOLS  # noqa: E402
from src.exchange_reconciler import SYNCED, reconcile_portfolio_state  # noqa: E402
from src.quant_invariants import verify_lot_neutrality_invariant  # noqa: E402

UNIVERSE_SYMBOLS: list[str] = list(LIQUID_8_SYMBOLS)
WARMUP_REQUIRED_HOURS: float = 720.0
DEFAULT_EVIDENCE_PATH: str = "docs/sprint21_burnin_evidence.md"
DEFAULT_LOG_PATH: str = "logs/sprint21_multi_cycle_burnin.jsonl"

_REDACT_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "api_secret",
        "apikey",
        "secretkey",
        "secret_key",
        "x-mbx-apikey",
        "spot_api_key",
        "spot_api_secret",
        "futures_api_key",
        "futures_api_secret",
        "signature",
        "signature_prefix",
    }
)

_SIGNATURE_RE = re.compile(
    r"""((?:signature|hmac)["']?\s*[:=]\s*["']?)[0-9a-fA-F]{16,}""",
    re.IGNORECASE,
)

_CREDENTIAL_ENV_KEYS: tuple[str, ...] = (
    ENV_API_KEY,
    ENV_API_SECRET,
    ENV_SPOT_API_KEY,
    ENV_SPOT_API_SECRET,
    ENV_FUTURES_API_KEY,
    ENV_FUTURES_API_SECRET,
)


def _secret_values(client: Any = None) -> list[str]:
    """Thu thap MOI gia tri credential da nap (chi de scrub, khong bao gio in).

    Bao gom ca khoa cua ``client`` duoc truyen tu ngoai (khong chi bien moi truong)
    de khong bao gio ghi khoa that vao log/evidence.
    """
    values: set[str] = set()
    try:
        loaded = load_dotenv_testnet(environ={}, extra_paths=())
    except Exception:  # pragma: no cover - thieu file
        loaded = {}
    import os

    for key in _CREDENTIAL_ENV_KEYS:
        for value in (loaded.get(key), os.environ.get(key)):
            if value and len(str(value)) >= 8:
                values.add(str(value))
    if client is not None:
        for venue in ("spot", "futures"):
            for attr in (f"{venue}_api_key", f"{venue}_api_secret", "api_key", "api_secret"):
                value = getattr(client, attr, None)
                if value and len(str(value)) >= 8:
                    values.add(str(value))
    return sorted(values, key=len, reverse=True)


def _scrub_secrets(text: str, secrets: Iterable[str] | None = None) -> str:
    """Thay moi gia tri credential bang ``[REDACTED]`` truoc khi ghi file/in ra."""
    if secrets is None:
        secrets = _secret_values()
    for value in secrets:
        if value and len(value) >= 8 and value in text:
            text = text.replace(value, "[REDACTED]")
    # Chu ky HMAC trong query string khong phai credential nhung van la vat lieu
    # nhay cam (audit MEDIUM) -> scrub luon.
    return _SIGNATURE_RE.sub(r"\1[REDACTED]", text)


def _sanitize(obj: Any) -> Any:
    """Loai bo truong co the chua khoa API truoc khi ghi log JSONL."""
    if isinstance(obj, dict):
        clean: dict[Any, Any] = {}
        for key, value in obj.items():
            if str(key).strip().lower() in _REDACT_KEYS:
                clean[key] = "[REDACTED]"
            else:
                clean[key] = _sanitize(value)
        return clean
    if isinstance(obj, (list, tuple)):
        return [_sanitize(item) for item in obj]
    return obj


def verify_warmup_gate(
    warmup_hours: float = WARMUP_REQUIRED_HOURS,
    required_hours: float = WARMUP_REQUIRED_HOURS,
) -> dict[str, Any]:
    """Cong 720h warmup: chuoi du lieu phai phu du toi thieu ``required_hours``."""
    available = float(warmup_hours or 0.0)
    required = float(required_hours)
    return {
        "gate": f"{int(required)}h",
        "warmup_hours": available,
        "required_hours": required,
        "ok": available >= required,
    }


def scope_snapshot_to_universe(
    snapshot: dict[str, Any],
    universe: Iterable[str] = UNIVERSE_SYMBOLS,
) -> dict[str, Any]:
    """Gioi han snapshot doi soat dung ro ``universe`` (tai san faucet tach rieng).

    Testnet cap san hang tram tai san "dust": doi soat toan tai khoan se luon bao
    ORPHAN_ON_EXCHANGE. Vi vay pham vi doi soat la ro giao dich that, con tai san
    ngoai ro duoc ghi lai de kiem toan minh bach.
    """
    universe_list = [str(s) for s in universe]
    universe_bases = {s.split("/")[0].upper() for s in universe_list if "/" in s}
    balances = snapshot.get("spot_balances") or {}
    scoped: dict[str, float] = {}
    untracked: dict[str, float] = {}
    for asset, qty in balances.items():
        name = str(asset).split("/")[0].strip().upper()
        if name in universe_bases:
            scoped[asset] = float(qty or 0.0)
        elif float(qty or 0.0) > 0.0:
            untracked[name] = float(qty or 0.0)
    scoped_snapshot = dict(snapshot)
    scoped_snapshot["spot_balances"] = scoped
    scoped_snapshot["untracked_spot_assets"] = untracked
    scoped_snapshot["reconcile_symbol_scope"] = universe_list
    return scoped_snapshot


def _reconcile_symbols(
    local_state: dict[str, Any],
    snapshot: dict[str, Any],
    prices: dict[str, float],
    universe: Iterable[str] = UNIVERSE_SYMBOLS,
) -> dict[str, Any]:
    """Doi soat toan ro (mac dinh 8 symbol) o che do ``testnet``."""
    universe_list = [str(s) for s in universe]
    scoped_snapshot = scope_snapshot_to_universe(snapshot, universe_list)
    report = reconcile_portfolio_state(
        local_state,
        scoped_snapshot,
        prices,
        EXCHANGE_LOT_SPECS,
        universe_symbols=universe_list,
        environment="testnet",
    )
    statuses = {str(item.get("symbol")): str(item.get("status")) for item in report.get("items", [])}
    report["symbol_statuses"] = statuses
    report["universe_size"] = len(universe_list)
    report["all_synced"] = bool(
        len(statuses) == len(universe_list) and all(v == SYNCED for v in statuses.values())
    )
    return report


def _empty_local_state(equity: float) -> dict[str, Any]:
    return {"carry_positions": {}, "overlay_positions": {}, "current_equity": float(equity)}


def _clone_baseline(
    baseline: dict[str, Any] | None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sao chep sau baseline cua Chu ky 1 de doi soat delta (khong mutate goc)."""
    state: dict[str, Any] = copy.deepcopy(baseline) if isinstance(baseline, dict) else {}
    state.setdefault("carry_positions", {})
    state.setdefault("overlay_positions", {})
    if snapshot is not None:
        state["current_equity"] = float(snapshot.get("exchange_equity_usd") or 0.0)
    else:
        state.setdefault("current_equity", 0.0)
    return state


def _bootstrap_local_state(
    snapshot: dict[str, Any],
    universe: Iterable[str] = UNIVERSE_SYMBOLS,
) -> dict[str, Any]:
    """Chu ky 1: adopt so du hien co tren san lam so sach ban dau (baseline).

    Testnet cap san faucet cho ca ro 8 symbol tu truoc. Neu gia dinh local = 0 thi
    reconciler se bao ORPHAN gia tao. Bootstrap ghi nhan dung so du that lam diem
    xuat phat, sau do moi chu ky tiep theo doi soat delta tren nen baseline nay.
    """
    balances = snapshot.get("spot_balances") or {}
    perp = snapshot.get("perp_positions") or {}
    carry: dict[str, dict[str, float]] = {}
    for sym in universe:
        base_asset = str(sym).split("/")[0].upper()
        spot_qty = float(balances.get(base_asset, 0.0) or 0.0)
        perp_qty = float(perp.get(str(sym), 0.0) or 0.0)
        # Reconciler tinh local_perp = -leg2_hedged_qty -> giu nguyen dau vi the.
        carry[str(sym)] = {"qty": spot_qty, "leg2_hedged_qty": -perp_qty}
    return {
        "carry_positions": carry,
        "overlay_positions": {},
        "current_equity": float(snapshot.get("exchange_equity_usd") or 0.0),
    }



def _prices_for(universe: Iterable[str], mid_price: float, symbol: str) -> dict[str, float]:
    """Gia tham chieu: symbol giao dich dung mid, cac ma khac dung cung mid (danh gia ro)."""
    prices = {str(s): float(mid_price) for s in universe}
    prices[str(symbol)] = float(mid_price)
    return prices


def _cycle1_bootstrap(
    client: Any,
    *,
    symbol: str,
    universe: list[str],
    warmup_hours: float,
) -> dict[str, Any]:
    """Chu ky 1: credential + time sync + warmup gate + doi soat toan ro."""
    detail: dict[str, Any] = {}
    # Mock transport (offline/deterministic) dung khoa gia nen KHONG the dat 64/64
    # -> danh dau `skipped` thay vi chan burn-in offline.
    is_mock_transport = getattr(client, "transport", None) is not None
    credential_ok = True
    format_report: dict[str, Any] = {}
    report_fn = getattr(client, "credential_format_report", None)
    if callable(report_fn):
        try:
            format_report = report_fn() or {}
        except Exception as exc:  # pragma: no cover - client la
            format_report = {"error": str(exc)}
    for venue in ("spot", "futures"):
        venue_report = format_report.get(venue) if isinstance(format_report, dict) else None
        if not venue_report:
            key = getattr(client, f"{venue}_api_key", None)
            secret = getattr(client, f"{venue}_api_secret", None)
            venue_report = validate_credential_format(key, secret, venue)
        if not is_mock_transport and not venue_report.get("ok"):
            credential_ok = False
    detail["credential_check"] = "skipped_mock_transport" if is_mock_transport else "enforced_64_64"
    detail["credentials_ok"] = bool(credential_ok)
    detail["credential_format"] = {
        venue: {
            "ok": bool((format_report.get(venue) or {}).get("ok", True)),
            "key_length": (format_report.get(venue) or {}).get("key_length"),
            "secret_length": (format_report.get(venue) or {}).get("secret_length"),
        }
        for venue in ("spot", "futures")
    }

    time_sync = client.sync_server_time("both") if hasattr(client, "sync_server_time") else {}
    detail["server_time_offsets_ms"] = {
        "spot": int(time_sync.get("spot_offset_ms", 0) or 0),
        "futures": int(time_sync.get("futures_offset_ms", 0) or 0),
    }

    warmup = verify_warmup_gate(warmup_hours, WARMUP_REQUIRED_HOURS)
    detail["warmup_gate"] = warmup

    ticker = client.fetch_book_ticker(symbol, "spot")
    mid_price = float(ticker.get("mid") or ticker.get("bid") or ticker.get("ask") or 0.0)
    prices = _prices_for(universe, mid_price, symbol)

    snapshot = client.fetch_exchange_reconcile_snapshot(
        [symbol], prices, universe_symbols=list(universe)
    )
    equity = float(snapshot.get("exchange_equity_usd") or 0.0)
    baseline_state = _bootstrap_local_state(snapshot, universe)
    report = _reconcile_symbols(baseline_state, snapshot, prices, universe)

    detail["snapshot_healthy"] = bool(snapshot.get("snapshot_healthy", False))
    detail["mid_price"] = mid_price
    detail["reconcile_status"] = report.get("status")
    detail["all_8_symbols_synced"] = bool(report.get("all_synced"))
    detail["symbol_statuses"] = report.get("symbol_statuses")
    detail["equity_gap_usd"] = report.get("equity_gap_usd")
    detail["untracked_spot_count"] = len(snapshot.get("untracked_spot_assets") or {})
    detail["bootstrap_baseline_symbols"] = len(baseline_state["carry_positions"])
    detail["bootstrap_equity_usd"] = equity
    spot_cash = float(snapshot.get("spot_cash_usd", 0.0) or 0.0)
    fut_wallet = float(snapshot.get("perp_wallet_balance_usd", 0.0) or 0.0)
    fut_upnl = float(snapshot.get("perp_unrealized_pnl_usd", 0.0) or 0.0)
    detail["pure_cash_and_margin_usd"] = float(spot_cash + fut_wallet + fut_upnl)
    recon_eq = report.get("reconcile_equity_usd")
    if recon_eq is not None:
        detail["reconcile_equity_usd"] = float(recon_eq)

    ok = bool(
        credential_ok
        and warmup.get("ok")
        and snapshot.get("snapshot_healthy")
        and report.get("all_synced")
        and mid_price > 0.0
    )
    return {
        "cycle": 1,
        "name": "Bootstrap & Reconcile",
        "ok": ok,
        "detail": detail,
        "_snapshot": snapshot,
        "_prices": prices,
        "_baseline_state": baseline_state,
        "_initial_equity": equity,
        "_mid_price": mid_price,
    }


def _cycle2_execution(
    client: Any,
    cycle1: dict[str, Any],
    *,
    symbol: str,
    universe: list[str],
    spot_qty: float,
    perp_qty: float,
    maker_ttl_sec: float,
) -> dict[str, Any]:
    """Chu ky 2: mo cap Delta-Neutral that + kiem chung lot neutrality + mid audit."""
    detail: dict[str, Any] = {}
    snapshot_pre = cycle1.get("_snapshot") or {}
    prices = cycle1.get("_prices") or {}
    mid_price = float(cycle1.get("_mid_price") or 0.0)

    base_asset = str(symbol.split("/")[0]).upper()
    base_before = float((snapshot_pre.get("spot_balances") or {}).get(base_asset, 0.0) or 0.0)
    perp_before = float((snapshot_pre.get("perp_positions") or {}).get(symbol, 0.0) or 0.0)
    detail["baseline_spot_qty"] = base_before
    detail["baseline_perp_qty"] = perp_before

    ticker = client.fetch_book_ticker(symbol, "spot")
    best_bid = float(ticker.get("bid") or mid_price)
    best_ask = float(ticker.get("ask") or mid_price)
    filters = client.fetch_symbol_filters(symbol, "spot")
    tick_size = float(filters.get("tick_size") or 0.0)

    pricing_buy = compute_maker_limit_price(
        "BUY", best_bid, best_ask, tick_size, improve_inside_spread=True
    )
    detail["best_bid"] = best_bid
    detail["best_ask"] = best_ask
    detail["tick_size"] = tick_size
    detail["maker_pricing"] = pricing_buy

    buy_price = float(pricing_buy.get("price") or 0.0)
    # Audit fix: price == 0.0 nghia la KHONG co gia maker an toan -> phai MARKET.
    # (Khong duoc dung `or best_bid` vi 0.0 la falsy -> vo hieu hoa co che bao ve.)
    maker_buy = buy_price > 0.0
    detail["maker_buy_enabled"] = maker_buy

    entry = client.execute_live_testnet_order_pair(
        symbol=symbol,
        spot_qty=float(spot_qty),
        perp_qty=float(perp_qty),
        spot_price=buy_price if maker_buy else best_ask,
        direction="open",
        post_only_spot=maker_buy,
        spot_use_limit_maker=maker_buy,
        perp_post_only=False,
        maker_ttl_sec=maker_ttl_sec,
        require_futures=False,
    )
    spot_entry = entry.get("spot") or {}
    perp_entry = entry.get("perp") or {}
    executed_qty = float(spot_entry.get("executedQty") or 0.0)
    perp_executed_qty = float(perp_entry.get("executedQty") or 0.0)
    fill_price = float(spot_entry.get("avgPrice") or mid_price)

    detail["entry_status"] = entry.get("status")
    detail["spot_entry_order_id"] = spot_entry.get("orderId")
    detail["perp_entry_order_id"] = perp_entry.get("orderId")
    detail["executed_qty"] = executed_qty
    detail["perp_executed_qty"] = perp_executed_qty
    detail["fill_price"] = fill_price

    neutrality = verify_lot_neutrality_invariant(
        leg_a_usd=executed_qty * fill_price,
        leg_b_usd=perp_executed_qty * (fill_price or mid_price),
        max_imbalance_usd=max(10.0, 0.02 * max(executed_qty * fill_price, 1.0)),
        raise_on_fail=False,
    )
    detail["lot_neutrality"] = neutrality
    detail["lot_neutral_ok"] = bool(
        neutrality.get("passed", neutrality.get("ok", False))
    )

    futures_auth = bool(getattr(client, "has_futures_credentials", lambda: False)())
    snapshot_mid = client.fetch_exchange_reconcile_snapshot(
        [symbol], prices, universe_symbols=list(universe)
    )
    local_state_mid = _clone_baseline(cycle1.get("_baseline_state"), snapshot_mid)
    traded = local_state_mid["carry_positions"].setdefault(
        symbol, {"qty": 0.0, "leg2_hedged_qty": 0.0}
    )
    traded["qty"] = base_before + executed_qty
    # Reconciler: local_perp = -leg2_hedged_qty -> giu nguyen dau vi the thuc te.
    traded["leg2_hedged_qty"] = -(perp_before - perp_executed_qty)
    local_state_mid["current_equity"] = float(snapshot_mid.get("exchange_equity_usd") or 0.0)
    report_mid = _reconcile_symbols(local_state_mid, snapshot_mid, prices, universe)

    detail["mid_reconcile_status"] = report_mid.get("status")
    detail["mid_all_8_symbols_synced"] = bool(report_mid.get("all_synced"))
    detail["mid_symbol_statuses"] = report_mid.get("symbol_statuses")
    detail["futures_positions_verified"] = bool(snapshot_mid.get("snapshot_healthy")) and bool(
        futures_auth
    ) and perp_executed_qty > 0.0
    # Sprint 21: do luong drift van that qua cac chu ky (khong re-anchor im lang).
    initial_equity = float(cycle1.get("_initial_equity") or 0.0)
    detail["equity_drift_vs_initial_usd"] = float(
        (snapshot_mid.get("exchange_equity_usd") or 0.0)
    ) - initial_equity

    dry_run = bool(getattr(client, "dry_run", False))
    entry_ok_status = ("FILLED", "DRY_RUN") if dry_run else ("FILLED",)
    ok = bool(
        entry.get("status") in entry_ok_status
        and executed_qty > 0.0
        and detail["lot_neutral_ok"]
        # Audit fix: chan Futures phai thuc su mo duoc vi the, khong duoc "pass am tham"
        # khi chi co chan Spot (key Futures thieu/sai).
        and detail["futures_positions_verified"]
        and report_mid.get("all_synced")
    )
    return {
        "cycle": 2,
        "name": "Live Dual-Leg Execution & Mid-Cycle Audit",
        "ok": ok,
        "detail": detail,
        "_executed_qty": executed_qty,
        "_fill_price": fill_price,
        "_base_before": base_before,
        "_perp_before": perp_before,
        "_prices": prices,
        "_baseline_state": cycle1.get("_baseline_state"),
        "_initial_equity": cycle1.get("_initial_equity"),
    }


def _cycle3_flatten(
    client: Any,
    cycle2: dict[str, Any],
    *,
    symbol: str,
    universe: list[str],
    maker_ttl_sec: float,
) -> dict[str, Any]:
    """Chu ky 3: dong phang ca 2 chan + xac minh zero residual tren toan ro."""
    detail: dict[str, Any] = {}
    prices = cycle2.get("_prices") or {}
    executed_qty = float(cycle2.get("_executed_qty") or 0.0)
    fill_price = float(cycle2.get("_fill_price") or 0.0)
    base_before = float(cycle2.get("_base_before") or 0.0)
    perp_before = float(cycle2.get("_perp_before") or 0.0)

    ticker = client.fetch_book_ticker(symbol, "spot")
    best_bid = float(ticker.get("bid") or fill_price)
    best_ask = float(ticker.get("ask") or fill_price)
    filters = client.fetch_symbol_filters(symbol, "spot")
    tick_size = float(filters.get("tick_size") or 0.0)
    pricing_sell = compute_maker_limit_price(
        "SELL", best_bid, best_ask, tick_size, improve_inside_spread=True
    )
    detail["maker_pricing"] = pricing_sell
    sell_price = float(pricing_sell.get("price") or 0.0)
    maker_sell = sell_price > 0.0
    detail["maker_sell_enabled"] = maker_sell

    exit_res = client.execute_live_testnet_order_pair(
        symbol=symbol,
        spot_qty=executed_qty,
        perp_qty=executed_qty,
        spot_price=sell_price if maker_sell else best_bid,
        direction="close",
        post_only_spot=maker_sell,
        spot_use_limit_maker=maker_sell,
        perp_post_only=False,
        maker_ttl_sec=maker_ttl_sec,
        require_futures=False,
    )
    spot_exit = exit_res.get("spot") or {}
    perp_exit = exit_res.get("perp") or {}
    exit_executed = float(spot_exit.get("executedQty") or 0.0)
    exit_perp_executed = float(perp_exit.get("executedQty") or 0.0)

    detail["exit_status"] = exit_res.get("status")
    detail["spot_exit_order_id"] = spot_exit.get("orderId")
    detail["perp_exit_order_id"] = perp_exit.get("orderId")
    detail["exit_executed_qty"] = exit_executed
    detail["exit_perp_executed_qty"] = exit_perp_executed

    snapshot_post = client.fetch_exchange_reconcile_snapshot(
        [symbol], prices, universe_symbols=list(universe)
    )
    local_state_post = _clone_baseline(cycle2.get("_baseline_state"), snapshot_post)
    report_post = _reconcile_symbols(local_state_post, snapshot_post, prices, universe)

    base_after = float((snapshot_post.get("spot_balances") or {}).get(symbol.split("/")[0].upper(), 0.0) or 0.0)
    perp_after = float((snapshot_post.get("perp_positions") or {}).get(symbol, 0.0) or 0.0)
    equity_gap = float(report_post.get("equity_gap_usd") or 0.0)
    spot_residual = base_after - base_before
    perp_residual = perp_after - perp_before

    detail["post_reconcile_status"] = report_post.get("status")
    detail["post_all_8_symbols_synced"] = bool(report_post.get("all_synced"))
    detail["post_symbol_statuses"] = report_post.get("symbol_statuses")
    detail["baseline_spot_qty"] = base_before
    detail["baseline_perp_qty"] = perp_before
    detail["spot_qty_after"] = base_after
    detail["perp_qty_after"] = perp_after
    detail["spot_residual_qty"] = spot_residual
    detail["perp_residual_qty"] = perp_residual
    detail["equity_gap_usd"] = equity_gap
    # Sai so cho phep: 0.2% khoi luong da giao dich (phi Spot co the tru vao base
    # asset tren mot so cap), van du chat de phat hien vi the treo that su.
    residual_tolerance = max(1e-9, abs(executed_qty) * 0.002)
    detail["residual_tolerance_qty"] = residual_tolerance
    detail["zero_residual"] = bool(
        abs(perp_residual) < 1e-9 and abs(spot_residual) <= residual_tolerance
    )
    detail["equity_gap_zero"] = bool(abs(equity_gap) < 1e-9)
    detail["equity_drift_vs_initial_usd"] = float(
        (snapshot_post.get("exchange_equity_usd") or 0.0)
    ) - float(cycle2.get("_initial_equity") or 0.0)
    # Audit fix: equity_gap luon 0 vi so sach duoc re-anchor theo baseline -> phai
    # kiem tra them drift THAT so voi von ban dau (chi cho phep sai so phi giao dich).
    drift_tolerance_usd = max(2.0, 0.01 * abs(executed_qty * (fill_price or 1.0)))
    detail["drift_tolerance_usd"] = drift_tolerance_usd
    detail["equity_drift_ok"] = bool(
        abs(detail["equity_drift_vs_initial_usd"]) <= drift_tolerance_usd
    )

    dry_run = bool(getattr(client, "dry_run", False))
    exit_ok_status = ("FILLED", "DRY_RUN") if dry_run else ("FILLED",)
    ok = bool(
        # Audit fix: ROLLED_BACK_SAFE nghia la CHUA dong duoc vi the -> khong duoc PASS.
        exit_res.get("status") in exit_ok_status
        and exit_executed > 0.0
        and exit_perp_executed > 0.0
        and report_post.get("all_synced")
        and detail["equity_gap_zero"]
        and detail["zero_residual"]
        and detail["equity_drift_ok"]
    )
    return {
        "cycle": 3,
        "name": "Clean Flatten & Zero-Residual Verification",
        "ok": ok,
        "detail": detail,
    }


def _public_cycles(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bo cac khoa tam ``_snapshot/_prices`` truoc khi ghi log."""
    cleaned: list[dict[str, Any]] = []
    for cycle in cycles:
        cleaned.append({k: v for k, v in cycle.items() if not str(k).startswith("_")})
    return cleaned


def _write_outputs(
    result: dict[str, Any],
    *,
    log_path: str,
    evidence_path: str,
    secrets: Iterable[str] | None = None,
) -> None:
    """Ghi JSONL audit + markdown evidence (da scrub secret)."""
    secrets = list(secrets) if secrets is not None else _secret_values()
    ts_iso = datetime.now(timezone.utc).isoformat()

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    record = dict(result)
    record["timestamp"] = ts_iso
    with log_file.open("a", encoding="utf-8") as handle:
        blob = json.dumps(_sanitize(record), ensure_ascii=False, default=str)
        handle.write(_scrub_secrets(blob, secrets) + "\n")

    ev_file = Path(evidence_path)
    ev_file.parent.mkdir(parents=True, exist_ok=True)
    cycles = result.get("cycles") or []
    lines = [
        "# Sprint 21 - Multi-Cycle Testnet Burn-In Evidence",
        "",
        f"- **Timestamp (UTC)**: `{ts_iso}`",
        f"- **Verdict**: `{result.get('verdict')}`",
        f"- **Symbol**: `{result.get('symbol')}`",
        f"- **Universe**: {len(result.get('universe') or [])} symbols (`{'`, `'.join(result.get('universe') or [])}`)",
        f"- **Secrets redacted**: `{result.get('secrets_redacted')}`",
        "",
        "| Cycle | Name | OK | Reconcile | All 8 SYNCED | Equity Gap (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for cycle in cycles:
        detail = cycle.get("detail") or {}
        lines.append(
            "| {cycle} | {name} | {ok} | {rec} | {synced} | {gap} |".format(
                cycle=cycle.get("cycle"),
                name=cycle.get("name"),
                ok=cycle.get("ok"),
                rec=detail.get("reconcile_status")
                or detail.get("mid_reconcile_status")
                or detail.get("post_reconcile_status")
                or "N/A",
                synced=detail.get("all_8_symbols_synced")
                if detail.get("all_8_symbols_synced") is not None
                else detail.get("mid_all_8_symbols_synced", detail.get("post_all_8_symbols_synced")),
                gap=detail.get("equity_gap_usd", "N/A"),
            )
        )
    lines.append("")
    lines.append("## Cycle Details")
    lines.append("")
    for cycle in cycles:
        lines.append(f"### Cycle {cycle.get('cycle')}: {cycle.get('name')}")
        lines.append("")
        lines.append("```json")
        lines.append(
            _scrub_secrets(
                json.dumps(_sanitize(cycle.get("detail") or {}), ensure_ascii=False, indent=2),
                secrets,
            )
        )
        lines.append("```")
        lines.append("")
    lines.append(f"_Log JSONL: `{log_path}`_")
    lines.append("")

    header_needed = not ev_file.exists() or ev_file.stat().st_size == 0
    mode = "w" if header_needed else "a"
    with ev_file.open(mode, encoding="utf-8") as handle:
        handle.write(_scrub_secrets("\n".join(lines) + "\n", secrets))


def run_multi_cycle_burnin(
    *,
    symbol: str = "BTC/USDT:USDT",
    spot_qty: float = 0.001,
    perp_qty: float = 0.001,
    maker_ttl_sec: float = 3.0,
    client: Any = None,
    universe: Iterable[str] = UNIVERSE_SYMBOLS,
    warmup_hours: float = WARMUP_REQUIRED_HOURS,
    write_files: bool = True,
    evidence_path: str = DEFAULT_EVIDENCE_PATH,
    log_path: str = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    """Chay 3 chu ky burn-in lien tiep tren Binance Testnet."""
    universe_list = [str(s) for s in universe]
    cycles: list[dict[str, Any]] = []

    if client is None:
        try:
            load_dotenv_testnet()
        except Exception:  # pragma: no cover - thieu file
            pass
        client = BinanceSignedTestnetClient(dry_run=False, recv_window=10000)

    # Scrub ca khoa cua client duoc truyen tu ngoai (khong chi bien moi truong).
    secrets = _secret_values(client)

    result: dict[str, Any] = {
        "verdict": "FAIL",
        "symbol": symbol,
        "universe": universe_list,
        "cycles": [],
        "log_path": log_path,
        "evidence_path": evidence_path,
        "secrets_redacted": True,
    }

    try:
        cycle1 = _cycle1_bootstrap(
            client, symbol=symbol, universe=universe_list, warmup_hours=warmup_hours
        )
        cycles.append(cycle1)
        if not cycle1["ok"]:
            raise RuntimeError("Cycle 1 (Bootstrap & Reconcile) that bai")

        cycle2 = _cycle2_execution(
            client,
            cycle1,
            symbol=symbol,
            universe=universe_list,
            spot_qty=spot_qty,
            perp_qty=perp_qty,
            maker_ttl_sec=maker_ttl_sec,
        )
        cycles.append(cycle2)
        if not cycle2["ok"]:
            raise RuntimeError("Cycle 2 (Live Dual-Leg Execution) that bai")

        cycle3 = _cycle3_flatten(
            client, cycle2, symbol=symbol, universe=universe_list, maker_ttl_sec=maker_ttl_sec
        )
        cycles.append(cycle3)
        if not cycle3["ok"]:
            raise RuntimeError("Cycle 3 (Clean Flatten) that bai")
    except Exception as exc:  # noqa: BLE001 - luon ghi evidence du that bai
        result["error"] = str(exc)
    finally:
        result["cycles"] = _public_cycles(cycles)
        result["cycle_count"] = len(cycles)
        all_ok = len(cycles) == 3 and all(bool(c.get("ok")) for c in cycles)
        result["verdict"] = "PASS" if all_ok else "FAIL"
        if write_files:
            _write_outputs(result, log_path=log_path, evidence_path=evidence_path, secrets=secrets)

    return result


def main(argv: list[str] | None = None) -> int:
    """Entrypoint CLI cho burn-in 3 chu ky."""
    parser = argparse.ArgumentParser(
        description="Sprint 21 - Binance Testnet 3-cycle multi-cycle burn-in"
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading symbol")
    parser.add_argument("--spot-qty", type=float, default=0.001, help="Spot quantity")
    parser.add_argument("--perp-qty", type=float, default=0.001, help="Perp quantity")
    parser.add_argument("--maker-ttl-sec", type=float, default=3.0, help="Maker TTL in sec")
    parser.add_argument(
        "--warmup-hours",
        type=float,
        default=WARMUP_REQUIRED_HOURS,
        help="So gio du lieu warmup da san (cong 720h)",
    )
    parser.add_argument(
        "--mock-transport",
        action="store_true",
        help="Chay offline voi mock transport deterministic (khong cham mang)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run sign only")
    parser.add_argument(
        "--evidence-doc", default=DEFAULT_EVIDENCE_PATH, help="Duong dan file evidence markdown"
    )
    parser.add_argument(
        "--log-jsonl", default=DEFAULT_LOG_PATH, help="Duong dan file log JSONL"
    )
    parser.add_argument("--no-write", action="store_true", help="Khong ghi evidence/log")
    parser.add_argument("--json", action="store_true", help="In 1 dong JSON tom tat")

    args = parser.parse_args(argv)

    try:
        client: Any = None
        if args.mock_transport:
            from scripts.run_testnet_live_roundtrip import build_mock_transport

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
            # Supervisor nay doi soat voi SO LENH/SO DU THAT tren san; client dry_run
            # khong tra snapshot -> chu ky 1 luon FAIL. Bao ro thay vi chay roi bao loi.
            print(
                "Verdict: FAIL\nError: --dry-run khong ho tro cho supervisor 3 chu ky "
                "(can snapshot so du that de doi soat). Dung --mock-transport."
            )
            return 2

        result = run_multi_cycle_burnin(
            symbol=args.symbol,
            spot_qty=args.spot_qty,
            perp_qty=args.perp_qty,
            maker_ttl_sec=args.maker_ttl_sec,
            client=client,
            warmup_hours=args.warmup_hours,
            write_files=not args.no_write,
            evidence_path=args.evidence_doc,
            log_path=args.log_jsonl,
        )

        if args.json:
            print(_scrub_secrets(json.dumps(_sanitize(result), ensure_ascii=False, default=str)))
        else:
            print(f"Verdict: {_scrub_secrets(str(result.get('verdict')))}")
            for cycle in result.get("cycles") or []:
                print(
                    f"  Cycle {cycle.get('cycle')}: {cycle.get('name')} "
                    f"-> ok={cycle.get('ok')}"
                )
            if result.get("error"):
                print(f"Error: {_scrub_secrets(str(result.get('error')))}")

        return 0 if result.get("verdict") == "PASS" else 2

    except Exception as exc:  # noqa: BLE001 - CLI khong duoc crash
        payload = {"verdict": "FAIL", "error": str(exc)}
        print(_scrub_secrets(json.dumps(payload, ensure_ascii=False)))
        return 2


if __name__ == "__main__":
    sys.exit(main())
