"""CLI khởi chạy Paper/Live Executor (24/7 hoặc --once)."""
import argparse
import dataclasses
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from run_backtest import generate_synthetic_ohlcv
from src.config import TradingConfig
from src.data_loader import DataLoader, _index_to_ms, timeframe_to_ms
from src.live_executor import LiveExecutor
from src.ml_filter import MetaLabelFilter
from src.notifier import TelegramNotifier
from src.risk_manager import RiskManager
from src.state_store import StateStore
from src.strategy import Strategy


class _OfflinePaperExchange:
    """Exchange giả lập phục vụ chạy --mode paper --once khi offline hoặc trong pytest."""

    def __init__(self, symbol: str = "BTC/USDT:USDT", timeframe: str = "15m") -> None:
        df = generate_synthetic_ohlcv(
            n_candles=320, symbol=symbol, timeframe=timeframe, seed=42
        )
        ts_ms = _index_to_ms(df.index)
        self.rows: list[list[float | int]] = [
            [
                int(ts_ms[i]),
                float(df["open"].iloc[i]),
                float(df["high"].iloc[i]),
                float(df["low"].iloc[i]),
                float(df["close"].iloc[i]),
                float(df["volume"].iloc[i]),
            ]
            for i in range(len(df))
        ]
        self.now_ms: int = int(ts_ms[-1]) + timeframe_to_ms(timeframe)

    def milliseconds(self) -> int:
        return self.now_ms

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> list[list[float | int]]:
        out = [r for r in self.rows if since is None or int(r[0]) >= int(since)]
        if limit is not None:
            out = out[: int(limit)]
        return [list(r) for r in out]


logger = logging.getLogger("run_live")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vận hành Paper/Live Quant Trading Bot (Binance USDT-M Futures)"
    )
    parser.add_argument(
        "--mode",
        choices=("paper", "live", "testnet", "shadow-live"),
        default="paper",
        help="Chế độ thực thi: paper, live, testnet hoặc shadow-live (read-only public feed, không đặt lệnh thật)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Chạy đúng 1 chu kỳ kiểm tra nến đóng rồi thoát",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Xoa trang thai cu va khoi tao lai von theo --capital",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="So chu ky daemon can chay truoc khi thoat",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=60.0,
        help="So giay giua 2 chu ky",
    )
    parser.add_argument(
        "--audit-log",
        default=None,
        help="Sprint 16: duong dan JSONL audit (ghi 1 dong moi chu ky), vd logs/sprint16_paper_audit.jsonl",
    )
    parser.add_argument(
        "--dry-run-sign",
        action="store_true",
        help="Sprint 16: chi ky va dong goi lenh testnet, khong gui (bat buoc khi thieu API key)",
    )
    parser.add_argument(
        "--testnet-send-orders",
        action="store_true",
        help="Sprint 16: cho phep gui lenh that len testnet (mac dinh TAT, chi ky/dong goi)",
    )
    parser.add_argument(
        "--reconcile-snapshot",
        default=None,
        help="Sprint 17: duong dan JSON snapshot san (spot_balances/perp_positions/exchange_equity_usd). "
        "Neu doi soat khong khop -> fail-closed, khong mo vi the moi.",
    )
    parser.add_argument(
        "--live-snapshot",
        action="store_true",
        help="Sprint 18: lay snapshot tai khoan THAT co ky HMAC-SHA256 tu Binance Testnet "
        "(/api/v3/account + /fapi/v2/account + /fapi/v2/positionRisk) roi doi soat truoc khi mo lenh.",
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT:USDT",
        help="Cặp giao dịch (mặc định: BTC/USDT:USDT)",
    )
    parser.add_argument(
        "--timeframe",
        default="15m",
        help="Khung thời gian nến (mặc định: 15m)",
    )
    parser.add_argument(
        "--strategy",
        choices=("directional", "dynamic_carry", "barbell"),
        default="directional",
        help="Chiến lược vận hành: directional (mặc định), dynamic_carry, hoặc barbell (Sprint 9)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=3000.0,
        help="Mức vốn vận hành cho chiến lược carry (mặc định: 3000.0 USDT)",
    )
    parser.add_argument(
        "--execution-style",
        choices=("maker_taker_hedge", "real_taker"),
        default="maker_taker_hedge",
        help="Cơ chế khớp lệnh: maker_taker_hedge (mặc định Sprint 9) hoặc real_taker",
    )
    parser.add_argument(
        "--strategy-mode",
        choices=("breakout", "trend_pullback", "mean_reversion", "regime_adaptive"),
        default="breakout",
        help="Chế độ chiến lược (mặc định: breakout)",
    )
    parser.add_argument(
        "--entry-mode",
        choices=("market", "limit_pullback"),
        default="market",
        help="Chế độ vào lệnh: market hoặc limit_pullback (mặc định: market)",
    )
    parser.add_argument(
        "--use-trailing",
        action="store_true",
        help="Bật cơ chế ATR Trailing Stop",
    )
    parser.add_argument(
        "--fee-aware",
        action="store_true",
        help="Bật cơ chế định cỡ vị thế tính phí (Fee-Aware Position Sizing)",
    )
    parser.add_argument(
        "--use-ml",
        action="store_true",
        help="Bật bộ lọc xác suất ML Meta-Labeling",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Đường dẫn file lưu trạng thái JSON (tùy chọn)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Dùng nguồn nến giả lập offline (không gọi mạng)",
    )
    parser.add_argument(
        "--shadow-timeout-sec",
        type=float,
        default=5.0,
        help="Timeout (giây) cho mỗi request public feed ở chế độ shadow-live (mặc định 5.0)",
    )
    parser.add_argument(
        "--offline-fixture",
        default=None,
        help="Đường dẫn JSON fixture cho shadow-live (không gọi mạng, dùng để test/CI)",
    )
    return parser.parse_args(argv)


def _load_shadow_fixture(fixture_path: Any) -> Any:
    """Nạp JSON fixture cho shadow-live; trả None nếu không có đường dẫn."""
    if not fixture_path:
        return None
    import json as _json

    path = Path(fixture_path)
    with open(path, "r", encoding="utf-8") as f:
        return _json.load(f)


def _run_shadow_live_preflight(args: argparse.Namespace, symbols: list[str]) -> bool:
    """Chốt an toàn trước khi chạy shadow-live: chỉ đọc public feed, fail-closed.

    Trả False (=> dừng, exit != 0) khi feed không healthy hoặc fixture lỗi.
    """
    from src.binance_public_feed import fetch_binance_shadow_snapshot

    try:
        fixture_data = _load_shadow_fixture(getattr(args, "offline_fixture", None))
    except Exception as e:
        logger.error("Shadow-live: cannot load offline fixture: %s", e)
        return False

    snapshot = fetch_binance_shadow_snapshot(
        symbols,
        timeout_sec=float(getattr(args, "shadow_timeout_sec", 5.0)),
        fixture_data=fixture_data,
    )
    logger.info(
        "Shadow-live snapshot: source=%s | healthy=%s | symbols=%d | errors=%d",
        snapshot.get("source"),
        snapshot.get("healthy"),
        len(snapshot.get("symbols", {})),
        len(snapshot.get("errors", [])),
    )
    for err in list(snapshot.get("errors", []))[:5]:
        logger.warning("Shadow-live feed issue: %s", err)
    if not snapshot.get("healthy", False):
        logger.error("Shadow-live: public feed unhealthy -> fail-closed, no cycles executed.")
        return False
    logger.info("Shadow-live: read-only feed OK (no orders can be placed in this mode).")
    return True


def _run_testnet_signing_preflight(args: argparse.Namespace, symbols: list[str]) -> bool:
    """Sprint 16/19: chot an toan cho mode testnet, fail-closed.

    dry_run=True  : chi ky + dong goi payload (khong gui lenh).
    dry_run=False : nap `.env.testnet`, dong bo gio may chu Spot/Futures, kiem tra
                    xac thuc tai khoan THAT cua tung venue (khong dat lenh o day -
                    dat lenh that do `scripts/run_testnet_live_roundtrip.py` dam nhiem).
    """
    from src.binance_signed_client import (
        DEFAULT_LIVE_RECV_WINDOW_MS,
        BinanceCredentialsMissingError,
        BinanceSignedTestnetClient,
        load_dotenv_testnet,
    )

    symbol = symbols[0] if symbols else "BTC/USDT:USDT"
    dry_run = not bool(getattr(args, "testnet_send_orders", False))
    try:
        load_dotenv_testnet()
    except Exception as e:  # thieu file .env.testnet la binh thuong
        logger.info("Testnet: khong nap duoc .env.testnet (%s) - dung bien moi truong.", e)
    try:
        client = BinanceSignedTestnetClient(
            dry_run=dry_run,
            recv_window=DEFAULT_RECV_WINDOW_MS if not dry_run else 5000,
        )
    except BinanceCredentialsMissingError as e:
        logger.error("Testnet: %s", e)
        return False

    logger.info(
        "Testnet signing preflight: symbol=%s | dry_run=%s | spot_creds=%s | futures_creds=%s",
        symbol,
        client.dry_run,
        client.has_spot_credentials(),
        client.has_futures_credentials(),
    )
    # Sprint 20: chan doan dinh dang 64 ky tu TRUOC khi goi mang (khong lo khoa).
    fmt = client.credential_format_report()
    for venue in ("spot", "futures"):
        report = (fmt.get(venue) or {}) if isinstance(fmt, dict) else {}
        if not report.get("ok"):
            logger.error(
                "Testnet credential format (%s): %s",
                venue,
                report.get("message"),
            )
    if not dry_run and not bool(fmt.get("all_ok", False)):
        logger.error(
            "Testnet: dinh dang khoa khong hop le (yeu cau dung 64 ky tu alnum) - BLOCKED."
        )
        return False
    if client.dry_run:
        try:
            proof = client.build_delta_neutral_order_pair(
                symbol=symbol,
                spot_qty=0.001,
                perp_qty=0.001,
                post_only_spot=False,
                timestamp_ms=1700000000000,
            )
        except Exception as e:
            logger.warning("Testnet: khong dung duoc payload mau (%s) - bo qua.", e)
            return True
        sig = str(proof["spot"]["params"].get("signature", ""))
        logger.info(
            "Testnet sign proof: spot=%s | perp=%s | sig_prefix=%s | dry_run=True (no order sent)",
            proof["spot"]["url"],
            proof["perp"]["url"],
            sig[:8] if sig else "<unsigned>",
        )
        return True

    # Sprint 19: dong bo gio may chu + xac thuc tai khoan THAT (khong dat lenh).
    sync = client.sync_server_time("both")
    logger.info(
        "Testnet server time sync: spot_offset=%sms | futures_offset=%sms | errors=%s",
        sync.get("spot_offset_ms"),
        sync.get("futures_offset_ms"),
        sync.get("errors"),
    )
    ok_any = False
    for venue in ("spot", "futures"):
        report = client.check_account_access(venue)
        if report.get("authenticated"):
            ok_any = True
            logger.info(
                "Testnet account OK: venue=%s | assets=%s | offset=%sms",
                venue,
                sorted(report.get("holdings", {}).keys()),
                report.get("server_time_offset_ms"),
            )
        else:
            logger.warning(
                "Testnet account FAIL: venue=%s | error=%s | hint=%s",
                venue,
                report.get("error"),
                report.get("hint"),
            )
    if not ok_any:
        logger.error("Testnet: khong venue nao xac thuc duoc - BLOCKED (fail-closed).")
    return True


def _run_reconcile_preflight(
    args: argparse.Namespace,
    executor: object,
    prices: dict[str, float] | None = None,
) -> dict | None:
    """Sprint 17: doi soat state local voi snapshot san truoc khi mo vi the moi.

    Fail-closed: file snapshot loi/khong doc duoc -> tra ve report BLOCKED, khong
    bao gio cho phep mo lenh moi. Tra ve None khi khong bat ``--reconcile-snapshot``.
    """
    path = getattr(args, "reconcile_snapshot", None)
    if not path:
        return None
    from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
    from src.exchange_reconciler import load_snapshot, reconcile_portfolio_state

    try:
        snapshot = load_snapshot(path)
    except Exception as e:  # fail-closed: khong doc duoc snapshot = khong giao dich
        logger.error("Reconcile: BLOCKED (SNAPSHOT_UNREADABLE) - %s", e)
        report = {
            "status": "SNAPSHOT_UNREADABLE",
            "trading_allowed": False,
            "repair_actions": [],
            "items": [],
            "orphans": [],
            "phantoms": [],
            "drifting": [],
            "equity_gap_usd": 0.0,
            "equity_gap_pct": 0.0,
            "summary": f"SNAPSHOT_UNREADABLE | {e}",
        }
        apply = getattr(executor, "apply_reconcile_report", None)
        if callable(apply):
            apply(report)
        return report

    px = dict(snapshot.get("prices") or {}) if isinstance(snapshot, dict) else {}
    if prices:
        px.update({k: v for k, v in prices.items() if v not in px})
    report = reconcile_portfolio_state(
        getattr(executor, "state", {}),
        snapshot,
        px,
        EXCHANGE_LOT_SPECS,
    )
    apply = getattr(executor, "apply_reconcile_report", None)
    if callable(apply):
        apply(report)
    else:  # executor toi gian (test double) - van ghi thuoc tinh
        setattr(executor, "reconcile_report", report)
        setattr(executor, "reconcile_status", report.get("status"))
        setattr(executor, "trading_allowed", bool(report.get("trading_allowed", False)))
    return report


def _run_signed_snapshot_preflight(
    args: argparse.Namespace,
    executor: object,
) -> dict | None:
    """Sprint 18: lay snapshot tai khoan THAT co ky HMAC-SHA256 (testnet) roi doi soat.

    Sprint 20: trong ``--mode testnet`` ham nay TU DONG chay (khong can
    ``--live-snapshot``) mien la khong truyen ``--reconcile-snapshot`` - nhu vay
    daemon luon doi soat tren snapshot song truoc khi vao chu ky giao dich.

    Fail-closed: thieu khoa / loi mang / payload la -> report BLOCKED, khong mo lenh moi.
    """
    auto_live = str(getattr(args, "mode", "")) == "testnet" and not getattr(
        args, "reconcile_snapshot", None
    )
    if not (getattr(args, "live_snapshot", False) or auto_live):
        return None
    from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
    from src.exchange_reconciler import reconcile_portfolio_state

    def _blocked(status: str, message: str) -> dict:
        report = {
            "status": status,
            "trading_allowed": False,
            "repair_actions": [],
            "items": [],
            "orphans": [],
            "phantoms": [],
            "drifting": [],
            "equity_gap_usd": 0.0,
            "equity_gap_pct": 0.0,
            "summary": f"{status} | {message}",
        }
        apply_fn = getattr(executor, "apply_reconcile_report", None)
        if callable(apply_fn):
            apply_fn(report)
        else:
            setattr(executor, "reconcile_report", report)
            setattr(executor, "reconcile_status", status)
            setattr(executor, "trading_allowed", False)
        logger.error("Reconcile(live): BLOCKED (%s) - %s", status, message)
        return report

    try:
        from src.binance_signed_client import (
            DEFAULT_LIVE_RECV_WINDOW_MS,
            BinanceSignedTestnetClient,
            load_dotenv_testnet,
        )

        # Sprint 20: mode testnet luon nap `.env.testnet` + dong bo gio 2 venue.
        try:
            load_dotenv_testnet()
        except Exception as exc:  # thieu file la binh thuong
            logger.info("Testnet: khong nap duoc .env.testnet (%s).", exc)
        # Sprint 19: recvWindow rong hon cho testnet that (bu do tre mang/offset gio).
        client = BinanceSignedTestnetClient(
            dry_run=False, recv_window=DEFAULT_LIVE_RECV_WINDOW_MS
        )
        client.sync_server_time("both")
    except Exception as exc:  # thieu khoa -> fail-closed
        return _blocked("SNAPSHOT_CREDENTIALS_MISSING", str(exc))

    symbols = list(getattr(executor, "symbols", []) or [])
    state = getattr(executor, "state", None)
    prices: dict[str, float] = {}
    for sym, pos in (getattr(state, "carry_positions", {}) or {}).items():
        if not isinstance(pos, dict):
            continue
        for key in ("entry_spot_price", "entry_perp_price"):
            px = float(pos.get(key, 0.0) or 0.0)
            if px > 0.0:
                prices[sym] = px
                break

    snapshot = client.fetch_exchange_reconcile_snapshot(symbols, prices)
    if not bool(snapshot.get("snapshot_healthy", False)):
        return _blocked("SNAPSHOT_UNHEALTHY", str(snapshot.get("error", "unknown")))

    # Sprint 20: doi soat theo UNIVERSE dang chay (bo qua dust testnet).
    snapshot = _scope_snapshot_to_symbols(snapshot, symbols)

    for item in snapshot.get("perp_positions_detail") or []:
        if isinstance(item, dict):
            px = float(item.get("mark_price", 0.0) or 0.0)
            sym = str(item.get("symbol") or "")
            if px > 0.0 and sym:
                prices[sym] = px

    report = reconcile_portfolio_state(
        state if state is not None else {},
        snapshot,
        prices,
        EXCHANGE_LOT_SPECS,
    )
    apply_fn = getattr(executor, "apply_reconcile_report", None)
    if callable(apply_fn):
        apply_fn(report)
    else:
        setattr(executor, "reconcile_report", report)
        setattr(executor, "reconcile_status", report.get("status"))
        setattr(executor, "trading_allowed", bool(report.get("trading_allowed", False)))
    logger.info("Reconcile(live): %s", report.get("summary", ""))
    return report


def _scope_snapshot_to_symbols(
    snapshot: dict[str, Any], symbols: list[str]
) -> dict[str, Any]:
    """Sprint 20: thu hep `spot_balances` ve dung cac base asset dang giao dich.

    Tai khoan Binance Testnet duoc cap san ~500 tai san "dust"; doi soat toan tai khoan
    se luon bao ``ORPHAN_ON_EXCHANGE`` cho cac tai san khong giao dich va chan vo ly
    moi chu ky. Pham vi doi soat cua daemon = universe dang chay; so tai san ngoai
    pham vi duoc ghi lai de kiem toan.
    """
    if not isinstance(snapshot, dict):
        return snapshot
    balances = snapshot.get("spot_balances")
    if not isinstance(balances, dict):
        return snapshot
    wanted: set[str] = set()
    for sym in symbols or []:
        canon = str(sym).replace(":USDT", "").replace("/", "")
        base = canon[:-4] if canon.upper().endswith("USDT") else canon
        if base:
            wanted.add(base.upper())
    scoped = {k: v for k, v in balances.items() if str(k).upper() in wanted}
    snapshot["untracked_spot_assets"] = max(0, len(balances) - len(scoped))
    snapshot["spot_balances"] = scoped
    snapshot["reconcile_symbol_scope"] = list(symbols or [])
    return snapshot


def _run_testnet_order_pair(
    args: argparse.Namespace,
    executor: object,
    symbols: list[str],
) -> dict | None:
    """Sprint 20: gui 1 cap lenh Delta-Neutral THAT (Spot BUY + Perp SELL) roi dong lai.

    Chi chay khi ``--mode testnet`` + ``--testnet-send-orders`` VA reconcile khong chan.
    Order id that duoc ghi vao ``--audit-log`` (JSONL) duoi khoa ``testnet_order_ids``.
    Fail-closed: dinh dang khoa sai / reconcile BLOCKED / loi mang -> khong gui lenh.
    """
    if str(getattr(args, "mode", "")) != "testnet":
        return None
    if not bool(getattr(args, "testnet_send_orders", False)):
        return None

    from src.carry_live_executor import append_audit_record

    symbol = symbols[0] if symbols else "BTC/USDT:USDT"
    reconcile_status = getattr(executor, "reconcile_status", None)
    trading_allowed = bool(getattr(executor, "trading_allowed", True))
    record: dict[str, Any] = {
        "schema": "sprint20.testnet_order_pair.v1",
        "event": "testnet_order_pair",
        "symbol": symbol,
        "reconcile_status": reconcile_status,
        "testnet_order_ids": {},
    }

    if reconcile_status is not None and not trading_allowed:
        record["status"] = "BLOCKED_RECONCILE"
        logger.error(
            "Testnet orders: BLOCKED vi reconcile=%s (fail-closed, khong gui lenh).",
            reconcile_status,
        )
        append_audit_record(getattr(args, "audit_log", None), record)
        setattr(executor, "testnet_order_ids", {})
        return record

    try:
        from src.binance_signed_client import (
            DEFAULT_LIVE_RECV_WINDOW_MS,
            BinanceSignedTestnetClient,
            load_dotenv_testnet,
            quantize_price,
        )

        try:
            load_dotenv_testnet()
        except Exception:  # pragma: no cover - thieu file la binh thuong
            pass
        client = BinanceSignedTestnetClient(
            dry_run=False, recv_window=DEFAULT_LIVE_RECV_WINDOW_MS
        )
        client.sync_server_time("both")

        fmt = client.credential_format_report()
        record["credential_format"] = {
            "spot_ok": bool((fmt.get("spot") or {}).get("ok")),
            "futures_ok": bool((fmt.get("futures") or {}).get("ok")),
        }
        if not bool(fmt.get("all_ok", False)):
            record["status"] = "CREDENTIAL_FORMAT_INVALID"
            record["diagnostics"] = [
                (fmt.get("spot") or {}).get("message", ""),
                (fmt.get("futures") or {}).get("message", ""),
            ]
            logger.error(
                "Testnet orders: BLOCKED vi dinh dang khoa sai (%s) - khong gui lenh.",
                record["diagnostics"],
            )
            append_audit_record(getattr(args, "audit_log", None), record)
            setattr(executor, "testnet_order_ids", {})
            return record

        ticker = client.fetch_book_ticker(symbol, "spot")
        mid = float(ticker.get("mid") or ticker.get("bid") or 0.0)
        qty = 0.001
        entry = client.execute_live_testnet_order_pair(
            symbol=symbol,
            spot_qty=qty,
            perp_qty=qty,
            spot_price=quantize_price(
                float(ticker.get("bid") or mid), 0.01, "BUY"
            ),
            direction="open",
            post_only_spot=True,
            spot_use_limit_maker=True,
            perp_post_only=False,
            maker_ttl_sec=2.0,
            require_futures=False,
            perp_available=client.has_futures_credentials(),
        )
        spot_entry = entry.get("spot") or {}
        perp_entry = entry.get("perp") or {}
        filled_qty = float(spot_entry.get("executedQty") or 0.0)
        ids: dict[str, Any] = {
            "spot_entry_order_id": spot_entry.get("orderId"),
            "perp_entry_order_id": perp_entry.get("orderId"),
            "spot_exit_order_id": None,
            "perp_exit_order_id": None,
        }
        exit_status = None
        if filled_qty > 0.0:
            exit_res = client.execute_live_testnet_order_pair(
                symbol=symbol,
                spot_qty=filled_qty,
                perp_qty=filled_qty,
                spot_price=quantize_price(
                    float(ticker.get("ask") or mid), 0.01, "SELL"
                ),
                direction="close",
                post_only_spot=True,
                spot_use_limit_maker=True,
                perp_post_only=False,
                maker_ttl_sec=2.0,
                require_futures=False,
                perp_available=client.has_futures_credentials(),
            )
            ids["spot_exit_order_id"] = (exit_res.get("spot") or {}).get("orderId")
            ids["perp_exit_order_id"] = (exit_res.get("perp") or {}).get("orderId")
            exit_status = exit_res.get("status")

        record["testnet_order_ids"] = ids
        record["entry_status"] = entry.get("status")
        record["exit_status"] = exit_status
        record["executed_qty"] = filled_qty
        record["status"] = "SENT" if entry.get("status") == "FILLED" else str(
            entry.get("status")
        )
    except Exception as exc:  # fail-closed: khong lam chet daemon
        record["status"] = "ERROR"
        record["error"] = str(exc)
        logger.error("Testnet orders: loi khi gui lenh (%s).", exc)

    append_audit_record(getattr(args, "audit_log", None), record)
    setattr(executor, "testnet_order_ids", record.get("testnet_order_ids") or {})
    logger.info(
        "Testnet orders: status=%s | ids=%s", record.get("status"), record.get("testnet_order_ids")
    )
    return record


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args(argv)
    in_pytest = "PYTEST_CURRENT_TEST" in os.environ

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    has_api_keys = bool(api_key and api_secret)

    exec_mode = args.mode
    if args.mode == "live":
        if not has_api_keys:
            logger.error("Live mode: thieu BINANCE_API_KEY hoac BINANCE_API_SECRET trong env.")
            return 2
    elif args.mode == "testnet":
        if not has_api_keys:
            logger.warning("Testnet: thieu API key, chay Public Live Feed Paper Mode")
            exec_mode = "paper"
        else:
            # Sprint 16: testnet luon chay paper executor, chi ky/dong goi lenh de kiem tra.
            logger.info("Testnet: ki + dong goi lenh (khong gui), executor van chay paper.")
            exec_mode = "paper"
    elif args.mode == "shadow-live":
        # Shadow-live: chỉ đọc public feed + mô phỏng paper, KHÔNG dùng API key, KHÔNG đặt lệnh thật.
        logger.info("Shadow-live: read-only public feed + paper executor (no real orders).")
        exec_mode = "paper"

    if args.state_file is not None:
        state_path = Path(args.state_file)
    elif in_pytest:
        state_path = Path(tempfile.gettempdir()) / "quant_trading_pytest_bot_state.json"
    else:
        state_path = TradingConfig().state_file_path

    strat = getattr(args, "strategy", "directional")
    if strat in ("dynamic_carry", "barbell"):
        logger.info("=" * 60)
        if strat == "barbell":
            logger.info("QUANT TRADING BOT STARTING (BARBELL CARRY SLEEVE)")
            logger.info("Mode: %s (exec_mode=%s) | Strategy: barbell (paper cross-exchange carry sleeve)", args.mode, exec_mode)
        else:
            logger.info("QUANT TRADING BOT STARTING (DYNAMIC CARRY)")
            logger.info("Mode: %s (exec_mode=%s) | Strategy: %s", args.mode, exec_mode, strat)
        logger.info(
            "Capital: %.2f USDT | Execution Style: %s",
            args.capital,
            args.execution_style,
        )
        logger.info("State file: %s", state_path)
        logger.info("=" * 60)

        from src.carry_live_executor import CarryLiveExecutor
        from src.dynamic_carry_basis import DynamicCarryConfig

        perp_exchange = None
        spot_exchange = None
        if not (args.synthetic or in_pytest):
            try:
                import ccxt

                ex_cfg: dict[str, Any] = {"enableRateLimit": True}
                if has_api_keys and args.mode in ("live", "testnet"):
                    ex_cfg["apiKey"] = api_key
                    ex_cfg["secret"] = api_secret

                perp_exchange = ccxt.binanceusdm(dict(ex_cfg))
                spot_exchange = ccxt.binance(dict(ex_cfg))

                if args.mode == "testnet" and has_api_keys:
                    for ex in (perp_exchange, spot_exchange):
                        if hasattr(ex, "set_sandbox_mode") and callable(ex.set_sandbox_mode):
                            try:
                                ex.set_sandbox_mode(True)
                            except Exception as ex_err:
                                logger.warning("Could not set sandbox mode: %s", ex_err)
            except Exception as e:
                logger.warning(
                    "Could not initialize live exchanges (%s), falling back to offline cache",
                    e,
                )
                perp_exchange = None
                spot_exchange = None

        carry_config = DynamicCarryConfig(initial_capital=float(args.capital))
        trading_config = dataclasses.replace(
            TradingConfig(),
            timeframe=args.timeframe,
            state_file_path=state_path,
        )

        if strat == "barbell":
            try:
                from src.cross_exchange_carry import CORE_4_SYMBOLS, LIQUID_8_SYMBOLS
            except (ImportError, AttributeError):
                CORE_4_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]
                LIQUID_8_SYMBOLS = list(CORE_4_SYMBOLS) + [
                    "DOGE/USDT:USDT",
                    "XRP/USDT:USDT",
                    "AVAX/USDT:USDT",
                    "LINK/USDT:USDT",
                ]
            from src.barbell_portfolio import BarbellConfig

            venues: dict[str, Any] = {
                "binance": perp_exchange,
                "binance_spot": spot_exchange,
            }
            if not (args.synthetic or in_pytest):
                try:
                    import ccxt

                    bybit_cfg: dict[str, Any] = {"enableRateLimit": True}
                    bybit_key = os.environ.get("BYBIT_API_KEY")
                    bybit_secret = os.environ.get("BYBIT_API_SECRET")
                    if bybit_key and bybit_secret and args.mode in ("live", "testnet"):
                        bybit_cfg["apiKey"] = bybit_key
                        bybit_cfg["secret"] = bybit_secret
                    bybit_ex = ccxt.bybit(bybit_cfg)
                    if args.mode == "testnet" and hasattr(bybit_ex, "set_sandbox_mode") and callable(bybit_ex.set_sandbox_mode):
                        try:
                            bybit_ex.set_sandbox_mode(True)
                        except Exception as ex_err:
                            logger.warning("Could not set sandbox mode on bybit: %s", ex_err)
                    venues["bybit"] = bybit_ex
                except Exception as e:
                    logger.warning("Could not initialize Bybit exchange for venues: %s", e)

            overlay_cfg = BarbellConfig(
                initial_capital=float(args.capital),
                carry_variant="tier1b",
                gating_mode="smooth_conviction",
            )
            # Sprint 15: carry gate phai dung dung Frozen Alpha Tier 1B (Sprint 11),
            # khong dung defaults Sprint 7 (entry 1.0 bps, score co basis blend).
            carry_config = DynamicCarryConfig(
                initial_capital=float(args.capital),
                ema_span=42,
                top_k=4,
                entry_threshold=0.000045,
                exit_threshold=0.0,
                min_hold_periods=63,
                buffer_rank=3,
                w_basis=0.0,
                basis_ema_hours=168,
                expected_hold_periods=63,
                switch_rule="tier1b_buffer_rank",
                eta_override=0.80,
            )
            barbell_symbols = list(LIQUID_8_SYMBOLS)
            logger.info(
                "Universe: LIQUID-8 | Venues: %s | Lot quantization: ON", sorted(venues.keys())
            )

            executor = CarryLiveExecutor(
                config=trading_config,
                carry_config=carry_config,
                mode=exec_mode,
                state_store=StateStore(state_path),
                perp_exchange=perp_exchange,
                spot_exchange=spot_exchange,
                capital=float(args.capital),
                execution_style=args.execution_style,
                symbols=barbell_symbols,
                venues=venues,
                overlay_config=overlay_cfg,
                enable_lot_quantization=True,
                initial_capital=float(args.capital),
                reset_state=bool(args.reset_state),
                strategy=strat,
                tier1b_gate=True,
            )
            logger.info(
                "Warmup requirement: >= 720h history (saturation gate, lookback_hours=%d).",
                int(getattr(executor, "lookback_hours", 720)),
            )
            if args.audit_log:
                executor.audit_log_path = str(args.audit_log)
                logger.info("Audit JSONL: %s", executor.audit_log_path)
            if args.mode == "shadow-live":
                if not _run_shadow_live_preflight(args, barbell_symbols):
                    return 1
            if args.mode == "testnet":
                if not _run_testnet_signing_preflight(args, barbell_symbols):
                    return 1
            # Sprint 17: doi soat san truoc khi mo vi the moi (fail-closed).
            _run_reconcile_preflight(args, executor)
            # Sprint 18/20: doi soat dong qua snapshot tai khoan co ky HMAC (testnet that).
            _run_signed_snapshot_preflight(args, executor)
            # Sprint 20: gui lenh THAT khi --testnet-send-orders (chi khi reconcile cho phep).
            _run_testnet_order_pair(args, executor, barbell_symbols)
            return executor.run_loop(
                once=bool(args.once),
                cycles=args.cycles,
                interval_sec=float(args.interval_sec),
            )

        executor = CarryLiveExecutor(
            config=trading_config,
            carry_config=carry_config,
            mode=exec_mode,
            state_store=StateStore(state_path),
            perp_exchange=perp_exchange,
            spot_exchange=spot_exchange,
            capital=float(args.capital),
            execution_style=args.execution_style,
            initial_capital=float(args.capital),
            reset_state=bool(args.reset_state),
            strategy=strat,
        )
        logger.info(
            "Warmup requirement: >= 720h history (saturation gate, lookback_hours=%d).",
            int(getattr(executor, "lookback_hours", 720)),
        )
        if args.audit_log:
            executor.audit_log_path = str(args.audit_log)
            logger.info("Audit JSONL: %s", executor.audit_log_path)
        if args.mode == "shadow-live":
            if not _run_shadow_live_preflight(args, [args.symbol]):
                return 1
        if args.mode == "testnet":
            if not _run_testnet_signing_preflight(args, [args.symbol]):
                return 1
        # Sprint 17: doi soat san truoc khi mo vi the moi (fail-closed).
        _run_reconcile_preflight(args, executor)
        # Sprint 18/20: doi soat dong qua snapshot tai khoan co ky HMAC (testnet that).
        _run_signed_snapshot_preflight(args, executor)
        # Sprint 20: gui lenh THAT khi --testnet-send-orders (chi khi reconcile cho phep).
        _run_testnet_order_pair(args, executor, [args.symbol])
        return executor.run_loop(
            once=bool(args.once),
            cycles=args.cycles,
            interval_sec=float(args.interval_sec),
        )

    config = dataclasses.replace(
        TradingConfig(),
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy_mode=args.strategy_mode,
        entry_mode=args.entry_mode,
        use_trailing_stop=bool(args.use_trailing),
        use_fee_aware_sizing=bool(args.fee_aware),
        use_ml=bool(args.use_ml),
        state_file_path=state_path,
    )

    logger.info("=" * 60)
    logger.info("QUANT TRADING BOT STARTING")
    logger.info("Mode: %s | Symbol: %s | Timeframe: %s", args.mode, config.symbol, config.timeframe)
    logger.info(
        "Strategy: %s | Entry: %s | Trailing: %s | Fee-Aware: %s | ML: %s",
        config.strategy_mode,
        config.entry_mode,
        config.use_trailing_stop,
        config.use_fee_aware_sizing,
        config.use_ml,
    )
    logger.info("State file: %s", state_path)
    logger.info("=" * 60)

    if args.synthetic or in_pytest:
        exchange: Any = _OfflinePaperExchange(
            symbol=config.symbol, timeframe=config.timeframe
        )
        data_loader = DataLoader(config, exchange=exchange)
    else:
        data_loader = DataLoader(config)

    ml_filter = MetaLabelFilter(config) if config.use_ml else None
    executor_directional = LiveExecutor(
        config=config,
        mode=exec_mode if exec_mode in ("paper", "live") else "paper",
        data_loader=data_loader,
        strategy=Strategy(config),
        risk_manager=RiskManager(config),
        state_store=StateStore(state_path),
        notifier=TelegramNotifier(env_file=None if in_pytest else Path(".env")),
        ml_filter=ml_filter,
    )
    try:
        return executor_directional.run_loop(
            once=bool(args.once),
            cycles=args.cycles,
            interval_sec=float(args.interval_sec),
        )
    except TypeError:
        return executor_directional.run_loop(once=bool(args.once))


if __name__ == "__main__":
    sys.exit(main())
