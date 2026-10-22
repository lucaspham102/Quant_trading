"""Sprint 20: test cho luong daemon testnet (scope snapshot, preflight, order pair, audit).

Bao phu:
1. ``_scope_snapshot_to_symbols`` chi giu base asset dang giao dich + dem dust.
2. ``parse_args`` parse dung co testnet (send orders / reconcile None / live_snapshot).
3. ``_run_signed_snapshot_preflight`` tra None o mode paper khi khong co --live-snapshot.
4. Preflight TU DONG chay o mode testnet (khong can --live-snapshot) va ap report vao executor.
5. ORPHAN_ON_EXCHANGE (spot co nhung local khong co) -> chan giao dich.
6. MARGIN_DANGER (maint margin ratio > 0.50) -> chan giao dich.
7. SYNCED (spot + perp hedge khop, equity gap 0) -> cho phep giao dich.
8. ``_run_testnet_order_pair`` tra None khi mode paper hoac thieu --testnet-send-orders.
9. BLOCKED_RECONCILE ghi audit JSONL va KHONG khoi tao client.
10. Happy path gui cap lenh that: audit JSONL co du 4 order id + set executor.testnet_order_ids.
11. ``build_cycle_audit_record`` mang theo testnet_order_ids (mac dinh {}).

Tat ca deu offline: client Binance duoc thay bang fake qua monkeypatch, khong goi mang.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

import run_live
from src.carry_live_executor import build_cycle_audit_record

BTC = "BTC/USDT:USDT"
CLIENT_TARGET = "src.binance_signed_client.BinanceSignedTestnetClient"
ORDER_ID_KEYS = (
    "spot_entry_order_id",
    "perp_entry_order_id",
    "spot_exit_order_id",
    "perp_exit_order_id",
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeState:
    """Toi gian hoa ``CarryLiveExecutor.state`` (thuoc tinh, khong phai dict)."""

    def __init__(
        self,
        carry_positions: dict[str, Any] | None = None,
        overlay_positions: dict[str, Any] | None = None,
        current_equity: float = 0.0,
    ) -> None:
        self.carry_positions = dict(carry_positions or {})
        self.overlay_positions = dict(overlay_positions or {})
        self.current_equity = float(current_equity)


class _FakeExecutor:
    """Executor gia lap, giu dung hop dong ma run_live can.

    ``apply_reconcile_report`` mo phong ``CarryLiveExecutor.apply_reconcile_report``:
    luu report, set ``reconcile_status``/``trading_allowed`` tu report.
    """

    def __init__(
        self,
        state: _FakeState | None = None,
        symbols: list[str] | None = None,
        reconcile_status: str | None = None,
        trading_allowed: bool = True,
    ) -> None:
        self.state = state if state is not None else _FakeState()
        self.symbols = list(symbols if symbols is not None else [BTC])
        self.reconcile_status = reconcile_status
        self.trading_allowed = bool(trading_allowed)
        self.reconcile_report: dict[str, Any] | None = None
        self.testnet_order_ids: dict[str, Any] = {}

    def apply_reconcile_report(self, report: dict[str, Any] | None) -> bool:
        """Nap report doi soat; tra True khi duoc phep mo vi the moi."""
        self.reconcile_report = dict(report) if isinstance(report, dict) else None
        if self.reconcile_report is None:
            self.reconcile_status = None
            self.trading_allowed = True
            return True
        self.reconcile_status = str(self.reconcile_report.get("status") or "BLOCKED")
        self.trading_allowed = bool(self.reconcile_report.get("trading_allowed", False))
        return self.trading_allowed


def _btc_carry(
    qty: float = 0.001,
    hedge_qty: float | None = None,
    entry_spot_price: float = 82000.0,
) -> dict[str, Any]:
    """Tra ve ``carry_positions`` voi 1 vi the BTC delta-neutral."""
    hedged = float(qty if hedge_qty is None else hedge_qty)
    return {
        BTC: {
            "qty": float(qty),
            "leg2_hedged_qty": hedged,
            "entry_spot_price": float(entry_spot_price),
        }
    }


def _snapshot(
    *,
    spot_balances: dict[str, float] | None = None,
    spot_cash_usd: float = 0.0,
    perp_wallet_balance_usd: float = 0.0,
    perp_unrealized_pnl_usd: float = 0.0,
    maint_margin_usd: float = 0.0,
    margin_balance_usd: float = 0.0,
    exchange_equity_usd: float = 0.0,
    perp_positions: dict[str, float] | None = None,
    perp_positions_detail: list[dict[str, Any]] | None = None,
    snapshot_healthy: bool = True,
) -> dict[str, Any]:
    """Dung snapshot dung shape ``fetch_exchange_reconcile_snapshot`` tra ve."""
    return {
        "snapshot_healthy": snapshot_healthy,
        "spot_balances": dict(spot_balances or {}),
        "spot_cash_usd": float(spot_cash_usd),
        "perp_wallet_balance_usd": float(perp_wallet_balance_usd),
        "perp_unrealized_pnl_usd": float(perp_unrealized_pnl_usd),
        "maint_margin_usd": float(maint_margin_usd),
        "margin_balance_usd": float(margin_balance_usd),
        "exchange_equity_usd": float(exchange_equity_usd),
        "perp_positions": dict(perp_positions or {}),
        "perp_positions_detail": list(perp_positions_detail or []),
    }


def _install_preflight_client(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_factory: Callable[[], dict[str, Any]],
) -> type:
    """Patch ``BinanceSignedTestnetClient`` bang fake tra snapshot tu factory."""

    class _FakePreflightClient:
        last_client: Any = None
        last_snapshot: dict[str, Any] | None = None

        def __init__(self, dry_run: bool = False, recv_window: int = 0) -> None:
            self.dry_run = dry_run
            self.recv_window = recv_window
            self.requested_symbols: list[str] | None = None
            self.requested_prices: dict[str, float] | None = None
            type(self).last_client = self

        def sync_server_time(self, venue: str = "both") -> dict[str, Any]:
            return {
                "venue": venue,
                "spot_offset_ms": 0,
                "futures_offset_ms": 0,
                "errors": [],
            }

        def fetch_exchange_reconcile_snapshot(
            self, symbols: list[str], prices: dict[str, float]
        ) -> dict[str, Any]:
            self.requested_symbols = list(symbols)
            self.requested_prices = dict(prices)
            snap = snapshot_factory()
            type(self).last_snapshot = snap
            return snap

    monkeypatch.setattr(CLIENT_TARGET, _FakePreflightClient)
    return _FakePreflightClient


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Doc audit JSONL, tra list dict (assert file ton tai truoc do)."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


# ---------------------------------------------------------------------------
# 1. _scope_snapshot_to_symbols
# ---------------------------------------------------------------------------


def test_scope_snapshot_keeps_only_traded_base_assets() -> None:
    snapshot = {
        "spot_balances": {"BTC": 0.001, "ETH": 0.5, "SOL": 2.0, "DOGE": 100.0},
    }
    symbols = ["BTC/USDT:USDT"]

    scoped = run_live._scope_snapshot_to_symbols(snapshot, symbols)

    assert scoped is snapshot  # mutate + tra ve chinh dict dau vao
    assert scoped["spot_balances"] == {"BTC": 0.001}
    assert scoped["untracked_spot_assets"] == 3
    assert scoped["reconcile_symbol_scope"] == symbols


# ---------------------------------------------------------------------------
# 2. parse_args
# ---------------------------------------------------------------------------


def test_parse_args_testnet_flags(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"

    args = run_live.parse_args(
        ["--mode", "testnet", "--testnet-send-orders", "--audit-log", str(audit)]
    )

    assert args.mode == "testnet"
    assert args.testnet_send_orders is True
    assert args.audit_log == str(audit)
    assert args.reconcile_snapshot is None
    assert hasattr(args, "live_snapshot")
    assert args.live_snapshot is False


# ---------------------------------------------------------------------------
# 3. Preflight khong chay o paper mode
# ---------------------------------------------------------------------------


def test_signed_snapshot_preflight_returns_none_for_paper(monkeypatch) -> None:
    class _NeverConstructed:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("client must not be constructed in paper mode")

    monkeypatch.setattr(CLIENT_TARGET, _NeverConstructed)
    executor = _FakeExecutor(state=_FakeState(carry_positions=_btc_carry()))
    args = run_live.parse_args(["--mode", "paper"])

    assert run_live._run_signed_snapshot_preflight(args, executor) is None
    assert executor.reconcile_report is None
    assert executor.reconcile_status is None


# ---------------------------------------------------------------------------
# 4-7. Preflight tu dong o mode testnet
# ---------------------------------------------------------------------------


def test_signed_snapshot_preflight_auto_runs_and_scopes(monkeypatch) -> None:
    def factory() -> dict[str, Any]:
        return _snapshot(
            spot_balances={"BTC": 0.001, "ETH": 0.0},
            spot_cash_usd=82.0,
            exchange_equity_usd=82.0,
        )

    client_cls = _install_preflight_client(monkeypatch, factory)
    executor = _FakeExecutor(
        state=_FakeState(carry_positions=_btc_carry(), current_equity=82.0),
        symbols=[BTC],
    )
    # Khong co --live-snapshot va khong co --reconcile-snapshot: auto path.
    args = run_live.parse_args(["--mode", "testnet"])
    assert args.reconcile_snapshot is None
    assert args.live_snapshot is False

    report = run_live._run_signed_snapshot_preflight(args, executor)

    assert report is not None
    assert isinstance(report, dict)
    assert "status" in report
    client = client_cls.last_client
    assert client is not None
    # Fake client nhan dung universe cua executor, va snapshot da duoc scope.
    assert client.requested_symbols == executor.symbols
    assert client.requested_prices.get(BTC) == pytest.approx(82000.0)
    assert client_cls.last_snapshot is not None
    assert client_cls.last_snapshot["reconcile_symbol_scope"] == executor.symbols
    assert client_cls.last_snapshot["untracked_spot_assets"] == 1
    # Executor da nhan report (apply_reconcile_report duoc goi).
    assert executor.reconcile_report is not None
    assert executor.reconcile_status == report["status"]
    assert executor.trading_allowed == bool(report["trading_allowed"])


def test_signed_snapshot_preflight_orphan_blocks(monkeypatch) -> None:
    _install_preflight_client(
        monkeypatch, lambda: _snapshot(spot_balances={"BTC": 0.001})
    )
    executor = _FakeExecutor(state=_FakeState(), trading_allowed=True)
    args = run_live.parse_args(["--mode", "testnet"])

    report = run_live._run_signed_snapshot_preflight(args, executor)

    assert report is not None
    assert report["status"] == "ORPHAN_ON_EXCHANGE"
    assert report["trading_allowed"] is False
    assert executor.reconcile_status == "ORPHAN_ON_EXCHANGE"
    assert executor.trading_allowed is False


def test_signed_snapshot_preflight_margin_danger_blocks(monkeypatch) -> None:
    def factory() -> dict[str, Any]:
        return _snapshot(
            spot_balances={"BTC": 0.001},
            spot_cash_usd=82.0,
            perp_wallet_balance_usd=1000.0,
            exchange_equity_usd=1082.0,
            perp_positions={BTC: -0.001},
            perp_positions_detail=[
                {"symbol": BTC, "qty": -0.001, "mark_price": 82000.0}
            ],
            maint_margin_usd=600.0,
            margin_balance_usd=1000.0,
        )

    _install_preflight_client(monkeypatch, factory)
    executor = _FakeExecutor(
        state=_FakeState(carry_positions=_btc_carry(), current_equity=1082.0)
    )
    args = run_live.parse_args(["--mode", "testnet"])

    report = run_live._run_signed_snapshot_preflight(args, executor)

    assert report is not None
    assert report["status"] in ("MARGIN_DANGER", "EQUITY_MISMATCH")
    assert report["status"] == "MARGIN_DANGER"
    assert report["margin_danger"] is True
    assert report["trading_allowed"] is False
    assert executor.trading_allowed is False


def test_signed_snapshot_preflight_synced_allows(monkeypatch) -> None:
    def factory() -> dict[str, Any]:
        return _snapshot(
            spot_balances={"BTC": 0.001},
            spot_cash_usd=82.0,
            perp_wallet_balance_usd=1000.0,
            exchange_equity_usd=1082.0,
            perp_positions={BTC: -0.001},
            perp_positions_detail=[
                {"symbol": BTC, "qty": -0.001, "mark_price": 82000.0}
            ],
        )

    _install_preflight_client(monkeypatch, factory)
    executor = _FakeExecutor(
        state=_FakeState(carry_positions=_btc_carry(), current_equity=1082.0)
    )
    args = run_live.parse_args(["--mode", "testnet"])

    report = run_live._run_signed_snapshot_preflight(args, executor)

    assert report is not None
    assert report["status"] == "SYNCED"
    assert report["trading_allowed"] is True
    assert executor.reconcile_status == "SYNCED"
    assert executor.trading_allowed is True


# ---------------------------------------------------------------------------
# 8-10. _run_testnet_order_pair
# ---------------------------------------------------------------------------


def test_testnet_order_pair_disabled_returns_none(tmp_path: Path) -> None:
    executor = _FakeExecutor(reconcile_status="SYNCED", trading_allowed=True)

    paper_args = run_live.parse_args(
        ["--mode", "paper", "--testnet-send-orders", "--audit-log", str(tmp_path / "paper.jsonl")]
    )
    assert run_live._run_testnet_order_pair(paper_args, executor, [BTC]) is None

    testnet_no_send = run_live.parse_args(
        ["--mode", "testnet", "--audit-log", str(tmp_path / "testnet.jsonl")]
    )
    assert run_live._run_testnet_order_pair(testnet_no_send, executor, [BTC]) is None

    assert not (tmp_path / "paper.jsonl").exists()
    assert not (tmp_path / "testnet.jsonl").exists()


def test_testnet_order_pair_blocked_by_reconcile(tmp_path: Path, monkeypatch) -> None:
    class _ExplodingClient:
        attempts = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            type(self).attempts += 1
            raise AssertionError("client must not be constructed when reconcile blocks")

    monkeypatch.setattr(CLIENT_TARGET, _ExplodingClient)
    audit = tmp_path / "blocked.jsonl"
    executor = _FakeExecutor(
        reconcile_status="ORPHAN_ON_EXCHANGE", trading_allowed=False
    )
    args = run_live.parse_args(
        ["--mode", "testnet", "--testnet-send-orders", "--audit-log", str(audit)]
    )

    record = run_live._run_testnet_order_pair(args, executor, [BTC])

    assert record is not None
    assert record["status"] == "BLOCKED_RECONCILE"
    assert record["testnet_order_ids"] == {}
    assert executor.testnet_order_ids == {}
    assert _ExplodingClient.attempts == 0
    lines = _read_jsonl(audit)
    assert len(lines) == 1
    assert lines[0]["event"] == "testnet_order_pair"
    assert lines[0]["status"] == "BLOCKED_RECONCILE"
    assert lines[0]["testnet_order_ids"] == {}


def test_testnet_order_pair_happy_path_writes_order_ids(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakeOrderClient:
        instances: list["_FakeOrderClient"] = []

        def __init__(self, dry_run: bool = False, recv_window: int = 0) -> None:
            self.dry_run = dry_run
            self.recv_window = recv_window
            self.executed: list[dict[str, Any]] = []
            type(self).instances.append(self)

        def sync_server_time(self, venue: str = "both") -> dict[str, Any]:
            return {"venue": venue, "spot_offset_ms": 0, "futures_offset_ms": 0}

        def credential_format_report(self) -> dict[str, Any]:
            return {
                "spot": {"ok": True},
                "futures": {"ok": True},
                "all_ok": True,
            }

        def fetch_book_ticker(self, symbol: str, venue: str) -> dict[str, float]:
            return {"bid": 82000.0, "ask": 82000.1, "mid": 82000.05}

        def has_futures_credentials(self) -> bool:
            return True

        def execute_live_testnet_order_pair(self, **kwargs: Any) -> dict[str, Any]:
            self.executed.append(dict(kwargs))
            if kwargs.get("direction") == "open":
                return {
                    "status": "FILLED",
                    "spot": {"orderId": 111, "executedQty": 0.001},
                    "perp": {"orderId": 222},
                }
            return {
                "status": "FILLED",
                "spot": {"orderId": 333},
                "perp": {"orderId": 444},
            }

    monkeypatch.setattr(CLIENT_TARGET, _FakeOrderClient)
    audit = tmp_path / "orders.jsonl"
    executor = _FakeExecutor(reconcile_status="SYNCED", trading_allowed=True)
    args = run_live.parse_args(
        ["--mode", "testnet", "--testnet-send-orders", "--audit-log", str(audit)]
    )

    record = run_live._run_testnet_order_pair(args, executor, [BTC])

    assert record is not None
    assert record["status"] == "SENT"
    expected_ids = {
        "spot_entry_order_id": 111,
        "perp_entry_order_id": 222,
        "spot_exit_order_id": 333,
        "perp_exit_order_id": 444,
    }
    assert record["testnet_order_ids"] == expected_ids
    assert executor.testnet_order_ids == expected_ids

    lines = _read_jsonl(audit)
    assert len(lines) == 1
    payload = lines[0]
    assert payload["event"] == "testnet_order_pair"
    assert payload["testnet_order_ids"] == expected_ids
    for key in ORDER_ID_KEYS:
        assert key in payload["testnet_order_ids"]

    # Ca 2 chang (open + close) deu duoc gui qua client that gia lap.
    assert len(_FakeOrderClient.instances) == 1
    directions = [c["direction"] for c in _FakeOrderClient.instances[0].executed]
    assert directions == ["open", "close"]


# ---------------------------------------------------------------------------
# 11. build_cycle_audit_record
# ---------------------------------------------------------------------------


def _cycle_kwargs() -> dict[str, Any]:
    return {
        "cycle": 1,
        "mode": "testnet",
        "execution_style": "maker_then_taker",
        "capital": 100.0,
        "equity": 100.0,
        "peak_equity": 100.0,
        "drawdown_pct": 0.0,
        "carry_spot_usd": 0.0,
        "carry_perp_usd": 0.0,
        "overlay_long_usd": 0.0,
        "overlay_short_usd": 0.0,
    }


def test_build_cycle_audit_record_carries_testnet_order_ids() -> None:
    ids = {
        "spot_entry_order_id": 1,
        "perp_entry_order_id": 2,
        "spot_exit_order_id": 3,
        "perp_exit_order_id": 4,
    }

    record = build_cycle_audit_record(**_cycle_kwargs(), testnet_order_ids=ids)

    assert record["testnet_order_ids"] == ids
    assert record["mode"] == "testnet"
    assert record["cycle"] == 1

    default_record = build_cycle_audit_record(**_cycle_kwargs())
    assert default_record["testnet_order_ids"] == {}
