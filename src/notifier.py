"""Cảnh báo Telegram chịu lỗi (Fault-Tolerant TelegramNotifier) với fallback ra logging."""
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import html
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Optional

import requests

from src.backtester import TradeRecord
from src.risk_manager import Position

logger = logging.getLogger(__name__)


def _num(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        if math.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return float(default)


def strip_html_tags(text: str) -> str:
    """Loại bỏ các thẻ HTML và unescape ký tự để gửi plain-text fallback."""
    no_tags = re.sub(r"<[^>]+>", "", str(text))
    return html.unescape(no_tags)


def _infer_trend_direction(
    live_ledger: Optional[Mapping[str, Any]],
    live_record: Optional[Mapping[str, Any]],
) -> str:
    """Xác định trạng thái nhánh Trend: FLAT, LONG, SHORT, hoặc LONG/SHORT."""
    pos_map = (
        live_ledger.get("positions_snapshot", {})
        if isinstance(live_ledger, Mapping)
        else (live_record.get("positions_snapshot", {}) if isinstance(live_record, Mapping) else {})
    )
    has_long = False
    has_short = False
    if isinstance(pos_map, Mapping):
        for pos in pos_map.values():
            if not isinstance(pos, Mapping):
                continue
            if str(pos.get("kind", "carry")) != "trend":
                continue
            qty = _num(pos.get("qty"), 0.0)
            signed = _num(pos.get("signed_usd"), qty * _num(pos.get("entry_price"), 0.0))
            if qty > 1e-12 or signed > 1e-9:
                has_long = True
            elif qty < -1e-12 or signed < -1e-9:
                has_short = True
    if not has_long and not has_short and isinstance(live_record, Mapping):
        t_gross = _num(live_record.get("trend_gross_usd"), 0.0)
        if t_gross > 1e-6:
            has_long = True
    if has_long and has_short:
        return "LONG/SHORT"
    if has_long:
        return "LONG"
    if has_short:
        return "SHORT"
    return "FLAT"


def _lookup_price(prices_map: Optional[Mapping[str, float]], symbol: str) -> float:
    """Tra cứu giá an toàn hỗ trợ cả key ngắn (SOL) và key sàn (SOL/USDT:USDT)."""
    if not prices_map:
        return 0.0
    s_clean = symbol.replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").strip()
    for k in (symbol, f"{s_clean}/USDT:USDT", f"{s_clean}/USDT", f"{s_clean}:USDT", s_clean):
        if k in prices_map and _num(prices_map.get(k), 0.0) > 0.0:
            return float(prices_map[k])
    return 0.0


def diff_cycle_executions(
    prev_positions: Optional[Mapping[str, Any]],
    curr_positions: Optional[Mapping[str, Any]],
    mid_prices: Optional[Mapping[str, float]] = None,
    fees_paid_delta: float = 0.0,
) -> list[dict[str, Any]]:
    """So sánh snapshot vị thế 2 chu kỳ liên tiếp để trích xuất danh sách lệnh đã thực thi."""
    p_map = prev_positions if isinstance(prev_positions, Mapping) else {}
    c_map = curr_positions if isinstance(curr_positions, Mapping) else {}
    all_keys = set(p_map.keys()) | set(c_map.keys())
    executions: list[dict[str, Any]] = []

    for key in sorted(all_keys):
        p = p_map.get(key, {})
        c = c_map.get(key, {})
        if not isinstance(p, Mapping):
            p = {}
        if not isinstance(c, Mapping):
            c = {}

        sym = str(c.get("symbol") or p.get("symbol") or str(key).split("|")[0]).replace("/USDT:USDT", "")
        kind = str(c.get("kind") or p.get("kind") or "overlay").lower()
        old_q = _num(p.get("qty"), 0.0)
        new_q = _num(c.get("qty"), 0.0)
        delta_q = new_q - old_q

        if abs(delta_q) <= 1e-8:
            continue

        if kind == "carry":
            sleeve = "Carry"
            side = "SCALE_UP" if new_q > old_q else "SCALE_DOWN"
        elif kind == "overlay":
            sleeve = "Overlay"
            side = "BUY" if delta_q > 0 else "SELL"
        elif kind == "trend":
            sleeve = "Trend"
            side = "BUY" if delta_q > 0 else "SELL"
        elif kind == "reversal":
            sleeve = "Reversal"
            side = "BUY" if delta_q > 0 else "SELL"
        else:
            sleeve = kind.capitalize()
            side = "BUY" if delta_q > 0 else "SELL"

        # Ưu tiên giá thị trường thực tế (current mark/mid price) lúc khớp lệnh trong chu kỳ
        market_px = _lookup_price(mid_prices, sym)
        if market_px <= 0.0:
            market_px = _num(c.get("mark_price"), 0.0)

        if market_px > 0.0:
            fill_px = market_px
        else:
            fill_px = _num(c.get("entry_price"), 0.0)
            if fill_px <= 0.0:
                fill_px = _num(p.get("entry_price"), 0.0)

        notional = abs(delta_q) * fill_px
        if notional <= 0.0:
            notional = abs(_num(c.get("signed_usd"), 0.0) - _num(p.get("signed_usd"), 0.0))

        rate = 0.0008 if kind == "carry" else 0.0004
        fee = notional * rate

        executions.append({
            "sleeve": sleeve,
            "symbol": sym,
            "side": side,
            "delta_qty": float(delta_q),
            "old_qty": float(old_q),
            "new_qty": float(new_q),
            "fill_price": float(fill_px),
            "notional": float(notional),
            "fee": float(fee),
        })

    return executions


def format_tournament_telegram_report(
    records: Sequence[Mapping[str, Any]],
    *,
    live_profile: str = "regime_omni_1k",
    live_ledger: Optional[Mapping[str, Any]] = None,
    reasons: Optional[Sequence[str]] = None,
    capital_usd: float = 1000.0,
    cycle_executions: Optional[Sequence[Mapping[str, Any]]] = None,
    prev_positions: Optional[Mapping[str, Any]] = None,
) -> str:
    """Format báo cáo Telegram HTML chuẩn Institutional Quant Desk."""
    rec_list = list(records)
    cycle = max((int(_num(r.get("cycle"), 0.0)) for r in rec_list), default=0)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    for r in rec_list:
        ts = r.get("timestamp")
        if isinstance(ts, str) and ts.strip():
            now_str = ts.replace("T", " ")[:19] + " UTC"
            break

    live_rec: Optional[Mapping[str, Any]] = next(
        (r for r in rec_list if str(r.get("profile", "")) == live_profile),
        rec_list[0] if rec_list else None,
    )
    if live_ledger is not None and live_rec is None:
        live_rec = live_ledger

    recon_status = str(
        (live_rec.get("reconcile_status") if live_rec else None)
        or (live_ledger.get("reconcile_status") if live_ledger else None)
        or "NOT_APPLICABLE"
    )
    drift_raw = (
        live_rec.get("recon_drift_usd")
        if (live_rec and live_rec.get("recon_drift_usd") is not None)
        else (live_ledger.get("recon_drift_usd") if live_ledger else None)
    )
    drift_str = "n/a" if drift_raw is None else f"{_num(drift_raw, 0.0):+.4f} USD"
    recon_badge = "🟢 SYNCED" if recon_status == "SYNCED" else f"🔴 {html.escape(recon_status)}"

    reason_tags = ", ".join(str(x) for x in (reasons or ["PERIODIC"]))
    lines: list[str] = [
        f"<b>🏆 QUANT TOURNAMENT — Cycle #{cycle}</b>",
        f"⏱ <b>Time:</b> <code>{html.escape(now_str)}</code>",
        f"🛡 <b>Reconcile:</b> {recon_badge} (<code>drift={html.escape(drift_str)}</code>)",
        f"🔔 <b>Trigger:</b> <code>{html.escape(reason_tags)}</code>",
        "",
        "<b>📊 BẢNG XẾP HẠNG TOURNAMENT</b>",
    ]

    sorted_recs = sorted(
        rec_list,
        key=lambda r: _num(r.get("pure_trading_equity_usd"), capital_usd),
        reverse=True,
    )

    table_lines = [
        f"{'Profile':<19} {'PureEq':>8} {'Ret%':>7} {'MaxDD':>6}",
        "-" * 42,
    ]
    for r in sorted_recs:
        pname = str(r.get("profile", "unknown"))
        pure_eq = _num(r.get("pure_trading_equity_usd"), capital_usd)
        ret_pct = ((pure_eq - capital_usd) / capital_usd * 100.0) if capital_usd > 0 else 0.0
        max_dd = _num(r.get("max_dd_pct"), 0.0)
        is_live = pname == live_profile
        tag = "*" if is_live else " "
        short_name = (pname + tag)[:19]
        table_lines.append(
            f"{short_name:<19} {pure_eq:>8.2f} {ret_pct:>+6.2f}% {max_dd:>5.2f}%"
        )
    lines.append("<pre>" + html.escape("\n".join(table_lines)) + "</pre>")

    # ---- KHỐI 2: ⚡ KHỚP LỆNH TRONG CHU KỲ (CYCLE EXECUTIONS)
    def _val(k: str, default: Any = None) -> Any:
        if isinstance(live_rec, Mapping) and live_rec.get(k) is not None:
            return live_rec.get(k)
        if isinstance(live_ledger, Mapping) and live_ledger.get(k) is not None:
            return live_ledger.get(k)
        return default

    src_detail: Mapping[str, Any] = live_rec or live_ledger or {}
    curr_pos_map = (
        live_ledger.get("positions_snapshot", {})
        if isinstance(live_ledger, Mapping)
        else (live_rec.get("positions_snapshot", {}) if isinstance(live_rec, Mapping) else {})
    )
    last_prices = dict(
        (live_ledger.get("last_prices") if isinstance(live_ledger, Mapping) else None)
        or (live_rec.get("last_prices") if isinstance(live_rec, Mapping) else None)
        or {}
    )

    # Xác định danh sách lệnh khớp
    executions: list[dict[str, Any]] = []
    if cycle_executions is not None:
        executions = [dict(x) for x in cycle_executions if isinstance(x, Mapping)]
    elif isinstance(src_detail.get("cycle_executions"), Sequence):
        executions = [dict(x) for x in src_detail["cycle_executions"] if isinstance(x, Mapping)]
    elif prev_positions is not None:
        executions = diff_cycle_executions(prev_positions, curr_pos_map, mid_prices=last_prices)
    elif isinstance(src_detail.get("prev_positions_snapshot"), Mapping):
        executions = diff_cycle_executions(
            src_detail["prev_positions_snapshot"], curr_pos_map, mid_prices=last_prices
        )

    lines.append("")
    if executions:
        lines.append("<b>⚡ KHỚP LỆNH TRONG CHU KỲ (CYCLE EXECUTIONS)</b>")
        for ex in executions:
            slv = str(ex.get("sleeve", "Trade"))
            side = str(ex.get("side", "ORDER")).upper()
            sym = str(ex.get("symbol", "")).replace("/USDT:USDT", "")
            dq = _num(ex.get("delta_qty"), 0.0)
            oq = _num(ex.get("old_qty"), 0.0)
            nq = _num(ex.get("new_qty"), 0.0)
            px = _num(ex.get("fill_price"), 0.0)
            notional = _num(ex.get("notional"), 0.0)
            fee = _num(ex.get("fee"), 0.0)
            lines.append(
                f"<code>- [{html.escape(slv)}] {side} {html.escape(sym)} | "
                f"Δ{dq:+.4f} ({oq:+.4f} -> {nq:+.4f}) @ ${px:.2f} (${notional:.1f}) | Fee: ${fee:.2f}</code>"
            )
    else:
        lines.append("⚡ <b>Executions:</b> <code>NONE (Hold Positions)</code>")

    # ---- KHỐI 3: 🔍 CHI TIẾT LIVE PROFILE
    pure_eq = _num(_val("pure_trading_equity_usd"), capital_usd)
    yield_eq = _num(_val("yield_stacked_equity_usd"), pure_eq)
    exch_eq = _val("exchange_recon_equity_usd")
    exch_str = f"${_num(exch_eq):.2f}" if exch_eq is not None else "n/a"

    if drift_raw is not None:
        d_val = _num(drift_raw, 0.0)
        d_bps = (d_val / capital_usd * 10000.0) if capital_usd > 0 else 0.0
        drift_text = f"{d_val:+.2f}$ ({d_bps:+.1f} bps)"
    else:
        drift_text = "n/a"

    funding_pnl = _num(_val("funding_pnl_usd"), 0.0)
    basis_pnl = _num(_val("basis_pnl_usd"), 0.0)
    carry_net = funding_pnl + basis_pnl
    overlay_pnl = _num(_val("overlay_pnl_usd"), 0.0)
    trend_pnl = _num(_val("trend_pnl_usd"), 0.0)
    fees_paid = _num(_val("fees_paid_usd"), 0.0)
    fut_lev = _num(_val("futures_leverage"), 0.0)
    trend_dir = _infer_trend_direction(live_ledger, live_rec)

    # Risk & exposure calculation
    carry_gross = 0.0
    trend_gross_pos = 0.0
    for p in curr_pos_map.values():
        if isinstance(p, Mapping):
            k = str(p.get("kind", "")).lower()
            mag = abs(_num(p.get("signed_usd"), 0.0))
            if k == "carry":
                carry_gross += mag
            elif k == "trend":
                trend_gross_pos += mag

    overlay_l = _num(_val("overlay_long_usd"), 0.0)
    overlay_s = _num(_val("overlay_short_usd"), 0.0)
    ovly_gross = overlay_l + overlay_s
    if ovly_gross <= 0.0:
        ovly_gross = sum(
            abs(_num(p.get("signed_usd"), 0.0))
            for p in curr_pos_map.values()
            if isinstance(p, Mapping) and str(p.get("kind", "")).lower() == "overlay"
        )
    t_gross = _num(_val("trend_gross_usd"), trend_gross_pos)
    gross_exp = carry_gross + ovly_gross + t_gross

    # Net delta: overlay net + trend net
    trend_net = sum(
        _num(p.get("signed_usd"), 0.0)
        for p in curr_pos_map.values()
        if isinstance(p, Mapping) and str(p.get("kind", "")).lower() == "trend"
    )
    net_delta = (overlay_l - overlay_s) + trend_net
    if overlay_l == 0.0 and overlay_s == 0.0:
        net_delta += sum(
            _num(p.get("signed_usd"), 0.0)
            for p in curr_pos_map.values()
            if isinstance(p, Mapping) and str(p.get("kind", "")).lower() == "overlay"
        )
    net_delta_pct = (net_delta / pure_eq * 100.0) if pure_eq > 0 else 0.0
    net_delta_sign = "+" if net_delta >= 0 else "-"

    lines.extend(
        [
            "",
            f"<b>🔍 CHI TIẾT LIVE PROFILE (<code>{html.escape(live_profile)}</code>)</b>",
            f"<code>- Vốn & Đối soát: Pure Eq: ${pure_eq:.2f} | Yield Eq: ${yield_eq:.2f} | Exch Eq: {exch_str} | Drift: {drift_text}</code>",
            f"<code>- Net PnL Attribution: Carry Net: {carry_net:+.2f}$ (Funding: {funding_pnl:+.2f}$ | Basis: {basis_pnl:+.2f}$) | Overlay: {overlay_pnl:+.2f}$ | Trend: {trend_pnl:+.2f}$ ({trend_dir}) | Fees: -${abs(fees_paid):.2f}</code>",
            f"<code>- Risk & Margin Telemetry: Fut Leverage: {fut_lev:.2f}x / 3.0x | Net Delta: {net_delta_sign}${abs(net_delta):.1f} ({net_delta_pct:+.1f}%) | Gross Exp: ${gross_exp:.1f} (Carry ${carry_gross:.1f} / Ovly ${ovly_gross:.1f} / Trend ${t_gross:.1f})</code>",
            "",
            "<b>📌 DANH MỤC ĐANG MỞ (ACTIVE POSITIONS)</b>",
        ]
    )

    # ---- KHỐI 4: 📌 DANH MỤC ĐANG MỞ
    pos_lines: list[str] = []
    if isinstance(curr_pos_map, Mapping):
        for key in sorted(curr_pos_map.keys()):
            pos = curr_pos_map[key]
            if not isinstance(pos, Mapping):
                continue
            sym = str(pos.get("symbol") or str(key).split("|")[0]).replace("/USDT:USDT", "")
            kind = str(pos.get("kind", "carry")).lower()
            qty = _num(pos.get("qty"), 0.0)
            entry = _num(pos.get("entry_price"), 0.0)
            signed_usd = _num(pos.get("signed_usd"), qty * entry)
            if abs(qty) <= 1e-12 and abs(signed_usd) <= 1e-9:
                continue

            if kind == "carry":
                spot_q = abs(_num(pos.get("spot_qty"), abs(qty)))
                perp_q = -abs(_num(pos.get("perp_qty"), abs(qty)))
                notional = abs(signed_usd)
                pos_lines.append(
                    f"<code>- {html.escape(sym):<5} [Carry] Spot {spot_q:+.4f} / Perp {perp_q:+.4f} (${notional:.1f}) | Net Δ: $0.0</code>"
                )
            else:
                sleeve_lbl = "Ovly" if kind == "overlay" else (kind[:5].capitalize())
                side_lbl = "LONG " if (qty > 0 or signed_usd > 0) else "SHORT"
                notional = abs(signed_usd)
                upnl = _num(pos.get("unrealized_pnl_usd"), 0.0)
                if upnl == 0.0 and entry > 0.0:
                    mid = _lookup_price(last_prices, sym)
                    if mid <= 0.0:
                        mid = _num(pos.get("mark_price"), entry)
                    if mid > 0.0:
                        signed_q = -abs(qty) if (side_lbl == "SHORT" or signed_usd < 0.0) else abs(qty)
                        upnl = signed_q * (mid - entry)
                upnl_str = f"{upnl:+.2f}$"
                pos_lines.append(
                    f"<code>- {html.escape(sym):<5} [{sleeve_lbl:<5}] Perp {side_lbl} {qty:+.4f} (${notional:.1f}) | Entry: ${entry:.2f} | uPnL: {upnl_str}</code>"
                )

    if pos_lines:
        lines.extend(pos_lines)
    else:
        lines.append("<code>- FLAT (Không có vị thế đang mở)</code>")

    return "\n".join(lines)


def positions_changed(
    prev_positions: Optional[Mapping[str, Any]],
    curr_positions: Optional[Mapping[str, Any]],
    prev_fees_usd: float = 0.0,
    curr_fees_usd: float = 0.0,
) -> bool:
    """Kiểm tra xem danh mục vị thế hoặc phí khớp lệnh có thay đổi so với chu kỳ trước không."""
    if abs(_num(curr_fees_usd, 0.0) - _num(prev_fees_usd, 0.0)) > 1e-5:
        return True
    p_map = prev_positions if isinstance(prev_positions, Mapping) else {}
    c_map = curr_positions if isinstance(curr_positions, Mapping) else {}
    if set(p_map.keys()) != set(c_map.keys()):
        return True
    for k, c_pos in c_map.items():
        p_pos = p_map.get(k)
        if not isinstance(c_pos, Mapping) or not isinstance(p_pos, Mapping):
            return True
        c_qty = _num(c_pos.get("qty"), 0.0)
        p_qty = _num(p_pos.get("qty"), 0.0)
        if abs(c_qty - p_qty) > 1e-8:
            return True
    return False


def should_send_tournament_telegram(
    *,
    cycle: int,
    is_startup: bool = False,
    trade_or_position_changed: bool = False,
    network_recovered: bool = False,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    live_profile: str = "regime_omni_1k",
    tg_report_every: int = 4,
    now_utc: Optional[datetime] = None,
    prev_utc_date: Optional[str] = None,
) -> tuple[bool, list[str]]:
    """Đánh giá 5 điều kiện gửi Telegram thông minh:
    1. Startup / Resume
    2. Trade Executed / Position Changed (trên live_profile)
    3. Network Recovered
    4. Risk / Reconcile Alert (reconcile_status != SYNCED, CB kích hoạt, hoặc status != OK)
    5. Periodic / Daily Milestone (mỗi N chu kỳ hoặc mốc 00:00 UTC chuyển ngày)
    """
    reasons: list[str] = []
    if is_startup:
        reasons.append("STARTUP_RESUME")
    if trade_or_position_changed:
        reasons.append("TRADE_OR_POSITION_CHANGED")
    if network_recovered:
        reasons.append("NETWORK_RECOVERED")

    rec_list = list(records or [])
    for r in rec_list:
        pname = str(r.get("profile", ""))
        status = str(r.get("status", "OK"))
        if status != "OK":
            reasons.append(f"EXEC_ERROR:{pname}:{status}")
        if pname == live_profile:
            recon_st = str(r.get("reconcile_status", "SYNCED"))
            if recon_st not in ("SYNCED", "NOT_APPLICABLE"):
                reasons.append(f"RECON_ALERT:{recon_st}")
            cb_st = str(r.get("circuit_breaker_state", "NORMAL"))
            if cb_st not in ("NORMAL", "DISABLED", "N/A (No-Trend)"):
                reasons.append(f"CB_ALERT:{cb_st}")

    utc_dt = now_utc or datetime.now(timezone.utc)
    cur_date_str = utc_dt.strftime("%Y-%m-%d")
    if prev_utc_date is not None and cur_date_str != prev_utc_date:
        reasons.append("DAILY_MILESTONE_00UTC")
    elif utc_dt.hour == 0 and prev_utc_date is None and not is_startup:
        reasons.append("DAILY_MILESTONE_00UTC")

    every_n = max(1, int(tg_report_every)) if tg_report_every and int(tg_report_every) > 0 else 4
    if int(cycle) > 0 and (int(cycle) % every_n == 0):
        reasons.append(f"PERIODIC_{every_n}C")

    return (len(reasons) > 0), reasons


def _parse_env_file(env_path: Optional[Path | str]) -> dict[str, str]:
    """Đọc thủ công file .env (KEY=VALUE, bỏ qua dòng trống và dòng bắt đầu bằng #)."""
    result: dict[str, str] = {}
    if env_path is None:
        return result
    try:
        path = Path(env_path)
        if not path.is_file():
            return result
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                if key:
                    result[key] = val
    except Exception as exc:  # noqa: BLE001
        logger.debug("Không thể đọc file .env (%s): %s", env_path, exc)
    return result


class TelegramNotifier:
    """Gửi thông báo qua Telegram Bot REST API kèm retry/timeout và fallback an toàn."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_sec: float = 5.0,
        max_retries: int = 2,
        env_file: Optional[Path | str] = Path(".env"),
        http_post: Optional[Callable[..., Any]] = None,
    ) -> None:
        env_vars: dict[str, str] = {}
        if not bot_token or not chat_id:
            env_vars = _parse_env_file(env_file)
            if not env_vars and env_file == Path(".env"):
                env_vars = _parse_env_file(Path(".env.testnet"))

        resolved_token = (
            bot_token
            or os.environ.get("TELEGRAM_BOT_TOKEN")
            or env_vars.get("TELEGRAM_BOT_TOKEN")
        )
        resolved_chat = (
            chat_id
            or os.environ.get("TELEGRAM_CHAT_ID")
            or env_vars.get("TELEGRAM_CHAT_ID")
        )

        self.bot_token: Optional[str] = resolved_token.strip() if resolved_token else None
        self.chat_id: Optional[str] = resolved_chat.strip() if resolved_chat else None
        self.timeout_sec: float = float(timeout_sec) if timeout_sec and float(timeout_sec) > 0 else 5.0
        self.max_retries: int = max(1, int(max_retries))
        self.env_file: Optional[Path] = Path(env_file) if env_file is not None else None
        self.http_post: Callable[..., Any] = http_post if http_post is not None else requests.post

    def _post_payload(self, url: str, payload: dict[str, Any]) -> tuple[bool, Optional[BaseException | str]]:
        last_error: Optional[BaseException | str] = None
        start_mono = time.monotonic()
        max_elapsed_sec = float(self.max_retries) * self.timeout_sec + 5.0

        for _ in range(self.max_retries):
            if time.monotonic() - start_mono > max_elapsed_sec:
                break
            try:
                response = self.http_post(url, json=payload, timeout=self.timeout_sec)
                status_code = getattr(response, "status_code", 200)
                is_ok = getattr(response, "ok", None)
                if status_code is not None and int(status_code) >= 400:
                    last_error = f"HTTP {status_code}"
                    continue
                if is_ok is False:
                    last_error = f"Telegram response not ok (status={status_code})"
                    continue
                return True, None
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        return False, last_error

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """Gửi tin nhắn Telegram; hỗ trợ parse_mode='HTML' kèm fallback về plain text nếu lỗi parse."""
        if not self.bot_token or not self.chat_id:
            logger.info("[Telegram Fallback] %s", text)
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload: dict[str, Any] = {"chat_id": self.chat_id, "text": str(text)}
        if parse_mode:
            payload["parse_mode"] = str(parse_mode)

        ok, last_error = self._post_payload(url, payload)
        if ok:
            return True

        # Fallback về plain text nếu gửi kèm parse_mode bị lỗi (ví dụ lỗi cú pháp HTML)
        if parse_mode:
            plain_text = strip_html_tags(str(text))
            fallback_payload = {"chat_id": self.chat_id, "text": plain_text}
            ok_fb, fb_error = self._post_payload(url, fallback_payload)
            if ok_fb:
                logger.info("Telegram sent via plain-text fallback after parse_mode=%s error.", parse_mode)
                return True
            last_error = fb_error or last_error

        logger.warning("Telegram send failed: %s | Message: %s", last_error, text)
        return False

    def send_tournament_report(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        live_profile: str = "regime_omni_1k",
        live_ledger: Optional[Mapping[str, Any]] = None,
        reasons: Optional[Sequence[str]] = None,
        capital_usd: float = 1000.0,
        cycle_executions: Optional[Sequence[Mapping[str, Any]]] = None,
        prev_positions: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Định dạng và gửi báo cáo giải đấu qua Telegram với parse_mode='HTML'."""
        try:
            msg = format_tournament_telegram_report(
                records,
                live_profile=live_profile,
                live_ledger=live_ledger,
                reasons=reasons,
                capital_usd=capital_usd,
                cycle_executions=cycle_executions,
                prev_positions=prev_positions,
            )
            return self.send_message(msg, parse_mode="HTML")
        except Exception as exc:  # noqa: BLE001
            logger.warning("send_tournament_report failed: %s", exc)
            return False

    def notify_trade_open(self, pos: Position, mode: str = "paper") -> bool:
        """Thông báo khi mở vị thế mới."""
        try:
            side_str = "LONG" if pos.side == 1 else "SHORT"
            msg = (
                f"[{mode.upper()}] OPEN {side_str} {pos.symbol} | "
                f"qty={pos.qty:.6f} @ {pos.entry_price:.2f} | "
                f"SL={pos.stop_loss:.2f} | TP={pos.take_profit:.2f}"
            )
            return self.send_message(msg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("notify_trade_open failed: %s", exc)
            return False

    def notify_trade_close(
        self,
        record: TradeRecord,
        equity: float,
        mode: str = "paper",
    ) -> bool:
        """Thông báo khi đóng vị thế."""
        try:
            side_str = "LONG" if record.side == 1 else "SHORT"
            msg = (
                f"[{mode.upper()}] CLOSE {side_str} ({record.exit_reason}) | "
                f"entry={record.entry_price:.2f} -> exit={record.exit_price:.2f} | "
                f"net_pnl={record.net_pnl:+.2f} USDT | equity={equity:.2f} USDT"
            )
            return self.send_message(msg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("notify_trade_close failed: %s", exc)
            return False

    def notify_error(self, context: str, error: BaseException | str) -> bool:
        """Thông báo khi xảy ra lỗi vận hành."""
        try:
            msg = f"[ERROR] {context}: {error}"
            return self.send_message(msg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("notify_error failed: %s", exc)
            return False
