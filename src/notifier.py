"""Cảnh báo Telegram chịu lỗi (Fault-Tolerant TelegramNotifier) với fallback ra logging."""
import logging
import os
from collections.abc import Callable
import time
from pathlib import Path
from typing import Any, Optional

import requests

from src.backtester import TradeRecord
from src.risk_manager import Position

logger = logging.getLogger(__name__)


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

    def send_message(self, text: str) -> bool:
        """Gửi tin nhắn Telegram; tuyệt đối không ném ngoại lệ khi thiếu cấu hình hoặc lỗi mạng."""
        if not self.bot_token or not self.chat_id:
            logger.info("[Telegram Fallback] %s", text)
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": str(text)}
        last_error: Optional[ BaseException | str ] = None
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
                return True
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        logger.warning("Telegram send failed: %s | Message: %s", last_error, text)
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
