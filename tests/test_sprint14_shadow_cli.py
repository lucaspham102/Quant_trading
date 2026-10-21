"""Test Sprint 14 Chang 3 - wiring CLI cho che do shadow-live (fail-closed).

Bao phu:
1. `--mode shadow-live` duoc argparse chap nhan, kem `--shadow-timeout-sec` va `--offline-fixture`.
2. Preflight voi fixture healthy -> True va KHONG goi mang.
3. Preflight voi book crossed -> False (fail-closed).
4. Preflight voi file fixture khong ton tai -> False (khong crash ra ngoai).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_live
import src.binance_public_feed


def test_parse_args_accepts_shadow_live_mode() -> None:
    args = run_live.parse_args(["--mode", "shadow-live", "--strategy", "barbell"])

    assert args.mode == "shadow-live"
    assert args.shadow_timeout_sec == pytest.approx(5.0)
    assert args.offline_fixture is None


def test_shadow_preflight_passes_with_healthy_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("network must not be touched in fixture mode")

    monkeypatch.setattr(src.binance_public_feed.urllib.request, "urlopen", _boom)

    fixture = {
        "BTC/USDT:USDT": {
            "bid": 100.0,
            "ask": 100.1,
            "mark_price": 100.05,
            "funding_rate": 0.0001,
        }
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    args = run_live.parse_args(["--mode", "shadow-live", "--offline-fixture", str(path)])

    assert run_live._run_shadow_live_preflight(args, ["BTC/USDT:USDT"]) is True


def test_shadow_preflight_fails_closed_on_crossed_book(tmp_path: Path) -> None:
    fixture = {"BTC/USDT:USDT": {"bid": 101.0, "ask": 100.0}}
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    args = run_live.parse_args(["--mode", "shadow-live", "--offline-fixture", str(path)])

    assert run_live._run_shadow_live_preflight(args, ["BTC/USDT:USDT"]) is False


def test_shadow_preflight_fails_closed_on_missing_fixture_file(tmp_path: Path) -> None:
    args = run_live.parse_args(
        ["--mode", "shadow-live", "--offline-fixture", str(tmp_path / "nope.json")]
    )

    assert run_live._run_shadow_live_preflight(args, ["BTC/USDT:USDT"]) is False
