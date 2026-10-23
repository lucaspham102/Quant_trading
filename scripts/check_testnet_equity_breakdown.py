"""Script ĐỌC-CHỈ: Lấy và in chi tiết cấu thành Equity (Spot + Futures) trên Binance Testnet.

Nguyên tắc an toàn:
- CHỈ đọc (GET account), KHÔNG đặt bất kỳ lệnh nào (dry_run=True).
- KHÔNG in giá trị API key / secret (chỉ in độ dài/tiền tố ẩn danh).
- Người dùng tự chạy: python scripts/check_testnet_equity_breakdown.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Đảm bảo import được src từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binance_signed_client import (
    DEFAULT_LIVE_RECV_WINDOW_MS,
    BinanceSignedTestnetClient,
    load_dotenv_testnet,
    _is_cash_asset,
    _mask_secret,
)

# Danh sách symbol universe chuẩn để đối soát vị thế ngoài danh mục
CANONICAL_UNIVERSE_SYMBOLS = frozenset({
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT",
    "TRXUSDT", "DOTUSDT", "LTCUSDT", "ADAUSDT",
})


def main() -> int:
    print("=" * 70)
    print("BINANCE TESTNET EQUITY BREAKDOWN (READ-ONLY)")
    print("=" * 70)

    # 1. Nạp .env.testnet
    loaded = load_dotenv_testnet()
    print(f"Loaded .env.testnet: {len(loaded)} keys found")

    # 2. Khởi tạo client đọc (dry_run=True, không cho phép đặt lệnh)
    client = BinanceSignedTestnetClient(
        dry_run=True,
        recv_window=DEFAULT_LIVE_RECV_WINDOW_MS,
    )

    fmt = client.credential_format_report()
    print(f"Spot credentials ok   : {(fmt.get('spot') or {}).get('ok')}")
    print(f"Futures credentials ok: {(fmt.get('futures') or {}).get('ok')}")

    if not client.has_spot_credentials() or not client.has_futures_credentials():
        print("[ERROR] Thiếu API key/secret cho ít nhất 1 venue (Spot hoặc Futures).")
        return 1

    # 3. Đồng bộ thời gian
    sync = client.sync_server_time("both")
    print(f"Server time offsets   : spot={sync.get('spot_offset_ms')}ms, futures={sync.get('futures_offset_ms')}ms")
    print("-" * 70)

    # 4. GET Spot Account
    print("1. SPOT TESTNET ACCOUNT (/api/v3/account)")
    try:
        spot_req = client.build_signed_get_request("/api/v3/account", futures=False)
        spot_acc = client._http_request(
            "GET", spot_req["url"], spot_req.get("params", {}), spot_req.get("headers", {})
        )
    except Exception as e:
        print(f"   [ERROR] Không đọc được Spot Account: {e}")
        spot_acc = {}

    spot_balances = spot_acc.get("balances", [])
    cash_assets: dict[str, float] = {}
    coin_assets: dict[str, float] = {}

    for b in spot_balances:
        asset = str(b.get("asset", "")).strip().upper()
        free = float(b.get("free", 0.0) or 0.0)
        locked = float(b.get("locked", 0.0) or 0.0)
        total = free + locked
        if total > 0:
            if _is_cash_asset(asset):
                cash_assets[asset] = total
            else:
                coin_assets[asset] = total

    print("   [Stablecoins / Cash]:")
    total_cash_usd = 0.0
    for a, q in sorted(cash_assets.items()):
        print(f"     - {a:<8}: {q:>15.4f}")
        total_cash_usd += q
    print(f"     => Total Spot Cash: {total_cash_usd:.2f} USD")

    print("   [Non-Cash Coins (Faucet / Dust)]:")
    for a, q in sorted(coin_assets.items()):
        print(f"     - {a:<8}: {q:>15.6f}")

    print("-" * 70)

    # 5. GET Futures Account
    print("2. USDS-M FUTURES TESTNET ACCOUNT (/fapi/v2/account)")
    try:
        fut_req = client.build_signed_get_request("/fapi/v2/account", futures=True)
        fut_acc = client._http_request(
            "GET", fut_req["url"], fut_req.get("params", {}), fut_req.get("headers", {})
        )
    except Exception as e:
        print(f"   [ERROR] Không đọc được Futures Account: {e}")
        fut_acc = {}

    # Chế độ Margin (Single-Asset vs Multi-Assets)
    is_multi_asset = bool(fut_acc.get("multiAssetsMargin", False))
    margin_mode_str = "Multi-Assets Margin Mode (ENABLED)" if is_multi_asset else "Single-Asset Margin Mode (USDT only)"
    print(f"   - Margin Mode           : {margin_mode_str}")

    fut_wallet = float(fut_acc.get("totalWalletBalance", 0.0) or 0.0)
    fut_upnl = float(fut_acc.get("totalUnrealizedProfit", 0.0) or 0.0)
    fut_margin = float(fut_acc.get("totalMarginBalance", 0.0) or 0.0)
    fut_maint = float(fut_acc.get("totalMaintMargin", 0.0) or 0.0)

    print(f"   - totalWalletBalance    : {fut_wallet:>15.2f} USDT")
    print(f"   - totalUnrealizedProfit : {fut_upnl:>15.2f} USDT")
    print(f"   - totalMarginBalance    : {fut_margin:>15.2f} USDT")
    print(f"   - totalMaintMargin      : {fut_maint:>15.2f} USDT")

    # Chi tiết số dư Futures theo từng Asset
    fut_assets = fut_acc.get("assets", [])
    print("   [Futures Asset Balances]:")
    for fa in fut_assets:
        if isinstance(fa, dict):
            fa_name = str(fa.get("asset", "")).strip().upper()
            wb = float(fa.get("walletBalance", 0.0) or 0.0)
            mb = float(fa.get("marginBalance", 0.0) or 0.0)
            upnl = float(fa.get("unrealizedProfit", 0.0) or 0.0)
            if abs(wb) > 0.0 or abs(mb) > 0.0 or abs(upnl) > 0.0:
                print(f"     - {fa_name:<8}: wallet={wb:>12.4f} | margin={mb:>12.4f} | uPnL={upnl:>10.4f}")

    # Vị thế Perp (trong và ngoài universe)
    positions = fut_acc.get("positions", [])
    open_pos = [p for p in positions if abs(float(p.get("positionAmt", 0.0) or 0.0)) > 0]
    print(f"   - Open Perp Positions   : {len(open_pos)}")
    for p in open_pos:
        sym = str(p.get("symbol", "")).upper()
        amt = float(p.get("positionAmt", 0.0))
        entry_p = float(p.get("entryPrice", 0.0))
        unrealized = float(p.get("unrealizedProfit", 0.0))
        in_univ = sym in CANONICAL_UNIVERSE_SYMBOLS
        tag = "[IN UNIVERSE]" if in_univ else "[OUTSIDE UNIVERSE WARNING]"
        print(f"     * {sym:<15}: qty={amt:>10.4f} | entryPrice={entry_p:>10.2f} | uPnL={unrealized:>8.2f} USDT {tag}")

    print("=" * 70)
    print("3. TỔNG KẾT TÀI SẢN THỰC TẾ (SÀN TESTNET):")
    print(f"   - Spot Cash (USD/USDT)   : {total_cash_usd:>15.2f} USD")
    print(f"   - Futures Wallet Balance : {fut_wallet:>15.2f} USD")
    print(f"   - Futures Unrealized PnL : {fut_upnl:>15.2f} USD")
    pure_cash_equity = total_cash_usd + fut_wallet + fut_upnl
    print(f"   => PURE CASH/MARGIN EQUAL: {pure_cash_equity:>15.2f} USD (Chưa bao gồm định giá coin Spot)")
    print(f"   - Số coin phi tiền mặt   : {len(coin_assets)} tài sản (Cần được trừ khỏi đối soát)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
