import sys, time, hmac, hashlib, urllib.parse, requests
sys.path.insert(0, r"D:\Quant_trading")
from src.binance_signed_client import BinanceSignedTestnetClient

client = BinanceSignedTestnetClient(dry_run=False, recv_window=10000)
client.sync_server_time("both")
print("[CRED]", client.credential_status())

def venue_creds():
    for v in ("futures", "perp"):
        try:
            return client.api_key_for(v), client.api_secret_for(v)
        except Exception:
            continue
    raise SystemExit("Không lấy được key futures từ client")

KEY, SECRET = venue_creds()
BASE = "https://testnet.binancefuture.com"

def position_risk():
    p = {"timestamp": int(time.time() * 1000), "recvWindow": 10000}
    qs = urllib.parse.urlencode(p)
    sig = hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r = requests.get(f"{BASE}/fapi/v2/positionRisk?{qs}&signature={sig}",
                     headers={"X-MBX-APIKEY": KEY}, timeout=15)
    if r.status_code != 200:
        raise SystemExit(f"LỖI positionRisk: {r.status_code} {r.text}")
    return [x for x in r.json() if float(x["positionAmt"]) != 0]

pos = position_risk()
if not pos:
    print("[FLAT] Perp đã phẳng.")
for x in pos:
    amt = float(x["positionAmt"])
    side = "SELL" if amt > 0 else "BUY"
    print(f"[FLATTEN] {x['symbol']} positionAmt={amt} -> {side} {abs(amt)}")
    res = client.place_futures_market_order(symbol=x["symbol"], side=side,
                                            quantity=abs(amt), reduce_only=True)
    print(" ->", res.get("orderId"), res.get("status"), res.get("dry_run", ""))

time.sleep(2)
rem = position_risk()
print(f"[VERIFY] Perp còn lại: {len(rem)} (ALL FLAT = {len(rem) == 0})")
for x in rem:
    print("   còn:", x["symbol"], x["positionAmt"])
sys.exit(1 if rem else 0)
