# Binance Testnet Live Roundtrip Evidence

| Timestamp (UTC) | Symbol | Verdict | Entry Order ID | Exit Order ID | Exec Qty | Fill Price | Entry Rec | Exit Rec | Spot Offset (ms) | Perp Offset (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-22T06:13:49.255764+00:00 | BTC/USDT:USDT | PASS_SPOT_ONLY | 5719354 | 5719358 | 0.001 | 82808.52 | SPOT_ONLY_SYNCED | SPOT_ONLY_SYNCED | 1260 | 1087 |
| 2026-10-22T06:13:50.171634+00:00 | BTC/USDT:USDT | PASS | 1001 | 1002 | 0.001 | 95990.0 | SYNCED | SYNCED | 123 | 456 |
| 2026-10-22T06:29:24.968667+00:00 | BTC/USDT:USDT | PASS_SPOT_ONLY | 5722879 | 5722894 | 0.001 | 82598.01 | SPOT_ONLY_SYNCED | SPOT_ONLY_SYNCED | 1101 | 1115 |

### Run Detail (2026-10-22T06:29:24.968667+00:00)
- **Symbol**: `BTC/USDT:USDT`
- **Verdict**: `PASS_SPOT_ONLY`
- **Spot Auth**: `True` (loaded=True (len=64, prefix=q0Gm***))
- **Futures Auth**: `False` (loaded=True (len=64, prefix=v6ew***))
- **Entry Order**: `5722879` (Qty: 0.001, Price: 82598.01)
- **Exit Order**: `5722894` (Qty: 0.001)
- **Entry Reconcile Status**: `SPOT_ONLY_SYNCED`
- **Exit Reconcile Status**: `SPOT_ONLY_SYNCED`


### Run Detail (2026-10-22T06:13:50.171634+00:00)
- **Symbol**: `BTC/USDT:USDT`
- **Verdict**: `PASS`
- **Spot Auth**: `True` (loaded=True (len=18, prefix=mock***))
- **Futures Auth**: `True` (loaded=True (len=21, prefix=mock***))
- **Entry Order**: `1001` (Qty: 0.001, Price: 95990.0)
- **Exit Order**: `1002` (Qty: 0.001)
- **Entry Reconcile Status**: `SYNCED`
- **Exit Reconcile Status**: `SYNCED`



### Run Detail (2026-10-22T06:13:49.255764+00:00)
- **Symbol**: `BTC/USDT:USDT`
- **Verdict**: `PASS_SPOT_ONLY`
- **Spot Auth**: `True` (loaded=True (len=64, prefix=q0Gm***))
- **Futures Auth**: `False` (loaded=True (len=64, prefix=v6ew***))
- **Entry Order**: `5719354` (Qty: 0.001, Price: 82808.52)
- **Exit Order**: `5719358` (Qty: 0.001)
- **Entry Reconcile Status**: `SPOT_ONLY_SYNCED`
- **Exit Reconcile Status**: `SPOT_ONLY_SYNCED`

