# Sprint 16 - Overlay neutrality, phan ra PnL theo nhanh va adapter testnet co ky

Nhanh: `agy/sprint16-neutrality-and-testnet-adapter` (tach tu `master` = `388ef13`, baseline 432 passed).
Muc tieu: (1) xoa ro ri delta rong cua sleeve overlay sau khi lam tron lot-size, (2) phan ra khoang
cach B - D' theo nhanh carry/overlay + ty le giai ngan + cua so 30d/151d, (3) adapter ky lenh
Binance HMAC-SHA256 cho testnet kem audit log JSONL, (4) chay lai paper/burn-in/fire drill va bao cao.

## 0. Ket qua tong hop

| Hang muc | Ket qua |
|---|---|
| Overlay neutrality (guard sau lam tron lot) | Dat - `NetDelta` tu **-76.62** xuong **< 1 USD** tren paper that |
| Phan ra PnL theo nhanh (A/B/D/D') | Dat - dang thuc carry + overlay = net khop `<= 1e-11` |
| Ty le giai ngan carry | Dat - A/B 3.63/4 slot (90.7%), D/D' 3.60/4 slot (89.9%) |
| Cua so 30d vs 151d | Dat - 30d dau = 40.9% khoang cach B - D' (hysteresis cash drag) |
| Client ky lenh testnet | Dat - HMAC-SHA256 + lot-size validate, fail-closed khi thieu key |
| Audit JSONL moi chu ky | Dat - `logs/sprint16_paper_audit.jsonl`, 2/2 chu ky |
| Shadow-live burn-in (mang that) | 10/10 PASS, 8 symbol, spread toi da 1.22 bps |
| Fire drill | 20/20 PASS |
| Test suite | **459 passed** (x2: 47.24s / 40.74s) |

## 1. Ro ri delta rong cua overlay (Chặng 1)

### Nguyen nhan

`build_barbell_allocations` tinh `qty = notional / price` roi lam tron xuong `step_size`
theo tung ma. Voi 1 chan long BTC va 1 chan short DOGE (hoac ETH), sai so lam tron
nhieu nhat 1 step cua moi ben - voi BTC step 0.001 tai gia ~96,558 USD la **96.5 USD/step**,
lon hon nhieu so voi von overlay (~330-410 USD). Ket qua paper Sprint 15:

- Long 331.06 vs Short 407.68 -> **NetDelta = -76.62 USDT** (23% von sleeve).

### Cach sua

- `src/barbell_portfolio.py`
  - `_floor_to_step(qty_abs, step)`: lam tron xuong theo step, 8 chu so thap phan khi step < 0.001.
  - `enforce_overlay_dollar_neutrality(target_notional_by_symbol, prices, lot_specs,
    max_imbalance_usd=10.0, min_viable_leg_usd=50.0)`: floor tung ma -> loai leg < `min_notional`
    -> neu mot chan < 50 USD thi tra **0.0 cho tat ca** (chong naked leg) -> neu lech > nguong
    thi scale chan lon roi vong lap <= 10,000 buoc ("giam 1 step chan lon" **hoac** "tang 1 step
    chan nho", chon buoc lam gap nho nhat va chi nhan khi gap giam that).
  - *Deviation co chu y so voi plan architect*: plan chi cho phep giam chan lon; thuc te chan BTC
    don le co 1 step = 96.5 USD nen se overshoot - phai cho phep tang chan nho moi dat `|gap| <= 10`.
- `src/carry_live_executor.py`: nhanh tier1b cua `build_barbell_allocations` dung gia thuc te
  (`prices`) + notional muc tieu (`w * equity_now`) -> goi guard -> allocation dung `abs(qty)`;
  them `overlay_max_imbalance_usd`, `overlay_min_viable_leg_usd`, `last_overlay_neutrality`,
  `last_overlay_delta`; dashboard hien `Overlay (Long=... vs Short=... | NetDelta=...)`.

### Bang chung

- Guard thu cong: `{BTC:+331.06, ETH:-407.68}` tai BTC 96,558 / ETH 4,200
  -> `BTC 0.003 (+289.67), ETH -0.07 (-294.00)`, gap **4.33 USD** (truoc do 76.62).
- Truong hop von nho khong the trung tinh: tra 0.0 cho ca 2 chan (khong de lai naked leg).
- Paper that (`--cycles 2`): dashboard `Overlay (Long=333.61 vs Short=333.54 | NetDelta=0.07)`.
- Test: `tests/test_sprint16_overlay_neutrality.py` (9 test).

## 2. Phan ra PnL theo nhanh, utilization, cua so 30d/151d (Chặng 2)

`src/sprint11_eval.py`:

- `compute_sleeve_attribution(attribution_series, holdings_log, ...)`: tach
  `carry_net = funding_pnl + basis_pnl - carry_fees`, `overlay_net = overlay_pnl`,
  `avg_carry_slots_held` / `carry_utilization_pct` tu `holdings_log.n_active`,
  cua so `first_30d` (720h) / `last_151d`.
- Bridge `compute_holdout_accounting_bridge` gan `sleeve_attribution` cho A/B/C/D/D' va xuat khoi
  tong hop `sleeve_attribution` gom `methods`, `identity` (sleeve/window gap), `delta_b_vs_d_aligned`.
- `scripts/gen_sprint16_attribution.py` tai lap lai A (reinvest) va B (fixed-base), D va D'
  (phase-aligned 48h) tren chuoi 2023-04-01 -> 2026-10-18.

Ket qua (`docs/sprint16_sleeve_and_utilization_audit.md`, `data/bridge_sprint16/attribution.json`):

| Phuong phap | Net | Carry | Overlay | Khop |
|---|---:|---:|---:|---:|
| A (compound) | 159.15 | 55.62 | 103.53 | -2.8e-14 |
| B (fixed-base) | 118.50 | 46.16 | 72.34 | 0.0 |
| D (cold-start) | 81.73 | 41.01 | 40.72 | 2.9e-11 |
| D' (phase-aligned 22h) | 76.09 | 41.01 | 35.09 | 2.4e-11 |

- A/B tai lap dung payload Sprint 11 (159.15 / 118.50) -> generator khong lech nguon.
- **B - D' = 42.41 USD**: carry 5.15, overlay 37.25; 30 ngay dau 17.35 (40.9%), 151 ngay sau 25.06.
- Pure inventory carryover sau phi mo vi the: 42.41 - 2.02 = **40.39 USD** - khop voi con so
  Sprint 15 (dong ho 48h), tuc la 13.5% khoang cach B - C di qua overlay chu khong phai carry.
- Utilization: A/B 3.63/4 slot (90.7%), D/D' 3.60/4 (89.9%) -> khong phai nut that co (carry
  luon gan day du 4 slot), xac nhan lai ket luan "carry lai cham vi clock/hysteresis, khong phai
  vi thieu slot".
- Test: `tests/test_sprint16_sleeve_attribution.py` (8 test).

## 3. Adapter testnet co ky + audit JSONL (Chặng 3)

- `src/binance_signed_client.py`
  - `sign_binance_payload(params, secret)`: HMAC-SHA256 tren query string da **sap xep key**,
    so duoc format thap phan (khong dung `e-05`) de chu ky khop body.
  - `BinanceSignedTestnetClient`: doc `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET`;
    `dry_run=True` mac dinh; thieu key ma `dry_run=False` -> `BinanceCredentialsMissingError`
    (**fail-closed**); `recvWindow=5000`; endpoint `testnet.binance.vision` (Spot) +
    `testnet.binancefuture.com` (USDⓈ-M).
  - `build_delta_neutral_order_pair(...)`: Spot BUY (LIMIT + `GTX` post-only khi co gia) va
    Perp SELL MARKET, validate `step_size` + `min_notional` truoc khi ky.
- `run_live.py`: them `--audit-log <path>`, `--dry-run-sign`, `--testnet-send-orders`
  (mac dinh TAT); mode `testnet` chay paper executor + `_run_testnet_signing_preflight`
  (chi ky/dong goi, khong gui lenh).
- `src/carry_live_executor.py`: `audit_log_path` + `build_cycle_audit_record` / `append_audit_record`
  -> moi chu ky ghi 1 dong JSONL (`schema=sprint16.cycle.v1`) gom equity, drawdown, notional
  carry 2 chan, overlay long/short + `overlay_net_delta_usd`, `overlay_neutrality`, kill-switch,
  diagnostics va actions. Loi ghi audit khong lam hong chu ky.
- Bang chung paper that: `logs/sprint16_paper_audit.jsonl` 2 dong, cycle 1
  `overlay_net_delta_usd = 0.0700`, `neutral_qty = {BTC: 0.004, DOGE: -4074.0}`.
- Test: `tests/test_sprint16_signed_client_and_audit_log.py` (10 test).

## 4. Kiem thu tong the (Chặng 4)

| Lenh | Ket qua |
|---|---|
| `python run_live.py --mode paper --strategy barbell --capital 3000 --reset-state --cycles 2 --interval-sec 1 --audit-log logs/sprint16_paper_audit.jsonl` | EXIT 0, 2 chu ky, khong ERROR |
| `python scripts/run_shadow_burnin.py` | 10/10 PASS (8 symbol, spread max 1.22 bps) |
| `python scripts/run_e2e_fire_drill.py` | 20/20 PASS |
| `python -m pytest -q` (lan 1) | 459 passed / 47.24s |
| `python -m pytest -q` (lan 2) | 459 passed / 40.74s |

## 5. Con lai / rui ro

- Guard chi bao dam trung tinh **sau** lam tron; neu von overlay < 2 x `min_viable_leg_usd` (100 USD)
  thi sleeve bi tat hoan toan (dung y do - tranh naked leg), can luu y khi ha capital.
- Adapter testnet chua gui lenh that: `--testnet-send-orders` moi bat, va chua co test tich hop
  voi API that (chi vector HMAC + validate payload).
- Verdict Sprint 11 van **NO-GO** (D' 2.54% < 3.5%); Sprint 16 khong thay doi tham so Frozen Alpha.
