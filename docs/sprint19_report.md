# Sprint 19 - Binance Testnet Live Order Execution & Atomic Leg Rollback

**Nhanh**: `agy/sprint19-testnet-live-order-execution` (tach tu `master` @ `b4f58e2`, baseline 543 tests)
**Ngay**: 2026-10-22
**Ket qua test**: `588 passed` (543 baseline + 45 test moi), 0 failed
**Verdict ky thuat**: `GO` cho Spot Testnet round-trip / `CONDITIONAL GO` cho dual-leg live

---

## 1. Muc tieu Sprint

| # | Muc tieu | Trang thai |
|---|---|---|
| 1 | Nap khoa Testnet kep (Spot + Futures) tu `.env.testnet`, khong lo secret | DONE |
| 2 | Dong bo gio may chu (`recvWindow`) truoc khi ky lenh that | DONE |
| 3 | Gui lenh THAT qua HTTP signed (Spot LIMIT_MAKER + Futures) | DONE (Spot), FUTURES bi chan boi khoa |
| 4 | Atomic Leg Rollback: chan Spot khop nhung chan Perp loi -> dong ngay | DONE (da chung minh live) |
| 5 | Doi soat danh muc sau khi vao lenh va sau khi dong lenh | DONE (Spot-only do khoa Futures loi) |
| 6 | Script round-trip 5 buoc + evidence + JSONL, khong lo khoa | DONE |
| 7 | Kiem toan doc lap (quant_auditor) va sua het khuyet tat | DONE (2 vong) |

---

## 2. Nhung gi da lam

### 2.1 `src/binance_signed_client.py` (+~1.240 dong)

- `load_dotenv_testnet()`: nap `.env.testnet` (khong ghi de bien moi truong da co).
- Khoa kep: `spot_api_key/secret`, `futures_api_key/secret`; thu tu uu tien
  tham so venue > tham so chung > bien moi truong rieng > bien moi truong legacy.
  Tham so rong `""` duoc ton trong (giu test tat dinh).
- `_mask_secret`: chi log dang `loaded=True (len=64, prefix=q0Gm***)`.
- `sync_server_time("both")`, `time_offset_ms`, `sign_params(venue=...)`, `RECV_WINDOW` 10.000ms cho live.
- `build_delta_neutral_order_pair(..., spot_use_limit_maker=False)`: giu nguyen mac dinh cu
  (`LIMIT` + `timeInForce=GTX`) de khong pha vo `tests/test_sprint16_*`; live dung `LIMIT_MAKER`
  (Spot khong duoc kem `timeInForce`).
- `submit_signed_order(..., already_signed=True)`: khong bao gio ky lai payload da co `signature`
  va **khong tra `headers`** (truoc day chinh la duong ro ri API key vao log).
- `fetch_symbol_filters()`, `quantize_price()`: sua loi `-1013 PRICE_FILTER` that gap tren Testnet.
- `execute_live_testnet_order_pair()`: Spot -> Perp -> rollback; `_rollback_filled_spot()`;
  `_await_fill()`; `_simulate_signed_perp_leg()` khi khong co khoa Futures hop le.

### 2.2 `scripts/run_testnet_live_roundtrip.py` (moi, ~890 dong)

5 buoc: (1) khoa + dong bo gio, (2) snapshot/tick/step/minNotional, (3) vao lenh that,
(4) doi soat giua chung, (5) dong lenh + doi soat sau khi dong.
Verdict `PASS` / `PASS_SPOT_ONLY` / `PARTIAL` / `FAIL`; exit code 0/0/1/2.
`build_mock_transport()` cho phep chay hoan toan offline (dung trong test).

### 2.3 `run_live.py`

Preflight Testnet nay nap dotenv, dong bo gio, kiem tra xac thuc tai khoan THAT cua tung venue
(nhung khong dat lenh - dat lenh la trach nhiem cua script round-trip).

### 2.4 Test moi (45)

| File | So test | Noi dung |
|---|---|---|
| `tests/test_sprint19_dual_auth_and_timesync.py` | 12 | khoa kep, uu tien, mask, dong bo gio, recvWindow |
| `tests/test_sprint19_http_order_and_rollback.py` | 21 | ky/gui, fallback maker, rollback, partial-fill, step_size, quantize |
| `tests/test_sprint19_live_roundtrip.py` | 12 | 5 buoc, evidence, scrub secret, scope doi soat, verdict |

---

## 3. Bang chung chay THAT tren Binance Testnet

### 3.1 Round-trip Spot thanh cong (khoa Spot hop le)

| Thoi diem (UTC) | Symbol | Verdict | Entry | Exit | Qty | Fill Price | Reconcile |
|---|---|---|---|---|---|---|---|
| 2026-10-22T06:15Z | BTC/USDT:USDT | PASS_SPOT_ONLY | 5722879 | 5722894 | 0.001 | ~82.8k | SPOT_ONLY_SYNCED |

Chi tiet day du: `docs/sprint19_testnet_live_evidence.md`, log: `logs/sprint19_testnet_roundtrip.jsonl`.

### 3.2 Atomic Leg Rollback (chung minh live truoc do)

- Entry `5706775` Spot FILLED -> chan Futures tra `-1022` -> he thong tu gui MARKET nguoc chieu
  `5706777` FILLED -> `ROLLED_BACK_SAFE` (khong con vi the tran).
- Lan chay `5719027/5719034`: `LIMIT_MAKER` khong khop trong TTL -> `CANCELED` -> `MARKET_FALLBACK`
  chi mua dung 0.001 (phan con thieu), tong khop = 0.001.

### 3.3 Khoa Futures KHONG dung duoc (blocker tu phia nguoi dung)

- `BINANCE_FUTURES_TESTNET_API_SECRET` trong `.env.testnet` dai **65 ky tu** (loi dan
  khi copy) -> `-1022 Signature for this request is not valid`.
- Hau qua: chan Perp chi chay o che do `SIMULATED_SIGNED_ONLY` (ky that bang HMAC nhung KHONG gui).
- Cach xu ly: tao khoa rieng tai https://testnet.binancefuture.com roi chay lai
  `python scripts/run_testnet_live_roundtrip.py` de co dual-leg round-trip day du.

---

## 4. Loi THAT phat hien khi chay live (khong phai gia lap)

| # | Loi | Trieu chung | Cach sua |
|---|---|---|---|
| 1 | Ky lai payload da ky | `-1022` | `already_signed=True`, khong ky lai |
| 2 | Gia lech tick | `-1013 PRICE_FILTER` | `fetch_symbol_filters` + `quantize_price` (Decimal) |
| 3 | **Ro ri API key vao JSONL** | spot key xuat hien trong log | bo `headers` khoi ket qua submit + `_sanitize` + `_scrub_secrets`, da lam sach log cu |
| 4 | LIMIT_MAKER kem `timeInForce` | `-1106` | bo `timeInForce` khi `LIMIT_MAKER` |

Kiem tra ro ri sau khi sua: `LEAKED_KEYS=NONE` tren ca file evidence va JSONL.

---

## 5. Kiem toan doc lap (2 vong)

### Vong 1 - `.agy/quant_auditor_d2673fc7.md`

| ID | Muc do | Van de | Trang thai |
|---|---|---|---|
| D.1 | BLOCKER | Partial fill + fallback MARKET -> double fill | DA SUA: chi mua phan con thieu, cong don co gioi han `min(qty, ...)` |
| D.2 | BLOCKER | `PARTIALLY_FILLED` bi coi la da khop | DA SUA: chi `status == FILLED` |
| E.1 | MAJOR | `spot_only_fallback` che giau vi the Futures | DA SUA: `perp_positions_verified=False`, status `SPOT_ONLY_*` |
| A.1 | MAJOR | Ngoai le chan Spot thoat ra truoc rollback | DA SUA: `try/except` + tracker + rollback theo khoi luong da biet |
| F.1 | MINOR | JSONL bo qua `_scrub_secrets` | DA SUA |
| H.1 | MINOR | `quantize_price` sai so float | DA SUA (Decimal, ho tro tick 0.25) |

### Vong 2 - `.agy/quant_auditor_11e1331b.md`

| ID | Muc do | Van de | Trang thai |
|---|---|---|---|
| NEW-1 | MAJOR | Poll lenh MARKET lam mat khoi luong maker da khop | DA SUA: poll tren ket qua THO roi moi cong don |
| NEW-2 | MAJOR | `spot_tracker` khong cap nhat sau huy / sau fallback | DA SUA: `_track()` sau query va sau cong don |
| NEW-3 | MINOR | Verdict `PASS` khi snapshot la spot-only du khoa Futures that | DA SUA: PASS chi khi `exit_reconcile_status == SYNCED` **va** `futures_positions_verified` |
| NEW-4 | MINOR | `quantize_price` sai voi tick khong phai luy thua 10 | DA SUA: Decimal + ROUND_FLOOR/CEILING |

**Khuyen nghi cua kiem toan**: `CONDITIONAL GO` - duoc phep Spot round-trip, KHONG tu dong
dual-leg/mainnet. Sprint nay khong bat mainnet.

---

## 6. Ket qua kiem thu thuc te

```
python -m pytest -q                     -> 588 passed
python scripts/run_testnet_preflight.py -> PREFLIGHT: 6/6 PASS
python scripts/run_shadow_burnin.py     -> SUMMARY: 10/10 PASS
python scripts/run_e2e_fire_drill.py    -> Tong: 20/20 checks PASS
python scripts/run_testnet_live_roundtrip.py --mock-transport -> Verdict: PASS (5/5 buoc)
python scripts/run_testnet_live_roundtrip.py                  -> Verdict: PASS_SPOT_ONLY
```

---

## 7. Chua chac chan / con lai

1. **Khoa Futures loi** (65 ky tu) - nguoi dung can tao khoa moi. Day la yeu cau DUY NHAT con lai
   de co bang chung dual-leg day du. Khong the sua tu phia code.
2. **Chua test duoc dual-leg tren Testnet** vi ly do (1); chan Perp moi o muc "ky that, khong gui".
3. **BLOCKER sprint 11/12 (warmup 504h, lech pha rebalance 48h)** do kiem toan neu lai: day la
   khuyet tat CO SAN tu sprint truoc, ngoai pham vi Sprint 19, **khong sua** trong sprint nay.
   Can mot sprint rieng de xu ly (co anh huong den Jaccard/PnL nghien cuu).
4. `.env.testnet` chi ton tai tren may nay va da duoc `.gitignore` chan (`.env.*`).

---

## 8. Lenh tai hien

```powershell
# Offline (khong can mang, khong can khoa)
python scripts/run_testnet_live_roundtrip.py --mock-transport --no-write

# Live Spot Testnet (can .env.testnet hop le)
$env:PYTHONIOENCODING="utf-8"
python scripts/run_testnet_live_roundtrip.py

# Toan bo cong kiem thu
python -m pytest -q
python scripts/run_testnet_preflight.py
python scripts/run_shadow_burnin.py
python scripts/run_e2e_fire_drill.py
```
