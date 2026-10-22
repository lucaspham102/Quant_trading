# SPRINT 20 REPORT — DUAL-VENUE LIVE TESTNET ROUND-TRIP (SPOT + FUTURES), LEGACY QUANT DEBT CLOSURE & DAEMON TESTNET WIRING

- **Nhanh**: `agy/sprint20-dual-leg-live-and-debt-closure` (tach tu `master`, baseline 588 passed)
- **Ngay**: 2026-10-22
- **Ket qua cuoi**: **616 passed** (2 lan lien tiep), live round-trip `PASS (Dual-Leg Live Verified)`, `quant_auditor` ket luan **UNCONDITIONAL GO (0 Blocker)**
- **Bao mat**: khong in/log gia tri `API_KEY` / `API_SECRET`; quet 4 khoa 64 ky tu tren toan bo git tree = `NONE`

---

## 1. CHANG 1 — KIEM DINH KHOA 64 KY TU & CHAY THAT 2 CHAN TREN BINANCE TESTNET

### 1.1. API moi trong `src/binance_signed_client.py`

| Thanh phan | Mo ta |
|---|---|
| `CREDENTIAL_EXPECTED_LENGTH = 64` | Do dai chuan HMAC-SHA256 cua Binance |
| `_clean_credential(value)` | Strip khoang trang / xuong dong / dau nhay boc `"` `'`; ap dung cho MOI duong nap khoa (tham so, env, `.env.testnet`) |
| `validate_credential_format(key, secret, venue) -> dict` | Kiem tra `len == 64` + `isalnum()`; tra `ok`, `key_length`, `secret_length`, `diagnostics`, `message` (khong chua gia tri khoa) |
| `BinanceSignedTestnetClient.credential_format_report()` | Chan doan dong thoi Spot + Futures: `{spot: {...}, futures: {...}, all_ok, dual_venue_ok}` |

Ma chan doan chinh xac (vi du khi secret 65 ky tu):
`spot_SECRET_INVALID_LENGTH_65_EXPECTED_64` — bao ngay TRUOC khi gui request mang, thay vi
nhan `-2015/-1022` mo ho tu san.

### 1.2. Bang chung chay that (`docs/sprint20_dual_leg_live_evidence.md`)

Lenh: `python scripts/run_testnet_live_roundtrip.py`

| Hang muc | Ket qua |
|---|---|
| Credential format | `spot_CREDENTIAL_FORMAT_OK`, `futures_CREDENTIAL_FORMAT_OK` (64/64) |
| Spot auth (`/api/v3/account`) | `True` |
| Futures auth (`/fapi/v2/account`) | `True` |
| Server time sync | spot offset **+1099 ms**, futures offset **+971 ms** |
| Spot entry order | **5746860** (BUY 0.001 BTC @ 82244.46) |
| Perp entry order | **28645525428** (SELL 0.001 BTC) |
| Spot exit order | **5746861** (SELL 0.001 BTC) |
| Perp exit order | **28645525452** (BUY 0.001 BTC) |
| Mid-trip reconcile | `SYNCED` (items=1, equity_gap=0.00 USD) |
| Post-flatten reconcile | `SYNCED` (items=1, equity_gap=0.00 USD) |
| **Verdict** | **PASS** (`verdict_scope = dual_leg_live_verified`) |

Sua loi khien Sprint 19 chi dat `PARTIAL`: `fetched_exchange_reconcile_snapshot` tra ve toan bo
~500 tai san "dust" ma faucet testnet cap cho tai khoan demo, nen doi soat toan tai khoan luon bao
`ORPHAN_ON_EXCHANGE` (vo nghia). Sprint 20 thu hep pham vi doi soat ve **symbol dang giao dich**
(base + cash), dong thoi ghi lai `untracked_spot_assets` de hau kiem — khong che giau vi the mo coi
that tren symbol dang giao dich (auditor da xac nhan rieng, xem muc 4).

---

## 2. CHANG 2 — DONG NO KY THUAT CU (CANONICAL DEFAULTS)

### 2.1. Thay doi trong `src/sprint11_eval.py`

```python
CANONICAL_SATURATED_WARMUP_HOURS: int = 720   # chuan bao hoa moi
LEGACY_SPRINT14_WARMUP_HOURS: int = 504       # chi de doi chieu lich su
SPRINT14_WARMUP_HOURS = LEGACY_SPRINT14_WARMUP_HOURS  # alias tuong thich nguoc

def run_indicator_warm_cash_start(
    ...,
    warmup_hours: int = CANONICAL_SATURATED_WARMUP_HOURS,
    align_rebalance_clock: bool = True,
    phase_aligned: bool | None = None,   # alias Sprint 15: neu khac None thi ghi de
    ...,
)
```

`aligned = align_rebalance_clock if phase_aligned is None else bool(phase_aligned)`; payload tra ve
ca `phase_aligned` va `align_rebalance_clock` de khong vo test cu.

### 2.2. Cac diem goi LICH SU duoc truyen tuong minh (khong con dua vao default)

| File | D (504h, unaligned) | D' (504h, aligned) | E (720h, aligned) |
|---|---|---|---|
| `scripts/gen_warm_cash_bridge.py` | `warmup_hours=504, align_rebalance_clock=False` | — | — |
| `scripts/gen_clock_phase_bridge.py` | co | co | — |
| `scripts/gen_sprint16_attribution.py` | co | co | — |
| `scripts/forensic_overlay_divergence.py` | co | co | `warmup_hours=CANONICAL_SATURATED_WARMUP_HOURS` |
| `tests/test_sprint15_clock_alignment.py` | co | co | — |

### 2.3. Bat bien so lieu (artifact `data/forensics_sprint17/bridge_sprint17.json`)

| Phep do | warmup | net PnL | Ghi chu |
|---|---|---|---|
| D (legacy) | 504h unaligned | **81.7274 USD** | giu nguyen so da cong bo Sprint 14 |
| D' (legacy) | 504h aligned | **76.0940 USD** | giu nguyen so da cong bo Sprint 15 |
| E (canonical) | 720h aligned | **114.2663 USD** | chuan moi, Jaccard(B, E) = **0.992** >= 0.98 |
| Accounting identity | — | `identity_gap_usd = 0.0` | `unexplained_total_usd = 0.0` |

Goi `run_indicator_warm_cash_start(data)` KHONG truyen tham so gio tuong duong phuong phap E
(720h + aligned clock) — dung yeu cau "mac dinh dat chuan".

---

## 3. CHANG 3 — NOI LUONG TESTNET VAO DAEMON `run_live.py`

1. `--mode testnet` luon: nap `.env.testnet` (`load_dotenv_testnet`), `sync_server_time("both")`,
   kiem dinh dinh dang 64 ky tu (`credential_format_report`), fail-closed khi dinh dang sai.
2. `_run_signed_snapshot_preflight` **tu dong chay** trong `--mode testnet` khi khong truyen
   `--reconcile-snapshot`: lay snapshot song co ky HMAC (`fetch_exchange_reconcile_snapshot`) roi
   `reconcile_portfolio_state` truoc khi vao chu ky giao dich.
3. `_scope_snapshot_to_symbols`: thu hep `spot_balances` ve universe dang chay + ghi
   `untracked_spot_assets` (cung ly do dust testnet nhu muc 1.2).
4. `_run_testnet_order_pair`: khi co `--testnet-send-orders`, gui 1 cap lenh Delta-Neutral THAT
   (Spot BUY + Perp SELL) roi dong lai; **chi gui khi reconcile khong BLOCKED** (fail-closed).
   4 order id that duoc ghi vao `--audit-log` duoi khoa `testnet_order_ids`.
5. `src/carry_live_executor.py`: `build_cycle_audit_record(..., testnet_order_ids=...)` — moi dong
   JSONL cua chu ky mang theo `testnet_order_ids` (mac dinh `{}`).

---

## 4. CHANG 4 — KIEM TOAN DOC LAP & CONG KIEM DINH

### 4.1. `quant_auditor` (`.agy/quant_auditor_bbbd8aed.md`)

| No cu | Phan quyet | Bang chung |
|---|---|---|
| BLOCKER 1: warmup 504h < 720h | **FIXED** | `src/sprint11_eval.py` (`CANONICAL_SATURATED_WARMUP_HOURS=720`), Jaccard(B,E)=0.992 |
| BLOCKER 2: rebalance clock phase shift | **PARTIALLY FIXED** — da dat o luong chay (call-site canonical), con no refactor tang ham | `src/sprint11_eval.py#L468-L505`; `src/barbell_portfolio.py#L409-L411` |

- **So BLOCKER toan hoc/tai chinh con lai: 0**
- **Ket luan: `UNCONDITIONAL GO (0 Blocker)`**
- Auditor xac nhan rieng: (i) thu hep pham vi reconcile **khong** che giau orphan that tren symbol
  dang giao dich; (ii) doi mac dinh 720h/aligned **khong** lam sai lech so lieu lich su D/D'.
- Khuyen nghi tiep (khong phai blocker): (a) refactor `build_rebalance_mask` sang UTC epoch tuyet doi
  de khong phu thuoc `phase_offset` truyen ngoai; (b) canh bao khi gia tri `untracked_spot_assets`
  vuot nguong; (c) thay polling REST `query_order` bang WebSocket User Data Stream.

### 4.2. Cong kiem dinh (chay that)

| Cong | Ket qua |
|---|---|
| `python scripts/run_testnet_live_roundtrip.py` | `PASS` (dual-leg live verified, 4 order id that) |
| `python scripts/run_testnet_preflight.py` | **6/6 PASS** |
| `python scripts/run_shadow_burnin.py` | **10/10 PASS** |
| `python scripts/run_e2e_fire_drill.py` | **20/20 PASS** |
| `pytest -q` (lan 1) | **616 passed**, 0 failed |
| `pytest -q` (lan 2) | **616 passed**, 0 failed |
| Quet ro ri khoa (4 khoa 64 ky tu tren 151 file tracked) | **NONE** |

### 4.3. Test moi (28 tests)

| File | So test | Noi dung |
|---|---|---|
| `tests/test_sprint20_credential_validation.py` | 9 | strip quote/whitespace, bat 65 ky tu, charset, missing, dual-venue report, dotenv parsing, helper cua script |
| `tests/test_sprint20_canonical_defaults.py` | 8 | hang so, default signature 720h/aligned, offset % 48, legacy 504h, alias override, artifact E = 114.27, accounting identity |
| `tests/test_sprint20_daemon_testnet_flow.py` | 11 | scope snapshot, auto live snapshot o mode testnet, BLOCKED khi `ORPHAN_ON_EXCHANGE`/`MARGIN_DANGER`, cho phep khi `SYNCED`, gui lenh that + audit `testnet_order_ids` |

Baseline 588 + 28 = **616**.

---

## 5. CON CHUA CHAC CHAN (residual)

1. `build_rebalance_mask` van tinh mask theo chi so mang cuc bo + `phase_offset` (khong phai epoch
   UTC). Voi duong chay canonical hien tai (offset duoc tinh tu `global_start_idx % 48`) ket qua
   trung khop backtest toan chuoi, nhung neu mot module tuong lai goi truc tiep
   `run_barbell_portfolio` tren DataFrame bi cat lat ma quen truyen offset — hoac du lieu 1h bi
   khuyet nen — thi van co the lech pha. Auditor xep day la khuyen nghi refactor, khong phai blocker.
2. Pham vi reconcile cua daemon dang gioi han o universe dang chay. Voi universe lon hon
   (`UNIVERSE_12_SYMBOLS`), can mo rong `_scope_snapshot_to_symbols` theo danh muc day du va them
   canh bao khi gia tri tai san ngoai pham vi vuot nguong.
3. Ket qua live round-trip phu thuoc thanh khoan testnet tai thoi diem chay: spot maker Post-Only
   co the bi huy TTL va fallback MARKET (lan chay nay xay ra 1 lan o chang exit) — verdict van PASS
   vi co che fallback + doi soat sau do dat `SYNCED`.
4. `.env.testnet` khong duoc track (dung); hai bien fallback chung
   `BINANCE_TESTNET_API_KEY/SECRET` dang la placeholder tro sang ten bien Spot (28/31 ky tu) — khong
   phai khoa that, nhung nen xoa de tranh nham lan khi debug.
