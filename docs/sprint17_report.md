# Sprint 17 - Kiem toan phap y overlay, nguong leg thich ung va doi soat trang thai san

Nhanh: `agy/sprint17-overlay-forensics-and-reconciler` (tach tu `master` = `c740f5e`, baseline **459 passed**).
Muc tieu: (1) tim nguyen nhan that cua khoang cach PnL nhanh Overlay giua B / C / D' va chuan hoa
phep do `E_true_fixed_warm_cash`, (2) nguong toi thieu moi chan overlay thich ung theo von de chay
duoc testnet von nho, (3) module doi soat trang thai local vs san (`src/exchange_reconciler.py`)
kem co che fail-closed `--reconcile-snapshot`, (4) chay lai paper / burn-in / fire drill va bao cao.

## 0. Ket qua tong hop

| Hang muc | Ket qua |
|---|---|
| Kiem toan phap y overlay (bar-by-bar) | Dat - dang thuc phan ra khop `<= 1e-13` USD, 4 gia thuyet co verdict |
| Nguyen nhan khoang cach overlay B - D' | Xac dinh: **H3 warmup EMA chua hoi tu** (weight effect -35.82 / -37.41 = 95.7%) |
| Phep do chuan `E_true_fixed_warm_cash` | Dat - E = **114.27 USD** = **3.81%** von hoat dong 3,000 (181 ngay) |
| Pure inventory carryover hieu chinh | **3.97 USD** (tu 40.39 cua Sprint 16) - khoang cach B - D' gan nhu toan bo la artifact do luong |
| Adaptive min-viable-leg | Dat - von 3,000 -> 45 USD, von 400-600 -> san 10 USD, khong bao gio de naked leg |
| Doi soat trang thai san (`exchange_reconciler`) | Dat - 5 trang thai, 4 repair action, `trading_allowed` chi True khi `SYNCED` |
| Fail-closed `--reconcile-snapshot` | Dat - snapshot loi/lech -> `Reconcile: BLOCKED (...)`, khong mo vi the moi |
| Shadow-live burn-in (mang that) | 10/10 PASS, 8 symbol, spread toi da 1.22 bps |
| Fire drill | 20/20 PASS |
| Test suite | **495 passed** (x2: 44.35s / 45.97s) |

## 1. Kiem toan phap y overlay (Chang 1)

### 1.1 Bo do moi

- `src/barbell_portfolio.py`: `_simulate_barbell_equity` xuat them mang `notional_base` (von co so
  tung gio) va `rebal_flag` (gio co tai can bang tai chinh); `run_barbell_portfolio` tra them
  `overlay_raw_weights`, `overlay_notional_base`, `overlay_rebal_flags`, `overlay_returns_by_symbol`,
  `overlay_funding_by_symbol`, `overlay_gross_series`, `overlay_fee_series`.
- `src/sprint11_eval.py`: `decompose_overlay_pnl_divergence(...)` - dang thuc
  `g_alt - g_ref = (A_alt - A_ref) * n_ref + A_alt * (n_alt - n_ref)` voi `A_t = sum(w * (ret - funding))`,
  `lag_weights=True` mac dinh (dich trong so 1 bar vi PnL cua gio t dung trong so cua bar truoc).
- `src/sprint11_eval.py`: `run_indicator_warm_cash_start(..., include_raw=True)` de lay raw result;
  `compute_holdout_accounting_bridge(..., warm_cash_converged=None, overlay_forensics=None)` them
  `e_true_fixed_warm_cash` va `overlay_divergence_forensics`, khong doi bat ky key cu nao.
- `scripts/forensic_overlay_divergence.py`: chay B (lien tuc, fixed base), C (cold start chi trong
  cua so holdout), D (504h, phase 0), D' (504h, phase can chinh), E (`warmup_hours=720`, phase can
  chinh) roi xuat `docs/sprint17_overlay_forensics.md` + 2 file JSON.

### 1.2 So lieu holdout (181 ngay, von hoat dong 3,000 USD)

| Phuong phap | Overlay net (USD) | Net tong (USD) | Ghi chu |
|---|---|---|---|
| A - continuous compound | 103.53 | 159.15 | tai dau tu lai, von phinh |
| B - continuous fixed base | **72.34** | **118.50** | von neo 3,000, trang thai am san |
| C - cold start trong cua so | 48.87 | 97.07 | khong co hang ton carry tu truoc |
| D - warm cash 504h, phase 0 | 40.72 | 81.73 | phase tai can bang sai |
| D' - warm cash 504h, phase can chinh | **35.09** | **76.09** | phuong phap Sprint 11 da dung |
| E - warm cash 720h, phase can chinh | **73.26** | **114.27** | phep do chuan Sprint 17 |

### 1.3 Phan ra khoang cach

`B - D'` (net gap -37.41, do lai -37.25):

| Thanh phan | USD | Ty trong |
|---|---|---|
| Weight effect | **-35.82** | 95.7% |
| Capital effect (notional base) | -0.13 | 0.3% |
| Fee effect | -1.47 | 3.9% |
| Residual | -2.8e-14 | ~0 |

`B - E` (net gap 0.78): weight +0.83, capital -0.05, fee -0.00, residual -9.9e-15.

So khop trong so (Jaccard tren tap symbol dang giu trong so):

| Cap | Jaccard | Ty le gio trong so giong het | Rebal hours | Timing match |
|---|---|---|---|---|
| B vs D' | 0.559 | 0.0% | 37 vs 39 | 0% (lech 76 gio) |
| B vs E | **0.992** | **93.3%** | 37 vs 37 | **100%** |
| D' vs E | 0.565 | - | - | - |

Notional base trung binh: B 3,000.00 | C 2,998.84 | D' 2,998.85 | E 2,999.79 -> **capital effect
khong the giai thich khoang cach** (lech < 1.2 USD tren toan ky).

### 1.4 Bon gia thuyet va verdict

| Gia thuyet | Ket qua | Bang chung |
|---|---|---|
| H1 - khac biet capital scaling / notional base | **BAC BO** | capital effect -0.13 USD; notional base lech < 0.05% |
| H2 - lech pha tai can bang 48h | **BAC BO** (chi la he qua) | D' da dung dung dong ho tuyet doi; phase offset 22 la dung, khong phai loi |
| H3 - EMA/score chua hoi tu do warmup 504h | **XAC NHAN** | 504h -> 76.09; 720/1440/2160/2880h -> 114.27 (bao hoa); Jaccard B/E 0.992 |
| H4 - mat lenh tai cau truc dau tien | **BAC BO** | `rebal_flag` bat tai `trade_start_idx` khi raw weight 0 -> khac 0; khong co gio nao bi bo qua |

**Root cause**: `warmup_hours=504` khong du de EMA funding (`ema_span=42`) va lookback overlay
(`overlay_lookback_hours=336`) hoi tu truoc khi cua so holdout bat dau. D' vi vay khoi dong voi
trong so overlay gan nhu tinh (chu yeu BTC/AVAX) va chi bat kip ke hoach that sau ~2-4 tuan, lam
mat phan lon PnL cua nhanh overlay. 720h la nguong bao hoa: tang tiep len 2880h khong doi ket qua.

### 1.5 Pure inventory carryover hieu chinh

- Sprint 16: `pure_inventory_carryover = 40.39 USD` (tinh tu B - D' sau khi tru cac thanh phan khac).
- Sprint 17: `118.50 - 114.27 - 0.27 (phi vao lenh cua chinh E) =` **3.97 USD**.

Y nghia: loi the duy nhat cua viec bat dau holdout khi da co hang ton carry (B) so voi khoi dong
lanh da hoi tu (E) chi con **3.97 USD / 181 ngay** thay vi 40.39 USD. Phan con lai cua "khoang
cach B - D'" la artifact cua phuong phap do (H3).

*Ghi chu ky thuat*: ban dau `e_true_fixed_warm_cash.revised_pure_carryover_usd` dung phi vao lenh
cua A/B (`holdout_resim_entry_fee_usd` = 2.02) cho ra 2.22, trong khi script phap y dung phi cua E
(0.27) cho ra 3.97. Da thong nhat theo phi cua chinh E (B da tra phi truoc split nen khong duoc
tru lan hai); co test hoi quy `test_revised_carryover_uses_e_own_entry_fee`.

### 1.6 Phan ra theo nhanh cua E

| Nhanh | Net (USD) | Phi (USD) |
|---|---|---|
| Carry | 41.01 | 11.82 |
| Overlay | 73.26 | 14.75 |
| Tong | 114.27 | 26.58 |

Ty le giai ngan carry 89.88% (3.60/4 slot); 30 ngay dau 2.10 USD, 151 ngay sau 112.17 USD.

## 2. Adaptive min-viable-leg (Chang 2)

- `src/barbell_portfolio.py`: `compute_adaptive_min_viable_leg_usd(active_capital, min_floor_usd=10.0,
  max_cap_usd=50.0, capital_ratio=0.015)` = `max(10, min(50, capital * 1.5%))`
  -> 3,000 USD = 45 USD; 600 USD = 10 USD; >= 3,334 USD = 50 USD.
- `enforce_overlay_dollar_neutrality(..., min_viable_leg_usd=None, active_capital=None)`: `None` (mac
  dinh) = tu tinh theo `active_capital` neu co, khong co thi 50.0 -> **toan bo test Sprint 16 chay
  nguyen khong sua**; truyen so cu the thi dung dung so do.
- `src/carry_live_executor.py`: `overlay_min_viable_leg_usd` mac dinh `None` (adaptive theo
  `equity_now`), dat so cu the de ghim cung; `last_overlay_neutrality` ghi them
  `min_viable_leg_usd` + `adaptive` de audit.
- Bao toan bat bien: mot chan bi lam tron ve 0 thi chan con lai **luon** bi khoa ve 0 (co test tai
  von 400/600/3,000/50,000).
- Test: `tests/test_sprint17_adaptive_neutrality.py` (9 test).

## 3. Doi soat trang thai san (Chang 3)

### 3.1 API

`src/exchange_reconciler.py`:

- `normalize_symbol` - `BTCUSDT` / `BTC/USDT` / `BTC/USDT:USDT` -> `BTC/USDT:USDT` (idempotent).
- `load_snapshot(path)` - doc JSON (`utf-8-sig`, chap nhan BOM), root phai la dict.
- `reconcile_portfolio_state(local_state, exchange_snapshot, prices, lot_specs,
  tolerance_usd=10.0, equity_tolerance_pct=1.5)` -> `status`, `trading_allowed`, `repair_actions`,
  `items`, `orphans`, `phantoms`, `drifting`, `equity_gap_usd`, `equity_gap_pct`, `summary`.

| Trang thai | Dieu kien | Repair action |
|---|---|---|
| `SYNCED` | moi symbol lech <= 10 USD va equity lech <= 1.5% | - |
| `ORPHAN_ON_EXCHANGE` | san co vi the >= `min_notional` ma local khong co | `CLOSE_ORPHAN_SPOT` / `CLOSE_ORPHAN_PERP` |
| `PHANTOM_IN_STATE` | local co vi the ma san khong co | `PURGE_PHANTOM_STATE` |
| `QTY_DRIFT` | hai ben deu co nhung lech > `tolerance_usd` | `RESYNC_QTY` |
| `EQUITY_MISMATCH` | `|equity_san - equity_local| / equity_local > 1.5%` | - (uu tien cao nhat) |

Uu tien status: `EQUITY_MISMATCH` > `ORPHAN_ON_EXCHANGE` > `PHANTOM_IN_STATE` > `QTY_DRIFT` > `SYNCED`.
`trading_allowed` chi `True` khi `SYNCED` (fail-closed).

### 3.2 Tich hop van hanh

- `run_live.py`: `--reconcile-snapshot <json>` -> `_run_reconcile_preflight(args, executor, prices=None)`
  chay truoc `run_loop` o ca hai nhanh (barbell va directional). File khong doc duoc -> status
  `SNAPSHOT_UNREADABLE`, `trading_allowed=False` (khong bao gio mac dinh cho phep).
- `src/carry_live_executor.py`:
  - `apply_reconcile_report(report)` + `_reconcile_dashboard_label()`.
  - Dashboard in them `| Reconcile=<OFF|SYNCED|BLOCKED (STATUS)>`.
  - Khi blocked: `build_barbell_allocations` tra allocation rong, va `resolved_actions` cua carry bi
    ha `OPEN_DELTA_NEUTRAL -> SKIP` voi reason `RECONCILE_BLOCKED_<STATUS>` (dong/giam vi the van duoc
    phep).
  - `build_cycle_audit_record(..., reconcile_status=None)` -> JSONL audit co truong `reconcile_status`.

### 3.3 Bang chung chay that (paper)

| Kich ban | Ket qua |
|---|---|
| Khong bat `--reconcile-snapshot` | `Reconcile=OFF`, Carry 583.80/583.54, Overlay 416.81/416.80, **NetDelta 0.01** |
| Snapshot co orphan (BTC 0.01 tren san, local rong) | `Reconcile: BLOCKED (ORPHAN_ON_EXCHANGE)`, chan 1 lenh carry moi, **Carry 0.00 / Overlay 0.00**, audit ghi `"reconcile_status": "ORPHAN_ON_EXCHANGE"` |
| Snapshot khop (rong, equity 3,000) | `Reconcile: SYNCED`, mo vi the binh thuong, NetDelta 0.01 |
| Snapshot co BOM (loi doc) | `SNAPSHOT_UNREADABLE` -> blocked (sau do da cho `load_snapshot` doc BOM) |

Test: `tests/test_sprint17_exchange_reconciler.py` (17 test).

## 4. Kiem thu va bang chung

| Lenh | Ket qua |
|---|---|
| `python -m pytest -q` (lan 1) | **495 passed** - 44.35s |
| `python -m pytest -q` (lan 2) | **495 passed** - 45.97s |
| `python scripts/forensic_overlay_divergence.py` | OK - B 72.34 / C 48.87 / D 40.72 / D' 35.09 / E 73.26; bridge 118.50 / 76.09 / 114.27, carryover 3.97 |
| `python run_live.py --mode paper --strategy barbell --capital 3000 --reset-state --cycles 2` | OK - NetDelta 0.01, Reconcile OFF |
| `python scripts/run_shadow_burnin.py` | **10/10 PASS** (feed mang that, 8 symbol, spread <= 1.22 bps) |
| `python scripts/run_e2e_fire_drill.py` | **20/20 PASS** |

Test moi: 9 (forensics) + 9 (adaptive neutrality) + 17 (reconciler) + 1 (hoi quy phi E) = 36 test moi
(459 -> 495).

## 5. Danh gia Go / No-Go cap nhat

| Chi so | Sprint 11 (D') | Sprint 17 (E) | Nguong |
|---|---|---|---|
| Return on active capital (181 ngay) | 2.54% | **3.81%** | 3.5% |
| Overlay net (USD) | 35.09 | 73.26 | - |
| Carry utilization | 89.9% | 89.9% | - |
| Pure inventory carryover | 40.39 (artifact) | 3.97 | - |

Doc theo so lieu: voi phep do dung (E), chien luoc **vuot nguong return 3.5%** (3.81%) trong khi
rui ro delta rong da duoc xu ly o Sprint 16 (NetDelta < 1 USD tren paper) va co them co che doi soat
fail-closed o Sprint 17. Ket luan "NO-GO" cua Sprint 11 bi anh huong boi artifact do luong (H3), khong
phai do chien luoc yeu.

Tuy nhien **quyet dinh cuoi cung van thuoc ve nguoi dung / architect**, vi 2 dieu kien chua co bang
chung that: (1) chua gui lenh that len testnet (moi o muc ky + dong goi, `--testnet-send-orders` van
TAT), (2) reconciler moi duoc kiem chung tren snapshot tinh, chua co du lieu san that.

## 6. Rui ro con lai

1. **H3 phu thuoc du lieu**: 720h la nguong bao hoa tren bo du lieu hien tai; can theo doi lai khi
   doi universe hoac span EMA.
2. **Adaptive threshold**: san 10 USD sat `min_notional` cua san (5-10 USD) - o von rat nho co the
   sinh nhieu lenh bi tu choi; da co guard khoa ca hai chan nhung chua do tren testnet that.
3. **Reconciler**: chua map duoc `margin`/`unrealized PnL` cua perp (chi doi soat qty + equity tong);
   `EQUITY_MISMATCH` se bat cac truong hop nay nhung khong chi ra duoc nguyen nhan.
4. **Chua chay testnet that** - day la dieu kien con thieu de chuyen tu paper sang GO.

## 7. File thay doi

| File | Loai |
|---|---|
| `src/barbell_portfolio.py` | sua - instrumentation phap y + adaptive min-viable-leg |
| `src/sprint11_eval.py` | sua - `decompose_overlay_pnl_divergence`, `include_raw`, `e_true_fixed_warm_cash`, add-back phi E |
| `src/carry_live_executor.py` | sua - adaptive threshold, `apply_reconcile_report`, gate fail-closed, dashboard + audit |
| `src/exchange_reconciler.py` | moi - module doi soat trang thai san |
| `run_live.py` | sua - `--reconcile-snapshot`, `_run_reconcile_preflight` |
| `scripts/forensic_overlay_divergence.py` | moi - generator bang chung phap y |
| `tests/test_sprint17_overlay_forensics.py` | moi - 10 test |
| `tests/test_sprint17_adaptive_neutrality.py` | moi - 9 test |
| `tests/test_sprint17_exchange_reconciler.py` | moi - 17 test |
| `docs/sprint17_overlay_forensics.md` | moi - bang chung tu dong (khong sua tay) |
| `docs/sprint17_report.md` | moi - bao cao nay |

Commit: `4f1e998` (Chang 1), `3a69fc1` (Chang 2), `c1da294` (Chang 3), commit cuoi (bao cao + hoan thien).
