# SPRINT 21 REPORT — CLOCK-ALIGNED REBALANCE, SMART INSIDE-SPREAD MAKER, UNIVERSE-WIDE RECONCILE & 3-CYCLE LIVE BURN-IN

- **Nhanh**: `agy/sprint21-execution-optimizations-and-burnin` (tach tu `master`, baseline 616 passed)
- **Ngay**: 2026-10-22
- **Ket qua cuoi**: **665 passed** (616 baseline + 49 test moi), burn-in 3 chu ky **PASS** tren Binance Spot + USDⓈ-M Futures Testnet, preflight 6/6, shadow 10/10, fire drill 20/20
- **Bao mat**: khong in/log gia tri `API_KEY` / `API_SECRET`; quet `[A-Za-z0-9]{64}` va `signature=<hex>` tren toan bo evidence/log = `NONE`

---

## 1. CHANG 1 — DONG HO REBALANCE THEO UTC EPOCH (KHONG TROI PHA KHI KHUYET NEN)

### 1.1. API moi trong `src/barbell_portfolio.py`

| Thanh phan | Mo ta |
|---|---|
| `_EPOCH_UTC` | Moc tham chieu `1970-01-01T00:00:00Z` (timezone-aware) |
| `_utc_epoch_hours(timestamps)` | Doi `DatetimeIndex` (naive -> coi la UTC) sang so gio epoch |
| `build_rebalance_mask(..., phase_offset=0, weights_changed=None, index=None, timestamps=None, use_utc_epoch_schedule=False)` | Bucket `floor((ts_utc_hours - phase_offset) / rebalance_hours)`; bat mask tai **bucket boundary crossing**, khong phu thuoc so nen da troi |

Loi duoc dong: truoc day mask tinh theo **vi tri nen** (`i % rebalance_hours == 0`), nen khi feed
khuyet 1 nen thi moc rebalance troi vinh vien va chu ky 48h bi lech. Nay moc bam **thoi gian thuc**:
khi nen dung moc 48h bi thieu, mask bat ngay o nen dau tien vuot moc (test
`test_rebalance_mask_utc_schedule_survives_missing_bar` chung minh khong troi pha).

`weights_changed` + `phase_offset` giu nguyen hanh vi cu khi `use_utc_epoch_schedule=False`
(tuong thich nguoc 100%). Nguon `timestamps` lay tu `index` neu khong truyen rieng; thieu ca hai ->
`ValueError` (khong doan bua).

---

## 2. CHANG 2 — SMART INSIDE-SPREAD MAKER + FALLBACK MARKET

### 2.1. `compute_maker_limit_price(side, best_bid, best_ask, tick_size, improve_inside_spread=True)`

| Mode | Dieu kien | Gia tra ve |
|---|---|---|
| `INSIDE_SPREAD` | spread >= 2 tick | `best_bid + 1 tick` (BUY) / `best_ask - 1 tick` (SELL) — strictly inside |
| `TOUCH_BEST` | spread <= 1 tick hoac `improve_inside_spread=False` | `best_bid` (BUY) / `best_ask` (SELL) |
| `FALLBACK_NO_BOOK` | book rong, mot phia, **hoac locked/crossed** (`ask <= bid`) | `0.0` -> caller bat buoc dung MARKET |
| `INVALID_SIDE` | side khac BUY/SELL | `0.0` |

Loi duoc dong: GTX/LIMIT_MAKER dat tai gia cat qua spread bi san tu choi (`-2010` / `-2021`),
gay fallback MARKET 100%. Nay gia luon nam trong `(best_bid, best_ask)`.

### 2.2. `place_maker_with_fallback(...)`

- Spot: `LIMIT_MAKER` (khong kem `timeInForce`) -> bi tu choi thi MARKET ngay, ghi event `LIMIT_MAKER_REJECTED`.
- Futures: `GTX` + TTL -> het TTL chua khop thi **cancel + query + MARKET phan con lai**, gop `executedQty`.
- **Audit fix (sau `quant_auditor`)**: `BinanceHttpError` khi gui GTX (post-only/crossing) duoc bat
  -> ban MARKET voi day du `qty` (`event = GTX_REJECTED`), khong crash ca chu ky.
- **Audit fix**: `filled` chi `True` khi `status == FILLED` **va** `executedQty > 0`; MARKET khop mot
  phan -> `filled = False` (khong con "fake FILLED" gay lech delta).
- **Audit fix**: cancel thanh cong nhung `query_order` loi -> `status = UNKNOWN_AFTER_CANCEL`, **khong**
  gui MARKET (tranh double fill).

---

## 3. CHANG 3 — DOI SOAT TOAN RO 8 SYMBOL + CHINH SACH TAI SAN NGOAI DANH MUC

### 3.1. `src/exchange_reconciler.py`

| Thanh phan | Mo ta |
|---|---|
| `BLOCKED` / `blocked` / `trading_allowed` | Trang thai chan giao dich (uu tien cao nhat khi tinh status) |
| `default_universe_symbols()` | Lazy import `LIQUID_8_SYMBOLS`, khong gay circular import |
| `_FALLBACK_UNIVERSE` | Ro du phong khi import that bai |
| `reconcile_portfolio_state(..., universe_symbols, environment="testnet", ignored_spot_assets)` | Doi soat toan ro; `environment` sai -> `ValueError` |
| `untracked_spot_assets` / `untracked_perp_positions` | Tai san co tren san nhung khong co trong so sach |
| `UNTRACKED_MAINNET_ASSET_WARNING` | Mainnet: canh bao; **perp la -> BLOCKED** (chan that) |
| `UNTRACKED_TESTNET_PERP_POSITION` | Testnet: canh bao ro rang nhung **khong chan** (chinh sach faucet) |

Loi duoc dong: testnet cap ~504 tai san faucet, neu dua het vao `canonical_symbols` thi moi chu ky
bao `ORPHAN_ON_EXCHANGE` gia. Nay `_add_snapshot_symbol` chi them symbol **thuoc ro giao dich**
(`BTC`, `ETH`, `SOL`, `BNB`, `DOGE`, `XRP`, `AVAX`, `LINK` + cash), con lai vao `untracked_spot_assets`
(hau kiem) — khong che giau vi the mo coi that tren symbol dang giao dich.

### 3.2. `src/binance_signed_client.py` — ve sinh env legacy

- `_credential_pair_is_valid`, `prune_legacy_fallback_env(environ)`: xoa `BINANCE_API_KEY/SECRET`
  (cap legacy) khi cap chuyen biet Spot/Futures da dat chuan 64/64 -> tranh nap nham cap cu.
- `validate_credential_format(..., legacy_key, legacy_secret)` -> them `legacy_fallback_present`,
  `legacy_fallback_ignored`.
- `load_dotenv_testnet(..., report=None)`: goi prune + tra metadata (khong bao gio chua gia tri khoa).
- `fetch_exchange_reconcile_snapshot(..., universe_symbols=None)`: quet dung ro, tra them
  `universe_symbols`, `untracked_spot_assets`.

---

## 4. CHANG 4 — BURN-IN 3 CHU KY THAT + CONG KIEM DINH

`scripts/run_testnet_multi_cycle_burnin.py` (moi):

| Cycle | Noi dung |
|---|---|
| 1 | Bootstrap & Reconcile: kiem tra credential 64/64, warmup gate 720h, snapshot healthy, doi soat 8/8 SYNCED, adopt baseline so du |
| 2 | Live Dual-Leg Execution & Mid-Cycle Audit: maker pricing -> spot entry + perp entry, LOT_NEUTRALITY, doi soat giua chu ky |
| 3 | Clean Flatten & Zero-Residual: dong 2 chan, residual <= tolerance 0.2% qty, `equity_gap_usd == 0` |

CLI: `--mock-transport`, `--dry-run`, `--json`, `--no-write`, `--evidence-doc`, `--log-jsonl`, `--warmup-hours`.

### 4.1. Bang chung chay that (lan cuoi, truoc commit)

| Hang muc | Ket qua |
|---|---|
| Credential format | spot 64/64 + futures 64/64 (`enforced_64_64`) |
| Warmup gate | `720h / 720h -> OK` |
| Cycle 1 | `SYNCED` 8/8, `untracked_spot_count = 504`, `equity_gap_usd = 0.0` |
| Cycle 2 | spot entry **5889700**, perp entry **28646282116**, `FILLED` 0.001 BTC, `LOT_NEUTRALITY PASS` (imbalance 0.00 USD), `futures_positions_verified = true` |
| Cycle 3 | spot exit **5889714**, perp exit **28646282232**, `spot_residual_qty = 0.0`, `perp_residual_qty = 0.0`, `equity_gap_usd = 0.0`, `exit_perp_executed_qty = 0.001` |
| Drift tai san | `equity_drift_vs_initial_usd = -0.2194` (phi giao dich), `equity_drift_ok = true` (tolerance = max(2 USD, 1% notional)) |
| **Verdict** | **PASS** |

Cong phu (chay lai sau moi thay doi):

| Cong | Ket qua |
|---|---|
| `python -m pytest -q` | **665 passed** (2 lan lien tiep) |
| `run_testnet_preflight.py` | **6/6 PASS** |
| `run_shadow_burnin.py` | **10/10 PASS** |
| `run_e2e_fire_drill.py` | **20/20 PASS** |
| `run_testnet_multi_cycle_burnin.py --mock-transport` | **PASS** (3/3 cycle) |
| `run_testnet_multi_cycle_burnin.py` (that) | **PASS** (3/3 cycle, 4 lenh that) |

---

## 5. TRIAGE BAO CAO `quant_auditor`

| # | Phat hien | Muc | Ket luan | Hanh dong |
|---|---|---|---|---|
| 1 | Futures GTX bi tu choi khong co `try/except` -> khong fallback MARKET | Blocker | **DUNG** | Da bat `BinanceHttpError` -> MARKET toan phan qty + test `test_maker_futures_gtx_rejection_falls_back_to_market` |
| 2 | `filled = executedQty > 0` -> bao FILLED gia khi MARKET khop mot phan | Blocker | **DUNG** | `filled` gio doi hoi `status == FILLED`; test `test_maker_futures_partial_market_fill_is_not_reported_as_filled` |
| 3 | Book locked/mot phia -> GTX tai gia doi ung -> `-2010` | High | **DUNG** | `compute_maker_limit_price` tra `0.0` khi `ask <= bid` hoac thieu mot phia; test locked/crossed |
| 4 | Cancel ok nhung query loi -> MARKET toan phan qty (double fill) | High | **DUNG** | `state_known=False` -> `UNKNOWN_AFTER_CANCEL`, khong ban MARKET; test `test_maker_futures_unknown_state_after_cancel_does_not_send_market` |
| 5 | Testnet perp ngoai ro van bao `SYNCED` im lang | Medium | **DUNG (mot phan)** | Them `UNTRACKED_TESTNET_PERP_POSITION` + `untracked_testnet_perp_warning=True`; van khong chan (theo chinh sach faucet) |
| 6 | `signature=<hex>` trong query string chua duoc scrub | Low | **DUNG** | `_SIGNATURE_RE` scrub `signature=` trong `_scrub_secrets` |
| 7 | Warmup 720h < `3 x span(42) x 8h = 1008h` | Blocker | **TU CHOI** | Sprint 18 `WARMUP_MIN_HOURS = 720` khoa cung o runtime va `docs/go_no_go_sprint18.md` chung minh bao hoa tai 720/1440/2160/2880h (cung 114.27 USD). Doi thanh 1008h se vi pham chuan da duyet va khong co bang chung dinh luong moi. Ghi chu: "do dai quan sat 3 chu ky" la khai niem khac "warmup hoi tu". |
| 8 | `phase_offset` nguoc dau giua epoch schedule va modulo cu | Low | **CHAP NHAN (tai lieu hoa)** | Cong thuc epoch `floor((ts - phase_offset)/hours)` la dac ta cua blueprint Sprint 21; khac biet chi anh huong cach dat `phase_offset` thu cong (khong co caller nao dung khac 0 ngoai test). Da ghi ro trong docstring. |
| 9 | `hourly_net_pnl.iloc[0] = 0.0` che phi mo vi the | Medium | **TU CHOI (ngoai pham vi)** | Chi bao metric Sprint 15 (`src/barbell_portfolio.py`), khong thuoc duong thuc thi Sprint 21, khong anh huong quyet dinh vao lenh; sua se doi ngu nghia bao cao lich su -> de Sprint 22. |
| 10 | `equity_gap_usd == 0` la tautology | Medium | **CHAP NHAN (tai lieu hoa)** | `equity_gap` co y nghia "so sach lanh manh sau reconcile" (tu dong re-anchor theo baseline moi chu ky). Sai lech tai san that duoc bao cao rieng bang `equity_drift_vs_initial_usd` (am do phi) — da ghi trong evidence. |

### 5.2. Vong review cuoi (truoc commit) — 8 phat hien, da xu ly

| # | Phat hien | Muc | Ket luan | Hanh dong |
|---|---|---|---|---|
| 11 | Sau khi GTX bi tu choi -> MARKET chua khop het -> code chay tiep nhanh TTL va ban MARKET lan 2 (double fill) | Blocker | **DUNG** | Co `gtx_submitted`; nhanh TTL chi chay khi lenh vua gui that su la GTX. Test `test_maker_futures_gtx_rejection_partial_market_does_not_double_submit` |
| 12 | Maker khop 100% trong luc huy -> `remaining_qty <= 0` -> khong cap nhat `order`, bao that bai sai | High | **DUNG** | Them nhanh `else` gan `order = final_state`, `filled = True`; test `test_maker_futures_maker_filled_during_cancel_reports_filled` |
| 13 | MARKET fallback bi tu choi (min notional/step size) -> crash giua chu ky | High | **DUNG** | Boc `try/except BinanceHttpError` -> `MARKET_FALLBACK_REJECTED` + `error`; test `test_maker_futures_market_fallback_rejected_returns_structured_error` |
| 14 | Caller burn-in dung `pricing["price"] or best_bid` -> `0.0` (falsy) bi thay bang best_bid, vo hieu hoa bao ve | High | **DUNG** | `maker_buy/maker_sell = price > 0`; khi `0.0` -> `spot_price = best_ask/best_bid`, `post_only_spot=False`, `spot_use_limit_maker=False`; ghi `maker_buy_enabled`/`maker_sell_enabled` vao evidence |
| 15 | `--dry-run` luon FAIL o Chu ky 2 | Low | **DUNG** | `--dry-run` khong ho tro supervisor (can snapshot san that) -> bao ro va thoat ma 2 truoc khi chay |
| 16 | Regex scrub chi bat `signature=` query, bo lot JSON `"signature": "..."` | Low | **DUNG** | Regex moi: `(signature\|hmac)["']?\s*[:=]\s*["']?[0-9a-fA-F]{16,}` |
| 17 | `main()` in JSON/error truc tiep ra STDOUT (khong qua `_scrub_secrets`) | Medium | **DUNG** | Moi `print` trong `main()` deu qua `_scrub_secrets` |
| 18 | Chu ky 2/3 co the PASS am tham khi khong co chan Futures; chu ky 3 chap nhan `ROLLED_BACK_SAFE`; `equity_gap_zero` la tautology | High | **DUNG** | Cycle 2: them `futures_positions_verified` (credential Futures + `perp_executed_qty > 0`) vao dieu kien `ok`. Cycle 3: chi chap nhan `FILLED` (+`DRY_RUN` khi client dry_run), yeu cau `exit_perp_executed_qty > 0`, them `equity_drift_ok` (drift that <= max(2 USD, 1% notional)) ben canh `equity_gap_zero` |
| 19 | `place_maker_with_fallback` tra `INVALID_PRICE` khi book rong thay vi tu fallback MARKET | Low | **TU CHOI** | Theo dac ta: `price = 0.0` la **tin hieu** cho caller chon MARKET (caller da duoc sua o #14). Tra `INVALID_PRICE` tuong minh, khong im lang doi hanh vi. |
| 20 | Ro 8 symbol lam che giau vi the mo coi ngoai ro | Medium | **CHAP NHAN (tai lieu hoa)** | Chinh sach blueprint Sprint 21: reconcile theo ro giao dich + `untracked_spot_assets`/`untracked_perp_positions` de hau kiem; mainnet perp ngoai ro -> **BLOCKED**, testnet -> warning ro rang. Khong the chan moi tai san faucet (504 tai san) vi se khoa toan bo testnet. |

---

## 6. RUI RO CON LAI & DE XUAT

1. Testnet spread BTC thuong = 1 tick -> `TOUCH_BEST` (khong "improved"). Nhanh `INSIDE_SPREAD` chi
   xac nhan bang mock (spread 2000 tick). Can do lai tren cap co spread rong hon truoc khi Mainnet.
2. Chua co do tre do tre mang thuc (khong do `signal -> order -> fill` latency); Sprint 22 nen do.
3. `equity_drift_vs_initial_usd` hien la tong phi tich luy qua cac lan chay trong ngay, chua tach
   rieng phi maker/taker.
4. Chua chay burn-in nhieu gio lien tuc (chi 1 lan 3 chu ky); "burn-in" dai han (>= 24h) van la hang muc Sprint 22.
5. Lenh that chi 0.001 BTC (~81 USD notional) — chua kiem chung slippage o size lon.

---

## 7. KET LUAN

- 4/4 chang cua Sprint 21 hoan thanh: clock-aligned rebalance, smart inside-spread maker + fallback,
  reconcile toan ro + ve sinh env legacy, burn-in 3 chu ky that.
- 665 test pass, 5 cong kiem dinh xanh, 4 lenh that khop va dong sach (residual 0), khong ro ri khoa.
- 2 vong review (review_code + quant_auditor): 14 phat hien duoc sua va co test bao ve; 6 phat hien
  con lai duoc tai lieu hoa kem ly do (xem muc 5.1 va 5.2). Khong co phat hien nao bi bo qua ma
  khong giai thich.
