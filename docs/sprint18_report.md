# Sprint 18 — Báo cáo tổng kết
## Live Runtime Invariant Enforcement (720h Saturation) · Perp Margin/uPnL Reconciler · Signed Account Snapshot Pipeline

**Nhánh:** `agy/sprint18-live-invariants-and-margin-reconciler` (tách từ `master`, baseline **510 passed**)
**Ngày:** 2026-10-21
**Phán quyết:** `QUANT ALPHA: GO (+3.81% Holdout)` — chi tiết tại `docs/go_no_go_sprint18.md`

---

## 1. Tóm tắt điều hành

Sprint 18 đóng 2 hạn chế còn lại của Sprint 17 và khóa cứng ngưỡng bão hòa warmup ở runtime:

1. **Hết "warmup 400h trên giấy"**: `run_live.py` / `CarryLiveExecutor` nay chặn mở vị thế mới khi lịch sử
   `< 720h` (`WARMUP_INSUFFICIENT_BELOW_720H`), và gọi `verify_lot_neutrality_invariant` trước khi phát lệnh
   Overlay (fail-closed nếu lệch > 10 USD hoặc có chân rỗng).
2. **Hết "EQUITY_MISMATCH không rõ nguyên nhân"**: reconciler phân rã chênh lệch equity thành
   `cash_and_funding_drift_usd` / `unrealized_pnl_drift_usd` / `unexplained_residual_drift_usd` +
   `primary_mismatch_cause`, đồng thời thêm guard ký quỹ & khoảng cách thanh lý perp (`MARGIN_DANGER`).
3. **Hết "reconciler chỉ đọc file tĩnh"**: `BinanceSignedTestnetClient` ký HMAC-SHA256 và lấy snapshot tài khoản
   thật (`/api/v3/account`, `/fapi/v2/account`, `/fapi/v2/positionRisk`) quy đổi thẳng về `exchange_snapshot`
   của reconciler, nối vào `run_live.py` qua cờ `--live-snapshot`.

---

## 2. Thay đổi theo 4 chặng

### Chặng 1 — Warmup 720h + runtime invariants (`run_live.py`, `src/carry_live_executor.py`)
- `WARMUP_MIN_HOURS = 720`, `WARMUP_MIN_FUNDING_PERIODS_8H = 90`, `WARMUP_BLOCK_REASON = "WARMUP_INSUFFICIENT_BELOW_720H"`.
- `compute_warmup_hours_loaded()` (hàm thuần) + `compute_warmup_readiness()` (chạy đầu mỗi `reconcile_once`).
- `warmup_ready=False` → `build_barbell_allocations` trả `warmup_blocked=True` (không sinh carry/overlay),
  `resolved_actions` đổi `OPEN_DELTA_NEUTRAL` → `SKIP` với reason `WARMUP_INSUFFICIENT_BELOW_720H`.
  Đóng/giảm vị thế vẫn được phép (giảm rủi ro không bị chặn).
- `verify_lot_neutrality_invariant(overlay_long_usd, overlay_short_usd, max_imbalance_usd=10.0)` trước khi
  trả allocations; vi phạm → huỷ overlay + `invariants_passed=False`.
- `build_cycle_audit_record()` thêm `invariants_passed`, `warmup_threshold_hours`, `warmup_hours_loaded`;
  Dashboard in `Warmup=720h (thr 720h) -> OK | ... | Invariants: OK`.
- Test mới: `tests/test_sprint18_live_invariants_gate.py` (**10 test**).

### Chặng 2 — Perp margin / uPnL reconciler (`src/exchange_reconciler.py`)
- `MARGIN_DANGER` (thêm vào `RECONCILE_STATUSES`).
- `analyze_equity_and_margin_health(...)`:
  - `equity_drift_attribution`: `total_equity_drift_usd`, `cash_and_funding_drift_usd`,
    `unrealized_pnl_drift_usd`, `unexplained_residual_drift_usd`, `primary_mismatch_cause`
    (`CASH_OR_FUNDING_DRIFT` / `UNREALIZED_PNL_DRIFT` / `UNEXPLAINED_EQUITY_DRIFT`, `None` khi khớp).
    uPnL local ước tính từ chính carry/overlay positions (không lấy từ sàn) để phân rã đúng nguyên nhân.
  - `margin_health`: `maint_margin_ratio` (chia 0 an toàn), `min_liq_distance_pct`,
    `perp_positions_detail` chuẩn hoá (nhận cả dict và list).
  - Vi phạm `maint_margin_ratio > 0.50` hoặc `min_liq_distance_pct < 25.0%` → `MARGIN_DANGER`,
    `trading_allowed=False`, repair action `EMERGENCY_DELEVERAGE_PERP`.
- `reconcile_portfolio_state(...)` thêm 2 tham số `max_maint_margin_ratio`, `min_liq_buffer_pct`
  (có default) → **tương thích ngược 100%** với 17 test Sprint 17.
- Test mới: `tests/test_sprint18_margin_and_upnl_reconciler.py` (**13 test**).

### Chặng 3 — Signed account snapshot pipeline (`src/binance_signed_client.py`, `scripts/run_testnet_preflight.py`)
- `build_signed_get_request(endpoint, params=None, futures=False)`: URL + query có `timestamp`/`recvWindow`/`signature`
  HMAC-SHA256 + header `X-MBX-APIKEY`; thiếu khoá → `BinanceCredentialsMissingError` (fail-closed).
- `fetch_exchange_reconcile_snapshot(symbols, prices, mock_responses=None, timeout_sec=5.0)`:
  - mock (test/dry-run) hoặc HTTP GET thật tới testnet; chuẩn hoá Spot `/api/v3/account` +
    Futures `/fapi/v2/account` + `/fapi/v2/positionRisk` thành đúng dict `exchange_snapshot`.
  - Tiền mặt/stablecoin quote bị loại khỏi `spot_balances` (`_is_cash_asset`) và gom vào `spot_cash_usd`,
    tránh reconciler báo `ORPHAN_ON_EXCHANGE` giả tạo cho USDT.
  - `exchange_equity_usd = spot_cash_usd + perp_wallet_balance_usd + perp_unrealized_pnl_usd`.
  - Lỗi mạng/khoá/payload → `snapshot_healthy=False` (không raise) để `run_live.py` chặn giao dịch.
- `run_live.py`: cờ `--live-snapshot` + `_run_signed_snapshot_preflight()` (BLOCKED với
  `SNAPSHOT_CREDENTIALS_MISSING` / `SNAPSHOT_UNHEALTHY`).
- `scripts/run_testnet_preflight.py`: 6/6 kiểm tra đầu-cuối (ký HMAC → mock payload → `SYNCED` →
  `MARGIN_DANGER`/`ORPHAN_ON_EXCHANGE` fail-closed → warmup 720h + 5 bất biến → 1 chu kỳ Barbell + JSONL audit).
- Test mới: `tests/test_sprint18_signed_snapshot_pipeline.py` (**10 test**).

### Chặng 4 — Kiểm toán, Go/No-Go, báo cáo
- `quant_auditor` kiểm toán độc lập 5 tiêu chí (xem §4).
- `docs/go_no_go_sprint18.md` (phán quyết GO + Stage-Gated Rollout) và tài liệu này.
- Không chạy `git merge` / `git push`.

---

## 3. Bằng chứng test thực tế (chạy lại trong phiên này)

| Lệnh | Kết quả |
|---|---|
| `python scripts/run_testnet_preflight.py` | **PREFLIGHT: 6/6 PASS** (exit 0) |
| `python scripts/run_shadow_burnin.py` | **10/10 PASS** (feed mạng thật, 8 symbol, spread ≤ 1.21 bps) |
| `python scripts/run_e2e_fire_drill.py` | **Tổng: 20/20 checks PASS** |
| `python run_live.py --mode paper --strategy barbell --capital 3000 --reset-state --cycles 2 --interval-sec 1` | OK — `Warmup=720h (thr 720h) -> OK`, `Invariants: OK`, Carry Spot 586.78 vs Perp 586.53, Overlay Long 418.95 vs Short 418.93 → **NetDelta 0.03 USDT** |
| `python -m pytest -q` (lần 1) | **543 passed** — 52.57s |
| `python -m pytest -q` (lần 2) | **543 passed** — 51.95s |

Phân rã test: 510 (baseline) + 10 (Chặng 1) + 13 (Chặng 2) + 10 (Chặng 3) = **543** (mục tiêu ≥ 534).

### Test cũ được cập nhật có chủ đích (do yêu cầu đổi hành vi, không phải nới ngưỡng để che lỗi)
| File | Thay đổi | Lý do |
|---|---|---|
| `tests/test_sprint14_diagnostics.py` | `warmup_min_required` 400 → 720; cold 719h / warm 720h | Ngưỡng bão hoà đổi theo đặc tả Sprint 18 |
| `tests/test_carry_live_executor.py` | Fixture OPEN→HOLD→CLOSE dùng `n_hours=720` | Luồng mở vị thế thật phải qua cổng warmup |
| `tests/test_sprint10_production.py` | Legacy overlay path: khẳng định **fail-closed** khi 2 chân lệch 14.22 USD > 10 USD; thêm nhánh nới ngưỡng 20 USD để vẫn kiểm tra step grid | Bất biến LOT_NEUTRALITY mới áp dụng cho mọi đường phát lệnh Overlay |

---

## 4. Biên bản kiểm toán độc lập (`quant_auditor`)

Nguồn: `.agy/quant_auditor_0a21e617.md` (read-only, không sửa file, không chạy Git).

| # | Tiêu chí | Kết quả |
|---|---|---|
| 1 | FROZEN ALPHA (Tier 1B + Smooth Conviction) | **PASS** — `ema_span=42`, `entry_threshold=0.000045`, `buffer_rank`, `min_hold_periods=63`, `eta=0.80`, `min_weight_change=0.08` giữ nguyên |
| 2 | WARMUP SATURATION 720h (không bypass) | **PASS** — 720h áp dụng nhất quán, ghi vào JSONL audit |
| 3 | MARGIN / LIQ GUARD (toán học + fail-closed) | **PASS** — chia 0 an toàn, đúng đơn vị %, `MARGIN_DANGER` khoá giao dịch |
| 4 | SNAPSHOT PIPELINE (chuẩn hoá, chống double-count) | **PASS** — cash tách khỏi `spot_balances`, equity nhất quán, không hardcode secret |
| 5 | TEST INTEGRITY (không test giả) | **PASS** — test mới chứng minh đúng hành vi mới, offline & deterministic |

**Kết luận kiểm toán: `QUANT VERDICT: GO` — 0 `[BLOCKER: QUANT_ARTIFACT]`.**

Xử lý 2 `[WARN]`:
1. `spot_cash_usd` dùng danh sách cứng thay vì `_is_cash_asset` → **đã sửa** (dùng `_is_cash_asset`, phủ thêm FDUSD/DAI/TUSD).
2. `self.warmup_ready = True` mặc định khi khởi tạo executor → **giữ nguyên by-design**: `compute_warmup_readiness`
   luôn chạy ở đầu mỗi `reconcile_once` trước khi dựng lệnh, nên luồng vận hành thực tế luôn fail-closed;
   default `True` chỉ để không phá các unit test gọi trực tiếp `build_barbell_allocations`.

---

## 5. Hướng dẫn vận hành Testnet → Canary

```powershell
# 0. Biến môi trường (không hardcode, không commit)
$env:BINANCE_TESTNET_API_KEY = "<testnet-key>"
$env:BINANCE_TESTNET_API_SECRET = "<testnet-secret>"

# 1. Pre-flight 6 bước (offline-safe, exit 0 = đủ điều kiện)
python scripts/run_testnet_preflight.py

# 2. Shadow burn-in (10 chu kỳ, feed thật, không đặt lệnh)
python scripts/run_shadow_burnin.py

# 3. Fire drill an toàn (kill-switch, orphan, reconcile fail-closed)
python scripts/run_e2e_fire_drill.py

# 4. Testnet live: snapshot tài khoản thật có ký + audit JSONL
python run_live.py --mode testnet --strategy barbell --capital 3000 `
  --live-snapshot --audit-log logs/sprint18_testnet_audit.jsonl --cycles 10 --interval-sec 60
```

Kiểm tra sau mỗi phiên:
- `Diagnostics:` phải có `Warmup=720h (thr 720h) -> OK` và `Invariants: OK`.
- `Reconcile=` phải là `SYNCED` (hoặc `OFF` nếu chưa bật snapshot); mọi `BLOCKED (...)` phải điều tra trước khi chạy tiếp.
- Mỗi dòng JSONL phải có `invariants_passed`, `warmup_threshold_hours = 720`, `reconcile_status`.

---

## 6. Rủi ro còn lại

1. **Chưa gửi lệnh thật lên testnet** trong sprint này (`--testnet-send-orders` vẫn TẮT); Bước 1 của rollout là
   bằng chứng còn thiếu duy nhất trước Canary.
2. **Ngưỡng 25% liq-buffer / 50% maint-margin** là ngưỡng an toàn bảo thủ, chưa hiệu chỉnh theo dữ liệu
   margin thật của từng cặp; nên theo dõi phân bố thực tế trong 48h testnet.
3. **uPnL local** trong phân rã drift là ước tính theo `entry_price`/`mark_price` trong state, không gồm phí
   funding đã trả/nhận giữa 2 lần reconcile → phần dư sẽ rơi vào `cash_and_funding_drift_usd` (đúng bản chất
   nhưng không tách được funding vs chuyển tiền).
4. **720h là ngưỡng bão hoà theo bộ dữ liệu hiện tại**; nếu đổi universe hoặc `ema_span`, phải chạy lại
   forensics (`scripts/forensic_overlay_divergence.py`) trước khi giữ nguyên 720h.
5. Legacy overlay path (không `tier1b`) có thể bị huỷ overlay do lot quantization > 10 USD — đây là hành vi
   fail-closed mong muốn; nếu cần chạy legacy, phải nới `overlay_max_imbalance_usd` có ý thức.

---

## 7. File thay đổi

| File | Loại |
|---|---|
| `src/carry_live_executor.py` | sửa — warmup 720h, invariant gate, audit fields |
| `src/exchange_reconciler.py` | sửa — `MARGIN_DANGER`, `analyze_equity_and_margin_health`, drift attribution |
| `src/binance_signed_client.py` | sửa — `build_signed_get_request`, `fetch_exchange_reconcile_snapshot`, `_is_cash_asset` |
| `run_live.py` | sửa — ngưỡng 720h, `--live-snapshot`, `_run_signed_snapshot_preflight` |
| `scripts/run_testnet_preflight.py` | mới — 6/6 pre-flight |
| `tests/test_sprint18_live_invariants_gate.py` | mới — 10 test |
| `tests/test_sprint18_margin_and_upnl_reconciler.py` | mới — 13 test |
| `tests/test_sprint18_signed_snapshot_pipeline.py` | mới — 10 test |
| `tests/test_sprint14_diagnostics.py`, `tests/test_carry_live_executor.py`, `tests/test_sprint10_production.py` | sửa — cập nhật kỳ vọng theo hành vi mới |
| `docs/go_no_go_sprint18.md`, `docs/sprint18_report.md` | mới — phán quyết & báo cáo |
