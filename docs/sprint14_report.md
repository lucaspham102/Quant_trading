# Sprint 14 — Warm-cash holdout bridge, live decision diagnostics, shadow-live feed

**Nhánh:** `agy/sprint14-warm-cash-and-shadow-live` (tách từ `master` = `1bdee8e`, baseline 377 test)
**Kết quả:** 407 test PASS (chạy 2 lần), E2E fire drill 20/20 checks PASS, paper daemon 2 chu kỳ OK,
shadow-live chạy được cả fixture offline và Binance public thật.

---

## 1. Mục tiêu Sprint 14

| # | Mục tiêu | Trạng thái |
|---|---|---|
| 1 | Method D — "indicator-warm cash-start": chỉ báo đã ấm nhưng tồn kho bằng tiền mặt | ✅ |
| 2 | Phân rã 3 thành phần độ trễ khởi động (phí mở, độ mù chỉ báo, chênh lệch tồn kho) | ✅ |
| 3 | Diagnostic quyết định tín hiệu trên dashboard (warmup / carry EMA / overlay delta) | ✅ |
| 4 | Chế độ `shadow-live` đọc Binance public REST, fail-closed, không đặt lệnh | ✅ |

---

## 2. Chặng 1 — Method D và bridge 4 phương pháp

### 2.1 Thay đổi code

- `src/cross_exchange_carry.py`: `run_binance_tier1_plus_carry(..., trade_start_idx=None)`. Khi có mốc,
  mọi bar trước mốc giữ nguyên `equity` (100% tiền mặt) và `continue` — chỉ báo (EMA-42 funding,
  momentum/vol 336h) vẫn được tính trên toàn cửa sổ mở rộng nên không bị "mù".
- `src/barbell_portfolio.py`: `run_barbell_portfolio(..., trade_start_idx=None)` — truyền xuống engine carry
  và zero toàn bộ overlay weight trước mốc.
- `src/sprint11_eval.py`:
  - `SPRINT14_WARMUP_HOURS = 504`, `run_indicator_warm_cash_start(...)` — cắt dữ liệu mở rộng thêm
    `warmup_hours` trước `split_date`, tính `trade_start_idx` bằng `searchsorted`, chạy 1 lần barbell,
    trả `net_pnl_usd`, `return_on_active_capital_pct`, `entry_fee_usd`, `warmup_hours`, `trade_start_idx`.
  - `compute_holdout_accounting_bridge(payload_compound, payload_fixed_base, warm_cash=None)` —
    thêm `indicator_warm_cash_start` (Method D) và `warm_cash_decomposition`.
- `run_sprint11.py`: tách `load_multi_aligned_data(raw_dir, symbols, start, end)` để script bridge dùng lại
  (không đổi hành vi CLI).
- `scripts/gen_warm_cash_bridge.py`: sinh `docs/sprint14_warm_cash_bridge.md`.

### 2.2 Kiểm chứng gate

Chạy trực tiếp trên 8 symbol, cửa sổ mở rộng 504h + holdout:

```
trade_start_idx = 504
max |equity[t] - 3000| với t < 504  = 0.0   (đúng 100% tiền mặt)
equity cuối = 3081.7273813065317
```

### 2.3 Số liệu thật (holdout 181 ngày, vốn active 3000 USDT)

| Phương pháp | Vốn đầu | Net PnL | Return |
|---|---:|---:|---:|
| A. Continuous compound | 4.091,29 | 159,15 | 3,89% |
| B. Continuous fixed-base | 3.946,72 | 118,50 | 3,95% |
| C. Standalone cold-start | 3.000,00 | 97,07 | 3,24% |
| **D. Indicator-warm cash-start** | **3.000,00** | **81,73** | **2,72%** |

Phân rã `B - C = 21,44 USD`:

| Thành phần | USD |
|---|---:|
| `cold_start_initial_entry_fee_usd` | 2,02 |
| `indicator_blindness_drag_usd` (= D − C) | −15,34 |
| `inventory_carryover_diff_usd` (= B − D − phí mở) | 34,76 |
| **Tổng giải thích** | **21,44** |

Đẳng thức: `B − C − [(B − D) + (D − C)] = 0,000000e+00` và `unexplained_total_usd = 0,000000e+00`.

### 2.4 Phát hiện quan trọng

1. **Chỉ báo ấm không giúp trong giai đoạn này** — D (2,72%) < C (3,24%). Trạng thái EMA-42/momentum mang
   regime cũ vào thời điểm holdout, đồng thời pha tái cân bằng overlay lệch so với cold-start
   (`np.arange(n_bars) % 48` neo theo bar 0 khác nhau). Vì vậy **không** được coi "warm start" là cải thiện
   hiệu suất; nó chỉ là mô phỏng trung thực hơn trạng thái bot thật.
2. Cả 4 phương pháp đều dưới 3,5% (B 3,95% tính trên 3000 nhưng vẫn là mức của B, không phải all-in mới)
   → **verdict NO-GO của Sprint 11 giữ nguyên**, Sprint 14 không mở khóa testnet.

---

## 3. Chặng 2 — Live decision diagnostics

- `CarryLiveExecutor.build_decision_diagnostics(aligned_universe, metrics_at_t, barbell_allocations)`
  trả 12 khóa: `warmup_hours_loaded`, `warmup_min_required=400`, `warmup_ok`, `carry_ema_span`,
  `max_carry_ema_bps`, `carry_ema_threshold_bps`, `carry_gate_pass`, `symbols_above_carry_threshold`,
  `max_overlay_delta`, `overlay_delta_threshold=0.08`, `overlay_gate_pass`, `summary`.
- Kết quả lưu ở `self.last_decision_diagnostics`, đưa vào `plan["diagnostics"]` (cả nhánh kill-switch
  và nhánh thường) và in trên dashboard ngay sau dòng `DASHBOARD`:
  `Diagnostics: Warmup=720h (thr 400h) -> OK | MaxCarryEMA=0.73 bps (thr 0.45) -> PASS | MaxOverlayDelta=0.000 (thr 0.08) -> BLOCK`
- Xóa 1 lời gọi `build_barbell_allocations` bị lặp trong `reconcile_once`.
- `run_live.py` log `Warmup requirement: >= 400h history (lookback_hours=720).`; mặc định 720h ≥ 400h.

**Quan sát cần theo dõi (chưa sửa — Frozen Alpha):** trong lần chạy paper, `MaxCarryEMA=0.73 bps`
vượt ngưỡng backtest 0,45 bps → `carry_gate_pass=True`, nhưng bảng reconcile vẫn `SKIP` với lý do
`SCORE_BELOW_ENTRY_THRESHOLD`. Nguyên nhân: executor live gate theo `score` (đã trừ chi phí payback),
không phải theo `f_ewma` thô. Đây là **khác biệt gate backtest vs live**, cần quyết định ở sprint sau
(hoặc hiển thị thêm ngưỡng score thực tế trong diagnostics).

---

## 4. Chặng 3 — Shadow-live public feed

`src/binance_public_feed.py`:

- `to_binance_symbol("BTC/USDT:USDT") -> "BTCUSDT"`.
- `fetch_binance_shadow_snapshot(symbols, timeout_sec=5.0, fixture_data=None)` đọc
  `/fapi/v1/ticker/bookTicker` + `/fapi/v1/premiumIndex` bằng `urllib.request` (stdlib, read-only).
- Trả `{source, healthy, fetched_at_utc, symbols{sym:{bid,ask,mid,spread_bps,mark_price,funding_rate_bps}}, errors}`.
- **Fail-closed:** lỗi mạng/timeout/JSON hỏng → `healthy=False`, `symbols={}` (không rò dữ liệu cũ);
  sổ lệnh crossed → `SPREAD_CROSSED_BOOK`, giá ≤ 0 → `SPREAD_INVALID_BOOK`, đều bị loại;
  `fixture_data` → 0 lời gọi mạng.
- `run_live.py`: `--mode shadow-live` (luôn ép `exec_mode=paper`, không dùng API key),
  `--shadow-timeout-sec`, `--offline-fixture`; pre-flight `_run_shadow_live_preflight` chặn trước khi vào
  vòng lặp nếu feed không healthy (exit 1).

Bằng chứng chạy thật:

```
fixture : Shadow-live snapshot: source=fixture | healthy=True | symbols=4 | errors=4   (4 symbol thiếu fixture)
network : Shadow-live snapshot: source=network | healthy=True | symbols=8 | errors=0
```

---

## 5. Kiểm thử và bằng chứng

| Hạng mục | Kết quả |
|---|---|
| pytest toàn bộ (lần 1) | **407 passed** |
| pytest toàn bộ (lần 2) | **407 passed** |
| `tests/test_sprint14_warm_cash_bridge.py` | 8 passed |
| `tests/test_sprint14_diagnostics.py` | 8 passed |
| `tests/test_sprint14_shadow_live.py` | 10 passed |
| `tests/test_sprint14_shadow_cli.py` | 4 passed |
| E2E fire drill (`scripts/run_e2e_fire_drill.py`) | **20/20 checks PASS** |
| Paper daemon (`--mode paper --strategy barbell --cycles 2`) | 2 chu kỳ OK, Diagnostics in đủ 3 gate |
| Shadow-live fixture + network | healthy=True, không đặt lệnh |

---

## 6. Rủi ro còn lại và khuyến nghị

1. **Gate backtest vs live lệch nhau** (mục 3) — cần thống nhất ngưỡng trước khi chạy tiền thật.
2. **Method D < C**: warm start không tạo alpha trong holdout hiện tại; mọi tuyên bố "cải thiện nhờ warm
   state" phải bị bác bỏ cho tới khi có bằng chứng khác.
3. Shadow-live mới kiểm tra tính sẵn sàng của feed và chạy paper song song, **chưa** dùng snapshot để
   chấm điểm lệch giá bid/ask thật so với mô hình maker fill.
4. Bridge vẫn dựa trên 1 holdout 181 ngày; chưa có kiểm định nhiều cửa sổ (walk-forward) → chưa đủ cơ sở GO.

**Kết luận:** Sprint 14 hoàn thành đúng phạm vi kỹ thuật, không thay đổi tham số alpha (Frozen Alpha),
nhưng **không** thay đổi verdict thương mại: vẫn NO-GO cho testnet tiền thật.

---

## 7. Commit trong sprint

| Commit | Nội dung |
|---|---|
| `e257ce6` | `feat(sprint14): add indicator-warm cash-start holdout evaluation and 3-way drag decomposition` |
| `8007ad2` | `feat(sprint14): add live signal gate diagnostics and warmup verification to dashboard` |
| `60a7705` | `feat(sprint14): implement shadow-live public Binance feed adapter and fail-closed network guards` |
