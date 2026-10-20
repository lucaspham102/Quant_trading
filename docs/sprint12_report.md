# Sprint 12 — Production Hardening, Capital-Scaling Parity & 24/7 Paper Daemon

**Ngày:** 2026-10-20
**Nhánh:** `agy/sprint12-production-hardening` (tách từ `master` tại `aeb55c8`)
**Baseline:** 305 passed → **342 passed** (chạy 2 lần liên tiếp, 47,4 s và 47,4 s)
**Phạm vi:** hạ tầng vận hành. **Không** chỉnh bất kỳ tham số tín hiệu nào.

---

## 1. Kết luận nhanh

| Mã | Cổng | Kết quả | Bằng chứng |
|---|---|---|---|
| PR-1 | Capital parity | **PASS** | 10 test `test_sprint12_capital.py`; chạy thật ra `equity = 3000.0` |
| PR-2 | Active vs compounded | **PASS** (báo cáo), nhưng số **khác** kỳ vọng ban đầu — xem §3 | payload có đủ 2 tỷ suất |
| PR-3 | Spread guard | **PASS** | 3 test + fail-closed `SPREAD_UNKNOWN` ở live/testnet |
| PR-4 | Delta drift repair | **PASS có điều kiện** | 3 test + ngưỡng hiệu dụng theo lot step |
| PR-5 | Kill-switch 6 % | **PASS sau khi vá CRITICAL** | 5 test + 2 test hồi quy intra-cycle |
| PR-6 | Daemon nhiều chu kỳ | **PASS** | `--cycles 2` chạy 2 chu kỳ, dashboard đủ 7 trường |
| PR-7 | testnet/live key gate | **PASS** | thiếu key: live thoát mã 2, testnet rơi về paper |
| PR-8 | Không hồi quy | **PASS** | 342 passed ×2, accounting gap < 1e-6 |

**Phán quyết Sprint 11 (đã khóa) vẫn là NO-GO.** Sprint 12 không mở lại phán quyết đó —
xem §3 để biết vì sao con số "3,75 %" không phải là cửa để lật kết luận.

---

## 2. Bốn module đã làm

### Module 1 — Capital-scaling parity (`src/barbell_portfolio.py`, `src/sprint11_eval.py`)
- `BarbellConfig.reinvest_profits: bool = True`. **True = công thức Sprint 11 nguyên vẹn**
  (carry và overlay đều compounding). **False = neo cứng cả carry và overlay vào
  `initial_capital`** (mô phỏng cash drag) — không dùng để tái lập số Sprint 11.
- Khi `False`, phần carry cộng vào equity dùng **PnL đô-la thật** của sleeve
  (`funding + basis − fees`), không dùng xấp xỉ `r_carry × 3000`, nhờ vậy bất biến kế toán
  vẫn khép kín (< 1e-6) ở cả hai chế độ.
- Báo cáo holdout xuất đồng thời:
  `holdout_return_on_active_capital_pct`, `holdout_return_on_compounded_equity_pct`,
  `holdout_net_pnl`, `holdout_equity_start`, `holdout_fees`,
  `holdout_return_resim_pct` (phương pháp cũ), `holdout_return_pct` (phương pháp slice).

### Module 2 — Sửa lỗi vốn CLI bị ghi đè (`src/state_store.py`, `run_live.py`)
- `BotState` thêm `initial_capital`, `peak_equity`, `strategy`, `kill_switch_triggered`.
- `StateStore.load(initial_capital=…, strategy=…, reset=…)`: state legacy (thiếu
  `initial_capital`) hoặc state chưa giao dịch → đồng bộ về `--capital`. State **đã có
  lãi/lỗ thực hiện** → giữ nguyên equity (không xóa lỗ), chỉ cập nhật vốn gốc.
- `--reset-state`, `--cycles`, `--interval-sec`, `--mode testnet` đã có trong CLI.

### Module 3 — Ba lớp bảo vệ tiền thật (`src/carry_live_executor.py`)
- **Spread guard:** `spread_bps > max_allowed_spread_bps` (10 bps) → hoãn đặt lệnh,
  ghi `SPREAD_TOO_WIDE`. Ở `live`/`testnet`, orderbook không đọc được → **fail-closed**
  (`SPREAD_UNKNOWN`). Ở `paper` giữ fallback giá để không chặn mô phỏng offline.
- **Delta drift:** audit `|spot_qty − perp_qty| × price`; ngưỡng hiệu dụng
  `max(10 USD, lot_step × price × 1.02)` để không lặp sửa bất khả thi; cờ
  `drift_unrepairable` được đặt khi lượng hóa ra 0 và chỉ thử lại khi drift ≥ 5× ngưỡng.
- **Kill-switch:** `current_equity < peak_equity × (1 − 6 %)` → cờ `kill_switch_triggered`
  (dính vĩnh viễn cho tới khi reset), hủy lệnh chờ, đóng overlay bằng market trên sàn
  (live/testnet), **xóa** vị thế carry khỏi state, và được kiểm tra **lần hai giữa chu kỳ**
  trước khi mở lệnh mới.

### Module 4 — Daemon & chế độ kết nối (`run_live.py`)
- `--cycles N` + `--interval-sec S` chạy đúng N chu kỳ; mỗi chu kỳ in dashboard:
  `Capital | Current Equity | Peak Equity | Drawdown % | Carry (Spot vs Perp) | Overlay (Long vs Short) | Kill-Switch`.
- `--mode testnet`: chỉ bật sandbox khi có `BINANCE_API_KEY`/`BINANCE_API_SECRET`; thiếu key
  → log cảnh báo và chạy Public Live Feed Paper Mode. `--mode live` thiếu key → thoát mã 2.

---

## 3. Đính chính số liệu Holdout (quan trọng)

Yêu cầu ban đầu ghi Holdout = **+3,75 %** trên vốn hoạt động 3.000 và +3,24 % trên equity
tích lũy. Kiểm tra lại trên dữ liệu thật, con số này **không đứng vững**:

| Cách đo | Net PnL holdout | Tỷ suất | Ghi chú |
|---|---|---|---|
| Re-sim sạch từ 3.000 (cách Sprint 11 đã khóa) | **96,69 USD** | **3,22 %** | đã trừ **toàn bộ** phí, gồm phí carry |
| Re-sim, lấy carry **gross** (funding + basis) + overlay net | 112,38 USD | 3,75 % | **bỏ sót ~15,7 USD phí carry** → không phải "lãi ròng" |
| Slice liên tục từ lần chạy 3,45 năm | **159,15 USD** | 3,89 % (compound) / 5,30 % (active) | equity đầu cửa sổ 4.091 USD, không phải 3.468 |

Ba điểm cần nói rõ:
1. **112,38 USD không phải lãi ròng.** Đó là carry gross cộng overlay net; phí carry
   (~15,7 USD trong cửa sổ 181 ngày) chưa bị trừ. Lãi ròng all-in là **96,69 USD → 3,22 %**.
2. **Equity đầu holdout khi chạy liên tục là 4.091 USD**, không phải 3.468 USD như giả định.
3. Vì vậy Sprint 12 xuất **cả hai** tỷ suất và yêu cầu **cả hai phương pháp** phải đạt 3,5 %
   thì mới GO (`holdout_consistency` cho slice 3,89 % — PASS; `holdout_consistency_resim`
   cho re-sim 3,24 % — FAIL). Kết quả: **NO-GO**, đúng như Sprint 11.
   Việc chọn phương pháp slice (đạt) thay cho re-sim (không đạt) sau khi đã thấy kết quả
   chính là hình thức dời cột mốc — nên bị chặn bằng luật "cả hai".

---

## 4. Kiểm toán độc lập và các lỗi đã vá

`quick_pro` kiểm toán toàn bộ cơ chế an toàn (`gpt-5.1`/Gemini high, chỉ đọc) phát hiện
**1 CRITICAL + 4 HIGH**. Tất cả đã được vá trong cùng nhánh:

| # | Mức | Lỗi | Cách vá | Test hồi quy |
|---|---|---|---|---|
| 1 | **CRITICAL** | Kill-switch chỉ kiểm tra 1 lần đầu chu kỳ; equity tụt ở bước đóng lệnh vẫn mở lệnh mới. Chỉ xóa overlay trên RAM, không đóng trên sàn | Kiểm tra lại giữa chu kỳ, chặn toàn bộ khối mở lệnh; đóng overlay bằng market; xóa hẳn vị thế carry khỏi state | `test_sprint12_audit_fixes.py::test_kill_switch_recheck_*` |
| 2 | HIGH | Strategy mismatch reset state → mất cờ kill-switch | Cờ an toàn "dính": giữ `kill_switch_triggered` từ state cũ, log error | `test_strategy_mismatch_keeps_kill_switch` |
| 3 | HIGH | Capital sync ghi đè equity → xóa lỗ thực hiện | Chỉ đồng bộ khi state legacy/chưa giao dịch; có lãi-lỗ thì giữ equity | `test_state_sync_keeps_realized_loss` |
| 4 | HIGH | Spread guard bị vô hiệu khi orderbook lỗi (spread = 0) | Fail-closed `SPREAD_UNKNOWN` ở live/testnet | `test_spread_unknown_fail_closed` (2 mode) |
| 5 | HIGH | Drift BTC (lot 0,001 ≈ 60 USD) không thể sửa dưới ngưỡng 10 USD → lặp vô hạn | Ngưỡng hiệu dụng theo lot, cờ `drift_unrepairable`, chỉ thử lại khi drift ≥ 5× | `test_drift_threshold_uses_lot_step`, `test_drift_unrepairable_flag` |
| 6 | HIGH | `gross_usd` overlay vẫn nhân `prev_eq` khi `reinvest_profits=False` | Base = `initial_capital` khi chế độ tĩnh | `test_static_capital_accounting_closes` |

Lỗi #1 là loại lỗi khiến "kill-switch" trở thành hình thức: cờ bật, nhưng tiền vẫn được
đặt thêm trong đúng chu kỳ xấu nhất. Đây là lý do Sprint 12 không thể tuyên bố
"sẵn sàng tiền thật" chỉ vì 3 guardrail đã tồn tại.

---

## 5. Kiểm định kỹ thuật

- `pytest -q` → **342 passed** ×2 (47,4 s / 47,4 s), không flaky.
- Chạy thật: `python run_live.py --mode paper --strategy barbell --capital 3000 --reset-state --cycles 2 --interval-sec 1`
  → 2 chu kỳ, mỗi chu kỳ in dashboard, `Capital=3000.00 | Equity=3000.00 | Peak=3000.00 | DD=0.00% | Kill-Switch=False`;
  file `data/state/bot_state.json` ghi `initial_capital=3000.0`, `current_equity=3000.0`,
  `peak_equity=3000.0`, `strategy="barbell"`, `kill_switch_triggered=false`.
- Bất biến kế toán ở cả `reinvest_profits=True` và `False`: `< 1e-6`.
- Không có lệnh nào bị gửi khi `--mode paper`; live/testnet không có key đều bị chặn.

---

## 6. Điều KHÔNG được kết luận

1. **Chưa chạy testnet/live thật.** Toàn bộ là paper trên dữ liệu cache; chưa có lệnh nào
   chạm API Binance có ký. `executor_functional=True` trong eval đến từ smoke test với stub
   exchange, không phải từ giao dịch thật.
2. **Chưa soak-test 24/7.** Chạy 2 chu kỳ chứng minh vòng lặp đúng, không chứng minh bộ nhớ
   không rò, không chứng minh chịu được reconnect/rate-limit.
3. **Delta drift với BTC/ETH vẫn thô.** Lot step 0,001 BTC ≈ 60 USD nghĩa là drift dưới
   60 USD **không thể** sửa bằng lệnh thị trường; hệ thống chỉ ghi nhận và bỏ qua, chấp nhận
   sai số delta-neutral ở mức đó.
4. **Chưa kiểm toán lại lần hai.** Các bản vá trên do chính nhóm viết và được test hồi quy
   bảo vệ, nhưng **chưa** có vòng audit độc lập thứ hai xác nhận đã sạch.
5. **Lần chạy paper không mở vị thế nào** (score dưới ngưỡng trên dữ liệu cache 8 mã), nên
   đường đi lệnh thật trong paper chưa được kích hoạt end-to-end ở lần chạy này — chỉ có
   unit test + smoke test chứng minh.

---

## 7. Đề xuất trước khi chạy testnet

| Ưu tiên | Việc | Lý do |
|---|---|---|
| 1 | Audit độc lập vòng 2 (chỉ đọc) trên đúng commit này | Lỗi CRITICAL vừa rồi lọt qua 342 test; test không thay được review |
| 2 | Testnet 7 ngày với `--cycles` hẹn giờ, log ra file | Kiểm chứng reconnect, rate-limit, partial fill thật |
| 3 | Nâng ngưỡng drift cho BTC/ETH theo lot step trong tài liệu vận hành | Tránh hiểu nhầm "delta luôn = 0" |
| 4 | Quyết định lại tiêu chí holdout (một phương pháp duy nhất, khóa trước khi chạy) | Hiện có 2 phương pháp cho 2 kết quả 3,24 % và 3,89 % |

---

## 8. Tệp thay đổi

| Tệp | Nội dung |
|---|---|
| `src/state_store.py` | 4 field mới, `load()` với capital sync / strategy / reset / cờ dính |
| `src/carry_live_executor.py` | spread guard (fail-closed), delta drift (ngưỡng theo lot), kill-switch (đóng thật + re-check giữa chu kỳ), dashboard, `--cycles` |
| `src/barbell_portfolio.py` | `reinvest_profits`, PnL đô-la thật khi neo vốn tĩnh, `g_base` cho overlay |
| `src/sprint11_eval.py` | 2 tỷ suất holdout, phương pháp slice vs re-sim, tiêu chí kép |
| `run_live.py` | `--reset-state`, `--cycles`, `--interval-sec`, `--mode testnet` |
| `tests/test_sprint12_capital.py` | 10 test capital parity |
| `tests/test_sprint12_hardening.py` | 11 test 3 guardrail |
| `tests/test_sprint12_reinvest.py` | 6 test chế độ vốn |
| `tests/test_sprint12_audit_fixes.py` | 10 test hồi quy sau kiểm toán |
| `docs/go_no_go_sprint12.md` | Tiêu chí PR-1..PR-8 (khóa trước khi sửa) |
| `docs/sprint11_evidence_tables.md` | Bằng chứng Sprint 11 (bất biến) |

**Commit:** `e6b82c7` → `b6877be` → `7ca4ced` → `dfd6243` → `c385e3c` (audit fixes) → `ad2b46e` (báo cáo này).
