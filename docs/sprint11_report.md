# Báo cáo Sprint 11 — Single-Venue Tier 1+ Barbell + Functional Executor

**Ngày kiểm định:** 2026-10-20
**Nhánh:** `agy/sprint11-tier1-barbell-live` (tách từ `master`, `master` chưa bị đụng)
**Vốn:** 3.000 USDT | **Universe:** LIQUID-8 (BTC, ETH, SOL, BNB, DOGE, XRP, AVAX, LINK)
**Dữ liệu:** `data/raw/*.parquet` 1h UTC, 2023-04-01 → 2026-10-19 (≈3,45 năm)
**Holdout:** 2026-04-21 → 2026-10-18 (181 ngày)

---

## 1. Phán quyết

> ### ❌ **NO-GO**

Theo đúng 5 nhóm tiêu chí đã khóa trong `docs/go_no_go_sprint11.md`, cấu hình Primary
**`tier1b / smooth_conviction`** đạt **9/10 tiêu chí**, trượt **duy nhất tiêu chí Holdout**:

| Tiêu chí | Giá trị thực tế | Ngưỡng | Kết quả |
|---|---|---|---|
| AnnRet toàn kỳ | **10,36 %** | ≥ 9,0 % | PASS |
| Số kỳ dương (7 kỳ) | **7/7** | ≥ 6/7 | PASS |
| Holdout return | **3,24 %** | ≥ 3,50 % | **FAIL** (−0,26 pp) |
| Holdout `carry_pnl` | **+54,10 $** | > 0 | PASS |
| Holdout `overlay_net_pnl` | **+58,28 $** | > 0 | PASS |
| Sharpe (1h, MTM) | **1,86** | ≥ 1,80 | PASS |
| MaxDD (MTM) | **4,44 %** | ≤ 7,5 % | PASS |
| \|β_BTC\| | **0,0034** | ≤ 0,15 | PASS |
| `basis_mtm_std` | **0,4577** | > 0 | PASS |
| `accounting_gap` | **2,3e-10** | < 1e-6 | PASS |
| `overlay_quantization_events` | **1.780** | > 0 | PASS |
| `executor_functional` | **True** | == True | PASS |

**Nguyên nhân NO-GO duy nhất:** Holdout 2026-04-21 → 2026-10-18 chỉ đạt **+3,24 %** trên
181 ngày (≈ 6,6 %/năm), thiếu **0,26 pp** so với ngưỡng +3,5 %. Cả hai nhánh đều dương
trong holdout (carry +54,10 $, overlay +58,28 $), nên đây **không phải** lỗi cấu trúc
chiến lược mà là **thiếu độ lớn lợi nhuận** trong nửa cuối 2026.

---

## 2. Thay đổi kiến trúc so với Sprint 10

| Hạng mục | Sprint 10 | Sprint 11 |
|---|---|---|
| Carry base | Cross-exchange (Binance + Bybit + OKX) | **Single-venue Binance**, Tier 1+ |
| Universe carry | CORE-4 | **LIQUID-8**, đầu vào lọc top_k = 4, buffer_rank = 3 |
| Ngưỡng vào/ra | `entry=0.000045`, `exit=-0.00001`, `min_hold=63` | giữ nguyên (Tier 1A/1B) |
| Đòn bẩy | nhị phân theo SNR | **`eta = 0.80`** trên `equity/top_k`, perp 4x, rebalance ±15 %, phí 5 bps |
| Cổng overlay | nhị phân SNR 1,35 / 0,95 | **smooth conviction**: `c_t = 0` nếu `D_t < 0,25`, ngược lại `clip(D_t/1,0; 0,5; 1,0)` |
| Vol targeting | nhị phân (bật/tắt) | `L_t = c_t · clip(σ_target/σ_pair; 0,15; 0,50)`, `σ_target = 7,5 %/năm` |
| Chống trade vụn | không | `min_weight_change_to_trade = 0,08` |
| Lượng hóa lot | một phần | **toàn bộ 2 nhánh** (`quantize_order_qty`, step size 8 mã) |
| Executor | parity 2 chân nhưng không có cờ thật | `executor_functional` là **cờ thật**, gắn với đường đi lệnh |

Ghi chú tương thích ngược (đã cố ý, có test bảo vệ):
- `BarbellConfig.gating_mode` mặc định **`"hard_binary"`** để giữ nguyên hành vi Sprint 10
  (`test_vol_targeting_reduces_leverage_when_pair_vol_doubles`). Sprint 11 phải set
  `gating_mode="smooth_conviction"` tường minh — mọi bảng dưới đây đều làm vậy.
- `BarbellConfig.carry_variant` mặc định **`"tier1b"`**, có `carry_fallback` sang
  `run_cross_exchange_carry` khi universe Tier 1 rỗng (giữ `tests/test_sprint9_barbell.py`).
  Trong lần chạy này `carry_fallback = False` ở cả 4 combo → **không** có fallback ngầm.

---

## 3. Ma trận đối chứng 4 combo (3,45 năm, 3.000 USDT)

| Combo | AnnRet % | Sharpe | MaxDD % | Kỳ dương | Holdout % | CarryPnL $ | OverlayNet $ | BasisStd | β_BTC | Gap | Equity cuối |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tier1a / hard_binary | 5,50 | 1,54 | 3,79 | 5/7 | −0,69 | 483,62 | 217,18 | 0,2444 | 0,0021 | 2,3e-10 | 3.625,53 |
| tier1a / smooth_conviction | 9,20 | 1,69 | 4,54 | 7/7 | +2,90 | 498,62 | 675,79 | 0,2520 | 0,0029 | 2,5e-10 | 4.095,75 |
| tier1b / hard_binary | 6,60 | 1,77 | 3,59 | 6/7 | −0,37 | 682,31 | 216,89 | 0,4365 | 0,0029 | 1,7e-10 | 3.760,80 |
| **tier1b / smooth_conviction** | **10,36** | **1,86** | **4,44** | **7/7** | **+3,24** | **705,53** | **691,09** | **0,4577** | **0,0034** | **2,3e-10** | **4.251,37** |

Đọc bảng:
1. **Universe Tier 1B (8 mã) > Tier 1A (4 mã):** +10,36 % vs +9,20 % AnnRet, carry PnL
   +705,53 $ vs +498,62 $. Xác nhận giả thuyết "mở rộng universe tăng capacity" ở mức
   lợi nhuận, dù Sharpe chỉ nhích +0,17.
2. **Smooth conviction > hard binary:** AnnRet 10,36 % vs 6,60 %, holdout +3,24 % vs −0,37 %.
   Chênh lệch gần như toàn bộ đến từ nhánh overlay (691,09 $ vs 216,89 $) — cổng nhị phân
   cắt mất các đoạn dispersion vừa phải (0,95 < D < 1,35).
3. **Chi phí đổi lấy độ mượt:** overlay fees tăng từ 48,73 $ (hard) lên 91,30 $ (smooth),
   quantize events 1.042 → 1.780. Tổng phí vẫn nhỏ so với lợi nhuận overlay ròng.
4. **Trung tính thị trường:** |β_BTC| ≤ 0,0034 ở mọi combo — đúng thiết kế delta-neutral.

### 3.1 Bảy kỳ lưới (Primary: tier1b / smooth_conviction)

| Kỳ | Lợi nhuận % |
|---|---|
| 1 | +1,07 |
| 2 | +7,54 |
| 3 | +0,91 |
| 4 | +16,03 |
| 5 | +5,83 |
| 6 | +1,15 |
| 7 | +4,04 |

7/7 kỳ dương. Kỳ yếu nhất là kỳ 3 (+0,91 %) — đúng giai đoạn dispersion thấp, nơi smooth
conviction hạ `c_t` và vol-targeting giảm đòn bẩy (leverage trung bình toàn kỳ chỉ 0,155).

### 3.2 Holdout — nơi phán quyết rơi

| Combo | Holdout % | Carry PnL $ | Overlay net $ |
|---|---|---|---|
| tier1a / hard_binary | −0,69 | 36,17 | −48,83 |
| tier1a / smooth_conviction | +2,90 | 36,95 | +58,40 |
| tier1b / hard_binary | −0,37 | 53,06 | −49,00 |
| **tier1b / smooth_conviction** | **+3,24** | **54,10** | **+58,28** |

Cấu trúc holdout **lành mạnh** (cả 2 nhánh dương, β ≈ 0), chỉ thiếu 0,26 pp biên độ.
Lưu ý carry PnL holdout chỉ 54,10 $ trong 181 ngày, tức ≈ 36 %/năm trên vốn carry —
thấp hơn nhiều so với mức 705,53 $/3,45 năm toàn kỳ, phản ánh funding 2026 đã co lại.

---

## 4. Executor: từ "parity hình thức" sang "cờ thật"

Lỗi Sprint 10 (GO-5): `executor_functional` chỉ là **tham số truyền tay** vào hàm đánh giá,
không có bằng chứng executor thực sự đi được đường lệnh. Sprint 11 sửa triệt để:

| Thành phần | Trạng thái |
|---|---|
| `execute_maker_then_taker_pair(...)` | Vòng TTL poll thật (`sleep_fn` inject được), partial fill hedge đúng phần khớp, hết hạn → `cancel_maker_leg1`, không bao giờ bỏ lệnh mồ côi |
| `execute_overlay_rebalance(...)` | So `overlay_positions` (state) với target của `build_barbell_allocations`, lọc `min_notional_usd = 10 $`, lượng hóa từng chân, ghi `overlay_orders` |
| `BotState` | Thêm `carry_orders`, `overlay_orders`, `overlay_positions`, `executor_functional` (load an toàn với state JSON cũ) |
| Cờ `executor_functional` | Chỉ bật `True` khi **cả hai** nhánh đã route lệnh thật (carry + overlay) |
| `smoke_test_executor` | Stub exchange in-memory: `methods_present=True`, `state_fields_present=True`, `carry_routed=True`, `overlay_routed=True`, **`executor_functional=True`** |

`run_live.py --mode paper --strategy barbell --capital 3000 --once` chạy hết chu trình
reconcile → build allocation → persist sleeve → overlay rebalance mà không cần mạng
(Universe: LIQUID-8). Đây là kiểm chứng đường đi lệnh, **không phải** kiểm chứng PnL live.

---

## 5. Kiểm định kỹ thuật

- **Bất biến kế toán** (đo trên cả 4 combo, dữ liệu thật):
  `sum(net_pnl) − (final_equity − initial_capital) ≤ 2,5e-10` — đạt yêu cầu < 1e-6.
- **Lượng hóa lot:** 1.780 sự kiện quantize ở Primary; mọi notional là bội số
  `step_size × price` (test `test_sprint11_barbell.py`).
- **Look-ahead:** EWMA funding `span = 42` trên chuỗi 8h rồi `shift(1)`; funding chỉ thu
  tại nến 0/8/16h; tất cả tín hiệu overlay dùng dữ liệu tới `t−1`.
- **Test suite:** `305 passed` (259 baseline + 7 Chặng 1 + 14 Chặng 2 + 10 executor +
  15 eval), chạy 2 lần liên tiếp không flaky.

---

## 6. Điều KHÔNG được kết luận từ báo cáo này

1. **Không** kết luận chiến lược "gần đạt" rồi nới ngưỡng. Ngưỡng 3,5 % đã khóa trước khi
   chạy; hạ nó xuống 3,2 % là tự lừa mình. Trạng thái đúng là **NO-GO**.
2. **Không** kết luận hiệu năng live. Toàn bộ số liệu là mô phỏng bar-close 1h, khớp lệnh
   giả định tại giá đóng cửa, phí 5 bps một chiều; chưa có trượt giá thực, chưa có rủi ro
   thanh lý thực, chưa có độ trễ API.
3. **Không** kết luận overlay "an toàn" — overlay đóng góp 691,09 $/3,45 năm nhưng phụ
   thuộc vào dispersion; kỳ 3 chỉ +0,91 % cho thấy nhánh này có thể gần như tắt.
4. **Không** kết luận `executor_functional=True` nghĩa là sẵn sàng chạy tiền thật: cờ này
   được bật bằng stub exchange trong smoke test + paper mode, chưa qua testnet có lệnh thật.

---

## 7. Đề xuất bước tiếp theo (chờ người dùng quyết)

| Hướng | Nội dung | Rủi ro |
|---|---|---|
| A. Mở rộng holdout | Chạy thêm cửa sổ holdout thứ hai (ví dụ 2025-10-01 → 2026-04-20) để xem 3,24 % là nhiễu hay xu hướng | Tăng số lần thử → nguy cơ overfit lựa chọn cửa sổ |
| B. Tăng hiệu suất carry holdout | Xem xét `eta`, `top_k`, ngưỡng `entry_funding` trong giai đoạn funding thấp | Chạm vào tham số đã khóa, phải khóa lại tiêu chí trước khi chạy |
| C. Chấp nhận NO-GO | Giữ Tier 1B + smooth làm cấu hình nghiên cứu, không triển khai vốn | Không có |

Việc chọn hướng là quyết định của người dùng — báo cáo này không tự hạ chuẩn.

---

## 8. Tệp sinh ra

| Tệp | Nội dung |
|---|---|
| `docs/go_no_go_sprint11.md` | Tiêu chí GO/NO-GO đã khóa (Chặng 0) |
| `data/sprint11_payload.json` | Payload đầy đủ: 4 combo, 7 kỳ, holdout, executor, verdict |
| `data/sprint11_report_tables.md` | 4 bảng markdown tự sinh từ payload |
| `src/sprint11_eval.py` | `run_sprint11_tiers`, `smoke_test_executor`, `evaluate_sprint11_go_no_go` |
| `run_sprint11.py` | CLI tái lập toàn bộ báo cáo: `python run_sprint11.py --capital 3000` |
| `tests/test_sprint11_tier1_carry.py` | 7 test carry Tier 1+ |
| `tests/test_sprint11_barbell.py` | 14 test barbell smooth + quantize + accounting |
| `tests/test_sprint11_executor.py` | 10 test executor (full/partial/TTL/overlay/state) |
| `tests/test_sprint11_eval.py` | 15 test eval & GO/NO-GO |

**Commit:** `1c9e411` (Chặng 0) → `ea7a875` (Chặng 1) → `c06cc2e` (Chặng 2) →
`4d70878` (Chặng 3) → commit Chặng 4/5.
