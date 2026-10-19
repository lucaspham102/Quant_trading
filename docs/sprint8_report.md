# Sprint 8 — Báo cáo Nghiệm thu: Dynamic 12-Asset Carry Rotation + Spot-Perp Basis Convergence + Multi-Period Validation & Live Carry Executor

- **Dự án**: `D:\Quant_trading`
- **Nhánh**: `agy/sprint8-dynamic-carry-basis` (tách từ `master`, không đụng `master`)
- **Khoảng dữ liệu**: `2023-05-07` → `2026-10-18` (**3.45 năm**), nến `1h`, 12 coin
- **Tiêu chí đã khóa trước khi code**: `docs/go_no_go_sprint8.md`
- **Dữ liệu thật 100%**: Perp 1h + Spot 1h + Funding 8h cho **cả 12 coin**
- **Kết quả điều phối**: 224 test PASSED (2 lần liên tiếp), 5 chặng hoàn tất, 4 commit.

---

## 0. Tóm tắt dành cho người quyết định

> **PHÁN QUYẾT: NO-GO.**
> Động cơ xoay vòng carry 12 coin **không đạt** mục tiêu lợi suất kinh tế. Trên dữ liệu thật 3.45 năm,
> cấu hình tốt nhất đạt **+1.80%/năm** (rổ 12 coin) và **+2.22%/năm** (rổ 8 coin) — **thấp hơn rất xa**
> ngưỡng khóa `8%/năm` và `6%/năm`, và thấp hơn cả baseline tĩnh Sprint 7.
>
> Nói thẳng: **độ phức tạp tăng lên (xoay vòng động + basis) đã làm hiệu năng GIẢM so với Static Carry 4 coin**
> của Sprint 7 (Sharpe 2.53 vs 4.50; 4/7 kỳ dương vs **7/7** kỳ dương; turnover 83.0 vs 10.9).
> Nguồn lãi gần như chỉ còn **Funding** (46.30 USDT), còn **Basis Convergence gần như vô giá trị** (2.37 USDT,
> ~5% tổng lãi) và bị **phí ăn mòn một nửa** (23.28 USDT).

---

## 1. Câu hỏi nghiên cứu và giả thuyết

Sprint 7 để lại **True MTM Carry** với Sharpe rất cao nhưng lợi suất năm hóa thấp (4.29% DEV / 0.96% Holdout)
vì hai lý do cấu trúc:

1. **Vốn chia tĩnh đều cho 4 coin lớn** (25% NAV mỗi coin) kể cả khi funding bằng 0 → vốn bị "chôn" vào
   tài sản zero-carry. Kiểm chứng dữ liệu: **BNB/USDT:USDT có 2.110/3.784 kỳ funding ĐÚNG BẰNG 0** (55.7%).
2. **Chưa khai thác chênh lệch giá Spot–Perp (Basis Convergence)** — chỉ thu funding.

**Giả thuyết Sprint 8**: xoay vòng động sang top-k coin có điểm Carry+Basis cao nhất sẽ (a) giải phóng vốn
khỏi coin funding thấp và (b) thu thêm Basis PnL, qua đó **gấp đôi lợi suất lên ≥ 8%/năm** mà vẫn giữ
Sharpe ≥ 2.0, MaxDD ≤ 5%.

**Kết quả: giả thuyết bị BÁC BỎ.**

---

## 2. Phương pháp

### 2.1 Dữ liệu
- 12 coin: `BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, TRX` (`UNIVERSE_12_SYMBOLS`).
- Mỗi coin đủ 3 luồng: **Perp OHLCV 1h**, **Spot OHLCV 1h** (bổ sung 8 coin mở rộng), **Funding 8h**.
- Đồng bộ qua `align_spot_perp_funding` — inner join timestamp, `0` NaN, `0` trùng timestamp,
  `30.240` nến/coin (Chặng 1 đã kiểm chứng).

### 2.2 Điểm Carry + Basis (chống look-ahead)
```
F_{i,t}   = EWMA(funding_rate_i, span=6)_{t-1}
B_{i,t}   = basis_pct_{i,t-1} - EMA_168(basis_pct_i)_{t-1}
Score_{i,t} = F_{i,t} + w_basis * B_{i,t} / expected_hold_periods(21)
basis_pct = (perp_close - spot_close) / spot_close
```
Xếp hạng **chỉ tại mốc funding 8h**, dùng dữ liệu `≤ t-1` (mọi chuỗi đều `.shift(1)`).

### 2.3 Quy tắc danh mục
- Ngưỡng vào `θ_entry = 0.00010` (1.0 bps/8h ≈ 10.95%/năm); ngưỡng ra `θ_exit = 0.00002`.
- Tối đa `top_k ∈ {2, 3}` coin, mỗi coin `1/top_k` vốn; buffer `buffer_rank = 1`.
- Hysteresis: `min_hold_periods = 9` (3 ngày) + `min_switch_advantage = 2·cost_round_trip/21`.
- Cấu trúc vốn `spot_perp_3x` (η = 0.75: 75% mua Spot + 25% ký quỹ Short Perp 3x) và `coinm_1leg` (η = 1.0).
- Chế độ phí: `real_taker` (Spot 7.5 + Perp 5 + Slip 2 bps) vs `passive_limit_maker` (Maker 2/5 bps, có điều kiện khớp).

### 2.4 Kiểm định
7 cửa sổ 180 ngày nối tiếp trên **một** Stitched Equity Curve; DSR với `K = 12`; Newey-West Alpha/Beta;
PnL attribution tách `Funding / Basis / Fees`. Đối chiếu 3 cấu hình: DYNAMIC_12, DYNAMIC_8 (Out-of-Universe),
STATIC_4 (baseline Sprint 7).

---

## 3. Kết quả toàn chuỗi 3.45 năm

| Cấu hình | AnnRet %/năm | Sharpe MTM | MaxDD % | DSR | Turnover | Funding $ | Basis $ | Fees $ | Kỳ dương | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| **DYNAMIC_12** (best: `k3\|w0\|passive`) | **1.80** | 2.53 | 0.35 | 0.999 | 83.01 | 46.30 | 2.37 | 23.28 | **4/7** | NO-GO |
| **DYNAMIC_8** (rổ 8 coin, Out-of-Universe) | **2.22** | 3.10 | 0.19 | 1.000 | 58.29 | 45.46 | 2.36 | 16.39 | **4/7** | NO-GO |
| **STATIC_4** (baseline Sprint 7) | **2.19** | **4.50** | **0.20** | 1.000 | **10.91** | 35.92 | 2.76 | **7.66** | **7/7** | BASELINE |

**Nhận xét then chốt**: Dynamic 12 coin **thua** baseline tĩnh 4 coin ở **mọi** chỉ số rủi ro/hiệu quả,
chỉ nhỉnh hơn về funding thô — nhưng phần funding tăng thêm bị **turnover tăng 7.6 lần** ăn hết.

### 3.1 Chi tiết 7 giai đoạn — DYNAMIC_12

| Kỳ | Giai đoạn | NetRet % | AnnRet % | Sharpe | MaxDD % | Funding $ | Basis $ | Fees $ | Dương |
|:--:|:--|---:|---:|---:|---:|---:|---:|---:|:--:|
| P1 | 2023-05 → 2023-11 | +0.07 | 0.15 | 0.37 | 0.10 | 1.08 | 0.40 | 1.18 | ✔ |
| P2 | 2023-11 → 2024-05 | **+5.30** | 11.03 | 6.85 | 0.18 | **33.29** | 0.74 | 12.84 | ✔ |
| P3 | 2024-05 → 2024-10 | +0.05 | 0.11 | 0.26 | 0.13 | 2.86 | 0.58 | 3.21 | ✔ |
| P4 | 2024-10 → 2025-04 | +1.08 | 2.21 | 2.65 | 0.15 | 8.22 | 0.35 | 4.00 | ✔ |
| P5 | 2025-04 → 2025-10 | −0.12 | −0.25 | −0.86 | 0.26 | 0.65 | 0.30 | 1.46 | ✘ |
| P6 | 2025-10 → 2026-04 | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | ✘ |
| P7 | 2026-04 → 2026-10 | −0.09 | −0.18 | −1.73 | 0.09 | 0.21 | 0.00 | 0.60 | ✘ |

**P2 một mình chiếm 33.3/46.3 USDT funding (~72% tổng lãi)** → toàn bộ "alpha" tập trung vào một cửa sổ
uptrend funding cao. **P6 = 0 tuyệt đối** (không mở bất kỳ lệnh nào suốt 6 tháng).

### 3.2 Phân rã nguồn lợi nhuận toàn chuỗi (DYNAMIC_12)

| Nguồn | Giá trị (USDT) | % vốn/năm | Nhận xét |
|---|---:|---:|---|
| Funding PnL | **+46.30** | ≈ +3.22%/năm gross | Nguồn lãi chính |
| Basis Convergence PnL | +2.37 | ≈ +0.17%/năm | **Gần như vô nghĩa (~5% tổng lãi)** |
| Fee Drag | **−23.28** | ≈ −1.62%/năm | **Ăn mòn ~50% funding thô** |
| **Net PnL** | **+25.38** | **+6.35% tổng 3.45 năm** | → AnnRet ≈ 1.80%/năm |

### 3.3 Toàn bộ lưới K = 12 (dấu hiệu quyết định)

| Cấu hình | AnnRet % | Sharpe | MaxDD % |
|---|---:|---:|---:|
| `k2\|w0\|real_taker` | −0.72 | −0.69 | 3.67 |
| `k2\|w0\|passive` | +1.73 | 2.05 | 0.40 |
| `k2\|w0.5\|real_taker` | **−5.34** | −3.19 | 17.25 |
| `k2\|w1\|real_taker` | **−8.06** | −4.16 | **25.19** |
| `k3\|w0\|real_taker` | −0.46 | −0.53 | 2.72 |
| **`k3\|w0\|passive`** (best) | **+1.80** | 2.53 | 0.35 |
| `k3\|w1\|passive` | +0.21 | 0.17 | 2.48 |

Hai sự thật trần trụi:
1. **MỌI cấu hình `real_taker` đều LỖ** (−0.46% đến −8.06%). Chỉ `passive_limit_maker` mới dương.
2. **Tăng `w_basis` làm hiệu năng SỤP**: turnover vọt từ 88.6 → 254.9, phí từ 58.5 → 168.4 USDT,
   trong khi lãi Basis chỉ tăng thêm **+3.93 USDT**. → **Basis Convergence không đủ bù phí.**

---

## 4. Phán quyết GO / NO-GO (theo `docs/go_no_go_sprint8.md`)

| # | Điều kiện đã khóa | Thực đo | Kết quả |
|:--:|---|---|:--:|
| 1 | AnnRet ≥ 8.0%/năm (rổ 12) **và** ≥ 6.0%/năm (rổ 8) | 1.80% / 2.22% | **FAIL** |
| 2 | ≥ 6/7 giai đoạn 6 tháng có Net Return > 0 | 4/7 | **FAIL** |
| 3 | Sharpe ≥ 2.00 **và** DSR ≥ 0.95 **và** MaxDD ≤ 5.0% | 2.53 / 0.999 / 0.35% | PASS |
| 4 | Carry Executor sẵn sàng vận hành | Chạy paper `--once` thành công, exit 0 | PASS |

> ### KẾT QUẢ: **NO-GO** — trượt `cond1_return` và `cond2_consistency`.

---

## 5. Kết quả Vận hành thực tế (Chặng 5)

Lệnh nghiệm thu:
```
python run_live.py --mode paper --strategy dynamic_carry --once
```
- **Thoát mã 0**. Tải dữ liệu **thật** 12 coin (Spot/Perp/Funding) từ sàn Binance, tính `CarryBasisScore`,
  đối chiếu `carry_positions` trong `data/state/bot_state.json`, in bảng ASCII 12 dòng.
- Bảng trạng thái (rút gọn): BTC `+0.69 bps` → SKIP; mọi coin `Score < θ_entry = 1.0 bps` →
  **danh mục 100% USDT** tại thời điểm chạy.
- State ghi **nguyên tử** với 2 field mới `carry_positions: {}` và
  `carry_last_reconcile_ts: "2026-10-19T13:00:00+00:00"` (backward-compatible với state cũ).
- Kết quả này quán quát với P6 trong backtest: **mặt bằng funding hiện tại thấp hơn cả chi phí vòng xoay**
  → bot đứng ngoài. Đây là hành vi phòng vệ hợp lý, nhưng đồng nghĩa **chiến lược không tạo thu nhập
  trong phần lớn thời gian thị trường nguội**.

---

## 6. Phản biện độc lập (Red Team) — các phát hiện quan trọng

| # | Phát hiện | Mức độ | Hàm ý |
|:--:|---|:--:|---|
| 1 | Mô hình `passive_limit_maker` **lạc quan giả tạo**: giả định cả 2 chân khớp đồng thời 100% ngay lập tức với phí Maker, không có legging risk, lệnh trượt bị hủy miễn phí. | **CRITICAL** | Toàn bộ lợi nhuận dương của cấu hình tốt nhất **đến từ giả định giảm phí**, không từ alpha. **Mức tham chiếu trung thực phải là `real_taker` → chiến lược LỖ RÒNG.** |
| 2 | `w_basis > 0` làm sập hiệu năng do **nhiễu vi cấu trúc 1h kích hoạt đảo lệnh**, không phải sai dấu MTM. | **CRITICAL** | Thành phần Basis hội tụ **không có giá trị kinh tế ròng** với tần suất 8h. |
| 3 | Mục tiêu `AnnRet ≥ 8%/năm` **bất khả thi về cấu trúc**: nền funding thô chỉ ~3.2%/năm, η = 0.75 làm giảm đòn bẩy vốn. Muốn đạt 8% phải dùng **đòn bẩy ~3.5–4.5x**, sẽ phá vỡ `MaxDD ≤ 5%`. | **CRITICAL** | **Ngưỡng GO đã khóa là phi thực tế** với rổ tài sản và chi phí này. |
| 4 | **Bất đồng bộ Backtest vs Live**: backtest chọn cấu hình Maker, nhưng `carry_live_executor` khi chạy `live` lại bắn **lệnh MARKET (Taker)** — đúng chế độ đã chứng minh thua lỗ. | **CRITICAL** | **Không được chạy live** với executor hiện tại. |
| 5 | P6 tê liệt dòng tiền do `θ_entry = 1.0 bps` **cứng**, không thích ứng regime. | **HIGH** | Cần ngưỡng động theo phân vị funding lịch sử. |
| 6 | Executor live thiếu **làm tròn khối lượng** (`amount_to_precision`, `minNotional`) và **thiếu rollback khi 1 chân thất bại** (legging risk). | **HIGH** | Không an toàn cho vốn thật. |
| 7 | `run_sprint8.py` xác nhận `cond4` bằng `Path(...).exists()` — **nghiệm thu hình thức**. | LOW | Nên kiểm thử chu trình reconcile thực. |
| 8 | Look-ahead, đếm trùng phí, timing funding | — | **KHÔNG có lỗi** (đã kiểm tra `shift(1)`, `merge_asof backward`, thứ tự MTM→Funding→Rebalance→Rotation). |

---

## 7. Kết luận

1. **Giả thuyết Sprint 8 bị bác bỏ.** Xoay vòng động 12 coin + Basis Convergence **không** tạo ra lợi suất
   kinh tế xứng đáng; nó **giảm** chất lượng so với Static Carry 4 coin của Sprint 7.
2. **Basis Convergence không có alpha ròng** trên dữ liệu này: lãi basis nhỏ (+0.17%/năm) không bù nổi
   phí giao dịch phát sinh do tăng tần suất xoay vòng.
3. **Trần kinh tế của carry delta-neutral với rổ này là ~2.5–3.5%/năm** khi không dùng đòn bẩy margin
   (mức đã kiểm chứng bằng cả `real_taker` lẫn baseline). Mục tiêu 8%/năm là **không khả thi** trong ràng buộc
   `MaxDD ≤ 5%`.
4. **Điều kiện 4 (Production Readiness) đạt về mặt kỹ thuật** (executor chạy, ghi state nguyên tử) nhưng
   **KHÔNG đủ an toàn để chạy vốn thật** do các lỗi CRITICAL ở mục 6 (market order, legging risk, lot size).

---

## 8. Khuyến nghị bước tiếp theo

1. **Dừng hướng Dynamic Basis Rotation (`w_basis > 0`).** Loại thành phần Basis khỏi tín hiệu chọn danh mục.
2. **Quay lại nền tảng Static/Semi-Static Carry của Sprint 7** và tìm cách nâng hiệu dụng vốn an toàn:
   `coinm_1leg` (η = 1.0) hoặc cross-margin được kiểm soát, thay vì xoay vòng liên tục.
3. **Trước khi nghĩ tới live**: (a) khóa mô hình chi phí ở `real_taker` hoặc hybrid thực tế,
   (b) thêm `amount_to_precision`/`minNotional`, (c) thực thi 2 chân nguyên tử + rollback,
   (d) thay `cond4` bằng kiểm thử reconcile thật.
4. **Nếu vẫn theo đuổi β cao**: cân nhắc ngưỡng entry **động theo phân vị funding** và mô hình Basis
   qua EMA/Kalman dài hạn thay vì snapshot 1h.

---

## 9. Hạn chế & tuyên bố trung thực

- **Chi phí**: kết quả tốt nhất dựa trên giả định Maker lạc quan; số liệu `real_taker` (âm) mới là
  tham chiếu thận trọng. Kết luận NO-GO vẫn giữ nguyên (thậm chí mạnh hơn) dưới giả định `real_taker`.
- **Thanh khoản/trượt giá thực tế**, rủi ro sàn, rủi ro depeg stablecoin, biến động basis giữa các coin
  altcoin nhỏ **chưa được mô hình hóa đầy đủ**.
- **Multiple-testing**: đã tính DSR với `K = 12`; các cửa sổ 6 tháng chồng lấn vốn dĩ không độc lập.
- **Baseline STATIC_4** không xuất `attribution_series` từng kỳ → attribution mỗi kỳ = 0 trong bảng
  so sánh kỳ (tổng chuỗi vẫn chính xác: funding 35.92, basis 2.76, fees 7.66 USDT). Đây là hạn chế
  trình bày, **không** ảnh hưởng kết luận (equity curve của baseline vẫn nguyên vẹn).
- **Không chạy live tiền thật** trong Sprint này.

---

## 10. Sản phẩm bàn giao

| Hạng mục | Đường dẫn |
|---|---|
| Tiêu chí khóa | `docs/go_no_go_sprint8.md` |
| Động cơ chính | `src/dynamic_carry_basis.py` |
| Bộ đánh giá 7 kỳ + PnL attribution | `src/sprint8_eval.py` |
| Runner nghiên cứu | `run_sprint8.py` |
| Executor Paper/Live | `src/carry_live_executor.py` |
| CLI vận hành | `run_live.py --strategy dynamic_carry` |
| State mở rộng (backward-compatible) | `src/state_store.py` |
| Test | `tests/test_dynamic_carry_basis.py` (17), `tests/test_sprint8_eval.py` (21), `tests/test_carry_live_executor.py` (7) |
| Kết quả số | `data/sprint8_grid.csv`, `data/sprint8_summary.csv`, `data/sprint8_periods.csv`, `data/sprint8_attribution.csv`, `data/sprint8_report_tables.md` |
| Blueprint & phản biện | `.agy/architect_3a5b8955.md`, `.agy/quick_pro_374db55c.md` |

**Kiểm thử**: `224 passed` (chạy 2 lần liên tiếp, không test nào đỏ).
