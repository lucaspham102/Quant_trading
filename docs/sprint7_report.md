# Sprint 7 — Cross-Sectional Breadth Expansion (12 Symbols) & True Spot-Perp MTM Carry

**Trạng thái:** HOÀN TẤT — **NO-GO cho cả hai hướng.**
**Ngày chạy:** 2026-10-19 · **Nhánh:** `agy/sprint7-xs12-and-mtm-carry`
**Tiêu chí đã khóa trước khi code:** [`docs/go_no_go_sprint7.md`](go_no_go_sprint7.md)
(commit `0735605`, trước mọi kết quả).

> Nguyên tắc tuân thủ: lưới tham số và ngưỡng GO/NO-GO được khóa **trước** khi có kết quả;
> Holdout 180 ngày **chỉ mở đúng 1 lần** ở Chặng 5; không sửa lưới/tiêu chí sau khi thấy số.

---

## 0. Tóm tắt phán quyết

| Hướng | Cấu hình báo cáo | DEV | HOLDOUT | Phán quyết |
|---|---|---|---|---|
| **True MTM Carry** | `coinm_1leg` + `hysteresis` | +13.23% (4.29%/năm), Sharpe 5.86, MaxDD 0.29% | +0.47% (0.96%/năm), Sharpe 3.39, MaxDD 0.034% | **NO-GO** — trượt ngưỡng 6.0%/năm ở cả DEV và HOLDOUT |
| **XS Momentum 12 coin** | `lb336 / buf2 / tilt0.25`, `top_k=3` | +111.42% (28.79%/năm), Sharpe 1.27, α_t 2.31, β −0.045, DSR 0.652, MaxDD 17.93% | **−8.31%**, Sharpe −1.15, MaxDD 12.78% | **NO-GO** — trượt DSR, MaxDD DEV, và toàn bộ điều kiện Holdout |
| **XS Momentum 8 coin mới (Out-of-Universe)** | cùng tham số, `top_k=2` | +41.33% (12.40%/năm), Sharpe 0.57, α_t 1.05, DSR 0.205, MaxDD 25.97% | **−13.51%**, Sharpe −1.36, MaxDD 19.94% | **NO-GO** — trượt cả 3 điều kiện |

**Kết luận:** không có cấu hình nào đạt tiêu chí GO. Không triển khai vốn thật.

---

## 1. Dữ liệu thật & kiểm tra chất lượng

Tải trực tiếp từ Binance (ccxt, `2023-05-07` → `2026-10-18`, nến 1h):

| Nhóm | Số symbol | Số nến / symbol | Trùng lặp | Khe hở thời gian |
|---|---|---|---|---|
| Perp OHLCV 1h (12 coin) | 12 | 30,275 | 0 | 0 |
| Funding Rate 8h (12 coin) | 12 | 3,785 kỳ | 0 | — |
| Spot OHLCV 1h (4 coin core) | 4 | 30,276 | 0 | 0 |

- Spot: `BTCUSDT_SPOT_1h`, `ETHUSDT_SPOT_1h`, `SOLUSDT_SPOT_1h`, `BNBUSDT_SPOT_1h`.
- Đồng bộ: **inner join chính xác theo timestamp UTC** giữa Spot và Perp, không ffill giá;
  `basis_pct = (perp_close − spot_close)/spot_close`.
- Basis thực tế (toàn kỳ DEV): trung bình −2.5 đến +0.2 bps, độ lệch chuẩn **4.2–6.5 bps**,
  cực trị −140 bps (SOL/BNB) — đây chính là nguồn rủi ro MTM mà Sprint 6 bỏ qua.

---

## 2. Kết quả DEV (`2023-05-07` → `2026-04-21`, ~2.96 năm)

### 2.1 True MTM Carry (4 coin core)

| Cấu hình | Net Ret % | Ann % | Sharpe | MaxDD % | MTM PnL (USD) | Funding (USD) | Fees (USD) | Rebalances |
|---|---|---|---|---|---|---|---|---|
| true_mtm · coinm_1leg · hysteresis | **13.23** | **4.29** | 5.86 | 0.29 | +4.65 | +52.68 | 4.40 | 0 |
| true_mtm · coinm_1leg · passive | 16.56 | 5.31 | 4.74 | 0.75 | +2.89 | +75.08 | 11.74 | 0 |
| true_mtm · spot_perp_2leg · hysteresis | 7.42 | 2.45 | 4.72 | 0.20 | +2.78 | +34.55 | 7.66 | 10 (price) |
| true_mtm · spot_perp_2leg · passive | 7.74 | 2.55 | 3.27 | 0.94 | +1.63 | +48.79 | 19.45 | 20 (price) |
| *idealized* · coinm_1leg · hysteresis (Sprint 6) | 12.02 | 3.91 | **12.23** | 0.08 | 0.00 | +52.46 | 4.38 | 0 |
| *idealized* · coinm_1leg · passive (Sprint 6) | 15.80 | 5.08 | **14.09** | 0.54 | 0.00 | +74.89 | 11.69 | 0 |
| *idealized* · spot_perp_2leg · hysteresis (Sprint 6) | 6.71 | 2.22 | **8.99** | 0.16 | 0.00 | +34.47 | 7.64 | 10 |
| *idealized* · spot_perp_2leg · passive (Sprint 6) | 7.33 | 2.42 | **7.91** | 0.84 | 0.00 | +48.72 | 19.41 | 20 |

**Xác nhận tái lập:** dòng *idealized* `coinm_1leg/hysteresis` cho **12.02% / Sharpe 12.22** —
khớp chính xác báo cáo Sprint 6 ⇒ phần MTM mới được cộng thêm, không làm thay đổi mô hình cũ.

**Tác động của MTM thật:** Sharpe **12.23 → 5.86** (−52%), MaxDD **0.08% → 0.29%**, xuất hiện
PnL basis hàng giờ (+4.65 USD/3 năm). Mô hình kế toán khớp tuyệt đối:
`Equity cuối = 400 + Funding + MTM − Fees` (sai số 0.0).

### 2.2 Cross-Sectional Momentum (16 cấu hình)

| Config | Net Ret % | Ann % | Sharpe | MaxDD % | DSR | α_t | β | Turnover | ΔTurnover vs buf0 |
|---|---|---|---|---|---|---|---|---|---|
| xs8 lb336 buf0 tilt0 | 11.98 | 3.90 | 0.28 | 38.17 | 0.093 | 0.55 | −0.036 | 337.0 | — |
| xs8 lb336 buf0 tilt0.25 | 45.81 | 13.59 | 0.60 | 36.43 | 0.222 | 1.12 | −0.049 | 353.0 | — |
| xs8 lb336 buf2 tilt0 | 37.86 | 11.46 | 0.55 | 31.10 | 0.195 | 1.04 | −0.039 | 143.0 | −57.6% |
| xs8 lb336 buf2 tilt0.25 | 41.33 | 12.40 | 0.57 | 25.97 | 0.205 | 1.05 | −0.035 | 157.0 | −55.5% |
| xs8 lb504 buf0 tilt0 | 2.17 | 0.73 | 0.16 | 36.94 | 0.064 | 0.39 | −0.044 | 277.0 | — |
| xs8 lb504 buf0 tilt0.25 | 50.84 | 14.90 | 0.65 | 32.14 | 0.247 | 1.21 | −0.048 | 302.5 | — |
| xs8 lb504 buf2 tilt0 | 7.25 | 2.39 | 0.22 | 31.31 | 0.078 | 0.50 | −0.044 | 103.5 | −62.6% |
| xs8 lb504 buf2 tilt0.25 | 25.60 | 8.01 | 0.42 | 30.69 | 0.142 | 0.86 | −0.048 | 110.5 | −63.5% |
| xs12 lb336 buf0 tilt0 | 104.97 | 27.45 | 1.197 | 21.35 | 0.604 | 2.149 | −0.028 | 319.3 | — |
| xs12 lb336 buf0 tilt0.25 | 63.61 | 18.10 | 0.863 | 25.59 | 0.376 | 1.581 | −0.040 | 336.3 | — |
| xs12 lb336 buf2 tilt0 | 99.62 | 26.32 | 1.192 | 21.08 | 0.599 | 2.108 | −0.036 | 172.3 | −46.0% |
| **xs12 lb336 buf2 tilt0.25** | **111.42** | **28.79** | **1.270** | **17.93** | **0.652** | **2.305** | **−0.045** | **173.7** | **−48.4%** |
| xs12 lb504 buf0 tilt0 | 54.03 | 15.72 | 0.762 | 20.50 | 0.313 | 1.399 | −0.032 | 258.3 | — |
| xs12 lb504 buf0 tilt0.25 | 58.28 | 16.79 | 0.818 | 19.52 | 0.348 | 1.548 | −0.046 | 276.3 | — |
| xs12 lb504 buf2 tilt0 | 45.01 | 13.38 | 0.696 | 21.74 | 0.274 | 1.333 | −0.040 | 124.7 | −51.7% |
| xs12 lb504 buf2 tilt0.25 | 70.92 | 19.86 | 0.954 | 18.51 | 0.438 | 1.838 | −0.054 | 125.0 | −54.8% |

**Nhận xét DEV:**
1. **Buffer band hoạt động đúng mục tiêu:** turnover giảm **46–64%** ở mọi cặp so sánh,
   gần như không mất lợi nhuận (thậm chí tăng Sharpe ở `xs12/lb336`).
2. **Mở rộng 4 → 12 coin là yếu tố tạo alpha mạnh nhất:** Sharpe 1.27 và α_t 2.31 (12 coin)
   so với 0.57 / 1.05 (8 coin mới). β ≈ 0 ở mọi cấu hình ⇒ market-neutral đúng nghĩa.
3. **Nhưng 8 coin mới độc lập KHÔNG xác nhận được alpha:** DSR ≤ 0.25, α_t ≤ 1.21,
   MaxDD 26–38% ⇒ nghi vấn lợi nhuận 12 coin đến từ 4 coin cũ (BTC/ETH/SOL/BNB).
4. **DSR là nút thắt:** kể cả cấu hình tốt nhất chỉ đạt **0.652** so với ngưỡng **0.90**
   (đã phạt multiple-testing với K=16).

---

## 3. HOLDOUT 180 ngày (`2026-04-21` → `2026-10-18`) — mở đúng 1 lần

Cấu hình đã khóa từ DEV (chọn theo DEV Sharpe, trước khi chạm Holdout):
- **Carry:** `true_mtm | coinm_1leg | hysteresis` (DEV Sharpe cao nhất trong 4 cấu hình MTM).
- **XS:** `lb336 / buf2 / tilt0.25` cho **cả** rổ 12 coin (`top_k=3`) và rổ 8 coin mới (`top_k=2`)
  — cùng một chiến lược, khác rổ tài sản (đúng nghĩa Out-of-Universe test).

| Hướng / rổ | Net Ret % | Ann % | Sharpe | MaxDD % | DSR | α_t | β | Phán quyết |
|---|---|---|---|---|---|---|---|---|
| Carry `coinm_1leg` hysteresis | **+0.47** | **+0.96** | 3.39 | **0.034** | 0.908 | 4.60 | −0.0003 | **NO-GO** |
| XS 12 coin (`top_k=3`) | **−8.31** | −16.13 | −1.15 | 12.78 | 0.005 | −0.75 | −0.050 | **NO-GO** |
| XS 8 coin mới (`top_k=2`) | **−13.51** | −25.50 | −1.36 | 19.94 | 0.003 | −0.94 | −0.110 | **NO-GO** |

### 3.1 Kiểm tra từng điều kiện đã khóa

**True MTM Carry:**

| Điều kiện | Ngưỡng | DEV | HOLDOUT | Kết quả |
|---|---|---|---|---|
| MaxDD MTM > 0 và ≤ 10% | 0 < DD ≤ 10 | 0.29 ✔ | 0.034 ✔ | ĐẠT |
| Ann Net Return | ≥ 6.0%/năm | 4.29 ✘ | 0.96 ✘ | **TRƯỢT** |
| Sharpe MTM | ≥ 1.50 | 5.86 ✔ | 3.39 ✔ | ĐẠT |

**Cross-Sectional (12 coin):**

| Điều kiện | Ngưỡng | DEV | HOLDOUT | Kết quả |
|---|---|---|---|---|
| α_t vs BTC / \|β\| | ≥ 2.20 / ≤ 0.15 | 2.31 / 0.045 ✔ | — | ĐẠT |
| Sharpe / DSR | ≥ 1.10 / ≥ 0.90 | 1.27 / 0.652 ✘ | — | **TRƯỢT** |
| Holdout Net / Sharpe / MaxDD | > 8% / ≥ 1.20 / ≤ 15% | — | −8.31 / −1.15 / 12.78 ✘ | **TRƯỢT** |

**Cross-Sectional (8 coin mới):** trượt **cả 3** điều kiện (α_t 1.05 < 2.00; Sharpe 0.57 < 1.10,
DSR 0.205 < 0.90; Holdout −13.51% < +8%, Sharpe −1.36, MaxDD 19.94% > 15%).

---

## 4. Phát hiện chính

1. **Sprint 6 đã phóng đại Sharpe của Carry.** Mô hình lý tưởng hóa (không MTM) cho Sharpe 12.23;
   khi nạp giá Spot/Perp thật từng giờ, Sharpe còn **5.86** và phát sinh PnL basis ±4–6 bps.
   Tuy vậy, MTM **không** đủ để phá hủy tính delta-neutral: MaxDD thực vẫn rất nhỏ (0.03–0.94%).
   Vấn đề của Carry là **mức sinh lời**, không phải rủi ro: 4.29%/năm (DEV) và 0.96%/năm (Holdout),
   đều dưới ngưỡng 6%/năm.
2. **Buffer band là cải tiến kỹ thuật thành công, giữ được qua mọi cấu hình:** turnover giảm
   46–64%, MaxDD DEV giảm (12 coin: 21.4% → 17.9%), Sharpe tăng nhẹ. Đây là thành quả nên giữ lại.
3. **Carry tilt (λ = 0.25) không ổn định:** giúp `xs12/lb336` (Sharpe 0.86 → 1.27) nhưng làm xấu
   `xs8/lb504` (0.65 → 0.42) và sụp đổ hoàn toàn trên Holdout ⇒ không nên coi là edge độc lập.
4. **Holdout đảo chiều mạnh với XS Momentum:** cả 12 coin và 8 coin đều lỗ hai chữ số trong 6 tháng.
   Đây là bằng chứng regime shift (thị trường 2026 H2 không còn trending như 2023–2026 H1),
   không phải lỗi kỹ thuật.
5. **Không có bằng chứng Out-of-Universe:** 8 coin chưa từng thấy ở Sprint 1–6 chỉ đạt α_t 1.05
   trên DEV và lỗ 13.5% trên Holdout ⇒ alpha 12 coin nhiều khả năng tập trung ở nhóm core cũ.

---

## 5. Kiểm định độc lập & tuân thủ phương pháp

- **Review độc lập (`quick_pro`) trước khi mở Holdout:** *không có BLOCKER*; xác nhận công thức
  MTM đúng docs mục 7, kế toán khớp 0.0 USD, `merge_asof(direction="backward")` chống look-ahead,
  score/EMA đều `.shift(1)`, DSR dùng K=16 nhất quán 8760 nến. Audit trail: `.agy/quick_pro_4a115b8f.md`.
- **Hai điểm review đã sửa trước khi mở Holdout:**
  1. Thêm **guardrail**: `--stage holdout` bắt buộc có `--select-csv`, nếu không sẽ từ chối chạy
     (chặn việc vô tình quét 16 cấu hình trên Holdout).
  2. Chọn **một bộ tham số duy nhất** trên rổ 12 coin rồi áp dụng y hệt cho rổ 8 coin mới
     (trước đó chọn riêng từng rổ, làm mất ý nghĩa kiểm định Out-of-Universe).
- **Giữ nguyên các quyết định đã khóa:** phí tái cân bằng ký quỹ vẫn nhân 2 (conservative,
  theo docs mục 6) dù review đánh dấu [LOW] là "có thể bảo thủ quá mức" — không sửa sau khi khóa.
- **Không tinh chỉnh sau khi thấy kết quả:** toàn bộ 16 + 4 cấu hình DEV và 3 lần chạy Holdout
  đều dùng đúng lưới đã khóa.

---

## 6. Hạn chế

1. **Chỉ 1 Holdout 180 ngày** — không đủ để kết luận thống kê về regime; kết quả lỗ có thể do
   giai đoạn, nhưng theo luật chơi đã khóa thì vẫn là NO-GO.
2. **Mô hình thực thi còn lý tưởng:** không mô hình hóa độ trễ rót lệnh Spot/Perp, rủi ro
   stablecoin/USDT, giới hạn rút/chuyển giữa 2 sàn, funding 8h chốt đúng nến, không có chi phí
   vay/margin thực tế ngoài phí taker.
3. **Carry chỉ chạy 4 coin core**, chưa mở rộng carry sang 8 coin mới (Sprint 7 chỉ mở rộng XS).
4. **Funding cho XS tính trên notional cuối kỳ trước**, chưa mô hình hóa funding kỳ vọng biến đổi
   trong nến; funding là nguồn lợi nhỏ (+25.9 USD/3 năm) so với phí (80.2 USD).
5. **DSR với K=16** là bảo thủ nhưng đúng cam kết; nếu dùng K=8 cho từng rổ, DSR vẫn chỉ ~0.4–0.5,
   không thay đổi phán quyết.

---

## 7. Khuyến nghị

1. **Không triển khai vốn thật** cho cả Carry và XS dựa trên Sprint 7.
2. **Giữ lại 2 cải tiến kỹ thuật** (đã có test): True MTM Carry và Buffer band — dùng làm nền cho
   các sprint sau, không cần làm lại.
3. Hướng đi tiếp có giá trị nhất:
   - **Carry:** tăng vốn phân bổ/đa dạng hoá sang 8 coin mới để nâng return tuyệt đối, và/hoặc
     chấp nhận hạ ngưỡng return kỳ vọng (3–5%/năm) nếu mục tiêu là "tiền gửi USD có bảo hiểm".
   - **XS:** kiểm định trên universe lớn hơn (20–30 coin) và thêm chế độ phòng thủ theo regime
     (ví dụ tắt momentum khi breadth/vol vượt ngưỡng) trước khi nghĩ tới GO.
4. Mọi thay đổi tiếp theo phải là **Sprint mới với tiêu chí khóa mới** — không sửa lưới Sprint 7.

---

## 8. Tái lập

```bash
# Dữ liệu (idempotent, có cache)
python run_sprint7.py --download --stage dev

# DEV
python run_sprint7.py --stage dev

# HOLDOUT (bắt buộc --select-csv; chỉ chạy 1 lần)
python run_sprint7.py --stage holdout --select-csv data/sprint7_xs_dev.csv \
    --out-md data/sprint7_holdout_tables.md
```

Artifact cục bộ (không commit vì `data/` nằm trong `.gitignore`):
`data/sprint7_carry_dev.csv`, `data/sprint7_xs_dev.csv`,
`data/sprint7_carry_holdout.csv`, `data/sprint7_xs_holdout.csv`.

**Kiểm thử:** `179 passed` (2 lần liên tiếp) trước khi mở Holdout — gồm 7 test MTM/Spot loader mới
và 5 test Top-K/Buffer/Carry-tilt mới.
