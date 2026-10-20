# Sprint 10 Report — Production-Grade Barbell (Lot-Size, True Basis MTM, SNR Gate, Vol Targeting)

**Ngày:** 2026-10-20
**Nhánh:** `agy/sprint10-production-barbell`
**Vốn:** 3,000.00 USDT
**Chuỗi:** 2023-05-07 → 2026-10-18 (30,241 nến 1h, 3.45 năm, 7 kỳ × 180 ngày), compounding
**Phí:** `maker_taker_hedge` (20 bps khứ hồi Perp-Perp), `lot_size_quantization=True`, True Basis MTM, phí rebalance
**Test suite:** 259 passed (Sprint 9: 241)
**Tiêu chí khóa:** `docs/go_no_go_sprint10.md`

---

## 1. Kết luận điều hành

> ## ❌ NO-GO — không triển khai tiền thật.

Sprint 10 đã sửa đúng 4 lỗi kỹ thuật của Sprint 9, nhưng **kết quả trung thực sau khi sửa lại xấu hơn**: mô hình Barbell phức tạp không còn vẻ ngoài 13%/năm.

| Tier | Final Eq | AnnRet | Sharpe | MaxDD | Kỳ dương | Holdout | Verdict |
|---|---|---|---|---|---|---|---|
| 1 — Binance Static Carry (long-horizon) | 3,493.66 | **4.51%** | **4.89** | **0.64%** | **7/7** | +1.86% | NO-GO (GO-1) |
| 2 — Multi-Exchange Best-Venue | 3,404.70 | 3.74% | 3.71 | 0.85% | 7/7 | +0.55% | NO-GO (GO-1/2) |
| 3 — Hybrid Cross-Exchange Carry | 3,314.43 | 2.93% | 2.16 | 2.40% | 5/7 | −1.02% | NO-GO |
| 4 — Production Barbell | 3,408.21 | 3.77% | 1.37 | 3.92% | 5/7 | −1.97% | NO-GO |

**Ba kết luận quan trọng:**
1. **Tier 1 (carry thuần Binance) đánh bại Tier 3 và Tier 4** trên mọi chiều: lợi nhuận cao hơn, Sharpe 4.89, MaxDD 0.64%, 7/7 kỳ dương. Chi phí giao dịch chỉ 97.25 USDT so với 279–287 USDT của Tier 3/4.
2. **Carry liên sàn Core-4 không đủ bù phí**: funding tăng thêm 8.95 USDT so với Tier 1 nhưng phí tăng 182 USDT. Chênh lệch funding 0.26 bps/8h đã bị HFT san phẳng.
3. **Overlay sau khi sửa gate chỉ đóng góp 92.94 USDT gross** (so với 1,080.54 USDT của Sprint 9 khi gate bị hỏng). Đó là con số thật; Sprint 9 chỉ đẹp vì regime gate luôn bật và basis MTM bị gán 0.

---

## 2. Đối chiếu Sprint 9 → Sprint 10

| Chỉ số | Sprint 9 | Sprint 10 | Diễn giải |
|---|---|---|---|
| Tier 3 AnnRet | 3.36% | 2.93% | Basis MTM thật + làm tròn lô + vào lệnh nhiều hơn → phí tăng |
| Tier 3 Sharpe | 9.35 (ảo) | **2.16** (thật) | Bóc trần: basis std 0.433 USDT/giờ |
| Tier 3 kỳ dương | 5/7 | 5/7 | Không đổi |
| Tier 4 AnnRet | 13.00% | 3.77% | Gate thật + vol targeting + DD cushion |
| Tier 4 Sharpe | 1.18 | 1.37 | Tăng nhẹ, vẫn < 1.80 |
| Tier 4 MaxDD | 13.86% | **3.92%** | Đạt GO-3 về DD |
| Tier 4 overlay active | 98.9% (hỏng) | **34.6%** | Đạt GO-4 [25%, 75%] |
| Tier 4 holdout | +5.86% | **−1.97%** | Overlay lỗ −13.87 USDT trong P7 |
| Tier 1 AnnRet | 2.85% | 4.51% | Long-horizon 42 kỳ + entry 0.5 bps + basis MTM |
| Test | 241 | 259 | +18 test Sprint 10 |

**Kết luận đối chiếu:** Sprint 9 "thắng" nhờ 3 lỗi cùng lúc (gate hằng đúng, basis = 0, không làm tròn lô). Sprint 10 sửa cả 3 → lợi nhuận giảm 9.2 điểm phần trăm. Đây là **giá của tính trung thực**, không phải thoái bộ kỹ thuật.

---

## 3. Bảng 7 kỳ

### Tier 1 — Binance Static Carry (long-horizon)
| Kỳ | Net % | Ann % | Sharpe | MaxDD % | Dương |
|---|---|---|---|---|---|
| P1 | +1.20 | 2.44 | 2.53 | 0.26 | ✔ |
| P2 | +7.22 | 15.17 | 13.41 | 0.16 | ✔ |
| P3 | +2.14 | 4.38 | 6.48 | 0.20 | ✔ |
| P4 | +2.55 | 5.24 | 3.60 | 0.64 | ✔ |
| P5 | +1.01 | 2.06 | 2.72 | 0.16 | ✔ |
| P6 | +0.53 | 1.07 | 2.28 | 0.19 | ✔ |
| P7 (holdout) | +0.91 | 1.86 | 2.80 | 0.16 | ✔ |

### Tier 3 — Hybrid Cross-Exchange Carry
| Kỳ | Net % | Ann % |
|---|---|---|
| P1 | +1.13 | 2.31 |
| P2 | +7.31 | 15.38 |
| P3 | +0.80 | 1.63 |
| P4 | +2.44 | 5.02 |
| P5 | +0.57 | 1.16 |
| P6 | −1.48 | −2.97 |
| P7 (holdout) | −0.51 | −1.02 |

### Tier 4 — Production Barbell
| Kỳ | Net % | Ann % | Sharpe | MaxDD % | Dương |
|---|---|---|---|---|---|
| P1 | +5.91 | 12.35 | 3.36 | 0.84 | ✔ |
| P2 | +4.59 | 9.52 | 2.39 | 2.81 | ✔ |
| P3 | +0.33 | 0.66 | 0.36 | 1.71 | ✔ |
| P4 | +3.61 | 7.46 | 2.52 | 1.34 | ✔ |
| P5 | +1.12 | 2.28 | 0.81 | 1.32 | ✔ |
| P6 | −1.46 | −2.95 | −2.51 | 1.56 | ✘ |
| P7 (holdout) | −0.98 | −1.97 | −1.04 | 1.84 | ✘ |

Holdout Tier 4: carry **+28.24 USDT**, overlay **−13.87 USDT** → cả hai nhánh không cùng dương ⇒ GO-2 fail.

---

## 4. Attribution (USDT, toàn chuỗi)

| Tier | Funding | Basis (thật) | Overlay (gross) | Phí | Net |
|---|---|---|---|---|---|
| 1 | 584.74 | 6.17 | — | 97.25 | 493.66 |
| 2 | 573.94 | 4.03 | — | 173.27 | 404.70 |
| 3 | 593.69 | 0.01 | — | 279.27 | 314.43 |
| 4 | 593.69 | 0.01 | 92.94 | 287.31 | 408.21 |

`basis_pnl` của Tier 3 gần 0 vì Mode Perp-Perp dùng basis Spot-Perp Binance (xem phát hiện #5). `basis_mtm_std` = 0.43 USDT/giờ, đủ điều kiện GO-4 (`> 0`).

---

## 5. Checklist GO/NO-GO

| Tiêu chí | Ngưỡng | Tier 3 | Tier 4 |
|---|---|---|---|
| GO-1 Lợi nhuận | ≥ 10.0%/năm | 2.93% ✘ | 3.77% ✘ |
| GO-2 Ổn định | ≥ 6/7, holdout ≥ 8%, kỳ xấu ≥ −3%, cả 2 nhánh dương | 5/7, −1.02% ✘ | 5/7, −1.97%, overlay âm ✘ |
| GO-3 Rủi ro | Sharpe ≥ 1.80, MaxDD ≤ 8%, \|β\| ≤ 0.15, 0 margin call | 2.16 / 2.40% / 0.0007 / 0 ✔ | 1.37 / 3.92% / 0.003 / 0 ✘ |
| GO-4 Gate & MTM | active ratio ∈ [25%, 75%], basis_mtm_std > 0 | 0.433 ✔ | 34.6% + 0.433 ✔ |
| GO-5 Executor | chạy paper barbell: lot-size, min_notional, cancel, 2 sleeve | static ✔ / **functional ✘** | static ✔ / **functional ✘** |

---

## 6. Việc đã làm và kiểm chứng

### Chặng 1 — `src/cross_exchange_carry.py`
`EXCHANGE_LOT_SPECS` (BTC 0.001, ETH 0.01, SOL 0.1, BNB 0.01), `quantize_order_qty`, min notional 10 USDT; entry 0.50 bps/8h, exit −0.10 bps/8h, payback 42 kỳ, EMA 21; **True Basis MTM từng giờ** (`-notional_start × Δbasis`), `basis_mtm_std` xuất ra ngoài.

### Chặng 2 — `src/barbell_portfolio.py`
`compute_snr_dispersion` (t-stat trend 14 ngày, không chuẩn hóa ngang), gate hysteresis 1.35/0.95, vol targeting 6.5%/năm với clip [0.10, 0.45], drawdown cushion 4%/6%. `overlay_active_ratio` = 34.6%.

### Chặng 3 — `src/carry_live_executor.py`, `run_live.py`, `src/state_store.py`
`quantize_leg`, làm tròn `filled_qty` trước khi hedge, `venues` đa sàn (binance + bybit), `build_barbell_allocations`, `persist_sleeve_state`, `carry_sleeve` + `momentum_overlay` trong `bot_state.json`, giữ `cancel_maker_leg1`.

### Chặng 4 — `src/sprint10_eval.py`, `run_sprint10.py`
Harness 4 Tier + 5 tiêu chí GO, `compute_period_contributions`, `check_executor_parity` (tĩnh), xuất `data/sprint10_{tiers_summary,periods,attribution}.csv` + `sprint10_report_tables.md`.

### Kiểm chứng thực tế
```
python run_sprint10.py --capital 3000            # 4 Tier + verdicts
python run_live.py --mode paper --strategy barbell --capital 3000 --once
   -> Venues: ['binance','binance_spot','bybit'] | Lot quantization: ON
   -> state.carry_sleeve: 2 allocations (BTC 0.018 / ETH 0.62), quantized
   -> state.momentum_overlay: active=False, dispersion 0.512
```
259 test pass (2 lần liên tiếp). 18 test mới trong `tests/test_sprint10_production.py`.

---

## 7. Red Team Audit lần 2 (độc lập) — 9 phát hiện

| # | Mức | Vấn đề | Ảnh hưởng | Trạng thái |
|---|---|---|---|---|
| 1 | CRITICAL | Executor chỉ đặt lệnh Spot-Perp Binance; **không có lệnh thật cho Overlay** và không có Perp-Perp Bybit | GO-5 fail về bản chất | **MỞ** |
| 2 | CRITICAL | `check_executor_parity()` chỉ kiểm tra chuỗi ký tự trong source → PASS giả | Che giấu #1 | **ĐÃ VÁ một phần**: thêm cờ `executor_functional=False` mặc định → cond5 FAIL trung thực |
| 3 | CRITICAL | Backtest Overlay **không lượng hóa lot-size**; ở leverage 5–15% lệnh $75–220/chân, BTC làm tròn ~7–9% | PnL overlay lý thuyết ≠ thực thi | **MỞ** |
| 4 | HIGH | Post-Only bị **hủy ngay trong cùng chu kỳ** (0 ms) → live không bao giờ khớp maker | Bot tê liệt ở live | **MỞ** (cần TTL 1–2 nến) |
| 5 | HIGH | Basis MTM của Mode Perp-Perp đang dùng basis Spot-Perp Binance | Tier 3 basis ≈ 0, bỏ sót rủi ro chéo sàn | **MỞ** |
| 6 | HIGH | Funding PnL tính trên notional entry, không phải mark value | Sai 5–15% dòng funding khi giá chạy 20% | **MỞ** |
| 7 | MEDIUM | Gate z-score cũ vẫn tồn tại trong `cross_sectional.py` | Technical debt, gọi trực tiếp sẽ hỏng | **MỞ** |
| 8 | MEDIUM | DD cushion dùng NAV tổng → carry lỗ kéo overlay tắt oan; ngưỡng 4%/6% quá nhạy | Overlay bị bóp nghẹt | **MỞ** |
| 9 | LOW | `EXCHANGE_LOT_SPECS` không phân biệt spot/perp | Spot hedge làm tròn thô | **MỞ** |

**Trạng thái 4 lỗi Sprint 9:**
1. Regime gate hằng-đúng → **ĐÓNG** ở tầng Barbell (34.6% active); còn debt ở `cross_sectional.py`.
2. Ngưỡng carry + basis = 0 → **ĐÓNG** (2.93% và Sharpe 2.16 là số thật).
3. Lot-size → **ĐÓNG MỘT PHẦN** (carry + executor có; overlay backtest chưa).
4. Executor Barbell/cross-exchange → **CHƯA ĐÓNG** (chỉ có phân bổ + state, chưa có lệnh).

---

## 8. Giới hạn dữ liệu

- Binance/Bybit: funding 8h đầy đủ 2023-05-07 → 2026-10-18.
- **OKX: chỉ ~3 tháng gần nhất** (296 kỳ từ 2026-07-13) do giới hạn API; Tier 2/3 thực chất là Binance vs Bybit trong 3.2 năm đầu.
- Giá perp/spot: cache 1h cục bộ, không có order book depth → mô hình maker fill vẫn là xấp xỉ.

---

## 9. Khuyến nghị (Sprint 11)

1. **Dùng Tier 1 làm Core Sleeve** thay Tier 3: 4.51%/năm, Sharpe 4.89, DD 0.64%, phí 97 USDT — nền tốt hơn hẳn.
2. **Mở rộng rổ carry sang 15–20 altcoin thanh khoản** (SUI, NEAR, AVAX, LINK, APT…) nơi spread funding 15–30%/năm; Core-4 không thể đạt 10%.
3. **Tái chuẩn overlay**: `min_snr_dispersion` 1.35 → 1.15, `target_vol_ann` 6.5% → 10%, tách DD cushion theo NAV riêng của overlay.
4. **Hoàn thiện executor thật**: TTL Post-Only 1–2 nến (không hủy 0 ms), đặt lệnh Overlay thật, Perp-Perp Bybit, basis chéo sàn, funding theo mark value.
5. **Lượng hóa lot-size trong backtest overlay** trước khi tin bất kỳ con số overlay nào ở vốn 3,000 USDT.

**Không chạy tiền thật.** Ngay cả Tier 1 (tốt nhất) cũng chỉ 4.51%/năm — dưới GO-1 — và live executor chưa đạt parity chức năng.

---

## 10. Phụ lục — cấu hình

- `BARBELL`: lookback 336h, rebalance 48h, top_k=1, buffer 1, target vol 6.5%/năm, leverage [0.10, 0.45], gate 1.35/0.95, DD cut 4%/off 6%.
- `CARRY`: entry 0.50 bps/8h, exit −0.10 bps/8h, payback 42 kỳ, min hold 42 kỳ, EMA 21, top_k=2, rebalance ±25% (5 bps), maker TTL 2 nến.
- LOT: BTC 0.001, ETH 0.01, SOL 0.1, BNB 0.01; min notional 10 USDT.
- Annualized `(final/initial)^(365.25/days)−1`; Sharpe mean/std 1h × √8760; MaxDD dương.
