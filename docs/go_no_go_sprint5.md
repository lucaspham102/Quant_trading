# Tiêu chí GO / NO-GO — Sprint 5 (4h & 1d) — KHÓA TRƯỚC KHI CHẠY

> Chốt & commit **TRƯỚC khi có bất kỳ kết quả Phần B nào**. Sau khi có kết quả:
> **KHÔNG sửa lưới, KHÔNG sửa tiêu chí, KHÔNG chạy lại với lưới mới.**

## 0. Phạm vi
- Symbol: BTC/ETH/SOL/BNB (USDT-M futures).
- Khung: **4h** và **1d**, resample từ dữ liệu 1h (không tải lại).
- Baseline & lưới: **cố định 8 tổ hợp** như Sprint 3 (`strategy_mode` ∈ {breakout, trend_pullback} × `entry_mode` ∈ {market, limit_pullback} × `use_trailing_stop` ∈ {false,true}); cố định `sl_atr_multiple=2.0`, `tp_atr_multiple=3.0`, `trailing_activation_r=0.75`, `use_fee_aware_sizing=True`, `min_stop_distance_pct=0.008`, `use_ml=False`. **Không thêm tham số.**
- Kịch bản phí: `zero`, `real`, `double` (như Sprint 3: real = taker 5bps, maker 2bps, slippage 2bps).

## 1. Chia dữ liệu & holdout
- **Holdout = 180 ngày CUỐI của dữ liệu.** Bị **loại khỏi MỌI bước phát triển và chọn config**. Chỉ được chạy **DUY NHẤT MỘT LẦN** ở cuối, sau khi đã khóa tiêu chí và chạy xong phần dev.
- Phần dev = toàn bộ dữ liệu TRỪ 180 ngày cuối.

## 2. Cửa sổ walk-forward co giãn theo khung
- **4h:** train 120 / test 60 / step 60 ngày (như Sprint 3).
- **1d:** train 360 / test 120 / step 120 ngày (~5–6 cửa sổ).
- **Lý do:** giữ đủ số nến cho warm-up chỉ báo và đủ số lệnh/cửa sổ ở khung lớn; 1d có quá ít nến nên cần cửa sổ dài hơn.
- **Bắt buộc báo cáo** số nến/cửa sổ, số cửa sổ, và **số lệnh mỗi cửa sổ** (đặc biệt 1d rất ít lệnh → độ tin cậy thấp, phải nêu).

## 3. candles_per_year
Với mỗi khung, `candles_per_year = candles_per_year_for_timeframe(tf)` (1h=8760, 4h=2190, 1d=365) và **phải kiểm tra lại** trước khi tính Sharpe; nếu sai thì Sharpe sai.

## 4. Điều kiện GO (phải thỏa ĐỒNG THỜI cả bốn)
1. **Alpha trừ beta (kịch bản `real`)** có **t-stat ≥ 2.5** ở **≥ 3/4 symbol** (t-stat Newey-West, hồi quy hourly return baseline lên buy-and-hold).
2. **Sharpe sau phí không thua buy-and-hold quá 0.1 VÀ drawdown tốt hơn rõ rệt**, HOẶC **alpha dương có ý nghĩa** (t-stat ≥ 2.5).
3. **Percentile so với null khớp exposure ≥ 95**.
4. **Dương trên holdout (180 ngày cuối) ở ≥ 3/4 symbol** (kịch bản `real`).

**GO ⇔ thỏa cả 1,2,3,4.** Chỉ cần một điều kiện không đạt → **NO-GO**.

## 5. Đếm số lần thử (multiple testing)
- Báo cáo **số tổ hợp = (số timeframe) × (số config) × (số symbol) = 2 × 8 × 4 = 64** (cho Phần B), cộng thêm dev+holdout.
- Ngưỡng t-stat đã nâng lên **2.5** (thay vì ~2.0) để bù cho việc thử nhiều tổ hợp.

## 6. Phần A (chẩn đoán, không phải điều kiện GO nhưng dùng để diễn giải)
- Exposure, drawdown so B&H, alpha/beta Newey-West (toàn bộ ~3.45y), block bootstrap theo ngày (CI95 Sharpe diff), null khớp exposure (1000 lần) → xem bảng trong báo cáo & `data/`.

## 7. Phần C (tùy chọn, ưu tiên thấp)
- Mô hình funding (nếu có API công khai), áp cho cả long & short. **Chỉ làm SAU khi xong Phần A và B.**

## 8. Kết luận
- Nếu GO: trình bày bằng chứng từng điều kiện + số liệu + cảnh báo multiple-testing & hạn chế.
- Nếu NO-GO: nói thẳng, nêu điều kiện trượt + số liệu; **KHÔNG chạy lại với lưới mới**.
