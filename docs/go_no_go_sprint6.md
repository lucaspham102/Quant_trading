# Tiêu chí GO / NO-GO — Sprint 6 (Funding Carry & Cross-Sectional) — KHÓA TRƯỚC KHI CODE

> Chốt & commit **TRƯỚC khi có bất kỳ kết quả nào**. Sau khi có kết quả:
> **KHÔNG sửa lưới, KHÔNG sửa tiêu chí, KHÔNG chạy lại với lưới mới.**

## 0. Phạm vi
- Symbol: BTC/ETH/SOL/BNB (USDT-M / Coin-M futures).
- Hai hướng nghiên cứu độc lập, mỗi hướng tự đánh giá GO/NO-GO:
  1. **Funding Carry (delta-neutral)**: `coinm_1leg`, `spot_perp_2leg`.
  2. **Cross-Sectional Market-Neutral**: `momentum`, `reversal`.
- Dữ liệu giá: OHLCV 1h/4h (cache sẵn, 2023-05-07 → 2026-10-18).
- Dữ liệu funding: Funding Rate 8h thật từ Binance (cùng khoảng).

## 1. Cửa sổ dữ liệu & Holdout
- **Tập Phát triển / Walk-forward**: `2023-05-07` → `2026-04-21` (~3 năm).
- **Tập Holdout**: `2026-04-21` → `2026-10-18` (180 ngày). Bị loại khỏi MỌI bước
  phát triển/chọn config; **chỉ chạy ĐÚNG 1 LẦN** sau khi toàn bộ code hoàn tất.

## 2. Sửa lỗi phương pháp từ Sprint 4–5
- **TUYỆT ĐỐI KHÔNG** đóng cưỡng bức vị thế hay reset vốn ở biên mỗi cửa sổ 60 ngày.
- Phải nối liền thành **một đường cong vốn liên tục (Stitched OOS Equity Curve)**,
  khởi đầu **400 USDT**: vị thế/ vốn cuối cửa sổ trước là trạng thái đầu cửa sổ sau;
  không zeroing, không flat-forced.
- Sharpe/Return/Drawdown của toàn bộ OOS được tính trên **đường cong nối liền** này,
  không phải trung bình của các cửa sổ rời rạc.

## 3. Điều kiện GO bắt buộc (kịch bản phí `real`) — cho MỖI hướng
Phải thỏa **ĐỒNG THỜI cả 4** điều kiện mới GO:

1. **Market-neutral thật sự**: Newey-West Alpha t-stat (hồi quy lợi nhuận chiến lược lên
   BTC Buy-and-Hold) **≥ 2.50** **VÀ** |β_BTC| **≤ 0.20**.
2. **Sharpe năm hóa sau phí + trượt giá (+ funding)**:
   - **Carry: ≥ 1.50**
   - **Cross-Sectional: ≥ 1.00**
3. **Deflated Sharpe Ratio (DSR) ≥ 0.95** (đã phạt skewness, kurtosis và tổng số cấu hình
   thử nghiệm `K` — Bailey & López de Prado).
4. **Tập Holdout 180 ngày** (kịch bản `real`): Net Return **> 0%**, Sharpe **> 0.80**,
   và Max Drawdown **≤ 12%** (Carry) hoặc **≤ 20%** (Cross-Sectional).

**GO ⇔ thỏa cả 1,2,3,4.** Chỉ cần một điều kiện không đạt → **NO-GO**.

## 4. Đếm số lần thử (multiple testing)
- Báo cáo `K` = tổng số cấu hình đã thử cho mỗi hướng (đưa vào công thức DSR).
- Lưới Carry: {coinm_1leg, spot_perp_2leg} × 3 kịch bản phí (zero/maker_optimistic/real_taker).
- Lưới Cross-Sectional: {momentum, reversal} × lookback {72,168,336}h × rebalance {24,48}h.

## 5. Định nghĩa kỹ thuật khóa
- **Funding PnL**: chân Short Perp nhận funding khi `funding_rate > 0`, chân Long trả funding.
- **Friction**: Carry 1 chân = 7 bps/chiều (14 bps khứ hồi); 2 chân = 16.5 bps/chiều (33 bps khứ hồi).
  Cross-Sectional: phí chỉ trên phần thay đổi tỷ trọng `|w_t − w_{t−1}|`.
- **Anti look-ahead**: tại nến `t` chỉ dùng funding đã chốt ở `≤ t` (merge_asof backward,
  timestamp dịch chuyển); điểm số xếp hạng dùng dữ liệu tới `t−1`.
- **DSR**: `dsr_prob = Φ( (SR̂ − SR*)·√(T−1) / √(1 − γ̂3·SR̂ + ((γ̂4−1)/4)·SR̂²) )`,
  `SR* = √V[{SR_k}] · ( (1−γ)·Φ⁻¹(1−1/K) + γ·Φ⁻¹(1−1/(K·e)) )`, γ ≈ 0.5772156649.

## 6. Kết luận
- Nếu GO: trình bày bằng chứng từng điều kiện + số liệu + cảnh báo multiple-testing & hạn chế.
- Nếu NO-GO: nói thẳng, nêu điều kiện trượt + số liệu; **KHÔNG chạy lại với lưới mới**.
