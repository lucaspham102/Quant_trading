# Sprint 10 — GO / NO-GO Criteria (Production-Grade Barbell)

**Khóa ngày:** 2026-10-20
**Vốn:** 3,000.00 USDT
**Cơ chế phí:** `maker_taker_hedge` — Perp-Perp 10 bps một chiều / 20 bps khứ hồi; Spot-Perp 11.25 / 22.5 bps
**Bắt buộc bổ sung:** `lot_size_quantization = True`, `True Basis MTM` (bar-by-bar, không được gán 0), phí rebalance
**Chuỗi kiểm định:** 3.45 năm (2023-05-07 → 2026-10-18), chia 7 kỳ 6 tháng liên tiếp, compounding

> Tài liệu này KHÓA tiêu chí. Không sửa ngưỡng sau khi đã chạy kết quả.

---

## 5 Tiêu chí GO bắt buộc

### GO-1 — Annualized Net Return
Lợi nhuận ròng năm hóa **≥ 10.0%/năm** (≥ +300 USDT/năm trên 3,000 USDT), đã trừ:
phí `maker_taker_hedge`, funding 2 chân, True Basis MTM, phí rebalance, và sai số làm tròn lot-size.

### GO-2 — Multi-Period Consistency
- Net Return > 0 ở **≥ 6/7 kỳ** 6 tháng.
- Kỳ holdout cuối `2026-04-21 → 2026-10-18`: Annualized Return **≥ 8.0%/năm**.
- Trong kỳ holdout, **cả hai nhánh** (Carry Sleeve và Momentum Overlay) đều đóng góp PnL dương.

### GO-3 — Risk & Sharpe Control
- Sharpe Ratio MTM thực tế **≥ 1.80**
- Max Drawdown MTM thực tế **≤ 8.0%**
- |β_BTC| **≤ 0.15**
- 0 margin call

### GO-4 — True Regime Gating & True MTM Verification
- `overlay_active_ratio` (tỷ lệ thời gian lớp Momentum Overlay thực sự có vị thế) nằm trong **[25%, 75%]**.
  → Bác bỏ ngay nếu > 85% (lỗi "hằng đúng" của Z-Score ở Sprint 9).
- Chuỗi Carry bắt buộc có `basis_mtm_std > 0`.
  → Bác bỏ nếu `basis_mtm` bằng 0 (giả định lạc quan của Sprint 9).

### GO-5 — Production Executor Parity
`python run_live.py --mode paper --strategy barbell --capital 3000 --once` phải:
- Định lượng hóa khối lượng theo `LOT_STEP_SIZES` (BTC 0.001, ETH 0.01, SOL 0.1, BNB 0.01)
- Kiểm tra `min_notional >= 10.0` USDT trước khi đặt lệnh
- Quản lý lệnh Post-Only có `cancel_maker_leg1()` (không để lệnh mồ côi)
- Khớp trạng thái **2 nhánh tách biệt**: `carry_sleeve` + `momentum_overlay`

---

## Điều kiện NO-GO (bác bỏ ngay)
- Bất kỳ tiêu chí GO nào ở trên không đạt.
- `basis_mtm_std == 0` hoặc `overlay_active_ratio > 0.85` → coi như lỗi triển khai, không phải kết quả chiến lược.
- Test suite đỏ, hoặc kết quả không tái lập được bằng `python run_sprint10.py --capital 3000`.

---

## Ghi chú kiểm định
- Annualized: `(final/initial)^(365.25/n_days) − 1`
- Sharpe: mean/std lợi nhuận 1h × √8760
- `maxdd_pct` là số dương
- 7 kỳ nối tiếp: vốn đầu kỳ sau = vốn cuối kỳ trước
- So sánh đối chứng với Sprint 9 (`docs/sprint9_report.md`) trong báo cáo cuối
