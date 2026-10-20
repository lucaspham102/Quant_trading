# Sprint 11 — GO / NO-GO Criteria (Single-Venue Tier 1+ Barbell)

**Khóa ngày:** 2026-10-20
**Vốn:** 3,000.00 USDT
**Sàn duy nhất:** Binance Unified (Spot + USDⓈ-M Perp)
**Cơ chế phí:** `maker_taker_hedge` (20 bps khứ hồi cho Perp-Perp; 22.5 bps khứ hồi cho Spot-Perp)
**Bắt buộc bổ sung:** `True Basis MTM` từng giờ (bar-by-bar, không được gán 0), `lot_size_quantization = True` trên **CẢ Carry lẫn Overlay**
**Chuỗi kiểm định:** 3.45 năm (2023-05-07 → 2026-10-18), chia 7 kỳ 6 tháng liên tiếp, compounding

> Tài liệu này KHÓA tiêu chí. Không sửa ngưỡng sau khi đã chạy kết quả.
> Bối cảnh Sprint 10: Tier 1 (Single-Exchange Binance Long-Horizon Carry) đạt 4.51%/năm,
> Sharpe 4.89, MaxDD 0.64%, thắng 7/7 kỳ, Holdout +1.86%; Tier 3 (đa sàn) chỉ 2.93%/năm và
> -1.02% Holdout. Sprint 11 đặt Barbell lên nền Tier 1+ và thay cổng nhị phân SNR bằng
> điều tiết mượt (smooth conviction).

---

## 5 Tiêu chí GO bắt buộc

### GO-1 — Annualized Net Return
Lợi nhuận ròng năm hóa **≥ 9.0%/năm** (mục tiêu chuẩn ≥ 10.0%/năm), đã trừ:
phí `maker_taker_hedge`, funding, True Basis MTM, phí rebalance và sai số làm tròn
lot-size (**áp dụng cho cả nhánh Carry và nhánh Momentum Overlay**).

### GO-2 — Multi-Period Consistency
- Net Return > 0 ở **ít nhất 6/7 kỳ** 6 tháng.
- Kỳ Holdout cuối `2026-04-21 → 2026-10-18` bắt buộc đạt **Net Return ≥ +3.5%**
  (≥ 7.0%/năm), với **cả `carry_pnl > 0` và `overlay_net_pnl > 0`**.

### GO-3 — Risk & Sharpe Control
- **Sharpe Ratio MTM thực tế ≥ 1.80**
- **Max Drawdown MTM thực tế ≤ 7.5%**
- **|β_BTC| ≤ 0.15**
- 0 margin call

### GO-4 — 100% Accounting & Quantization Parity
- Cả nhánh Carry và nhánh Momentum Overlay trong Backtest đều phải áp dụng
  `quantize_order_qty` theo `EXCHANGE_LOT_SPECS`.
- `basis_mtm_std > 0` (True Basis MTM từng giờ, không được gán 0).
- Phương trình kế toán khớp tuyệt đối (sai số < 1e-6):
  `funding_pnl + basis_pnl + overlay_gross_pnl - total_fees == net_pnl`.

### GO-5 — Genuine Live Executor (`executor_functional == True`)
`CarryLiveExecutor` thực thi trên 1 sàn Binance duy nhất phải có:
- (a) Vòng lặp chờ khớp lệnh Post-Only có TTL (`maker_ttl_sec > 0`,
  `poll_interval_sec > 0`, xử lý `partial_fill` trước khi gọi `cancel_maker_leg1`).
- (b) Đặt lệnh thật cho cả `carry_sleeve` (Spot + Perp) lẫn `momentum_overlay`
  (Perp Long + Perp Short) qua giao diện exchange.
- Cờ `executor_functional = True` chỉ được bật khi cả `carry_orders` và `overlay_orders`
  đều được định tuyến thực tế.

---

## Điều kiện NO-GO (bác bỏ ngay)
- Bất kỳ tiêu chí GO nào ở trên không đạt.
- `basis_mtm_std == 0` hoặc phương trình kế toán lệch ≥ 1e-6 → lỗi triển khai.
- Test suite đỏ, hoặc kết quả không tái lập được bằng `python run_sprint11.py --capital 3000`.

---

## Ghi chú kiểm định
- Annualized: `(final/initial)^(365.25/n_days) − 1`
- Sharpe: mean/std lợi nhuận 1h × √8760
- `maxdd_pct` là số dương
- 7 kỳ nối tiếp: vốn đầu kỳ sau = vốn cuối kỳ trước
- Ma trận đối chứng 5 cấu hình: Tier 1A, Tier 1B, Barbell Ablation 1 (hard binary),
  Barbell Ablation 2 (smooth overlay trên Tier 1A), Barbell Production (Tier 1B + smooth overlay)
- So sánh đối chứng với Sprint 9 (`docs/sprint9_report.md`) và Sprint 10 (`docs/sprint10_report.md`)
