# Sprint 9 — GO / NO-GO Criteria Lock (Cross-Exchange Perp-Perp Carry + Maker-Taker Hedge + Core-4 Barbell)

**Ngày khóa:** 2026-10-20
**Nhánh:** `agy/sprint9-cross-exchange-barbell`
**Vốn vận hành mặc định:** `initial_capital = 3000.0` USDT (mở rộng được lên 5.000 USDT)
**Sàn giao dịch:** Binance (binanceusdm) + Bybit + OKX
**Rổ tài sản lõi:** `CORE_4_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]`
**Chuỗi đánh giá:** `2023-05-07` → `2026-10-18`, chia 7 giai đoạn 6 tháng liên tiếp (stitched equity curve).

> Mục đích của sprint này là **chạy tiền thật kiếm lời thực tế**. Mọi con số phải phản ánh chi phí thực thi thật, không dùng giả định lý tưởng hóa.

---

## 0. Cơ chế phí thực tế BẮT BUỘC — `maker_taker_hedge`

Mặc định của toàn bộ backtest Sprint 9 là `execution = "maker_taker_hedge"`.
**Cấm tuyệt đối** dùng giả định "2 chân Maker lý tưởng hóa" (ví dụ maker 2 bps cho cả 2 chân) làm kết quả chính.

| Thành phần | Cơ chế | Chi phí |
|---|---|---|
| **Chân 1 — Entry/Exit Leg 1** | Limit `Post-Only (GTX)` tại best bid/ask | Maker Perp fee **2 bps** (`0.0002`) + slippage **0 bps** + **Adverse Selection Penalty 1 bp** (`0.0001`) |
| **Chân 1 — rủi ro không khớp** | Nếu giá không đi qua mức treo, lệnh không khớp | Xác suất no-fill được mô hình hóa (không được coi là luôn khớp) |
| **Chân 2 — Hedge Leg 2** | Market Taker ngay khi Chân 1 khớp | Taker Perp fee **5 bps** (`0.0005`) + slippage **2 bps** (`0.0002`) |

**Tổng chi phí mỗi chiều (Perp-Perp):** `2 + 1 + 5 + 2 = 10 bps` → **khứ hồi = 20 bps**.
**Biến thể Spot-Perp (1 chân Spot Maker 5 bps + 1 chân Perp Taker 5 bps + slippage):** khứ hồi ≈ **22.5 bps**.
Mọi báo cáo phải nêu rõ chế độ phí đang dùng; `maker_taker_hedge` là chuẩn so sánh duy nhất khi ra phán quyết GO/NO-GO.

---

## 1. Bốn tiêu chí GO bắt buộc (chuỗi 3.45 năm, 7 giai đoạn 6 tháng, vốn gốc 3.000 USDT)

Tất cả đều tính **sau** phí `maker_taker_hedge`, funding **cả 2 chân**, và phí **rebalance ký quỹ liên sàn** (`rebalance_cost_bps = 5.0` bps mỗi sự kiện).

### GO-1 — Annualized Net Return
- **Annualized Net Return ≥ 10.0 %/năm** trên chuỗi stitched 3.45 năm.
- Tương đương **≥ +300 USDT/năm** trên vốn 3.000 USDT (≥ ~1.035 USDT/ngày trung bình).
- Đo trên `net_pnl` sau toàn bộ chi phí, không đo trên gross funding.

### GO-2 — Multi-Period Consistency
- Lợi nhuận ròng **dương ở ≥ 6/7 giai đoạn 6 tháng**.
- **Giai đoạn Holdout cuối** `2026-04-21` → `2026-10-18` **bắt buộc dương** và đạt **Annualized Return ≥ 8.0 %/năm**.
- Không được có giai đoạn âm nào vượt quá -3.0 % trên 6 tháng.

### GO-3 — Rủi ro MTM thực tế
- **Sharpe Ratio MTM (annualized, 1h bars) ≥ 1.80**
- **Max Drawdown MTM ≤ 8.0 %**
- **|β_BTC| ≤ 0.15** (delta-neutral thực sự, đo trên returns 1h của equity curve so với BTC)
- Không có giai đoạn nào bị liquidation / margin call (equity < maintenance margin).

### GO-4 — Đồng nhất Backtest ↔ Live Executor (Zero Execution Drift)
- `src/carry_live_executor.py` phải cài đúng máy trạng thái:
  `PLACE_MAKER_LEG_1 (Post-Only)` → `VERIFY_MAKER_FILL_AND_HEDGE_LEG_2 (Market Taker đúng filled_qty)`.
- **Cấm tuyệt đối** bắn 2 lệnh Market mù (lỗi CRITICAL #3 của Red Team Sprint 8).
- Không bao giờ hedge vượt quá `filled_qty` của Chân 1; partial fill → hedge partial; reject → không hedge.
- Tham số chi phí trong executor phải khớp 100% bảng phí ở mục 0.

---

## 2. Ba nguyên nhân gốc phải triệt tiêu (từ Red Team Sprint 8)

| # | Vấn đề Sprint 8 | Yêu cầu Sprint 9 |
|---|---|---|
| 1 | Taker cả 2 chân Spot+Perp = **33 bps** khứ hồi, ăn 50% funding | `maker_taker_hedge` = **20 bps** khứ hồi (Perp-Perp); đo lại toàn bộ hiệu quả |
| 2 | Chỉ Binance → trần funding ~3.2 %/năm, chân Spot không sinh lời | Đa sàn Binance+Bybit+OKX, chế độ Perp-vs-Perp với η = 1.00–1.25 |
| 3 | Live Executor bắn Market trong khi backtest chỉ lãi ở Maker | Máy trạng thái Post-Only → Market Hedge, test partial fill/reject |

---

## 3. Điều kiện NO-GO (bất kỳ điều nào đúng → NO-GO)

- Annualized Net Return < 10.0 %/năm trên chuỗi 3.45 năm.
- Holdout cuối ≤ 0 hoặc Annualized < 8.0 %/năm.
- Chỉ đạt được tiêu chí khi dùng phí lý tưởng hóa (2 chân Maker), tức kết quả `maker_taker_hedge` không đạt.
- Sharpe < 1.80 hoặc MaxDD > 8.0 % hoặc |β_BTC| > 0.15.
- Executor còn nhánh bắn Market không đối chiếu `filled_qty` của Chân 1.
- Có look-ahead bias trong `align_multi_exchange_funding` (dùng funding của chính cây nến t).
- Có 1 giai đoạn 6 tháng lỗ > 3.0 %.

---

## 4. Cách đo (định nghĩa kỹ thuật, tránh tranh cãi khi ra phán quyết)

- **Annualized Return:** `(final_equity / initial_capital) ** (365.25 / n_days) - 1`, `n_days` = độ dài chuỗi stitched.
- **Sharpe MTM:** mean/std của returns 1h của equity curve (MTM đánh giá lại mỗi giờ theo giá mark/close), annualize bằng `sqrt(24*365)`; risk-free = 0.
- **Max Drawdown MTM:** `min(equity / cummax(equity) - 1)` trên chuỗi MTM 1h, không dùng equity chỉ cập nhật lúc đóng lệnh.
- **β_BTC:** OLS slope của returns 1h equity vs returns 1h BTC (close-to-close), không lag.
- **Stitched equity:** mỗi giai đoạn 6 tháng chạy độc lập, nối tiếp bằng cách nhân chuỗi lợi suất (không reset về 3.000 ở mỗi kỳ).
- **Tier so sánh:** (1) Single-Exchange Binance Static Carry, (2) Multi-Exchange Best-Venue Carry, (3) Hybrid Cross-Exchange Perp-Perp Carry, (4) Full Barbell (Carry + Core-4 Buffered XS Momentum overlay ≤ 0.30–0.50). Tất cả Tier đều phải dùng cùng chuẩn phí `maker_taker_hedge`.

---

## 5. Ranh giới triển khai (không được vượt)

- Chỉ sửa/tạo file trong `src/`, `tests/`, `docs/`, `data/` và các script `run_*.py`.
- Không `git merge` / `git push` / `git reset --hard` / xóa dữ liệu `data/raw`.
- Mọi module mới phải có unit test; `pytest -q` phải xanh trước mỗi commit.
- Dữ liệu funding đa sàn phải là dữ liệu thật tải từ API, lưu cache Parquet, 0 trùng lặp, 0 NaN ở cột `funding_rate`.
