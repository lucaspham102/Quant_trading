# Tiêu chí GO / NO-GO — Sprint 7 (Cross-Sectional 12 Symbols + True Spot-Perp MTM Carry) — KHÓA TRƯỚC KHI CODE

> Chốt & commit **TRƯỚC khi có bất kỳ kết quả nào**.
> Sau khi có kết quả: **KHÔNG sửa lưới, KHÔNG sửa tiêu chí, KHÔNG chạy lại với lưới mới.**

## 0. Phạm vi

- **Rổ Core (Sprint 1–6)**: `BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`.
- **Rổ mới (Out-of-Universe, CHƯA TỪNG được nhìn thấy ở Sprint 1–6)**:
  `XRP`, `DOGE`, `ADA`, `AVAX`, `LINK`, `DOT`, `LTC`, `TRX` (USDT-M perps).
- **Rổ đầy đủ**: `UNIVERSE_12_SYMBOLS` = 4 core + 8 mới.
- Hai hướng đánh giá độc lập, mỗi hướng tự quyết GO / NO-GO:
  1. **True Spot-Perp MTM Carry** (`spot_perp_2leg`, `coinm_1leg`; chế độ `hysteresis` và `passive_long_term`).
  2. **Cross-Sectional Momentum** (Top-K / Bottom-K, buffer band, carry tilt).
- Dữ liệu giá: Perp OHLCV 1h + Spot OHLCV 1h + Funding Rate 8h thật từ Binance,
  cùng khoảng `2023-05-07` → `2026-10-18`.

## 1. Phân tách dữ liệu & Holdout

- **Tập Phát triển (DEV)**: `2023-05-07` → `2026-04-21` (~3 năm).
- **Tập Holdout**: `2026-04-21` → `2026-10-18` (180 ngày).
  - Loại khỏi MỌI bước phát triển/chọn config.
  - Chạy trên **cùng một đường cong vốn nối tiếp (Stitched Equity Curve)** với DEV
    (không reset vốn, không đóng cưỡng bức ở biên).
  - **Chỉ mở khóa ĐÚNG 1 LẦN**, ở Chặng 5, sau khi toàn bộ code & lưới DEV đã đóng băng.

## 2. Kiểm định chống nhìn trộm Holdout (Blind Sub-window Check)

Sprint 6 đã biết cấu hình `336h/48h` dương trên Holdout 4 coin ⇒ Holdout tổng không còn
hoàn toàn "mù". Vì vậy ngoài Holdout tổng 180 ngày, chiến lược **bắt buộc phải có lãi dương
trên cả 8 đồng coin mới** (`XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, TRX`) — rổ **Out-of-Universe**
chưa từng được nhìn thấy ở Sprint 1–6. Đây là bằng chứng chống overfit mạnh hơn Holdout tổng.

## 3. Điều kiện GO bắt buộc — Cross-Sectional

Kịch bản phí khóa cứng: **`real_taker` = Taker 5 bps + Slippage 2 bps mỗi chiều**.

Phải thỏa **ĐỒNG THỜI cả 3** điều kiện:

1. **DEV Newey-West Alpha t-stat** (so với BTC Buy-and-Hold) **≥ 2.20** trên rổ 12 coin
   **hoặc ≥ 2.00** trên rổ 8 coin mới độc lập, **VÀ** |β_BTC| **≤ 0.15**.
2. **DEV Sharpe năm hóa sau phí & funding ≥ 1.10** **VÀ** **DSR ≥ 0.90**.
3. **HOLDOUT 180 ngày — trên CẢ rổ 12 coin VÀ rổ 8 coin mới**:
   Net Return **> +8.0%**, Sharpe **≥ 1.20**, Max Drawdown **≤ 15.0%**.

**GO ⇔ thỏa cả 1, 2, 3.** Trượt bất kỳ điều kiện nào → **NO-GO**.

## 4. Điều kiện GO bắt buộc — True MTM Carry

Sau khi tính **biến động Basis từng giờ thực tế giữa Spot và Perp** (không còn giả định
đường vốn phẳng / basis noise giả lập):

1. **Max Drawdown thực tế (MTM) > 0%** (bác bỏ mô hình nếu MaxDD vẫn bằng 0%)
   **VÀ ≤ 10.0%**.
2. **Annualized Net Return (DEV & HOLDOUT) ≥ +6.0%/năm** (vượt lãi suất tiết kiệm USD phi rủi ro)
   **VÀ Sharpe MTM thực tế ≥ 1.50**.

**GO ⇔ thỏa cả 1, 2.** Trượt bất kỳ điều kiện nào → **NO-GO**.

## 5. Đếm số lần thử (multiple testing) — khóa `K`

- **Carry**: {`spot_perp_2leg`, `coinm_1leg`} × {`hysteresis`, `passive_long_term`} = **K = 4**.
- **Cross-Sectional**: {rổ 8 mới (`top_k=2`), rổ 12 (`top_k=3`)} × lookback {336h, 504h}
  × buffer {0, 2} × carry tilt {0.0, 0.25} = **K = 16**.
- Tổng số cấu hình đã thử dùng cho DSR: `K_carry = 4`, `K_xs = 16`.

## 6. Lưới tham số đóng băng (không mở rộng sau khi thấy kết quả)

- Lookback: `336h` (14 ngày), `504h` (21 ngày). Rebalance: `48h` cố định.
- `top_k`: `2` cho rổ 8 coin, `3` cho rổ 12 coin. `buffer_rank`: `0` hoặc `2`.
- `carry_tilt_weight`: `0.0` hoặc `0.25`. `gross_leverage`: `1.0`.
- Carry `passive_long_term`: entry khi EMA funding 21 kỳ (7 ngày) `> 0.00005`;
  exit khi EMA funding 42 kỳ (14 ngày) `< -0.00002`; `min_hold_periods = 21`.
- Carry `maintenance_margin_ratio = 0.05`; đệm ký quỹ 1/3 ở `spot_perp_2leg`;
  tái cân bằng ký quỹ trừ phí 2 chiều.

## 7. Định nghĩa kỹ thuật khóa

- **MTM từng nến (2 chân)**:
  `ΔMTM_t = Q·(spot_t − spot_{t−1}) − Q·(perp_t − perp_{t−1}) + FundingPnL_t − Fees_t`,
  với `Q = η·C_i / spot_{t0}`, `η = 2/3`, tính ở **MỌI nến đang giữ vị thế**.
- **MTM từng nến (1 chân)**:
  `ΔMTM_t = −Notional_{t−1}·(basis_pct_t − basis_pct_{t−1}) + FundingPnL_t − Fees_t`.
- **Basis**: `basis_pct_t = (perp_close_t − spot_close_t) / spot_close_t`.
- **Điểm xếp hạng Cross-Sectional** (chỉ dùng dữ liệu tới `t−1`):
  `M_{i,t} = (Close_{i,t−1}/Close_{i,t−1−L} − 1) / (Std(ret_{i,t−1−L..t−1}) + 1e−8)`;
  `Z^mom` chuẩn hóa ngang; `Score_{i,t} = sign_mode·Z^mom_{i,t} − carry_tilt_weight·Z^fund_{i,t}`.
- **Trọng số**: `+gross_leverage/(2·top_k)` cho Long, `−gross_leverage/(2·top_k)` cho Short,
  ràng buộc `Σ w_{i,t} = 0`.
- **Anti look-ahead**: ranking dùng dữ liệu `≤ t−1`; funding chỉ dùng giá trị đã chốt `≤ t`
  (`merge_asof` backward).
- **Fill/align**: Perp và Spot khớp chính xác theo `timestamp` UTC (inner join), không ffill
  giá sang nến không có dữ liệu thật khi tính MTM.

## 8. Kết luận

- Nếu GO: trình bày bằng chứng từng điều kiện + số liệu + cảnh báo multiple-testing & hạn chế.
- Nếu NO-GO: nói thẳng, nêu điều kiện trượt + số liệu; **KHÔNG chạy lại với lưới mới**.
