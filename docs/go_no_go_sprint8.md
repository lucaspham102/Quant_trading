# Tiêu chí GO / NO-GO — Sprint 8 (Dynamic 12-Asset Carry Rotation + Spot-Perp Basis Convergence) — KHÓA TRƯỚC KHI CODE

> Chốt & commit **TRƯỚC khi có bất kỳ kết quả nào của Sprint 8**.
> Sau khi có kết quả: **KHÔNG sửa lưới, KHÔNG sửa tiêu chí, KHÔNG chạy lại với lưới mới.**

## 0. Bối cảnh & mục tiêu kinh tế

Sprint 7 để lại **True MTM Carry** là chiến lược duy nhất có Sharpe thực rất cao
(5.86 DEV / 3.39 Holdout, MaxDD < 0.3%), nhưng **Annualized Net Return chỉ 4.29% (DEV)**
và **0.96% (Holdout)** vì hai lý do cấu trúc:

1. **Vốn bị chia tĩnh đều cho 4 đồng coin lớn** (25% NAV mỗi coin) kể cả khi funding bằng 0
   ⇒ phần lớn thời gian vốn nằm ở tài sản có carry thấp.
2. **Chưa khai thác chênh lệch giá Spot–Perp (Basis Convergence)** — chỉ thu funding,
   không thu phần basis co về trung bình.

Sprint 8 phải chứng minh được **lợi suất thực sự đủ lớn để đáng vận hành thực tế**,
không chỉ đẹp về Sharpe.

## 1. Phạm vi dữ liệu

- **Rổ 4 core**: `BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`.
- **Rổ 8 mở rộng**: `XRP`, `DOGE`, `ADA`, `AVAX`, `LINK`, `DOT`, `LTC`, `TRX` (USDT-M perps).
- **Rổ 12**: `UNIVERSE_12_SYMBOLS` = 4 core + 8 mở rộng.
- **Ba luồng dữ liệu thật 100% cho CẢ 12 mã**:
  - Perp OHLCV `1h`
  - Spot OHLCV `1h`
  - Funding Rate `8h`
  - Ghép qua `align_spot_perp_funding` (inner join timestamp, chống look-ahead).
- Khoảng thời gian: `2023-05-07` → `2026-10-18` (**3.45 năm**).

## 2. Kiểm định Đa Giai Đoạn (7 cửa sổ 6 tháng liên tiếp)

- Chia chuỗi thành **7 giai đoạn 180 ngày nối tiếp nhau**, phủ trọn `2023-05-07` → `2026-10-18`.
- Chạy trên **một đường cong vốn nối tiếp (Stitched Equity Curve)** duy nhất:
  không reset vốn, không đóng cưỡng bức vị thế ở biên giai đoạn.
- Báo cáo **cho từng giai đoạn** và **cho toàn bộ 3.45 năm**:
  Return %, Annualized Return %, Sharpe MTM, MaxDD MTM, Turnover,
  Tổng Funding thu được, Tổng Basis PnL, Tổng Phí đã trả.
- Đối chiếu trên **3 cấu hình độc lập**:
  1. Dynamic Carry + Basis trên **rổ 12 coin**.
  2. Dynamic Carry + Basis trên **rổ 8 coin mở rộng độc lập** (Out-of-Universe).
  3. **Static Carry 4 coin của Sprint 7** (baseline so sánh).

## 3. Điều kiện GO bắt buộc

Phải thỏa **ĐỒNG THỜI cả 4** điều kiện sau:

1. **Lợi suất ròng năm hóa (Annualized Net Return)**
   - `≥ 8.0%/năm` trên **toàn chuỗi 3.45 năm** với **rổ 12 coin**
     (gấp đôi mức 4.29% của Sprint 7), **VÀ**
   - `≥ 6.0%/năm` trên **rổ 8 coin mở rộng**.
   - Đã trừ **phí Taker/Maker thực tế** và tính **MTM thực tế**.

2. **Tính ổn định qua các chế độ thị trường (Sub-period Consistency)**
   - Net Return **> 0%** ở **ít nhất 6/7 giai đoạn 6 tháng**.

3. **Chất lượng rủi ro**
   - **Sharpe MTM thực tế ≥ 2.00**
   - **DSR ≥ 0.95**
   - **Max Drawdown MTM thực tế ≤ 5.0%** tại mọi thời điểm.

4. **Sẵn sàng vận hành (Production Readiness)**
   - Module `src/carry_live_executor.py` + `run_live.py --strategy dynamic_carry` chạy thành công
     **chu kỳ đối soát danh mục Delta-Neutral trên dữ liệu thực tế từ sàn**
     (paper mode, `--once`), in bảng trạng thái danh mục và ghi state nguyên tử.

**GO ⇔ thỏa cả 1, 2, 3, 4.** Trượt bất kỳ điều kiện nào → **NO-GO**.

## 4. Đếm số lần thử (multiple testing) — khóa `K`

- **Dynamic Carry**: `top_k ∈ {2, 3}` × `entry_threshold ∈ {0.00010}` ×
  `ema_span ∈ {6}` × `w_basis ∈ {0.0, 0.5, 1.0}` ×
  `execution ∈ {real_taker, passive_limit_maker}` = **K = 12**.
- DSR tính với `K = 12` (số cấu hình thực sự đã thử trước khi chọn).
- Baseline Static Carry Sprint 7 **không tính vào K** (đã công bố ở Sprint 7); chỉ dùng
  để so sánh.

## 5. Lưới tham số đóng băng (không mở rộng sau khi thấy kết quả)

- `ema_span` (EWMA funding) = `6` kỳ (48h).
- `expected_hold_periods` = `21` kỳ (7 ngày) cho quy đổi Basis về lợi suất mỗi 8h.
- `basis_lookback_hours` = `168` (7 ngày) cho EMA của `basis_pct`.
- `entry_threshold`:

  | Tham số | Giá trị |
  | --- | --- |
  | `entry_threshold` (θ_entry) | `0.00010` (1.0 bps/8h) |
  | `exit_threshold` (θ_exit) | `0.00002` (0.2 bps/8h) |
  | `min_hold_periods` | `9` kỳ (3 ngày) |
  | `top_k` | `2` hoặc `3` |
  | `buffer_rank` | `1` |
  | `min_switch_advantage` | `2 × cost_round_trip / expected_hold_periods` |

- **Cấu trúc vốn**: `coinm_1leg` (η = 1.0) và `spot_perp_3x` (η = 0.75:
  75% vốn mua Spot, 25% làm ký quỹ Short Perp đòn bẩy 3x, tự rebalance khi lệch tỷ lệ ký quỹ).
- **Mô hình phí khóa cứng**:
  - `real_taker`: Spot **7.5 bps** + Perp **5 bps** + Slippage **2 bps** mỗi chân.
  - `passive_limit_maker`: Perp Maker **2 bps** + Spot Maker **5 bps**; mô hình xác suất khớp
    chỉ khớp khi Basis nới rộng qua mức đặt lệnh ở nến kế tiếp.

## 6. Định nghĩa kỹ thuật khóa

### 6.1 Điểm Kỳ Vọng Carry + Basis

Với tài sản `i` tại thời điểm `t` (mốc thanh toán funding, 8h/lần), **chỉ dùng dữ liệu đã đóng ở ≤ t−1**:

```
F_{i,t} = EWMA(funding_rate_i, span=ema_span)_{t-1}

B_{i,t} = basis_pct_{i,t-1} - EMA_168(basis_pct_i)_{t-1}
        với basis_pct = (perp_close - spot_close) / spot_close

Score_{i,t} = F_{i,t} + w_basis * B_{i,t} / expected_hold_periods
```

- `B_{i,t} > 0` nghĩa là Perp đang giao dịch cao hơn bình thường so với Spot
  ⇒ Short Perp / Long Spot tại `t` kỳ vọng thu thêm lợi nhuận khi Basis co về trung bình.

### 6.2 Quy tắc chọn danh mục tập trung (Buffer + Hysteresis)

1. Chỉ xét coin có `Score_{i,t} ≥ θ_entry`.
2. Chọn tối đa `top_k` coin điểm cao nhất; mỗi coin nhận `1/top_k` tổng vốn.
3. Coin **đang nắm giữ từ kỳ trước được GIỮ NGUYÊN** (không đóng) chừng nào:
   - chưa đủ `min_hold_periods`, **HOẶC**
   - `Score ≥ θ_exit` **VÀ** hạng `≤ top_k + buffer_rank`,
   - **VÀ** chênh lệch điểm so với coin mới tốt nhất chưa vượt `min_switch_advantage`
     (chỉ đổi coin khi coin mới trả nhiều hơn hẳn phần phí khứ hồi).

### 6.3 Định giá True Bar-by-Bar Spot-Perp MTM

- `spot_perp_3x` (2 chân):
  `ΔMTM_t = Q·(spot_t − spot_{t−1}) − Q·(perp_t − perp_{t−1}) + FundingPnL_t − Fees_t`,
  với `Q = η·C_i / spot_{t0}`, `η = 0.75`, tính ở **mọi nến đang giữ vị thế**.
- `coinm_1leg`:
  `ΔMTM_t = −Notional_{t−1}·(basis_pct_t − basis_pct_{t−1}) + FundingPnL_t − Fees_t`.
- `FundingPnL_t = Notional_{t−1} · funding_rate_event_t` tại các nến 00/08/16 UTC.
- Phí trừ tại thời điểm mở/đóng/rebalance, không dồn cuối kỳ.

### 6.4 Anti look-ahead

- Xếp hạng tại `t` chỉ dùng dữ liệu `≤ t−1`.
- Funding chỉ dùng giá trị **đã chốt** (`merge_asof` backward).
- MTM chỉ dùng giá **thật** tại `t` và `t−1`; cấm ffill giá sang nến thiếu dữ liệu để tính MTM.
- Spot và Perp khớp chính xác theo timestamp UTC (inner join).

## 7. Phân rã nguồn lợi nhuận (PnL Attribution)

Mọi báo cáo phải tách bạch:

- **Funding PnL** (tổng funding thu được),
- **Basis Convergence PnL** (lãi/lỗ MTM giá Spot vs Perp),
- **Fee Drag** (tổng phí đã trả),
- **Net PnL** = Funding + Basis − Fees.

## 8. Kết luận

- Nếu GO: trình bày bằng chứng từng điều kiện + số liệu + cảnh báo multiple-testing & hạn chế
  (thanh khoản, trượt giá thực tế, rủi ro sàn, rủi ro depeg stablecoin).
- Nếu NO-GO: nói thẳng, nêu điều kiện trượt + số liệu; **KHÔNG chạy lại với lưới mới**.
