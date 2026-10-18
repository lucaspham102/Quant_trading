# Tiêu chí GO / NO-GO — Sprint 3 Alpha Engine (chốt TRƯỚC khi chạy)

> Tài liệu này được **chốt và commit TRƯỚC khi có bất kỳ kết quả walk-forward nào**
> của Đợt 6. Sau khi có kết quả, **KHÔNG được sửa lưới tham số, không được sửa tiêu chí**
> để chạy lại nhằm đạt GO. Mọi thay đổi lưới/tiêu chí phải là một Đợt mới, minh bạch.

## 0. Phạm vi & dữ liệu cố định
- Khung thời gian: **1h**.
- Symbol: **BTC/ETH/SOL/BNB** (USDT-M futures).
- Dữ liệu: ~**1260 ngày** (≈3.45 năm) qua cơ chế backfill cache, yêu cầu chất lượng:
  - 0 timestamp trùng, 0 khe hở nến (mọi khoảng cách = 1h), số nến ≈ 24 × số ngày.
- Walk-forward: **train 120 ngày / test 60 ngày / step 60 ngày** → **~18 cửa sổ/symbol**.
  - Cửa sổ test KHÔNG chồng lấn; train nằm ngay trước test (không chồng test).
  - Mỗi cửa sổ equity reset về 400 USDT; không mang vị thế qua biên cửa sổ.
- Lưới ứng viên CỐ ĐỊNH (8 tổ hợp, không tinh chỉnh):
  - `strategy_mode` ∈ {breakout, trend_pullback}
  - `entry_mode` ∈ {market, limit_pullback}
  - `use_trailing_stop` ∈ {false, true}
  - Cố định: `sl_atr_multiple=2.0`, `tp_atr_multiple=3.0`, `trailing_activation_r=0.75`,
    `use_fee_aware_sizing=True`, `min_stop_distance_pct=0.008`, `use_breakeven=True`,
    `max_holding_bars=48`, `use_ml=False`, không tinh chỉnh thêm tham số nào.
- Chọn ứng viên CHỈ trên TRAIN: điều kiện `total_trades ≥ 15` và `max_drawdown_pct ≤ 8.0`;
  chọn max Sharpe (tie-break MaxDD nhỏ hơn, rồi tên). Không ứng viên đạt → cửa sổ đứng ngoài
  (return = 0.0, dd = 0.0, sharpe = 0.0, trades = 0).
- Baseline cố định (để so sánh): breakout / market / trailing-on, chạy trên đúng các cửa sổ test.

## 1. Kịch bản phí & slippage (BƯỚC 4)
Chạy cùng một lưới & cùng cửa sổ với **3 kịch bản** (cột `fee_scenario`):
- `zero`: `taker_fee=0`, `maker_fee=0`, `slippage=0`.
- `real`: `taker_fee=0.0005`, `maker_fee=0.0002`, `slippage=0.0002` (giá trị mặc định hiện tại).
- `double`: gấp đôi `real` → `taker_fee=0.001`, `maker_fee=0.0004`, `slippage=0.0004`.

## 2. Điều kiện GO (phải thỏa ĐỒNG THỜI cả 4)
1. **Tính dương rộng (ở kịch bản `real`)**: có **≥ 60% cửa sổ test dương** ở **≥ 3/4 symbol**.
2. **Thắng baseline (ở kịch bản `real`)**: `mean test Sharpe` của ứng viên chọn-theo-train
   **> baseline** ở **≥ 3/4 symbol**.
3. **Vẫn dương với phí thực tế**: ở kịch bản `real` (phí + slippage thực tế), kết quả tổng hợp
   **vẫn dương** (mean test return > 0 trên đa số symbol, cụ thể ≥ 3/4 symbol có mean test return > 0).
   Phải kiểm tra thêm kịch bản `double` để nêu rủi ro chi phí (không bắt buộc dương, nhưng phải báo cáo).
4. **Thắng null test (BƯỚC 5)**: test return của cấu hình chọn-theo-train phải **nằm trên
   percentile 75** của phân phối null (chọn ngẫu nhiên cấu hình, 100 lần/cửa sổ).
   Nếu không → bước chọn trên train **không có giá trị**.

**GO ⇔ thỏa cả 1, 2, 3, 4.** Chỉ cần một điều kiện không đạt → **NO-GO**.

## 3. Cách diễn giải (quy ước, chốt trước)
- `% cửa sổ test dương` = 100 × (số cửa sổ test có `test_return_pct > 0`) / tổng số cửa sổ.
- `mean test Sharpe` = trung bình `test_sharpe_ratio` qua các cửa sổ (cửa sổ đứng ngoài tính là 0.0).
- Baseline so sánh **cùng kịch bản phí**.
- Null test: mỗi cửa sổ, lặp 100 lần chọn **ngẫu nhiên** một cấu hình trong lưới (seed cố định,
  minh bạch) rồi chạy trên **test frame** của cửa sổ đó → thu phân phối test return; lấy
  percentile 75 làm ngưỡng. So sánh test return của cấu hình chọn-theo-train với ngưỡng này.

## 4. Kết luận
- Nếu **GO**: chỉ ra bằng chứng từng điều kiện kèm số liệu; nêu rõ đây là kết quả in-sample
  của quy trình (walk-forward) nhưng vẫn cần kiểm toán look-ahead độc lập (checklist bên dưới) trước khi tin dùng.
- Nếu **NO-GO**: nói thẳng, liệt kê điều kiện trượt và số liệu; **không** tự ý đổi lưới/tiêu chí.

## 5. Checklist để Argon audit sau (khi quota hồi phục)
Kiểm tra `src/walkforward.py` + pipeline feature:
1. Tín hiệu tại nến `t` chỉ vào lệnh ở **open của nến `t+1`** (không dùng close/high/low của `t+1` để quyết định).
2. Feature không dùng **nến chưa đóng** (không shift âm, không rolling centered; nến cuối bị `drop_incomplete_last_candle`).
3. `drop_warmup` không cắt lệch giữa train và test (warm-up tính trên lịch sử đầy đủ, hai tập rời nhau).
4. Feature tính MỘT LẦN trên toàn lịch sử, truyền slice vào Backtester không tính lại (tránh mất warm-up ở mép & tránh lộ tương lai).
5. `candles_per_year` đúng theo timeframe khi tính Sharpe.
