# Sprint 5 — Deep Analysis: tiến độ & quyết định (ĐỌC FILE NÀY TRƯỚC KHI LÀM TIẾP)

> File này tồn tại để chống mất ngữ cảnh khi agent bị nén/chuyển session.
> **Quy ước:** cập nhật + commit sau MỖI bước. Kết quả phải nằm trong `data/*.csv` và `docs/*.md`, KHÔNG chỉ trong chat.
> Nhánh: `agy/sprint5-deep-analysis` (tạo từ `master` — master đã merge Sprint 3; LƯU Ý: master KHÔNG có code Sprint 4 diagnostics).

## 0. Mục tiêu Đợt 8
Xóa các điểm mù của Đợt 7 và kiểm định khung thời gian lớn hơn với tiêu chí khóa trước:
- **Phần A (chẩn đoán baseline 1h):** exposure (time-in-market, notional/equity, tỷ lệ long/short, holding time); drawdown baseline vs buy-and-hold; **alpha trừ beta** (hồi quy hourly return lên B&H, sai số chuẩn Newey-West, chạy toàn bộ ~3.45 năm); **block bootstrap theo ngày** (CI95% cho chênh lệch Sharpe baseline vs B&H); **null khớp exposure** (vào lệnh ngẫu nhiên cùng tần suất/holding/long-short, 1000 lần).
- **Phần B (4h & 1d):** resample từ 1h; walk-forward co giãn theo khung (1h/4h: 120/60/60 ngày; 1d: 360/120/120 ngày); lưới cố định 8 tổ hợp; **holdout 180 ngày cuối** chạy DUY NHẤT 1 lần ở cuối.
- **Phần C (tùy chọn, ưu tiên thấp):** mô hình funding. Chỉ làm sau A & B.

## 1. Tiêu chí đã khóa
- Phần B go/no-go: xem `docs/go_no_go_sprint5.md` (chốt & commit TRƯỚC khi có kết quả B).
- Nguyên tắc bất biến: **KHÔNG sửa chiến lược, KHÔNG sửa tiêu chí sau khi thấy kết quả, KHÔNG chạy lại với lưới mới.**

## 2. Dữ liệu nền
- 1h, 4 symbol (BTC/ETH/SOL/BNB), ~3.45 năm (1260 ngày) đã backfill ở Đợt 6 → cache `data/raw/*.parquet`.
- 4h/1d resample từ 1h (không tải lại). Kiểm tra lại `candles_per_year_for_timeframe` cho từng khung.
- Baseline cố định = `Candidate("breakout","market",use_trailing_stop=True)`, `base = TradingConfig(use_fee_aware_sizing=True, min_stop_distance_pct=0.008)`, kịch bản phí `zero`/`real`.

## 3. Trạng thái các bước
| Bước | Nội dung | Trạng thái | Commit | Ghi chú |
|---|---|---|---|---|
| S0 | Tạo nhánh + progress doc | DONE | (xem log) | bước này |
| S1 | `docs/go_no_go_sprint5.md` (khóa tiêu chí Phần B) | DONE | | chốt trước kết quả |
| A1 | Bảng exposure baseline 1h | IN_PROGRESS | | `src/deep_analysis.py` |
| A2 | Drawdown baseline vs B&H | TODO | | |
| A3 | Alpha/beta Newey-West (toàn 3.45y) | TODO | | |
| A4 | Block bootstrap theo ngày (CI95 Sharpe diff) | TODO | | |
| A5 | Null khớp exposure (1000 lần) | TODO | | |
| B1 | Resample 4h/1d + kiểm candles_per_year | TODO | | |
| B2 | Walk-forward 4h & 1d (dev, KHÔNG holdout) | TODO | | |
| B3 | Chạy holdout 1 lần duy nhất | TODO | | |
| C | Funding model (tùy chọn) | TODO | | sau A&B |
| R | Báo cáo GO/NO-GO | TODO | | |

## 4. Quyết định đã đưa ra & lý do
- **Branch từ master** (theo chỉ thị). Vì master thiếu code Sprint 4, Phần A tự cài đặt metric (không phụ thuộc `src/diagnostics.py`).
- **Alpha/beta chạy trên TOÀN BỘ 3.45 năm** (không từng cửa sổ) để có đủ mẫu cho t-stat Newey-West — theo chỉ thị.
- **Holdout**: 180 ngày cuối bị cô lập khỏi mọi bước dev/chọn; chỉ chạy 1 lần cuối. Ghi lại để tránh đụng holdout sớm.
- **Đếm số lần thử** (timeframe × config × symbol) để bù multiple-testing; t-stat ngưỡng nâng lên 2.5 (đã khóa trong go_no_go_sprint5.md).

## 5. Cảnh báo cho session sau
- Nếu phải viết lại: bắt đầu từ `docs/go_no_go_sprint5.md` + file này.
- Rule git: KHÔNG đụng master, KHÔNG merge/push/reset --hard/branch -D.
- Python 3.14, pandas 3.0. PowerShell dùng `;` không `&&`. `PYTHONIOENCODING=utf-8` khi in tiếng Việt.
