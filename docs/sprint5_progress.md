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
| A1 | Bảng exposure baseline 1h | DONE | | `data/sprint5_exposure.csv` |
| A2 | Drawdown baseline vs B&H | DONE | | `data/sprint5_drawdown.csv` |
| A3 | Alpha/beta Newey-West (toàn 3.45y) | DONE | | `data/sprint5_alpha_beta.csv` |
| A4 | Block bootstrap theo ngày (CI95 Sharpe diff) | DONE | | `data/sprint5_bootstrap.csv` |
| A5 | Null khớp exposure (1000 lần) | DONE | | `data/sprint5_null.csv` |
| A6 | Báo cáo Phần A | DONE | | `docs/sprint5_results_partA.md` |
| B1 | Resample 4h/1d + kiểm candles_per_year | DONE | | 4h=2190, 1d=365 (dùng `candles_per_year_for_timeframe`) |
| B2 | Walk-forward 4h & 1d (dev, KHÔNG holdout) | DONE | | 4h 15 cửa sổ, 1d 4 cửa sổ |
| B3 | Chạy holdout 1 lần duy nhất | DONE | | 4h=1080 nến, 1d=180 nến |
| C | Funding model (tùy chọn) | TODO | | sau A&B (ưu tiên thấp) |
| R | Báo cáo GO/NO-GO | DONE | | **NO-GO** (xem §8) |

## 4. Quyết định đã đưa ra & lý do
- **Branch từ master** (theo chỉ thị). Vì master thiếu code Sprint 4, Phần A tự cài đặt metric (không phụ thuộc `src/diagnostics.py`).
- **Alpha/beta chạy trên TOÀN BỘ 3.45 năm** (không từng cửa sổ) để có đủ mẫu cho t-stat Newey-West — theo chỉ thị.
- **Holdout**: 180 ngày cuối bị cô lập khỏi mọi bước dev/chọn; chỉ chạy 1 lần cuối. Ghi lại để tránh đụng holdout sớm.
- **Đếm số lần thử** (timeframe × config × symbol) để bù multiple-testing; t-stat ngưỡng nâng lên 2.5 (đã khóa trong go_no_go_sprint5.md).

## 5. KẾT QUẢ PHẦN A (baseline 1h, 1260 ngày, 4 symbol) — đã chạy
Nguồn: `data/sprint5_*.csv` + `docs/sprint5_results_partA.md`.
- **Exposure**: time-in-market chỉ **~10.8–13.6%**; notional/equity trung bình **~0.43–0.70**; long_ratio **~0.48–0.56** (cân bằng); holding trung bình **~7.6–9.3 nến (giờ)**.
- **Drawdown (pooled)**: baseline zero **8.0–15.2%**, real **17.1–26.9%**, buy-and-hold **53.8–78.7%**. → baseline ít DD hơn nhiều (nhờ exposure thấp), nhưng per-window DD ~1–8% vs B&H 7–50%.
- **Alpha/beta (Newey-West, lags=14)**: alpha annualized BTC −3.1%, ETH −6.2%, SOL +1.0%, BNB −2.5%; **alpha_t ∈ [−1.13, +0.16] → đều KHÔNG có ý nghĩa (|<2|)**; beta ≈ **0.001–0.004** (gần như bằng 0 vì exposure rất thấp).
- **Block bootstrap (ngày, 1000 lần)**: Sharpe diff mean −0.87…−1.22; **CI95 đều chứa 0** (ci_high ~+0.30…+0.59); p(diff≤0) ≈ 0.88–0.95 → baseline Sharpe nghiêng về **kém hơn** B&H.
- **Null khớp exposure (1000 lần)**: percentile BTC 86.7, SOL 72.0, BNB 57.1, ETH 36.7 → **TẤT CẢ < 95**; metric per-trade return của baseline ~ null mean → **không có bằng chứng kỹ năng chọn thời điểm**.
- **Kết luận Phần A**: không có alpha có ý nghĩa; "lợi thế" chỉ là **exposure thấp → return thấp + DD thấp**. Đây là lý do phải so với B&H, và là cơ sở để thử khung lớn hơn (giảm nhiễu/turnover).

## 6. CHI TIẾT CẦN CHO PHẦN B (session sau)
- Holdout = **180 ngày cuối**. Dữ liệu kết thúc ~**2026-10-18 14:00 UTC** → holdout ≈ từ **2026-04-21**. Dev = 2023-05-01 → ~2026-04-21. **Chạy holdout 1 lần duy nhất ở cuối.**
- Resample 4h/1d từ 1h (`data/raw/*.parquet`). candles_per_year: 4h=2190, 1d=365 (kiểm lại).
- Windows: 4h = 120/60/60; 1d = 360/120/120.
- Lưới cố định 8 tổ hợp; baseline = breakout/market/trailing=True. Số lần thử Phần B = 2×8×4 = 64.

## 7. KẾT QUẢ PHẦN B (4h & 1d, dev 2023-05→2026-04; holdout 180 ngày cuối) — đã chạy
Nguồn: `data/sprint5_partB_*.csv` + `docs/sprint5_results_partB.md`. Code: `src/deep_analysis.py` (thêm `resample_ohlcv`, `sharpe_of`, `analyze_timeframe`, `holdout_return`), `run_sprint5_partB.py`, `tests/test_deep_analysis_partB.py` (5 test). Full suite **128 passed ×2**.
- **Cửa sổ**: 4h = 15; 1d = 4 (ít hơn dự kiến 5–6 vì warm-up ~200 nến bị drop). trade-day bootstrap 4h=1047, 1d=877.
- **Số lệnh/dev**: 4h ~70–78; **1d chỉ 3–10 (≈1 lệnh/cửa sổ)** → power rất thấp.
- **Alpha/beta (NW)**: 4h alpha **dương** BTC +2.17%/năm (t 0.84), SOL +3.65 (1.25), BNB +3.09 (1.04), ETH +0.57 (0.18); 1d nhỏ ±1% (|t|≤0.99). **Đều không ý nghĩa (<2.5).**
- **Sharpe (real) vs B&H**: 4h base 0.10–0.78 vs bh 0.47–1.14; 1d base −0.23…0.68 vs bh 0.42–1.05 → **đều thua** (base−bh: 4h −0.28…−0.51; 1d −0.21…−1.08).
- **Drawdown pooled (real)**: 4h 4.7–7.9% vs B&H 50–73%; 1d 1.2–2.2% vs B&H 50–70% → phòng vệ tốt nhưng đổi bằng return.
- **Null khớp exposure**: percentile 4h max 93.4 (SOL), 1d max 69.0 → **<95**.
- **Holdout (real)**: 4h 1/4 dương (SOL +0.20%); 1d 1/4 dương (SOL +1.72; BTC 0 lệnh).
- **Kết luận: NO-GO** (trượt cả 4 điều kiện). Khung lớn cải thiện Sharpe/alpha so với 1h nhưng vẫn dưới B&H & không ý nghĩa.
- **Bug đã sửa**: holdout ban đầu bị warm-up ăn hết (1d → 0 nến); đã đổi `holdout_return(symbol, tf, context_df, cutoff)` (tính chỉ báo trên toàn chuỗi rồi mới cắt từ cutoff).

## 8. Còn lại sau Đợt 8
- **Argon audit look-ahead** (checklist ở `docs/go_no_go_sprint3.md` §5) — chờ quota hồi phục.
- **Phần C (funding)** — tùy chọn, ưu tiên thấp.
- `src/deep_analysis.py` là bản sao độc lập metric (nhánh này thiếu `src/diagnostics.py` của Sprint 4).

## 9. Cảnh báo cho session sau
- Nếu phải viết lại: bắt đầu từ `docs/go_no_go_sprint5.md` + file này.
- Rule git: KHÔNG đụng master, KHÔNG merge/push/reset --hard/branch -D.
- Python 3.14, pandas 3.0. PowerShell dùng `;` không `&&`. `PYTHONIOENCODING=utf-8` khi in tiếng Việt.
