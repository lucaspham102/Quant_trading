# Sprint 12 — Production Readiness Gate (tiêu chí nghiệm thu đã khóa)

**Khóa ngày:** 2026-10-20, trước khi sửa bất kỳ dòng code Sprint 12 nào.
**Nhánh:** `agy/sprint12-production-hardening` (tách từ `master` tại `aeb55c8`)
**Nguyên tắc:** Sprint 12 **không** chỉnh tham số tín hiệu (entry/exit funding, top_k,
eta, smooth overlay). Mọi thay đổi thuộc hạ tầng vận hành. Tiêu chí dưới đây khóa cứng
trước khi chạy, không được hạ sau khi thấy kết quả.

---

## A. 6 cổng bắt buộc (PR-1 … PR-6)

| Mã | Tên cổng | Tiêu chí đạt | Cách kiểm chứng |
|---|---|---|---|
| **PR-1** | Capital parity | (a) State mới/reset: `bot_state.json` ghi `initial_capital == current_equity == --capital` (3.000,0). (b) State cũ còn `400.0` nhưng không có vị thế mở: tự đồng bộ lên `--capital`. (c) State có vị thế mở: **không** tự ghi đè, chỉ cảnh báo. (d) `--reset-state` xóa vị thế và đặt lại vốn. | Unit test + `run_live.py --capital 3000 --reset-state` đọc lại file JSON |
| **PR-2** | Active vs compounded capital | Báo cáo holdout xuất **cả hai**: `holdout_return_on_active_capital_pct = net_pnl / initial_active_capital` và `holdout_return_on_compounded_equity_pct`. Với Sprint 11: 3,75 % và 3,24 % (sai số < 0,01 pp). `reinvest_profits=True` phải làm thay đổi quy mô phân bổ (khác `False`), `False` phải tái lập đúng số Sprint 11. | Unit test trên payload giả + so khớp số đã lưu ở `docs/sprint11_evidence_tables.md` |
| **PR-3** | Spread guard | Khi `spread_bps > max_allowed_spread_bps` (mặc định 10,0 bps): **không** đặt lệnh chân 1, ghi lý do `SPREAD_TOO_WIDE`, chu kỳ vẫn hoàn tất (không crash). Khi spread ≤ 10 bps: lệnh vẫn được đặt như cũ. | Unit test với orderbook giả 2 mức spread |
| **PR-4** | Delta drift repair | Đầu mỗi chu kỳ, mọi vị thế có `|spot_qty − perp_short_qty| × price ≥ 10,0 USDT` phải sinh hành động `REPAIR_DELTA_DRIFT` và sau khi sửa, độ lệch còn lại < 10,0 USDT. Lượng sửa phải lượng hóa theo lot-size. | Unit test 2 kịch bản (lệch 5 $ → không sửa; lệch ≥ 10 $ → sửa) |
| **PR-5** | Drawdown kill-switch | Khi `current_equity < peak_equity × (1 − 0,06)`: đặt `kill_switch_triggered = True` vào state, hủy toàn bộ lệnh chờ, đóng vị thế, **không** đặt lệnh mới trong các chu kỳ sau cho tới khi reset. Ngưỡng 6 % là mặc định, cấu hình được. | Unit test: equity 0,93 × peak → bật; 0,97 × peak → không bật; chu kỳ sau kill-switch vẫn chặn |
| **PR-6** | Daemon nhiều chu kỳ | `--cycles N` + `--interval-sec S` chạy đúng N chu kỳ rồi thoát mã 0; mỗi chu kỳ in dashboard gồm Capital, Current Equity, Peak Equity, Drawdown %, Carry (Spot vs Perp), Overlay (Long vs Short), Kill-Switch Status. `--cycles` không có → hành vi cũ (`--once` hoặc loop vô hạn). | Chạy thật `--cycles 2 --interval-sec 1`, kiểm tra output + state |

## B. Cổng chế độ kết nối (PR-7, PR-8)

| Mã | Tiêu chí |
|---|---|
| **PR-7** | `--mode testnet`: chỉ bật sandbox khi có API key trong biến môi trường (`BINANCE_API_KEY`/`BINANCE_API_SECRET`); thiếu key → log cảnh báo rõ và rơi về *Public Live Feed Paper Mode* (giá/funding/orderbook thật, khớp giả lập), **không** được âm thầm gửi lệnh thật. `--mode live` không có key → dừng với thông báo lỗi rõ ràng, mã thoát ≠ 0. |
| **PR-8** | Không hồi quy: toàn bộ test cũ vẫn xanh, baseline `305 passed` + test mới; bất biến kế toán Sprint 11 giữ nguyên (`accounting_gap < 1e-6`); `executor_functional` vẫn là cờ thật. |

## C. Điều kiện NO-GO (bất kỳ điều nào xảy ra)

1. Bất kỳ cổng PR-1 … PR-8 không có test chứng minh hoặc test đỏ.
2. Bất kỳ thay đổi nào chạm vào tham số tín hiệu (`entry_funding_threshold`,
   `exit_funding_threshold`, `min_hold_periods`, `buffer_rank`, `top_k`, `eta`,
   `carry_tilt_weight`, `smooth_target_vol_ann`, `min_weight_change_to_trade`).
3. State hỏng/không đọc được sau khi daemon chạy 2 chu kỳ, hoặc equity trong state khác
   `--capital` khi chưa có vị thế.
4. Kill-switch không chặn được lệnh mới ở chu kỳ kế tiếp.
5. Có đường code nào gửi lệnh thật khi thiếu API key hoặc khi đang ở `--mode paper`.
6. Báo cáo `docs/sprint12_report.md` không nêu rõ số liệu nào là mô phỏng, số liệu nào là
   testnet/live thật.

## D. Bằng chứng phải nộp kèm

- `docs/sprint11_evidence_tables.md` (đã lưu, bất biến — không sửa số).
- Kết quả `pytest -q` 2 lần liên tiếp (số test, thời gian).
- Output console của `run_live.py --mode paper --strategy barbell --capital 3000 --reset-state --cycles 2 --interval-sec 1`.
- Nội dung `data/state/bot_state.json` sau lần chạy trên (equity, peak_equity, kill_switch).
- `docs/sprint12_report.md` với bảng đối chiếu "trước Sprint 12 / sau Sprint 12" cho 4 module.
