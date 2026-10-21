# Sprint 13 — Biên bản Kiểm toán Độc lập Vòng 2 (Round-2 Audit)

**Ngày:** 2026-10-20
**Nhánh:** `agy/sprint13-audit2-and-e2e-drills` (tách từ `master` tại `06beb0b`)
**Baseline xác nhận:** `pytest -q` → **342 passed** (34,94 s), trước khi sửa bất kỳ dòng nào.
**Phạm vi:** 4 lỗ hổng edge-case còn lại sau bản vá `c385e3c` của Sprint 12.
**Nguyên tắc:** Alpha đóng băng — không đổi tham số `tier1b` / `smooth_conviction`.
**Phương pháp:** đọc trực tiếp mã nguồn + đối chiếu hành vi, không dựa vào báo cáo Sprint 12.

---

## R2-1 — Partial Emergency Flatten Resilience

**Kết luận: LỖ HỔNG XÁC NHẬN (CRITICAL).**

Bằng chứng (`src/carry_live_executor.py`):
- `check_kill_switch` dòng 1516–1537: với mỗi vị thế carry, khối `try/except` chỉ **log warning**
  khi lệnh đóng spot/perp thất bại (`e_spot`, `e_perp`), rồi **luôn** chạy
  `closed_carry_count += 1` và `self.state.carry_positions.pop(sym, None)`.
- Không có biến `close_ok`; không kiểm tra `status` trả về của `create_order`.
- Không tồn tại `kill_switch_pending_close` trong `BotState` (`src/state_store.py` dòng 169
  chỉ có `kill_switch_triggered`).
- Dòng 1161 và 2067 cũng `pop` overlay position mà không xác nhận đóng thành công.

**Hệ quả thực chiến:** kill-switch nổ trong lúc API lỗi/timeout → vị thế **vẫn còn trên sàn**
nhưng **đã biến mất khỏi state**. Chu kỳ sau bot coi như phẳng, không thử đóng lại, và có thể
mở vị thế mới chồng lên vị thế mồ côi. Đây là kịch bản mất tiền thật, không phải lỗi lý thuyết.

**Yêu cầu vá:** chỉ pop khi cả 2 chân xác nhận đóng (`close_ok is True`); giữ nguyên phần chưa
đóng; bật `kill_switch_pending_close=True`; đầu chu kỳ sau tự thử đóng nốt trước khi kết thúc;
chỉ tắt cờ khi `carry_positions` và `overlay_positions` đều rỗng.

---

## R2-2 — Self-Healing Sub-Lot Drift Flag

**Kết luận: LỖ HỔNG XÁC NHẬN (HIGH).**

Bằng chứng (`src/carry_live_executor.py` dòng 1381–1386):
```python
if pos.get("drift_unrepairable", False):
    if drift_usd >= threshold * 5:
        ... pop flag ...
    else:
        continue
```
- Ngưỡng gỡ cờ là `threshold * 5` với `threshold ≈ lot_step × price × 1.02`.
  Với BTC (step 0,001; giá 60.000) `threshold ≈ 61 USD`, nên phải lệch **≈ 306 USD** mới được thử lại.
  Kịch bản Blueprint (lệch 0,003 BTC ≈ 180 USD, đã đủ 1 lô) **vẫn bị bỏ qua** → sai.
- Cờ được lưu trên `pos["drift_unrepairable"]` (bool) chứ không có registry
  `drift_unrepairable_symbols` trong `BotState`; sau restart state cũ vẫn dính cờ.
- Việc quyết định "sửa được hay không" dựa trên `drift_usd` chứ không dựa trên
  `floor_to_step(abs(drift_qty), step) >= min_qty`.

**Yêu cầu vá:** mỗi chu kỳ tính lại `repair_qty = floor_to_step(abs(drift_qty), step)`; nếu
`repair_qty >= min_qty` và `drift_usd >= max(10, min_qty × mark × 1.05)` thì **tự động gỡ cờ**
và sửa; nếu `repair_qty < min_qty` thì đánh cờ dust và **không** đặt lệnh 0 lot; nếu drift về 0
thì gỡ cờ.

---

## R2-3 — Active Capital Downside Clamp khi `reinvest_profits=False`

**Kết luận: LỖ HỔNG XÁC NHẬN (HIGH).**

Bằng chứng (`src/barbell_portfolio.py`):
- `_simulate_barbell_equity` dòng 345–348: khi `reinvest_profits=False` thì
  `eq_mid = float(initial_capital)` — hằng số, không phụ thuộc equity hiện tại.
- Không có chỗ nào dùng `min(initial_capital, current_equity)`.
- `g_base` cho overlay (dòng ~333) cũng dùng `initial_capital` tuyệt đối.

**Hệ quả:** nếu tài khoản đang lỗ (ví dụ equity 2.850 khi vốn gốc 3.000), chế độ "neo vốn tĩnh"
vẫn đặt lệnh theo 3.000 → **đánh đòn bẩy vượt vốn thực còn lại**, đúng lúc rủi ro cao nhất.
Điều này chưa từng lộ ra trong backtest vì 7/7 kỳ đều dương.

**Yêu cầu vá:** `active_capital = max(0.0, min(initial_capital, current_equity))` tại mỗi lần
rebalance; khi equity ≥ vốn gốc thì kết quả lịch sử **không đổi một chữ số thập phân nào**.

---

## R2-4 — Orderbook Stale/Crossed Book Guard

**Kết luận: LỖ HỔNG XÁC NHẬN (HIGH).**

Bằng chứng (`src/carry_live_executor.py` dòng 302–340):
- `book_ok = (best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > best_bid)`
- Nhưng nhánh chặn chỉ chạy khi `not book_ok and self.mode in ("live", "testnet")`.
- Không có kiểm tra tường minh `ask < bid` (crossed book) ở bất kỳ chế độ nào; điều kiện
  `best_ask > best_bid` nằm trong `book_ok`, và ở `paper` sai `book_ok` vẫn đi tiếp bằng giá
  fallback (`ref_price ± 0,5 bps`).

**Hệ quả:** paper mode — nơi dùng để kiểm thử trước khi lên testnet — có thể đặt lệnh trên
sổ lệnh chéo/lỗi mà không cảnh báo, tạo bằng chứng giả rằng guard hoạt động.

**Yêu cầu vá:** `bid/ask` có truyền vào mà `≤ 0` hoặc `ask < bid` → trả `False` (chặn) ở **cả
`paper`, `testnet`, `live`**; riêng `testnet`/`live` thì thiếu orderbook (`None`) cũng chặn.

---

## Bảng tổng hợp

| Mã | Mức | Hiện trạng | Ảnh hưởng nếu không vá |
|---|---|---|---|
| R2-1 | CRITICAL | Vị thế bị pop dù đóng thất bại | Vị thế mồ côi trên sàn, mất dấu rủi ro |
| R2-2 | HIGH | Cờ dust chỉ gỡ ở 5× ngưỡng (~306 USD với BTC) | Drift 1–5 lô bị bỏ qua vĩnh viễn |
| R2-3 | HIGH | Đặt lệnh theo 3.000 dù equity đã dưới 3.000 | Đòn bẩy vượt vốn thực khi đang lỗ |
| R2-4 | HIGH | Crossed book chỉ bị chặn ở live/testnet | Paper cho qua sổ lệnh lỗi, bằng chứng giả |

**Kết luận Chặng 0:** 4/4 lỗ hổng trong Blueprint được xác nhận là **có thật** trên mã hiện tại,
không phải phòng xa. Chặng 1 vá đúng 4 điểm này, mỗi điểm kèm test hồi quy.
