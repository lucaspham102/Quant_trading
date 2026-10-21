# Sprint 13 — Round-2 Audit, Warm-vs-Cold Holdout Bridge & 5-Cycle E2E Fire Drill

**Ngày:** 2026-10-21
**Nhánh:** `agy/sprint13-audit2-and-e2e-drills` (tách từ `master` tại `06beb0b`)
**Baseline:** 342 passed → **377 passed** (chạy 2 lần: 51,49 s và 51,14 s)
**Alpha:** đóng băng — không đổi bất kỳ siêu tham số nào của `tier1b` / `smooth_conviction`.

---

## 1. Kết luận nhanh

| Hạng mục | Kết quả |
|---|---|
| Round-2 audit (R2-1…R2-4) | **4/4 lỗ hổng có thật**, đã vá, 18 test hồi quy bảo vệ |
| Holdout bridge (A/B/C) | Bảng đối soát khép kín, residual = **0.00e+00** |
| Fire drill 5 chu kỳ | **20/20 check PASS**, chạy cả từ CLI và pytest |
| Phán quyết Sprint 11 | **VẪN NO-GO** (cold-start 3,24% < 3,5%) |
| Sẵn sàng testnet | **Chưa** — xem §6 |

---

## 2. Round-2 audit: 4 lỗ hổng đều là thật

Biên bản đầy đủ: `docs/sprint13_round2_audit.md` (đọc mã trước khi sửa, không dựa báo cáo cũ).

| Mã | Mức | Trước khi vá | Sau khi vá |
|---|---|---|---|
| **R2-1** | CRITICAL | `check_kill_switch` `pop` vị thế **bất kể** lệnh đóng thành công hay không; overlay cũng bị pop vô điều kiện ở 2 chỗ | Helper `_close_carry_position_emergency` / `_close_overlay_position_emergency` trả `bool`; chỉ `pop` khi `close_ok is True`; giữ lại vị thế lỗi + bật `kill_switch_pending_close`; đầu chu kỳ tự `_flush_pending_emergency_close()` |
| **R2-2** | HIGH | Cờ `drift_unrepairable` chỉ gỡ ở `threshold × 5` (BTC ≈ 306 USD) → lệch 0,003 BTC (180 USD, đã đủ lô) bị bỏ qua vĩnh viễn | Mỗi vòng tính lại `repair_qty = floor(drift/step)*step`; `repair_qty >= min_qty` và `drift_usd >= max(10, min_qty×mark×1.05)` → **tự gỡ cờ** và sửa; dust ghi vào `state.drift_unrepairable_symbols` |
| **R2-3** | HIGH | `reinvest_profits=False` neo cứng 3.000 kể cả khi equity đã 2.850 | `active_capital = max(0, min(initial_capital, equity))` cho cả carry và `g_base` overlay |
| **R2-4** | HIGH | Crossed book (`ask < bid`) và giá `≤ 0` chỉ bị chặn ở live/testnet | Chặn ở **mọi chế độ**, reason riêng `SPREAD_CROSSED_BOOK` / `SPREAD_INVALID_BOOK` |

Test hồi quy: `tests/test_sprint13_round2_hardening.py` — **18 test** (R2-1: 5, R2-2: 4, R2-3: 2, R2-4: 5, persistence: 1, cross-book parametrize 3 mode).

Điểm đáng chú ý: lỗi R2-1 khiến kill-switch "bật cờ nhưng mất dấu vị thế" — dạng lỗi chỉ lộ ra khi sàn từ chối/timeout đúng lúc cần đóng, tức lúc thị trường xấu nhất. Đây là lý do phải kiểm toán vòng 2 thay vì tin vào 342 test xanh.

---

## 3. Cầu nối kế toán Holdout

Sinh tự động: `scripts/gen_holdout_bridge.py` → `docs/sprint13_holdout_bridge.md`
(từ 2 lần chạy `run_sprint11.py`: `--reinvest 1` và `--reinvest 0`, cùng dữ liệu, cùng 7 kỳ).

| Phương pháp | Vốn đầu cửa sổ | PnL ròng | Tỷ suất | Bản chất |
|---|---|---|---|---|
| **A. Continuous compound** (`reinvest=True`) | 4.091,29 | **159,15 USDT** | **3,89%** trên equity đầu kỳ | Vị thế ấm, quy mô phình theo lãi kép |
| **B. Continuous fixed-base** (`reinvest=False`) | 3.946,72 (vốn hoạt động 3.000) | **118,50 USDT** | **3,95%** trên 3.000 | Vị thế ấm, quy mô neo 3.000 suốt 3,45 năm |
| **C. Standalone cold-start** | 3.000,00 | **97,07 USDT** | **3,24%** | Bắt đầu 100% tiền mặt, phải mở lại danh mục |

Phân rã: `A − B = +40,64` (hiệu ứng quy mô do lãi kép) và `B − C = 21,44` (ma sát cold-start),
trong đó **2,02 USD** là phí mở danh mục ngày đầu, còn **19,42 USD** là lệch đường đi do
`ema_span=42` / `min_hold=63` chưa ấm.

Đẳng thức: `159,147661 = 97,066418 + 40,644522 + 21,436720`, **residual = 0,00e+00**.

### Điều này nói gì về con số 3,75%
Con số 3,75% từng được trích dẫn là **carry gross** (funding + basis) cộng overlay net — chưa trừ
phí carry. Lãi ròng all-in thực tế của phương pháp cold-start là **3,24%**. Sprint 12 đã đính chính
một lần; Sprint 13 xác nhận lại bằng một lần chạy độc lập và bằng đẳng thức kế toán khép kín.

Ba phương pháp cho ba con số vì **cùng chiến lược, khác trạng thái vốn** — không phải vì
phương pháp nào "đúng" hơn. Vì vậy tiêu chí kép được giữ: cả slice (3,89%) và re-sim (3,24%)
phải ≥ 3,5% → hiện **NO-GO**.

---

## 4. Fire drill 5 chu kỳ

`scripts/run_e2e_fire_drill.py` → `docs/sprint13_fire_drill_evidence.md`, hàm
`run_five_cycle_fire_drill(tmp_state_path)` chạy cả từ CLI và từ pytest, offline, không mạng.

| Chu kỳ | Nội dung | Check | Kết quả |
|---|---|---|---|
| 1 | Mở Carry (maker→taker) + Overlay, ghi state | 4 | PASS — vị thế mở, state đọc lại được, qty bội số lot step |
| 2A | BTC lệch 0,0004 (dưới lô) + SOL lệch 0,25 | 3 | PASS — BTC gắn `drift_unrepairable`, **0 lệnh** cho BTC; SOL được sửa |
| 2B | BTC lệch 0,002 (≈ 2 lô) | 2 | PASS — cờ tự gỡ, lệnh sửa lệch được đặt |
| 3 | Spread 18,5 bps + crossed book | 3 | PASS — `SPREAD_TOO_WIDE` / `SPREAD_CROSSED_BOOK`, 0 lệnh |
| 4 | Equity 2.812,50 (−6,25% từ peak 3.000) | 5 | PASS — kill-switch bật, đóng sạch carry + overlay, ghi cờ xuống đĩa, chỉ có lệnh market đóng |
| 5A | Restart từ đĩa, đổi `capital=5000` | 1 | PASS — cờ dính, `actions == []` |
| 5B | `reset=True` | 2 | PASS — cờ tắt, equity về 3.000, **đã ghi lại xuống đĩa** |

**Tổng: 20/20 check PASS.** Test: `tests/test_sprint13_fire_drill.py` — 10 test (gồm kiểm tra
tính deterministic khi chạy lại trên đường dẫn khác).

Khác biệt so với Sprint 12: lần này **có vị thế thật được mở** (tín hiệu tổng hợp), nên đường đi
lệnh maker→taker→lưu JSON→đóng khẩn cấp được kích hoạt thật, không chỉ chạy vòng lặp rỗng.

---

## 5. Kiểm định kỹ thuật

- `python -m pytest -q` → **377 passed**, hai lần liên tiếp (51,49 s / 51,14 s), 0 failed.
- `python scripts/run_e2e_fire_drill.py` → 20/20 PASS, sinh bằng chứng Markdown.
- `python run_live.py --mode paper --strategy barbell --capital 3000 --reset-state --cycles 2 --interval-sec 1`
  → 2 chu kỳ, dashboard đủ 7 trường, `Capital=3000.00 | Equity=3000.00 | DD=0.00% | Kill-Switch=False`.
- Bất biến kế toán giữ nguyên ở cả `reinvest=True`/`False` (< 1e-6); bridge residual = 0.

---

## 6. Bảng kiểm tra điều kiện lên Testnet

| # | Điều kiện | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Kill-switch đóng được vị thế **và** giữ dấu khi đóng lỗi | ✅ | R2-1 vá + 5 test + drill cycle 4 |
| 2 | Drift tự phục hồi khi vượt 1 lô | ✅ | R2-2 + drill cycle 2B |
| 3 | Spread/crossed book chặn ở mọi chế độ | ✅ | R2-4 + drill cycle 3 |
| 4 | Vốn co theo equity khi đang lỗ | ✅ | R2-3 + 2 test |
| 5 | Cờ an toàn dính qua restart, gỡ bằng reset | ✅ | drill cycle 5, test persistence |
| 6 | Đường đi lệnh end-to-end có vị thế thật | ✅ (stub) | drill dùng `FakeExchange`, **chưa phải sàn thật** |
| 7 | Chạy testnet 7 ngày liên tục với key thật | ❌ | Chưa làm — cần API key + soak test |
| 8 | Kiểm toán độc lập bởi bên thứ ba | ❌ | Vòng 2 do chính nhóm thực hiện |
| 9 | Số liệu holdout đạt ngưỡng 3,5% (cold-start) | ❌ | 3,24% → NO-GO |
| 10 | Tiêu chí holdout chốt **một** phương pháp trước khi chạy | ❌ | Hiện có 2 phương pháp cho 2 kết quả |

---

## 7. Điều KHÔNG được kết luận

1. **Chưa có lệnh nào chạm sàn thật.** Toàn bộ drill dùng `FakeExchange`; chưa ký lệnh Binance nào.
2. **Chưa soak-test 24/7.** 5 chu kỳ chứng minh logic tuần tự, không chứng minh chịu reconnect,
   rate-limit, partial fill, hay rò rỉ bộ nhớ qua nhiều ngày.
3. **`run_live.py --cycles 2` vẫn mở 0 vị thế** trên dữ liệu cache 8 mã (điểm dưới ngưỡng) —
   hạn chế của Sprint 12 chỉ được khắc phục trong drill có tín hiệu tổng hợp, không phải trên
   dữ liệu thị trường thật.
4. **Drift dưới 1 lô vẫn không sửa được** với BTC/ETH (lot step ≈ 60 USD) — hệ thống ghi nhận và
   bỏ qua, chấp nhận sai số delta-neutral ở mức đó; đây là giới hạn của sàn, không phải bug.
5. **Audit vòng 2 do cùng nhóm thực hiện.** 4/4 lỗ hổng được tìm thấy là tín hiệu tốt, nhưng
   không thay thế được review độc lập.

---

## 8. Tệp thay đổi

| Tệp | Nội dung |
|---|---|
| `src/state_store.py` | `kill_switch_pending_close`, `drift_unrepairable_symbols`; reset sạch, sticky khi sync |
| `src/carry_live_executor.py` | R2-1 (helper đóng khẩn cấp + flush pending), R2-2 (self-healing drift), R2-4 (crossed/invalid book), `_reject_leg1_spread` |
| `src/barbell_portfolio.py` | R2-3 active-capital clamp cho carry + overlay |
| `src/sprint11_eval.py` | `reinvest_profits` param, `holdout_resim_entry_fee_usd`, `compute_holdout_accounting_bridge` |
| `run_sprint11.py` | `--reinvest {0,1}` |
| `scripts/run_e2e_fire_drill.py` | Drill 5 chu kỳ + xuất bằng chứng |
| `scripts/gen_holdout_bridge.py` | Sinh bảng đối soát holdout |
| `tests/test_sprint13_round2_hardening.py` | 18 test |
| `tests/test_sprint13_holdout_bridge.py` | 7 test |
| `tests/test_sprint13_fire_drill.py` | 10 test |
| `docs/sprint13_round2_audit.md` | Biên bản audit vòng 2 |
| `docs/sprint13_holdout_bridge.md` | Bảng đối soát A/B/C + phân rã |
| `docs/sprint13_fire_drill_evidence.md` | Bằng chứng 20/20 check |

**Commit:** `8f3d58c` → `8c641be` → `effa142` → `934636e` → (báo cáo này).
