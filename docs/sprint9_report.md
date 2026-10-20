# Sprint 9 Report — Cross-Exchange Perp-Perp Carry + Maker-Taker Hedge + Core-4 Barbell

**Ngày:** 2026-10-20
**Vốn giả định:** 3,000 USDT
**Chuỗi kiểm thử:** 2023-05-07 → 2026-10-18 (30,241 nến 1h, 3.45 năm, 7 kỳ 6 tháng)
**Nhánh:** `agy/sprint9-cross-exchange-barbell`
**Test suite:** 241 passed (baseline trước Sprint 9: 224)

---

## 1. Kết luận điều hành

> ## ❌ NO-GO — KHÔNG triển khai tiền thật ở trạng thái hiện tại.

Không Tier nào đạt đủ 4 tiêu chí GO đã khóa trong `docs/go_no_go_sprint9.md`:

| Tier | Mô tả | AnnRet | Sharpe | MaxDD | Kỳ dương | Holdout | Phán quyết |
|---|---|---|---|---|---|---|---|
| 1 | Single-Exchange Binance Static Carry | 2.85% | 11.07 | 0.20% | 3/7 | 0.00% | FAIL |
| 2 | Multi-Exchange Best-Venue Carry | 3.27% | 10.55 | 0.15% | 3/7 | 0.00% | FAIL |
| 3 | Hybrid Cross-Exchange Perp-Perp | 3.36% | 9.35 | 0.44% | 5/7 | 0.00% | **NO-GO** |
| 4 | Full Barbell Portfolio | **13.00%** | **1.18** | **13.86%** | 6/7 | 5.86% | **NO-GO** |

- **Tier 3 (carry thuần):** trượt GO-1 (3.36% < 10.0%) và GO-2 (5/7 < 6/7, holdout = 0.00%).
- **Tier 4 (carry + overlay):** đạt GO-1 nhưng trượt GO-2 (holdout 5.86% < 8.0%) và GO-3 (Sharpe 1.18 < 1.80, MaxDD 13.86% > 8.0%). Toàn bộ 13% lợi nhuận đến từ overlay momentum, không phải carry.
- **Chẩn đoán gốc:** chênh lệch funding liên sàn trên rổ Core-4 đã bị HFT san phẳng (trung bình ≈ 0.26 bps/8h; funding đơn sàn trung bình ≈ 0.79 bps/8h). Với phí khứ hồi thực tế 20 bps, chiến lược chỉ vào lệnh được khi spread ≥ 1.67 bps/8h → đứng ngoài thị trường phần lớn thời gian. Đây là **giới hạn của thị trường**, không phải lỗi tham số đơn thuần.

**Cảnh báo đỏ từ Red Team:** có 3 phát hiện CRITICAL về an toàn vận hành. Hai lỗi trong nhóm này **đã được vá trong Sprint 9** (hủy lệnh Post-Only mồ côi; sửa double-count phí overlay). Lỗi còn lại — executor chưa hỗ trợ cross-exchange Perp-Perp — **vẫn mở** và là lý do GO-4 chỉ nên coi là "máy trạng thái đạt, kiến trúc đa sàn chưa đạt".

---

## 2. Mô hình phí bắt buộc (không dùng giả định 2-leg maker)

| Chế độ | Chân 1 | Chân 2 | Một chiều | Khứ hồi |
|---|---|---|---|---|
| Perp-Perp | Post-Only GTX: maker 2 bps + adverse selection 1 bp | Market: taker 5 bps + slippage 2 bps | **10 bps** | **20 bps** |
| Spot-Perp | Maker spot 5 bps + adverse 1 bp | Market perp 5 bps + slippage 0.25 bp | **11.25 bps** | **22.5 bps** |

Rebalance ký quỹ khi lệch ≥ ±25% tính thêm 5 bps notional (xem phát hiện #4 — mức này còn lạc quan, xem mục 6).

---

## 3. Bảng 7 kỳ (compounded, vốn đầu kỳ sau = vốn cuối kỳ trước)

### Tier 3 — Hybrid Cross-Exchange Perp-Perp
| Kỳ | Khoảng | Net % | Ann % | Sharpe | MaxDD % | Dương |
|---|---|---|---|---|---|---|
| P1 | 2023-05-07 → 2023-11-03 | +0.58 | 1.18 | 3.58 | 0.19 | ✔ |
| P2 | 2023-11-03 → 2024-05-01 | +7.81 | 16.48 | 23.09 | 0.08 | ✔ |
| P3 | 2024-05-01 → 2024-10-28 | +1.13 | 2.30 | 9.05 | 0.25 | ✔ |
| P4 | 2024-10-28 → 2025-04-26 | +2.40 | 4.93 | 10.88 | 0.18 | ✔ |
| P5 | 2025-04-26 → 2025-10-23 | −0.26 | −0.53 | −2.92 | 0.26 | ✘ |
| P6 | 2025-10-23 → 2026-04-21 | +0.05 | 0.11 | 0.95 | 0.06 | ✔ |
| **P7 (holdout)** | **2026-04-21 → 2026-10-18** | **0.00** | **0.00** | 0.00 | 0.00 | ✘ |

### Tier 4 — Full Barbell (carry 1.0x + Core-4 momentum overlay 0.40x)
| Kỳ | Khoảng | Net % | Ann % | Sharpe | MaxDD % | Dương |
|---|---|---|---|---|---|---|
| P1 | 2023-05-07 → 2023-11-03 | +18.17 | 40.29 | 3.04 | 5.63 | ✔ |
| P2 | 2023-11-03 → 2024-05-01 | +9.86 | 21.01 | 1.29 | **13.86** | ✔ |
| P3 | 2024-05-01 → 2024-10-28 | +7.16 | 15.05 | 1.47 | 4.16 | ✔ |
| P4 | 2024-10-28 → 2025-04-26 | +8.30 | 17.55 | 1.39 | 7.65 | ✔ |
| P5 | 2025-04-26 → 2025-10-23 | −1.98 | −3.97 | −0.36 | 6.85 | ✘ |
| P6 | 2025-10-23 → 2026-04-21 | +0.35 | 0.71 | 0.13 | 5.28 | ✔ |
| **P7 (holdout)** | **2026-04-21 → 2026-10-18** | **+2.85** | **5.86** | 0.90 | 4.69 | ✔ |

**Nhận xét:** P1 của Tier 4 (+18.17%, Sharpe 3.04) là giai đoạn duy nhất overlay thực sự tốt; P5 lỗ, P6 gần như đi ngang. Không có bằng chứng edge ổn định ngoài mẫu.

---

## 4. Attribution (USDT, toàn chuỗi)

| Tier | Funding | Basis | Overlay (gross) | Phí | Net |
|---|---|---|---|---|---|
| 1 | 323.17 | 0.00 | — | 17.62 | 305.54 |
| 2 | 378.17 | 0.00 | — | 25.68 | 352.49 |
| 3 | 430.49 | 0.00 | — | 68.76 | 361.73 |
| 4 | 430.49 | 0.00 | 1,080.54 | 177.96 | 1,572.68 |

Đẳng thức kế toán: `funding + basis + overlay_gross − fees = net_pnl` đã được khóa bằng test (trước Sprint 9, overlay là net nên phí bị trừ hai lần, lệch ~109 USDT trên toàn chuỗi — **đã vá**).
`basis_pnl = 0` tuyệt đối là **giả định lạc quan** (xem phát hiện #5).

---

## 5. Giới hạn dữ liệu (bắt buộc công bố)

- **Binance:** funding 8h đầy đủ 2023-05-07 → 2026-10-18.
- **Bybit:** funding 8h đầy đủ (3,780 kỳ, từ 2023-05-07 08:00, phân trang `limit=200` hoạt động).
- **OKX:** API công khai `fetch_funding_rate_history` **chỉ trả ~3 tháng gần nhất** (296 kỳ, từ 2026-07-13), bất kể `since`/`params.before`. Toàn bộ 2023-07/2026-07 là `NaN`.
  → **Tier 2 và Tier 3 thực chất chỉ chọn giữa Binance và Bybit trong 3.2 năm đầu.** Tính năng "3 sàn" chưa từng được kiểm chứng qua bear market 2023 và halving 2024. Muốn kiểm chứng phải mua dữ liệu lịch sử (Tardis/Kaiko/CSV archive).
- Giá: nến 1h perp + spot cache cục bộ (`data/raw/*.parquet`, gitignored).

---

## 6. Red Team Audit (độc lập, `quick_pro`) — 11 phát hiện

| # | Mức | Vị trí | Vấn đề | Ảnh hưởng | Trạng thái |
|---|---|---|---|---|---|
| 1 | CRITICAL | `carry_live_executor.py` | Đặt GTX rồi verify ngay; không khớp → xóa state nhưng **không cancel** → lệnh mồ côi, có thể thành vị thế trần trụi | Rủi ro cháy tài khoản khi giá quét qua limit | **ĐÃ VÁ** (`cancel_maker_leg1`, test mới) |
| 2 | CRITICAL | `carry_live_executor.py` | Executor vẫn dùng `DynamicCarryConfig` + Spot-Perp Sprint 8, hardcode Binance; chưa có logic Cross-Exchange Perp-Perp; `detect_executor_state_machine` chỉ kiểm tra hình thức | GO-4 "PASS" là **pass giả**; live ≠ backtest | **MỞ** — chặn go-live |
| 3 | CRITICAL | `cross_sectional.py:250-270` | Regime gate dùng `z_mom` (std=1) nên `max−min ≥ 2.0` **luôn** ≥ 0.75 → gate bật 100% (thực tế 98.9% thời gian có vị thế) | Overlay = momentum không gate; MaxDD 13.86% | **MỞ** — cần thiết kế lại gate |
| 4 | HIGH | `cross_exchange_carry.py:537-566` | Rebalance liên sàn 5 bps: không thể rút lãi chưa thực hiện; phí on-chain 1–2 USDT/lần ≈ 27–53 bps trên lệnh $375 | Chi phí thực cao gấp 5–10 lần | **MỞ** — cần mô hình partial-close |
| 5 | HIGH | `cross_exchange_carry.py:795` | `hourly_basis = zeros`; 62% thời gian đứng ngoài → Sharpe 9.35 là ảo ảnh | Live Sharpe dự kiến 1–2, không phải 9.35 | **MỞ** — cần MTM basis 2 sàn |
| 6 | HIGH | `barbell_portfolio.py` | `overlay_pnl` lấy từ return net nhưng vẫn cộng phí overlay vào cột phí → lệch kế toán ~109 USDT | Sai lệch báo cáo | **ĐÃ VÁ** (overlay_gross trong attribution) |
| 7 | HIGH | `cross_exchange_carry.py:54,111-115` | Payback ≤ 12 kỳ ⇒ cần ≥ 1.67 bps/8h trong khi spread thực 0.26 bps | Kẹt tiền mặt 62% thời gian, Tier 3 ≈ 3.4%/năm | **MỞ** — nới payback / trailing exit |
| 8 | MEDIUM | `cross_exchange_carry.py:476-533` | Exit khớp 100% tại close nhưng tính phí maker (bất đối xứng với entry) | Lạc quan ở chiều thoát | **MỞ** |
| 9 | MEDIUM | `multi_exchange_loader.py` | OKX thiếu 92% chuỗi lịch sử | Chưa kiểm chứng 3 sàn | **ĐÃ CÔNG BỐ** (mục 5) |
| 10 | MEDIUM | `cross_exchange_carry.py:701-706` | Không lượng tử hóa lot size / min notional; $750/chân ⇒ lệch delta 8–12% | Không delta-neutral hoàn hảo ở vốn nhỏ | **MỞ** |
| 11 | LOW | `multi_exchange_loader.py:424-436` | Gộp funding 4h vào mốc 8h, sai lệch 1–3 bps khi sàn đổi lịch | Sai số cục bộ | **MỞ** |

---

## 7. Checklist GO/NO-GO

| Tiêu chí | Ngưỡng | Tier 3 | Tier 4 | Kết quả |
|---|---|---|---|---|
| GO-1 Lợi nhuận | AnnRet ≥ 10.0% (≥ +300 USDT/năm trên 3000) | 3.36% | 13.00% | Tier 3 ✘ / Tier 4 ✔ |
| GO-2 Ổn định | ≥ 6/7 kỳ dương **và** holdout ≥ 8.0%/năm **và** kỳ xấu ≥ −3.0% | 5/7, 0.00%, −0.26% | 6/7, 5.86%, −1.98% | ✘ cả hai (holdout) |
| GO-3 Rủi ro | Sharpe ≥ 1.80, MaxDD ≤ 8.0%, \|β_BTC\| ≤ 0.15, 0 margin call | 9.35 / 0.44% / 0.00 / 0 | 1.18 / 13.86% / 0.00 / 0 | Tier 3 ✔ (Sharpe ảo) / Tier 4 ✘ |
| GO-4 Thực thi | `POST_ONLY_LEG_1` → `MARKET_HEDGE_LEG_2`, không bắn 2 market mù | Máy trạng thái ✔, không orphan ✔ | như Tier 3 | ⚠️ **một phần** (kiến trúc đa sàn chưa có) |

---

## 8. Đã thay đổi trong Sprint 9

- `src/multi_exchange_loader.py` — tải/chuẩn hóa funding Binance + Bybit + OKX, `align_multi_exchange_funding` (ffill + shift(1), không look-ahead).
- `src/cross_exchange_carry.py` — carry engine Perp-Perp & Spot-Perp, phí 20/22.5 bps, hysteresis payback ≤ 12 kỳ, rebalance ±25%.
- `src/barbell_portfolio.py` — barbell carry + overlay 0.40x, attribution 7 cột, sửa double-count phí.
- `src/carry_live_executor.py` — máy trạng thái maker→taker, hedge partial fill, `cancel_maker_leg1` chống lệnh mồ côi.
- `src/sprint9_eval.py`, `run_sprint9.py` — harness 4 Tier + GO/NO-GO, xuất CSV/Markdown.
- `tests/`: +17 test (multi-exchange loader 6, barbell/eval 8, executor 3) → **241 passed**.

Tái lập: `python run_sprint9.py --capital 3000` (thêm `--download` nếu chưa có cache).

---

## 9. Lộ trình để đạt GO một cách trung thực (đề xuất Sprint 10)

1. **Mở rộng rổ carry** sang 15–25 altcoin thanh khoản (volume 24h > 50M USDT: SUI, NEAR, AVAX, LINK, APT…), nơi spread funding liên sàn thường 15–35%/năm thay vì 2.8%.
2. **Nới payback + trailing exit**: `max_payback_periods` 90–135 kỳ; thoát khi spread về 0/đảo dấu 3 kỳ liên tiếp → tăng time-in-market từ 38% lên 75–85%.
3. **Sửa regime gate**: dùng spread tuyệt đối (ví dụ `abs(max−min) ≥ 2.5%`) hoặc z-score **theo thời gian** của spread (rolling 720h), kèm vol-targeting overlay 0.40x → 0.15x khi vol rổ tăng, mục tiêu MaxDD ≤ 8%.
4. **MTM basis thật** giữa 2 sàn trong equity curve — bắt buộc trước khi tin bất kỳ Sharpe nào của carry.
5. **Mô hình rebalance không cần chuyển tiền**: partial de-leverage/re-hedge thay vì rút on-chain.
6. **Hợp nhất executor**: port logic `cross_exchange_carry` sang `carry_live_executor` với `exchange_map` 3 sàn, lot-size quantization, TTL đa chu kỳ cho Post-Only.
7. **Dữ liệu OKX lịch sử** từ nguồn trả phí nếu muốn tuyên bố "3 sàn".

**Khuyến nghị:** tiếp tục paper trading; chỉ cân nhắc tiền thật sau khi (2)+(4)+(6) xong và có ≥ 1 kỳ holdout mới đạt ≥ 8%/năm với MaxDD ≤ 8%.

---

## 10. Phụ lục — cấu hình

- `CORE_4_SYMBOLS = [BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT, BNB/USDT:USDT]`
- Carry: `top_k=2`, EWMA halflife 24h, rebalance threshold 25%, `max_payback_periods=12`, yêu cầu expected 8h yield ≥ round_trip/12 = 1.67 bps.
- Overlay: XS momentum lookback 336h, rebalance 48h, gross 0.40x, long top-2/short bottom-2, `min_spread=0.75` (xem phát hiện #3).
- Annualized: `(final/initial)^(365.25/days) − 1`; Sharpe = mean/std lợi nhuận 1h × √8760; MaxDD là số dương.
