# Sprint 11 — Bằng chứng định lượng lưu trữ vĩnh viễn

**Nguồn:** `data/sprint11_payload.json` (sinh bởi `python run_sprint11.py --capital 3000`)
**Ngày sinh payload:** 2026-10-20 | **Nhánh:** `agy/sprint11-tier1-barbell-live` (đã merge vào `master` tại `aeb55c8`)
**Phạm vi:** 2023-04-01 → 2026-10-19 (≈3,45 năm, 1h UTC), vốn 3.000 USDT, 8 symbol LIQUID-8
**Holdout:** 2026-04-21 → 2026-10-18 (181 ngày)
**Lưu ý:** `data/` nằm trong `.gitignore`, nên tệp này là bản sao được Git theo dõi. Số liệu
bên dưới là nguyên văn từ payload, không chỉnh sửa.

---

## Bảng 1 — Ma trận 4 combo (toàn kỳ)

| Combo | AnnRet% | Sharpe | MaxDD% | PositivePeriods | HoldoutRet% | CarryPnL$ | OverlayNet$ | BasisStd | BetaBTC | AccountingGap | Executor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tier1a/hard_binary | 5.50% | 1.54 | 3.79% | 5/7 | -0.69% | 483.62 | 217.18 | 0.244426 | 0.0021 | 2.28e-10 | PASS |
| tier1a/smooth_conviction | 9.20% | 1.69 | 4.54% | 7/7 | 2.90% | 498.62 | 675.79 | 0.252024 | 0.0029 | 2.53e-10 | PASS |
| tier1b/hard_binary | 6.60% | 1.77 | 3.59% | 6/7 | -0.37% | 682.31 | 216.89 | 0.436501 | 0.0029 | 1.65e-10 | PASS |
| **tier1b/smooth_conviction** | **10.36%** | **1.86** | **4.44%** | **7/7** | **3.24%** | **705.53** | **691.09** | **0.457734** | **0.0034** | **2.30e-10** | **PASS** |

## Bảng 2 — Bảy kỳ lưới (Primary: tier1b/smooth_conviction)

| Kỳ | Bắt đầu | Kết thúc | Lợi nhuận (%) |
| --- | --- | --- | --- |
| 1 | 2023-04-07T02:00:00+00:00 | 2023-10-08T12:00:00+00:00 | 1.07% |
| 2 | 2023-10-08T13:00:00+00:00 | 2024-04-09T22:00:00+00:00 | 7.54% |
| 3 | 2024-04-09T23:00:00+00:00 | 2024-10-11T08:00:00+00:00 | 0.91% |
| 4 | 2024-10-11T09:00:00+00:00 | 2025-04-13T18:00:00+00:00 | 16.03% |
| 5 | 2025-04-13T19:00:00+00:00 | 2025-10-15T04:00:00+00:00 | 5.83% |
| 6 | 2025-10-15T05:00:00+00:00 | 2026-04-17T14:00:00+00:00 | 1.15% |
| 7 | 2026-04-17T15:00:00+00:00 | 2026-10-19T00:00:00+00:00 | 4.04% |

## Bảng 3 — Holdout (181 ngày)

| combo | holdout% | holdout_carry_pnl | holdout_overlay_net_pnl | holdout_days |
| --- | --- | --- | --- | --- |
| tier1a/hard_binary | -0.69% | 36.17 | -48.83 | 181.0 |
| tier1a/smooth_conviction | 2.90% | 36.95 | 58.40 | 181.0 |
| tier1b/hard_binary | -0.37% | 53.06 | -49.00 | 181.0 |
| **tier1b/smooth_conviction** | **3.24%** | **54.10** | **58.28** | **181.0** |

## Bảng 4 — GO/NO-GO Sprint 11 (nguyên trạng phán quyết)

| name | value | threshold | passed |
| --- | --- | --- | --- |
| annualized_return_pct | 10.36 | >= 9.0% | PASS |
| min_positive_periods | 7 | >= 6/7 | PASS |
| holdout_consistency | return_pct=3.24, carry_pnl=54.10, overlay_net_pnl=58.28 | ret >= 3.5% & carry_pnl > 0 & overlay_net > 0 | FAIL |
| sharpe | 1.86 | >= 1.8 | PASS |
| maxdd_pct | 4.44 | <= 7.5% | PASS |
| abs_beta_btc | 0.0034 | <= 0.15 | PASS |
| basis_mtm_std | 0.4577 | > 0 | PASS |
| accounting_gap | 0.0000 | < 1e-6 | PASS |
| overlay_quantization_events | 1780 | > 0 | PASS |
| executor_functional | True | == True | PASS |

**PHAN QUYET: NO-GO** — lý do duy nhất: Holdout 3,24% < 3,5%.

## Bảng 5 — Chi tiết Primary (số gốc từ payload)

| Chỉ số | Giá trị |
| --- | --- |
| `final_equity` | 4.251,37 USDT |
| `net_pnl` toàn kỳ | 1.251,37 USDT |
| `annualized_return_pct` | 10,3592 % |
| `sharpe` | 1,8602 |
| `maxdd_pct` | 4,4407 % |
| `carry_pnl` toàn kỳ | 705,53 USDT |
| `overlay_gross_pnl` / `overlay_net_pnl` | (xem payload) / 691,09 USDT |
| `overlay_fees` | 91,30 USDT |
| `overlay_quantization_events` | 1.780 |
| `overlay_mean_leverage` | 0,1545 |
| `overlay_gate_on_ratio` | 0,9660 |
| `basis_mtm_std` | 0,45773 |
| `beta_btc` | 0,00337 |
| `accounting_gap` | 2,30e-10 |
| `carry_fallback` | False |
| `holdout_days` | 181 |
| `holdout_carry_pnl` | 54,10 USDT |
| `holdout_overlay_net_pnl` | 58,28 USDT |
| `holdout_net_pnl` (carry + overlay) | **112,38 USDT** |
| Holdout trên **vốn hoạt động** 3.000 | **+3,75 %** |
| Holdout trên **equity tích lũy** đầu kỳ | **+3,24 %** |

### Ghi chú phương pháp (đính chính cách đọc, không đổi phán quyết)

Sprint 11 neo quy mô lệnh theo `initial_capital = 3000` (tham số `reinvest_profits = False`
hiện là mặc định ngầm). Vì equity đầu kỳ Holdout đã nở lên khoảng 3.468 USDT, mẫu số
"equity tích lũy" làm tỷ suất Holdout hiển thị 3,24 %, trong khi tỷ suất trên **vốn thực sự
được triển khai** (3.000) là 112,38 / 3.000 = **3,75 %**. Cả hai con số đều đúng; Sprint 12
Module 1 sẽ xuất minh bạch cả hai chỉ số thay vì chỉ in một con số dễ gây tranh luận.
Phán quyết NO-GO của Sprint 11 **giữ nguyên** vì tiêu chí đã khóa trước khi chạy.
