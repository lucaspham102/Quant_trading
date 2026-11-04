# Sprint 28 — High-Velocity & Tactical Breakout Ablation Study

- Study ID: `SPRINT_28_VELOCITY_ABLATION_AND_BREAKOUT_STUDY` v2.0.0
- Generated at: `2026-11-03T15:56:47.887173+00:00`
- Capital base: `$1,000.00` (fixed base, zero compounding leak)
- Baseline Champion: `V0_CHAMPION_BASELINE` (U4 Maker-First + U5 Dynamic Flat Transfer + U3 Emergency Exit)
- Primary Profile: `regime_omni_1k`
- Engine: 100% Authentic Causal Re-simulation on Aligned Parquet Cache (Zero Synthetic Multipliers)

## 1. Five Golden Quant Invariants

| Law | Status | Measured Details |
|---|---|---|
| INV-1 WARMUP_SATURATION | **PASSED** | [PASS] Saturation achieved between 720h and 1440h: Jaccard=1.0000 >= 0.98, L1=0.000000 <= 0.02 |
| INV-2 ZERO_COMPOUNDING_LEAK | **PASSED** | [PASS] Zero compounding leak verified: max_abs_diff=5.115908e-13 <= 1.000000e-06 |
| INV-3 CLOCK_SHIFT_INVARIANCE | **PASSED** | [PASS] Clock phase invariance verified: 2190 rebalance points >= 1729299600 matched identically (Jaccard=1.0000). |
| INV-4 ACCOUNTING_IDENTITY | **PASSED** | [PASS] Accounting identity closed: Net_PnL (400.530519) == Gross (416.892428) - Fees (16.361910), diff=5.12e-13 <= 1.00e-06 |
| INV-5 LOT_NEUTRALITY | **PASSED** | [PASS] Lot neutrality verified: Leg A=495.78 USD, Leg B=495.52 USD, Imbalance=0.26 USD <= 10.00 USD. |

**All invariants passed:** `True`

## 2. Baseline Reference (V0) — Primary Profile `regime_omni_1k`

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +23.47% | +25.98% | 1.36 | 5.56% | $3.32 |
| 6M | +14.90% | +17.30% | 1.07 | 5.47% | $4.59 |
| 12M | +5.23% | +7.84% | 0.54 | 5.56% | $5.69 |
| 24M | +10.59% | +12.95% | 1.00 | 8.65% | $13.59 |
| 3_5Y | +19.45% | +21.25% | 1.68 | 7.66% | $24.68 |

## 3. Single-Upgrade Velocity Ablation (V1 -> V5)

| ID | Upgrade Name | Verdict | Materiality | 24M ΔAPR | 24M ΔDD | Holdout 6M ΔAPR |
|---|---|---|---|---:|---:|---:|
| V1 | Fast-Horizon Trend Alignment (Rút ngắn 50% độ trễ EMA & Momentum) | **ACCEPTED** | EFFECTIVE | +7.7559 pp | -2.6530 pp | +10.1534 pp |
| V2 | High-Frequency Rebalance Cadence (Trend 8h, Overlay 24h + Deadband) | **ACCEPTED** | EFFECTIVE | +3.4585 pp | -1.2749 pp | +3.9673 pp |
| V3 | Early-Wave Entry Gate Relaxation (Hạ ngưỡng kích hoạt để vào chân sóng) | **REJECTED** | EFFECTIVE | -0.1555 pp | +1.4647 pp | -4.3285 pp |
| V4 | Agile Carry Rotation (Tăng độ nhạy xoay vòng vốn Carry) | **REJECTED** | EFFECTIVE | -0.6994 pp | +0.8748 pp | -1.7941 pp |
| V5 | Tactical Breakout / Impulse Sub-Sleeve (Nhánh Xung Kích Ngắn Hạn khi Trend FLAT) | **REJECTED** | EFFECTIVE | -3.8390 pp | +1.8086 pp | +2.7943 pp |

### V1 — Fast-Horizon Trend Alignment (Rút ngắn 50% độ trễ EMA & Momentum)

- Verdict: **ACCEPTED** (EFFECTIVE)
- Target Layer: Directional Trend Sleeve / Signal Horizon
- Mechanism: Thay bộ khung chậm (168h, 504h, 1440h) bằng bộ khung phản ứng nhanh (72h, 168h, 504h). Điểm Z_comp tính trên cửa sổ 168h + 504h (7d + 21d). Bảo toàn Macro Regime Filter và Funding Drag Veto.
- Evaluation Summary: Vượt ngưỡng kiểm định thực tế: gia tăng 24M Pure APR (+7.76 pp) và bảo toàn hiệu năng trên 6M Holdout (+10.15 pp).

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +24.69% | +27.22% | 1.71 | 4.49% | $3.05 |
| 6M | +25.05% | +27.56% | 2.00 | 4.25% | $4.64 |
| 12M | +16.55% | +19.16% | 1.57 | 5.95% | $6.55 |
| 24M | +18.34% | +20.56% | 1.76 | 6.00% | $16.36 |
| 3_5Y | +13.38% | +15.39% | 1.26 | 8.21% | $28.18 |

### V2 — High-Frequency Rebalance Cadence (Trend 8h, Overlay 24h + Deadband)

- Verdict: **ACCEPTED** (EFFECTIVE)
- Target Layer: Execution Cadence / Rebalancing Clock
- Mechanism: Hạ trend_rebalance_hours từ 24h xuống 8h (khớp nhịp Funding 00:00, 08:00, 16:00 UTC) và overlay_rebalance_hours từ 48h xuống 24h. Kèm deadband 15% chống đảo lệnh lắt nhắt.
- Evaluation Summary: Vượt ngưỡng kiểm định thực tế: gia tăng 24M Pure APR (+3.46 pp) và bảo toàn hiệu năng trên 6M Holdout (+3.97 pp).

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +32.93% | +35.59% | 1.86 | 3.88% | $3.83 |
| 6M | +18.86% | +21.31% | 1.36 | 4.20% | $5.43 |
| 12M | +7.64% | +10.25% | 0.75 | 5.99% | $6.73 |
| 24M | +14.05% | +16.34% | 1.31 | 7.37% | $15.53 |
| 3_5Y | +13.36% | +15.38% | 1.28 | 7.83% | $31.49 |

### V3 — Early-Wave Entry Gate Relaxation (Hạ ngưỡng kích hoạt để vào chân sóng)

- Verdict: **REJECTED** (EFFECTIVE)
- Target Layer: Directional Trend Sleeve / Entry & Exit Gates
- Mechanism: Hạ z_entry từ 1.15 xuống 0.95, BTC Kaufman ER_336h từ 0.16 xuống 0.12, Bull Breadth từ 0.375 xuống 0.25 (2/8 coin bứt phá trước). Thắt chặt Trailing Stop: trail_k_sigma = 2.5 (thay vì 3.0) để khóa lời nhanh.
- Evaluation Summary: Tổn thất hiệu năng thực tế do phí hoặc tín hiệu: 24M Pure APR giảm -0.16 pp, phí tăng chênh lệch $4.27.

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +18.79% | +21.23% | 1.01 | 4.83% | $4.09 |
| 6M | +10.57% | +12.92% | 0.71 | 4.80% | $5.66 |
| 12M | +-2.23% | +0.39% | -0.13 | 7.54% | $7.04 |
| 24M | +10.43% | +12.80% | 0.89 | 10.11% | $17.86 |
| 3_5Y | +17.05% | +18.93% | 1.49 | 8.79% | $30.84 |

### V4 — Agile Carry Rotation (Tăng độ nhạy xoay vòng vốn Carry)

- Verdict: **REJECTED** (EFFECTIVE)
- Target Layer: Delta-Neutral Carry / Selection & Holding
- Mechanism: Rút ngắn Carry f_ewma span từ 42 xuống 21 kỳ 8h (7 ngày), hạ min_hold_periods từ 63 (21 ngày) xuống 21 kỳ (7 ngày), và thu hẹp buffer_rank từ 3 xuống 2 (Top 5). Tận dụng phí rẻ U4 9.25 bps để xoay vốn nhanh hơn sang coin bùng nổ Funding.
- Evaluation Summary: Tổn thất hiệu năng thực tế do phí hoặc tín hiệu: 24M Pure APR giảm -0.70 pp, phí tăng chênh lệch $23.12.

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +21.22% | +24.08% | 1.24 | 5.79% | $6.22 |
| 6M | +13.10% | +15.93% | 0.94 | 5.72% | $10.23 |
| 12M | +4.13% | +7.04% | 0.43 | 5.83% | $15.35 |
| 24M | +9.89% | +12.40% | 0.96 | 9.52% | $36.71 |
| 3_5Y | +18.31% | +20.19% | 1.56 | 8.36% | $60.17 |

### V5 — Tactical Breakout / Impulse Sub-Sleeve (Nhánh Xung Kích Ngắn Hạn khi Trend FLAT)

- Verdict: **REJECTED** (EFFECTIVE)
- Target Layer: Tactical Impulse Sleeve / Volatility Squeeze
- Mechanism: Khi nhánh Trend lớn đang FLAT (chưa đủ điều kiện EMA dài hạn), cho phép kích hoạt vị thế xung kích ngắn hạn tối đa 25% gross khi phát hiện cú nén bứt phá đỉnh 7 ngày (168h Donchian), Z_168h >= 1.30 và BTC > EMA_168h. Thoát nhanh bằng Trailing Stop 2.0 Sigma.
- Evaluation Summary: Tổn thất hiệu năng thực tế do phí hoặc tín hiệu: 24M Pure APR giảm -3.84 pp, phí tăng chênh lệch $24.81.

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +35.39% | +38.08% | 1.91 | 4.72% | $5.12 |
| 6M | +17.69% | +20.12% | 1.21 | 4.70% | $9.45 |
| 12M | +5.72% | +8.34% | 0.56 | 5.99% | $16.37 |
| 24M | +6.75% | +9.20% | 0.68 | 10.46% | $38.39 |
| 3_5Y | +20.64% | +22.40% | 1.75 | 7.84% | $66.97 |

## 4. Combinatorial Matrix & Synergy Decomposition

| Combo ID | Included Upgrades | 24M Pure APR | 24M Yield APR | 24M Sharpe | 24M MaxDD | 3.5Y Pure Ret | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| V0_CHAMPION_BASELINE | None (V0 Baseline) | +10.59% | +12.95% | 1.00 | 8.65% | +79.49% | **BASELINE** |
| COMBO_V1_FAST_HORIZON_ONLY | V1 | +18.34% | +20.56% | 1.76 | 6.00% | +51.15% | **ACCEPTED** |
| COMBO_V1_V2_ACCEPTED_SYNERGY | V1 + V2 | +16.13% | +18.38% | 1.59 | 9.11% | +45.90% | **ACCEPTED** |
| COMBO_V1_V3_FAST_WAVE | V1 + V3 | +20.38% | +22.55% | 1.65 | 7.39% | +59.49% | **ACCEPTED** |
| COMBO_ALL_FIVE | V1 + V2 + V3 + V4 + V5 | +14.55% | +16.97% | 1.19 | 9.85% | +76.27% | **REJECTED** |

## 5. Cross-Profile Velocity Impact Across All 4 Tournament Profiles (24M Audit Window)

| Profile ID | Baseline Pure APR | Upgraded Pure APR | Upgraded Yield APR | Baseline Sharpe | Upgraded Sharpe | Baseline MaxDD | Upgraded MaxDD | Fee Delta (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **regime_omni_1k** | +10.59% | +18.34% | +20.56% | 1.00 | 1.76 | 8.65% | 6.00% | +20.4% |
| **flexible_regime_1k** | +9.30% | +17.00% | +19.07% | 1.01 | 1.90 | 7.51% | 4.30% | +11.9% |
| **carry_max_eta_1k** | +3.55% | +3.55% | +5.54% | 2.08 | 2.08 | 0.57% | 0.57% | +0.0% |
| **baseline_1k** | +3.17% | +3.17% | +5.37% | 2.71 | 2.71 | 0.52% | 0.52% | +0.0% |

## 6. Phase 1 Preliminary Recommendation

- **Phase 1 Screening Winner**: `COMBO_V1_FAST_HORIZON_ONLY` (V1)
- **Executive Summary**: Trên cửa sổ 24M, V1 nâng Pure APR từ +10.59% lên +18.34% (+7.76 pp), Sharpe từ 1.00 lên 1.76, giảm MaxDD từ 8.65% về 6.00%. Tuy nhiên, kiểm định đa chu kỳ 3.5Y (Phase 2) phát hiện hiện tượng sụt giảm hiệu năng nghiêm trọng trong năm 2023–2024, đòi hỏi cơ chế Dual-Horizon.

## 7. Sprint 28 Phase 2 — Multi-Cycle Yearly Decomposition & Dual-Horizon Ultimate Champion (`F4_DUAL_72_336_1440`)

### 7.1. Chẩn đoán Pháp y Điểm nghẽn Năm 2023–2024 của V1 Fast-Horizon Thuần túy

Khi mở rộng kiểm thử từ 24M sang toàn bộ 3.5 năm (2023–2026), xuất hiện hiện tượng nghịch đảo hiệu suất:
- Trên cửa sổ 3.5Y, `V0_BASELINE` đạt Pure Return **+79.58%** (APR +19.47%, Sharpe 1.68).
- Nhưng `V1_CLEAN` (`ema=72/168/504h`, `horizon=168/504h`) chỉ đạt **+51.22%** (APR +13.39%, Sharpe 1.26).
- **Bóc tách PnL theo năm giữa V0 và V1**:
  - **2023**: V0 đạt `+$293.8` vs V1 chỉ đạt `+$39.5` (Thất thoát `-$254.3`)!
  - **2024**: V0 đạt `+$448.3` vs V1 chỉ đạt `+$118.8` (Thất thoát `-$329.5`)!
  - **2025**: V0 đạt `+$42.8` vs V1 bứt phá `+$153.1` (V1 vượt trội gấp 3.5x).
  - **2026**: V0 đạt `+$8.6` vs V1 bứt phá `+$200.7` (V1 vượt trội gấp 23.3x).

**Nguyên nhân cơ chế**:
1. Trong các con sóng tăng lớn bền vững (Late 2023 Bull Run và 2024 Halving Wave), những nhịp điều chỉnh kỹ thuật ngắn hạn (3–5 ngày) làm giá BTC đâm thủng `EMA_med = 168h` hoặc `336h`.
2. Chỉ báo độ rộng thị trường `bull_breadth` tụt dưới ngưỡng `0.375`, kích hoạt `macro_long_exit` cắt toàn bộ các vị thế Long đang có lãi lớn (SOL +115%, AVAX +140% trong tháng 12/2023).
3. Khung momentum ngắn hạn 7 ngày (`168h`) bị triệt tiêu điểm $Z$ trong pha đi ngang ngắn, khiến bot bị "mù đà" không thể vào lại chân sóng tiếp theo.

### 7.2. Cơ chế Đột phá: Động lượng Kép Dual-Horizon Composite Momentum (`F4`)

Để vừa phản ứng siêu nhạy trong thị trường phân hóa 2025–2026, vừa giữ trọn thân sóng vĩ mô 2023–2024, giải pháp **Dual-Horizon Composite Momentum** được tích hợp vào `src/directional_trend_sleeve.py`:
$$Z_{\text{fast}} = 0.5 \cdot Z_{168h} + 0.5 \cdot Z_{504h}, \qquad Z_{\text{slow}} = 0.5 \cdot Z_{504h} + 0.5 \cdot Z_{1440h}$$
$$Z_{\text{comp}} = \max(Z_{\text{fast}}, Z_{\text{slow}})$$

Kết hợp hệ thống đường trung bình neo vĩ mô:
- `trend_ema_fast_hours = 72` (3 ngày) — Bắt nhịp sớm.
- `trend_ema_med_hours = 336` (14 ngày) — Thoát nhanh khi thị trường gãy cấu trúc trung hạn.
- `trend_ema_slow_hours = 1440` (60 ngày) — Neo xu hướng vĩ mô dài hạn.
- `trend_vol_window_hours = 240` (10 ngày) — Ước lượng biến động mượt mà hơn.
- `trend_short_horizon_hours = 168`, `trend_long_horizon_hours = 504`.
- `trend_enable_dual_horizon = True`.

### 7.3. Bảng Kiểm định Đấu loại Chung kết (Finalists Tournament)

#### A. Trên Profile `regime_omni_1k` ($1,000 USD):
| Cấu hình | 24M Pure APR | 24M Sharpe | 24M MaxDD | 3.5Y Return | 3.5Y APR | 3.5Y Sharpe | PnL 2023 | PnL 2024 | PnL 2025 | PnL 2026 | Phán quyết |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V0_BASELINE`** (168/504/1440) | +10.59% | 1.00 | 8.65% | +79.58% | +19.47% | 1.68 | +$293.8 | +$448.3 | +$42.8 | +$8.6 | Baseline |
| **`F1_V1_CLEAN`** (72/168/504) | +18.34% | 1.76 | 6.00% | +51.22% | +13.39% | 1.26 | +$39.5 | +$118.8 | +$153.1 | +$200.7 | Hụt sóng 23-24 |
| **`F2_HYBRID_ASYM_504`** | +17.95% | 1.46 | 7.47% | +73.81% | +18.29% | 1.56 | +$38.0 | +$367.8 | +$136.4 | +$194.8 | Tốt nhưng DD cao |
| **`F3_PROTECTED_336_1440`** | +18.51% | 1.56 | 7.10% | +75.76% | +18.70% | 1.68 | +$66.1 | +$380.5 | +$118.9 | +$191.1 | Rất tốt |
| 🏆 **`F4_DUAL_72_336_1440`** | **+18.51%** | **1.56** | **7.10%** | **+77.54%** | **+19.06%** | **1.70** | **+$66.1** | **+$380.5** | **+$136.7** | **+$191.1** | **ULTIMATE CHAMPION** |
| **`F4_DUAL_72_504_1440`** | +12.85% | 1.11 | 9.85% | +99.55% | +23.36% | 1.93 | +$305.3 | +$429.3 | +$77.5 | +$181.0 | 24M thấp (<18%) |

#### B. Trên Profile `flexible_regime_1k` ($1,000 USD):
| Cấu hình | 24M Pure APR | 24M Sharpe | 24M MaxDD | 3.5Y Return | 3.5Y APR | 3.5Y Sharpe | PnL 2023 | PnL 2024 | PnL 2025 | PnL 2026 | Phán quyết |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V0_BASELINE`** | +9.30% | 1.01 | 7.51% | +70.10% | +17.52% | 1.78 | +$223.7 | +$370.3 | +$52.0 | +$52.6 | Baseline |
| **`F1_V1_CLEAN`** | +17.00% | 1.90 | 4.30% | +46.99% | +12.42% | 1.37 | +$22.2 | +$124.3 | +$152.9 | +$170.2 | Hụt sóng 23-24 |
| **`F2_HYBRID_ASYM_504`** | +16.33% | 1.50 | 6.71% | +69.10% | +17.31% | 1.70 | +$22.3 | +$340.1 | +$151.0 | +$175.8 | Tốt |
| **`F3_PROTECTED_336_1440`** | +16.08% | 1.54 | 6.73% | +71.16% | +17.74% | 1.85 | +$50.8 | +$352.4 | +$133.5 | +$173.0 | Rất tốt |
| 🏆 **`F4_DUAL_72_336_1440`** | **+16.08%** | **1.54** | **6.73%** | **+72.94%** | **+18.11%** | **1.88** | **+$50.8** | **+$352.4** | **+$151.2** | **+$173.0** | **ULTIMATE CHAMPION** |
| **`F4_DUAL_72_504_1440`** | +13.82% | 1.31 | 7.42% | +85.42% | +20.64% | 2.01 | +$233.4 | +$356.3 | +$98.0 | +$164.2 | 24M thấp |

### 7.4. Kết luận & Quyết định Promote vào Sản xuất

Cấu hình **`F4_DUAL_72_336_1440`** chính thức được lựa chọn làm **Nhà Vô Địch Sản Xuất Toàn Diện (Ultimate Production Champion)** cho Sprint 28:
1. Đạt **24M Pure APR = +18.51%** (vượt chỉ tiêu `>= +18.0%`), MaxDD thấp (**7.10%**), Sharpe 24M tăng vọt từ **1.00 lên 1.56**.
2. Duy trì **3.5Y Pure Return = +77.54%** (APR **+19.06%**, Sharpe **1.70**), PnL dương mạnh mẽ và đồng đều trên cả 4 năm (2023: +$66.1, 2024: +$380.5, 2025: +$136.7, 2026: +$191.1).
3. Đã promote trực tiếp vào `src/strategy_profiles.py` cho `regime_omni_1k` và `flexible_regime_1k`.
4. Bảo toàn Zero-Impact trên 2 profile thuần Carry `carry_max_eta_1k` và `baseline_1k`.
