# Sprint 27 — Upgrade Ablation & Combinatorial Study

- Study ID: `SPRINT_27_UPGRADE_ABLATION_AND_COMBO_STUDY` v2.0.0
- Generated at: `2026-11-02T14:35:29.235371+00:00`
- Capital base: `$1,000.00` (fixed, zero compounding leak)
- Engine: `scripts/run_1k_profile_tournament_backtest.py::evaluate_window` (real parquet re-simulation)
- Data: `30,265` hourly bars, `2023-05-07 00:00:00+00:00` -> `2026-10-19 00:00:00+00:00`
- Runtime: `406.3s`

## 1. Five Golden Quant Invariants

| Law | Status | Measured |
|---|---|---|
| INV-1 WARMUP_SATURATION | **PASSED** | Jaccard=1.0000, L1=0.000000 |
| INV-2 ZERO_COMPOUNDING_LEAK | **PASSED** | max_abs_diff=2.16e-12, sizing_capital_std=0.0 |
| INV-3 EPOCH_CLOCK_INVARIANCE | **PASSED** | diff_count=0, jaccard=1.0000 |
| INV-4 ACCOUNTING_IDENTITY | **PASSED** | baseline_full_error=2.16e-12 |
| INV-5 LOT_SIZE_NEUTRALITY | **PASSED** | end_imbalance=$0.2593, max_delta=$0.8568 |

**All invariants passed:** `True`

## 2. Baseline (U0) — Primary Profile `regime_omni_1k`

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +23.26% | +25.77% | 1.35 | 5.56% | 3.75 |
| 6M | +14.77% | +17.17% | 1.06 | 5.47% | 5.17 |
| 12M | +5.15% | +7.76% | 0.53 | 5.57% | 6.49 |
| 24M | +10.51% | +12.88% | 0.99 | 8.74% | 16.25 |
| 3_5Y | +18.49% | +20.39% | 1.63 | 7.76% | 29.13 |

## 3. Single-Upgrade Ablation (U1 -> U6)

| ID | Upgrade | Verdict | Materiality | 24M ΔAPR | 24M ΔDD | Holdout 6M ΔAPR |
|---|---|---|---|---:|---:|---:|
| U1 | Smart Lot Remainder Re-allocation (quét tiền lẻ Carry vào coin lot nhỏ) | **REJECTED** | EFFECTIVE | +0.0303 pp | -0.0729 pp | -0.0210 pp |
| U2 | Controlled Dynamic Basis Weight (w_basis=0.15 + clip +/-10 bps + veto discount <= -8 bps) | **REJECTED** | EFFECTIVE | +0.0031 pp | -0.0011 pp | +0.0174 pp |
| U3 | Emergency Negative Funding Exit (thoát sớm sau 12 kỳ nếu f_ewma < -1.0 bps/8h) | **ACCEPTED** | NEGLIGIBLE | +0.0000 pp | +0.0000 pp | +0.0000 pp |
| U4 | Maker-First Blended Cost 9.25 bps/chiều (entry_threshold giữ 0.45 bps) | **ACCEPTED** | EFFECTIVE | +0.0739 pp | -0.0861 pp | +0.1258 pp |
| U5 | Regime-Adaptive Split-Wallet Rebalancing (carry_max 70/30 + FLAT universal transfer) | **ACCEPTED** | EFFECTIVE | +0.0000 pp | +0.0000 pp | +0.0000 pp |
| U6 | Mở rộng Rổ Tài sản từ LIQUID_8 lên LIQUID_12 (bổ sung SUI, ADA, NEAR, DOT) | **REJECTED** | EFFECTIVE | +0.2798 pp | +1.4428 pp | -7.7329 pp |

### U1 — Smart Lot Remainder Re-allocation (quét tiền lẻ Carry vào coin lot nhỏ)

- Verdict: **REJECTED** (EFFECTIVE)
- Config: `{"u1_smart_lot_sweep": true, "u1_min_sweep_usd": 25.0, "u2_basis_weight": 0.0, "u2_basis_clip_bps": 10.0, "u2_discount_veto_bps": 0.0, "u3_emergency_exit_min_periods": 0, "u3_emergency_exit_funding_bps": -1.0, "u4_blended_carry_cost_bps": 0.0, "u4_entry_threshold_bps": 0.0, "u5_spot_wallet_usd": 0.0, "u5_futures_wallet_usd": 0.0, "u5_dynamic_transfer_flat": false, "enable_liquid_12_universe": false, "u6_liquid_12_universe": false}`
- Mechanism: Sau equal-weight + floor-to-lot, gom unallocated carry budget và sweep vào các symbol đang active có lot step nhỏ (DOGE/XRP/AVAX/SOL/LINK), không vượt carry_max_single_weight và không mở symbol ngoài top-k.
- Falsification reason: regime_omni_1k 6M: Pure APR giảm 0.0210 pp (14.770% -> 14.749%) | flexible_regime_1k 24M: Pure APR giảm 0.2144 pp (9.167% -> 8.952%) | carry_max_eta_1k 24M: Pure APR giảm 0.0132 pp (2.349% -> 2.336%)

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +23.04% | +25.47% | 1.34 | 5.54% | 3.91 |
| 6M | +14.75% | +17.12% | 1.07 | 5.45% | 5.34 |
| 12M | +5.20% | +7.73% | 0.54 | 5.54% | 6.70 |
| 24M | +10.54% | +12.86% | 1.00 | 8.66% | 16.59 |
| 3_5Y | +18.44% | +20.33% | 1.63 | 7.72% | 29.78 |

### U2 — Controlled Dynamic Basis Weight (w_basis=0.15 + clip +/-10 bps + veto discount <= -8 bps)

- Verdict: **REJECTED** (EFFECTIVE)
- Config: `{"u1_smart_lot_sweep": false, "u1_min_sweep_usd": 25.0, "u2_basis_weight": 0.15, "u2_basis_clip_bps": 10.0, "u2_discount_veto_bps": 8.0, "u3_emergency_exit_min_periods": 0, "u3_emergency_exit_funding_bps": -1.0, "u4_blended_carry_cost_bps": 0.0, "u4_entry_threshold_bps": 0.0, "u5_spot_wallet_usd": 0.0, "u5_futures_wallet_usd": 0.0, "u5_dynamic_transfer_flat": false, "enable_liquid_12_universe": false, "u6_liquid_12_universe": false}`
- Mechanism: Bật w_basis=0.15 với basis_dev bị kẹp +/-10 bps; veto không mở lệnh mới khi Perp discount sâu hơn 8 bps so với Spot.
- Falsification reason: carry_max_eta_1k 24M: Pure APR giảm 0.0543 pp (2.349% -> 2.295%) | baseline_1k 24M: Pure APR giảm 0.0109 pp (3.016% -> 3.005%) | baseline_1k 6M: Pure APR giảm 0.0490 pp (2.571% -> 2.522%)

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +23.34% | +25.85% | 1.35 | 5.57% | 3.71 |
| 6M | +14.79% | +17.19% | 1.06 | 5.49% | 5.13 |
| 12M | +5.16% | +7.77% | 0.53 | 5.58% | 6.44 |
| 24M | +10.52% | +12.88% | 0.99 | 8.73% | 16.21 |
| 3_5Y | +18.33% | +20.24% | 1.61 | 7.79% | 29.03 |

### U3 — Emergency Negative Funding Exit (thoát sớm sau 12 kỳ nếu f_ewma < -1.0 bps/8h)

- Verdict: **ACCEPTED** (NEGLIGIBLE)
- Config: `{"u1_smart_lot_sweep": false, "u1_min_sweep_usd": 25.0, "u2_basis_weight": 0.0, "u2_basis_clip_bps": 10.0, "u2_discount_veto_bps": 0.0, "u3_emergency_exit_min_periods": 12, "u3_emergency_exit_funding_bps": -1.0, "u4_blended_carry_cost_bps": 0.0, "u4_entry_threshold_bps": 0.0, "u5_spot_wallet_usd": 0.0, "u5_futures_wallet_usd": 0.0, "u5_dynamic_transfer_flat": false, "enable_liquid_12_universe": false, "u6_liquid_12_universe": false}`
- Mechanism: Giữ min_hold=63 cho dao động thường; nếu đã giữ >=12 kỳ và f_ewma < -1.0 bps/8h thì đóng vị thế trước hạn.
- Forensic note: f_ewma nội bộ dùng alpha nhanh (span=9, unshifted) trong khi carry_score dùng chu kỳ chậm (span=42, shifted); nhờ đó trên 24M/6M không cần kích hoạt (0 lần), nhưng trên 3.5Y (bear market 2023) đã cắt sớm được 1 đợt funding shock âm ở kỳ 12 (trước min_hold=63), giúp tăng +0.1602 pp APR.
- Falsification reason: Không có thay đổi kinh tế >= 0.01 pp trên 24M/6M ở mọi profile (upgrade không tạo giá trị).

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +23.26% | +25.77% | 1.35 | 5.56% | 3.75 |
| 6M | +14.77% | +17.17% | 1.06 | 5.47% | 5.17 |
| 12M | +5.15% | +7.76% | 0.53 | 5.57% | 6.49 |
| 24M | +10.51% | +12.88% | 0.99 | 8.74% | 16.25 |
| 3_5Y | +18.65% | +20.47% | 1.63 | 7.71% | 28.87 |

### U4 — Maker-First Blended Cost 9.25 bps/chiều (entry_threshold giữ 0.45 bps)

- Verdict: **ACCEPTED** (EFFECTIVE)
- Config: `{"u1_smart_lot_sweep": false, "u1_min_sweep_usd": 25.0, "u2_basis_weight": 0.0, "u2_basis_clip_bps": 10.0, "u2_discount_veto_bps": 0.0, "u3_emergency_exit_min_periods": 0, "u3_emergency_exit_funding_bps": -1.0, "u4_blended_carry_cost_bps": 9.25, "u4_entry_threshold_bps": 0.0, "u5_spot_wallet_usd": 0.0, "u5_futures_wallet_usd": 0.0, "u5_dynamic_transfer_flat": false, "enable_liquid_12_universe": false, "u6_liquid_12_universe": false}`
- Mechanism: Limit-chase chân Spot ~75% Maker, instant taker hedge chân Perp; blended carry cost 9.25 bps/chiều (18.5 bps khứ hồi), giữ nguyên entry threshold 0.45 bps. Biến thể hạ threshold 0.40 bps được chạy riêng trong parameter sweep.

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +23.47% | +25.98% | 1.36 | 5.56% | 3.32 |
| 6M | +14.90% | +17.30% | 1.07 | 5.47% | 4.59 |
| 12M | +5.23% | +7.84% | 0.54 | 5.56% | 5.69 |
| 24M | +10.59% | +12.95% | 1.00 | 8.65% | 13.59 |
| 3_5Y | +19.22% | +21.10% | 1.67 | 7.70% | 24.95 |

### U5 — Regime-Adaptive Split-Wallet Rebalancing (carry_max 70/30 + FLAT universal transfer)

- Verdict: **ACCEPTED** (EFFECTIVE)
- Config: `{"u1_smart_lot_sweep": false, "u1_min_sweep_usd": 25.0, "u2_basis_weight": 0.0, "u2_basis_clip_bps": 10.0, "u2_discount_veto_bps": 0.0, "u3_emergency_exit_min_periods": 0, "u3_emergency_exit_funding_bps": -1.0, "u4_blended_carry_cost_bps": 0.0, "u4_entry_threshold_bps": 0.0, "u5_spot_wallet_usd": 0.0, "u5_futures_wallet_usd": 0.0, "u5_dynamic_transfer_flat": true, "enable_liquid_12_universe": false, "u6_liquid_12_universe": false}`
- Mechanism: carry_max_eta_1k chuyển 78/22 -> 70/30 để giải phóng cap; các profile có trend sleeve mô phỏng Universal Transfer phần Spot nhàn rỗi sang Futures khi Trend=FLAT.

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +23.26% | +25.77% | 1.35 | 5.56% | 3.75 |
| 6M | +14.77% | +17.17% | 1.06 | 5.47% | 5.17 |
| 12M | +5.15% | +7.76% | 0.53 | 5.57% | 6.49 |
| 24M | +10.51% | +12.88% | 0.99 | 8.74% | 16.25 |
| 3_5Y | +18.49% | +20.39% | 1.63 | 7.76% | 29.13 |

### U6 — Mở rộng Rổ Tài sản từ LIQUID_8 lên LIQUID_12 (bổ sung SUI, ADA, NEAR, DOT)

- Verdict: **REJECTED** (EFFECTIVE)
- Config: `{"u1_smart_lot_sweep": false, "u1_min_sweep_usd": 25.0, "u2_basis_weight": 0.0, "u2_basis_clip_bps": 10.0, "u2_discount_veto_bps": 0.0, "u3_emergency_exit_min_periods": 0, "u3_emergency_exit_funding_bps": -1.0, "u4_blended_carry_cost_bps": 0.0, "u4_entry_threshold_bps": 0.0, "u5_spot_wallet_usd": 0.0, "u5_futures_wallet_usd": 0.0, "u5_dynamic_transfer_flat": false, "enable_liquid_12_universe": true, "u6_liquid_12_universe": true}`
- Mechanism: Mở rộng rổ tài sản từ 8 lên 12 đồng coin top thanh khoản có lot step nhỏ (SUI, ADA, NEAR, DOT). Carry, Momentum Overlay và Directional Trend chấm điểm và phân bổ trên toàn bộ 12 coin với cơ chế dynamic active universe (warmup >= 1440h).
- Falsification reason: regime_omni_1k 24M: MaxDD tăng 1.4428 pp (8.735% -> 10.178%) | regime_omni_1k 6M: Pure APR giảm 7.7329 pp (14.770% -> 7.037%) | regime_omni_1k 6M: MaxDD tăng 0.7503 pp (5.474% -> 6.224%) | flexible_regime_1k 24M: Pure APR giảm 0.5455 pp (9.167% -> 8.621%) | flexible_regime_1k 24M: MaxDD tăng 1.7552 pp (7.595% -> 9.350%) | flexible_regime_1k 6M: Pure APR giảm 5.5365 pp (11.481% -> 5.944%) | flexible_regime_1k 6M: MaxDD tăng 0.9027 pp (5.349% -> 6.251%) | carry_max_eta_1k 24M: Pure APR giảm 1.2167 pp (2.349% -> 1.132%) | carry_max_eta_1k 6M: Pure APR giảm 1.9348 pp (2.013% -> 0.078%) | baseline_1k 24M: Pure APR giảm 1.0226 pp (3.016% -> 1.993%) | baseline_1k 6M: Pure APR giảm 1.2640 pp (2.571% -> 1.307%)

| Window | Pure APR | Yield APR | Pure Sharpe | Max DD | Fees (USD) |
|---|---:|---:|---:|---:|---:|
| 3M | +20.49% | +22.91% | 1.00 | 6.13% | 5.51 |
| 6M | +7.04% | +9.55% | 0.48 | 6.22% | 10.16 |
| 12M | -4.07% | -1.44% | -0.26 | 8.32% | 15.69 |
| 24M | +10.79% | +13.09% | 1.01 | 10.18% | 32.44 |
| 3_5Y | +20.62% | +22.21% | 1.60 | 8.03% | 60.13 |

## 4. Combinatorial Matrix

| Combo | Included | 24M Pure APR | 24M Yield APR | 24M Sharpe | 24M MaxDD | 3.5Y Pure Return | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| C0_BASELINE_U0 | None (Baseline) | +10.51% | +12.88% | 0.99 | 8.74% | +74.74% | **BASELINE** |
| COMBO_A_STRUCTURAL_ALPHA | U1 + U2 + U3 | +10.54% | +12.84% | 1.00 | 8.65% | +78.42% | **REJECTED** |
| COMBO_B_EXECUTION_WALLET | U4 + U5 | +10.59% | +12.95% | 1.00 | 8.65% | +78.33% | **ACCEPTED** |
| BEST_COMBO | U4 + U5 | +10.59% | +12.95% | 1.00 | 8.65% | +78.33% | **ACCEPTED** |
| COMBO_C_CHAMPION_PLUS_U6 | U4 + U5 + U6 | +10.88% | +13.18% | 1.02 | 10.28% | +85.67% | **REJECTED** |

### Synergy decomposition (primary profile, 24M)

| Combo | Realised ΔAPR | Sum of single ΔAPR | Synergy |
|---|---:|---:|---:|
| COMBO_A_STRUCTURAL_ALPHA | +0.0244 pp | +0.0334 pp | -0.0090 pp |
| COMBO_B_EXECUTION_WALLET | +0.0739 pp | +0.0739 pp | +0.0000 pp |
| BEST_COMBO | +0.0739 pp | +0.0739 pp | +0.0000 pp |
| COMBO_C_CHAMPION_PLUS_U6 | +0.3664 pp | +0.3537 pp | +0.0127 pp |

## 5. Cross-Profile Champion Impact (24M)

| Profile | Baseline Pure APR | Best Combo Pure APR | Yield APR | Baseline MaxDD | Best Combo MaxDD | Fee Saved |
|---|---:|---:|---:|---:|---:|---:|
| regime_omni_1k | +10.51% | +10.59% | +12.95% | 8.74% | 8.65% | $2.67 |
| flexible_regime_1k | +9.17% | +9.30% | +11.52% | 7.59% | 7.51% | $2.89 |
| carry_max_eta_1k | +2.35% | +3.55% | +5.54% | 0.47% | 0.57% | $8.34 |
| baseline_1k | +3.02% | +3.17% | +5.37% | 0.52% | 0.52% | $3.09 |

## 6. U4 Entry-Threshold Sensitivity (24M / Holdout 6M)

| Profile | Window | U4 cost-only APR | U4 + 0.40 bps APR | Δ APR | Δ MaxDD |
|---|---|---:|---:|---:|---:|
| regime_omni_1k | 24M | +10.59% | +10.36% | -0.2243 pp | +0.3731 pp |
| regime_omni_1k | 6M | +14.90% | +14.86% | -0.0355 pp | +0.0008 pp |
| flexible_regime_1k | 24M | +9.30% | +9.04% | -0.2627 pp | +0.8642 pp |
| flexible_regime_1k | 6M | +11.62% | +12.35% | +0.7223 pp | -0.0163 pp |
| carry_max_eta_1k | 24M | +2.66% | +2.59% | -0.0657 pp | +0.0000 pp |
| carry_max_eta_1k | 6M | +2.36% | +2.16% | -0.1960 pp | -0.0190 pp |
| baseline_1k | 24M | +3.17% | +3.08% | -0.0839 pp | +0.0000 pp |
| baseline_1k | 6M | +2.72% | +2.73% | +0.0121 pp | -0.0014 pp |

## 7. Production Recommendation

- Selected: **BEST_COMBO**
- Included upgrades: `U4, U5`
- Conservative fallback: **U4_MAKER_FIRST_BLENDED_COST**
- Summary: Best Combo = U4 + U5

## 8. Zero-Interference Notes

- Live path `scripts/run_parallel_live_demo.py` does not import `src/upgrade_variants.py` or the study runner.
- All new engine parameters are opt-in; `upgrade=None` / `UpgradeFlags()` is a byte-for-byte Baseline no-op (tested).
- Dashboard Layer 5 is 100% read-only and reads the generated JSON.
