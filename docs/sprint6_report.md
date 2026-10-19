# Sprint 6 — Báo cáo nghiệm thu

**Cửa sổ dữ liệu:** Phân đoạn liền mạch: DEV (2023-05-07 → 2026-04-21) & HOLDOUT (2026-04-21 → 2026-10-18)
**Khung thời gian:** 1h | **Danh sách symbol (4):** BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT, BNB/USDT:USDT
**Benchmark so sánh:** BTC/USDT:USDT Buy-and-Hold

## Định nghĩa kịch bản chi phí:
- `zero`: perp_fee = 0.0, spot_fee = 0.0, slippage = 0.0.
- `maker_optimistic`: perp_fee = 0.0002 (2 bps), spot_fee = 0.0002 (2 bps), slippage = 0.0001 (1 bps).
- `real_taker`: perp_fee = 0.0005 (5 bps), spot_fee = 0.00075 (7.5 bps), slippage = 0.0002 (2 bps).

## 1. Kết quả chiến lược Funding Carry (Delta-Neutral)
*Tổng số cấu hình thử nghiệm K = 6*

### Phân đoạn DEV (Kịch bản real_taker):
| segment | structure | fee_scenario | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | coinm_1leg | real_taker | 12.0202 | 3.9107 | 12.2180 | 0.0830 | 1.0000 | 0.0227 | 0.0000 | 0.0001 | 17.3395 | 1.3034 | 3.8334 | 46.7994 | 4.3798 | 54.7299 | 15.6420 |
| dev | spot_perp_2leg | real_taker | 6.9417 | 2.2941 | 9.4240 | 0.1679 | 1.0000 | 0.0227 | 0.0000 | 0.0001 | 13.9963 | 1.4643 | 2.2658 | 30.7279 | 6.7215 | 54.7299 | 10.1841 |

### Phân đoạn DEV (Toàn bộ cấu hình):
| segment | structure | fee_scenario | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | coinm_1leg | zero | 13.1614 | 4.2673 | 13.7930 | 0.0247 | 1.0000 | 0.0227 | 0.0000 | 0.0001 | 19.1761 | 1.1210 | 4.1765 | 46.9765 | 0.0000 | 54.7299 | 15.7165 |
| dev | coinm_1leg | maker_optimistic | 12.6710 | 4.1144 | 13.2125 | 0.0344 | 1.0000 | 0.0227 | 0.0000 | 0.0001 | 18.4875 | 1.2055 | 4.0295 | 46.9005 | 1.8821 | 54.7299 | 15.6845 |
| dev | coinm_1leg | real_taker | 12.0202 | 3.9107 | 12.2180 | 0.0830 | 1.0000 | 0.0227 | 0.0000 | 0.0001 | 17.3395 | 1.3034 | 3.8334 | 46.7994 | 4.3798 | 54.7299 | 15.6420 |
| dev | spot_perp_2leg | zero | 8.6697 | 2.8498 | 13.7659 | 0.0165 | 1.0000 | 0.0227 | 0.0000 | 0.0000 | 19.1302 | 1.1220 | 2.8083 | 30.9098 | 0.0000 | 54.7299 | 10.2602 |
| dev | spot_perp_2leg | maker_optimistic | 8.0384 | 2.6475 | 12.4726 | 0.0477 | 1.0000 | 0.0227 | 0.0000 | 0.0000 | 17.6197 | 1.2810 | 2.6111 | 30.8435 | 2.4558 | 54.7299 | 10.2324 |
| dev | spot_perp_2leg | real_taker | 6.9417 | 2.2941 | 9.4240 | 0.1679 | 1.0000 | 0.0227 | 0.0000 | 0.0001 | 13.9963 | 1.4643 | 2.2658 | 30.7279 | 6.7215 | 54.7299 | 10.1841 |

### Phân đoạn HOLDOUT (Kịch bản real_taker):
| segment | structure | fee_scenario | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | coinm_1leg | real_taker | 0.4770 | 0.9693 | 20.6188 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 15.7514 | 1.7045 | 0.9640 | 7.8017 | 0.0000 | 100.0000 | 0.0000 |
| holdout | spot_perp_2leg | real_taker | 0.3180 | 0.6458 | 20.6176 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 15.7501 | 1.7052 | 0.6432 | 5.1225 | 0.0000 | 100.0000 | 0.0000 |

### Phân đoạn HOLDOUT (Toàn bộ cấu hình):
| segment | structure | fee_scenario | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | coinm_1leg | zero | 0.4769 | 0.9692 | 20.6188 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 15.7514 | 1.7045 | 0.9638 | 7.8312 | 0.0000 | 100.0000 | 0.0000 |
| holdout | coinm_1leg | maker_optimistic | 0.4769 | 0.9692 | 20.6188 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 15.7514 | 1.7045 | 0.9639 | 7.8186 | 0.0000 | 100.0000 | 0.0000 |
| holdout | coinm_1leg | real_taker | 0.4770 | 0.9693 | 20.6188 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 15.7514 | 1.7045 | 0.9640 | 7.8017 | 0.0000 | 100.0000 | 0.0000 |
| holdout | spot_perp_2leg | zero | 0.3179 | 0.6456 | 20.6176 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 15.7501 | 1.7052 | 0.6431 | 5.1528 | 0.0000 | 100.0000 | 0.0000 |
| holdout | spot_perp_2leg | maker_optimistic | 0.3180 | 0.6457 | 20.6176 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 15.7501 | 1.7052 | 0.6431 | 5.1418 | 0.0000 | 100.0000 | 0.0000 |
| holdout | spot_perp_2leg | real_taker | 0.3180 | 0.6458 | 20.6176 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 15.7501 | 1.7052 | 0.6432 | 5.1225 | 0.0000 | 100.0000 | 0.0000 |

### Đánh giá nghiệm thu GO / NO-GO (Hướng Carry):
- **Cấu trúc `coinm_1leg` (real_taker)**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): ĐẠT (alpha_t=17.34, beta=0.0001)
  - Điều kiện 2 (Sharpe >= 1.50 [DEV]): ĐẠT (Sharpe=12.22)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): ĐẠT (DSR=1.0000, SR*=0.0227)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): ĐẠT (NetRet=0.48%, Sharpe=20.62, MaxDD=0.00%)
  - **Kết luận:** `GO`
- **Cấu trúc `spot_perp_2leg` (real_taker)**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): ĐẠT (alpha_t=14.00, beta=0.0001)
  - Điều kiện 2 (Sharpe >= 1.50 [DEV]): ĐẠT (Sharpe=9.42)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): ĐẠT (DSR=1.0000, SR*=0.0227)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): ĐẠT (NetRet=0.32%, Sharpe=20.62, MaxDD=0.00%)
  - **Kết luận:** `GO`

## 2. Kết quả chiến lược Cross-Sectional Market-Neutral
*Tổng số cấu hình thử nghiệm K = 12*

### Phân đoạn DEV (Toàn bộ cấu hình):
| segment | mode | lookback_hours | rebalance_hours | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | momentum | 72 | 24 | -8.8831 | -3.0951 | 0.0385 | 61.0913 | 0.0116 | 0.0145 | -0.0000 | 0.0269 | -0.0024 | 2.7226 | -0.0401 | 15.2354 | 377.9939 | 99.6258 | 976.0000 |
| dev | momentum | 72 | 48 | -20.7726 | -7.5677 | -0.1240 | 53.0414 | 0.0054 | 0.0145 | -0.0000 | 0.0109 | -0.2425 | 1.1404 | -4.0892 | 13.8192 | 187.8020 | 99.5332 | 649.0000 |
| dev | momentum | 168 | 24 | -5.5360 | -1.9063 | 0.0763 | 46.0524 | 0.0138 | 0.0145 | 0.0000 | 0.0131 | 0.1006 | 1.2784 | 1.6206 | 15.1388 | 237.4331 | 99.3480 | 662.0000 |
| dev | momentum | 168 | 48 | -8.7255 | -3.0385 | 0.0315 | 50.8878 | 0.0113 | 0.0145 | 0.0000 | 0.0085 | 0.0335 | 0.9239 | 0.5211 | 15.9950 | 154.9323 | 99.3480 | 453.0000 |
| dev | momentum | 336 | 24 | 42.4447 | 12.7007 | 0.5615 | 32.1539 | 0.0851 | 0.0145 | 0.0000 | 0.0045 | 0.9885 | 0.4447 | 15.8284 | 17.3627 | 180.3403 | 98.6998 | 446.0000 |
| dev | momentum | 336 | 48 | 78.9677 | 21.7386 | 0.8360 | 35.3852 | 0.1842 | 0.0145 | 0.0000 | 0.0019 | 1.4708 | 0.1997 | 23.5905 | 18.0902 | 147.5739 | 98.6998 | 316.0000 |
| dev | reversal | 72 | 24 | -78.2587 | -40.2927 | -1.6201 | 82.2093 | 0.0000 | 0.0145 | -0.0001 | -0.0265 | -2.7996 | -2.6882 | -46.1575 | -2.8444 | 90.6567 | 99.6258 | 976.0000 |
| dev | reversal | 72 | 48 | -60.4466 | -26.9091 | -0.9288 | 63.8402 | 0.0000 | 0.0145 | -0.0000 | -0.0105 | -1.5855 | -1.1027 | -26.6341 | -5.8579 | 92.0811 | 99.5332 | 649.0000 |
| dev | reversal | 168 | 24 | -67.1603 | -31.3626 | -1.1682 | 76.8050 | 0.0000 | 0.0145 | -0.0000 | -0.0130 | -2.0561 | -1.2735 | -32.9459 | -5.5994 | 78.6336 | 99.3480 | 662.0000 |
| dev | reversal | 168 | 48 | -54.0862 | -23.1313 | -0.7912 | 65.4467 | 0.0001 | 0.0145 | -0.0000 | -0.0085 | -1.4155 | -0.9263 | -21.9538 | -6.0811 | 64.8537 | 99.3480 | 453.0000 |
| dev | reversal | 336 | 24 | -70.4491 | -33.7673 | -1.3015 | 72.7221 | 0.0000 | 0.0145 | -0.0000 | -0.0044 | -2.3162 | -0.4419 | -36.9320 | -5.9770 | 55.3209 | 98.6998 | 446.0000 |
| dev | reversal | 336 | 48 | -71.6826 | -34.7149 | -1.3644 | 73.3590 | 0.0000 | 0.0145 | -0.0000 | -0.0023 | -2.4064 | -0.2341 | -38.5277 | -5.2017 | 38.2459 | 98.6998 | 316.0000 |

### Phân đoạn HOLDOUT (Toàn bộ cấu hình):
| segment | mode | lookback_hours | rebalance_hours | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | momentum | 72 | 24 | -2.6076 | -5.2156 | -0.2507 | 13.0728 | 0.0098 | 0.0329 | -0.0000 | 0.0161 | -0.1934 | 0.8672 | -4.4048 | 2.5398 | 61.5789 | 100.0000 | 159.0000 |
| holdout | momentum | 72 | 48 | -6.6704 | -13.0599 | -0.8177 | 12.8678 | 0.0032 | 0.0329 | -0.0000 | 0.0009 | -0.5884 | 0.0463 | -12.7942 | 2.3037 | 33.2777 | 100.0000 | 115.0000 |
| holdout | momentum | 168 | 24 | 4.0348 | 8.3493 | 0.5655 | 13.0233 | 0.0383 | 0.0329 | 0.0000 | 0.0438 | 0.3341 | 1.9723 | 8.4395 | 2.5237 | 38.7353 | 100.0000 | 108.0000 |
| holdout | momentum | 168 | 48 | 6.0242 | 12.5911 | 0.7910 | 12.8863 | 0.0533 | 0.0329 | 0.0000 | 0.0394 | 0.5162 | 1.9518 | 12.3995 | 2.6665 | 24.9670 | 100.0000 | 73.0000 |
| holdout | momentum | 336 | 24 | 14.8836 | 32.4834 | 1.8140 | 10.9571 | 0.1866 | 0.0329 | 0.0000 | 0.0375 | 1.2966 | 2.0968 | 28.6245 | 2.8945 | 27.4958 | 100.0000 | 68.0000 |
| holdout | momentum | 336 | 48 | 18.3670 | 40.7542 | 2.2052 | 8.6466 | 0.2691 | 0.0329 | 0.0000 | 0.0305 | 1.5632 | 1.6752 | 34.8095 | 3.0157 | 21.0153 | 100.0000 | 45.0000 |
| holdout | reversal | 72 | 24 | -18.8675 | -34.5500 | -2.5372 | 23.4557 | 0.0000 | 0.0329 | -0.0000 | -0.0143 | -1.8120 | -0.7662 | -40.7630 | -0.4742 | 14.7689 | 100.0000 | 159.0000 |
| holdout | reversal | 72 | 48 | -9.8769 | -19.0086 | -1.2729 | 12.7268 | 0.0011 | 0.0329 | -0.0000 | 0.0005 | -0.9246 | 0.0276 | -19.8754 | -0.9766 | 16.3164 | 100.0000 | 115.0000 |
| holdout | reversal | 168 | 24 | -18.4865 | -33.9253 | -2.4062 | 23.6401 | 0.0000 | 0.0329 | -0.0000 | -0.0431 | -1.5655 | -1.9355 | -39.1072 | -0.9335 | 12.8284 | 100.0000 | 108.0000 |
| holdout | reversal | 168 | 48 | -16.0188 | -29.8071 | -2.0250 | 20.5675 | 0.0001 | 0.0329 | -0.0000 | -0.0385 | -1.3922 | -1.9026 | -33.1383 | -1.0138 | 10.4510 | 100.0000 | 73.0000 |
| holdout | reversal | 336 | 24 | -21.8803 | -39.3833 | -3.0085 | 24.8895 | 0.0000 | 0.0329 | -0.0001 | -0.0366 | -2.1796 | -2.0429 | -47.9442 | -0.9964 | 8.4346 | 100.0000 | 68.0000 |
| holdout | reversal | 336 | 48 | -21.6812 | -39.0696 | -3.0021 | 22.4604 | 0.0000 | 0.0329 | -0.0001 | -0.0308 | -2.1381 | -1.6896 | -47.5766 | -0.8671 | 5.4464 | 100.0000 | 45.0000 |

### Đánh giá nghiệm thu GO / NO-GO (Hướng Cross-Sectional):
- **Cấu hình `momentum` lookback=72h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=-0.00, beta=0.0269)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=0.04)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0116, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=-2.61%, Sharpe=-0.25, MaxDD=13.07%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=72h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=-0.24, beta=0.0109)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=-0.12)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0054, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=-6.67%, Sharpe=-0.82, MaxDD=12.87%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=168h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=0.10, beta=0.0131)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=0.08)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0138, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=4.03%, Sharpe=0.57, MaxDD=13.02%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=168h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=0.03, beta=0.0085)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=0.03)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0113, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=6.02%, Sharpe=0.79, MaxDD=12.89%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=336h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=0.99, beta=0.0045)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=0.56)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0851, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): ĐẠT (NetRet=14.88%, Sharpe=1.81, MaxDD=10.96%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=336h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=1.47, beta=0.0019)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=0.84)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.1842, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): ĐẠT (NetRet=18.37%, Sharpe=2.21, MaxDD=8.65%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=72h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=-2.80, beta=-0.0265)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=-1.62)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0000, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=-18.87%, Sharpe=-2.54, MaxDD=23.46%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=72h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=-1.59, beta=-0.0105)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=-0.93)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0000, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=-9.88%, Sharpe=-1.27, MaxDD=12.73%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=168h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=-2.06, beta=-0.0130)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=-1.17)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0000, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=-18.49%, Sharpe=-2.41, MaxDD=23.64%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=168h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=-1.42, beta=-0.0085)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=-0.79)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0001, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=-16.02%, Sharpe=-2.03, MaxDD=20.57%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=336h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=-2.32, beta=-0.0044)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=-1.30)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0000, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=-21.88%, Sharpe=-3.01, MaxDD=24.89%)
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=336h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20 [DEV]): KHÔNG ĐẠT (alpha_t=-2.41, beta=-0.0023)
  - Điều kiện 2 (Sharpe >= 1.00 [DEV]): KHÔNG ĐẠT (Sharpe=-1.36)
  - Điều kiện 3 (DSR >= 0.95 [DEV]): KHÔNG ĐẠT (DSR=0.0000, SR*=0.0145)
  - Điều kiện 4 (Holdout 180 ngày [HOLDOUT]): KHÔNG ĐẠT (NetRet=-21.68%, Sharpe=-3.00, MaxDD=22.46%)
  - **Kết luận:** `NO-GO`

---

## 3. KẾT LUẬN TỔNG HỢP & HẠN CHẾ

### 3.1. Bảng tổng hợp (kịch bản phí `real_taker`, cấu hình tiêu biểu mỗi hướng)
| Hướng / cấu hình | Net Return % (DEV) | Net Return % (HOLDOUT) | Sharpe (DEV) | DSR (DEV) | NW Alpha_t (DEV) | Beta (DEV) | MaxDD % (HOLDOUT) | Phán quyết |
|---|---|---|---|---|---|---|---|---|
| **Carry — coinm_1leg** | 12.02 | **+0.48** | 12.22 | 1.00 | 17.34 | 0.0001 | 0.00 | **GO** |
| **Carry — spot_perp_2leg** | 6.94 | **+0.32** | 9.42 | 1.00 | 14.00 | 0.0001 | 0.00 | **GO** |
| **XS — momentum 336h/48h** (tốt nhất) | 78.97 | +18.37 | 0.84 | 0.18 | 1.47 | 0.0019 | 8.65 | NO-GO |
| (tham chiếu) XS momentum 336h/24h | 42.44 | +14.88 | 0.56 | 0.09 | 0.99 | 0.0045 | 10.96 | NO-GO |
| (tham chiếu) XS reversal (mọi cấu hình) | −54…−78 | −10…−22 | −0.79…−1.62 | ~0 | −1.4…−2.8 | <0 | 12.7…24.9 | NO-GO |

- **Số lần thử**: Carry `K = 6` (2 structure × 3 kịch bản phí); Cross-Sectional `K = 12` (2 mode × 3 lookback × 2 rebalance).
- **HOLDOUT chỉ chạy 1 lần** (2026-04-21 → 2026-10-18) trên cùng một đường cong vốn liền mạch với DEV (không đóng cưỡng bức/reset ở biên).

### 3.2. Phán quyết theo tiêu chí đã khóa (`docs/go_no_go_sprint6.md`)
- **Funding Carry (delta-neutral): GO** — cả `coinm_1leg` và `spot_perp_2leg` đạt ĐỒNG THỜI 4 điều kiện ở kịch bản `real_taker`.
- **Cross-Sectional Market-Neutral: NO-GO** — trượt Điều kiện 1 (alpha_t < 2.5), Điều kiện 2 (Sharpe < 1.0) và Điều kiện 3 (DSR < 0.95) ở mọi cấu hình phía DEV; riêng momentum lookback 336h đạt Điều kiện 4 (holdout) nhưng không đủ.

### 3.3. HẠN CHẾ NGHIÊM TRỌNG (đọc kèm — ảnh hưởng đến độ tin cậy của GO)
Kết quả `GO` của Carry **tuân thủ đúng tiêu chí đã khóa**, nhưng rủi ro phương pháp đã được phản biện độc lập (`.agy/quick_pro_8d53d4f2.md`) và phải được nêu minh bạch:
1. **Bỏ qua basis risk mark-to-market (mức độ CAO)**: model delta-neutral lý tưởng hóa — `basis_noise` chỉ lấy mẫu 1 lần khi ĐÓNG lệnh; không nạp giá, không có PnL basis hàng giờ, notional cố định. → Biến động danh mục bị dồn về ≈0.32%/năm → **Sharpe 12.2 bị phóng đại**. Nếu mô hình hóa đúng basis vol thực tế (~6%/năm), Sharpe thực ước lượng **~0.5–1.0** → **sẽ TRƯỢT Điều kiện 2 (≥1.50)**. Nói cách khác: **GO về mặt kỹ thuật, nhưng KHÔNG đáng tin về mặt kinh tế.**
2. **Notional cố định & không giả lập ký quỹ/liquidation**: không phản ánh cấu trúc hợp đồng Coin-M/Perp thực; chưa mô phỏng margin call khi giá biến động mạnh.
3. **Kẹt vị thế khi funding chuyển âm kéo dài**: hysteresis + `min_hold_periods=3` có thể gánh funding âm nhiều kỳ.
4. **Giới hạn lô tối thiểu Coin-M (100 USD/coin)** & **phân mảnh ký quỹ 2 chân** → khó tái cân bằng với vốn 400 USDT.
5. **Annualization**: Sharpe năm hóa theo 8760 (nến 1h) trong khi funding chỉ 3 kỳ/ngày → nhất quán theo tiêu chí đã khóa nhưng cần lưu ý khi diễn giải.

### 3.4. Khuyến nghị
- **Không triển khai vốn thật cho Carry** dựa trên kết quả này. Muốn GO "thật", cần bổ sung mô hình **basis risk MTM** (nạp giá/ futures basis) và chạy lại như một Sprint mới với tiêu chí khóa mới — **KHÔNG sửa model/tiêu chí trong Sprint 6**.
- Cross-Sectional momentum lookback dài (336h) là ứng viên duy nhất có tín hiệu dương trên holdout; cần thêm bằng chứng (và vượt ĐK1–3) trước khi xem xét.

