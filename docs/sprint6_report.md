# Sprint 6 — Báo cáo nghiệm thu

**Cửa sổ dữ liệu:** DEV (Phát triển / Walk-forward) (2023-05-07 → 2026-04-21)
**Khung thời gian:** 1h | **Danh sách symbol (4):** BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT, BNB/USDT:USDT
**Benchmark so sánh:** BTC/USDT:USDT Buy-and-Hold

## Định nghĩa kịch bản chi phí:
- `zero`: perp_fee = 0.0, spot_fee = 0.0, slippage = 0.0.
- `maker_optimistic`: perp_fee = 0.0002 (2 bps), spot_fee = 0.0002 (2 bps), slippage = 0.0001 (1 bps).
- `real_taker`: perp_fee = 0.0005 (5 bps), spot_fee = 0.00075 (7.5 bps), slippage = 0.0002 (2 bps).

## 1. Kết quả chiến lược Funding Carry (Delta-Neutral)
*Tổng số cấu hình thử nghiệm K = 6*

### Bảng tổng hợp kịch bản chi phí thực tế (real_taker):
| structure | fee_scenario | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coinm_1leg | real_taker | 12.0218 | 3.9110 | 12.2193 | 0.0830 | 1.0000 | 0.0050 | 0.0000 | 0.0001 | 17.3417 | 1.3042 | 3.8337 | 52.4701 | 4.3798 | 54.7008 | 15.6420 |
| spot_perp_2leg | real_taker | 6.9427 | 2.2943 | 9.4251 | 0.1679 | 1.0000 | 0.0080 | 0.0000 | 0.0001 | 13.9982 | 1.4651 | 2.2660 | 34.4939 | 6.7215 | 54.7008 | 10.1841 |

### Toàn bộ cấu hình (kèm kịch bản zero & maker_optimistic):
| structure | fee_scenario | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coinm_1leg | zero | 13.1630 | 4.2676 | 13.7942 | 0.0247 | 1.0000 | 0.0043 | 0.0000 | 0.0001 | 19.1783 | 1.1218 | 4.1768 | 52.6554 | 0.0000 | 54.7008 | 15.7165 |
| coinm_1leg | maker_optimistic | 12.6726 | 4.1147 | 13.2137 | 0.0344 | 1.0000 | 0.0043 | 0.0000 | 0.0001 | 18.4897 | 1.2064 | 4.0298 | 52.5758 | 1.8821 | 54.7008 | 15.6845 |
| coinm_1leg | real_taker | 12.0218 | 3.9110 | 12.2193 | 0.0830 | 1.0000 | 0.0050 | 0.0000 | 0.0001 | 17.3417 | 1.3042 | 3.8337 | 52.4701 | 4.3798 | 54.7008 | 15.6420 |
| spot_perp_2leg | zero | 8.6707 | 2.8500 | 13.7671 | 0.0165 | 1.0000 | 0.0043 | 0.0000 | 0.0000 | 19.1324 | 1.1228 | 2.8085 | 34.6846 | 0.0000 | 54.7008 | 10.2602 |
| spot_perp_2leg | maker_optimistic | 8.0394 | 2.6477 | 12.4738 | 0.0477 | 1.0000 | 0.0048 | 0.0000 | 0.0000 | 17.6219 | 1.2819 | 2.6113 | 34.6151 | 2.4558 | 54.7008 | 10.2324 |
| spot_perp_2leg | real_taker | 6.9427 | 2.2943 | 9.4251 | 0.1679 | 1.0000 | 0.0080 | 0.0000 | 0.0001 | 13.9982 | 1.4651 | 2.2660 | 34.4939 | 6.7215 | 54.7008 | 10.1841 |

### Đánh giá nghiệm thu GO / NO-GO (Hướng Carry):
- **Cấu trúc `coinm_1leg` (real_taker)**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): ĐẠT (alpha_t=17.34, beta=0.0001)
  - Điều kiện 2 (Sharpe >= 1.50): ĐẠT (Sharpe=12.22)
  - Điều kiện 3 (DSR >= 0.95): ĐẠT (DSR=1.0000)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu trúc `spot_perp_2leg` (real_taker)**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): ĐẠT (alpha_t=14.00, beta=0.0001)
  - Điều kiện 2 (Sharpe >= 1.50): ĐẠT (Sharpe=9.43)
  - Điều kiện 3 (DSR >= 0.95): ĐẠT (DSR=1.0000)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`

## 2. Kết quả chiến lược Cross-Sectional Market-Neutral
*Tổng số cấu hình thử nghiệm K = 12*

### Bảng tổng hợp cấu hình Cross-Sectional (real_taker):
| mode | lookback_hours | rebalance_hours | total_return_pct | annualized_return_pct | sharpe | max_dd_pct | dsr_prob | sr_star | alpha | beta | alpha_t | beta_t | alpha_annualized_pct | total_funding | total_fees | time_in_market_pct | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| momentum | 72 | 24 | -8.9276 | -3.1110 | 0.0379 | 61.0913 | 0.0548 | 0.0103 | -0.0000 | 0.0269 | -0.0036 | 2.7223 | -0.0596 | 17.5529 | 398.1852 | 99.6258 | 976.0000 |
| momentum | 72 | 48 | -20.8113 | -7.5827 | -0.1246 | 53.0414 | 0.0301 | 0.0103 | -0.0000 | 0.0109 | -0.2436 | 1.1401 | -4.1067 | 15.7434 | 196.1392 | 99.5332 | 649.0000 |
| momentum | 168 | 24 | -5.5786 | -1.9212 | 0.0757 | 46.0524 | 0.0624 | 0.0103 | 0.0000 | 0.0131 | 0.0995 | 1.2782 | 1.6039 | 16.6341 | 245.7262 | 99.3480 | 662.0000 |
| momentum | 168 | 48 | -8.7668 | -3.0531 | 0.0310 | 50.8878 | 0.0535 | 0.0103 | 0.0000 | 0.0085 | 0.0325 | 0.9236 | 0.5049 | 17.6948 | 159.7865 | 99.3480 | 453.0000 |
| momentum | 336 | 24 | 42.3751 | 12.6816 | 0.5610 | 32.1539 | 0.2425 | 0.0103 | 0.0000 | 0.0045 | 0.9875 | 0.4444 | 15.8109 | 18.4194 | 176.9639 | 98.6999 | 446.0000 |
| momentum | 336 | 48 | 78.8802 | 21.7176 | 0.8354 | 35.3852 | 0.4117 | 0.0103 | 0.0000 | 0.0019 | 1.4698 | 0.1995 | 23.5730 | 18.3243 | 143.1486 | 98.6999 | 316.0000 |
| reversal | 72 | 24 | -78.2481 | -40.2817 | -1.6195 | 82.2093 | 0.0000 | 0.0103 | -0.0001 | -0.0265 | -2.7984 | -2.6879 | -46.1363 | -3.2669 | 96.9493 | 99.6258 | 976.0000 |
| reversal | 72 | 48 | -60.4272 | -26.8962 | -0.9282 | 63.8402 | 0.0006 | 0.0103 | -0.0000 | -0.0105 | -1.5844 | -1.1025 | -26.6155 | -6.6541 | 96.4235 | 99.5332 | 649.0000 |
| reversal | 168 | 24 | -67.1455 | -31.3511 | -1.1676 | 76.8050 | 0.0001 | 0.0103 | -0.0000 | -0.0130 | -2.0551 | -1.2733 | -32.9280 | -6.2172 | 82.8410 | 99.3480 | 662.0000 |
| reversal | 168 | 48 | -54.0655 | -23.1188 | -0.7907 | 65.4467 | 0.0012 | 0.0103 | -0.0000 | -0.0085 | -1.4144 | -0.9261 | -21.9368 | -6.6674 | 67.0613 | 99.3480 | 453.0000 |
| reversal | 336 | 24 | -70.4346 | -33.7553 | -1.3009 | 72.7221 | 0.0000 | 0.0103 | -0.0000 | -0.0044 | -2.3151 | -0.4416 | -36.9137 | -6.6613 | 59.0504 | 98.6999 | 446.0000 |
| reversal | 336 | 48 | -71.6688 | -34.7030 | -1.3638 | 73.3590 | 0.0000 | 0.0103 | -0.0000 | -0.0023 | -2.4054 | -0.2338 | -38.5095 | -5.7014 | 40.6349 | 98.6999 | 316.0000 |

### Đánh giá nghiệm thu GO / NO-GO (Hướng Cross-Sectional):
- **Cấu hình `momentum` lookback=72h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=-0.00, beta=0.0269)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=0.04)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0548)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=72h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=-0.24, beta=0.0109)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=-0.12)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0301)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=168h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=0.10, beta=0.0131)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=0.08)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0624)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=168h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=0.03, beta=0.0085)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=0.03)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0535)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=336h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=0.99, beta=0.0045)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=0.56)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.2425)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `momentum` lookback=336h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=1.47, beta=0.0019)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=0.84)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.4117)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=72h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=-2.80, beta=-0.0265)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=-1.62)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0000)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=72h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=-1.58, beta=-0.0105)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=-0.93)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0006)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=168h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=-2.06, beta=-0.0130)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=-1.17)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0001)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=168h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=-1.41, beta=-0.0085)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=-0.79)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0012)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=336h rebalance=24h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=-2.32, beta=-0.0044)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=-1.30)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0000)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
- **Cấu hình `reversal` lookback=336h rebalance=48h**:
  - Điều kiện 1 (Market Neutral: alpha_t >= 2.50 & |beta| <= 0.20): KHÔNG ĐẠT (alpha_t=-2.41, beta=-0.0023)
  - Điều kiện 2 (Sharpe >= 1.00): KHÔNG ĐẠT (Sharpe=-1.36)
  - Điều kiện 3 (DSR >= 0.95): KHÔNG ĐẠT (DSR=0.0000)
  - Điều kiện 4 (Holdout 180 ngày): CHƯA CHẠY / KHÔNG ĐẠT
  - **Kết luận:** `NO-GO`
