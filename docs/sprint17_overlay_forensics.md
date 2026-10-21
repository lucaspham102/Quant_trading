# Sprint 17 - Phap y khoang cach PnL nhanh Overlay (B vs C vs D' vs E)

Holdout: 2026-04-21 -> 2026-10-18 | von active 3,000.00 USDT | tham so Frozen Alpha giu nguyen

## 1. PnL overlay (net) tung phep do

| Phep do | Mo ta | Warmup | Phase | Overlay net (USD) | Net (USD) |
|---|---|---:|---:|---:|---:|
| B | Fixed-base, chuoi day du tu 2023 | full | 0 | 72.34 | 118.50 |
| C | Standalone cold-start (chi holdout) | 0 | 0 | 48.87 | - |
| D | Warm cash, pha cat lat | 504 | 0 | 40.72 | 81.73 |
| D' | Warm cash, pha tuyet doi | 504 | 22 | 35.09 | 76.09 |
| **E** | Warm cash hoi tu (720h), pha tuyet doi | 720 | 46 | 73.26 | 114.27 |

## 2. Phan ra chinh xac chenh lech overlay

| Cap | Weight effect | Capital effect | Fee effect | Net gap | Residual |
|---|---:|---:|---:|---:|---:|
| B_minus_Dp | -35.82 | -0.13 | -1.47 | -37.41 | -2.84e-14 |
| B_minus_E | 0.83 | -0.05 | -0.00 | 0.78 | -9.88e-15 |
| B_minus_C | -24.92 | -0.08 | 1.13 | -23.87 | 7.11e-15 |
| E_minus_Dp | -36.63 | -0.10 | -1.47 | -38.19 | -7.11e-15 |

## 3. Moc tai can bang 48h

| Cap | So moc (A) | So moc (B) | Trung | Jaccard |
|---|---:|---:|---:|---:|
| B_vs_Dp | 124 | 127 | 90 | 0.559 |
| B_vs_E | 124 | 125 | 124 | 0.992 |
| Dp_vs_E | 127 | 125 | 91 | 0.565 |

## 4. Von co so trung binh / ty le gio trung trong so

- Notional base trung binh: B 3,000.00 | C 2,998.84 | D' 2,998.85 | E 2,999.79 USD
- Ty le gio trong so giong B: D' 0.0% | E 93.3% | C 74.3%
- Equity B tai split: 3,946.72 USD

## 5. Ket luan 4 gia thuyet

| Gia thuyet | Ket qua | Bang chung dinh luong |
|---|---|---|
| H1 - Overlay capital scaling / compounding leak | **BAC BO** | capital_effect = -0.13 USD (0.3% gap); notional base B 3,000.00 vs D' 2,998.85 USD (B bi clamp o initial_capital) |
| H2 - Rebalance timestamp / modulo shift | **BAC BO (la he qua, khong phai nguyen nhan)** | Jaccard B/D' = 0.559 nhung B/E = 0.992; pha 48h da neo dung dong ho tuyet doi o ca D va D' |
| H3 - Signal (EMA/momentum/vol) khac do warmup chua hoi tu | **XAC NHAN (nguyen nhan chinh)** | weight_effect = -35.82 USD (95.7% gap); ty le gio trong so giong B: D' 0.0% vs E 93.3%; nang warmup 504h -> 720h dua overlay tu 35.09 len 73.26 USD (B = 72.34) |
| H4 - Warmup zeroing artifact (mat lenh co cau dau tien) | **BAC BO** | D'/E deu vao vi the dung tai trade_start_idx (raw weights 0 -> khac 0 kich hoat rebalance); fee effect chi -1.47 USD |

### Ket luan tong

- Khoang cach overlay B - D' = 37.25 USD truoc hieu chinh; sau khi do lai bang E (warmup 720h) chi con -0.92 USD.
- `E_true_fixed_warm_cash`: net 114.27 USD vs B 118.50 USD -> gap 4.23 USD (truoc day bao cao Sprint 16 la 42.41 USD).
- Pure inventory carryover hieu chinh: 3.97 USD (thay vi 40.39 USD) -> phan lon 'khoang cach live vs backtest' la ao giac do warmup 504h.

