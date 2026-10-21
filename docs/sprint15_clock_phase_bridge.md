# Sprint 15 - Dong ho tai can bang 48h cua Method D

Holdout: 181 ngay | Von active: 3,000 USDT | Symbol: 8

## 1. Bon moc so sanh

| Phuong phap | Net PnL (USD) | Return (%) | Ghi chu |
|---|---:|---:|---|
| B. Continuous fixed-base | 118.50 | 3.95 | chuoi day du |
| C. Standalone cold-start | 97.07 | 3.24 | khong warm |
| D. Method D (unaligned clock) | 81.73 | 2.72 | moc 48h neo nen dau chuoi cat |
| D'. Method D (clock aligned) | 76.09 | 2.54 | offset=22h |

## 2. Phan ra B - C thanh 4 thanh phan

| Thanh phan | USD |
|---|---:|
| Fee mo vi the ngay dau | 2.02 |
| Indicator blindness (D - C) | -15.34 |
| Clock phase drift (D' - D) | -5.63 |
| Pure inventory carryover (B - D' - Fee) | 40.39 |
| **Tong giai thich** | **21.44** |
| B - C thuc te | 21.44 |

### Kiem tra dang thuc

- `identity_gap_usd` = 0.000000e+00 (yeu cau < 1e-6)
- `unexplained_total_usd` = 0.000000e+00 (yeu cau < 1e-6)

## 3. Ket luan

- Lech pha dong ho 48h chi chiem 26.28% cua chenh lech B - C, nen khong phai nguyen nhan chinh cua drag khoi dong.
- Sau khi tach lech pha, phan con lai la pure inventory carryover - dong nghia voi viec ton kho duoc mang sang tu chuoi day du la nguon loi the that su.
