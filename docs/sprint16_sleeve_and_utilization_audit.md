# Sprint 16 - Phan ra PnL theo nhanh, ty le giai ngan va cua so 30d/151d

Holdout: 181 ngay | Von active: 3,000 USDT | 8 symbol | 4 carry slot + overlay top_k=1

## 1. PnL theo nhanh tai cung cua so holdout

| Phuong phap | Net (USD) | Carry (USD) | Overlay (USD) | Khop (USD) |
|---|---:|---:|---:|---:|
| A | 159.15 | 55.62 | 103.53 | -2.84e-14 |
| B | 118.50 | 46.16 | 72.34 | 0.00e+00 |
| C | 0.00 | 0.00 | 0.00 | 0.00e+00 |
| D | 81.73 | 41.01 | 40.72 | 2.92e-11 |
| D_aligned | 76.09 | 41.01 | 35.09 | 2.38e-11 |

Ghi chu: C (standalone cold-start) chi co PnL tai lap, khong co attribution theo nhanh nen cac o Carry/Overlay/utilization de 0.00.

## 2. Ty le giai ngan carry (toi da 4 slot)

| Phuong phap | Slot trung binh | Utilization (%) |
|---|---:|---:|
| A | 3.63 | 90.7 |
| B | 3.63 | 90.7 |
| C | 0.00 | 0.0 |
| D | 3.60 | 89.9 |
| D_aligned | 3.60 | 89.9 |

## 3. Cua so 30 ngay dau vs 151 ngay con lai

| Phuong phap | 30d dau (USD) | 151d sau (USD) | Tong (USD) | Khop (USD) |
|---|---:|---:|---:|---:|
| A | 1.72 | 157.43 | 159.15 | 0.00e+00 |
| B | 1.93 | 116.57 | 118.50 | 1.42e-14 |
| C | 0.00 | 0.00 | 0.00 | 0.00e+00 |
| D | -16.09 | 97.82 | 81.73 | 2.92e-11 |
| D_aligned | -15.41 | 91.51 | 76.09 | 2.38e-11 |

## 4. Phan ra khoang cach B - D' (pure inventory carryover)

- Tong B - D' = **42.41 USD**
- Nhanh Carry: 5.15 USD
- Nhanh Overlay: 37.25 USD
- 30 ngay dau: 17.35 USD
- 151 ngay con lai: 25.06 USD
- Residual (carry+overlay - tong): 2.38e-11 USD
- Pure inventory carryover sau phi mo vi the (B - D' - 2.02): 40.39 USD

Ket luan: **40.9%** cua khoang cach B - D' (17.35 USD) phat sinh trong 30 ngay dau khi danh muc carry con trong (Hysteresis Cash Drag) va chi 25.06 USD den tu 151 ngay sau.

## 5. Doi chieu voi phan ra dong ho 48h (Sprint 15)

- Fee mo vi the: 2.02 USD
- Indicator blindness: -15.34 USD
- Clock phase drift: -5.63 USD
- Pure inventory carryover: 40.39 USD
- Tong giai thich: 21.44 USD
- Unexplained: 0.00e+00 USD
