# Sprint 14 - Warm-cash holdout bridge (A/B/C/D)

Holdout: 181 ngay | Von active: 3,000 USDT | Symbol: 8

## 1. Bon phuong phap tai cung mot moc holdout

| Phuong phap | Von dau (USD) | Net PnL (USD) | Return (%) | Ghi chu |
|---|---:|---:|---:|---|
| A. Continuous compound | 4,091.29 | 159.15 | 3.89 | tai dau tu lai, von truot |
| B. Continuous fixed-base | 3,946.72 | 118.50 | 3.95 | neo 3000 USDT, chi bao da am |
| C. Standalone cold-start | 3,000.00 | 97.07 | 3.24 | nap thang vao holdout, moi thu lanh |
| D. Indicator-warm cash-start | 3,000.00 | 81.73 | 2.72 | chi bao am 504h, ton kho = tien mat |

## 2. Phan ra chenh lech B - C (tong do trep khoi dong)

| Thanh phan | USD | Y nghia |
|---|---:|---|
| cold_start_initial_entry_fee_usd | 2.02 | phi mo vi the ngay dau (24 bar dau holdout) |
| indicator_blindness_drag_usd | -15.34 | (D - C): do tre do EMA-42/momentum chua du lich su |
| inventory_carryover_diff_usd | 34.76 | (B - D) tru phi mo: chenh lech ton kho duoc mang sang |
| **Tong giai thich** | **21.44** | B - C = 21.44 |

### Kiem tra dang thuc

- `B - C - [(B - D) + (D - C)]` = 0.000000e+00 (yeu cau < 1e-6)
- `unexplained_total_usd` = 0.000000e+00 (yeu cau < 1e-6)

## 3. Ket luan so bo

- Method D dat 2.72% tren von active 3000 USDT, chua dat nguong 3,5% cua Sprint 11.
- Phan lon do trep khoi dong den tu do mu chi bao (D - C), khong phai phi mo vi the; dieu nay xac nhan can pre-warm indicator truoc khi vao live.

## 4. Doc ket qua va han che

- `indicator_blindness_drag_usd` am (-15,34 USD): trong cua so holdout nay, **chi bao da am lai lam KEM hon** trang thai lanh. Day la ket qua that tu mo phong, khong phai loi tinh toan: equity cua D bi khoa dung 3000,000000 USDT trong 504 bar dau (da kiem chung `max |equity - 3000| = 0,0` truoc moc) va chi giao dich tu `trade_start_idx` tro di.
- Giai thich kha di: (a) EMA-42 funding va momentum/vol 336h tinh tren chuoi 2026-04 tro ve truoc mang trang thai regime cu, nen tai moc holdout bot uu tien nhom symbol dang huong funding cu thay vi nhom tot nhat cua holdout; (b) pha tai can bang overlay (`np.arange(n_bars) % 48`) o D lech so voi C vi C bat dau dem tu bar 0 cua holdout -> khac thoi diem vao lenh. Vi vay C khong phai proxy hoan hao cho "bot moi tinh", va D cung khong phai "ban tot hon" tuyet doi.
- Ket luan an toan: **ca 4 phuong phap deu duoi nguong 3,5%** (A 3,89% tinh tren von truot, B 3,95% tren 3000, C 3,24%, D 2,72%) -> khuyen nghi testnet cua Sprint 11 giu nguyen **NO-GO**; khong duoc dung ket qua Sprint 14 de noi long nguong.
- Chi phi khoi dong (21,44 USD ~ 0,71% von) van la yeu to quyet dinh: 2,02 USD phi mo + phan con lai la do trep hanh vi (duong hoac am tuy regime).
