# Sprint 13 — Cầu nối kế toán Holdout (warm-state vs cold-start)

Sinh tự động bởi `scripts/gen_holdout_bridge.py` từ 2 lần chạy `run_sprint11.py`
(compound: `data/bridge_compound`, fixed-base: `data/bridge_fixed`). Cửa sổ holdout: 181 ngày, vốn gốc 3000.00 USDT.

## Ba phương pháp (tất cả đều là lãi ròng all-in sau phí carry + phí overlay)

| Phương pháp | Vốn đầu cửa sổ | PnL ròng (USDT) | Tỷ suất | Cách hiểu |
|---|---|---|---|---|
| A. Continuous compound (`reinvest=True`) | 4091.29 | 159.15 | 3.89% | Vị thế ấm, quy mô phóng theo lãi kép |
| B. Continuous fixed-base (`reinvest=False`) | 3946.72 (vốn hoạt động 3000.00) | 118.50 | 3.95% trên vốn 3.000 | Vị thế ấm, quy mô neo 3.000 suốt 3,45 năm |
| C. Standalone cold-start re-sim | 3000.00 | 97.07 | 3.24% | Bắt đầu 100% tiền mặt, phải mở lại danh mục |

## Phân rã chênh lệch

| Thành phần | USD | Ý nghĩa |
|---|---|---|
| Hiệu ứng quy mô do lãi kép (A − B) | 40.64 | Vị thế lớn hơn nhờ lãi tích lũy, không phải alpha tốt hơn |
| Tổng ma sát cold-start (B − C) | 21.44 | Giá phải trả khi bắt đầu lại từ tiền mặt |
| — trong đó phí mở danh mục ngày đầu | 2.02 | 24 bar đầu của lần chạy standalone |
| — còn lại: lệch đường đi warmup/min_hold | 19.42 | Do `ema_span=42` / `min_hold=63` chưa ấm ở đầu cửa sổ |

## Đẳng thức bảo toàn

```
A.net_pnl = C.net_pnl + (A − B) + (B − C)
159.147661 = 97.066418 + 40.644522 + 21.436720
residual = 0.00e+00
```

## Kết luận

- Con số **+3,75%** từng được trích dẫn là carry **gross** (funding + basis) cộng overlay net,
  chưa trừ phí carry — không phải lãi ròng all-in. Không dùng để kết luận GO.
- Ba phương pháp cho ba con số khác nhau vì **cùng một chiến lược, khác trạng thái vốn**.
  Cửa sổ holdout ngắn (181 ngày) nên chênh lệch quy mô vài phần trăm vốn tạo khác biệt ~0,5 điểm %.
- Tiêu chí giữ nguyên: cả slice lẫn re-sim phải ≥ 3,5% → hiện **NO-GO**.
