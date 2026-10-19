# Sprint 6 — Tiến độ & quyết định (ĐỌC TRƯỚC KHI LÀM TIẾP)

> Nhánh: `agy/sprint6-carry-and-xs` (từ `master` đã merge Sprint 4+5).
> Quy ước: cập nhật + commit sau mỗi chặng. Kết quả nằm trong `data/*.csv` + `docs/*.md`.
> Kết quả dữ liệu (data/, CSV) bị gitignore; funding cache ở `data/raw/{SYM}_funding.parquet`.

## 0. Mục tiêu
Hai hướng độc lập trên baseline 400 USDT, tiêu chí khóa ở `docs/go_no_go_sprint6.md`:
- **Carry delta-neutral** (`coinm_1leg`, `spot_perp_2leg`).
- **Cross-Sectional market-neutral** (`momentum`, `reversal`).
Yêu cầu §2: **KHÔNG đóng cưỡng bức/reset vốn ở biên cửa sổ** — đường cong vốn liên tục (stitched).

## 1. Trạng thái
| Chặng | Nội dung | Trạng thái | Commit |
|---|---|---|---|
| 0 | Nhánh + khóa tiêu chí | DONE | `9b0f3d7` |
| 1 | `funding_loader.py` + `carry_strategy.py` + tests | DONE | (Chặng 1) |
| 2 | `cross_sectional.py` + tests | DONE | (Chặng 2) |
| 3 | `sprint6_eval.py` + `run_sprint6.py` + tests | DONE | (Chặng 3) |
| 4 | Tải funding thật + chạy DEV | DONE | (này) |
| 4b | Sửa DSR `sr_variance` + holdout stitched | IN_PROGRESS | |
| 5 | Holdout 1 lần + báo cáo | TODO | |

Baseline test: 131 → Chặng1 136 → Chặng2 143 → Chặng3 152 passed.

## 2. Dữ liệu
- Funding 8h thật: 3784 kỳ/mã (BTC/ETH/SOL/BNB), 2023-05-07 → 2026-10-19, 0 trùng, 0 NaN.
- OHLCV 1h sẵn có trong `data/raw/*_1h.parquet`.
- DEV = 2023-05-07 → 2026-04-21. HOLDOUT = 2026-04-21 → 2026-10-18.

## 3. Kết quả DEV (sơ bộ, TRƯỚC khi sửa DSR)
- **Carry coinm_1leg real_taker**: Sharpe năm hóa **12.22**, alpha_t 17.34, beta 0.0001, DSR 1.0, return 12.0%, MaxDD 0.083%, time-in-market 54.7%.
- **Carry spot_perp_2leg real_taker**: Sharpe 9.43, alpha_t 14.0, DSR 1.0, return 6.9%.
- **XS momentum 336h/48h** (tốt nhất): Sharpe 0.84, alpha_t 1.47, DSR 0.41, return +78.9%, MaxDD 35.4%.
- **XS reversal**: thua lỗ nặng (Sharpe âm) ở mọi cấu hình.

## 4. Review phương pháp (quick_pro, `.agy/quick_pro_8d53d4f2.md`)
- **(CAO) Bỏ qua basis risk MTM**: `basis_noise` chỉ lấy mẫu 1 lần khi ĐÓNG lệnh; không có PnL basis hàng giờ, notional cố định, không nạp giá. → biến động danh mục ≈ 0.32%/năm → **Sharpe 12 bị phóng đại**. Ước lượng Sharpe thực nếu thêm basis vol ~6%: ~0.65. **Đây là đặc thù model đã khóa (blueprint chỉ định basis_noise khi đóng) → KHÔNG sửa model, phải CÔNG BỐ như hạn chế.**
- **(TRUNG) DSR proxy sai**: `compute_deflated_sharpe_ratio` dùng proxy `1/(n-1)` thay vì phương sai ngang `V[{SR_k}]` của K cấu hình → SR*≈0 → DSR luôn 1.0. **CẦN SỬA.**
- **(TRUNG) Holdout reset vốn**: chạy holdout riêng sẽ reset 400 + flat-forced → trái §2. **CẦN SỬA (segment-aware từ 1 lần chạy liền mạch).**
- **(TRUNG) 1d bỏ sót 2/3 kỳ funding**: không ảnh hưởng (ta chạy 1h).
- Khác: K nhưng fee scenario là môi trường; min lot Coin-M 100$; margin segregation.

## 5. Quyết định sửa (không đổi chiến lược/tiêu chí)
- Sửa **DSR**: truyền `sr_variance = var(Sharpe_k per-period)` của K cấu hình vào.
- Sửa **holdout**: chạy 1 lần liền mạch cả dải; tách metric DEV (`< split`) và HOLDOUT (`>= split`) từ cùng đường cong.
- Giữ nguyên model carry/xs đã khóa; công bố hạn chế basis risk trong báo cáo cuối.

## 6. Cảnh báo session sau
- KHÔNG đụng master; KHÔNG merge/push/reset --hard/branch -D.
- Py3.14, pandas 3.0 (không chained assignment; `.as_unit("ns")`). PowerShell `;`. `PYTHONIOENCODING=utf-8` khi in tiếng Việt.
- codec CLI in tiếng Việt ra file cần `encoding="utf-8"`.
