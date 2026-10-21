# Sprint 18 — GO / NO-GO Chính thức (Live Runtime Invariants + Perp Margin Reconciler)

**Ngày chốt:** 2026-10-21
**Nhánh:** `agy/sprint18-live-invariants-and-margin-reconciler` (tách từ `master`, baseline 510 passed)
**Trạng thái kỹ thuật:** HOÀN TẤT — `python -m pytest -q` = **543 passed**, 0 failed (2 lần liên tiếp)

---

## 1. Phán quyết

> ## ✅ **QUANT ALPHA: GO**
> **Holdout +3.81% ≥ 3.50% target · 7/7 cửa sổ 6M dương · Sharpe 1.86 · MaxDD 4.44%**
> (Nâng từ `NO-GO (measurement artifact)` của Sprint 11 → `GO` sau pháp y Sprint 17 + kiểm toán độc lập Sprint 18)

Phán quyết này chỉ thay đổi **trạng thái định lượng của chiến lược**, KHÔNG đồng nghĩa được
bơm toàn bộ 3.000 USDT vào sàn ngay. Việc triển khai vốn tuân theo **Stage-Gated Rollout** ở §4.

---

## 2. Căn cứ định lượng (Pháp y Sprint 17)

| Hạng mục | Method B (chuẩn cũ) | Method E (`Indicator-Warm >= 720h`, `Clock-Aligned`, `Fixed-Base 3.000 USDT`) |
|---|---|---|
| Net PnL 181 ngày | +118.50 USD | **+114.27 USD** |
| Return on active capital | +3.95% | **+3.81%** (ngưỡng 3.50%) |
| Jaccard trọng số (B vs E) | — | **0.992** |
| NetDelta overlay | — | **≤ 0.07 USDT** |
| Cửa sổ 6M dương | 7/7 | **7/7** |

Kết luận pháp y: chênh lệch B/E là do **warmup chưa hội tụ (504h < 720h)**, không phải do
chiến lược yếu. Với cửa sổ 720h trở lên, kết quả bão hòa hoàn toàn (720/1440/2160/2880h đều
cho 114.27 USD). Vì vậy Sprint 18 **khóa cứng 720h ở runtime** thay vì chỉ ghi trong tài liệu.

---

## 3. 5 Định luật Bất biến (Golden Quant Invariants) — trạng thái sau Sprint 18

| # | Bất biến | Trạng thái | Cơ chế thực thi runtime |
|---|---|---|---|
| 1 | WARMUP_SATURATION | **PASS** | `WARMUP_MIN_HOURS = 720`; `compute_warmup_readiness()` chạy đầu mỗi chu kỳ; `< 720h` → chặn mở vị thế mới (`WARMUP_INSUFFICIENT_BELOW_720H`) |
| 2 | ZERO_COMPOUNDING_LEAK | **PASS** | Không đổi (Frozen Alpha, `reinvest_profits=False`) |
| 3 | CLOCK_SHIFT_INVARIANCE | **PASS** | Neo timestamp tuyệt đối (không dùng `local_idx % 48`) |
| 4 | ACCOUNTING_IDENTITY | **PASS** | Net = Gross − Fees, kiểm trong test bất biến |
| 5 | LOT_NEUTRALITY | **PASS (fail-closed)** | `verify_lot_neutrality_invariant(...)` gọi trước khi phát lệnh Overlay; vi phạm → hủy toàn bộ lệnh Overlay chu kỳ đó |

Bằng chứng runtime: dòng `Diagnostics:` trên Dashboard in
`Warmup=720h (thr 720h) -> OK | ... | Invariants: OK`, và mỗi dòng JSONL `--audit-log` có
`invariants_passed`, `warmup_threshold_hours = 720`, `warmup_hours_loaded`.

---

## 4. Live Capital Deployment Checklist (Stage-Gated Rollout)

| Bước | Phạm vi | Vốn | Thời lượng | Điều kiện chuyển bước | Rollback |
|---|---|---|---|---|---|
| **1** | Testnet live keys (Binance Spot + USDⓈ-M Futures), `--live-snapshot` BẬT | 0 (testnet) | **48 giờ liên tục** | `python scripts/run_testnet_preflight.py` = 6/6 PASS; `scripts/run_shadow_burnin.py` = 10/10; không có `SNAPSHOT_UNHEALTHY`, không có `MARGIN_DANGER`; `NetDelta ≤ 10 USDT` | Tắt tiến trình, giữ state, không mở lệnh mới |
| **2** | **Canary** tiền thật, 1 cặp (BTC), `tier1b_gate=True` | **400 USDT** | **7 ngày** | Equity drift (sau phân rã `equity_drift_attribution`) ≤ 0.5%/ngày và `primary_mismatch_cause` không phải `UNEXPLAINED_EQUITY_DRIFT` 2 ngày liên tiếp; `maint_margin_ratio < 0.30`; `liq_distance ≥ 40%` | Kill-switch → đóng toàn bộ vị thế, quay lại Bước 1 |
| **3** | **Full** Barbell (Carry + Overlay), universe LIQUID-8 | **3.000 USDT** | liên tục | Bước 2 đạt 7/7 ngày; Drawdown live < 6% (kill-switch); reconcile `SYNCED` ≥ 95% số chu kỳ | Hạ về Canary 400 USDT |

Điều kiện tiên quyết chung: API key **chỉ có quyền đọc + giao dịch spot/futures**, KHÔNG bật quyền rút tiền;
`BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` đặt qua biến môi trường (không hardcode).

---

## 5. Điều kiện NO-GO / tạm dừng (áp dụng ngay khi vi phạm)

1. `WARMUP_INSUFFICIENT_BELOW_720H` xuất hiện khi hệ thống đã có vị thế mở (dữ liệu bị cắt ngắn bất thường).
2. `status = MARGIN_DANGER` hoặc `min_liq_distance_pct < 25%` bất kỳ chu kỳ nào.
3. `invariants_passed = false` (đặc biệt `NAKED_LEG`) xuất hiện ≥ 2 chu kỳ liên tiếp.
4. `equity_drift_attribution.primary_mismatch_cause = UNEXPLAINED_EQUITY_DRIFT` với
   `|unexplained_residual_drift_usd| > 10 USD` trong 2 chu kỳ liên tiếp.
5. `pytest -q` bất kỳ lần nào < toàn bộ PASS (không được sửa test để hợp thức hóa code sai).
6. Thay đổi bất kỳ công thức/tham số Frozen Alpha (Tier 1B `ema_span=42`, `entry_threshold=0.45 bps`,
   `buffer_rank`, `min_hold_periods=63`; Smooth Conviction `c_t`, `vol_target`, leverage band, `min_weight_change=0.08`)
   → tự động vô hiệu phán quyết GO này, phải chạy lại toàn bộ holdout.

---

## 6. Phạm vi công bố

- Phán quyết GO áp dụng cho: **Binance Spot + USDⓈ-M Futures, universe 12 coin, spot-perp carry + momentum overlay,
  capital 3.000 USDT, phí `maker_taker_hedge`**.
- KHÔNG áp dụng cho: universe khác, đòn bẩy perp > 3x, hoặc chạy khi `warmup < 720h`.
- Kiểm toán độc lập (Sprint 18, role `quant_auditor`): **0 `[BLOCKER: QUANT_ARTIFACT]`**, 2 `[WARN]` (1 đã sửa, 1 là by-design).
