# Sprint 15 - Backtest-vs-Live parity, dong ho 48h va shadow-live burn-in

Nhanh: `agy/sprint15-backtest-live-parity` (tach tu `master` = `ce64d3d`, baseline 407 passed).
Muc tieu: sua lech gate carry/overlay giua backtest Frozen Alpha (Sprint 11) va live executor,
kiem toan dong ho tai can bang 48h cua Method D, chay shadow-live burn-in tren orderbook that.

## 0. Ket qua tong hop

| Hang muc | Ket qua |
|---|---|
| Carry gate parity (live vs Tier 1B) | Dat - live mo vi the dung tai 0.45 bps (pure `f_ewma`) |
| Overlay gate parity (smooth_conviction) | Dat - `MaxOverlayDelta` 0.136-0.138, 2 chan long/short |
| Diagnostics | Co `CarryGate` / `OverlayGate` / `TradesExecuted` + dem lenh thuc thi |
| Dong ho 48h (Method D) | Dat - tach duoc `clock_phase_drift_usd` = -5.63, dang thuc khep kin 0.000000 |
| Shadow-live burn-in (fixture) | 3/3 PASS |
| Shadow-live burn-in (mang that) | 10/10 PASS, 8 symbol, spread toi da 1.23 bps |
| Fire drill | 20/20 PASS |
| Test suite | **432 passed** (x2: 53.22s / 52.27s) |

## 1. Carry gate parity

### Nguyen nhan lech

`run_live.py` (nhanh barbell) dung `DynamicCarryConfig()` voi defaults Sprint 7:

- `ema_span=6`, `top_k=2`, `entry_threshold=0.00010` (1.0 bps), `exit_threshold=0.00002`,
  `min_hold_periods=9`, `buffer_rank=1`, `w_basis=0.5`.
- Score dua vao gate: `score = f_ewma + 0.5 * basis_dev / 21`.
- Ket qua: `FundEWMA = 0.73 bps` nhung `score` bi basis keo xuong duoi 1.0 bps
  -> `SKIP / SCORE_BELOW_ENTRY_THRESHOLD`, trong khi backtest Tier 1B mo vi the o 0.45 bps
  tren `f_ewma` thuan.

### Cach sua

- `src/dynamic_carry_basis.py`
  - `DynamicCarryConfig` them `switch_rule: Literal["advantage", "tier1b_buffer_rank"]="advantage"`
    va `eta_override: float | None = None`; `notional_eta()` tra `eta_override` khi duoc set.
  - `select_portfolio(..., switch_rule=None)`: nhanh `tier1b_buffer_rank` mo phong dung luat
    backtest - EXIT khi `periods_held >= min_hold_periods` va `score <= exit_threshold`;
    SWITCH chi khi `rank > top_k + buffer_rank` **va** co candidate `>= entry_threshold`
    (nguoc lai `HOLD_BUFFER_RANK`); ENTRY lap day slot `top_k - held`.
- `src/carry_live_executor.py`
  - Co moi `tier1b_gate: bool = False`; helper `_carry_selection_scores()` tra `f_ewma` thuan
    khi bat co nay; `select_portfolio(..., switch_rule="tier1b_buffer_rank")`.
- `run_live.py` nhanh `strat == "barbell"`: dung dung bo tham so Frozen Alpha
  (`ema_span=42, top_k=4, entry_threshold=0.000045, exit_threshold=0.0, min_hold_periods=63,
  buffer_rank=3, w_basis=0.0, expected_hold_periods=63, switch_rule="tier1b_buffer_rank",
  eta_override=0.80`) va `tier1b_gate=True`.

Mac dinh cu duoc giu nguyen nen toan bo test Sprint 7-10 van xanh.

### Bang chung chay that (paper, 2026-10-21)

```
| BTC/USDT:USDT   |      +0.45 |         +0.45 |    -0.05% |         +0.11 | OPEN_DELTA_NEUTRAL | 579.35 |
| ETH/USDT:USDT   |      +0.36 |         +0.36 |    -0.04% |         +0.07 | SKIP               |   0.00 |
```

- BTC co `f_ewma = 0.45 bps` = nguong entry -> OPEN (backtest cung mo o dung nguong nay).
- ETH `0.36 bps` -> SKIP, dung luat Tier 1B (khong con bi basis blend keo xuong duoi 1.0 bps).
- Notional 579.35 USD = `(3000/4) * 0.80` = 600 USD sau luong hoa lot-size, khop `eta=0.80`.

## 2. Overlay gate parity

- Ham moi `compute_live_overlay_targets()` trong `src/barbell_portfolio.py`:
  cat `lookback_hours=720` nen cuoi (khong look-ahead), chay lai `run_cross_sectional_portfolio`
  voi `momentum 336h / rebalance 48h / min_spread 0.75 / top_k=1 / buffer_rank=1 / carry_tilt=0.20`,
  roi ap gate `smooth_conviction`: `c_t = clip(snr/1.00, 0.50, 1.00)` khi `snr >= 0.25` va du
  hai chan; vol targeting `L_t = c_t * clip(0.075/sigma_pair, 0.15, 0.50)`; `w_i = raw_w_i * L_t`,
  bo qua thay doi duoi `min_weight_change = 0.08`.
- `build_barbell_allocations()` dung ham nay khi `tier1b_gate=True` (nhanh barbell), luong cu
  giu nguyen cho cac test Sprint 9-10.
- Bang chung: `Overlay (Long=331.06 vs Short=407.68)`, `MaxOverlayDelta=0.136 -> PASS`
  (truoc khi sua: `overlay_active=False`, `MaxOverlayDelta=0.000`).

## 3. Diagnostics

Format moi (van giu cac field cu de khong vo test Sprint 14):

```
Warmup=720h (thr 400h) -> OK | MaxCarryEMA=0.45 bps (thr 0.45) -> PASS | MaxOverlayDelta=0.136 (thr 0.08) -> PASS |
CarryGate: PASS (selected=1, maxEMA=0.45 bps vs thr 0.45) |
OverlayGate: PASS (selected=2, maxDelta=0.136 vs thr 0.08) |
TradesExecuted: carryTrades=1, overlayTrades=2
```

- `carryTrades`: so vi the carry mo/dong/thay doi notional trong chu ky (so sanh snapshot state).
- `overlayTrades`: so lenh overlay co status `filled`/`paper_filled` tu `execute_overlay_rebalance`.
- Dashboard khong con in cung `Long=0.00 vs Short=0.00`; doc tu `state.overlay_positions`.

## 4. Dong ho tai can bang 48h cua Method D

- `run_barbell_portfolio(..., rebalance_phase_offset=0)` + `build_rebalance_mask()`:
  mask = `(bar + offset) % 48 == 0`, union voi cac nen trong so doi.
- `run_indicator_warm_cash_start(..., phase_aligned=True)`: offset = `global_start_idx % 48`,
  voi `global_start_idx` la vi tri cua `split_date - warmup` trong chuoi day du -> moc tai can
  bang trung voi backtest B/C.
- `compute_holdout_accounting_bridge(..., warm_cash_aligned=...)` tach 4 thanh phan.

Ket qua that (`scripts/gen_clock_phase_bridge.py`, holdout 181 ngay, von 3000):

| Phuong phap | Net PnL (USD) | Return (%) |
|---|---:|---:|
| B. Continuous fixed-base | 118.50 | 3.95 |
| C. Standalone cold-start | 97.07 | 3.24 |
| D. Method D (unaligned clock) | 81.73 | 2.72 |
| D'. Method D (clock aligned, offset 22h) | 76.09 | 2.54 |

| Thanh phan | USD |
|---|---:|
| Fee mo vi the ngay dau | 2.02 |
| Indicator blindness (D - C) | -15.34 |
| Clock phase drift (D' - D) | **-5.63** |
| Pure inventory carryover (B - D' - Fee) | 40.39 |
| Tong giai thich | 21.44 |
| B - C thuc te | 21.44 |

- `identity_gap_usd` = 0.000000e+00, `unexplained_total_usd` = 0.000000e+00.
- Lech pha dong ho chiem 26.28% chenh lech B - C: co y nghia nhung khong phai nguon chinh.
  Loi the that su cua backtest la ton kho duoc mang sang (pure inventory carryover 40.39 USD).

## 5. Shadow-live burn-in

`scripts/run_shadow_burnin.py` (paper, read-only, khong dat lenh that):

- Moi chu ky: snapshot orderbook Binance public -> kiem feed (du symbol, `bid < ask`,
  spread <= 10 bps) -> `reconcile_once()` -> kiem state (hedge khop, khong so du am, dust duoc
  danh dau) -> kiem risk (drawdown < 6%, kill-switch khong bat).
- Fixture offline: `3/3 PASS`.
- Mang that: `10/10 PASS`, `feed=network symbols=8`, `max_spread=1.23 bps`.
- `--strict` tra exit 1 khi co chu ky FAIL (co test rieng).

## 6. Kiem thu

- `pytest -q`: **432 passed** (53.22s) va **432 passed** (52.27s) - muc tieu >= 425 dat.
- Test moi:
  - `tests/test_sprint15_backtest_live_parity.py` (11 test)
  - `tests/test_sprint15_clock_alignment.py` (6 test)
  - `tests/test_sprint15_shadow_burnin.py` (8 test)
- `scripts/run_e2e_fire_drill.py`: 20/20 PASS.
- Paper barbell 2 chu ky: `equity=2999.35`, `carryTrades=1`, `overlayTrades=2`, khong loi.

## 7. Ket luan va trang thai

- Lech Backtest-vs-Live da duoc dong: live gio mo vi the theo dung nguong 0.45 bps tren
  `f_ewma` thuan, dung luat switch buffer-rank, dung eta 0.80 va dung cong thuc overlay
  smooth_conviction.
- Method D sau khi can dong ho chi dat 2.54% (D') - van duoi nguong 3.5% cua Sprint 11.
- **Verdict Sprint 11 giu nguyen: NO-GO cho testnet.** Sprint 15 chi khoi phuc tinh dung dan
  giua backtest va live, khong thay doi ket luan loi nhuan.
- Khong merge/push: cho nguoi dung quyet dinh.
