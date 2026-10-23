# Sprint 21 - Multi-Cycle Testnet Burn-In Evidence

- **Timestamp (UTC)**: `2026-10-22T15:57:22.253852+00:00`
- **Verdict**: `FAIL`
- **Symbol**: `BTC/USDT:USDT`
- **Universe**: 8 symbols (`BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`, `DOGE/USDT:USDT`, `XRP/USDT:USDT`, `AVAX/USDT:USDT`, `LINK/USDT:USDT`)
- **Secrets redacted**: `True`

| Cycle | Name | OK | Reconcile | All 8 SYNCED | Equity Gap (USD) |
|---|---|---|---|---|---|
| 1 | Bootstrap & Reconcile | False | ORPHAN_ON_EXCHANGE | False | 0.0 |

## Cycle Details

### Cycle 1: Bootstrap & Reconcile

```json
{
  "credential_check": "enforced_64_64",
  "credentials_ok": true,
  "credential_format": {
    "spot": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    },
    "futures": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    }
  },
  "server_time_offsets_ms": {
    "spot": 1780,
    "futures": 1760
  },
  "warmup_gate": {
    "gate": "720h",
    "warmup_hours": 720.0,
    "required_hours": 720.0,
    "ok": true
  },
  "snapshot_healthy": true,
  "mid_price": 81702.265,
  "reconcile_status": "ORPHAN_ON_EXCHANGE",
  "all_8_symbols_synced": false,
  "symbol_statuses": {
    "AVAX/USDT:USDT": "ORPHAN_ON_EXCHANGE",
    "BNB/USDT:USDT": "ORPHAN_ON_EXCHANGE",
    "BTC/USDT:USDT": "ORPHAN_ON_EXCHANGE",
    "DOGE/USDT:USDT": "ORPHAN_ON_EXCHANGE",
    "ETH/USDT:USDT": "ORPHAN_ON_EXCHANGE",
    "LINK/USDT:USDT": "ORPHAN_ON_EXCHANGE",
    "SOL/USDT:USDT": "ORPHAN_ON_EXCHANGE",
    "XRP/USDT:USDT": "ORPHAN_ON_EXCHANGE"
  },
  "equity_gap_usd": 0.0,
  "untracked_spot_count": 504
}
```

_Log JSONL: `logs/sprint21_multi_cycle_burnin.jsonl`_

# Sprint 21 - Multi-Cycle Testnet Burn-In Evidence

- **Timestamp (UTC)**: `2026-10-22T15:59:48.999129+00:00`
- **Verdict**: `PASS`
- **Symbol**: `BTC/USDT:USDT`
- **Universe**: 8 symbols (`BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`, `DOGE/USDT:USDT`, `XRP/USDT:USDT`, `AVAX/USDT:USDT`, `LINK/USDT:USDT`)
- **Secrets redacted**: `True`

| Cycle | Name | OK | Reconcile | All 8 SYNCED | Equity Gap (USD) |
|---|---|---|---|---|---|
| 1 | Bootstrap & Reconcile | True | SYNCED | True | 0.0 |
| 2 | Live Dual-Leg Execution & Mid-Cycle Audit | True | SYNCED | True | N/A |
| 3 | Clean Flatten & Zero-Residual Verification | True | SYNCED | True | 0.0 |

## Cycle Details

### Cycle 1: Bootstrap & Reconcile

```json
{
  "credential_check": "enforced_64_64",
  "credentials_ok": true,
  "credential_format": {
    "spot": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    },
    "futures": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    }
  },
  "server_time_offsets_ms": {
    "spot": 1975,
    "futures": 1765
  },
  "warmup_gate": {
    "gate": "720h",
    "warmup_hours": 720.0,
    "required_hours": 720.0,
    "ok": true
  },
  "snapshot_healthy": true,
  "mid_price": 81668.36499999999,
  "reconcile_status": "SYNCED",
  "all_8_symbols_synced": true,
  "symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "equity_gap_usd": 0.0,
  "untracked_spot_count": 504,
  "bootstrap_baseline_symbols": 8,
  "bootstrap_equity_usd": 484266735.8214776
}
```

### Cycle 2: Live Dual-Leg Execution & Mid-Cycle Audit

```json
{
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "best_bid": 81668.36,
  "best_ask": 81668.37,
  "tick_size": 0.01,
  "maker_pricing": {
    "side": "BUY",
    "best_bid": 81668.36,
    "best_ask": 81668.37,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81668.36
  },
  "entry_status": "FILLED",
  "spot_entry_order_id": 5873563,
  "perp_entry_order_id": 28646182519,
  "executed_qty": 0.001,
  "perp_executed_qty": 0.001,
  "fill_price": 81668.37,
  "lot_neutrality": {
    "passed": true,
    "invariant_name": "LOT_NEUTRALITY",
    "max_error": 0.0,
    "imbalance_usd": 0.0,
    "details": "[PASS] Lot neutrality verified: Leg A=81.67 USD, Leg B=81.67 USD, Imbalance=0.00 USD <= 10.00 USD."
  },
  "lot_neutral_ok": true,
  "mid_reconcile_status": "SYNCED",
  "mid_all_8_symbols_synced": true,
  "mid_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "futures_positions_verified": true
}
```

### Cycle 3: Clean Flatten & Zero-Residual Verification

```json
{
  "maker_pricing": {
    "side": "SELL",
    "best_bid": 81668.36,
    "best_ask": 81668.37,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81668.37
  },
  "exit_status": "FILLED",
  "spot_exit_order_id": 5873575,
  "perp_exit_order_id": 28646182553,
  "exit_executed_qty": 0.001,
  "post_reconcile_status": "SYNCED",
  "post_all_8_symbols_synced": true,
  "post_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "spot_qty_after": 1.0,
  "perp_qty_after": 0.0,
  "spot_residual_qty": 0.0,
  "perp_residual_qty": 0.0,
  "equity_gap_usd": 0.0,
  "zero_residual": true,
  "equity_gap_zero": true
}
```

_Log JSONL: `logs/sprint21_multi_cycle_burnin.jsonl`_

# Sprint 21 - Multi-Cycle Testnet Burn-In Evidence

- **Timestamp (UTC)**: `2026-10-22T16:21:10.404354+00:00`
- **Verdict**: `PASS`
- **Symbol**: `BTC/USDT:USDT`
- **Universe**: 8 symbols (`BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`, `DOGE/USDT:USDT`, `XRP/USDT:USDT`, `AVAX/USDT:USDT`, `LINK/USDT:USDT`)
- **Secrets redacted**: `True`

| Cycle | Name | OK | Reconcile | All 8 SYNCED | Equity Gap (USD) |
|---|---|---|---|---|---|
| 1 | Bootstrap & Reconcile | True | SYNCED | True | 0.0 |
| 2 | Live Dual-Leg Execution & Mid-Cycle Audit | True | SYNCED | True | N/A |
| 3 | Clean Flatten & Zero-Residual Verification | True | SYNCED | True | 0.0 |

## Cycle Details

### Cycle 1: Bootstrap & Reconcile

```json
{
  "credential_check": "enforced_64_64",
  "credentials_ok": true,
  "credential_format": {
    "spot": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    },
    "futures": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    }
  },
  "server_time_offsets_ms": {
    "spot": 1922,
    "futures": 1948
  },
  "warmup_gate": {
    "gate": "720h",
    "warmup_hours": 720.0,
    "required_hours": 720.0,
    "ok": true
  },
  "snapshot_healthy": true,
  "mid_price": 81642.42499999999,
  "reconcile_status": "SYNCED",
  "all_8_symbols_synced": true,
  "symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "equity_gap_usd": 0.0,
  "untracked_spot_count": 504,
  "bootstrap_baseline_symbols": 8,
  "bootstrap_equity_usd": 484112937.39886034
}
```

### Cycle 2: Live Dual-Leg Execution & Mid-Cycle Audit

```json
{
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "best_bid": 81642.42,
  "best_ask": 81642.43,
  "tick_size": 0.01,
  "maker_pricing": {
    "side": "BUY",
    "best_bid": 81642.42,
    "best_ask": 81642.43,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81642.42
  },
  "entry_status": "FILLED",
  "spot_entry_order_id": 5879294,
  "perp_entry_order_id": 28646229130,
  "executed_qty": 0.001,
  "perp_executed_qty": 0.001,
  "fill_price": 81642.43000000001,
  "lot_neutrality": {
    "passed": true,
    "invariant_name": "LOT_NEUTRALITY",
    "max_error": 0.0,
    "imbalance_usd": 0.0,
    "details": "[PASS] Lot neutrality verified: Leg A=81.64 USD, Leg B=81.64 USD, Imbalance=0.00 USD <= 10.00 USD."
  },
  "lot_neutral_ok": true,
  "mid_reconcile_status": "SYNCED",
  "mid_all_8_symbols_synced": true,
  "mid_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "futures_positions_verified": true,
  "equity_drift_vs_initial_usd": -0.03615158796310425
}
```

### Cycle 3: Clean Flatten & Zero-Residual Verification

```json
{
  "maker_pricing": {
    "side": "SELL",
    "best_bid": 81642.42,
    "best_ask": 81642.43,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81642.43
  },
  "exit_status": "FILLED",
  "spot_exit_order_id": 5879309,
  "perp_exit_order_id": 28646229270,
  "exit_executed_qty": 0.001,
  "post_reconcile_status": "SYNCED",
  "post_all_8_symbols_synced": true,
  "post_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "spot_qty_after": 1.0,
  "perp_qty_after": 0.0,
  "spot_residual_qty": 0.0,
  "perp_residual_qty": 0.0,
  "equity_gap_usd": 0.0,
  "residual_tolerance_qty": 2e-06,
  "zero_residual": true,
  "equity_gap_zero": true,
  "equity_drift_vs_initial_usd": -0.04329437017440796
}
```

_Log JSONL: `logs/sprint21_multi_cycle_burnin.jsonl`_

# Sprint 21 - Multi-Cycle Testnet Burn-In Evidence

- **Timestamp (UTC)**: `2026-10-22T16:40:11.226768+00:00`
- **Verdict**: `PASS`
- **Symbol**: `BTC/USDT:USDT`
- **Universe**: 8 symbols (`BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`, `DOGE/USDT:USDT`, `XRP/USDT:USDT`, `AVAX/USDT:USDT`, `LINK/USDT:USDT`)
- **Secrets redacted**: `True`

| Cycle | Name | OK | Reconcile | All 8 SYNCED | Equity Gap (USD) |
|---|---|---|---|---|---|
| 1 | Bootstrap & Reconcile | True | SYNCED | True | 0.0 |
| 2 | Live Dual-Leg Execution & Mid-Cycle Audit | True | SYNCED | True | N/A |
| 3 | Clean Flatten & Zero-Residual Verification | True | SYNCED | True | 0.0 |

## Cycle Details

### Cycle 1: Bootstrap & Reconcile

```json
{
  "credential_check": "enforced_64_64",
  "credentials_ok": true,
  "credential_format": {
    "spot": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    },
    "futures": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    }
  },
  "server_time_offsets_ms": {
    "spot": 2015,
    "futures": 1862
  },
  "warmup_gate": {
    "gate": "720h",
    "warmup_hours": 720.0,
    "required_hours": 720.0,
    "ok": true
  },
  "snapshot_healthy": true,
  "mid_price": 81538.005,
  "reconcile_status": "SYNCED",
  "all_8_symbols_synced": true,
  "symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "equity_gap_usd": 0.0,
  "untracked_spot_count": 504,
  "bootstrap_baseline_symbols": 8,
  "bootstrap_equity_usd": 483493831.1097642
}
```

### Cycle 2: Live Dual-Leg Execution & Mid-Cycle Audit

```json
{
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "best_bid": 81538.0,
  "best_ask": 81538.01,
  "tick_size": 0.01,
  "maker_pricing": {
    "side": "BUY",
    "best_bid": 81538.0,
    "best_ask": 81538.01,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81538.0
  },
  "entry_status": "FILLED",
  "spot_entry_order_id": 5884229,
  "perp_entry_order_id": 28646255760,
  "executed_qty": 0.001,
  "perp_executed_qty": 0.001,
  "fill_price": 81538.01,
  "lot_neutrality": {
    "passed": true,
    "invariant_name": "LOT_NEUTRALITY",
    "max_error": 0.0,
    "imbalance_usd": 0.0,
    "details": "[PASS] Lot neutrality verified: Leg A=81.54 USD, Leg B=81.54 USD, Imbalance=0.00 USD <= 10.00 USD."
  },
  "lot_neutral_ok": true,
  "mid_reconcile_status": "SYNCED",
  "mid_all_8_symbols_synced": true,
  "mid_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "futures_positions_verified": true,
  "equity_drift_vs_initial_usd": -0.06684255599975586
}
```

### Cycle 3: Clean Flatten & Zero-Residual Verification

```json
{
  "maker_pricing": {
    "side": "SELL",
    "best_bid": 81534.0,
    "best_ask": 81534.01,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81534.01
  },
  "exit_status": "FILLED",
  "spot_exit_order_id": 5884284,
  "perp_exit_order_id": 28646255827,
  "exit_executed_qty": 0.001,
  "post_reconcile_status": "SYNCED",
  "post_all_8_symbols_synced": true,
  "post_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "spot_qty_after": 1.0,
  "perp_qty_after": 0.0,
  "spot_residual_qty": 0.0,
  "perp_residual_qty": 0.0,
  "equity_gap_usd": 0.0,
  "residual_tolerance_qty": 2e-06,
  "zero_residual": true,
  "equity_gap_zero": true,
  "equity_drift_vs_initial_usd": -0.12283527851104736
}
```

_Log JSONL: `logs/sprint21_multi_cycle_burnin.jsonl`_

# Sprint 21 - Multi-Cycle Testnet Burn-In Evidence

- **Timestamp (UTC)**: `2026-10-22T16:57:36.112644+00:00`
- **Verdict**: `PASS`
- **Symbol**: `BTC/USDT:USDT`
- **Universe**: 8 symbols (`BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`, `DOGE/USDT:USDT`, `XRP/USDT:USDT`, `AVAX/USDT:USDT`, `LINK/USDT:USDT`)
- **Secrets redacted**: `True`

| Cycle | Name | OK | Reconcile | All 8 SYNCED | Equity Gap (USD) |
|---|---|---|---|---|---|
| 1 | Bootstrap & Reconcile | True | SYNCED | True | 0.0 |
| 2 | Live Dual-Leg Execution & Mid-Cycle Audit | True | SYNCED | True | N/A |
| 3 | Clean Flatten & Zero-Residual Verification | True | SYNCED | True | 0.0 |

## Cycle Details

### Cycle 1: Bootstrap & Reconcile

```json
{
  "credential_check": "enforced_64_64",
  "credentials_ok": true,
  "credential_format": {
    "spot": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    },
    "futures": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    }
  },
  "server_time_offsets_ms": {
    "spot": 1897,
    "futures": 1870
  },
  "warmup_gate": {
    "gate": "720h",
    "warmup_hours": 720.0,
    "required_hours": 720.0,
    "ok": true
  },
  "snapshot_healthy": true,
  "mid_price": 81167.39499999999,
  "reconcile_status": "SYNCED",
  "all_8_symbols_synced": true,
  "symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "equity_gap_usd": 0.0,
  "untracked_spot_count": 504,
  "bootstrap_baseline_symbols": 8,
  "bootstrap_equity_usd": 481296484.29692876
}
```

### Cycle 2: Live Dual-Leg Execution & Mid-Cycle Audit

```json
{
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "best_bid": 81167.39,
  "best_ask": 81167.4,
  "tick_size": 0.01,
  "maker_pricing": {
    "side": "BUY",
    "best_bid": 81167.39,
    "best_ask": 81167.4,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81167.39
  },
  "maker_buy_enabled": true,
  "entry_status": "FILLED",
  "spot_entry_order_id": 5889700,
  "perp_entry_order_id": 28646282116,
  "executed_qty": 0.001,
  "perp_executed_qty": 0.001,
  "fill_price": 81210.09999999999,
  "lot_neutrality": {
    "passed": true,
    "invariant_name": "LOT_NEUTRALITY",
    "max_error": 0.0,
    "imbalance_usd": 0.0,
    "details": "[PASS] Lot neutrality verified: Leg A=81.21 USD, Leg B=81.21 USD, Imbalance=0.00 USD <= 10.00 USD."
  },
  "lot_neutral_ok": true,
  "mid_reconcile_status": "SYNCED",
  "mid_all_8_symbols_synced": true,
  "mid_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "futures_positions_verified": true,
  "equity_drift_vs_initial_usd": -0.17148107290267944
}
```

### Cycle 3: Clean Flatten & Zero-Residual Verification

```json
{
  "maker_pricing": {
    "side": "SELL",
    "best_bid": 81210.09,
    "best_ask": 81210.1,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81210.1
  },
  "maker_sell_enabled": true,
  "exit_status": "FILLED",
  "spot_exit_order_id": 5889714,
  "perp_exit_order_id": 28646282232,
  "exit_executed_qty": 0.001,
  "exit_perp_executed_qty": 0.001,
  "post_reconcile_status": "SYNCED",
  "post_all_8_symbols_synced": true,
  "post_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "spot_qty_after": 1.0,
  "perp_qty_after": 0.0,
  "spot_residual_qty": 0.0,
  "perp_residual_qty": 0.0,
  "equity_gap_usd": 0.0,
  "residual_tolerance_qty": 2e-06,
  "zero_residual": true,
  "equity_gap_zero": true,
  "equity_drift_vs_initial_usd": -0.21936899423599243,
  "drift_tolerance_usd": 2.0,
  "equity_drift_ok": true
}
```

_Log JSONL: `logs/sprint21_multi_cycle_burnin.jsonl`_

# Sprint 21 - Multi-Cycle Testnet Burn-In Evidence

- **Timestamp (UTC)**: `2026-10-23T11:04:50.400032+00:00`
- **Verdict**: `PASS`
- **Symbol**: `BTC/USDT:USDT`
- **Universe**: 8 symbols (`BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BNB/USDT:USDT`, `DOGE/USDT:USDT`, `XRP/USDT:USDT`, `AVAX/USDT:USDT`, `LINK/USDT:USDT`)
- **Secrets redacted**: `True`

| Cycle | Name | OK | Reconcile | All 8 SYNCED | Equity Gap (USD) |
|---|---|---|---|---|---|
| 1 | Bootstrap & Reconcile | True | SYNCED | True | 0.0 |
| 2 | Live Dual-Leg Execution & Mid-Cycle Audit | True | SYNCED | True | N/A |
| 3 | Clean Flatten & Zero-Residual Verification | True | SYNCED | True | 0.0 |

## Cycle Details

### Cycle 1: Bootstrap & Reconcile

```json
{
  "credential_check": "enforced_64_64",
  "credentials_ok": true,
  "credential_format": {
    "spot": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    },
    "futures": {
      "ok": true,
      "key_length": 64,
      "secret_length": 64
    }
  },
  "server_time_offsets_ms": {
    "spot": 1531,
    "futures": 1565
  },
  "warmup_gate": {
    "gate": "720h",
    "warmup_hours": 720.0,
    "required_hours": 720.0,
    "ok": true
  },
  "snapshot_healthy": true,
  "mid_price": 81658.005,
  "reconcile_status": "SYNCED",
  "all_8_symbols_synced": true,
  "symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "equity_gap_usd": 0.0,
  "untracked_spot_count": 504,
  "bootstrap_baseline_symbols": 8,
  "bootstrap_equity_usd": 484205310.76755995
}
```

### Cycle 2: Live Dual-Leg Execution & Mid-Cycle Audit

```json
{
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "best_bid": 81658.0,
  "best_ask": 81658.01,
  "tick_size": 0.01,
  "maker_pricing": {
    "side": "BUY",
    "best_bid": 81658.0,
    "best_ask": 81658.01,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81658.0
  },
  "maker_buy_enabled": true,
  "entry_status": "FILLED",
  "spot_entry_order_id": 6145739,
  "perp_entry_order_id": 28647724834,
  "executed_qty": 0.001,
  "perp_executed_qty": 0.001,
  "fill_price": 81658.0,
  "lot_neutrality": {
    "passed": true,
    "invariant_name": "LOT_NEUTRALITY",
    "max_error": 0.0,
    "imbalance_usd": 0.0,
    "details": "[PASS] Lot neutrality verified: Leg A=81.66 USD, Leg B=81.66 USD, Imbalance=0.00 USD <= 10.00 USD."
  },
  "lot_neutral_ok": true,
  "mid_reconcile_status": "SYNCED",
  "mid_all_8_symbols_synced": true,
  "mid_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "futures_positions_verified": true,
  "equity_drift_vs_initial_usd": -0.03332740068435669
}
```

### Cycle 3: Clean Flatten & Zero-Residual Verification

```json
{
  "maker_pricing": {
    "side": "SELL",
    "best_bid": 81629.48,
    "best_ask": 81629.49,
    "tick_size": 0.01,
    "spread_ticks": 1,
    "improve_inside_spread": true,
    "improved": false,
    "mode": "TOUCH_BEST",
    "price": 81629.49
  },
  "maker_sell_enabled": true,
  "exit_status": "FILLED",
  "spot_exit_order_id": 6145802,
  "perp_exit_order_id": 28647724892,
  "exit_executed_qty": 0.001,
  "exit_perp_executed_qty": 0.001,
  "post_reconcile_status": "SYNCED",
  "post_all_8_symbols_synced": true,
  "post_symbol_statuses": {
    "AVAX/USDT:USDT": "SYNCED",
    "BNB/USDT:USDT": "SYNCED",
    "BTC/USDT:USDT": "SYNCED",
    "DOGE/USDT:USDT": "SYNCED",
    "ETH/USDT:USDT": "SYNCED",
    "LINK/USDT:USDT": "SYNCED",
    "SOL/USDT:USDT": "SYNCED",
    "XRP/USDT:USDT": "SYNCED"
  },
  "baseline_spot_qty": 1.0,
  "baseline_perp_qty": 0.0,
  "spot_qty_after": 1.0,
  "perp_qty_after": 0.0,
  "spot_residual_qty": 0.0,
  "perp_residual_qty": 0.0,
  "equity_gap_usd": 0.0,
  "residual_tolerance_qty": 2e-06,
  "zero_residual": true,
  "equity_gap_zero": true,
  "equity_drift_vs_initial_usd": -0.09448504447937012,
  "drift_tolerance_usd": 2.0,
  "equity_drift_ok": true
}
```

_Log JSONL: `logs/sprint22_multi_cycle_burnin.jsonl`_

