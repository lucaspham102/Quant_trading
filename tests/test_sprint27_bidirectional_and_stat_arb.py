"""Sprint 27 & 28: bidirectional trend gating, macro regime filter, reversal, 1k profiles."""
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.directional_trend_sleeve import (  # noqa: E402
    TrendSleeveConfig,
    TrendSleeveState,
    compute_triple_confirmed_trend_weights,
    kaufman_efficiency_ratio,
    trend_weights_to_target_qty,
)
from src.stat_arb_sleeve import (  # noqa: E402
    ReversalSleeveConfig,
    compute_funding_conditioned_reversal_weights,
    reversal_weights_to_target_qty,
)
from src.strategy_profiles import (  # noqa: E402
    PROFILE_REGISTRY,
    enforce_split_wallet_futures_cap,
    get_strategy_profile,
)

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
N_BARS = 2000
NEW_PROFILES = ("bi_trend_carry_1k", "carry_stat_arb_1k", "regime_omni_1k")
ALL_S27_S28_PROFILES = (
    "bi_trend_carry_1k",
    "carry_stat_arb_1k",
    "regime_omni_1k",
    "flexible_regime_1k",
)


def _walk(drift: float, seed: int, n: int = N_BARS, sigma: float = 0.004) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = drift + sigma * rng.standard_normal(n)
    return pd.Series(100.0 * np.exp(np.cumsum(rets)))


def _bear_prices() -> dict:
    return {BTC: _walk(-0.0006, 1), ETH: _walk(-0.0006, 2)}


def _short_cfg(**kw) -> TrendSleeveConfig:
    return TrendSleeveConfig(allow_short_trend=True, **kw)


# --------------------------------------------------------------------------- 1) short trend
def test_short_trend_enters_in_bear_regime_with_neutral_funding():
    w = compute_triple_confirmed_trend_weights(
        _bear_prices(), N_BARS - 1, _short_cfg(), TrendSleeveState(),
        funding_bps={BTC: 0.0, ETH: 0.0},
    )
    assert w, "bear breakdown with funding ~0 must open shorts"
    assert all(v < 0.0 for v in w.values())
    assert sum(abs(v) for v in w.values()) <= 0.55 + 1e-9


def test_short_trend_disabled_is_long_only():
    w = compute_triple_confirmed_trend_weights(
        _bear_prices(), N_BARS - 1, TrendSleeveConfig(), TrendSleeveState(),
        funding_bps={BTC: 0.0, ETH: 0.0},
    )
    assert all(v > 0.0 for v in w.values())
    assert w == {}


def test_short_trend_blocked_when_btc_not_bear():
    prices = {BTC: _walk(0.0006, 3), ETH: _walk(-0.0006, 2)}
    w = compute_triple_confirmed_trend_weights(
        prices, N_BARS - 1, _short_cfg(), TrendSleeveState(),
        funding_bps={BTC: 0.0, ETH: 0.0},
    )
    assert not any(v < 0.0 for v in w.values())
    assert ETH not in w


def test_short_trend_blocked_by_negative_funding_below_floor():
    cfg = _short_cfg()
    assert cfg.funding_short_entry_floor_bps == pytest.approx(-0.25)
    w = compute_triple_confirmed_trend_weights(
        _bear_prices(), N_BARS - 1, cfg, TrendSleeveState(),
        funding_bps={BTC: -0.30, ETH: -0.30},
    )
    assert w == {}


def test_short_trend_allowed_at_funding_floor_boundary():
    w = compute_triple_confirmed_trend_weights(
        _bear_prices(), N_BARS - 1, _short_cfg(), TrendSleeveState(),
        funding_bps={BTC: -0.25, ETH: -0.25},
    )
    assert w and all(v < 0.0 for v in w.values())


def test_short_trend_requires_conviction_z_short():
    # Dang xuong rat nhe -> |Z_short| < 1.10 -> khong vao short.
    prices = {BTC: _walk(-0.00003, 5, sigma=0.001), ETH: _walk(-0.00003, 6, sigma=0.001)}
    w = compute_triple_confirmed_trend_weights(
        prices, N_BARS - 1, _short_cfg(z_short_entry=50.0), TrendSleeveState(),
        funding_bps={BTC: 0.0, ETH: 0.0},
    )
    assert w == {}


def test_short_trend_anti_squeeze_exit_on_deep_negative_funding():
    cfg = _short_cfg()
    state = TrendSleeveState()
    prices = _bear_prices()
    w1 = compute_triple_confirmed_trend_weights(
        prices, N_BARS - 1, cfg, state, funding_bps={BTC: 0.0, ETH: 0.0},
    )
    assert w1 and all(state.side[s] == -1 for s in w1)
    w2 = compute_triple_confirmed_trend_weights(
        prices, N_BARS - 1, cfg, state,
        funding_bps={BTC: -0.60, ETH: -0.60},
    )
    assert w2 == {}
    assert not any(state.active.get(s, False) for s in (BTC, ETH))


def test_short_trend_holds_while_funding_above_exit_threshold():
    cfg = _short_cfg()
    state = TrendSleeveState()
    prices = _bear_prices()
    compute_triple_confirmed_trend_weights(
        prices, N_BARS - 1, cfg, state, funding_bps={BTC: 0.0, ETH: 0.0},
    )
    w2 = compute_triple_confirmed_trend_weights(
        prices, N_BARS - 1, cfg, state, funding_bps={BTC: -0.40, ETH: -0.40},
    )
    assert w2 and all(v < 0.0 for v in w2.values())


def test_short_trend_chandelier_stop_on_price_spike():
    cfg = _short_cfg()
    state = TrendSleeveState()
    prices = _bear_prices()
    w1 = compute_triple_confirmed_trend_weights(
        prices, N_BARS - 1, cfg, state, funding_bps={BTC: 0.0, ETH: 0.0},
    )
    assert w1
    spiked = {s: ser.copy() for s, ser in prices.items()}
    for s in spiked:
        spiked[s].iloc[-1] = float(spiked[s].iloc[-1]) * 1.30
    w2 = compute_triple_confirmed_trend_weights(
        spiked, N_BARS - 1, cfg, state, funding_bps={BTC: 0.0, ETH: 0.0},
    )
    assert w2 == {}
    assert not any(state.active.get(s, False) for s in (BTC, ETH))
    assert not state.trough_close


def test_short_trend_is_causal_no_lookahead():
    cfg = _short_cfg()
    base = _bear_prices()
    w_a = compute_triple_confirmed_trend_weights(
        base, 1800, cfg, TrendSleeveState(), funding_bps={BTC: 0.0, ETH: 0.0},
    )
    mutated = {s: ser.copy() for s, ser in base.items()}
    for s in mutated:
        mutated[s].iloc[1801:] = 1e6
    w_b = compute_triple_confirmed_trend_weights(
        mutated, 1800, cfg, TrendSleeveState(), funding_bps={BTC: 0.0, ETH: 0.0},
    )
    assert w_a == w_b


def test_trend_target_qty_signed_for_short_weights():
    q = trend_weights_to_target_qty(
        {BTC: -0.2, ETH: 0.2}, {BTC: 100.0, ETH: 100.0}, 1000.0, {BTC: 0.001, ETH: 0.001},
    )
    assert q[BTC] < 0.0 < q[ETH]
    assert abs(q[BTC]) == pytest.approx(q[ETH])


# ------------------------------------------- 1b) macro efficiency & breadth regime filter
def test_kaufman_efficiency_ratio_extremes():
    smooth = pd.Series(np.linspace(100.0, 200.0, 400))
    assert kaufman_efficiency_ratio(smooth, 336) == pytest.approx(1.0)
    choppy = pd.Series([100.0 + (1.0 if i % 2 else -1.0) for i in range(400)])
    assert kaufman_efficiency_ratio(choppy, 336) < 0.01
    assert kaufman_efficiency_ratio(smooth.iloc[:100], 336) == 0.0


def test_macro_regime_filter_long_breadth_and_er_gate():
    syms = [BTC, ETH, "SOL/USDT:USDT", "BNB/USDT:USDT"]
    up = _walk(0.0012, 10, sigma=0.0015)
    down = _walk(-0.0012, 11, sigma=0.0015)
    # 3/4 symbols in uptrend -> bull_breadth = 0.75 >= 0.375
    prices_ok = {BTC: up, ETH: up.copy(), "SOL/USDT:USDT": up.copy(), "BNB/USDT:USDT": down}
    cfg = TrendSleeveConfig(
        allow_short_trend=True,
        require_macro_regime_filter=True,
        short_gross_scale=0.65,
    )
    assert cfg.min_bull_breadth == pytest.approx(0.375)
    w_ok = compute_triple_confirmed_trend_weights(
        prices_ok, N_BARS - 1, cfg, TrendSleeveState(),
    )
    assert w_ok and all(v > 0.0 for v in w_ok.values())

    # Only 1/4 symbol above EMA_med -> bull_breadth = 0.25 < 0.375 -> Long blocked!
    prices_low_breadth = {
        BTC: up,
        ETH: down.copy(),
        "SOL/USDT:USDT": down.copy(),
        "BNB/USDT:USDT": down.copy(),
    }
    w_blocked = compute_triple_confirmed_trend_weights(
        prices_low_breadth, N_BARS - 1, cfg, TrendSleeveState(),
    )
    assert not any(v > 0.0 for v in w_blocked.values())
    assert syms[0] not in w_blocked


def test_macro_regime_filter_short_synchronized_panic_and_immediate_exit():
    s_down = _walk(-0.0012, 21, sigma=0.0015)
    s_up = _walk(0.0012, 22, sigma=0.0015)
    syms = [
        BTC,
        ETH,
        "SOL/USDT:USDT",
        "BNB/USDT:USDT",
        "LINK/USDT:USDT",
        "AVAX/USDT:USDT",
        "XRP/USDT:USDT",
        "DOGE/USDT:USDT",
    ]
    # 7/8 symbols below EMA_med & EMA_slow -> bear_breadth = 0.875 >= 0.75
    panic_prices = {s: (s_up.copy() if s == "DOGE/USDT:USDT" else s_down.copy()) for s in syms}
    cfg_macro = TrendSleeveConfig(
        allow_short_trend=True,
        require_macro_regime_filter=True,
        short_gross_scale=0.65,
        max_gross_pct=0.70,
    )
    cfg_unscaled = TrendSleeveConfig(
        allow_short_trend=True,
        require_macro_regime_filter=False,
        short_gross_scale=1.0,
        max_gross_pct=0.70,
    )
    state = TrendSleeveState()
    w_panic = compute_triple_confirmed_trend_weights(
        panic_prices, N_BARS - 1, cfg_macro, state,
    )
    w_full = compute_triple_confirmed_trend_weights(
        panic_prices, N_BARS - 1, cfg_unscaled, TrendSleeveState(),
    )
    assert w_panic and all(v < 0.0 for v in w_panic.values())
    assert sum(abs(v) for v in w_panic.values()) == pytest.approx(
        0.65 * sum(abs(v) for v in w_full.values()), rel=1e-6
    )

    # 5/8 bear breadth = 0.625 < 0.75 -> blocks new Short entries
    mild_bear = {
        s: (s_up.copy() if i >= 5 else s_down.copy()) for i, s in enumerate(syms)
    }
    assert (
        compute_triple_confirmed_trend_weights(
            mild_bear, N_BARS - 1, cfg_macro, TrendSleeveState(),
        )
        == {}
    )

    # Immediate exit when bear_breadth drops < 0.50 (even if ETH itself is still falling)
    broad_rebound = {
        s: (s_down.copy() if i < 3 else s_up.copy()) for i, s in enumerate(syms)
    }
    w_after_rebound = compute_triple_confirmed_trend_weights(
        broad_rebound, N_BARS - 1, cfg_macro, state,
    )
    assert not any(v < 0.0 for v in w_after_rebound.values())

    # Immediate exit when BTC rebounds above EMA_fast (168h) even if 7/8 altcoins stay bear
    state2 = TrendSleeveState()
    compute_triple_confirmed_trend_weights(panic_prices, N_BARS - 1, cfg_macro, state2)
    btc_rebound = {s: ser.copy() for s, ser in panic_prices.items()}
    btc_rebound[BTC].iloc[-1] = float(btc_rebound[BTC].iloc[-200]) * 1.25
    w_btc_bounce = compute_triple_confirmed_trend_weights(
        btc_rebound, N_BARS - 1, cfg_macro, state2,
    )
    assert w_btc_bounce == {}


# --------------------------------------------------------------------------- 2) reversal
REV_SYMS = [f"S{i}/USDT:USDT" for i in range(8)]


def _reversal_prices(n: int = 400, up: str = "S0/USDT:USDT", down: str = "S1/USDT:USDT") -> dict:
    out = {}
    for i, s in enumerate(REV_SYMS):
        rng = np.random.default_rng(100 + i)
        rets = 0.003 * rng.standard_normal(n)
        if s == up:
            rets[-120:] += 0.003
        if s == down:
            rets[-120:] -= 0.003
        out[s] = pd.Series(100.0 * np.exp(np.cumsum(rets)))
    return out


def test_reversal_shorts_overextended_and_longs_oversold():
    cfg = ReversalSleeveConfig(capital_usd=1000.0)
    w = compute_funding_conditioned_reversal_weights(
        _reversal_prices(), 399, funding_bps={}, config=cfg,
    )
    assert w
    assert w["S0/USDT:USDT"] < 0.0
    assert w["S1/USDT:USDT"] > 0.0


def test_reversal_is_dollar_neutral_within_ten_usd():
    cfg = ReversalSleeveConfig(capital_usd=1000.0)
    prices = _reversal_prices()
    w = compute_funding_conditioned_reversal_weights(prices, 399, funding_bps={}, config=cfg)
    assert w
    assert abs(sum(w.values())) <= 1e-12
    px = {s: float(ser.iloc[-1]) for s, ser in prices.items()}
    qty = reversal_weights_to_target_qty(w, px, 1000.0, {s: 0.001 for s in px})
    long_usd = sum(q * px[s] for s, q in qty.items() if q > 0.0)
    short_usd = sum(-q * px[s] for s, q in qty.items() if q < 0.0)
    assert long_usd > 0.0 and short_usd > 0.0
    assert abs(long_usd - short_usd) <= 10.0
    assert long_usd + short_usd <= 1000.0 * cfg.gross_pct + 1e-6


def test_reversal_is_causal_no_lookahead():
    cfg = ReversalSleeveConfig(capital_usd=1000.0)
    base = _reversal_prices(n=420)
    w_a = compute_funding_conditioned_reversal_weights(base, 399, funding_bps={}, config=cfg)
    mutated = {s: ser.copy() for s, ser in base.items()}
    for s in mutated:
        mutated[s].iloc[400:] = mutated[s].iloc[400:] * 50.0
    w_b = compute_funding_conditioned_reversal_weights(mutated, 399, funding_bps={}, config=cfg)
    assert w_a == w_b


def test_reversal_flat_when_market_quiet_or_history_short():
    cfg = ReversalSleeveConfig(capital_usd=1000.0)
    flat = {s: pd.Series([100.0] * 400) for s in REV_SYMS}
    assert compute_funding_conditioned_reversal_weights(flat, 399, {}, cfg) == {}
    assert compute_funding_conditioned_reversal_weights(_reversal_prices(), 50, {}, cfg) == {}


def test_reversal_funding_tilt_changes_score_but_keeps_neutrality():
    cfg = ReversalSleeveConfig(capital_usd=1000.0)
    prices = _reversal_prices()
    funding = {s: (1.0 if s == "S0/USDT:USDT" else -1.0 if s == "S1/USDT:USDT" else 0.0)
               for s in REV_SYMS}
    w = compute_funding_conditioned_reversal_weights(prices, 399, funding, cfg)
    assert w
    assert abs(sum(w.values())) <= 1e-12


def test_reversal_config_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ReversalSleeveConfig(gross_pct=0.0)
    with pytest.raises(ValueError):
        ReversalSleeveConfig(funding_tilt=1.5)


# --------------------------------------------------------------------------- 3) profiles
@pytest.mark.parametrize("name", NEW_PROFILES)
def test_new_profiles_registered_with_550_450_split(name):
    assert name in PROFILE_REGISTRY
    p = get_strategy_profile(name)
    assert p.capital_usd == pytest.approx(1000.0)
    assert p.spot_wallet_usd == pytest.approx(550.0)
    assert p.futures_wallet_usd == pytest.approx(450.0)
    assert p.max_futures_leverage <= 3.0
    assert p.futures_wallet_usd * p.max_futures_leverage == pytest.approx(1350.0)


def test_flexible_regime_1k_profile_registered():
    assert "flexible_regime_1k" in PROFILE_REGISTRY
    p = get_strategy_profile("flexible_regime_1k")
    assert p.capital_usd == pytest.approx(1000.0)
    assert p.spot_wallet_usd == pytest.approx(600.0)
    assert p.futures_wallet_usd == pytest.approx(400.0)
    assert p.carry_eta == pytest.approx(0.60)
    assert p.carry_top_k == 3
    assert p.carry_min_hold_bars == 63
    assert p.enable_trend_sleeve is True
    assert p.allow_short_trend is True
    assert p.require_macro_regime_filter is True
    assert p.short_gross_scale == pytest.approx(0.65)
    assert p.trend_target_vol_ann == pytest.approx(0.15)
    assert p.trend_max_gross_pct == pytest.approx(0.70)
    assert p.enable_yield_stacking is True


def test_new_profiles_feature_flags():
    bi = PROFILE_REGISTRY["bi_trend_carry_1k"]
    assert bi.enable_trend_sleeve and bi.allow_short_trend and not bi.enable_reversal_sleeve
    assert not bi.require_macro_regime_filter and bi.short_gross_scale == pytest.approx(1.0)
    arb = PROFILE_REGISTRY["carry_stat_arb_1k"]
    assert arb.enable_reversal_sleeve and not arb.enable_trend_sleeve
    assert arb.reversal_gross_pct > 0.0
    omni = PROFILE_REGISTRY["regime_omni_1k"]
    assert omni.enable_trend_sleeve and omni.allow_short_trend and omni.enable_reversal_sleeve
    assert omni.dynamic_carry_to_alpha_margin


def test_profile_validation_rejects_inconsistent_sprint27_flags():
    arb = PROFILE_REGISTRY["carry_stat_arb_1k"]
    with pytest.raises(ValueError):
        dataclasses.replace(arb, allow_short_trend=True)
    with pytest.raises(ValueError):
        dataclasses.replace(arb, require_macro_regime_filter=True)
    with pytest.raises(ValueError):
        dataclasses.replace(arb, short_gross_scale=1.5)
    with pytest.raises(ValueError):
        dataclasses.replace(arb, reversal_gross_pct=0.0)
    bi = PROFILE_REGISTRY["bi_trend_carry_1k"]
    with pytest.raises(ValueError):
        dataclasses.replace(bi, spot_wallet_usd=600.0)


@pytest.mark.parametrize("name", ALL_S27_S28_PROFILES)
def test_new_profiles_leverage_cap_never_exceeds_3x(name):
    p = PROFILE_REGISTRY[name]
    res = enforce_split_wallet_futures_cap(
        carry_perp_notional_usd=900.0,
        desired_overlay_gross_usd=1000.0,
        futures_wallet_equity_usd=p.futures_wallet_usd,
        max_futures_leverage=p.max_futures_leverage,
    )
    assert res["effective_futures_leverage"] <= 3.0 + 1e-9
    assert res["allowed_overlay_gross_usd"] <= p.futures_wallet_usd * 3.0 - 900.0 + 1e-9


# --------------------------------------------------------------------------- tournament script
def _load_tournament():
    path = _ROOT / "scripts" / "run_1k_profile_tournament_backtest.py"
    spec = importlib.util.spec_from_file_location("tournament_s27", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tournament_s27"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tournament_lists_new_profiles():
    mod = _load_tournament()
    available = mod.build_tournament_profiles()
    for name in ALL_S27_S28_PROFILES:
        assert name in available
        assert name in mod.DEFAULT_PROFILES.split(",")
        assert name in mod.PROFILE_ORDER


def _synthetic_market(n: int = 2200) -> dict:
    syms = [BTC, ETH, "SOL/USDT:USDT", "BNB/USDT:USDT", "LINK/USDT:USDT"]
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    perp = {}
    for i, s in enumerate(syms):
        perp[s] = _walk(-0.0002, 200 + i, n=n, sigma=0.006).to_numpy()
    perp_df = pd.DataFrame(perp, index=index)[syms]
    is_f = np.isin(index.hour, [0, 8, 16]) & (index.minute == 0)
    fund = pd.DataFrame(
        np.where(is_f[:, None], 0.0001, 0.0) * np.ones((n, len(syms))), index=index, columns=syms,
    )
    zeros = pd.DataFrame(0.0, index=index, columns=syms)
    return {
        "index": index,
        "perp_close": perp_df,
        "spot_close": perp_df.copy(),
        "funding_event": fund,
        "carry_score": zeros.copy(),
        "funding_bps": fund * 10000.0,
        "snr": zeros.copy(),
        "snr_dispersion": pd.Series(0.0, index=index),
    }


@pytest.mark.parametrize("name", ALL_S27_S28_PROFILES)
def test_tournament_simulation_accounting_and_leverage(name):
    mod = _load_tournament()
    market = _synthetic_market()
    res = mod.simulate_profile(PROFILE_REGISTRY[name], market)
    assert res["accounting_delta_usd"] == 0.0
    assert res["short_trend_pnl_usd"] + res["long_trend_pnl_usd"] == pytest.approx(
        res["trend_pnl_usd"], abs=1e-6,
    )
    assert res["max_observed_futures_leverage"] <= 3.0 + 1e-6
    assert res["max_reversal_imbalance_usd"] <= 10.0 + 1e-6 or res["reversal_skipped_imbalance"] > 0
    if not PROFILE_REGISTRY[name].enable_reversal_sleeve:
        assert res["reversal_pnl_usd"] == 0.0
    if not PROFILE_REGISTRY[name].enable_trend_sleeve:
        assert res["trend_pnl_usd"] == 0.0
