import sys; sys.path.insert(0, ".")
import dataclasses, numpy as np, pandas as pd
from data.synthetic_5m import generate_5m
from data.synthetic_ctx import make_ctx
from features.intraday import build_intraday_features
from features.extra_indicators import add_extra
from features.context import add_context
from patterns.chart_patterns import add_pattern_columns, detect
from strategy.intraday_components import generate_signals, CANDIDATES
from backtest.intraday import run_intraday, CostModel
from research.ml import meta_label
from research.ablation import ablate
from research.study import sl, expR

RAW = generate_5m(40, seed=5)
EXT = add_pattern_columns(add_extra(build_intraday_features(RAW)))
def ext(raw): return add_pattern_columns(add_extra(build_intraday_features(raw)))

def test_extra_indicators_and_patterns_no_lookahead():
    cols = ["st_dir", "psar", "ichi_pos", "aroon_osc", "mfi_14", "cci_20", "hist_vol", "hma_21", "stochrsi", "kc_pos", "avwap_prev_low", "rsi_div_bull", "obv_div_bear"]
    pat = [c for c in EXT.columns if c.startswith("pat_")]
    for t in (431, 1333, 2010):
        tr = ext(RAW.iloc[:t + 1])
        for c in cols:
            a, b = EXT[c].iloc[t], tr[c].iloc[t]; assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b, rtol=1e-9), (c, t, a, b)
        for c in pat: assert EXT[c].iloc[t] == tr[c].iloc[t], (c, t)

def test_context_is_causal():
    ctx = make_ctx(EXT); T = 1500; ts = EXT.index[T]; base = add_context(EXT, ctx)
    bad = {k: v.copy() for k, v in ctx.items()}
    for k in ("index", "tech", "sector"): bad[k].loc[bad[k].index > ts, "close"] *= 3
    bad["vol"].loc[bad["vol"].index.date >= ts.date(), "close"] = 999.0
    alt = add_context(EXT, bad)
    for c in [c for c in base.columns if c.startswith("ctx_") or c == "rel_strength_12"]:
        a, b = base[c].iloc[:T + 1], alt[c].iloc[:T + 1]; assert ((a == b) | (a.isna() & b.isna())).all(), c

def test_market_context_component_gates_direction():
    f = add_context(EXT, make_ctx(EXT)); cfg = {"direction": "both", "entry": {"mkt_ctx": {"index": True}}, "min_bar_of_day": 0, "no_entry_last_bars": 0}
    L, S, _ = generate_signals(f, cfg); assert (f.ctx_index_trend.values[L] > 0).all() and (f.ctx_index_trend.values[S] < 0).all() and L.any() and S.any()

def test_double_bottom_detected_at_neckline_break_only():
    p = np.interp(np.arange(60), [0, 10, 20, 30, 40, 59], [100, 90, 96, 90.1, 97, 98])
    df = pd.DataFrame({"high": p + .2, "low": p - .2, "close": p, "atr_14": 1.0, "session_id": 0})
    F = detect(df); hit = np.flatnonzero(F["double_bottom"])
    assert len(hit) >= 1 and hit.min() >= 36 and hit.min() <= 42      # fires when the neckline breaks, never before the 2nd low is confirmed

def test_ml_ignores_test_outcomes_when_fitting_and_choosing_threshold():
    raw = generate_5m(150, seed=9); f = ext(raw); ns = int(f.session_id.max()) + 1
    L, S, sc = generate_signals(f, CANDIDATES[0]); T = run_intraday(f, L, S, sc, stop_atr=1.5, target_rr=2.0, max_hold_bars=6, cost=CostModel())
    rg = {"train": (0, int(ns * .5)), "val": (int(ns * .5), int(ns * .75)), "test": (int(ns * .75), ns)}
    a = meta_label(f, T, rg, "X"); T2 = [dataclasses.replace(t, net_r=-1.0) if t.session_idx >= rg["test"][0] else t for t in T]; b = meta_label(f, T2, rg, "X")
    assert a[0]["status"] != "-" and len(a) == len(b)
    for x, y in zip(a, b): assert x["auc_val"] == y["auc_val"] and x["threshold"] == y["threshold"] and x["model"] == y["model"]

def test_ablation_variant_count_and_full_matches_direct_run():
    f = ext(generate_5m(60, seed=2)); cfg = {**CANDIDATES[0], "entry": {"ema_align": True, "vwap_side": True, "adx_min": 20}}; c = (6, 2.0, 1.5, None)
    rg = {"val": (30, 45), "test": (45, 60)}; rows = ablate(f, cfg, c, rg, CostModel(), "K")
    assert len(rows) == 1 + 3 + 3 + 3
    L, S, sc = generate_signals(f, cfg); T = run_intraday(f, L, S, sc, stop_atr=1.5, target_rr=2.0, max_hold_bars=6, cost=CostModel())
    v = sl(T, 30, 45); assert rows[0]["variant"] == "FULL" and rows[0]["n_val"] == len(v) and (not v or abs(rows[0]["exp_val_R"] - round(expR(v), 3)) < 1e-9)

def test_combo_outcomes_use_next_bar_open_execution():
    from learn.combos import forward_outcomes
    # Signal bar closes at 100, but the next bar opens at 110. A 5% target
    # should therefore NOT be credited from the signal-bar close.
    df = pd.DataFrame({
        "open":  [100.0, 110.0, 110.0],
        "high":  [101.0, 111.0, 111.0],
        "low":   [99.0, 109.0, 109.0],
        "close": [100.0, 110.0, 110.0],
        "session_id": [0, 0, 0],
    })
    out = forward_outcomes(df, "long", tp=0.05, sl=0.05, horizon=1, cost=0.0)
    assert np.isnan(out[0]), "target must be evaluated from next-bar open, not signal-bar close"
