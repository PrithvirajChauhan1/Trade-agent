"""Component ablation + single/pair/full comparison under the strategy's FROZEN exit config. Decisions use VALIDATION; test shown for information only."""
from itertools import combinations
import numpy as np
from strategy.intraday_components import generate_signals
from backtest.intraday import run_intraday
from research.study import sl, expR

def ablate(f, cfg, c, ranges, cost, key):
    hold, rr, st, tr = c; comps = list(cfg["entry"]); rows = []
    def run(entry, label):
        c2 = {**cfg, "entry": {k: cfg["entry"][k] for k in entry}}; L, S, sc = generate_signals(f, c2)
        T = run_intraday(f, L, S, sc, stop_atr=st, target_rr=rr, max_hold_bars=hold, trail_atr=tr, cost=cost); v, t = sl(T, *ranges["val"]), sl(T, *ranges["test"])
        rows.append({"strategy": key, "variant": label, "n_val": len(v), "exp_val_R": round(expR(v), 3) if v else np.nan, "n_test_info": len(t), "exp_test_R_info": round(expR(t), 3) if t else np.nan})
    run(comps, "FULL")
    if len(comps) > 1:
        for k in comps: run([x for x in comps if x != k], f"minus:{k}")
    for k in comps: run([k], f"only:{k}")
    for a, b in combinations(comps, 2): run([a, b], f"pair:{a}+{b}")
    return rows
