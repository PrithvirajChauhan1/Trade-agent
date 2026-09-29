"""
Is the frozen algorithm's OOS edge real, or just market drift + luck?
Compares the strategy's OOS trades with RANDOM entries (same count, same direction mix, same
12%/7% exits, same OOS window). Reports the unconditional base rate and a permutation p-value.
Usage: python baseline_check.py --source yahoo --symbol ^BSESN
"""
import sys, argparse, warnings; sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import numpy as np
from features.indicators import compute_all
from features.custom_indicators import apply_custom_indicators
from learn.combos import attach_combo_columns, forward_outcomes
from learn.freeze import load_frozen
from strategy.ensemble import ComboEnsembleStrategy
from backtest.fixed_exit import run_fixed_exit_backtest
from risk.engine import RiskConfig

ap = argparse.ArgumentParser(); ap.add_argument("--source", default="demo", choices=["demo", "yahoo"])
ap.add_argument("--symbol", default="^BSESN"); ap.add_argument("--n", type=int, default=10000); a = ap.parse_args()
if a.source == "yahoo":
    from data.provider import get_history; raw = get_history(a.symbol, "2000-01-01")
else:
    from data.demo_series import make_demo_full; raw = make_demo_full(); print("DEMO — SYNTHETIC DATA")
params, combos, meta, sha = load_frozen("frozen/algo_v1.json")
full = attach_combo_columns(apply_custom_indicators(compute_all(raw), params), combos)
oos = run_fixed_exit_backtest(full.loc["2020-01-01":"2025-12-31"], [ComboEnsembleStrategy(combos)],
                              RiskConfig(min_risk_reward=1.5), max_concurrent=3)
ts = oos.trades; n = len(ts); assert n, "no OOS trades"
lo = full.index.get_loc(full.loc["2020-01-01":].index[0]); hi = full.index.get_loc(full.loc[:"2025-12-31"].index[-1])
out = {d: forward_outcomes(full, d) for d in ("long", "short")}
pool = {d: (lambda x: x[~np.isnan(x)])(out[d][lo:hi + 1]) for d in out}
nl = sum(t.direction == "long" for t in ts); ns = n - nl
obs_mean = float(np.mean([t.net_pct for t in ts])); obs_wr = float(np.mean([t.win for t in ts]))
rng = np.random.default_rng(0); sims = np.empty(a.n); wrs = np.empty(a.n)
for k in range(a.n):
    draw = np.concatenate([rng.choice(pool["long"], nl), rng.choice(pool["short"], ns) if ns else []])
    sims[k] = draw.mean(); wrs[k] = (draw > 0).mean()
print(f"\nStrategy OOS: {n} trades ({nl} long / {ns} short)  win {obs_wr*100:.1f}%  exp {obs_mean*100:+.2f}%/trade")
print(f"Random entries, same exits/window: win {np.mean(wrs)*100:.1f}% (95% range {np.percentile(wrs,2.5)*100:.0f}-{np.percentile(wrs,97.5)*100:.0f}%)"
      f"  exp {sims.mean()*100:+.2f}%/trade")
print(f"Base rate — ALL long entries in window: win {(pool['long']>0).mean()*100:.1f}%  exp {pool['long'].mean()*100:+.2f}%")
p = float((sims >= obs_mean).mean()); print(f"Permutation p-value (strategy beats random entries): {p:.4f}")
print("VERDICT:", "edge over random entries is statistically visible" if p < 0.05 else
      "NOT distinguishable from random entries — the win rate is mostly market drift / luck")
