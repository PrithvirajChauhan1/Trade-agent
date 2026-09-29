"""
learn(2000-2019) -> freeze -> OOS(2020-2025) -> regime edge -> apply to 2026 -> tech signal
-> fundamental filter -> ENTRY / WAIT / NO-TRADE
Usage:  python run_pipeline.py --source yahoo --symbol ^BSESN
        python run_pipeline.py --source demo      (synthetic wiring test, NOT a market signal)
"""
import sys, json, argparse, datetime as dt, warnings
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from features.indicators import compute_all
from features.custom_indicators import train_params, apply_custom_indicators
from learn.combos import search_combos, attach_combo_columns, combo_name
from learn.freeze import freeze, load_frozen
from learn.regime_edge import regime_edge, breakeven_win_rate
from strategy.ensemble import ComboEnsembleStrategy
from backtest.fixed_exit import run_fixed_exit_backtest
from fundamental.filter import load_inputs
from decision.engine import decide
from risk.engine import RiskConfig

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="demo", choices=["demo", "yahoo"])
ap.add_argument("--symbol", default="^BSESN")
ap.add_argument("--today", default=None)
a = ap.parse_args()

if a.source == "yahoo":
    from data.provider import get_history
    raw = get_history(a.symbol, "2000-01-01"); banner = f"REAL DATA: {a.symbol}"
else:
    from data.demo_series import make_demo_full
    raw = make_demo_full(); banner = "DEMO — SYNTHETIC DATA. NOT A MARKET SIGNAL."
today = dt.date.fromisoformat(a.today) if a.today else (raw.index[-1].date() if a.source == "demo" else dt.date.today())
print("=" * 70, f"\n  {banner}\n" + "=" * 70)

tr_end, oos_a, oos_b, cur_a = "2019-12-31", "2020-01-01", "2025-12-31", "2026-01-01"
# 1) LEARN on training only
train = compute_all(raw.loc[:tr_end]); params = train_params(train)
train = apply_custom_indicators(train, params)
ia = int((train.index <= "2012-12-31").sum())
combos, n_tested, n_pass = search_combos(train, [(0, ia), (ia, len(train))])
print(f"\nLearned combos: tested {n_tested}, {n_pass} profitable in both training folds, kept {len(combos)}")
for c in combos:
    print(f"  {combo_name(c['direction'], c['names']):<48} folds(n,exp%,wr%)=" +
          " | ".join(f"{f['n']},{f['exp']*100:+.1f} (base {f['base']*100:+.1f}),{f['wr']*100:.0f}" for f in c["folds"]))
# 2) FREEZE
sha = freeze("frozen/algo_v1.json", params, combos, {"trained_through": tr_end, "source": banner, "n_tested": n_tested})
print(f"\nFROZEN → frozen/algo_v1.json  sha256={sha[:16]}…")

# 3) OOS with the reloaded frozen file
params, combos, meta, sha = load_frozen("frozen/algo_v1.json")
full = attach_combo_columns(apply_custom_indicators(compute_all(raw), params), combos)
ens = ComboEnsembleStrategy(combos)
oos = run_fixed_exit_backtest(full.loc[oos_a:oos_b], [ens], RiskConfig(min_risk_reward=1.5),
                              period_label="OOS 2020-2025", max_concurrent=3)
s = oos.summary()
print(f"\nOOS 2020-2025: {s['n_trades']} trades | win {s['win_rate_pct']}% (breakeven {breakeven_win_rate()*100:.1f}%) | "
      f"PF {s['profit_factor']} | expectancy {s['expectancy_per_trade_pct']}%/trade | maxDD {s['max_drawdown_pct']}%")
print("80% win-rate target:", "demonstrated" if s["win_rate_pct"] >= 80 and s["n_trades"] >= 30 else "NOT statistically demonstrated")

# 4) REGIME EDGE
table = regime_edge(oos.trades); json.dump(table, open("reports/regime_table.json", "w"), indent=2)
print("\nRegime verdicts (OOS):")
for r, v in table.items():
    print(f"  {r:<16} n={v['n']:<3} win={v['win_rate_pct']:>5}%  exp={v['expectancy_pct']:>+6.2f}%  p={v['p_vs_breakeven']:<6} → {v['status']}")

# 5-7) CURRENT STATE → SIGNAL → FUNDAMENTAL FILTER → DECISION
cur = full.loc[:pd.Timestamp(today)]
dec = decide(cur, ens, table, load_inputs("config/fundamentals_input.json"), today)
print("\n" + "=" * 70 + f"\n  DECISION as of {dec.bar_date}: {dec.action}\n" + "=" * 70)
print(f"  regime: {dec.regime} ({dec.regime_status})  fundamental: {dec.fundamental or 'n/a'}")
if dec.direction: print(f"  {dec.direction.upper()} entry {dec.entry:.2f} stop {dec.stop:.2f} target {dec.target:.2f}")
for r in dec.reasons: print("  -", r)
json.dump({"action": dec.action, "reasons": dec.reasons, "regime": dec.regime, "bar_date": dec.bar_date,
           "source": banner}, open("reports/latest_decision.json", "w"), indent=2)
