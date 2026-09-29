"""5-minute research cycle. Strict temporal protocol:
 sessions -> TRAIN 50% | VALIDATION 25% | TEST 25%.  Holding period / R:R / stop / trailing chosen on train+validation ONLY,
 frozen, then evaluated once on TEST.  Never auto-deploys: strategies only reach EXPERIMENTAL/VALIDATED/REJECTED here.
Usage: python -m research.run_research --source synthetic|csv|yahoo [--csv f.csv] [--symbol X] [--cap 200]"""
import sys, os, json, argparse, datetime as dt, warnings
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from data.session import NSE, US
from data.quality import validate_intraday
from features.intraday import build_intraday_features
from strategy.intraday_components import generate_signals, CANDIDATES
from backtest.intraday import CostModel, run_intraday, summarize, breakdown, trades_df, cap_first_n
from research.study import *
from research.registry import Registry
from research.docs import write_library, append_log
from research.feature_stats import feature_table
from patterns.candles import candle_stats
from charts.make import session_panel, trade_chart

FATAL = ("TIMEZONE", "DUPLICATE", "zero/negative", "inconsistent", "NaNs", "not monotonic")
FEATURES = ["ema_9/21/50", "rsi_14", "adx_14", "session_vwap", "rel_volume_tod", "or_high/low", "htf_trend_15m/1h", "bb", "atr_14"]

def load(a, sess):
    if a.source == "synthetic":
        from data.synthetic_5m import generate_5m; return generate_5m(a.sessions, sess=sess), "SYNTHETIC DATA — machinery test, NOT a market signal"
    if a.source == "csv":
        d = pd.read_csv(a.csv, index_col=0, parse_dates=True); return d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]], f"CSV {a.csv}"
    import yfinance as yf
    d = yf.download(a.symbol, interval="5m", period="60d", progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    return d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(), f"YAHOO {a.symbol} 5m (max ~60 days)"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="synthetic"); ap.add_argument("--csv"); ap.add_argument("--symbol", default="^NSEI")
    ap.add_argument("--sessions", type=int, default=260); ap.add_argument("--cap", type=int, default=200); ap.add_argument("--date"); ap.add_argument("--market", default="NSE")
    ap.add_argument("--registry", default="strategy_store/registry.json"); a = ap.parse_args()
    sess = NSE if a.market == "NSE" else US; day = a.date or str(dt.date.today()); out = f"reports/{day}"; os.makedirs(out + "/charts", exist_ok=True)
    raw, banner = load(a, sess); q = validate_intraday(raw, sess); json.dump(q, open(f"{out}/data_quality.json", "w"), indent=2, default=str)
    print(f"[{banner}]\nDATA QUALITY ok={q['ok']}"); [print("  -", i) for i in q["issues"]]
    if any(i.startswith(FATAL) for i in q["issues"]): sys.exit("ABORT: fatal data-quality issue(s). Fix data; nothing was run.")
    f = build_intraday_features(raw, sess); ns = int(f.session_id.max()) + 1
    tr, va, te = (0, int(ns * .5)), (int(ns * .5), int(ns * .75)), (int(ns * .75), ns); ranges = {"train": tr, "val": va, "test": te}
    d0 = lambda k: f.index[f.session_id == k][0].date()
    period = f"train {d0(0)}..{d0(va[0]-1)} | validation {d0(va[0])}..{d0(te[0]-1)} | OOS test {d0(te[0])}..{f.index[-1].date()}  ({ns} sessions)"
    print(period); cost = CostModel(); reg = Registry(a.registry); rows, alltr, siglog, regrows, log, chosen = [], [], [], [], [], {}
    for cfg in CANDIDATES:
        key = reg.register(cfg, period, FEATURES); sig = generate_signals(f, cfg); grid = run_grid(f, sig, cost, a.symbol, cfg["id"])
        best, elig = select(grid, tr, va); n_cfg = len(grid)
        info = max(((c, expR(sl(t, *va))) for c, t in grid.items() if len(sl(t, *va)) >= 15), key=lambda x: x[1], default=(None, np.nan))
        r = {"strategy": key, "configs_tested": n_cfg, "eligible_configs": elig, "config": None, "flag": ""}
        if best is None:
            verdict, why = "REJECTED", "no configuration profitable (net of costs) in BOTH train and validation"
            if info[0]: r["info_only_test_exp_R"] = round(expR(sl(grid[info[0]], *te)), 3) if sl(grid[info[0]], *te) else np.nan; r["info_config"] = str(info[0])
            r.update({"verdict": verdict, "reason": why}); rows.append(r)
            reg.record_run(key, {"date": day, "config": None, "notes": why}); 
            if reg.data[key]["status"] == "EXPERIMENTAL": reg.set_status(key, "REJECTED", why)
            log.append({"key": key, "hypothesis": cfg["hypothesis"], "components": list(cfg["entry"]), "tested": f"{n_cfg} exit configs on train/val",
                        "result": why, "decision": "REJECTED"}); continue
        c = best[0]; T = grid[c]; chosen[key] = (cfg, c, sig, T)
        trn, val, tst = sl(T, *tr), sl(T, *va), sl(T, *te); sample = cap_first_n(tst, a.cap); s = summarize(sample)
        lo, hi = bootstrap_ci([t.net_r for t in sample]); mcm, mc95 = mc_drawdown([t.net_r for t in sample])
        cs = {}
        for m in (0, 1, 2, 3):
            tt = sl(run_intraday(f, *sig, stop_atr=c[2], target_rr=c[1], max_hold_bars=c[0], trail_atr=c[3], cost=CostModel(mult=m)), *te); cs[f"x{m}"] = round(expR(tt), 3) if tt else np.nan
        nb, nbn = neighbor_stability(grid, c, va); folds, oos = walk_forward(grid, ns); fv = [x for x in folds if x["n"] > 0]
        wf_pos = round(np.mean([x["expR"] > 0 for x in fv]), 2) if fv else np.nan
        if len(sample) < 30: verdict, why = "EXPERIMENTAL", f"only {len(sample)} OOS trades (<30): inconclusive"
        elif s["expectancy_R"] <= 0: verdict, why = "REJECTED", f"negative OOS net expectancy ({s['expectancy_R']}R)"
        elif lo > 0 and wf_pos == wf_pos and wf_pos >= .6 and nb == nb and nb >= .5: verdict, why = "VALIDATED", "OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable"
        else: verdict, why = "EXPERIMENTAL", "OOS expectancy positive but CI/walk-forward/sensitivity not conclusive"
        flag = " ".join(filter(None, ["UNSTABLE" if ((wf_pos == wf_pos and wf_pos < .5) or (nb == nb and nb < .5)) else "",
                                      "GRID_EDGE(optimum may lie outside searched range)" if (c[0] == HOLD[-1] or c[1] == RR[-1] or c[2] == STOP[-1]) else ""]))
        r.update({"config": str(c), "flag": flag, "verdict": verdict, "reason": why, "train_n": len(trn), "train_exp_R": round(expR(trn), 3), "val_n": len(val), "val_exp_R": round(expR(val), 3),
                  "test_n_total": len(tst), "test_n_sample": len(sample), "cap": a.cap, "test_exp_R": s["expectancy_R"], "test_win_rate": s["win_rate"], "test_PF": s["profit_factor"],
                  "sharpe": s["sharpe"], "sortino": s["sortino"], "calmar": s["calmar"], "max_dd_R": s["max_dd_R"], "exp_R_CI95": f"[{lo:.3f},{hi:.3f}]", "MC_maxDD_R_p95": round(mc95, 1),
                  "cost_sens_exp_R": json.dumps(cs), "wf_folds": len(folds), "wf_positive_frac": wf_pos, "wf_oos_trades": len(oos), "neighbors_profitable": None if nb != nb else round(nb, 2),
                  "avg_hold_bars": s["avg_hold_bars"], "ambiguous_bars": s["ambiguous_bars"], "avg_cost_R": s["avg_cost_R"]}); rows.append(r)
        reg.record_run(key, {"date": day, "config": c, "train_exp": r["train_exp_R"], "val_exp": r["val_exp_R"], "test_exp": s["expectancy_R"], "test_n": len(sample), "test_win": s["win_rate"],
                             "test_pf": s["profit_factor"], "wf_pos": wf_pos, "neighbors": r["neighbors_profitable"], "flag": flag,
                             "notes": "; ".join(filter(None, [why, flag, "fails in trend days" if "MEAN" in key else ""]))})
        cur = reg.data[key]["status"]
        if cur == "EXPERIMENTAL" and verdict in ("VALIDATED", "REJECTED"): reg.set_status(key, verdict, why)
        elif cur == "VALIDATED" and verdict != "VALIDATED": reg.set_status(key, "DEGRADED", why)
        for split, rg in ranges.items():
            for t in sl(T, *rg): alltr.append({**trades_df([t]).iloc[0].to_dict(), "split": split, "status": reg.data[key]["status"]})
            b = breakdown(sl(T, *rg), "regime")
            if len(b): regrows.append(b.reset_index().assign(strategy=key, split=split))
        taken = {t.signal_time for t in T}; L, S_, sc = sig
        for i in np.flatnonzero((L | S_) & (f.session_id.values >= te[0])):
            siglog.append({"strategy": key, "time": f.index[i], "direction": "long" if L[i] else "short", "score_uncalibrated": sc[i], "regime": f.regime.iloc[i], "tod": f.tod.iloc[i], "taken": f.index[i] in taken})
        log.append({"key": key, "hypothesis": cfg["hypothesis"], "components": list(cfg["entry"]), "tested": f"{n_cfg} exit configs (hold x R:R x stop x trail); selected {c} on validation among {elig} eligible",
                    "result": f"train {r['train_exp_R']}R / val {r['val_exp_R']}R / OOS {s['expectancy_R']}R over {len(sample)} trades; CI {r['exp_R_CI95']}; WF+ {wf_pos}; neighbours+ {r['neighbors_profitable']}", "decision": f"{verdict}{' (UNSTABLE)' if flag else ''}: {why}"})
    # ---- reports ----
    res = pd.DataFrame(rows); res.to_csv(f"{out}/strategy_results.csv", index=False)
    pd.DataFrame(alltr).to_csv(f"{out}/trade_log.csv", index=False); pd.DataFrame(siglog).to_csv(f"{out}/signal_log.csv", index=False)
    (pd.concat(regrows) if regrows else pd.DataFrame()).to_csv(f"{out}/regime_results.csv", index=False)
    ft, red = feature_table(f, ranges); ft.to_csv(f"{out}/feature_statistics.csv", index=False)
    candle_stats(f, ranges).to_csv(f"{out}/pattern_stats.csv", index=False)
    session_panel(f, f.session_date.iloc[-1], f"{out}/charts/session_panel.png")
    for key, (cfg, c, sig, T) in chosen.items():
        tt = sl(T, *te)
        for tag, sel in (("win", [t for t in tt if t.net_r > 0]), ("loss", [t for t in tt if t.net_r <= 0])):
            if sel: trade_chart(f, sel[0], f"{out}/charts/{key}_{tag}.png")
    L_ = [f"# Daily research summary — {day}", f"**Data:** {banner}", f"**Protocol:** {period}", f"**Costs:** {cost}", ""]
    last = f.iloc[-1]; L_ += ["## Market summary", f"- Regime (last bar): {last.regime}; high relative volume: {bool(last.high_volume)}; gap day: {bool(last.gap_day)}", "- Volatility percentile / SPY / QQQ / VIX / sector leadership: not supplied (market-context data not wired yet)", ""]
    cols = [c for c in ("strategy", "verdict", "flag", "config", "test_n_sample", "test_win_rate", "test_exp_R", "test_PF", "sharpe", "max_dd_R", "wf_positive_frac", "neighbors_profitable") if c in res.columns]
    L_ += ["## Strategy summary (OOS test, net of costs)", res[cols].to_markdown(index=False) if len(res) else "none", "", "## Active setups on the last closed bar (research only; entry = NEXT bar open)"]
    act = 0
    for key, (cfg, c, sig, T) in chosen.items():
        Lb, Sb, sc = sig
        if Lb[-1] or Sb[-1]:
            d = 1 if Lb[-1] else -1; px = last.close; r_ = c[2] * last.atr_14; act += 1
            L_.append(f"- {key} {'LONG' if d>0 else 'SHORT'} ~{px:.2f} | stop ~{px-d*r_:.2f} | target ~{px+d*c[1]*r_:.2f} | max hold {c[0]} bars | uncalibrated score {sc[-1]:.0f} | status {reg.data[key]['status']}")
    if not act: L_.append("- none — NO TRADE")
    n_ok = int((res.get("verdict") == "VALIDATED").sum()) if "verdict" in res else 0
    L_ += ["", "## Top research findings"]
    if "test_exp_R" in res and res.test_exp_R.notna().any():
        b, w = res.loc[res.test_exp_R.idxmax()], res.loc[res.test_exp_R.idxmin()]
        L_ += [f"- Best OOS setup: {b.strategy} ({b.test_exp_R}R/trade, n={b.test_n_sample}); worst: {w.strategy} ({w.test_exp_R}R/trade)."]
        gap = (res.train_exp_R + res.val_exp_R) / 2 - res.test_exp_R; L_ += [f"- Train/val -> OOS expectancy decay (overfitting evidence): " + ", ".join(f"{s}: {g:+.2f}R" for s, g in zip(res.strategy, gap) if g == g)]
        L_ += [f"- Cost sensitivity (OOS exp R at cost x0/x1/x2/x3): " + "; ".join(f"{s}: {v}" for s, v in zip(res.strategy, res.cost_sens_exp_R))]
    L_ += [f"- Strategies REJECTED with no train+validation-profitable exit config: {', '.join(res[res.config.isna()].strategy) or 'none'}",
           f"- Features with sign-stable IC across train/val/test: {', '.join(ft[ft.sign_stable].feature) or 'none'}", f"- Redundant feature pairs (|rho|>0.85): {red or 'none'}",
           f"- VALIDATED strategies: {n_ok}. None is deployed; paper/live use requires explicit human approval (registry blocks it otherwise).", "",
           "_Multiple-testing note: each strategy tried 270 exit configs; validation-selected results are optimistic, only the OOS test is unbiased._"]
    open(f"{out}/daily_summary.md", "w").write("\n".join(L_)); write_library(reg); append_log(log)
    print("\n".join(L_)); print(f"\nreports -> {out}/")

if __name__ == "__main__": main()
