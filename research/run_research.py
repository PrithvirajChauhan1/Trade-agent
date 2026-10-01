"""5-minute research cycle. Strict temporal protocol:
 sessions -> TRAIN 50% | VALIDATION 25% | TEST 25%.  Holding period / R:R / stop / trailing chosen on train+validation ONLY,
 frozen, then evaluated once on TEST.  Never auto-deploys: strategies only reach EXPERIMENTAL/VALIDATED/REJECTED here.
Usage:
  Synthetic machinery test:
    python -m research.run_research --source synthetic
  Real long-history 5m research:
    python -m research.run_research --source csv --csv data/5m.csv \
      --train-start 2000-01-01 --train-end 2019-12-31 \
      --val-start 2018-01-01 --val-end 2019-12-31 \
      --test-start 2020-01-01 --test-end 2025-12-31 --cap 200
The real-market path requires a genuine long-history 5m dataset. Yahoo's
~60-day 5m retention is intentionally rejected for the requested study."""
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
from features.extra_indicators import add_extra
from patterns.chart_patterns import add_pattern_columns, pattern_flags
from features.context import add_context
from research.ml import meta_label
from research.ablation import ablate
from research.robustness import robustness_rows

FATAL = ("TIMEZONE", "DUPLICATE", "zero/negative", "inconsistent", "NaNs", "not monotonic")
FEATURES = ["ema_9/21/50", "rsi_14", "adx_14", "session_vwap", "rel_volume_tod", "or_high/low", "htf_trend_15m/1h", "bb", "atr_14"]

def load(a, sess):
    if a.source == "synthetic":
        from data.synthetic_5m import generate_5m
        return generate_5m(a.sessions, sess=sess), "SYNTHETIC DATA — machinery test, NOT a market signal"
    if a.source == "csv":
        if not a.csv:
            raise SystemExit("--csv is required for --source csv")
        d = pd.read_csv(a.csv, index_col=0, parse_dates=True)
        d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        if d.index.tz is None:
            d.index = d.index.tz_localize(sess.tz)
        else:
            d.index = d.index.tz_convert(sess.tz)
        return d, f"REAL 5m CSV {a.csv}"
    raise SystemExit("Yahoo 5m is intentionally unsupported for long-history research; use a real archived 5m CSV.")


def _date_split(f, a):
    if a.source == "synthetic":
        ns = int(f.session_id.max()) + 1
        return (0, int(ns * .5)), (int(ns * .5), int(ns * .75)), (int(ns * .75), ns)
    starts = pd.to_datetime([a.train_start, a.val_start, a.test_start]).date
    ends = pd.to_datetime([a.train_end, a.val_end, a.test_end]).date
    sd = pd.Series(f.session_date).astype("datetime64[ns]").dt.date
    def bounds(start, end):
        ids = np.flatnonzero((sd >= start) & (sd <= end))
        if not len(ids):
            raise SystemExit(f"No sessions found for requested range {start}..{end}")
        return int(f.session_id.iloc[ids[0]]), int(f.session_id.iloc[ids[-1]]) + 1
    return bounds(starts[0], ends[0]), bounds(starts[1], ends[1]), bounds(starts[2], ends[2])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="synthetic"); ap.add_argument("--csv"); ap.add_argument("--symbol", default="^NSEI")
    ap.add_argument("--sessions", type=int, default=260); ap.add_argument("--cap", type=int, default=200); ap.add_argument("--date"); ap.add_argument("--market", default="NSE")
    ap.add_argument("--train-start", default="2000-01-01"); ap.add_argument("--train-end", default="2019-12-31")
    ap.add_argument("--val-start", default="2018-01-01"); ap.add_argument("--val-end", default="2019-12-31")
    ap.add_argument("--test-start", default="2020-01-01"); ap.add_argument("--test-end", default="2025-12-31")
    ap.add_argument("--registry", default="strategy_store/registry.json"); ap.add_argument("--ctx", default="none", help="none | synthetic | index=f.csv,tech=f.csv,sector=f.csv,vol=f.csv"); a = ap.parse_args()
    sess = NSE if a.market == "NSE" else US; day = a.date or str(dt.date.today()); out = f"reports/{day}"; os.makedirs(out + "/charts", exist_ok=True)
    raw, banner = load(a, sess); q = validate_intraday(raw, sess); json.dump(q, open(f"{out}/data_quality.json", "w"), indent=2, default=str)
    print(f"[{banner}]\nDATA QUALITY ok={q['ok']}"); [print("  -", i) for i in q["issues"]]
    if any(i.startswith(FATAL) for i in q["issues"]): sys.exit("ABORT: fatal data-quality issue(s). Fix data; nothing was run.")
    f = add_pattern_columns(add_extra(build_intraday_features(raw, sess))); ns = int(f.session_id.max()) + 1
    if a.ctx == "synthetic":
        from data.synthetic_ctx import make_ctx; f = add_context(f, make_ctx(f))
    elif a.ctx != "none":
        ctx = {}
        for part in a.ctx.split(","):
            k, path = part.split("="); d = pd.read_csv(path, index_col=0, parse_dates=True).rename(columns=str.lower); d.index = d.index.tz_localize(sess.tz) if d.index.tz is None else d.index; ctx[k] = d
        f = add_context(f, ctx)
    tr, va, te = _date_split(f, a); ranges = {"train": tr, "val": va, "test": te}
    d0 = lambda k: f.index[f.session_id == k][0].date()
    period = f"train {d0(tr[0])}..{d0(tr[1]-1)} | validation {d0(va[0])}..{d0(va[1]-1)} | OOS test {d0(te[0])}..{d0(te[1]-1)}"
    if a.source != "synthetic":
        assert d0(tr[0]) <= dt.date(2019, 1, 1) and d0(te[0]) >= dt.date(2020, 1, 1), "requested long-history split not present"
    print(period); cost = CostModel(); reg = Registry(a.registry); rows, alltr, siglog, regrows, log, chosen = [], [], [], [], [], {}; abl, mlr, rob = [], [], []
    for cfg in CANDIDATES:
        if "mkt_ctx" in cfg["entry"] and "ctx_index_trend" not in f:
            print(f"SKIP {cfg['id']}: needs market-context data (--ctx); none supplied"); continue
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
        elif lo > 0 and wf_pos == wf_pos and wf_pos >= .6 and nb == nb and nb >= .5:
            if a.source == "synthetic":
                verdict, why = "SYNTHETIC_VALIDATED", "Synthetic-data machinery passed statistical checks; this is NOT evidence of market viability."
            else:
                verdict, why = "VALIDATED", "Real-market OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable"
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
        if a.source != "synthetic":
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
        abl += ablate(f, cfg, c, ranges, cost, key); mlr += meta_label(f, T, ranges, key)
        for split, rg in ranges.items(): rob += robustness_rows(f, sl(T, *rg), key, tr, split)
    # ---- reports ----
    res = pd.DataFrame(rows); res.to_csv(f"{out}/strategy_results.csv", index=False)
    pd.DataFrame(alltr).to_csv(f"{out}/trade_log.csv", index=False); pd.DataFrame(siglog).to_csv(f"{out}/signal_log.csv", index=False)
    # The research quota is global across the selected strategies, chronological only.
    # It is never used to cherry-pick a strategy or parameter set.
    global_oos = []
    for key, (_, _, _, T) in chosen.items():
        for t in sl(T, *te):
            global_oos.append({**trades_df([t]).iloc[0].to_dict(), "strategy": key})
    global_sample = sorted(global_oos, key=lambda x: x["entry_time"])[:a.cap] if global_oos else []
    pd.DataFrame(global_sample).to_csv(f"{out}/global_oos_sample.csv", index=False)
    (pd.concat(regrows) if regrows else pd.DataFrame()).to_csv(f"{out}/regime_results.csv", index=False)
    pd.DataFrame(abl).to_csv(f"{out}/ablation.csv", index=False); pd.DataFrame(mlr).to_csv(f"{out}/ml_results.csv", index=False); pd.DataFrame(rob).to_csv(f"{out}/robustness.csv", index=False)
    ft, red = feature_table(f, ranges); ft.to_csv(f"{out}/feature_statistics.csv", index=False)
    candle_stats(f, ranges).to_csv(f"{out}/pattern_stats.csv", index=False); candle_stats(f, ranges, flags=pattern_flags(f)).to_csv(f"{out}/chart_pattern_stats.csv", index=False)
    session_panel(f, f.session_date.iloc[-1], f"{out}/charts/session_panel.png")
    for key, (cfg, c, sig, T) in chosen.items():
        tt = sl(T, *te)
        for tag, sel in (("win", [t for t in tt if t.net_r > 0]), ("loss", [t for t in tt if t.net_r <= 0])):
            if sel: trade_chart(f, sel[0], f"{out}/charts/{key}_{tag}.png")
    L_ = [f"# Daily research summary — {day}", f"**Data:** {banner}", f"**Protocol:** {period}", f"**Costs:** {cost}", f"**Global OOS research sample:** first {a.cap} chronological trades across selected strategies (actual n={len(global_sample)}); not a forced quota.", ""]
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
    A = pd.DataFrame(abl)
    if len(A):
        full = A[A.variant == "FULL"].set_index("strategy").exp_val_R; m = A[A.variant.str.startswith("minus:")].copy(); m["gain"] = m.exp_val_R.values - full.reindex(m.strategy).values
        red_c = m[m.gain > 0.02]; L_ += ["- Ablation (validation): components whose REMOVAL raised expectancy >0.02R (candidate redundancy): " + (", ".join(f"{r.strategy}:{r.variant[6:]}(+{r.gain:.2f})" for r in red_c.itertuples()) or "none")]
    M = pd.DataFrame(mlr)
    if len(M) and "auc_test" in M: L_ += ["- ML meta-label (test AUC / filtered vs unfiltered expR): " + "; ".join(f"{r.strategy}/{r.model}: {r.auc_test} / {getattr(r, 'test_filtered_expR', 'n/a')} vs {r.test_unfiltered_expR}" for r in M.dropna(subset=["auc_test"]).itertuples())]
    L_ += [f"- Strategies REJECTED with no train+validation-profitable exit config: {', '.join(res[res.config.isna()].strategy) or 'none'}",
           f"- Features with sign-stable IC across train/val/test: {', '.join(ft[ft.sign_stable].feature) or 'none'}", f"- Redundant feature pairs (|rho|>0.85): {red or 'none'}",
           f"- Real-market VALIDATED strategies: {n_ok}. Synthetic validation is never promoted. None is deployed; paper/live use requires explicit human approval (registry blocks it otherwise).", "",
           "_Multiple-testing note: each strategy tried 880 exit configs; validation-selected results are optimistic, only the OOS test is unbiased._"]
    open(f"{out}/daily_summary.md", "w").write("\n".join(L_)); write_library(reg); append_log(log)
    print("\n".join(L_)); print(f"\nreports -> {out}/")

if __name__ == "__main__": main()
