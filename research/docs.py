import os, datetime as dt
INDICATORS = [
 ("EMA (9/21/50/200)", "Trend direction/alignment", "EMA_t = a*P_t + (1-a)*EMA_{t-1}, a=2/(n+1)", "span", "Lags; whipsaws in ranges", "Sideways chop; overnight gaps distort continuity"),
 ("RSI(14)", "Momentum / stretch", "100 - 100/(1+avgGain/avgLoss) (Wilder smoothing)", "period", "Stays extreme in strong trends", "Trend days: 'oversold' keeps falling"),
 ("MACD(12,26,9) histogram", "Momentum change", "EMA12-EMA26 minus its 9-EMA", "fast, slow, signal", "Lagging, noisy on 5m", "Range-bound sessions"),
 ("ATR(14)", "Volatility scale for stops/targets", "Wilder EMA of true range", "period", "Backward-looking; jumps at open", "Volatility regime shifts intraday"),
 ("ADX(14)", "Trend strength (not direction)", "Wilder-smoothed |+DI - -DI|/(+DI + -DI)", "period", "Slow to react", "Trend reversals after a strong ADX reading"),
 ("Bollinger(20,2) / %B / width", "Stretch and compression", "SMA20 +/- 2 sigma", "period, k", "Bands widen after the move", "Band-walks in trends break mean reversion"),
 ("Session VWAP", "Intraday fair-value anchor", "cum(TP*V)/cum(V) reset each session", "session", "Meaningless with zero volume (indexes)", "Very early bars; low-volume names"),
 ("Relative volume (time-of-day)", "Participation vs normal for that bar", "V_t / mean(V at same bar-of-day, prior 20 sessions)", "20 sessions", "Needs history; news days distort", "Volume-less instruments"),
 ("Opening range (first 6 bars)", "Early-session balance breakout level", "max/min of bars 0..5, valid from bar 6", "or_bars", "Few breakouts per day", "Gap days, low-volume opens"),
 ("HTF trend (15m/1h)", "Higher-timeframe agreement", "sign(EMA8-EMA21) of COMPLETED HTF closes", "k bars", "Stale by up to one HTF bar", "Fast reversals"),
 ("Candle patterns", "Bar-level structure", "see patterns/candles.py", "-", "Weak alone; measured, not assumed", "Most are cost-negative alone"),
]

def write_library(reg, path="docs/STRATEGY_LIBRARY.md"):
    L = ["# Strategy Library", "", "_Auto-generated from `strategy_store/registry.json`. All versions are kept; nothing is overwritten._", "", "## Indicators", ""]
    for n, p, f, par, w, fm in INDICATORS:
        L += [f"### {n}", f"- **Purpose:** {p}", f"- **Formula:** {f}", f"- **Parameters:** {par}", f"- **Known weaknesses:** {w}", f"- **Failure modes:** {fm}", ""]
    L += ["## Strategies", ""]
    for k, s in reg.data.items():
        last = s["runs"][-1] if s["runs"] else {}; P = s["parameters"]
        L += [f"### {k}  —  status: **{s['status']}**", f"- **Objective / hypothesis:** {s['hypothesis']}", f"- **Created:** {s['created_at']}",
              f"- **Entry conditions:** `{P['entry']}` (direction: {P.get('direction')}; tod: {P.get('tod_allow')}; regime: {P.get('regime_allow')})",
              "- **Entry timing:** signal on bar T close, fill at open of T+1", "- **Stop/target/exit/hold:** selected on validation, see latest run",
              f"- **Training / validation / OOS:** {s['training_period']}", f"- **Features:** {', '.join(s['features'])}"]
        if last:
            L += [f"- **Selected config (hold, R:R, stop ATR, trail):** {last.get('config')}", f"- **Train / Val / OOS expectancy (R):** {last.get('train_exp')} / {last.get('val_exp')} / {last.get('test_exp')}",
                  f"- **OOS trades / win rate / PF:** {last.get('test_n')} / {last.get('test_win')} / {last.get('test_pf')}", f"- **Walk-forward positive folds:** {last.get('wf_pos')}",
                  f"- **Parameter sensitivity (neighbors profitable):** {last.get('neighbors')}", f"- **Flags:** {last.get('flag') or 'none'}",
                  f"- **Failure modes / limitations:** {last.get('notes')}"]
        else: L += ["- No run recorded yet."]
        L += [f"- **Status history:** {[(h['to'], h['reason']) for h in s['status_history']]}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, "w").write("\n".join(L))

def append_log(entries, path="docs/ALGORITHM_DEVELOPMENT_LOG.md"):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write("# Algorithm Development Log\n\n_Append-only. Failed experiments are kept on purpose._\n")
        for e in entries: f.write(f"\n## {dt.datetime.now():%Y-%m-%d %H:%M} — {e['key']}\n- **Why / hypothesis:** {e['hypothesis']}\n- **Combined components:** {e['components']}\n"
                                  f"- **Tested:** {e['tested']}\n- **Result:** {e['result']}\n- **Decision:** {e['decision']}\n")
