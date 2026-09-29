from scipy.stats import binomtest
import numpy as np
from learn.combos import TP, SL, COST


def breakeven_win_rate(): return (SL + COST) / (TP + SL + 2 * COST)


def regime_edge(trades, min_n=20, alpha=0.10):
    """Per-regime OOS verdict: ENABLED / DISABLED / INSUFFICIENT_SAMPLE."""
    p0 = breakeven_win_rate(); table = {}
    for reg in sorted({t.regime for t in trades}):
        ts = [t for t in trades if t.regime == reg]; n = len(ts); w = sum(t.win for t in ts)
        exp = float(np.mean([t.net_pct for t in ts])) * 100
        pv = binomtest(w, n, p0, alternative="greater").pvalue if n else 1.0
        status = ("INSUFFICIENT_SAMPLE" if n < min_n else "ENABLED" if (exp > 0 and pv < alpha) else "DISABLED")
        table[reg] = {"n": n, "win_rate_pct": round(100 * w / n, 1), "expectancy_pct": round(exp, 3),
                      "p_vs_breakeven": round(float(pv), 4), "status": status}
    return table
