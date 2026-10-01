"""Holding-period / exit-mode grid, train->validation selection, walk-forward, robustness tests."""
import itertools, numpy as np, pandas as pd
from backtest.intraday import run_intraday, CostModel, summarize

HOLD = [1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36]; RR = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]; STOP = [0.75, 1.0, 1.5, 2.0, 2.5]; TRAIL = [None, 1.5]

def run_grid(df, sig, cost, symbol="SYM", strategy="S"):
    L, S, sc = sig; g = {}
    for hold, rr, st, tr in itertools.product(HOLD, RR, STOP, TRAIL):
        g[(hold, rr, st, tr)] = run_intraday(df, L, S, sc, stop_atr=st, target_rr=rr, max_hold_bars=hold, trail_atr=tr, cost=cost, symbol=symbol, strategy=strategy)
    return g

def sl(trades, lo, hi): return [t for t in trades if lo <= t.session_idx < hi]
def expR(trades): return float(np.mean([t.net_r for t in trades])) if trades else float("nan")

def select(grid, tr, va, min_tr=30, min_va=15):
    best, elig = None, 0
    for cfg, trades in grid.items():
        a, b = sl(trades, *tr), sl(trades, *va)
        if len(a) < min_tr or len(b) < min_va: continue
        ea, eb = expR(a), expR(b)
        if ea > 0 and eb > 0:
            elig += 1
            if best is None or eb > best[1]: best = (cfg, eb, ea, len(a), len(b))
    return best, elig

def neighbor_stability(grid, cfg, rng, min_n=10):
    axes = [HOLD, RR, STOP, TRAIL]; pos = tot = 0
    for ax in range(3):
        i = axes[ax].index(cfg[ax])
        for j in (i - 1, i + 1):
            if 0 <= j < len(axes[ax]):
                nb = list(cfg); nb[ax] = axes[ax][j]; t = sl(grid[tuple(nb)], *rng)
                if len(t) >= min_n: tot += 1; pos += expR(t) > 0
    return (pos / tot) if tot else float("nan"), tot

def walk_forward(grid, n_sess, init_frac=0.4, test_len=20):
    a, folds, oos = int(n_sess * init_frac), [], []
    while a + test_len <= n_sess:
        inner = int(a * 2 / 3); best, elig = select(grid, (0, inner), (inner, a), min_tr=20, min_va=10)
        te = sl(grid[best[0]], a, a + test_len) if best else []
        folds.append({"train_end": a, "test": f"{a}-{a+test_len}", "config": best[0] if best else None, "n": len(te), "expR": expR(te) if te else np.nan})
        oos += te; a += test_len
    return folds, oos

def bootstrap_ci(r, n=10000, seed=0):
    r = np.asarray(r); rng = np.random.default_rng(seed)
    if len(r) < 5: return (np.nan, np.nan)
    m = rng.choice(r, (n, len(r))).mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def mc_drawdown(r, n=5000, seed=0):
    r = np.asarray(r); rng = np.random.default_rng(seed)
    if len(r) < 5: return (np.nan, np.nan)
    dd = []
    for _ in range(n):
        e = np.cumsum(rng.permutation(r)); dd.append((np.maximum.accumulate(e) - e).max())
    return float(np.median(dd)), float(np.percentile(dd, 95))
