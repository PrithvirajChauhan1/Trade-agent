"""Candlestick flags (causal: use bar t and earlier) + conditional-expectancy statistics.
Nothing here is assumed predictive: every pattern is measured against costs and the unconditional baseline."""
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

def candle_flags(df):
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close")); rng = (h - l).replace(0, np.nan)
    body = (c - o).abs(); up = h - np.maximum(o, c); lo = np.minimum(o, c) - l
    po, ph, pl, pc = o.shift(1), h.shift(1), l.shift(1), c.shift(1); b = body / rng
    f = {
     "doji": (b < 0.1, "both"),
     "hammer": ((lo >= 2 * body) & (up <= 0.5 * body) & (b > 0.05) & (c.shift(3) > c), "long"),
     "inverted_hammer": ((up >= 2 * body) & (lo <= 0.5 * body) & (b > 0.05) & (c.shift(3) > c), "long"),
     "shooting_star": ((up >= 2 * body) & (lo <= 0.5 * body) & (b > 0.05) & (c.shift(3) < c), "short"),
     "bull_engulfing": ((c > o) & (pc < po) & (c >= po) & (o <= pc), "long"),
     "bear_engulfing": ((c < o) & (pc > po) & (c <= po) & (o >= pc), "short"),
     "bull_harami": ((c > o) & (pc < po) & (c <= po) & (o >= pc), "long"),
     "bear_harami": ((c < o) & (pc > po) & (c >= po) & (o <= pc), "short"),
     "inside_bar": ((h <= ph) & (l >= pl), "both"), "outside_bar": ((h > ph) & (l < pl), "both"),
     "marubozu_bull": ((c > o) & (b >= 0.9), "long"), "marubozu_bear": ((c < o) & (b >= 0.9), "short"),
     "pin_bar_bull": ((lo / rng >= 0.6) & (b <= 0.3), "long"), "pin_bar_bear": ((up / rng >= 0.6) & (b <= 0.3), "short"),
     "morning_star": ((c.shift(2) < o.shift(2)) & (body.shift(2) / rng.shift(2) > 0.5) & (body.shift(1) < 0.35 * body.shift(2)) & (c > o) & (c > (o.shift(2) + c.shift(2)) / 2), "long"),
     "evening_star": ((c.shift(2) > o.shift(2)) & (body.shift(2) / rng.shift(2) > 0.5) & (body.shift(1) < 0.35 * body.shift(2)) & (c < o) & (c < (o.shift(2) + c.shift(2)) / 2), "short"),
    }
    return {k: (m.fillna(False).values.astype(bool), d) for k, (m, d) in f.items()}

def forward_arrays(df, H=6):
    """Entry at open of T+1, exit at close of T+H (same session only). Returns ret, mfe, mae as fractions (long view)."""
    n = len(df); o, h, l, c = (df[k].values for k in ("open", "high", "low", "close")); sid = df["session_id"].values
    ret, mfe, mae = (np.full(n, np.nan) for _ in range(3))
    if n > H + 1:
        m = n - H
        e = o[1:m + 1]; ret[:m] = c[H:H + m] / e - 1
        mfe[:m] = sliding_window_view(h[1:], H)[:m].max(1) / e - 1; mae[:m] = sliding_window_view(l[1:], H)[:m].min(1) / e - 1
        bad = np.zeros(n, bool); bad[:m] = sid[:m] != sid[H:H + m]; bad[m:] = True
        ret[bad] = mfe[bad] = mae[bad] = np.nan
    return ret, mfe, mae

def candle_stats(df, ranges, H=6, cost_bps=5.0, min_n=30):
    """ranges: {'train': (lo, hi), ...} session-index ranges. Regime/volume groupings use TRAIN only."""
    ret, mfe, mae = forward_arrays(df, H); flags = candle_flags(df); sidx = df["session_id"].values; rows = []
    groups = [("ALL", "ALL", np.ones(len(df), bool))] + \
             [("regime", r, (df.regime == r).values) for r in df.regime.unique()] + [("high_volume", "yes", df.high_volume.values), ("high_volume", "no", ~df.high_volume.values)]
    for name, (mask, direction) in flags.items():
        for dirn in (["long", "short"] if direction == "both" else [direction]):
            sgn = 1 if dirn == "long" else -1
            r = sgn * ret * 1e4 - cost_bps; fav = (mfe if sgn > 0 else -mae) * 1e4; adv = (mae if sgn > 0 else -mfe) * 1e4
            for split, (lo, hi) in ranges.items():
                inr = (sidx >= lo) & (sidx < hi)
                for gtype, gval, gm in groups:
                    if gtype != "ALL" and split != "train": continue
                    sel = mask & inr & gm & ~np.isnan(r); x = r[sel]
                    if len(x) < (min_n if gtype != "ALL" else 5): continue
                    base = r[inr & gm & ~np.isnan(r)] if dirn == "long" else r[inr & gm & ~np.isnan(r)]
                    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
                    rows.append({"pattern": name, "direction": dirn, "split": split, "group": f"{gtype}={gval}", "n": len(x), "win_rate": round(100 * (x > 0).mean(), 1),
                                 "avg_net_bps": round(x.mean(), 2), "median_net_bps": round(np.median(x), 2), "avg_MFE_bps": round(np.nanmean(fav[sel]), 1),
                                 "avg_MAE_bps": round(np.nanmean(adv[sel]), 1), "profit_factor": round(pos / neg, 2) if neg > 0 else np.inf,
                                 "baseline_avg_net_bps": round(float(np.nanmean(sgn * ret[inr & gm] * 1e4 - cost_bps)), 2)})
    return pd.DataFrame(rows)
