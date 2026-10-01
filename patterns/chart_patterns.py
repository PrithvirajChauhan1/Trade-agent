"""Causal chart-pattern detector. Pivots are CONFIRMED k bars after they occur (never used earlier), patterns fire on the bar that completes them
(breakout of the neckline/boundary). Per-session only. Patterns are measured statistically, not assumed to work."""
import numpy as np, pandas as pd

def detect(df, k=2, max_age=30):
    n = len(df); h, l, c, a = (df[x].values for x in ("high", "low", "close", "atr_14")); sid = df.session_id.values
    orh = df.or_high.values if "or_high" in df else np.full(n, np.nan); orl = df.or_low.values if "or_low" in df else np.full(n, np.nan)
    names = ["double_bottom", "double_top", "inverse_head_shoulders", "head_shoulders", "failed_breakdown", "failed_breakout",
             "rectangle_break_up", "rectangle_break_down", "triangle_break_up", "triangle_break_down", "flag_break_up", "flag_break_down", "pennant_break_up", "pennant_break_down"]
    F = {x: np.zeros(n, bool) for x in names}
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]); ends = np.r_[starts[1:], n]
    for s0, s1 in zip(starts, ends):
        PH, PL = [], []
        for t in range(s0, s1):
            i = t - k
            if i - k >= s0:
                if h[i] > h[i - k:i].max() and h[i] >= h[i + 1:t + 1].max(): PH.append(i)
                if l[i] < l[i - k:i].min() and l[i] <= l[i + 1:t + 1].min(): PL.append(i)
            if np.isnan(a[t]) or t - s0 < 8: continue
            up = c[t] > c[t - 1]
            if len(PL) >= 2 and len(PH) >= 1:
                l1, l2 = PL[-2], PL[-1]
                if l2 - l1 >= 6 and abs(l[l2] - l[l1]) <= 0.6 * a[t] and t - l2 <= max_age:
                    neck = h[l1:l2 + 1].max()
                    if c[t] > neck >= c[t - 1]: F["double_bottom"][t] = True
            if len(PH) >= 2 and len(PL) >= 1:
                h1, h2 = PH[-2], PH[-1]
                if h2 - h1 >= 6 and abs(h[h2] - h[h1]) <= 0.6 * a[t] and t - h2 <= max_age:
                    neck = l[h1:h2 + 1].min()
                    if c[t] < neck <= c[t - 1]: F["double_top"][t] = True
            if len(PL) >= 3:
                p1, p2, p3 = PL[-3:]
                if l[p2] < l[p1] and l[p2] < l[p3] and abs(l[p1] - l[p3]) <= 0.8 * a[t] and t - p3 <= max_age:
                    neck = h[p1:p3 + 1].max()
                    if c[t] > neck >= c[t - 1]: F["inverse_head_shoulders"][t] = True
            if len(PH) >= 3:
                p1, p2, p3 = PH[-3:]
                if h[p2] > h[p1] and h[p2] > h[p3] and abs(h[p1] - h[p3]) <= 0.8 * a[t] and t - p3 <= max_age:
                    neck = l[p1:p3 + 1].min()
                    if c[t] < neck <= c[t - 1]: F["head_shoulders"][t] = True
            if t >= 3 and not np.isnan(orh[t]):
                if c[t - 3:t].max() > orh[t] and c[t] <= orh[t] and c[t - 1] > orh[t]: F["failed_breakout"][t] = True
                if c[t - 3:t].min() < orl[t] and c[t] >= orl[t] and c[t - 1] < orl[t]: F["failed_breakdown"][t] = True
            w = 20
            if t - s0 >= w:
                rng = h[t - w:t].max() - l[t - w:t].min()
                if rng <= 3.0 * a[t]:
                    if c[t] > h[t - w:t].max(): F["rectangle_break_up"][t] = True
                    if c[t] < l[t - w:t].min(): F["rectangle_break_down"][t] = True
            tri = len(PH) >= 2 and len(PL) >= 2 and h[PH[-1]] < h[PH[-2]] and l[PL[-1]] > l[PL[-2]] and t - min(PH[-1], PL[-1]) <= max_age
            imp = t - s0 >= 25 and abs(c[t - 6] - c[t - 25]) >= 2.5 * a[t]
            if tri and c[t] > h[PH[-1]] >= c[t - 1]: F["triangle_break_up"][t] = True; F["pennant_break_up"][t] = imp and c[t - 6] > c[t - 25]
            if tri and c[t] < l[PL[-1]] <= c[t - 1]: F["triangle_break_down"][t] = True; F["pennant_break_down"][t] = imp and c[t - 6] < c[t - 25]
            if t - s0 >= 12:
                imp2 = c[t - 6] - c[t - 12]
                if abs(imp2) >= 2.5 * a[t] and (h[t - 6:t].max() - l[t - 6:t].min()) <= 0.4 * abs(imp2):
                    if imp2 > 0 and c[t] > h[t - 6:t].max(): F["flag_break_up"][t] = True
                    if imp2 < 0 and c[t] < l[t - 6:t].min(): F["flag_break_down"][t] = True
    return F

DIRECTION = {x: ("long" if x in ("double_bottom", "inverse_head_shoulders", "failed_breakdown") or x.endswith("_up") else "short") for x in
             ["double_bottom", "double_top", "inverse_head_shoulders", "head_shoulders", "failed_breakdown", "failed_breakout", "rectangle_break_up", "rectangle_break_down",
              "triangle_break_up", "triangle_break_down", "flag_break_up", "flag_break_down", "pennant_break_up", "pennant_break_down"]}

def add_pattern_columns(df):
    o = df.copy(); F = detect(o)
    for k, m in F.items(): o[f"pat_{k}"] = m
    o["pat_reversal_long"] = o[["pat_double_bottom", "pat_inverse_head_shoulders", "pat_failed_breakdown"]].any(axis=1)
    o["pat_reversal_short"] = o[["pat_double_top", "pat_head_shoulders", "pat_failed_breakout"]].any(axis=1)
    o["pat_cont_long"] = o[[f"pat_{x}_break_up" for x in ("rectangle", "triangle", "flag", "pennant")]].any(axis=1)
    o["pat_cont_short"] = o[[f"pat_{x}_break_down" for x in ("rectangle", "triangle", "flag", "pennant")]].any(axis=1)
    return o

def pattern_flags(df): return {k: (df[f"pat_{k}"].values, DIRECTION[k]) for k in DIRECTION}
