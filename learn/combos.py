"""
Learn indicator COMBINATIONS from training data only.
Method: build a library of boolean conditions (long/short mirrored), exhaustively
test 2- and 3-condition AND-combinations, score each by forward outcome under the
fixed 12% target / 7% stop rule, and keep only combos that are profitable in
BOTH training folds (early vs late) with a minimum signal count.
NOTE: ~hundreds of candidates are tested, so some will look good by chance.
The frozen combos are judged only by the untouched 2020-2025 OOS run.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from itertools import combinations

TP, SL, COST = 0.12, 0.07, 0.0005


def forward_outcomes(df, direction, tp=TP, sl=SL, horizon=250, cost=COST):
    """Net first-touch return using the same execution convention as the intraday
    engine: signal on bar i, entry at the NEXT bar open, then evaluate exits
    forward from that entry bar. Same-bar TP+SL is conservatively a loss.

    If ``session_id`` exists, the trade cannot cross a session boundary.
    """
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    n = len(df)
    out = np.full(n, np.nan)
    sid = df["session_id"].to_numpy() if "session_id" in df.columns else None

    for i in range(n - 1):
        if sid is not None and sid[i + 1] != sid[i]:
            continue
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        last = min(n, i + 2 + horizon)
        if sid is not None:
            same = np.flatnonzero(sid[i + 1:last] != sid[i])
            if len(same):
                last = i + 1 + int(same[0])
        hh, ll = h[i + 1:last], l[i + 1:last]
        if direction == "long":
            tp_hit = hh >= entry * (1 + tp)
            sl_hit = ll <= entry * (1 - sl)
            tp_ret, sl_ret = tp, -sl
        else:
            tp_hit = ll <= entry * (1 - tp)
            sl_hit = hh >= entry * (1 + sl)
            tp_ret, sl_ret = tp, -sl
        t = int(tp_hit.argmax()) if tp_hit.any() else None
        s = int(sl_hit.argmax()) if sl_hit.any() else None
        if t is None and s is None:
            continue
        out[i] = (sl_ret - cost) if (s is not None and (t is None or s <= t)) else (tp_ret - cost)
    return out


def condition_library(df):
    """name -> (long_mask, short_mask) as boolean Series. All causal."""
    c = df["close"]; up = df["ema_21"] > df["ema_50"]; dn = df["ema_21"] < df["ema_50"]
    L = {}
    L["trend_ema"] = (up, dn)
    L["trend_200"] = (c > df["ema_200"], c < df["ema_200"])
    L["adx_strong"] = (df["adx_14"] > 25, df["adx_14"] > 25)
    L["adx_weak"] = (df["adx_14"] < 20, df["adx_14"] < 20)
    if "momentum_score" in df:
        L["mom"] = (df["momentum_score"] > 0.2, df["momentum_score"] < -0.2)
    if "mpo" in df:
        L["mpo_hi"] = (df["mpo"] > 55, df["mpo"] > 55)
    if "aema_cross" in df:
        L["aema"] = (df["aema_cross"] == 1, df["aema_cross"] == 0)
    L["rsi_side"] = (df["rsi_14"] > 50, df["rsi_14"] < 50)
    L["rsi_pullback"] = ((df["rsi_14"] < 45) & up, (df["rsi_14"] > 55) & dn)
    L["macd_side"] = (df["hist"] > 0, df["hist"] < 0)
    L["donch_break"] = (c > df["donchian_upper"].shift(1), c < df["donchian_lower"].shift(1))
    near = (c / df["ema_21"] - 1).abs() < 0.015
    L["near_ema21"] = (near, near)
    sq = df["bb_width"].rolling(252, min_periods=60).rank(pct=True) < 0.3
    L["bb_squeeze"] = (sq, sq)
    if df["volume"].median() > 0 and df["rel_volume"].notna().any():
        ve = df["rel_volume"] > 1.2
        L["vol_expansion"] = (ve, ve)
    return L


def thin(mask, gap=5):
    """Causal thinning: keep a signal only if >= gap bars since the last kept one."""
    out = np.zeros(len(mask), dtype=bool); last = -10**9
    for i in np.flatnonzero(mask):
        if i - last >= gap:
            out[i] = True; last = i
    return out


def combo_name(direction, names): return f"{direction}:{'+'.join(names)}"


def _mask(conds, names, idx):
    return np.logical_and.reduce([conds[n][idx].fillna(False).values.astype(bool) for n in names])


def search_combos(df, fold_bounds, sizes=(2, 3), min_signals=25, top_k=4):
    conds = condition_library(df); names = list(conds)
    outs = {"long": forward_outcomes(df, "long"), "short": forward_outcomes(df, "short")}
    rows, n_tested = [], 0
    # unconditional (all-bars) expectancy per fold: a combo must beat THIS, not zero,
    # otherwise any long-only rule "works" in a market that drifts upward
    base = {d: [float(np.nanmean(outs[d][a:b])) for a, b in fold_bounds] for d in outs}
    for k in sizes:
        for combo in combinations(names, k):
            for d, idx in (("long", 0), ("short", 1)):
                n_tested += 1
                m = thin(_mask(conds, combo, idx))
                stats = []
                for fi, (a, b) in enumerate(fold_bounds):
                    o = outs[d][a:b][m[a:b]]; o = o[~np.isnan(o)]
                    if len(o) < min_signals: stats = None; break
                    stats.append({"n": int(len(o)), "exp": float(o.mean()), "wr": float((o > 0).mean()),
                                  "base": base[d][fi], "excess": float(o.mean()) - base[d][fi]})
                if stats:
                    rows.append({"direction": d, "names": list(combo), "folds": stats,
                                 "score": min(s["excess"] for s in stats)})
    rows = [r for r in rows if r["score"] > 0 and all(f["exp"] > 0 for f in r["folds"])]  # beat baseline AND profit outright; rows.sort(key=lambda r: -r["score"])
    chosen = []
    for d in ("long", "short"):
        sel = []
        for r in (x for x in rows if x["direction"] == d):
            if all(len(set(r["names"]) & set(s["names"])) / len(set(r["names"]) | set(s["names"])) < 0.5 for s in sel):
                sel.append(r)
            if len(sel) >= top_k: break
        chosen += sel
    return chosen, n_tested, len(rows)


def attach_combo_columns(df, combos, gap=5):
    out = df.copy(); conds = condition_library(out)
    for cb in combos:
        missing = [n for n in cb["names"] if n not in conds]
        if missing: raise ValueError(f"conditions unavailable on this data: {missing}")
        m = thin(_mask(conds, cb["names"], 0 if cb["direction"] == "long" else 1), gap)
        out[f"combo::{combo_name(cb['direction'], cb['names'])}"] = m
    return out
