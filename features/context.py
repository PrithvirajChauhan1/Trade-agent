"""Market-wide context (index/tech index/sector/volatility index). Data must be SUPPLIED by the user; nothing is fetched or invented.
Causality: 5m context bars are joined as-of (forward-fill by timestamp); the volatility index uses the PREVIOUS session's close."""
import numpy as np, pandas as pd

def add_context(df, ctx):
    o = df.copy()
    for name in ("index", "tech", "sector"):
        if name in ctx:
            c = ctx[name]["close"]; c = c.tz_convert(df.index.tz) if c.index.tz is not None else c.tz_localize(df.index.tz)
            tr = np.sign(c.ewm(span=9, adjust=False).mean() - c.ewm(span=21, adjust=False).mean())
            o[f"ctx_{name}_trend"] = tr.reindex(df.index, method="ffill"); o[f"ctx_{name}_ret12"] = c.pct_change(12).reindex(df.index, method="ffill")
    if "index" in ctx: o["rel_strength_12"] = df.close.pct_change(12) - o["ctx_index_ret12"]
    if "vol" in ctx:
        v = ctx["vol"]["close"]; d = v.groupby(pd.DatetimeIndex(v.index).date).last().shift(1)
        o["ctx_vol_level"] = o.session_date.map(d); pct = d.rolling(60, min_periods=20).rank(pct=True); o["ctx_vol_pct"] = o.session_date.map(pct)
    return o
