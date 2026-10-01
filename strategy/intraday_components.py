"""Config-driven strategy builder: reusable components -> boolean long/short signal arrays.
A signal at bar T uses information <= T; the engine enters at the OPEN of bar T+1."""
from __future__ import annotations
import numpy as np, pandas as pd

def _pair(l, s): return l.fillna(False).astype(bool), s.fillna(False).astype(bool)

def c_ema_align(df, p=True):  return _pair((df.ema_9 > df.ema_21) & (df.ema_21 > df.ema_50), (df.ema_9 < df.ema_21) & (df.ema_21 < df.ema_50))
def c_vwap_side(df, p=True):  return _pair(df.close > df.vwap, df.close < df.vwap)
def c_rel_vol_min(df, p):     x = df.rel_volume_tod >= p; return _pair(x, x)
def c_adx_min(df, p):         x = df.adx_14 >= p; return _pair(x, x)
def c_adx_max(df, p):         x = df.adx_14 <= p; return _pair(x, x)
def c_rsi_momo(df, p):        return _pair(df.rsi_14 >= p["long_min"], df.rsi_14 <= p["short_max"])
def c_htf_15m(df, p=True):    return _pair(df.htf_trend_15m > 0, df.htf_trend_15m < 0)
def c_htf_1h(df, p=True):     return _pair(df.htf_trend_1h > 0, df.htf_trend_1h < 0)
def c_orb(df, p=True):
    lb = (df.close > df.or_high) & (df.close.shift(1) <= df.or_high); sb = (df.close < df.or_low) & (df.close.shift(1) >= df.or_low)
    return _pair(lb, sb)
def c_pullback(df, p):
    tol = p * df.atr_14
    return _pair((df.low <= df.ema_21 + tol) & (df.close > df.ema_21) & (df.close > df.open),
                 (df.high >= df.ema_21 - tol) & (df.close < df.ema_21) & (df.close < df.open))
def c_bb_extreme(df, p):      return _pair((df.close < df.bb_lower) & (df.rsi_14 < p["rsi_lo"]), (df.close > df.bb_upper) & (df.rsi_14 > p["rsi_hi"]))
def c_vwap_stretch(df, p):    return _pair(df.vwap_dist_atr <= -p, df.vwap_dist_atr >= p)   # stretched away from VWAP -> fade
def c_gap_ok(df, p):          x = df.gap_pct.abs() <= p; return _pair(x, x)

def c_pat_reversal(df, p=True):     return _pair(df.pat_reversal_long, df.pat_reversal_short)
def c_pat_continuation(df, p=True): return _pair(df.pat_cont_long, df.pat_cont_short)
def c_mkt_ctx(df, p):
    """p: {'index':bool,'tech':bool,'sector':bool,'vol_max':level}. Requires supplied context columns (KeyError otherwise)."""
    L = pd.Series(True, index=df.index); S = L.copy()
    for k in ("index", "tech", "sector"):
        if p.get(k): L &= df[f"ctx_{k}_trend"] > 0; S &= df[f"ctx_{k}_trend"] < 0
    if p.get("vol_max") is not None: ok = df["ctx_vol_level"] <= p["vol_max"]; L &= ok; S &= ok
    return _pair(L, S)

COMPONENTS = {k[2:]: v for k, v in globals().items() if k.startswith("c_")}
BONUS = {"rel_vol_1.5": lambda d: d.rel_volume_tod >= 1.5, "htf_1h": lambda d: None}

def generate_signals(df, cfg):
    n = len(df); L = np.ones(n, bool); S = np.ones(n, bool)
    for name, param in cfg["entry"].items():
        l, s = COMPONENTS[name](df, param); L &= l.values; S &= s.values
    d = cfg.get("direction", "both")
    if d == "long": S[:] = False
    if d == "short": L[:] = False
    allow = np.ones(n, bool)
    if cfg.get("tod_allow"): allow &= df.tod.isin(cfg["tod_allow"]).values
    if cfg.get("regime_allow"): allow &= df.regime.isin(cfg["regime_allow"]).values
    nb = int(df.bar_of_day.max()) + 1
    allow &= (df.bar_of_day.values >= cfg.get("min_bar_of_day", 1)) & (df.bar_of_day.values <= nb - 1 - cfg.get("no_entry_last_bars", 6))
    L &= allow; S &= allow; both = L & S; L &= ~both; S &= ~both
    # uncalibrated confirmation score (NOT a probability): share of optional confirmations present
    b1 = (df.rel_volume_tod >= 1.5).fillna(False).values
    b2 = np.where(L, df.htf_trend_1h.values > 0, df.htf_trend_1h.values < 0)
    b3 = (df.adx_14 >= 25).values
    score = 40 + 20 * (b1.astype(int) + b2.astype(int) + b3.astype(int))
    return L, S, score.astype(float)

CANDIDATES = [
 {"id": "TREND_VWAP", "version": 1, "hypothesis": "Trend continuation: EMA stack + price on the right side of session VWAP + ADX + volume + RSI momentum + 15m trend agreement.",
  "direction": "both", "entry": {"ema_align": True, "vwap_side": True, "adx_min": 20, "rel_vol_min": 1.0, "rsi_momo": {"long_min": 55, "short_max": 45}, "htf_15m": True},
  "tod_allow": ["mid_morning", "lunch", "afternoon"]},
 {"id": "ORB_VOLUME", "version": 1, "hypothesis": "Opening-range breakout with relative-volume spike, EMA alignment and 15m trend agreement.",
  "direction": "both", "entry": {"orb": True, "rel_vol_min": 1.5, "ema_align": True, "htf_15m": True}, "min_bar_of_day": 6},
 {"id": "TREND_PULLBACK", "version": 1, "hypothesis": "Buy/sell EMA21 pullbacks with rejection candle inside an aligned trend, VWAP side and 1h trend agreement.",
  "direction": "both", "entry": {"ema_align": True, "pullback": 0.3, "vwap_side": True, "htf_1h": True, "adx_min": 18}},
 {"id": "MEAN_REVERSION", "version": 1, "hypothesis": "Fade Bollinger extremes stretched from VWAP in low-ADX (range) conditions; expects reversion, fails in trend days.",
  "direction": "both", "entry": {"bb_extreme": {"rsi_lo": 30, "rsi_hi": 70}, "vwap_stretch": 1.5, "adx_max": 22, "gap_ok": 0.006}},
 {"id": "PATTERN_REVERSAL", "version": 1, "hypothesis": "Confirmed double top/bottom, head-and-shoulders (and inverse) and failed opening-range breaks, with normal-or-better participation.",
  "direction": "both", "entry": {"pat_reversal": True, "rel_vol_min": 1.0}, "tod_allow": ["mid_morning", "lunch", "afternoon"]},
 {"id": "PATTERN_CONTINUATION", "version": 1, "hypothesis": "Rectangle/triangle/flag/pennant breakouts in the direction of EMA alignment and 15m trend, with volume expansion.",
  "direction": "both", "entry": {"pat_continuation": True, "ema_align": True, "htf_15m": True, "rel_vol_min": 1.2}},
 {"id": "CONTEXT_TREND", "version": 1, "hypothesis": "TREND_VWAP entries additionally required to agree with index and sector trend and to occur when the volatility index is not elevated. Tests whether market context adds value.",
  "direction": "both", "entry": {"ema_align": True, "vwap_side": True, "adx_min": 20, "rel_vol_min": 1.0, "htf_15m": True, "mkt_ctx": {"index": True, "sector": True, "vol_max": 30}},
  "tod_allow": ["mid_morning", "lunch", "afternoon"]},
]
