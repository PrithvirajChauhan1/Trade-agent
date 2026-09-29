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
]
