"""Extended indicator library (causal). Each is evaluated for incremental information by research/feature_stats + ablation; none is assumed useful."""
import numpy as np, pandas as pd
from features.indicators import ema, sma, rsi, atr

def wma(s, n):
    w = np.arange(1, n + 1, dtype=float); return s.rolling(n).apply(lambda x: (x * w).sum() / w.sum(), raw=True)
def hma(s, n): return wma(2 * wma(s, max(2, n // 2)) - wma(s, n), max(2, int(np.sqrt(n))))

def aroon(df, n=25):
    up = 100 * df.high.rolling(n + 1).apply(lambda x: np.argmax(x), raw=True) / n
    dn = 100 * df.low.rolling(n + 1).apply(lambda x: np.argmin(x), raw=True) / n
    return up, dn

def supertrend(df, n=10, m=3.0):
    a = atr(df, n).values; hl2 = ((df.high + df.low) / 2).values; c = df.close.values; N = len(df)
    ub, lb = hl2 + m * a, hl2 - m * a; fu, fl = ub.copy(), lb.copy(); d = np.ones(N); line = np.full(N, np.nan)
    for i in range(1, N):
        if np.isnan(a[i]): continue
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1] or np.isnan(fu[i - 1])) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1] or np.isnan(fl[i - 1])) else fl[i - 1]
        d[i] = 1 if c[i] > fu[i - 1] else (-1 if c[i] < fl[i - 1] else d[i - 1]); line[i] = fl[i] if d[i] > 0 else fu[i]
    return pd.Series(d, index=df.index), pd.Series(line, index=df.index)

def psar(df, af0=0.02, afmax=0.2):
    h, l = df.high.values, df.low.values; N = len(df); out = np.full(N, np.nan); up = True; af = af0; ep = h[0]; sar = l[0]
    for i in range(1, N):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < sar: up, sar, ep, af = False, ep, l[i], af0
            elif h[i] > ep: ep, af = h[i], min(af + af0, afmax)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar: up, sar, ep, af = True, ep, h[i], af0
            elif l[i] < ep: ep, af = l[i], min(af + af0, afmax)
        out[i] = sar
    return pd.Series(out, index=df.index)

def ichimoku(df):
    t = (df.high.rolling(9).max() + df.low.rolling(9).min()) / 2; k = (df.high.rolling(26).max() + df.low.rolling(26).min()) / 2
    a = ((t + k) / 2).shift(26); b = ((df.high.rolling(52).max() + df.low.rolling(52).min()) / 2).shift(26)   # cloud AS PLOTTED NOW (built from the past)
    top, bot = np.maximum(a, b), np.minimum(a, b); pos = np.where(df.close > top, 1, np.where(df.close < bot, -1, 0)); pos = pd.Series(pos, index=df.index).where(top.notna())
    return t, k, pos      # Chikou span intentionally omitted: it is defined with future bars

def add_extra(df):
    o = df.copy(); c, h, l, v = o.close, o.high, o.low, o.volume
    o["wma_20"] = wma(c, 20); o["hma_21"] = hma(c, 21)
    o["aroon_up"], o["aroon_dn"] = aroon(o); o["aroon_osc"] = o.aroon_up - o.aroon_dn
    o["st_dir"], o["st_line"] = supertrend(o); o["psar"] = psar(o); o["psar_dir"] = np.sign(c - o.psar)
    o["ichi_tenkan"], o["ichi_kijun"], o["ichi_pos"] = ichimoku(o)
    hh, ll = h.rolling(14).max(), l.rolling(14).min(); o["stoch_k"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan); o["stoch_d"] = o.stoch_k.rolling(3).mean()
    r = rsi(c, 14); o["stochrsi"] = (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min()).replace(0, np.nan)
    o["willr_14"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan); o["roc_10"] = c / c.shift(10) - 1; o["mom_10"] = c - c.shift(10)
    tp = (h + l + c) / 3; o["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True))
    mf = tp * v; pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum(); neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum(); o["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    o["ppo"] = (ema(c, 12) - ema(c, 26)) / ema(c, 26) * 100
    kc_mid = ema(c, 20); o["kc_upper"] = kc_mid + 2 * o.atr_14; o["kc_lower"] = kc_mid - 2 * o.atr_14; o["kc_pos"] = (c - kc_mid) / (2 * o.atr_14)
    lr = np.log(c).diff(); o["hist_vol"] = lr.rolling(20).std() * np.sqrt(75 * 252); o["vol_pctile"] = o.hist_vol.rolling(375, min_periods=75).rank(pct=True)
    o["range_exp"] = (h - l) / (h - l).rolling(20).mean(); o["atr_exp"] = o.atr_14 / o.atr_14.rolling(50).mean()
    o["vol_accel"] = ema(v, 3) / ema(v, 10); o["vol_spike"] = (v > 3 * v.rolling(20).median()).astype(int)
    o["rsi_div_bear"] = ((c >= c.rolling(20).max()) & (o.rsi_14 < o.rsi_14.rolling(20).max() - 3)).astype(int)
    o["rsi_div_bull"] = ((c <= c.rolling(20).min()) & (o.rsi_14 > o.rsi_14.rolling(20).min() + 3)).astype(int)
    o["obv_div_bear"] = ((c >= c.rolling(20).max()) & (o.obv < o.obv.rolling(20).max())).astype(int)
    o["obv_div_bull"] = ((c <= c.rolling(20).min()) & (o.obv > o.obv.rolling(20).min())).astype(int)
    if "session_id" in o:   # anchored VWAP from the bar of the PREVIOUS session's low (known by then)
        pv = (tp * v).cumsum().values; cv = v.cumsum().values; sid = o.session_id.values; lows = pd.Series(l.values).groupby(sid).idxmin().to_dict()
        av = np.full(len(o), np.nan)
        for s in np.unique(sid):
            if s == 0: continue
            a = lows[s - 1]; m = sid == s; av[m] = (pv[m] - pv[a - 1 if a > 0 else 0]) / np.maximum(cv[m] - cv[a - 1 if a > 0 else 0], 1e-9)
        o["avwap_prev_low"] = av; o["avwap_dist_atr"] = (c - o.avwap_prev_low) / o.atr_14
    return o
