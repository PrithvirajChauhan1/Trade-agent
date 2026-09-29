"""Session-aware intraday features. Everything is causal: value at bar t uses bars <= t only."""
from __future__ import annotations
import numpy as np, pandas as pd
from data.session import SessionConfig, NSE
from features.indicators import compute_all
from features.regime import classify_regime

def _htf_trend(df, k, fast=8, slow=21):
    """Trend of higher-timeframe bars (k x 5m), using only COMPLETED HTF bars (shifted one HTF bar)."""
    key = [df["session_date"], df["bar_of_day"] // k]
    g = df.groupby(key, sort=True)["close"].last()
    tr = np.sign(g.ewm(span=fast, adjust=False).mean() - g.ewm(span=slow, adjust=False).mean()).shift(1)
    vals = tr.reindex(pd.MultiIndex.from_arrays(key)).values
    return pd.Series(vals, index=df.index).fillna(0)

def build_intraday_features(raw, sess: SessionConfig = NSE, or_bars=6):
    if raw.index.tz is None: raise ValueError("index must be tz-aware (see data.quality)")
    out = compute_all(raw.tz_convert(sess.tz))
    loc = out.index
    mins = loc.hour * 60 + loc.minute - sess.open_min
    out["session_date"] = loc.date; out["bar_of_day"] = mins // sess.bar_minutes
    out["session_id"] = pd.factorize(out["session_date"])[0]
    out["weekday"] = loc.weekday
    sess_len = sess.close_min - sess.open_min
    out["tod"] = np.select([mins < 30, mins < 120, mins < sess_len - 135, mins < sess_len - 60],
                           ["open_orb", "mid_morning", "lunch", "afternoon"], default="final_hour")
    sd = out["session_date"]
    # session VWAP (replaces the cumulative-over-all-history VWAP from features.indicators)
    tp = (out.high + out.low + out.close) / 3
    out["vwap"] = (tp * out.volume).groupby(sd).cumsum() / out.volume.groupby(sd).cumsum().replace(0, np.nan)
    out["vwap_dist_atr"] = (out.close - out.vwap) / out.atr_14
    # time-of-day relative volume: same bar-of-day, prior 20 sessions only
    out["rel_volume_tod"] = out.volume / out.groupby("bar_of_day")["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    out["bb_pctb"] = (out.close - out.bb_lower) / (out.bb_upper - out.bb_lower)
    # previous-session levels and gap
    daily = out.groupby("session_date").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"), o=("open", "first"))
    prev = daily.shift(1)
    for name, col in (("prev_high", "h"), ("prev_low", "l"), ("prev_close", "c")):
        out[name] = sd.map(prev[col])
    out["gap_pct"] = sd.map(daily["o"]) / out["prev_close"] - 1
    # opening range: defined only from bar `or_bars` onward (needs the OR bars to be complete)
    first = out[out.bar_of_day < or_bars].groupby("session_date")
    out["or_high"] = np.where(out.bar_of_day >= or_bars, sd.map(first["high"].max()), np.nan)
    out["or_low"] = np.where(out.bar_of_day >= or_bars, sd.map(first["low"].min()), np.nan)
    out["day_high"] = out.groupby("session_date")["high"].cummax(); out["day_low"] = out.groupby("session_date")["low"].cummin()
    out["htf_trend_15m"] = _htf_trend(out, 3); out["htf_trend_1h"] = _htf_trend(out, 12)
    out["regime"] = classify_regime(out, lookback_vol=5 * sess.expected_bars)
    out["high_volume"] = out["rel_volume_tod"] > 1.5; out["gap_day"] = out["gap_pct"].abs() > 0.005
    return out
