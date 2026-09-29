"""Data-quality gate for intraday OHLCV. Returns a report; never silently repairs."""
import numpy as np, pandas as pd
from data.session import SessionConfig, NSE

def validate_intraday(df, sess: SessionConfig = NSE, max_missing_frac=0.02):
    issues, info = [], {}
    if not isinstance(df.index, pd.DatetimeIndex): return {"ok": False, "issues": ["index is not a DatetimeIndex"], "info": {}}
    if df.index.tz is None: issues.append("TIMEZONE: index is tz-naive (localize to the exchange timezone first)")
    else: df = df.tz_convert(sess.tz)
    if df.index.duplicated().any(): issues.append(f"DUPLICATE candles: {int(df.index.duplicated().sum())}")
    if not df.index.is_monotonic_increasing: issues.append("timestamps not monotonic increasing")
    if (df[["open", "high", "low", "close"]] <= 0).any().any(): issues.append("zero/negative prices present")
    if df.isna().any().any(): issues.append(f"NaNs present: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")
    bad = (df.high < df[["open", "close", "low"]].max(axis=1)) | (df.low > df[["open", "close", "high"]].min(axis=1))
    if bad.any(): issues.append(f"inconsistent OHLC bars: {int(bad.sum())}")
    m = df.index.hour * 60 + df.index.minute
    off = (m < sess.open_min) | (m >= sess.close_min)
    if off.any(): issues.append(f"bars outside regular session (pre/post-market mixed in?): {int(off.sum())}")
    if (m % sess.bar_minutes != 0).any(): issues.append("timestamps not aligned to bar grid")
    d = df[~off]; per = d.groupby(d.index.date).size()
    short = per[per < sess.expected_bars * (1 - max_missing_frac)]
    info["sessions"] = int(len(per)); info["expected_bars_per_session"] = sess.expected_bars
    info["short_sessions"] = {str(k): int(v) for k, v in short.head(10).items()}
    if len(short): issues.append(f"MISSING candles: {len(short)} sessions below {(1-max_missing_frac)*100:.0f}% of {sess.expected_bars} bars"
                                  " (half-days/holidays/outages — inspect before trusting)")
    if (df.volume == 0).mean() > 0.2: issues.append(f"volume mostly zero ({(df.volume==0).mean():.0%}) — volume features unreliable (index data?)")
    vol = df.volume[df.volume > 0]
    if len(vol) and (vol > 30 * vol.median()).any(): issues.append(f"volume anomalies: {int((vol > 30*vol.median()).sum())} bars > 30x median")
    info["not_verifiable"] = ["corporate actions/splits/dividends (use adjusted vendor data)", "exchange holiday calendar"]
    return {"ok": not issues, "issues": issues, "info": info}
