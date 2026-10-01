"""SYNTHETIC context series (wiring/test only): index correlated with the stock, a vol index reacting to |moves|."""
import numpy as np, pandas as pd
def make_ctx(df, seed=11):
    rng = np.random.default_rng(seed); lr = np.log(df.close).diff().fillna(0).values
    idx = 100 * np.exp(np.cumsum(0.6 * lr + rng.normal(0, 0.0004, len(df)))); mk = lambda x: pd.DataFrame({"close": x}, index=df.index)
    days = df.groupby(df.index.date).close.agg(lambda s: abs(np.log(s.iloc[-1] / s.iloc[0])))
    vix = pd.DataFrame({"close": 14 + 400 * days.rolling(3, min_periods=1).mean().values}, index=pd.DatetimeIndex(pd.to_datetime(days.index)).tz_localize(df.index.tz))
    return {"index": mk(idx), "tech": mk(idx * (1 + rng.normal(0, 0.0003, len(df)).cumsum())), "sector": mk(100 * np.exp(np.cumsum(0.4 * lr + rng.normal(0, 0.0005, len(df))))), "vol": vix}
