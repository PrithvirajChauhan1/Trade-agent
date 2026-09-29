"""
20-year synthetic daily OHLCV (2000-01-01 to 2025-01-01) with historically-
plausible regime segments matching major market epochs.

Regimes mapped to approximate real-world periods:
  2000-03 → dot-com crash (strong bear)
  2003-07 → recovery bull
  2008-09 → GFC crash (strong bear, high vol)
  2009-20 → long bull + corrections
  2020-03 → COVID crash (extreme high vol)
  2020-04 onwards → recovery + 2022 correction + 2023-24 bull

CRITICAL: This is SYNTHETIC data with statistical properties SIMILAR to
those historical periods — it is NOT real market data. No trading decisions
should be based on this alone. Wire in real yfinance data via
data/real_source.py before any live or paper trading.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EpochSegment:
    label: str
    start: str
    end: str
    drift_annual: float   # annualised log-return
    vol_annual: float     # annualised volatility (log)
    vol_spike: float = 1.0  # volatility multiplier mid-crash events


def _trading_days(start: str, end: str) -> int:
    return len(pd.bdate_range(start, end))


def generate_long_ohlcv(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, test_df):
      train_df: 2000-01-03 to 2019-12-31 (~5040 trading days)
      test_df:  2020-01-02 to 2024-12-31 (~1258 trading days)
    """
    rng = np.random.default_rng(seed)

    epochs = [
        EpochSegment("dotcom_crash",      "2000-01-03", "2002-12-31", -0.28, 0.35),
        EpochSegment("recovery_bull",     "2003-01-02", "2007-10-31",  0.14, 0.14),
        EpochSegment("gfc_crash",         "2007-11-01", "2009-03-31", -0.45, 0.55, vol_spike=1.8),
        EpochSegment("qe_bull_1",         "2009-04-01", "2011-06-30",  0.22, 0.18),
        EpochSegment("correction_2011",   "2011-07-01", "2011-12-31", -0.15, 0.28),
        EpochSegment("steady_bull",       "2012-01-02", "2015-06-30",  0.16, 0.13),
        EpochSegment("flash_crash_2015",  "2015-07-01", "2016-02-29", -0.12, 0.26),
        EpochSegment("bull_2016_2018",    "2016-03-01", "2018-09-30",  0.20, 0.14),
        EpochSegment("q4_2018_sell",      "2018-10-01", "2018-12-31", -0.20, 0.30),
        EpochSegment("late_bull_2019",    "2019-01-02", "2019-12-31",  0.29, 0.14),
        # Test period begins 2020
        EpochSegment("covid_crash",       "2020-01-02", "2020-03-23", -0.35, 0.70, vol_spike=2.5),
        EpochSegment("covid_recovery",    "2020-03-24", "2021-12-31",  0.42, 0.22),
        EpochSegment("rate_hike_bear",    "2022-01-03", "2022-12-31", -0.26, 0.28),
        EpochSegment("ai_bull_2023_24",   "2023-01-02", "2024-12-31",  0.28, 0.16),
    ]

    all_rows = []
    price = 100.0
    trading_cal = pd.bdate_range("2000-01-03", "2024-12-31")
    cal_idx = 0

    for ep in epochs:
        days = _trading_days(ep.start, ep.end)
        daily_drift = ep.drift_annual / 252
        daily_vol = ep.vol_annual / np.sqrt(252)

        for d in range(days):
            if cal_idx >= len(trading_cal):
                break
            date = trading_cal[cal_idx]
            cal_idx += 1

            # vol can spike at start of crash epochs
            spike = ep.vol_spike if d < 15 else 1.0
            v = daily_vol * spike

            steps = rng.normal(daily_drift / 8, v / np.sqrt(8), size=8)
            path = price * np.exp(np.cumsum(steps))
            o = price
            c = path[-1]
            h = max(o, c, path.max()) * (1 + rng.exponential(0.001))
            l = min(o, c, path.min()) * (1 - rng.exponential(0.001))
            bar_range = (h - l) / price
            vol = max(1000, rng.normal(50000, 12000) * (1 + 4 * bar_range))

            all_rows.append({
                "date": date,
                "epoch": ep.label,
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(l, 4),
                "close": round(c, 4),
                "volume": round(vol, 0),
            })
            price = c

    df = pd.DataFrame(all_rows).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "timestamp"

    train = df[df.index.year < 2020].copy()
    test  = df[df.index.year >= 2020].copy()
    return train, test
