"""SYNTHETIC 5-minute sessions for testing the machinery only. Not market data."""
import numpy as np, pandas as pd
from data.session import SessionConfig, NSE

def generate_5m(n_sessions=260, start="2023-01-02", sess: SessionConfig = NSE, seed=7, start_price=100.0):
    rng = np.random.default_rng(seed); nb = sess.expected_bars; days = pd.bdate_range(start, periods=n_sessions)
    u = 1 + 0.8 * (np.abs(np.linspace(-1, 1, nb)) ** 2)          # U-shaped intraday vol/volume
    price, drift, rows, idx = start_price, 0.0, [], []
    for d in days:
        drift = 0.6 * drift + rng.normal(0, 0.00012)               # persistent daily drift (per bar)
        gap = rng.normal(0, 0.003); vol_day = 0.0009 * np.exp(rng.normal(0, 0.3))
        for b in range(nb):
            o = price * (1 + gap) if b == 0 else price
            steps = rng.normal(drift / 5, vol_day * u[b] / np.sqrt(5), 5); p = o * np.exp(np.cumsum(steps)); c = p[-1]
            rows.append((o, max(o, p.max()) * (1 + abs(rng.normal(0, 2e-4))), min(o, p.min()) * (1 - abs(rng.normal(0, 2e-4))),
                         c, max(100.0, rng.lognormal(9.0, 0.35) * u[b]))); price = c
            idx.append(pd.Timestamp(d) + pd.Timedelta(minutes=sess.open_min + b * sess.bar_minutes))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(idx))
    df.index = df.index.tz_localize(sess.tz); df.index.name = "timestamp"; return df
