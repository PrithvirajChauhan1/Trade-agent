import numpy as np, pandas as pd
from data.longterm import generate_long_ohlcv

def make_demo_full(seed=42):
    """SYNTHETIC 2000-2026 series for wiring tests only. Not market data."""
    tr, te = generate_long_ohlcv(seed); hist = pd.concat([tr, te])
    rng = np.random.default_rng(seed + 1); price = float(hist["close"].iloc[-1]); rows = []
    for d in pd.bdate_range("2025-01-01", "2026-09-29"):
        st = rng.normal(0.10 / 252 / 8, 0.15 / np.sqrt(252) / np.sqrt(8), 8)
        p = price * np.exp(np.cumsum(st)); o, c = price, p[-1]
        rows.append(dict(date=d, epoch="demo_2025_26", open=o, high=max(o, c, p.max()) * (1 + rng.exponential(.001)),
                         low=min(o, c, p.min()) * (1 - rng.exponential(.001)), close=c,
                         volume=max(1000, rng.normal(50000, 12000))))
        price = c
    ext = pd.DataFrame(rows).set_index("date"); ext.index.name = "timestamp"
    return pd.concat([hist, ext])
