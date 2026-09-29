"""
Synthetic OHLCV generator for testing the pipeline offline.

IMPORTANT: This is a testing/development tool only. It generates data with
KNOWN statistical properties (specific regime segments, specific volatility)
so we can verify the backtester computes correct metrics. It is NOT a
substitute for real market data validation. Before trusting any strategy
in paper trading, it must be re-validated against real OHLCV (yfinance,
broker API, etc.) — see data/real_source.py for that adapter.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RegimeSegment:
    name: str          # 'strong_bull' | 'weak_bull' | 'sideways' | 'strong_bear' | 'weak_bear' | 'high_vol_chop'
    n_bars: int
    drift: float        # mean log-return per bar
    vol: float           # std of log-return per bar


def generate_ohlcv(
    segments: list[RegimeSegment],
    start_price: float = 100.0,
    bar_minutes: int = 5,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Build a synthetic OHLCV series by concatenating regime segments.
    Each bar's OHLC is derived from a simulated intrabar path so wicks/bodies
    are realistic rather than just close-to-close.
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = start_price
    t0 = pd.Timestamp("2024-01-01 09:15:00")
    idx = 0

    for seg in segments:
        for _ in range(seg.n_bars):
            # simulate 10 intrabar steps to build a realistic H/L/O/C
            steps = rng.normal(seg.drift / 10, seg.vol / np.sqrt(10), size=10)
            path = price * np.exp(np.cumsum(steps))
            o = price
            c = path[-1]
            h = max(o, c, path.max())
            l = min(o, c, path.min())
            # volume: base + noise, with a mild bump on big-range bars (crude but fine for testing)
            rng_range = (h - l) / price
            vol = max(100, rng.normal(1000, 250) * (1 + 3 * rng_range))

            rows.append({
                "timestamp": t0 + pd.Timedelta(minutes=bar_minutes * idx),
                "regime_label": seg.name,  # ground truth, for validating regime detector
                "open": o, "high": h, "low": l, "close": c, "volume": vol,
            })
            price = c
            idx += 1

    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def default_mixed_regime_series(bar_minutes: int = 5, seed: int = 42) -> pd.DataFrame:
    """A ~2500-bar series cycling through all regime types, for backtesting/unit tests."""
    segments = [
        RegimeSegment("strong_bull", 300, drift=0.0012, vol=0.004),
        RegimeSegment("sideways", 250, drift=0.0000, vol=0.003),
        RegimeSegment("weak_bear", 200, drift=-0.0004, vol=0.004),
        RegimeSegment("high_vol_chop", 180, drift=0.0000, vol=0.010),
        RegimeSegment("strong_bull", 350, drift=0.0015, vol=0.005),
        RegimeSegment("strong_bear", 220, drift=-0.0018, vol=0.006),
        RegimeSegment("sideways", 300, drift=0.0001, vol=0.0025),
        RegimeSegment("weak_bull", 260, drift=0.0005, vol=0.0035),
        RegimeSegment("strong_bear", 240, drift=-0.0014, vol=0.0055),
        RegimeSegment("sideways", 200, drift=0.0000, vol=0.002),
    ]
    return generate_ohlcv(segments, bar_minutes=bar_minutes, seed=seed)
