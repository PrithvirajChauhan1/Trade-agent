"""
Real market data adapter. NOT runnable in this sandbox (no network access) —
run this on your local machine after `pip install yfinance`.

Usage (locally):
    from data.real_source import fetch_yfinance
    df = fetch_yfinance("AAPL", interval="5m", period="60d")
"""
from __future__ import annotations
import pandas as pd


def fetch_yfinance(symbol: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance and normalize to the schema the rest of the
    pipeline expects: lowercase columns [open, high, low, close, volume],
    DatetimeIndex named 'timestamp'.

    Notes on yfinance limitations (know these before trusting results):
    - Intraday intervals (1m/5m/15m) are only available for the trailing ~60 days.
    - 1m data is capped at ~7 days per request.
    - Data can have gaps around holidays/halts; the feature layer does NOT
      currently interpolate — gaps will show up as larger bar-to-bar moves.
    - This is delayed/free-tier data, not suitable for live execution timing,
      only for research/backtesting.
    """
    import yfinance as yf  # imported lazily — not installed in this sandbox

    raw = yf.download(symbol, interval=interval, period=period, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {symbol} ({interval}, {period})")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "timestamp"
    return df


def validate_ohlcv_schema(df: pd.DataFrame) -> list[str]:
    """Sanity checks before feeding data into the pipeline. Returns a list of problems (empty = OK)."""
    problems = []
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        problems.append(f"missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        problems.append("index is not a DatetimeIndex")
    if df.isna().any().any():
        problems.append(f"contains NaNs: {df.isna().sum().to_dict()}")
    if (df.get("high", pd.Series(dtype=float)) < df.get("low", pd.Series(dtype=float))).any():
        problems.append("found bars where high < low")
    if len(df) < 50:
        problems.append(f"only {len(df)} bars — too few for reliable indicator warmup")
    return problems
