import pandas as pd

class DataUnavailable(Exception): pass

def get_history(symbol="^BSESN", start="2000-01-01", end=None):
    """Real daily OHLCV from Yahoo Finance (needs network + `pip install yfinance`)."""
    try:
        import yfinance as yf
        raw = yf.download(symbol, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    except Exception as e:
        raise DataUnavailable(f"Yahoo fetch failed: {e}")
    if raw is None or raw.empty: raise DataUnavailable("Yahoo returned no data")
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None); df.index.name = "timestamp"
    return df
