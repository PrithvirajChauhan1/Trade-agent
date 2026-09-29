"""
Market regime detection (spec Section 20).

Classifies each bar into a regime label using ADX (trend strength),
EMA slope/ordering (trend direction), and ATR percentile (volatility).
This is a rules-based classifier, not ML — deliberately, so it's
inspectable and doesn't itself need out-of-sample validation to trust.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


REGIMES = [
    "strong_bull", "weak_bull", "strong_bear", "weak_bear",
    "sideways", "high_volatility", "low_volatility",
]


def classify_regime(df: pd.DataFrame, adx_col: str = "adx_14",
                     atr_col: str = "atr_14", lookback_vol: int = 100) -> pd.Series:
    """
    df must already have adx_14, ema_21, ema_50, atr_14 columns (see indicators.compute_all).
    Returns a Series of regime labels (str), aligned to df.index.
    """
    required = {adx_col, atr_col, "ema_21", "ema_50", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"classify_regime requires columns {missing} — run indicators.compute_all first")

    trend_up = df["ema_21"] > df["ema_50"]
    slope = df["ema_21"].diff(5)
    strong_trend = df[adx_col] > 25
    weak_trend = (df[adx_col] > 15) & (df[adx_col] <= 25)

    atr_pct = df[atr_col].rolling(lookback_vol, min_periods=20).rank(pct=True)
    high_vol = atr_pct > 0.80
    low_vol = atr_pct < 0.20

    labels = pd.Series("sideways", index=df.index, dtype=object)

    labels[strong_trend & trend_up & (slope > 0)] = "strong_bull"
    labels[strong_trend & (~trend_up) & (slope < 0)] = "strong_bear"
    labels[weak_trend & trend_up] = "weak_bull"
    labels[weak_trend & (~trend_up)] = "weak_bear"

    # volatility flags override to communicate regime risk, but only where
    # trend classification didn't already find a strong signal — high vol
    # doesn't erase a strong trend, it's a modifier the strategy engine checks separately
    no_trend = ~(strong_trend | weak_trend)
    labels[no_trend & high_vol] = "high_volatility"
    labels[no_trend & low_vol] = "low_volatility"

    return labels


def regime_confidence(df: pd.DataFrame, adx_col: str = "adx_14") -> pd.Series:
    """
    Crude confidence score (0-100) for the regime call: how far ADX is from
    the ambiguous 15-25 band, scaled. Not a statistical confidence interval —
    a heuristic to flag borderline classifications for the NO-TRADE filter.
    """
    adx_val = df[adx_col]
    dist_from_ambiguous = np.where(
        adx_val <= 15, (15 - adx_val),
        np.where(adx_val >= 25, (adx_val - 25), 0)
    )
    conf = 50 + np.clip(dist_from_ambiguous * 3, 0, 45)
    return pd.Series(conf, index=df.index)
