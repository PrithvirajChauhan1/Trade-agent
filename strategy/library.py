"""
Concrete strategies, each mapping directly to spec sections:
- TrendPullback: Section 2 trend-following + Section 5 pattern combos
- BollingerMeanReversion: Section 2 mean reversion
- DonchianBreakout: Section 2 breakout strategies

Each requires MULTIPLE confirming conditions (never a single indicator
threshold), per Section 3's explicit prohibition on "RSI < 30 -> BUY".
"""
from __future__ import annotations
import pandas as pd
from strategy.base import Strategy, StrategyMeta, StrategySignal


class TrendPullbackStrategy(Strategy):
    meta = StrategyMeta(
        name="trend_pullback",
        compatible_regimes=["strong_bull", "strong_bear", "weak_bull", "weak_bear"],
        min_risk_reward=1.5,
        suitable_timeframes=["5m", "15m", "1h"],
        description="Enter on pullback to EMA21/VWAP within an established trend, "
                    "confirmed by rejection candle and volume.",
    )

    def evaluate(self, df: pd.DataFrame, i: int):
        if i < 55:  # warmup for ema_50 etc.
            return None
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        bull_trend = row["ema_21"] > row["ema_50"] > row["ema_200"] if "ema_200" in df.columns else row["ema_21"] > row["ema_50"]
        bear_trend = row["ema_21"] < row["ema_50"] < row["ema_200"] if "ema_200" in df.columns else row["ema_21"] < row["ema_50"]

        near_ema = abs(row["close"] - row["ema_21"]) / row["ema_21"] < 0.006
        bullish_rejection = row["close"] > row["open"] and (row["close"] - row["low"]) > 1.5 * abs(row["close"] - row["open"])
        bearish_rejection = row["close"] < row["open"] and (row["high"] - row["close"]) > 1.5 * abs(row["open"] - row["close"])
        vol_confirm = row.get("rel_volume", 1.0) > 1.1
        momentum_up = row["hist"] > prev["hist"]
        momentum_down = row["hist"] < prev["hist"]

        evidence = []
        if bull_trend and near_ema and bullish_rejection and vol_confirm and momentum_up:
            evidence = [
                "EMA21 > EMA50 (uptrend structure)",
                "Price pulled back near EMA21",
                "Bullish rejection candle (long lower wick)",
                "Relative volume > 1.1x average",
                "MACD histogram rising",
            ]
            entry = row["close"]
            stop = min(row["low"], prev["low"]) - 0.5 * row["atr_14"]
            target = entry + 2.0 * (entry - stop)
            return StrategySignal("long", entry, stop, target, confidence=70,
                                   evidence=evidence,
                                   invalidation=f"Close below {stop:.2f}")

        if bear_trend and near_ema and bearish_rejection and vol_confirm and momentum_down:
            evidence = [
                "EMA21 < EMA50 (downtrend structure)",
                "Price pulled back near EMA21",
                "Bearish rejection candle (long upper wick)",
                "Relative volume > 1.1x average",
                "MACD histogram falling",
            ]
            entry = row["close"]
            stop = max(row["high"], prev["high"]) + 0.5 * row["atr_14"]
            target = entry - 2.0 * (stop - entry)
            return StrategySignal("short", entry, stop, target, confidence=70,
                                   evidence=evidence,
                                   invalidation=f"Close above {stop:.2f}")
        return None


class BollingerMeanReversionStrategy(Strategy):
    meta = StrategyMeta(
        name="bb_mean_reversion",
        compatible_regimes=["sideways", "low_volatility"],
        min_risk_reward=1.3,
        suitable_timeframes=["5m", "15m"],
        description="Fade extreme closes outside Bollinger Bands in a non-trending "
                    "regime, confirmed by RSI extreme and low ADX.",
    )

    def evaluate(self, df: pd.DataFrame, i: int):
        if i < 30:
            return None
        row = df.iloc[i]
        low_adx = row["adx_14"] < 20  # explicit regime gate, not just trusting the label
        oversold = row["close"] < row["bb_lower"] and row["rsi_14"] < 32
        overbought = row["close"] > row["bb_upper"] and row["rsi_14"] > 68

        if low_adx and oversold:
            entry = row["close"]
            stop = entry - 1.2 * row["atr_14"]
            target = row["bb_mid"]
            if target <= entry:
                return None
            return StrategySignal("long", entry, stop, target, confidence=62,
                                   evidence=["Close below lower Bollinger Band",
                                             "RSI14 < 32 (oversold)",
                                             "ADX < 20 (non-trending regime)"],
                                   invalidation=f"Close below {stop:.2f}")

        if low_adx and overbought:
            entry = row["close"]
            stop = entry + 1.2 * row["atr_14"]
            target = row["bb_mid"]
            if target >= entry:
                return None
            return StrategySignal("short", entry, stop, target, confidence=62,
                                   evidence=["Close above upper Bollinger Band",
                                             "RSI14 > 68 (overbought)",
                                             "ADX < 20 (non-trending regime)"],
                                   invalidation=f"Close above {stop:.2f}")
        return None


class DonchianBreakoutStrategy(Strategy):
    meta = StrategyMeta(
        name="donchian_breakout",
        compatible_regimes=["strong_bull", "strong_bear", "high_volatility"],
        min_risk_reward=1.8,
        suitable_timeframes=["15m", "1h"],
        description="Enter on a close beyond the 20-bar Donchian channel with "
                    "volume expansion, confirming a breakout regime.",
    )

    def evaluate(self, df: pd.DataFrame, i: int):
        if i < 25:
            return None
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        vol_expansion = row.get("rel_volume", 1.0) > 1.3

        breakout_up = row["close"] > prev["donchian_upper"] and vol_expansion
        breakout_down = row["close"] < prev["donchian_lower"] and vol_expansion

        if breakout_up:
            entry = row["close"]
            stop = row["donchian_mid"]
            target = entry + 2.2 * (entry - stop)
            if stop >= entry:
                return None
            return StrategySignal("long", entry, stop, target, confidence=65,
                                   evidence=["Close broke above 20-bar Donchian upper channel",
                                             "Volume > 1.3x average (expansion)"],
                                   invalidation=f"Close back below {stop:.2f}")

        if breakout_down:
            entry = row["close"]
            stop = row["donchian_mid"]
            target = entry - 2.2 * (stop - entry)
            if stop <= entry:
                return None
            return StrategySignal("short", entry, stop, target, confidence=65,
                                   evidence=["Close broke below 20-bar Donchian lower channel",
                                             "Volume > 1.3x average (expansion)"],
                                   invalidation=f"Close back above {stop:.2f}")
        return None


ALL_STRATEGIES = [TrendPullbackStrategy(), BollingerMeanReversionStrategy(), DonchianBreakoutStrategy()]
