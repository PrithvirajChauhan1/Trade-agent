"""
Custom strategies using learned parameters from 2000-2020 training.
Exit rules: +12% profit target (hard), -7% stop loss (hard), per requirement.

Five strategies covering different market regimes:
  1. AdaptiveEMATrend    — slope-based EMA trend with MPO + momentum confirm
  2. MPOBreakout         — MPO surge above 60 with structure break
  3. RegimeRSIReversion  — trained RSI thresholds in ranging regime
  4. MomentumSurge       — composite momentum score crossings
  5. StructureBreak      — swing high/low break with volume + MPO
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategy.base import Strategy, StrategyMeta, StrategySignal
from features.custom_indicators import LearnedParams

PROFIT_TARGET_PCT = 0.12
STOP_LOSS_PCT     = 0.07

def _fixed_exits(entry: float, direction: str):
    if direction == "long":
        return entry * (1 - STOP_LOSS_PCT), entry * (1 + PROFIT_TARGET_PCT)
    else:
        return entry * (1 + STOP_LOSS_PCT), entry * (1 - PROFIT_TARGET_PCT)


class AdaptiveEMATrendStrategy(Strategy):
    meta = StrategyMeta(
        name="adaptive_ema_trend",
        compatible_regimes=["strong_bull", "weak_bull", "strong_bear", "weak_bear"],
        min_risk_reward=1.5,
        suitable_timeframes=["1D"],
        description="Trained adaptive EMA slope + MPO energy filter. Exits: +12%/-7%.",
    )

    def evaluate(self, df, i):
        if i < 50:
            return None
        row, prev = df.iloc[i], df.iloc[i - 1]
        if pd.isna(row.get("aema_fast")) or pd.isna(row.get("mpo")):
            return None

        fast_rising  = float(row["aema_fast"]) > float(prev["aema_fast"]) and int(row["aema_cross"]) == 1
        fast_falling = float(row["aema_fast"]) < float(prev["aema_fast"]) and int(row["aema_cross"]) == 0
        mpo_ok = float(row["mpo"]) > 50
        mom_long  = float(row.get("momentum_score", 0)) > 0.2
        mom_short = float(row.get("momentum_score", 0)) < -0.2

        if fast_rising and mpo_ok and mom_long:
            stop, target = _fixed_exits(row["close"], "long")
            return StrategySignal("long", row["close"], stop, target, confidence=72,
                evidence=["Adaptive EMA slope rising (bullish)",
                           f"MPO={row['mpo']:.1f}", f"MomentumScore={row.get('momentum_score',0):.2f}"],
                invalidation=f"Close below {stop:.2f}")

        if fast_falling and mpo_ok and mom_short:
            stop, target = _fixed_exits(row["close"], "short")
            return StrategySignal("short", row["close"], stop, target, confidence=72,
                evidence=["Adaptive EMA slope falling (bearish)",
                           f"MPO={row['mpo']:.1f}", f"MomentumScore={row.get('momentum_score',0):.2f}"],
                invalidation=f"Close above {stop:.2f}")
        return None


class MPOBreakoutStrategy(Strategy):
    meta = StrategyMeta(
        name="mpo_breakout",
        compatible_regimes=["strong_bull", "strong_bear", "weak_bull", "weak_bear", "high_volatility"],
        min_risk_reward=1.5,
        suitable_timeframes=["1D"],
        description="MPO above 60 with Donchian structure break. Exits: +12%/-7%.",
    )

    def evaluate(self, df, i):
        if i < 30:
            return None
        row, prev = df.iloc[i], df.iloc[i - 1]
        if pd.isna(row.get("mpo")):
            return None

        mpo_high = float(row["mpo"]) > 60
        db_up   = row["close"] > prev.get("donchian_upper", np.inf)
        db_down = row["close"] < prev.get("donchian_lower", -np.inf)
        trend_up   = row["ema_21"] > row["ema_50"]
        trend_down = row["ema_21"] < row["ema_50"]

        if mpo_high and db_up and trend_up:
            stop, target = _fixed_exits(row["close"], "long")
            return StrategySignal("long", row["close"], stop, target, confidence=68,
                evidence=["MPO>60", "Donchian breakout up", "EMA trend aligned"],
                invalidation=f"Close below {stop:.2f}")

        if mpo_high and db_down and trend_down:
            stop, target = _fixed_exits(row["close"], "short")
            return StrategySignal("short", row["close"], stop, target, confidence=68,
                evidence=["MPO>60", "Donchian breakdown", "EMA trend aligned"],
                invalidation=f"Close above {stop:.2f}")
        return None


class RegimeRSIReversionStrategy(Strategy):
    meta = StrategyMeta(
        name="regime_rsi_reversion",
        compatible_regimes=["sideways", "low_volatility", "weak_bull", "weak_bear"],
        min_risk_reward=1.5,
        suitable_timeframes=["1D"],
        description="Trained RSI extreme thresholds in ranging/low-vol regime. Exits: +12%/-7%.",
    )

    def evaluate(self, df, i):
        if i < 20:
            return None
        row = df.iloc[i]
        rsi_sig = int(row.get("rsi_signal", 0))
        low_adx = float(row["adx_14"]) < 30

        if rsi_sig == 1 and low_adx:
            stop, target = _fixed_exits(row["close"], "long")
            return StrategySignal("long", row["close"], stop, target, confidence=63,
                evidence=[f"RSI oversold (trained threshold)", f"ADX={row['adx_14']:.1f}<30"],
                invalidation=f"Close below {stop:.2f}")

        if rsi_sig == -1 and low_adx:
            stop, target = _fixed_exits(row["close"], "short")
            return StrategySignal("short", row["close"], stop, target, confidence=63,
                evidence=[f"RSI overbought (trained threshold)", f"ADX={row['adx_14']:.1f}<30"],
                invalidation=f"Close above {stop:.2f}")
        return None


class MomentumSurgeStrategy(Strategy):
    meta = StrategyMeta(
        name="momentum_surge",
        compatible_regimes=["strong_bull", "strong_bear", "weak_bull", "weak_bear"],
        min_risk_reward=1.5,
        suitable_timeframes=["1D"],
        description="Composite momentum score (trained weights) extremes. Exits: +12%/-7%.",
    )

    def evaluate(self, df, i):
        if i < 30:
            return None
        row, prev = df.iloc[i], df.iloc[i - 1]
        if pd.isna(row.get("momentum_score")):
            return None

        ms = float(row["momentum_score"])
        ms_prev = float(prev["momentum_score"])
        mpo_ok = float(row.get("mpo", 0)) > 45
        trend_long  = row["ema_21"] > row["ema_50"]
        trend_short = row["ema_21"] < row["ema_50"]

        # Cross above 0.40 threshold (trained)
        cross_long  = ms > 0.40 and ms_prev <= 0.40
        cross_short = ms < -0.40 and ms_prev >= -0.40

        if cross_long and mpo_ok and trend_long:
            stop, target = _fixed_exits(row["close"], "long")
            return StrategySignal("long", row["close"], stop, target, confidence=66,
                evidence=[f"MomentumScore crossed +0.40: {ms:.2f}", f"MPO={row.get('mpo',0):.1f}", "EMA trend up"],
                invalidation=f"Close below {stop:.2f}")

        if cross_short and mpo_ok and trend_short:
            stop, target = _fixed_exits(row["close"], "short")
            return StrategySignal("short", row["close"], stop, target, confidence=66,
                evidence=[f"MomentumScore crossed -0.40: {ms:.2f}", f"MPO={row.get('mpo',0):.1f}", "EMA trend down"],
                invalidation=f"Close above {stop:.2f}")
        return None


class StructureBreakStrategy(Strategy):
    meta = StrategyMeta(
        name="structure_break",
        compatible_regimes=["strong_bull", "strong_bear", "sideways", "weak_bull", "weak_bear"],
        min_risk_reward=1.5,
        suitable_timeframes=["1D"],
        description="15-bar swing break + MPO>55 + volume surge. Exits: +12%/-7%.",
    )

    def evaluate(self, df, i):
        if i < 20:
            return None
        row = df.iloc[i]
        window = df.iloc[i - 15:i]
        swing_high = window["high"].max()
        swing_low  = window["low"].min()
        mpo_ok   = float(row.get("mpo", 0)) > 55
        vol_surge = float(row.get("rel_volume", 1.0)) > 1.15

        if row["close"] > swing_high and mpo_ok and vol_surge:
            stop, target = _fixed_exits(row["close"], "long")
            return StrategySignal("long", row["close"], stop, target, confidence=65,
                evidence=[f"Close broke 15-bar swing high {swing_high:.2f}",
                           f"MPO={row.get('mpo',0):.1f}", f"RelVol={row.get('rel_volume',1):.2f}x"],
                invalidation=f"Close below {stop:.2f}")

        if row["close"] < swing_low and mpo_ok and vol_surge:
            stop, target = _fixed_exits(row["close"], "short")
            return StrategySignal("short", row["close"], stop, target, confidence=65,
                evidence=[f"Close broke 15-bar swing low {swing_low:.2f}",
                           f"MPO={row.get('mpo',0):.1f}", f"RelVol={row.get('rel_volume',1):.2f}x"],
                invalidation=f"Close above {stop:.2f}")
        return None


def get_custom_strategies() -> list[Strategy]:
    return [
        AdaptiveEMATrendStrategy(),
        MPOBreakoutStrategy(),
        RegimeRSIReversionStrategy(),
        MomentumSurgeStrategy(),
        StructureBreakStrategy(),
    ]
