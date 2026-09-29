"""
Strategy schema (spec Section 7). Every strategy is a structured object with
explicit entry/exit/stop logic — never a black box. Strategies emit
StrategySignal objects that the risk engine and backtester consume identically
whether in backtest or live/paper mode (same code path = no train/live skew).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import pandas as pd

Direction = Literal["long", "short"]


@dataclass
class StrategySignal:
    direction: Direction
    entry_price: float
    stop_loss: float
    target: float
    confidence: float          # 0-100, strategy-internal confidence
    evidence: list[str] = field(default_factory=list)
    invalidation: str = ""

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        return round(reward / risk, 3) if risk > 0 else 0.0


@dataclass
class StrategyMeta:
    """Static description of a strategy — the 'card' the spec asks for in Section 7."""
    name: str
    compatible_regimes: list[str]
    min_risk_reward: float
    suitable_timeframes: list[str]
    description: str = ""


class Strategy:
    """Base class. Subclasses implement `evaluate`, called once per bar with
    all history up to and including that bar (no lookahead)."""

    meta: StrategyMeta

    def evaluate(self, df: pd.DataFrame, i: int) -> Optional[StrategySignal]:
        """
        df: full indicator-enriched OHLCV frame.
        i: integer position of the current bar (df.iloc[i] is 'now').
        Return a StrategySignal if this strategy's entry conditions fire on
        this bar, else None. Must not look at df.iloc[i+1:].
        """
        raise NotImplementedError

    def regime_ok(self, current_regime: str) -> bool:
        return current_regime in self.meta.compatible_regimes
