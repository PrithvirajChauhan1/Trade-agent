"""
Risk engine (spec Section 12). Every entry must pass through this before
becoming an executable trade. Also implements circuit breakers that halt
trading — these are checked by the backtester/live loop, not bypassable
by the strategy or signal layer.
"""
from __future__ import annotations
from dataclasses import dataclass
from strategy.base import StrategySignal


@dataclass
class RiskConfig:
    max_risk_per_trade_pct: float = 0.5      # % of account risked per trade
    daily_loss_limit_pct: float = 2.0        # halt trading for the day past this
    max_consecutive_losses: int = 4          # halt + flag for re-evaluation
    max_open_positions: int = 1
    min_risk_reward: float = 1.5
    transaction_cost_pct: float = 0.03       # round-trip, e.g. brokerage
    slippage_pct: float = 0.02               # estimated adverse fill slippage


@dataclass
class PositionSizeResult:
    shares_or_units: float
    dollar_risk: float
    account_risk_pct: float
    rejected: bool = False
    rejection_reason: str = ""


class RiskEngine:
    def __init__(self, config: RiskConfig, account_equity: float):
        self.cfg = config
        self.equity = account_equity
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.open_positions = 0
        self.halted = False
        self.halt_reason = ""

    def check_circuit_breakers(self) -> bool:
        """Returns True if trading is currently allowed."""
        if self.halted:
            return False
        if self.daily_pnl <= -abs(self.cfg.daily_loss_limit_pct / 100 * self.equity):
            self.halted = True
            self.halt_reason = "Daily loss limit exceeded"
            return False
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.halted = True
            self.halt_reason = "Max consecutive losses reached — halted for re-evaluation"
            return False
        if self.open_positions >= self.cfg.max_open_positions:
            return False
        return True

    def size_position(self, signal: StrategySignal) -> PositionSizeResult:
        if signal.risk_reward < self.cfg.min_risk_reward:
            return PositionSizeResult(0, 0, 0, rejected=True,
                                       rejection_reason=f"R:R {signal.risk_reward} below minimum {self.cfg.min_risk_reward}")

        risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        if risk_per_unit <= 0:
            return PositionSizeResult(0, 0, 0, rejected=True, rejection_reason="Zero/invalid stop distance")

        dollar_risk = self.equity * (self.cfg.max_risk_per_trade_pct / 100)
        units = dollar_risk / risk_per_unit

        return PositionSizeResult(
            shares_or_units=round(units, 4),
            dollar_risk=round(dollar_risk, 2),
            account_risk_pct=self.cfg.max_risk_per_trade_pct,
        )

    def apply_costs(self, gross_pnl: float, notional: float) -> float:
        cost = notional * (self.cfg.transaction_cost_pct + self.cfg.slippage_pct) / 100
        return gross_pnl - cost

    def record_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        self.open_positions = max(0, self.open_positions - 1)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset_day(self):
        self.daily_pnl = 0.0
        # Daily-loss halts are session-scoped; the consecutive-loss breaker is
        # intentionally sticky across days.
        if self.halt_reason == "Daily loss limit exceeded":
            self.halted = False
            self.halt_reason = ""
