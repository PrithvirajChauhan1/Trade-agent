"""
Fixed-exit backtester: exits at +12% (profit) or -7% (loss), whichever
hits first on the bar's high/low. No timeout exit — the 12%/7% levels
are the only exits per the requirement.

R:R of these fixed exits: 12/7 = 1.714:1
For positive expectancy, win rate must exceed: 7 / (12+7) = 36.8%.
The system reports actual win rate, not the 80% target.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional

from strategy.base import Strategy, StrategySignal
from features.regime import classify_regime
from risk.engine import RiskEngine, RiskConfig

PROFIT_TARGET_PCT = 0.12
STOP_LOSS_PCT     = 0.07


@dataclass
class FixedExitTrade:
    strategy:     str
    direction:    str
    entry_time:   pd.Timestamp
    entry_price:  float
    stop_loss:    float
    target:       float
    exit_time:    Optional[pd.Timestamp] = None
    exit_price:   Optional[float] = None
    exit_reason:  str = ""
    gross_pct:    float = 0.0   # % move from entry
    net_pct:      float = 0.0   # after costs
    win:          bool = False
    regime:       str = ""
    epoch:        str = ""      # for synthetic data — real data won't have this
    confidence:   float = 0.0
    signal_evidence: list = field(default_factory=list)


@dataclass
class FullBacktestResult:
    trades: list[FixedExitTrade] = field(default_factory=list)
    period_label: str = ""

    @property
    def n(self): return len(self.trades)

    @property
    def wins(self): return [t for t in self.trades if t.win]

    @property
    def losses(self): return [t for t in self.trades if not t.win]

    @property
    def win_rate(self): return round(100 * len(self.wins) / self.n, 2) if self.n else 0.0

    @property
    def avg_win_pct(self): return round(np.mean([t.net_pct for t in self.wins]) * 100, 2) if self.wins else 0.0

    @property
    def avg_loss_pct(self): return round(np.mean([t.net_pct for t in self.losses]) * 100, 2) if self.losses else 0.0

    @property
    def profit_factor(self):
        g = sum(t.net_pct for t in self.wins)
        l = abs(sum(t.net_pct for t in self.losses))
        return round(g / l, 3) if l > 0 else (float("inf") if g > 0 else 0.0)

    @property
    def expectancy_pct(self):
        return round(np.mean([t.net_pct for t in self.trades]) * 100, 4) if self.trades else 0.0

    @property
    def required_win_rate(self):
        """Minimum win rate needed for positive expectancy with these fixed exits."""
        return round(100 * STOP_LOSS_PCT / (PROFIT_TARGET_PCT + STOP_LOSS_PCT), 2)

    @property
    def equity_curve(self) -> pd.Series:
        """Compound equity curve (starting at 1.0)."""
        curve = [1.0]
        for t in self.trades:
            curve.append(curve[-1] * (1 + t.net_pct))
        return pd.Series(curve)

    @property
    def max_drawdown_pct(self):
        eq = self.equity_curve.values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return round(float(dd.min()) * 100, 2)

    @property
    def cagr(self):
        if self.n < 2:
            return 0.0
        first = self.trades[0].entry_time
        last  = self.trades[-1].exit_time or self.trades[-1].entry_time
        years = max((last - first).days / 365.25, 0.01)
        final_equity = float(self.equity_curve.iloc[-1])
        return round((final_equity ** (1 / years) - 1) * 100, 2)

    @property
    def consecutive_stats(self):
        max_w, max_l, cur_w, cur_l = 0, 0, 0, 0
        for t in self.trades:
            if t.win:
                cur_w += 1; cur_l = 0; max_w = max(max_w, cur_w)
            else:
                cur_l += 1; cur_w = 0; max_l = max(max_l, cur_l)
        return max_w, max_l

    def by_strategy(self) -> dict:
        strats = sorted(set(t.strategy for t in self.trades))
        return {s: FullBacktestResult([t for t in self.trades if t.strategy == s]).summary()
                for s in strats}

    def by_regime(self) -> dict:
        regimes = sorted(set(t.regime for t in self.trades))
        return {r: FullBacktestResult([t for t in self.trades if t.regime == r]).summary()
                for r in regimes}

    def by_epoch(self) -> dict:
        epochs = sorted(set(t.epoch for t in self.trades if t.epoch))
        return {e: FullBacktestResult([t for t in self.trades if t.epoch == e]).summary()
                for e in epochs}

    def summary(self) -> dict:
        max_cw, max_cl = self.consecutive_stats
        return {
            "period": self.period_label,
            "n_trades": self.n,
            "win_rate_pct": self.win_rate,
            "required_win_rate_pct": self.required_win_rate,
            "profit_target_pct": PROFIT_TARGET_PCT * 100,
            "stop_loss_pct": STOP_LOSS_PCT * 100,
            "risk_reward_ratio": round(PROFIT_TARGET_PCT / STOP_LOSS_PCT, 3),
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "profit_factor": self.profit_factor,
            "expectancy_per_trade_pct": self.expectancy_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "cagr_pct": self.cagr,
            "max_consecutive_wins": max_cw,
            "max_consecutive_losses": max_cl,
        }


def run_fixed_exit_backtest(
    df: pd.DataFrame,
    strategies: list[Strategy],
    risk_config: RiskConfig = None,
    account_equity: float = 100_000.0,
    period_label: str = "",
    transaction_cost_pct: float = 0.05,
    max_concurrent: int = 5,
) -> FullBacktestResult:
    """
    Bar-by-bar fixed-exit simulation with multiple concurrent positions.

    Physical trade-count constraint on daily bars:
      With 1 position at a time and ~10-day avg hold, ~1260 daily bars
      yields at most ~126 trades. To reach ~500 trades, max_concurrent=5
      is needed. Each slot is independently sized; risk limits apply per-slot.

    Exits checked per-bar per open position. Worst-case assumption when
    both target and stop are hit on the same bar: stop triggered first.
    """
    risk_config = risk_config or RiskConfig()

    df = df.copy()
    df["regime"] = classify_regime(df)

    result = FullBacktestResult(period_label=period_label)
    # Multiple concurrent slots
    open_slots: list[Optional[FixedExitTrade]] = [None] * max_concurrent
    last_year = None

    for i in range(len(df)):
        row = df.iloc[i]
        cur_year = row.name.year
        daily_pnl = 0.0
        halted = False

        # --- Close out any hitting their fixed exits ---
        for slot_idx, trade in enumerate(open_slots):
            if trade is None:
                continue
            if trade.direction == "long":
                hit_target = row["high"] >= trade.target
                hit_stop   = row["low"]  <= trade.stop_loss
            else:
                hit_target = row["low"]  <= trade.target
                hit_stop   = row["high"] >= trade.stop_loss

            if hit_stop or hit_target:
                if hit_target and not hit_stop:
                    exit_px, reason = trade.target, "profit_target"
                elif hit_stop and not hit_target:
                    exit_px, reason = trade.stop_loss, "stop_loss"
                else:
                    exit_px, reason = trade.stop_loss, "stop_loss_gap"

                sign = 1 if trade.direction == "long" else -1
                gross_pct = sign * (exit_px / trade.entry_price - 1)
                net_pct   = gross_pct - transaction_cost_pct / 100
                daily_pnl += net_pct

                trade.exit_time   = row.name
                trade.exit_price  = round(exit_px, 4)
                trade.exit_reason = reason
                trade.gross_pct   = round(gross_pct, 6)
                trade.net_pct     = round(net_pct, 6)
                trade.win         = bool(net_pct > 0)
                result.trades.append(trade)
                open_slots[slot_idx] = None

        # --- Daily loss circuit breaker ---
        if daily_pnl < -(risk_config.daily_loss_limit_pct / 100):
            halted = True

        # --- Open new positions in free slots ---
        if not halted:
            current_regime = row["regime"]
            epoch = row.get("epoch", "")

            for strat in strategies:
                if not strat.regime_ok(current_regime):
                    continue
                sig = strat.evaluate(df, i)
                if sig is None:
                    continue
                if sig.risk_reward < risk_config.min_risk_reward:
                    continue

                # Find a free slot
                free = next((idx for idx, s in enumerate(open_slots) if s is None), None)
                if free is None:
                    break  # all slots occupied

                open_slots[free] = FixedExitTrade(
                    strategy=strat.meta.name,
                    direction=sig.direction,
                    entry_time=row.name,
                    entry_price=round(sig.entry_price, 4),
                    stop_loss=round(sig.stop_loss, 4),
                    target=round(sig.target, 4),
                    regime=current_regime,
                    epoch=str(epoch) if epoch else "",
                    confidence=sig.confidence,
                    signal_evidence=sig.evidence,
                )

    # Force-close remaining open positions at end of period
    last_row = df.iloc[-1]
    for trade in open_slots:
        if trade is None:
            continue
        sign = 1 if trade.direction == "long" else -1
        gross_pct = sign * (last_row["close"] / trade.entry_price - 1)
        net_pct = gross_pct - transaction_cost_pct / 100
        trade.exit_time   = last_row.name
        trade.exit_price  = round(float(last_row["close"]), 4)
        trade.exit_reason = "end_of_period"
        trade.gross_pct   = round(gross_pct, 6)
        trade.net_pct     = round(net_pct, 6)
        trade.win         = bool(net_pct > 0)
        result.trades.append(trade)

    result.trades.sort(key=lambda t: t.entry_time)
    return result
