"""
Backtesting engine (Sections 13-14). Bar-by-bar simulation, no lookahead:
at bar i we only use df.iloc[:i+1]. Supports walk-forward split so the
80%-win-rate hypothesis is tested out-of-sample, never on the fit data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from strategy.base import Strategy, StrategySignal
from features.regime import classify_regime
from risk.engine import RiskEngine, RiskConfig


@dataclass
class Trade:
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    target: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    exit_reason: str = ""
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    regime_at_entry: str = ""


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        return round(100 * wins / len(self.trades), 2)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        return round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    @property
    def expectancy(self) -> float:
        """Average net PnL per trade, in account-currency terms."""
        if not self.trades:
            return 0.0
        return round(sum(t.net_pnl for t in self.trades) / len(self.trades), 4)

    @property
    def expectancy_r(self) -> float:
        """Average R-multiple per trade — the size-independent version."""
        if not self.trades:
            return 0.0
        return round(np.mean([t.r_multiple for t in self.trades]), 4)

    @property
    def max_drawdown_pct(self) -> float:
        if not self.trades:
            return 0.0
        equity_curve = np.cumsum([t.net_pnl for t in self.trades])
        peak = np.maximum.accumulate(equity_curve)
        dd = equity_curve - peak
        return round(float(dd.min()), 4)

    @property
    def avg_win(self) -> float:
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        return round(np.mean(wins), 4) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        return round(np.mean(losses), 4) if losses else 0.0

    def summary(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "win_rate_pct": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy_per_trade": self.expectancy,
            "expectancy_R": self.expectancy_r,
            "max_drawdown": self.max_drawdown_pct,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
        }

    def by_strategy(self) -> dict[str, dict]:
        out = {}
        for name in sorted(set(t.strategy for t in self.trades)):
            sub = BacktestResult([t for t in self.trades if t.strategy == name])
            out[name] = sub.summary()
        return out

    def by_regime(self) -> dict[str, dict]:
        out = {}
        for name in sorted(set(t.regime_at_entry for t in self.trades)):
            sub = BacktestResult([t for t in self.trades if t.regime_at_entry == name])
            out[name] = sub.summary()
        return out


def run_backtest(
    df: pd.DataFrame,
    strategies: list[Strategy],
    risk_config: RiskConfig = None,
    account_equity: float = 100_000.0,
    max_holding_bars: int = 60,
) -> BacktestResult:
    """
    Single-position-at-a-time bar-by-bar backtest. df must already have
    indicators (features.indicators.compute_all) applied.
    """
    risk_config = risk_config or RiskConfig()
    risk = RiskEngine(risk_config, account_equity)

    df = df.copy()
    df["regime"] = classify_regime(df)

    result = BacktestResult()
    open_trade: Trade = None
    open_strategy_signal: StrategySignal = None
    entry_bar_idx = None
    last_date = None

    for i in range(len(df)):
        row = df.iloc[i]

        # daily reset (approx: date change)
        cur_date = row.name.date()
        if last_date is not None and cur_date != last_date:
            risk.reset_day()
        last_date = cur_date

        if open_trade is not None:
            hit_stop = (row["low"] <= open_trade.stop_loss) if open_trade.direction == "long" else (row["high"] >= open_trade.stop_loss)
            hit_target = (row["high"] >= open_trade.target) if open_trade.direction == "long" else (row["low"] <= open_trade.target)
            timed_out = (i - entry_bar_idx) >= max_holding_bars

            if hit_stop or hit_target or timed_out:
                if hit_stop:
                    exit_price, reason = open_trade.stop_loss, "stop_loss"
                elif hit_target:
                    exit_price, reason = open_trade.target, "target"
                else:
                    exit_price, reason = row["close"], "timeout"

                sign = 1 if open_trade.direction == "long" else -1
                gross = sign * (exit_price - open_trade.entry_price)
                notional = open_trade.entry_price
                net = risk.apply_costs(gross, notional)
                risk_per_unit = abs(open_trade.entry_price - open_trade.stop_loss)

                open_trade.exit_time = row.name
                open_trade.exit_price = exit_price
                open_trade.exit_reason = reason
                open_trade.gross_pnl = round(gross, 4)
                open_trade.net_pnl = round(net, 4)
                open_trade.r_multiple = round(net / risk_per_unit, 4) if risk_per_unit > 0 else 0.0

                result.trades.append(open_trade)
                risk.record_trade_result(net)
                open_trade = None
                open_strategy_signal = None
                entry_bar_idx = None
            continue  # one position at a time — don't evaluate new entries mid-trade

        if not risk.check_circuit_breakers():
            continue

        current_regime = row["regime"]
        for strat in strategies:
            if not strat.regime_ok(current_regime):
                continue
            sig = strat.evaluate(df, i)
            if sig is None:
                continue
            sizing = risk.size_position(sig)
            if sizing.rejected:
                continue

            open_trade = Trade(
                strategy=strat.meta.name,
                direction=sig.direction,
                entry_time=row.name,
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                target=sig.target,
                regime_at_entry=current_regime,
            )
            open_strategy_signal = sig
            entry_bar_idx = i
            risk.open_positions += 1
            break  # first qualifying strategy wins this bar

    return result


def walk_forward_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    """Returns (train_df, val_df, test_df). test_df is held out for the final,
    single out-of-sample check — never touched during strategy tuning."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
