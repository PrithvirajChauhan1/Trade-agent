from __future__ import annotations
from strategy.base import Strategy, StrategyMeta, StrategySignal
from features.regime import REGIMES
from learn.combos import combo_name, TP, SL


class ComboEnsembleStrategy(Strategy):
    """Fires when any frozen learned combo fires (one trade per bar; conflicts -> None)."""
    def __init__(self, combos):
        self.combos = combos
        self.meta = StrategyMeta(name="learned_combo_ensemble", compatible_regimes=list(REGIMES),
                                 min_risk_reward=1.5, suitable_timeframes=["1D"],
                                 description="Daily OR of frozen learned indicator combinations; legacy +12%/-7% exits. Not a 5m strategy.")

    def evaluate(self, df, i):
        longs, shorts = [], []
        for cb in self.combos:
            col = f"combo::{combo_name(cb['direction'], cb['names'])}"
            if col in df.columns and bool(df[col].iat[i]):
                (longs if cb["direction"] == "long" else shorts).append(col[7:])
        if bool(longs) == bool(shorts):
            return None
        d = "long" if longs else "short"; px = float(df["close"].iat[i])
        stop = px * (1 - SL) if d == "long" else px * (1 + SL)
        tgt = px * (1 + TP) if d == "long" else px * (1 - TP)
        return StrategySignal(d, px, stop, tgt, confidence=60,
                              evidence=[f"learned combo fired: {c}" for c in (longs or shorts)],
                              invalidation=f"Close {'below' if d=='long' else 'above'} {stop:.2f}")
