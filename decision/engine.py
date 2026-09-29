from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from features.regime import classify_regime
from fundamental.filter import evaluate

NEEDED = ["ema_200", "adx_14", "atr_14", "rsi_14", "mpo", "momentum_score"]

@dataclass
class Decision:
    action: str                       # ENTRY | WAIT | NO-TRADE
    reasons: list = field(default_factory=list)
    direction: str = ""
    entry: float = 0.0; stop: float = 0.0; target: float = 0.0
    regime: str = ""; regime_status: str = ""; fundamental: str = ""; bar_date: str = ""

def decide(df, ensemble, regime_table, fund_inputs, today: dt.date, max_stale_days=5) -> Decision:
    last_date = df.index[-1].date(); d = Decision("NO-TRADE", bar_date=str(last_date))
    if (today - last_date).days > max_stale_days:
        d.reasons.append(f"market data stale: last bar {last_date}"); return d
    if df[NEEDED].iloc[-1].isna().any():
        d.reasons.append("insufficient history for indicators"); return d
    d.regime = classify_regime(df).iloc[-1]
    d.regime_status = regime_table.get(d.regime, {}).get("status", "INSUFFICIENT_SAMPLE")
    if d.regime_status != "ENABLED":
        d.reasons.append(f"regime '{d.regime}' not validated out-of-sample ({d.regime_status})"); return d
    sig = ensemble.evaluate(df, len(df) - 1)
    if sig is None:
        d.action = "WAIT"; d.reasons.append("regime enabled, but no learned setup on the latest bar"); return d
    d.direction = sig.direction
    d.fundamental, why = evaluate(fund_inputs, sig.direction, today)
    d.reasons += sig.evidence + [f"fundamental: {d.fundamental} — " + "; ".join(why)]
    d.entry, d.stop, d.target = sig.entry_price, sig.stop_loss, sig.target
    d.action = {"PASS": "ENTRY", "BLOCK": "NO-TRADE"}.get(d.fundamental, "WAIT")
    return d
