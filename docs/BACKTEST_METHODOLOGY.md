# Backtest methodology (5-minute research)

**Protocol.** Sessions are split chronologically: TRAIN 50% / VALIDATION 25% / TEST 25%. Exit configuration (holding period 1-24 bars, R:R 1-3, stop 1-2 ATR,
optional 1.5 ATR trailing stop) is chosen on TRAIN+VALIDATION only: a config is eligible only if net expectancy > 0 in both, with minimum trade counts; the best
validation expectancy wins. The choice is frozen, then evaluated once on TEST. 270 configs are tried per strategy, so validation figures are optimistic by construction.

**Execution.** Signal uses bars <= T. Entry = open of T+1, same session only. Exits: stop, target, trailing stop (tightens from the next bar), time exit at the close of the
last hold bar, forced exit at session end (no overnight). Gap through a stop fills at the open (worse than the stop); gap through a target fills at the target (conservative).
**Same-candle TP and SL:** predetermined rule = stop first; such bars are counted (`ambiguous_bars`). Lower-timeframe/tick resolution is not available here.

**Costs.** Configurable `CostModel`: commission per side, half the spread and slippage per side (adverse), round-trip statutory levies. Defaults are illustrative
(1 bp each) and MUST be set to your broker/exchange schedule. All headline numbers are net of costs; cost sensitivity at x0/x1/x2/x3 is reported.

**Sizing/risk.** Risk-based units (0.5% equity / stop distance) via `RiskEngine`; daily loss halt resets each session; consecutive-loss halt is disabled inside research
runs to avoid truncating the sample (reported separately if enabled).

**200-trade rule.** The OOS research sample is the first 200 trades chronologically (cap fixed in advance, `--cap`). If fewer occur, the actual count is reported.

**Verdicts.** VALIDATED needs: >=30 OOS trades, OOS net expectancy > 0, bootstrap 95% CI lower bound > 0, >=60% positive walk-forward folds, >=50% profitable parameter
neighbours. REJECTED: no config profitable in both train and validation, or negative OOS expectancy. Otherwise EXPERIMENTAL. Flags: UNSTABLE, GRID_EDGE.
No path here can set PAPER_TRADING or live status without a named human approver.

**Walk-forward.** Expanding window (first 40% of sessions), 20-session test blocks, re-selection inside each block using an inner 2/3-1/3 split.

**Known limitations.** Single symbol; synthetic data in this environment; Yahoo 5m history is only ~60 days (insufficient for train/val/test/walk-forward); no
corporate-action handling; index data usually has no volume; fundamentals/market context not connected.
