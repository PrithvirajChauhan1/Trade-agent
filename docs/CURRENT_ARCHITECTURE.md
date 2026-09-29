# Current architecture and audit (5-minute upgrade)

## Module map
| Area | Module | Role |
|---|---|---|
| Data | `data/session.py`, `data/quality.py`, `data/synthetic_5m.py`, `data/provider.py`, `data/real_source.py` | session config (NSE/US), data-quality gate, synthetic 5m generator (tests only), Yahoo adapters |
| Indicators | `features/indicators.py` (daily-era, reused), `features/intraday.py` (session-aware layer) | EMA/SMA/RSI/MACD/ATR/ADX/BB/Donchian/OBV; session VWAP, time-of-day rel. volume, prev-day levels, gap, opening range, 15m/1h trend, tod buckets |
| Regime | `features/regime.py` (reused, lookback now parameterised) | trend/vol regime labels |
| Strategy | `strategy/intraday_components.py` (new, config-driven), `strategy/base.py`, legacy daily strategies | reusable components -> long/short signal arrays; 4 candidate composites |
| Execution | `backtest/intraday.py` (new) | T+1 open entry, stop/target/trail/time/session-end exits, costs, metrics |
| Risk | `risk/engine.py` (reused + fixed) | sizing, daily loss halt, consecutive-loss halt |
| Research | `research/study.py`, `research/run_research.py`, `research/registry.py`, `research/feature_stats.py`, `research/docs.py`, `patterns/candles.py`, `charts/make.py` | grid/selection/walk-forward/robustness, versioned registry, reports, charts |
| Legacy daily | `backtest/fixed_exit.py`, `learn/*`, `fundamental/*`, `decision/*`, `run_pipeline.py` | daily-bar study; untouched, not used by the 5m path |

## Audit findings
| # | Finding | Evidence | Status |
|---|---|---|---|
| 1 | `features.indicators.vwap` is cumulative over the entire series (never resets per session) | 3-session test: day-2 VWAP 100.3 vs true ~120 | FIXED for intraday (`features/intraday.py` session VWAP). Legacy function kept for daily data; do not use on intraday |
| 2 | `RiskEngine` daily-loss halt never lifted by `reset_day()` | audit script + `test_daily_loss_halt_lifts_next_session` | FIXED. Consecutive-loss halt intentionally persists (test added) |
| 3 | Legacy backtesters (`engine.py`, `fixed_exit.py`) enter at the signal bar's own close (not T+1) | code inspection | OPEN for legacy; new engine enters at next open. Recommend retiring legacy engines after the daily study is archived |
| 4 | Two legacy backtest engines already overlapped in function | `backtest/engine.py` vs `fixed_exit.py` | OPEN (documented, not deleted, tests still depend on them) |
| 5 | Regime ATR-percentile lookback = 100 bars (~1.3 sessions at 5m) | `classify_regime(lookback_vol=100)` | FIXED via parameter (5 sessions on intraday). ADX thresholds 15/25 are not validated for 5m |
| 6 | `rel_volume` (20-bar) mixes times of day (open volume >> lunch) | design | FIXED: `rel_volume_tod` (same bar-of-day, prior 20 sessions) |
| 7 | No timezone/session handling anywhere | code inspection | FIXED: tz-aware requirement, `SessionConfig`, quality gate |
| 8 | Overnight gaps flow through EMA/ATR (continuous series) | design | OPEN, accepted; gap features exposed separately |
| 9 | Daily-study custom indicators (MPO, adaptive EMA, learned combos) were tuned on daily bars | history | Not ported; not used by 5m path |
| 10 | No indicators warm-up guard on some legacy strategies | code inspection | Signals treat NaN as False in the new path |

## Not implemented yet (deliberately deferred; see ALGORITHM_DEVELOPMENT_LOG)
Market-wide context (SPY/QQQ/VIX/breadth/sector), ML probability models, chart-pattern engine (double top, H&S, triangles, flags), extra indicators
(Ichimoku, SAR, Aroon, Supertrend, HMA, Keltner, MFI, CCI, StochRSI, Williams %R, anchored VWAP), multi-symbol universe, 1-minute/tick same-candle resolution,
corporate-action adjustment, per-year/sector/symbol robustness tables (single-symbol here), pre/post-market handling.
