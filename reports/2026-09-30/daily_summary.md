# Daily research summary — 2026-09-30
**Data:** SYNTHETIC DATA — machinery test, NOT a market signal
**Protocol:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
**Costs:** CostModel(commission_bps=1.0, spread_bps=1.0, slippage_bps=1.0, tax_bps=1.0, mult=1.0)

## Market summary
- Regime (last bar): strong_bull; high relative volume: False; gap day: False
- Volatility percentile / SPY / QQQ / VIX / sector leadership: not supplied (market-context data not wired yet)

## Strategy summary (OOS test, net of costs)
| strategy           | verdict      | flag                                              | config               |   test_n_sample |   test_win_rate |   test_exp_R |   test_PF |   sharpe |   max_dd_R |   wf_positive_frac |   neighbors_profitable |
|:-------------------|:-------------|:--------------------------------------------------|:---------------------|----------------:|----------------:|-------------:|----------:|---------:|-----------:|-------------------:|-----------------------:|
| TREND_VWAP_001     | VALIDATED    | GRID_EDGE(optimum may lie outside searched range) | (24, 3.0, 2.0, None) |             129 |           45.74 |       0.2701 |     1.544 |     3.85 |      14.1  |               1    |                      1 |
| ORB_VOLUME_001     | REJECTED     |                                                   | nan                  |             nan |          nan    |     nan      |   nan     |   nan    |     nan    |             nan    |                    nan |
| TREND_PULLBACK_001 | EXPERIMENTAL | GRID_EDGE(optimum may lie outside searched range) | (24, 3.0, 2.0, None) |              67 |           52.24 |       0.2839 |     1.572 |     3.54 |       6.89 |               0.71 |                      1 |
| MEAN_REVERSION_001 | REJECTED     |                                                   | nan                  |             nan |          nan    |     nan      |   nan     |   nan    |     nan    |             nan    |                    nan |

## Active setups on the last closed bar (research only; entry = NEXT bar open)
- none — NO TRADE

## Top research findings
- Best OOS setup: TREND_PULLBACK_001 (0.2839R/trade, n=67.0); worst: TREND_VWAP_001 (0.2701R/trade).
- Train/val -> OOS expectancy decay (overfitting evidence): TREND_VWAP_001: -0.00R, TREND_PULLBACK_001: +0.12R
- Cost sensitivity (OOS exp R at cost x0/x1/x2/x3): TREND_VWAP_001: {"x0": 0.442, "x1": 0.27, "x2": 0.108, "x3": -0.056}; ORB_VOLUME_001: nan; TREND_PULLBACK_001: {"x0": 0.454, "x1": 0.284, "x2": 0.132, "x3": -0.02}; MEAN_REVERSION_001: nan
- Strategies REJECTED with no train+validation-profitable exit config: ORB_VOLUME_001, MEAN_REVERSION_001
- Features with sign-stable IC across train/val/test: rsi_14, ema9_21_spread_atr, ema21_50_spread_atr, vwap_dist_atr, bb_pctb, bb_width, dist_prev_high_atr, dist_prev_low_atr, dist_or_high_atr, htf_trend_15m, htf_trend_1h, day_range_pos
- Redundant feature pairs (|rho|>0.85): [('rsi_14', 'ema9_21_spread_atr', 0.94), ('rsi_14', 'vwap_dist_atr', 0.9), ('rsi_14', 'bb_pctb', 0.9), ('rsi_14', 'day_range_pos', 0.9), ('ema9_21_spread_atr', 'vwap_dist_atr', 0.88), ('ema9_21_spread_atr', 'day_range_pos', 0.85), ('ema21_50_spread_atr', 'dist_prev_high_atr', 0.9), ('ema21_50_spread_atr', 'dist_prev_low_atr', 0.89), ('vwap_dist_atr', 'dist_or_high_atr', 0.92), ('vwap_dist_atr', 'day_range_pos', 0.95), ('dist_or_high_atr', 'day_range_pos', 0.9)]
- VALIDATED strategies: 1. None is deployed; paper/live use requires explicit human approval (registry blocks it otherwise).

_Multiple-testing note: each strategy tried 270 exit configs; validation-selected results are optimistic, only the OOS test is unbiased._