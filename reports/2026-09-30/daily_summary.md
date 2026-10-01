# Daily research summary — 2026-09-30
**Data:** SYNTHETIC DATA — machinery test, NOT a market signal
**Protocol:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
**Costs:** CostModel(commission_bps=1.0, spread_bps=1.0, slippage_bps=1.0, tax_bps=1.0, mult=1.0)

## Market summary
- Regime (last bar): strong_bull; high relative volume: False; gap day: False
- Volatility percentile / SPY / QQQ / VIX / sector leadership: not supplied (market-context data not wired yet)

## Strategy summary (OOS test, net of costs)
| strategy                 | verdict      | flag                                              | config               |   test_n_sample |   test_win_rate |   test_exp_R |   test_PF |   sharpe |   max_dd_R |   wf_positive_frac |   neighbors_profitable |
|:-------------------------|:-------------|:--------------------------------------------------|:---------------------|----------------:|----------------:|-------------:|----------:|---------:|-----------:|-------------------:|-----------------------:|
| TREND_VWAP_001           | VALIDATED    | GRID_EDGE(optimum may lie outside searched range) | (24, 3.0, 2.0, None) |             129 |           45.74 |       0.2701 |     1.544 |     3.85 |      14.1  |               1    |                      1 |
| ORB_VOLUME_001           | REJECTED     |                                                   | nan                  |             nan |          nan    |     nan      |   nan     |   nan    |     nan    |             nan    |                    nan |
| TREND_PULLBACK_001       | EXPERIMENTAL | GRID_EDGE(optimum may lie outside searched range) | (24, 3.0, 2.0, None) |              67 |           52.24 |       0.2839 |     1.572 |     3.54 |       6.89 |               0.71 |                      1 |
| MEAN_REVERSION_001       | REJECTED     |                                                   | nan                  |             nan |          nan    |     nan      |   nan     |   nan    |     nan    |             nan    |                    nan |
| PATTERN_REVERSAL_001     | REJECTED     |                                                   | nan                  |             nan |          nan    |     nan      |   nan     |   nan    |     nan    |             nan    |                    nan |
| PATTERN_CONTINUATION_001 | REJECTED     |                                                   | nan                  |             nan |          nan    |     nan      |   nan     |   nan    |     nan    |             nan    |                    nan |
| CONTEXT_TREND_001        | VALIDATED    | GRID_EDGE(optimum may lie outside searched range) | (24, 3.0, 2.0, None) |             120 |           45.83 |       0.2727 |     1.542 |     3.73 |      14.86 |               1    |                      1 |

## Active setups on the last closed bar (research only; entry = NEXT bar open)
- none — NO TRADE

## Top research findings
- Best OOS setup: TREND_PULLBACK_001 (0.2839R/trade, n=67.0); worst: TREND_VWAP_001 (0.2701R/trade).
- Train/val -> OOS expectancy decay (overfitting evidence): TREND_VWAP_001: -0.00R, TREND_PULLBACK_001: +0.12R, CONTEXT_TREND_001: +0.05R
- Cost sensitivity (OOS exp R at cost x0/x1/x2/x3): TREND_VWAP_001: {"x0": 0.442, "x1": 0.27, "x2": 0.108, "x3": -0.056}; ORB_VOLUME_001: nan; TREND_PULLBACK_001: {"x0": 0.454, "x1": 0.284, "x2": 0.132, "x3": -0.02}; MEAN_REVERSION_001: nan; PATTERN_REVERSAL_001: nan; PATTERN_CONTINUATION_001: nan; CONTEXT_TREND_001: {"x0": 0.449, "x1": 0.273, "x2": 0.113, "x3": -0.035}
- Ablation (validation): components whose REMOVAL raised expectancy >0.02R (candidate redundancy): TREND_VWAP_001:adx_min(+0.02), TREND_VWAP_001:rsi_momo(+0.03), CONTEXT_TREND_001:ema_align(+0.02)
- ML meta-label (test AUC / filtered vs unfiltered expR): TREND_VWAP_001/logreg: 0.528 / 0.785 vs 0.27; TREND_VWAP_001/grad_boost: 0.59 / 0.475 vs 0.27; TREND_PULLBACK_001/logreg: 0.501 / nan vs 0.284; TREND_PULLBACK_001/grad_boost: 0.6 / 0.451 vs 0.284; CONTEXT_TREND_001/logreg: 0.547 / 0.353 vs 0.273; CONTEXT_TREND_001/grad_boost: 0.555 / 0.341 vs 0.273
- Strategies REJECTED with no train+validation-profitable exit config: ORB_VOLUME_001, MEAN_REVERSION_001, PATTERN_REVERSAL_001, PATTERN_CONTINUATION_001
- Features with sign-stable IC across train/val/test: rsi_14, ema9_21_spread_atr, ema21_50_spread_atr, vwap_dist_atr, bb_pctb, bb_width, dist_prev_high_atr, dist_prev_low_atr, dist_or_high_atr, htf_trend_15m, htf_trend_1h, day_range_pos, st_dir, psar_dir, ichi_pos, aroon_osc, stoch_k, stochrsi, willr_14, roc_10, cci_20, mfi_14, ppo, kc_pos, vol_pctile, atr_exp, avwap_dist_atr, rsi_div_bull, rsi_div_bear, obv_div_bull, obv_div_bear, ctx_index_trend, ctx_sector_trend, rel_strength_12
- Redundant feature pairs (|rho|>0.85): [('rsi_14', 'ema9_21_spread_atr', 0.94), ('rsi_14', 'vwap_dist_atr', 0.9), ('rsi_14', 'bb_pctb', 0.9), ('rsi_14', 'day_range_pos', 0.9), ('rsi_14', 'stoch_k', 0.87), ('rsi_14', 'willr_14', 0.87), ('rsi_14', 'cci_20', 0.89), ('rsi_14', 'ppo', 0.89), ('rsi_14', 'kc_pos', 0.98), ('ema9_21_spread_atr', 'vwap_dist_atr', 0.88), ('ema9_21_spread_atr', 'day_range_pos', 0.85), ('ema9_21_spread_atr', 'aroon_osc', 0.86), ('ema9_21_spread_atr', 'ppo', 0.97), ('ema9_21_spread_atr', 'kc_pos', 0.89), ('ema21_50_spread_atr', 'dist_prev_high_atr', 0.9), ('ema21_50_spread_atr', 'dist_prev_low_atr', 0.89), ('ema21_50_spread_atr', 'ichi_pos', 0.86), ('ema21_50_spread_atr', 'ppo', 0.88), ('ema21_50_spread_atr', 'avwap_dist_atr', 0.9), ('vwap_dist_atr', 'dist_or_high_atr', 0.92), ('vwap_dist_atr', 'day_range_pos', 0.95), ('vwap_dist_atr', 'kc_pos', 0.86), ('vwap_dist_atr', 'avwap_dist_atr', 0.87), ('bb_pctb', 'stoch_k', 0.95), ('bb_pctb', 'willr_14', 0.95), ('bb_pctb', 'roc_10', 0.88), ('bb_pctb', 'cci_20', 0.98), ('bb_pctb', 'kc_pos', 0.95), ('dist_prev_low_atr', 'avwap_dist_atr', 0.89), ('dist_or_high_atr', 'day_range_pos', 0.9), ('dist_or_high_atr', 'avwap_dist_atr', 0.87), ('day_range_pos', 'kc_pos', 0.88), ('ichi_pos', 'ppo', 0.85), ('stoch_k', 'willr_14', 1.0), ('stoch_k', 'roc_10', 0.88), ('stoch_k', 'cci_20', 0.92), ('stoch_k', 'kc_pos', 0.92), ('willr_14', 'roc_10', 0.88), ('willr_14', 'cci_20', 0.92), ('willr_14', 'kc_pos', 0.92), ('roc_10', 'cci_20', 0.87), ('roc_10', 'kc_pos', 0.88), ('cci_20', 'kc_pos', 0.93), ('ppo', 'avwap_dist_atr', 0.86)]
- VALIDATED strategies: 2. None is deployed; paper/live use requires explicit human approval (registry blocks it otherwise).

_Multiple-testing note: each strategy tried 270 exit configs; validation-selected results are optimistic, only the OOS test is unbiased._