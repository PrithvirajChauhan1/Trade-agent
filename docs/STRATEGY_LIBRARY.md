# Strategy Library

_Auto-generated from `strategy_store/registry.json`. All versions are kept; nothing is overwritten._

## Indicators

### EMA (9/21/50/200)
- **Purpose:** Trend direction/alignment
- **Formula:** EMA_t = a*P_t + (1-a)*EMA_{t-1}, a=2/(n+1)
- **Parameters:** span
- **Known weaknesses:** Lags; whipsaws in ranges
- **Failure modes:** Sideways chop; overnight gaps distort continuity

### RSI(14)
- **Purpose:** Momentum / stretch
- **Formula:** 100 - 100/(1+avgGain/avgLoss) (Wilder smoothing)
- **Parameters:** period
- **Known weaknesses:** Stays extreme in strong trends
- **Failure modes:** Trend days: 'oversold' keeps falling

### MACD(12,26,9) histogram
- **Purpose:** Momentum change
- **Formula:** EMA12-EMA26 minus its 9-EMA
- **Parameters:** fast, slow, signal
- **Known weaknesses:** Lagging, noisy on 5m
- **Failure modes:** Range-bound sessions

### ATR(14)
- **Purpose:** Volatility scale for stops/targets
- **Formula:** Wilder EMA of true range
- **Parameters:** period
- **Known weaknesses:** Backward-looking; jumps at open
- **Failure modes:** Volatility regime shifts intraday

### ADX(14)
- **Purpose:** Trend strength (not direction)
- **Formula:** Wilder-smoothed |+DI - -DI|/(+DI + -DI)
- **Parameters:** period
- **Known weaknesses:** Slow to react
- **Failure modes:** Trend reversals after a strong ADX reading

### Bollinger(20,2) / %B / width
- **Purpose:** Stretch and compression
- **Formula:** SMA20 +/- 2 sigma
- **Parameters:** period, k
- **Known weaknesses:** Bands widen after the move
- **Failure modes:** Band-walks in trends break mean reversion

### Session VWAP
- **Purpose:** Intraday fair-value anchor
- **Formula:** cum(TP*V)/cum(V) reset each session
- **Parameters:** session
- **Known weaknesses:** Meaningless with zero volume (indexes)
- **Failure modes:** Very early bars; low-volume names

### Relative volume (time-of-day)
- **Purpose:** Participation vs normal for that bar
- **Formula:** V_t / mean(V at same bar-of-day, prior 20 sessions)
- **Parameters:** 20 sessions
- **Known weaknesses:** Needs history; news days distort
- **Failure modes:** Volume-less instruments

### Opening range (first 6 bars)
- **Purpose:** Early-session balance breakout level
- **Formula:** max/min of bars 0..5, valid from bar 6
- **Parameters:** or_bars
- **Known weaknesses:** Few breakouts per day
- **Failure modes:** Gap days, low-volume opens

### HTF trend (15m/1h)
- **Purpose:** Higher-timeframe agreement
- **Formula:** sign(EMA8-EMA21) of COMPLETED HTF closes
- **Parameters:** k bars
- **Known weaknesses:** Stale by up to one HTF bar
- **Failure modes:** Fast reversals

### Candle patterns
- **Purpose:** Bar-level structure
- **Formula:** see patterns/candles.py
- **Parameters:** -
- **Known weaknesses:** Weak alone; measured, not assumed
- **Failure modes:** Most are cost-negative alone

### WMA / HMA
- **Purpose:** Lower-lag moving averages
- **Formula:** WMA linear weights; HMA = WMA(2*WMA(n/2)-WMA(n), sqrt(n))
- **Parameters:** n
- **Known weaknesses:** More noise than EMA
- **Failure modes:** Chop

### Aroon / Supertrend / Parabolic SAR
- **Purpose:** Trend state and trailing levels
- **Formula:** Aroon: bars since extreme; Supertrend: ATR bands flip; SAR: accelerating stop
- **Parameters:** n, mult, af
- **Known weaknesses:** Whipsaw in ranges; SAR path-dependent
- **Failure modes:** Sideways sessions

### Ichimoku (tenkan/kijun/cloud)
- **Purpose:** Trend + support/resistance zone
- **Formula:** midpoints of 9/26/52-bar ranges; cloud shifted from the PAST
- **Parameters:** 9/26/52
- **Known weaknesses:** Designed for daily bars; Chikou span omitted (uses future bars)
- **Failure modes:** 5m noise

### Stoch / StochRSI / Williams %R / ROC / CCI / MFI / PPO
- **Purpose:** Momentum family
- **Formula:** standard definitions (see features/extra_indicators.py)
- **Parameters:** n
- **Known weaknesses:** Highly correlated with RSI; redundancy is measured
- **Failure modes:** Trend days

### Keltner, hist. vol, vol percentile, range/ATR expansion
- **Purpose:** Volatility state
- **Formula:** EMA20 +/- 2 ATR; annualised stdev of log returns; ratios to rolling means
- **Parameters:** n
- **Known weaknesses:** Backward-looking
- **Failure modes:** Regime breaks

### RSI/OBV divergence, anchored VWAP (prev-session low)
- **Purpose:** Divergence and anchored fair value
- **Formula:** price extreme without confirming RSI/OBV; VWAP from previous session low bar
- **Parameters:** 20 bars
- **Known weaknesses:** Divergence is frequent and weak alone
- **Failure modes:** Strong trends

### Chart patterns (double top/bottom, H&S, failed break, rectangle, triangle, flag, pennant)
- **Purpose:** Structure events
- **Formula:** pivots confirmed after k=2 bars; fire on neckline/boundary break; per session
- **Parameters:** k, tolerance in ATR
- **Known weaknesses:** Rare; rule-based approximations
- **Failure modes:** Choppy pivots

### Market context (index/tech/sector trend, vol index)
- **Purpose:** Regime and relative strength
- **Formula:** EMA9/21 sign of supplied context series; vol index = previous session close
- **Parameters:** -
- **Known weaknesses:** Requires user-supplied data
- **Failure modes:** Data misalignment

## Strategies

### TREND_VWAP_001  —  status: **EXPERIMENTAL**
- **Objective / hypothesis:** Trend continuation: EMA stack + price on the right side of session VWAP + ADX + volume + RSI momentum + 15m trend agreement.
- **Created:** 2026-09-30T02:03:10
- **Entry conditions:** `{'ema_align': True, 'vwap_side': True, 'adx_min': 20, 'rel_vol_min': 1.0, 'rsi_momo': {'long_min': 55, 'short_max': 45}, 'htf_15m': True}` (direction: both; tod: ['mid_morning', 'lunch', 'afternoon']; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** (36, 4.0, 2.5, None)
- **Train / Val / OOS expectancy (R):** 0.637 / 0.543 / -0.2618
- **OOS trades / win rate / PF:** 28 / 28.57 / 0.622
- **Walk-forward positive folds:** 1.0
- **Parameter sensitivity (neighbors profitable):** 1.0
- **Flags:** GRID_EDGE(optimum may lie outside searched range)
- **Failure modes / limitations:** only 28 OOS trades (<30): inconclusive; GRID_EDGE(optimum may lie outside searched range)
- **Status history:** [('VALIDATED', 'OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable'), ('EXPERIMENTAL', 'Historical result was generated on synthetic 5m data; real-market validation has not been demonstrated.')]

### ORB_VOLUME_001  —  status: **REJECTED**
- **Objective / hypothesis:** Opening-range breakout with relative-volume spike, EMA alignment and 15m trend agreement.
- **Created:** 2026-09-30T02:03:31
- **Entry conditions:** `{'orb': True, 'rel_vol_min': 1.5, 'ema_align': True, 'htf_15m': True}` (direction: both; tod: None; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** None
- **Train / Val / OOS expectancy (R):** None / None / None
- **OOS trades / win rate / PF:** None / None / None
- **Walk-forward positive folds:** None
- **Parameter sensitivity (neighbors profitable):** None
- **Flags:** none
- **Failure modes / limitations:** no configuration profitable (net of costs) in BOTH train and validation
- **Status history:** [('REJECTED', 'no configuration profitable (net of costs) in BOTH train and validation')]

### TREND_PULLBACK_001  —  status: **EXPERIMENTAL**
- **Objective / hypothesis:** Buy/sell EMA21 pullbacks with rejection candle inside an aligned trend, VWAP side and 1h trend agreement.
- **Created:** 2026-09-30T02:03:32
- **Entry conditions:** `{'ema_align': True, 'pullback': 0.3, 'vwap_side': True, 'htf_1h': True, 'adx_min': 18}` (direction: both; tod: None; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** (36, 4.0, 1.5, None)
- **Train / Val / OOS expectancy (R):** 0.439 / 1.016 / -0.7546
- **OOS trades / win rate / PF:** 13 / 15.38 / 0.228
- **Walk-forward positive folds:** 1.0
- **Parameter sensitivity (neighbors profitable):** 1.0
- **Flags:** GRID_EDGE(optimum may lie outside searched range)
- **Failure modes / limitations:** only 13 OOS trades (<30): inconclusive; GRID_EDGE(optimum may lie outside searched range)
- **Status history:** []

### MEAN_REVERSION_001  —  status: **REJECTED**
- **Objective / hypothesis:** Fade Bollinger extremes stretched from VWAP in low-ADX (range) conditions; expects reversion, fails in trend days.
- **Created:** 2026-09-30T02:03:40
- **Entry conditions:** `{'bb_extreme': {'rsi_lo': 30, 'rsi_hi': 70}, 'vwap_stretch': 1.5, 'adx_max': 22, 'gap_ok': 0.006}` (direction: both; tod: None; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** None
- **Train / Val / OOS expectancy (R):** None / None / None
- **OOS trades / win rate / PF:** None / None / None
- **Walk-forward positive folds:** None
- **Parameter sensitivity (neighbors profitable):** None
- **Flags:** none
- **Failure modes / limitations:** no configuration profitable (net of costs) in BOTH train and validation
- **Status history:** [('REJECTED', 'no configuration profitable (net of costs) in BOTH train and validation')]

### PATTERN_REVERSAL_001  —  status: **REJECTED**
- **Objective / hypothesis:** Confirmed double top/bottom, head-and-shoulders (and inverse) and failed opening-range breaks, with normal-or-better participation.
- **Created:** 2026-09-30T02:03:41
- **Entry conditions:** `{'pat_reversal': True, 'rel_vol_min': 1.0}` (direction: both; tod: ['mid_morning', 'lunch', 'afternoon']; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** None
- **Train / Val / OOS expectancy (R):** None / None / None
- **OOS trades / win rate / PF:** None / None / None
- **Walk-forward positive folds:** None
- **Parameter sensitivity (neighbors profitable):** None
- **Flags:** none
- **Failure modes / limitations:** no configuration profitable (net of costs) in BOTH train and validation
- **Status history:** [('REJECTED', 'no configuration profitable (net of costs) in BOTH train and validation')]

### PATTERN_CONTINUATION_001  —  status: **REJECTED**
- **Objective / hypothesis:** Rectangle/triangle/flag/pennant breakouts in the direction of EMA alignment and 15m trend, with volume expansion.
- **Created:** 2026-09-30T02:03:46
- **Entry conditions:** `{'pat_continuation': True, 'ema_align': True, 'htf_15m': True, 'rel_vol_min': 1.2}` (direction: both; tod: None; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** None
- **Train / Val / OOS expectancy (R):** None / None / None
- **OOS trades / win rate / PF:** None / None / None
- **Walk-forward positive folds:** None
- **Parameter sensitivity (neighbors profitable):** None
- **Flags:** none
- **Failure modes / limitations:** no configuration profitable (net of costs) in BOTH train and validation
- **Status history:** [('REJECTED', 'no configuration profitable (net of costs) in BOTH train and validation')]

### CONTEXT_TREND_001  —  status: **EXPERIMENTAL**
- **Objective / hypothesis:** TREND_VWAP entries additionally required to agree with index and sector trend and to occur when the volatility index is not elevated. Tests whether market context adds value.
- **Created:** 2026-09-30T02:03:47
- **Entry conditions:** `{'ema_align': True, 'vwap_side': True, 'adx_min': 20, 'rel_vol_min': 1.0, 'htf_15m': True, 'mkt_ctx': {'index': True, 'sector': True, 'vol_max': 30}}` (direction: both; tod: ['mid_morning', 'lunch', 'afternoon']; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** [24, 3.0, 2.0, None]
- **Train / Val / OOS expectancy (R):** 0.332 / 0.316 / 0.2727
- **OOS trades / win rate / PF:** 120 / 45.83 / 1.542
- **Walk-forward positive folds:** 1.0
- **Parameter sensitivity (neighbors profitable):** 1.0
- **Flags:** GRID_EDGE(optimum may lie outside searched range)
- **Failure modes / limitations:** OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable; GRID_EDGE(optimum may lie outside searched range)
- **Status history:** [('VALIDATED', 'OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable'), ('EXPERIMENTAL', 'Historical result was generated on synthetic 5m data; real-market validation has not been demonstrated.')]
