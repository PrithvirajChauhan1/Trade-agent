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

## Strategies

### TREND_VWAP_001  —  status: **VALIDATED**
- **Objective / hypothesis:** Trend continuation: EMA stack + price on the right side of session VWAP + ADX + volume + RSI momentum + 15m trend agreement.
- **Created:** 2026-09-29T19:06:24
- **Entry conditions:** `{'ema_align': True, 'vwap_side': True, 'adx_min': 20, 'rel_vol_min': 1.0, 'rsi_momo': {'long_min': 55, 'short_max': 45}, 'htf_15m': True}` (direction: both; tod: ['mid_morning', 'lunch', 'afternoon']; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** (24, 3.0, 2.0, None)
- **Train / Val / OOS expectancy (R):** 0.294 / 0.245 / 0.2701
- **OOS trades / win rate / PF:** 129 / 45.74 / 1.544
- **Walk-forward positive folds:** 1.0
- **Parameter sensitivity (neighbors profitable):** 1.0
- **Flags:** GRID_EDGE(optimum may lie outside searched range)
- **Failure modes / limitations:** OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable; GRID_EDGE(optimum may lie outside searched range)
- **Status history:** [('VALIDATED', 'OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable')]

### ORB_VOLUME_001  —  status: **REJECTED**
- **Objective / hypothesis:** Opening-range breakout with relative-volume spike, EMA alignment and 15m trend agreement.
- **Created:** 2026-09-29T19:06:45
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
- **Created:** 2026-09-29T19:06:46
- **Entry conditions:** `{'ema_align': True, 'pullback': 0.3, 'vwap_side': True, 'htf_1h': True, 'adx_min': 18}` (direction: both; tod: None; regime: None)
- **Entry timing:** signal on bar T close, fill at open of T+1
- **Stop/target/exit/hold:** selected on validation, see latest run
- **Training / validation / OOS:** train 2023-01-02..2023-06-30 | validation 2023-07-03..2023-09-29 | OOS test 2023-10-02..2023-12-29  (260 sessions)
- **Features:** ema_9/21/50, rsi_14, adx_14, session_vwap, rel_volume_tod, or_high/low, htf_trend_15m/1h, bb, atr_14
- **Selected config (hold, R:R, stop ATR, trail):** (24, 3.0, 2.0, None)
- **Train / Val / OOS expectancy (R):** 0.285 / 0.532 / 0.2839
- **OOS trades / win rate / PF:** 67 / 52.24 / 1.572
- **Walk-forward positive folds:** 0.71
- **Parameter sensitivity (neighbors profitable):** 1.0
- **Flags:** GRID_EDGE(optimum may lie outside searched range)
- **Failure modes / limitations:** OOS expectancy positive but CI/walk-forward/sensitivity not conclusive; GRID_EDGE(optimum may lie outside searched range)
- **Status history:** []

### MEAN_REVERSION_001  —  status: **REJECTED**
- **Objective / hypothesis:** Fade Bollinger extremes stretched from VWAP in low-ADX (range) conditions; expects reversion, fails in trend days.
- **Created:** 2026-09-29T19:06:53
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
