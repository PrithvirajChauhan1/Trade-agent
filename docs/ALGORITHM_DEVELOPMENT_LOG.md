# Algorithm Development Log

_Append-only. Failed experiments are kept on purpose._

## 2026-09-30 02:04 — TREND_VWAP_001
- **Why / hypothesis:** Trend continuation: EMA stack + price on the right side of session VWAP + ADX + volume + RSI momentum + 15m trend agreement.
- **Combined components:** ['ema_align', 'vwap_side', 'adx_min', 'rel_vol_min', 'rsi_momo', 'htf_15m']
- **Tested:** 880 exit configs (hold x R:R x stop x trail); selected (24, 3.0, 2.0, None) on validation among 75 eligible
- **Result:** train 0.294R / val 0.245R / OOS 0.2701R over 129 trades; CI [0.021,0.531]; WF+ 1.0; neighbours+ 1.0
- **Decision:** VALIDATED (UNSTABLE): OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable

## 2026-09-30 02:04 — ORB_VOLUME_001
- **Why / hypothesis:** Opening-range breakout with relative-volume spike, EMA alignment and 15m trend agreement.
- **Combined components:** ['orb', 'rel_vol_min', 'ema_align', 'htf_15m']
- **Tested:** 880 exit configs on train/val
- **Result:** no configuration profitable (net of costs) in BOTH train and validation
- **Decision:** REJECTED

## 2026-09-30 02:04 — TREND_PULLBACK_001
- **Why / hypothesis:** Buy/sell EMA21 pullbacks with rejection candle inside an aligned trend, VWAP side and 1h trend agreement.
- **Combined components:** ['ema_align', 'pullback', 'vwap_side', 'htf_1h', 'adx_min']
- **Tested:** 880 exit configs (hold x R:R x stop x trail); selected (24, 3.0, 2.0, None) on validation among 76 eligible
- **Result:** train 0.285R / val 0.532R / OOS 0.2839R over 67 trades; CI [-0.061,0.631]; WF+ 0.71; neighbours+ 1.0
- **Decision:** EXPERIMENTAL (UNSTABLE): OOS expectancy positive but CI/walk-forward/sensitivity not conclusive

## 2026-09-30 02:04 — MEAN_REVERSION_001
- **Why / hypothesis:** Fade Bollinger extremes stretched from VWAP in low-ADX (range) conditions; expects reversion, fails in trend days.
- **Combined components:** ['bb_extreme', 'vwap_stretch', 'adx_max', 'gap_ok']
- **Tested:** 880 exit configs on train/val
- **Result:** no configuration profitable (net of costs) in BOTH train and validation
- **Decision:** REJECTED

## 2026-09-30 02:04 — PATTERN_REVERSAL_001
- **Why / hypothesis:** Confirmed double top/bottom, head-and-shoulders (and inverse) and failed opening-range breaks, with normal-or-better participation.
- **Combined components:** ['pat_reversal', 'rel_vol_min']
- **Tested:** 880 exit configs on train/val
- **Result:** no configuration profitable (net of costs) in BOTH train and validation
- **Decision:** REJECTED

## 2026-09-30 02:04 — PATTERN_CONTINUATION_001
- **Why / hypothesis:** Rectangle/triangle/flag/pennant breakouts in the direction of EMA alignment and 15m trend, with volume expansion.
- **Combined components:** ['pat_continuation', 'ema_align', 'htf_15m', 'rel_vol_min']
- **Tested:** 880 exit configs on train/val
- **Result:** no configuration profitable (net of costs) in BOTH train and validation
- **Decision:** REJECTED

## 2026-09-30 02:04 — CONTEXT_TREND_001
- **Why / hypothesis:** TREND_VWAP entries additionally required to agree with index and sector trend and to occur when the volatility index is not elevated. Tests whether market context adds value.
- **Combined components:** ['ema_align', 'vwap_side', 'adx_min', 'rel_vol_min', 'htf_15m', 'mkt_ctx']
- **Tested:** 880 exit configs (hold x R:R x stop x trail); selected (24, 3.0, 2.0, None) on validation among 76 eligible
- **Result:** train 0.332R / val 0.316R / OOS 0.2727R over 120 trades; CI [0.017,0.534]; WF+ 1.0; neighbours+ 1.0
- **Decision:** VALIDATED (UNSTABLE): OOS net expectancy > 0 with bootstrap CI above 0; walk-forward and parameter neighbourhood stable

## 2026-10-01 17:54 — TREND_VWAP_001
- **Why / hypothesis:** Trend continuation: EMA stack + price on the right side of session VWAP + ADX + volume + RSI momentum + 15m trend agreement.
- **Combined components:** ['ema_align', 'vwap_side', 'adx_min', 'rel_vol_min', 'rsi_momo', 'htf_15m']
- **Tested:** 880 exit configs (hold x R:R x stop x trail); selected (36, 4.0, 2.5, None) on validation among 263 eligible
- **Result:** train 0.637R / val 0.543R / OOS -0.2618R over 28 trades; CI [-0.691,0.209]; WF+ 1.0; neighbours+ 1.0
- **Decision:** EXPERIMENTAL (UNSTABLE): only 28 OOS trades (<30): inconclusive

## 2026-10-01 17:54 — ORB_VOLUME_001
- **Why / hypothesis:** Opening-range breakout with relative-volume spike, EMA alignment and 15m trend agreement.
- **Combined components:** ['orb', 'rel_vol_min', 'ema_align', 'htf_15m']
- **Tested:** 880 exit configs on train/val
- **Result:** no configuration profitable (net of costs) in BOTH train and validation
- **Decision:** REJECTED

## 2026-10-01 17:54 — TREND_PULLBACK_001
- **Why / hypothesis:** Buy/sell EMA21 pullbacks with rejection candle inside an aligned trend, VWAP side and 1h trend agreement.
- **Combined components:** ['ema_align', 'pullback', 'vwap_side', 'htf_1h', 'adx_min']
- **Tested:** 880 exit configs (hold x R:R x stop x trail); selected (36, 4.0, 1.5, None) on validation among 488 eligible
- **Result:** train 0.439R / val 1.016R / OOS -0.7546R over 13 trades; CI [-1.164,-0.076]; WF+ 1.0; neighbours+ 1.0
- **Decision:** EXPERIMENTAL (UNSTABLE): only 13 OOS trades (<30): inconclusive

## 2026-10-01 17:54 — MEAN_REVERSION_001
- **Why / hypothesis:** Fade Bollinger extremes stretched from VWAP in low-ADX (range) conditions; expects reversion, fails in trend days.
- **Combined components:** ['bb_extreme', 'vwap_stretch', 'adx_max', 'gap_ok']
- **Tested:** 880 exit configs on train/val
- **Result:** no configuration profitable (net of costs) in BOTH train and validation
- **Decision:** REJECTED

## 2026-10-01 17:54 — PATTERN_REVERSAL_001
- **Why / hypothesis:** Confirmed double top/bottom, head-and-shoulders (and inverse) and failed opening-range breaks, with normal-or-better participation.
- **Combined components:** ['pat_reversal', 'rel_vol_min']
- **Tested:** 880 exit configs on train/val
- **Result:** no configuration profitable (net of costs) in BOTH train and validation
- **Decision:** REJECTED

## 2026-10-01 17:54 — PATTERN_CONTINUATION_001
- **Why / hypothesis:** Rectangle/triangle/flag/pennant breakouts in the direction of EMA alignment and 15m trend, with volume expansion.
- **Combined components:** ['pat_continuation', 'ema_align', 'htf_15m', 'rel_vol_min']
- **Tested:** 880 exit configs on train/val
- **Result:** no configuration profitable (net of costs) in BOTH train and validation
- **Decision:** REJECTED
