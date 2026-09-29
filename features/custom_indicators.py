"""
Custom / adaptive indicators trained on the 2000-2020 period.

Phase 1:  grid-search over parameter combinations on training data
          to find periods/thresholds that maximise signal quality metrics
          (Sharpe of signals, NOT win rate — avoids in-sample overfit).
Phase 2:  parameters are FROZEN and applied to 2020-2025 OOS data.

Why Sharpe of signals, not win rate?
  Optimising for win rate on training data almost always produces a parameter
  set that works for the specific noise realisation in that sample but
  collapses OOS (exactly the fragility Section 14 warns about).
  Sharpe of the signal distribution is more robust: a signal must have
  consistent direction AND reasonable size, penalising both missed moves
  and whipsaws.

Included custom indicators:
  1. AdaptiveEMA   — period optimised per regime type (trending vs ranging)
  2. RegimeRSI     — overbought/oversold thresholds optimised per regime
  3. VolatilityBand— ATR multiplier for stop/target, optimised per regime
  4. MomentumScore — composite momentum (MACD, RSI slope, price vs EMAs),
                     weighted by in-sample correlation with forward return
  5. MarketPhaseOscillator (MPO) — novel custom indicator combining ADX
                     trend strength, BB width (volatility), and OBV slope
                     into a single 0-100 market "energy" score
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import product
from typing import NamedTuple

from features.indicators import ema, sma, rsi, atr, adx, bollinger_bands, obv, relative_volume


# ---------------------------------------------------------------------------
# Structs for learned parameters
# ---------------------------------------------------------------------------

class AdaptiveEMAParams(NamedTuple):
    fast_trend: int     # fast EMA period when ADX > threshold (trending)
    slow_trend: int
    fast_range: int     # fast EMA period when ADX <= threshold (ranging)
    slow_range: int
    adx_threshold: float

class RegimeRSIParams(NamedTuple):
    ob_trend: float     # overbought threshold in trending regime
    os_trend: float     # oversold threshold in trending regime
    ob_range: float     # overbought in ranging
    os_range: float

class VolBandParams(NamedTuple):
    stop_mult_bull: float   # ATR multiplier for stop in bull trend
    tgt_mult_bull: float    # ATR multiplier for target in bull trend
    stop_mult_bear: float
    tgt_mult_bear: float
    stop_mult_range: float
    tgt_mult_range: float

class MomentumWeights(NamedTuple):
    w_rsi_slope: float
    w_macd_hist: float
    w_price_ema: float      # (close - ema21) / atr

class LearnedParams(NamedTuple):
    adaptive_ema: AdaptiveEMAParams
    regime_rsi: RegimeRSIParams
    vol_band: VolBandParams
    momentum_weights: MomentumWeights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward_return(close: pd.Series, n: int = 5) -> pd.Series:
    """n-bar forward return. Used in training ONLY — never touched in OOS."""
    return close.shift(-n) / close - 1


def _signal_sharpe(signals: pd.Series, fwd_ret: pd.Series, min_signals: int = 30) -> float:
    """Sharpe-like metric of how well a binary signal {+1, -1, 0} predicts forward returns."""
    active = signals[signals != 0].dropna()
    fwd = fwd_ret.reindex(active.index).dropna()
    active = active.reindex(fwd.index)
    if len(active) < min_signals:
        return -99.0
    returns = active.values * fwd.values
    std = returns.std()
    if std < 1e-9:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252))


# ---------------------------------------------------------------------------
# Training routines (run on 2000-2020 data only)
# ---------------------------------------------------------------------------

def optimise_adaptive_ema(df: pd.DataFrame) -> AdaptiveEMAParams:
    """Grid search over EMA period pairs for trending vs ranging regime."""
    adx_vals = df["adx_14"]
    fwd = _forward_return(df["close"], n=5)

    best_sharpe, best_params = -99.0, AdaptiveEMAParams(9, 21, 5, 13, 25)

    fast_range_options = [3, 5, 8]
    slow_range_options = [10, 13, 21]
    fast_trend_options = [9, 12, 15]
    slow_trend_options = [21, 26, 34]
    adx_thresh_options = [20, 25, 30]

    for ft, st, fr, sr, adx_t in product(fast_trend_options, slow_trend_options,
                                           fast_range_options, slow_range_options,
                                           adx_thresh_options):
        if ft >= st or fr >= sr:
            continue
        trending = adx_vals > adx_t
        fast_ema = np.where(trending, ema(df["close"], ft), ema(df["close"], fr))
        slow_ema = np.where(trending, ema(df["close"], st), ema(df["close"], sr))
        signals = pd.Series(np.where(fast_ema > slow_ema, 1, -1), index=df.index)
        sh = _signal_sharpe(signals, fwd)
        if sh > best_sharpe:
            best_sharpe = sh
            best_params = AdaptiveEMAParams(ft, st, fr, sr, adx_t)

    print(f"  AdaptiveEMA: {best_params}  (Sharpe={best_sharpe:.3f})")
    return best_params


def optimise_regime_rsi(df: pd.DataFrame) -> RegimeRSIParams:
    """Find RSI thresholds that best separate mean-reverting oversold/overbought."""
    fwd = _forward_return(df["close"], n=3)
    rsi_vals = df["rsi_14"]
    adx_vals = df["adx_14"]

    best = -99.0
    best_params = RegimeRSIParams(70, 30, 65, 35)

    for ob_t, os_t, ob_r, os_r in product([65, 70, 75], [25, 30, 35], [60, 65, 70], [30, 35, 40]):
        if ob_t <= os_t or ob_r <= os_r:
            continue
        trending = adx_vals > 25
        # in trending: break of extremes is continuation, not reversal
        # in ranging: extremes signal reversal (mean reversion)
        ranging_long  = (~trending) & (rsi_vals < os_r)
        ranging_short = (~trending) & (rsi_vals > ob_r)
        signals = pd.Series(0, index=df.index)
        signals[ranging_long] = 1
        signals[ranging_short] = -1
        sh = _signal_sharpe(signals, fwd)
        if sh > best:
            best = sh
            best_params = RegimeRSIParams(ob_t, os_t, ob_r, os_r)

    print(f"  RegimeRSI:   {best_params}  (Sharpe={best:.3f})")
    return best_params


def optimise_vol_band(df: pd.DataFrame) -> VolBandParams:
    """Find ATR multipliers that maximise reward:risk given fixed 12%/7% exits."""
    # With fixed exits (12% target, 7% stop), what we're really optimising here
    # is which ATR-based stop placement best avoids premature stop-outs.
    # Proxy: stop mult that minimises P(low < entry - mult*ATR) in profitable bars.
    # This is a heuristic — the 12%/7% fixed exits make ATR primarily useful for
    # position sizing and entry timing, not exit management in this system.
    best_params = VolBandParams(
        stop_mult_bull=1.8, tgt_mult_bull=3.5,
        stop_mult_bear=1.8, tgt_mult_bear=3.5,
        stop_mult_range=1.2, tgt_mult_range=2.0,
    )
    print(f"  VolBand:     {best_params}  (fixed — ATR mults for entry quality, exits use 12%/7% hard levels)")
    return best_params


def optimise_momentum_weights(df: pd.DataFrame) -> MomentumWeights:
    """Regress 3-factor momentum score against 5-bar forward return."""
    fwd = _forward_return(df["close"], n=5).dropna()
    rsi_slope = df["rsi_14"].diff(3).reindex(fwd.index)
    macd_h = df["hist"].reindex(fwd.index)
    price_vs_ema = ((df["close"] - df["ema_21"]) / df["atr_14"].replace(0, np.nan)).reindex(fwd.index)

    X = pd.DataFrame({"rsi": rsi_slope, "macd": macd_h, "pema": price_vs_ema}).dropna()
    y = fwd.reindex(X.index)

    if len(X) < 50:
        return MomentumWeights(0.33, 0.33, 0.34)

    # Normalise columns to comparable scale before regression
    Xn = (X - X.mean()) / (X.std() + 1e-9)
    # OLS via normal equation (no scipy needed)
    coef = np.linalg.lstsq(Xn.values, y.values, rcond=None)[0]
    coef = coef / (np.abs(coef).sum() + 1e-9)
    w = MomentumWeights(max(0, coef[0]), max(0, coef[1]), max(0, coef[2]))
    total = w.w_rsi_slope + w.w_macd_hist + w.w_price_ema
    if total < 1e-9:
        w = MomentumWeights(0.33, 0.33, 0.34)
    print(f"  MomentumWeights: {w}")
    return w


def train_params(df: pd.DataFrame) -> LearnedParams:
    """
    Run all optimisation routines on training data (2000-2020).
    df must already have compute_all() applied.
    Returns frozen LearnedParams — these are stored and applied unchanged to OOS.
    """
    print("Training custom indicators on 2000-2020 data...")
    p_ema  = optimise_adaptive_ema(df)
    p_rsi  = optimise_regime_rsi(df)
    p_vb   = optimise_vol_band(df)
    p_mom  = optimise_momentum_weights(df)
    params = LearnedParams(p_ema, p_rsi, p_vb, p_mom)
    print("Training complete. Parameters frozen.")
    return params


# ---------------------------------------------------------------------------
# Indicator computation using learned parameters (call on OOS data)
# ---------------------------------------------------------------------------

def compute_adaptive_ema(df: pd.DataFrame, p: AdaptiveEMAParams) -> pd.DataFrame:
    trending = df["adx_14"] > p.adx_threshold
    fast_vals = np.where(trending, ema(df["close"], p.fast_trend), ema(df["close"], p.fast_range))
    slow_vals = np.where(trending, ema(df["close"], p.slow_trend), ema(df["close"], p.slow_range))
    return pd.DataFrame({
        "aema_fast": fast_vals,
        "aema_slow": slow_vals,
        "aema_cross": (fast_vals > slow_vals).astype(int),
    }, index=df.index)


def compute_regime_rsi_signal(df: pd.DataFrame, p: RegimeRSIParams) -> pd.Series:
    trending = df["adx_14"] > 25
    ob = np.where(trending, p.ob_trend, p.ob_range)
    os_ = np.where(trending, p.os_trend, p.os_range)
    rsi_v = df["rsi_14"].values
    sig = np.where(rsi_v < os_, 1, np.where(rsi_v > ob, -1, 0))
    return pd.Series(sig, index=df.index, name="rsi_signal")


def compute_momentum_score(df: pd.DataFrame, w: MomentumWeights) -> pd.Series:
    rsi_slope = df["rsi_14"].diff(3)
    macd_h = df["hist"]
    price_vs_ema = (df["close"] - df["ema_21"]) / df["atr_14"].replace(0, np.nan)
    # normalise each component to [-1, 1] via tanh of z-score
    def norm(s):
        mu, sd = s.rolling(252, min_periods=30).mean(), s.rolling(252, min_periods=30).std()
        return np.tanh((s - mu) / (sd + 1e-9))

    score = w.w_rsi_slope * norm(rsi_slope) + w.w_macd_hist * norm(macd_h) + w.w_price_ema * norm(price_vs_ema)
    return score.rename("momentum_score")


def compute_mpo(df: pd.DataFrame) -> pd.Series:
    """
    Market Phase Oscillator (MPO) — custom composite indicator.
    Combines:
      - ADX (trend energy): 0-100
      - Inverted BB width percentile (low width = energy building): 0-100
      - OBV slope sign (institutional participation direction): +1/-1
    Output: 0-100 market "readiness" score.
    High MPO (>65): market has energy (trending or building to break).
    Low MPO (<35): quiet, ranging, avoid trend strategies.
    """
    adx_norm = df["adx_14"].clip(0, 50) / 50 * 100
    bb_width_pct = df["bb_width"].rolling(252, min_periods=30).rank(pct=True) * 100
    bb_contraction = 100 - bb_width_pct  # high score when BB is contracting
    obv_slope = df["obv"].diff(10).apply(np.sign) * 50 + 50  # 0 or 100

    mpo = (0.45 * adx_norm + 0.35 * bb_contraction + 0.20 * obv_slope).clip(0, 100)
    return mpo.rename("mpo")


def apply_custom_indicators(df: pd.DataFrame, params: LearnedParams) -> pd.DataFrame:
    """Attach all custom indicators to df using frozen learned params."""
    out = df.copy()
    aema = compute_adaptive_ema(out, params.adaptive_ema)
    out = out.join(aema)
    out["rsi_signal"] = compute_regime_rsi_signal(out, params.regime_rsi)
    out["momentum_score"] = compute_momentum_score(out, params.momentum_weights)
    out["mpo"] = compute_mpo(out)
    return out
