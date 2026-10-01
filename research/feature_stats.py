"""Does a feature carry information? Spearman IC vs forward 6-bar return per split + redundancy pairs."""
import numpy as np, pandas as pd
from patterns.candles import forward_arrays

def feature_table(df, ranges, H=6):
    ret, _, _ = forward_arrays(df, H)
    X = pd.DataFrame({"rsi_14": df.rsi_14, "adx_14": df.adx_14, "macd_hist": df["hist"] / df.atr_14,
                      "ema9_21_spread_atr": (df.ema_9 - df.ema_21) / df.atr_14, "ema21_50_spread_atr": (df.ema_21 - df.ema_50) / df.atr_14,
                      "vwap_dist_atr": df.vwap_dist_atr, "rel_volume_tod": df.rel_volume_tod, "bb_pctb": df.bb_pctb, "bb_width": df.bb_width,
                      "gap_pct": df.gap_pct, "dist_prev_high_atr": (df.close - df.prev_high) / df.atr_14, "dist_prev_low_atr": (df.close - df.prev_low) / df.atr_14,
                      "dist_or_high_atr": (df.close - df.or_high) / df.atr_14, "htf_trend_15m": df.htf_trend_15m, "htf_trend_1h": df.htf_trend_1h,
                      "day_range_pos": (df.close - df.day_low) / (df.day_high - df.day_low).replace(0, np.nan)}, index=df.index)
    for c in ("st_dir", "psar_dir", "ichi_pos", "aroon_osc", "stoch_k", "stochrsi", "willr_14", "roc_10", "cci_20", "mfi_14", "ppo", "kc_pos", "vol_pctile", "range_exp", "atr_exp", "vol_accel", "avwap_dist_atr", "rsi_div_bull", "rsi_div_bear", "obv_div_bull", "obv_div_bear", "ctx_index_trend", "ctx_sector_trend", "rel_strength_12", "ctx_vol_level"):
        if c in df: X[c] = df[c]
    y = pd.Series(ret, index=df.index); sid = df.session_id.values; rows = []
    for c in X.columns:
        row = {"feature": c}
        for name, (lo, hi) in ranges.items():
            m = (sid >= lo) & (sid < hi); d = pd.concat([X[c][m], y[m]], axis=1).dropna()
            row[f"ic_{name}"] = round(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"), 4) if len(d) > 200 else np.nan
        s = [row.get("ic_train"), row.get("ic_val"), row.get("ic_test")]
        row["sign_stable"] = bool(all(v == v for v in s) and len({np.sign(v) for v in s}) == 1)
        row["abs_ic_train"] = abs(row.get("ic_train", np.nan)); rows.append(row)
    corr = X.dropna().corr(method="spearman").abs(); red = [(a, b, round(float(corr.loc[a, b]), 2)) for i, a in enumerate(X.columns) for b in X.columns[i + 1:] if corr.loc[a, b] > 0.85]
    return pd.DataFrame(rows).drop(columns="abs_ic_train"), red
