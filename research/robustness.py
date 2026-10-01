import numpy as np, pandas as pd
from backtest.intraday import trades_df

def robustness_rows(f, trades, key, train_range, split):
    if not trades: return []
    t = trades_df(trades); ix = f.index.get_indexer(t.signal_time); ap = (f.atr_14 / f.close).values
    q = np.nanquantile(ap[f.session_id.values < train_range[1]], [1 / 3, 2 / 3])          # terciles from TRAIN data only
    t["vol_bucket"] = np.select([ap[ix] < q[0], ap[ix] < q[1]], ["low", "mid"], "high"); t["high_volume"] = f.high_volume.values[ix]; t["year"] = pd.DatetimeIndex(t.entry_time).year
    rows = []
    for g in ("weekday", "tod", "regime", "year", "vol_bucket", "high_volume"):
        for v, d in t.groupby(g):
            rows.append({"strategy": key, "split": split, "group": g, "value": v, "n": len(d), "win_rate": round(100 * (d.net_r > 0).mean(), 1), "expectancy_R": round(d.net_r.mean(), 3), "total_R": round(d.net_r.sum(), 1)})
    return rows
