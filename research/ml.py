"""ML meta-labelling: P(trade wins | features at signal time), strictly chronological.
Model fit on TRAIN trades only; probability threshold chosen on VALIDATION; TEST evaluated once. Features are known at signal time only.
Does not change any strategy status automatically; results are research evidence."""
import numpy as np, pandas as pd
from scipy.stats import rankdata

def _features(f, idx, d):
    r = f.iloc[idx]; a = r.atr_14.values
    return pd.DataFrame({"rsi_c": (r.rsi_14.values - 50) / 50 * d, "ema9_21": (r.ema_9.values - r.ema_21.values) / a * d, "ema21_50": (r.ema_21.values - r.ema_50.values) / a * d,
        "vwap_dist": r.vwap_dist_atr.values * d, "macd": r["hist"].values / a * d, "htf15": r.htf_trend_15m.values * d, "htf1h": r.htf_trend_1h.values * d,
        "adx": r.adx_14.values / 50, "relvol": r.rel_volume_tod.values, "pctb": (r.bb_pctb.values - .5) * d, "gap": r.gap_pct.values * 100 * d,
        "range_pos": ((r.close.values - r.day_low.values) / np.maximum(r.day_high.values - r.day_low.values, 1e-9) - .5) * d,
        "bar": r.bar_of_day.values / 75, "atr_pct": a / r.close.values * 100})

def _auc(y, p):
    if y.min() == y.max(): return np.nan
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

class LogReg:
    def __init__(self, l2=1.0, lr=0.2, iters=600): self.l2, self.lr, self.iters = l2, lr, iters
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), np.where(X.std(0) == 0, 1, X.std(0)); Z = (X - self.mu) / self.sd; self.w = np.zeros(Z.shape[1]); self.b = 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b))); g = p - y
            self.w -= self.lr * (Z.T @ g / len(y) + self.l2 * self.w / len(y)); self.b -= self.lr * g.mean()
        return self
    def predict_proba(self, X): return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))

def meta_label(f, trades, ranges, key=""):
    sel = lambda rg: [t for t in trades if rg[0] <= t.session_idx < rg[1]]; A, B, C = sel(ranges["train"]), sel(ranges["val"]), sel(ranges["test"])
    if len(A) < 80 or len(B) < 30 or len(C) < 30: return [{"strategy": key, "model": "-", "status": f"insufficient trades (train {len(A)}, val {len(B)}, test {len(C)})"}]
    def xy(ts):
        idx = f.index.get_indexer([t.signal_time for t in ts]); d = np.array([1 if t.direction == "long" else -1 for t in ts])
        return _features(f, idx, d), np.array([t.net_r > 0 for t in ts], float), np.array([t.net_r for t in ts])
    (Xa, ya, _), (Xb, yb, rb), (Xc, yc, rc) = xy(A), xy(B), xy(C); med = Xa.median(); Xa, Xb, Xc = (x.fillna(med).values for x in (Xa, Xb, Xc))
    models = {"logreg": LogReg().fit(Xa, ya)}
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        models["grad_boost"] = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0).fit(Xa, ya)
    except Exception: pass
    base = ya.mean(); out = []
    for name, m in models.items():
        pb, pc = m.predict_proba(Xb)[:, -1] if name != "logreg" else m.predict_proba(Xb), m.predict_proba(Xc)[:, -1] if name != "logreg" else m.predict_proba(Xc)
        best = None
        for thr in np.arange(0.40, 0.66, 0.02):
            k = pb >= thr
            if k.sum() >= 15 and rb[k].mean() > rb.mean() and (best is None or rb[k].mean() > best[1]): best = (thr, rb[k].mean())
        row = {"strategy": key, "model": name, "n_train": len(A), "n_val": len(B), "n_test": len(C), "auc_val": round(_auc(yb, pb), 3), "auc_test": round(_auc(yc, pc), 3),
               "brier_test": round(float(((pc - yc) ** 2).mean()), 4), "brier_test_baseline": round(float(((base - yc) ** 2).mean()), 4), "threshold": None if best is None else round(best[0], 2),
               "test_unfiltered_expR": round(float(rc.mean()), 3), "status": "ok"}
        if best is None: row["status"] = "no threshold improved validation expectancy — ML filter not used"
        else:
            k = pc >= best[0]; row.update({"test_filtered_n": int(k.sum()), "test_filtered_expR": round(float(rc[k].mean()), 3) if k.any() else np.nan})
        q = pd.qcut(pc, 4, duplicates="drop"); row["calibration_test(pred->win)"] = "; ".join(f"{g.mean():.2f}->{yc[q == i].mean():.2f}" for i in q.categories for g in [pc[q == i]])
        out.append(row)
    return out
