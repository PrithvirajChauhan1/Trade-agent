"""Intraday backtester: signal at bar T -> entry at OPEN of T+1 (same session only) -> stop/target/time/session-end exit.
Same-candle TP+SL ambiguity is resolved by a PREDETERMINED conservative rule: stop first (counted and reported).
Reuses risk.engine.RiskEngine for sizing and circuit breakers."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np, pandas as pd
from risk.engine import RiskEngine, RiskConfig
from strategy.base import StrategySignal

@dataclass
class CostModel:
    commission_bps: float = 1.0   # per side
    spread_bps: float = 1.0       # full round-trip spread (half paid each side)
    slippage_bps: float = 1.0     # per side, adverse
    tax_bps: float = 1.0          # round-trip statutory levies (approximation; configure per market)
    mult: float = 1.0
    def adverse_bps(self): return self.mult * (self.spread_bps / 2 + self.slippage_bps)
    def fee_frac(self): return self.mult * (2 * self.commission_bps + self.tax_bps) / 1e4

@dataclass
class Trade:
    strategy: str; symbol: str; direction: str; signal_time: pd.Timestamp; entry_time: pd.Timestamp
    entry: float; stop: float; target: float; exit_time: pd.Timestamp; exit: float; exit_reason: str
    bars_held: int; units: float; gross_r: float; net_r: float; net_pnl: float; cost_r: float
    mfe_r: float; mae_r: float; ambiguous: bool; score: float; regime: str; tod: str; weekday: int
    session_idx: int; expected_rr: float

def run_intraday(df, long_sig, short_sig, score, *, stop_atr, target_rr, max_hold_bars, cost: CostModel = None,
                 trail_atr=None, risk_cfg: RiskConfig = None, equity=1_000_000.0, symbol="SYM", strategy="S"):
    cost = cost or CostModel(); risk_cfg = risk_cfg or RiskConfig(min_risk_reward=min(1.0, target_rr), max_consecutive_losses=10**6)
    risk = RiskEngine(risk_cfg, equity)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close")); atr = df["atr_14"].values
    sid = df["session_id"].values; n = len(df); idx = df.index
    last_of = pd.Series(np.arange(n)).groupby(sid).max().to_dict()
    regime = df["regime"].values if "regime" in df else np.array([""] * n); tod = df["tod"].values if "tod" in df else np.array([""] * n)
    wd = df["weekday"].values if "weekday" in df else np.zeros(n, int)
    trades, cur_sid, busy_until = [], None, -1
    for i in np.flatnonzero(long_sig | short_sig):
        if i < busy_until or i + 1 >= n or sid[i + 1] != sid[i] or np.isnan(atr[i]) or atr[i] <= 0: continue
        if sid[i] != cur_sid: risk.reset_day(); cur_sid = sid[i]
        if not risk.check_circuit_breakers(): continue
        d = 1 if long_sig[i] else -1; j = i + 1
        entry = o[j] * (1 + d * cost.adverse_bps() / 1e4)
        rdist = stop_atr * atr[i]; stop = entry - d * rdist; target = entry + d * target_rr * rdist
        sizing = risk.size_position(StrategySignal("long" if d > 0 else "short", entry, stop, target, 0.0))
        if sizing.rejected: continue
        risk.open_positions += 1
        end_bar = min(j + max_hold_bars - 1, last_of[sid[j]]); cur_stop = stop
        exit_px, reason, k_exit, amb, mfe, mae = None, None, end_bar, False, 0.0, 0.0
        for k in range(j, end_bar + 1):
            if k > j:   # gap through a level at the open
                if (d > 0 and o[k] <= cur_stop) or (d < 0 and o[k] >= cur_stop): exit_px, reason, k_exit = o[k], "stop_gap", k; break
                if (d > 0 and o[k] >= target) or (d < 0 and o[k] <= target): exit_px, reason, k_exit = target, "target", k; break
            hit_s = l[k] <= cur_stop if d > 0 else h[k] >= cur_stop
            hit_t = h[k] >= target if d > 0 else l[k] <= target
            fav = (h[k] - entry) if d > 0 else (entry - l[k]); adv = (entry - l[k]) if d > 0 else (h[k] - entry)
            mfe, mae = max(mfe, fav / rdist), max(mae, adv / rdist)
            if hit_s: exit_px, reason, k_exit, amb = cur_stop, ("stop_ambiguous" if hit_t else ("trail_stop" if cur_stop != stop else "stop")), k, hit_t; break
            if hit_t: exit_px, reason, k_exit = target, "target", k; break
            if trail_atr:   # trailing stop only tightens, applied from the NEXT bar
                cand = c[k] - d * trail_atr * atr[i]; cur_stop = max(cur_stop, cand) if d > 0 else min(cur_stop, cand)
        if exit_px is None:
            exit_px, k_exit = c[end_bar], end_bar; reason = "session_end" if end_bar == last_of[sid[j]] and end_bar < j + max_hold_bars - 1 else "time"
        exit_fill = exit_px * (1 - d * cost.adverse_bps() / 1e4)
        gross = d * (exit_px - o[j]) ; net = d * (exit_fill - entry) - cost.fee_frac() * entry
        pnl = net * sizing.shares_or_units; risk.record_trade_result(pnl)
        trades.append(Trade(strategy, symbol, "long" if d > 0 else "short", idx[i], idx[j], entry, stop, target, idx[k_exit], exit_fill, reason,
                            k_exit - j + 1, sizing.shares_or_units, gross / rdist, net / rdist, pnl, (gross / rdist) - (net / rdist),
                            mfe, mae, bool(amb), float(score[i]), str(regime[i]), str(tod[i]), int(wd[i]), int(sid[j]), target_rr))
        busy_until = k_exit
    return trades

def trades_df(trades): return pd.DataFrame([asdict(t) for t in trades])

def summarize(trades):
    if not trades: return {"n": 0}
    t = trades_df(trades); r = t.net_r.values; w = r > 0
    eq = np.cumsum(r); dd = eq - np.maximum.accumulate(eq)
    streak = lambda m: max((len(list(g)) for k, g in __import__("itertools").groupby(m) if k), default=0)
    daily = t.groupby("session_idx").net_r.sum(); sh = so = np.nan
    if len(daily) >= 10 and daily.std() > 0: sh = daily.mean() / daily.std() * np.sqrt(252)
    dn = daily[daily < 0]
    if len(daily) >= 10 and len(dn) > 1 and dn.std() > 0: so = daily.mean() / np.sqrt((dn ** 2).mean()) * np.sqrt(252)
    maxdd = float(-dd.min()); calmar = (daily.mean() * 252 / maxdd) if maxdd > 0 else np.nan
    return {"n": len(t), "wins": int(w.sum()), "losses": int((~w).sum()), "timeouts": int(t.exit_reason.isin(["time", "session_end"]).sum()),
            "ambiguous_bars": int(t.ambiguous.sum()), "win_rate": round(100 * w.mean(), 2), "loss_rate": round(100 * (~w).mean(), 2),
            "avg_win_R": round(float(r[w].mean()), 3) if w.any() else 0.0, "avg_loss_R": round(float(r[~w].mean()), 3) if (~w).any() else 0.0,
            "median_R": round(float(np.median(r)), 3), "largest_win_R": round(float(r.max()), 3), "largest_loss_R": round(float(r.min()), 3),
            "expectancy_R": round(float(r.mean()), 4), "profit_factor": round(float(r[w].sum() / -r[~w].sum()), 3) if (~w).any() and r[~w].sum() < 0 else np.inf,
            "max_dd_R": round(maxdd, 2), "dd_duration_trades": int(_dd_dur(dd)), "sharpe": round(float(sh), 2) if sh == sh else np.nan,
            "sortino": round(float(so), 2) if so == so else np.nan, "calmar": round(float(calmar), 2) if calmar == calmar else np.nan,
            "max_consec_wins": streak(w), "max_consec_losses": streak(~w), "avg_hold_bars": round(float(t.bars_held.mean()), 1),
            "avg_MFE_R": round(float(t.mfe_r.mean()), 2), "avg_MAE_R": round(float(t.mae_r.mean()), 2), "avg_cost_R": round(float(t.cost_r.mean()), 3),
            "trades_per_session": round(len(t) / max(1, t.session_idx.nunique()), 2)}

def _dd_dur(dd):
    best = cur = 0
    for x in dd: cur = cur + 1 if x < 0 else 0; best = max(best, cur)
    return best

def breakdown(trades, key):
    if not trades: return pd.DataFrame()
    t = trades_df(trades); g = t.groupby(key).net_r
    return pd.DataFrame({"n": g.size(), "win_rate": (t.assign(w=t.net_r > 0).groupby(key).w.mean() * 100).round(1),
                         "expectancy_R": g.mean().round(3), "total_R": g.sum().round(2)})

def cap_first_n(trades, n=200):
    """Predetermined research-sample cap: first n trades chronologically (never best-n)."""
    return sorted(trades, key=lambda x: x.entry_time)[:n]
