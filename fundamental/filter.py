"""
Fundamental veto filter for an INDEX (Sensex). Read this before trusting it:
- Point-in-time fundamental history is not available here, so this filter is NOT
  backtested. Thresholds are heuristics (config below). It can only VETO or delay
  a technical signal; it never creates one.
- Missing or stale inputs => UNVERIFIED (never guessed).
"""
from __future__ import annotations
import json, datetime as dt
from dataclasses import dataclass, fields
from typing import Optional

@dataclass
class FundamentalInputs:
    as_of: Optional[str] = None                 # YYYY-MM-DD
    pe_percentile_10y: Optional[float] = None   # 0-100, index trailing PE vs own 10y history
    earnings_growth_yoy_pct: Optional[float] = None
    india_vix: Optional[float] = None
    fii_net_flow_20d_cr: Optional[float] = None # INR crore, + = net buying
    usdinr_3m_change_pct: Optional[float] = None  # + = rupee weakening
    brent_3m_change_pct: Optional[float] = None
    rbi_stance: Optional[str] = None            # easing | neutral | tightening

CFG = dict(max_age_days=45, vix_block=30, pe_block_long=95, pe_caution_long=85, pe_block_short=10,
           pe_caution_short=25, fii_big=20000, usdinr_caution=3.0, brent_caution=20.0,
           eps_neg=0.0, eps_hot=15.0)
REQUIRED = ["pe_percentile_10y", "india_vix", "earnings_growth_yoy_pct"]

def load_inputs(path):
    try: raw = json.load(open(path))
    except FileNotFoundError: return FundamentalInputs()
    return FundamentalInputs(**{f.name: raw.get(f.name) for f in fields(FundamentalInputs)})

def evaluate(fi: FundamentalInputs, direction: str, today: dt.date, cfg=CFG):
    """Returns (verdict, reasons): PASS | CAUTION | BLOCK | UNVERIFIED"""
    if fi.as_of is None: return "UNVERIFIED", ["no fundamental data supplied"]
    age = (today - dt.date.fromisoformat(fi.as_of)).days
    if age > cfg["max_age_days"]: return "UNVERIFIED", [f"fundamental data {age}d old (> {cfg['max_age_days']}d)"]
    miss = [k for k in REQUIRED if getattr(fi, k) is None]
    if miss: return "UNVERIFIED", [f"missing required fields: {miss}"]
    block, caution, L = [], [], direction == "long"
    if fi.india_vix > cfg["vix_block"]: block.append(f"India VIX {fi.india_vix} > {cfg['vix_block']}")
    if L:
        if fi.pe_percentile_10y > cfg["pe_block_long"]: block.append(f"PE at {fi.pe_percentile_10y}th pct of 10y")
        elif fi.pe_percentile_10y > cfg["pe_caution_long"]: caution.append("valuation stretched")
        if fi.earnings_growth_yoy_pct < cfg["eps_neg"]: caution.append("earnings contracting")
        if fi.fii_net_flow_20d_cr is not None and fi.fii_net_flow_20d_cr < -cfg["fii_big"]: caution.append("heavy FII selling")
        if fi.usdinr_3m_change_pct is not None and fi.usdinr_3m_change_pct > cfg["usdinr_caution"]: caution.append("rupee weakening")
        if fi.brent_3m_change_pct is not None and fi.brent_3m_change_pct > cfg["brent_caution"]: caution.append("crude spike")
        if fi.rbi_stance == "tightening": caution.append("RBI tightening")
    else:
        if fi.pe_percentile_10y < cfg["pe_block_short"]: block.append(f"PE at {fi.pe_percentile_10y}th pct (cheap)")
        elif fi.pe_percentile_10y < cfg["pe_caution_short"]: caution.append("valuation cheap")
        if fi.earnings_growth_yoy_pct > cfg["eps_hot"]: caution.append("strong earnings growth")
        if fi.fii_net_flow_20d_cr is not None and fi.fii_net_flow_20d_cr > cfg["fii_big"]: caution.append("heavy FII buying")
        if fi.rbi_stance == "easing": caution.append("RBI easing")
    if block: return "BLOCK", block + caution
    if len(caution) >= 2: return "CAUTION", caution
    return "PASS", caution or ["no fundamental objections"]
