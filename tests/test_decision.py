import sys; sys.path.insert(0, ".")
import datetime as dt, pandas as pd, numpy as np
from fundamental.filter import FundamentalInputs, evaluate
from decision.engine import decide
import decision.engine as E

TODAY = dt.date(2026, 9, 29)
def fi(**k):
    base = dict(as_of="2026-09-20", pe_percentile_10y=50, earnings_growth_yoy_pct=10, india_vix=15)
    base.update(k); return FundamentalInputs(**base)

def test_missing_data_unverified():   assert evaluate(FundamentalInputs(), "long", TODAY)[0] == "UNVERIFIED"
def test_stale_unverified():          assert evaluate(fi(as_of="2026-01-01"), "long", TODAY)[0] == "UNVERIFIED"
def test_high_vix_blocks():           assert evaluate(fi(india_vix=35), "long", TODAY)[0] == "BLOCK"
def test_expensive_blocks_long():     assert evaluate(fi(pe_percentile_10y=97), "long", TODAY)[0] == "BLOCK"
def test_two_cautions():              assert evaluate(fi(earnings_growth_yoy_pct=-3, rbi_stance="tightening"), "long", TODAY)[0] == "CAUTION"
def test_clean_pass():                assert evaluate(fi(), "long", TODAY)[0] == "PASS"

class FakeEns:
    def __init__(self, sig): self.sig = sig
    def evaluate(self, df, i): return self.sig

def frame(days_old=0):
    idx = pd.bdate_range(end=pd.Timestamp(TODAY) - pd.Timedelta(days=days_old), periods=300)
    df = pd.DataFrame(1.0, index=idx, columns=E.NEEDED); return df

def test_stale_market_data():
    assert decide(frame(30), FakeEns(None), {}, fi(), TODAY).action == "NO-TRADE"

def test_regime_not_enabled_blocks(monkeypatch=None):
    E.classify_regime = lambda d: pd.Series("sideways", index=d.index)
    d = decide(frame(), FakeEns(None), {"sideways": {"status": "DISABLED"}}, fi(), TODAY)
    assert d.action == "NO-TRADE"

def test_enabled_no_signal_waits():
    E.classify_regime = lambda d: pd.Series("strong_bull", index=d.index)
    assert decide(frame(), FakeEns(None), {"strong_bull": {"status": "ENABLED"}}, fi(), TODAY).action == "WAIT"

def test_signal_plus_pass_enters_and_block_vetoes():
    from strategy.base import StrategySignal
    E.classify_regime = lambda d: pd.Series("strong_bull", index=d.index)
    sig = StrategySignal("long", 100, 93, 112, 60, ["x"])
    tbl = {"strong_bull": {"status": "ENABLED"}}
    assert decide(frame(), FakeEns(sig), tbl, fi(), TODAY).action == "ENTRY"
    assert decide(frame(), FakeEns(sig), tbl, fi(india_vix=40), TODAY).action == "NO-TRADE"
    assert decide(frame(), FakeEns(sig), tbl, FundamentalInputs(), TODAY).action == "WAIT"

if __name__ == "__main__":
    ok = bad = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            try: v(); ok += 1; print("PASS", k)
            except AssertionError as e: bad += 1; print("FAIL", k, e)
    print(f"{ok} passed, {bad} failed"); sys.exit(bad)
