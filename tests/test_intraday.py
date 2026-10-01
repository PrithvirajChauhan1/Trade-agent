import sys, os, tempfile; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from collections import namedtuple
from data.synthetic_5m import generate_5m
from data.quality import validate_intraday
from features.intraday import build_intraday_features
from strategy.intraday_components import generate_signals, CANDIDATES
from backtest.intraday import run_intraday, CostModel, cap_first_n
from risk.engine import RiskEngine, RiskConfig
from research.registry import Registry
from research import study

RAW = generate_5m(40, seed=3); FULL = build_intraday_features(RAW)
FCOLS = ["vwap", "rel_volume_tod", "or_high", "or_low", "htf_trend_15m", "htf_trend_1h", "prev_high", "gap_pct", "ema_21", "atr_14", "adx_14", "day_high", "bb_pctb", "rsi_14"]

def test_session_vwap_resets_each_session():
    f = FULL; first = f[f.bar_of_day == 0].iloc[5]
    assert abs(first.vwap - (first.high + first.low + first.close) / 3) < 1e-9

def test_intraday_features_no_lookahead():
    for t in (150, 431, 1333, 2010):   # includes mid-session and opening-range bars
        tr = build_intraday_features(RAW.iloc[:t + 1])
        for c in FCOLS:
            a, b = FULL[c].iloc[t], tr[c].iloc[t]
            assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-9), f"{c} differs at {t}: {a} vs {b}"
        assert FULL.regime.iloc[t] == tr.regime.iloc[t]

def test_opening_range_undefined_until_complete():
    assert FULL.loc[FULL.bar_of_day < 6, "or_high"].isna().all() and FULL.loc[FULL.bar_of_day >= 6, "or_high"].notna().all()

def test_signals_no_lookahead():
    from features.extra_indicators import add_extra
    from patterns.chart_patterns import add_pattern_columns
    ex = lambda r: add_pattern_columns(add_extra(build_intraday_features(r)))
    t = 1500; full, tr = ex(RAW), ex(RAW.iloc[:t + 1])
    for cfg in CANDIDATES:
        if "mkt_ctx" in cfg["entry"]: continue          # needs supplied context data; covered in test_extended
        a = generate_signals(full, cfg); b = generate_signals(tr, cfg)
        assert a[0][t] == b[0][t] and a[1][t] == b[1][t], cfg["id"]

def mk(rows, sids, atr=1.0):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float); df["atr_14"] = atr; df["session_id"] = sids
    df.index = pd.date_range("2024-01-01 09:15", periods=len(df), freq="5min"); return df

def one(df, i, **kw):
    n = len(df); L = np.zeros(n, bool); L[i] = True; S = np.zeros(n, bool)
    a = dict(stop_atr=1.0, target_rr=2.0, max_hold_bars=5, cost=CostModel(0, 0, 0, 0)); a.update(kw)
    return run_intraday(df, L, S, np.zeros(n), **a)

FLAT = [100, 100.5, 99.5, 100]

def test_entry_at_next_bar_open():
    df = mk([FLAT, FLAT, [100, 101, 99.5, 100.5], [100.5, 102.5, 100, 102]], [0] * 4); t = one(df, 1)[0]
    assert t.entry_time == df.index[2] and t.entry == 100.0 and t.exit_reason == "target" and abs(t.net_r - 2.0) < 1e-9

def test_same_candle_tp_sl_is_stop_first():
    df = mk([FLAT, FLAT, [100, 102.5, 98.5, 100], FLAT], [0] * 4); t = one(df, 1)[0]
    assert t.exit_reason == "stop_ambiguous" and t.ambiguous and abs(t.net_r + 1.0) < 1e-9

def test_time_exit_at_close_of_last_hold_bar():
    df = mk([FLAT] * 8, [0] * 8); t = one(df, 1, max_hold_bars=3)[0]
    assert t.exit_reason == "time" and t.bars_held == 3 and t.exit_time == df.index[4]

def test_session_end_exit_and_no_overnight():
    df = mk([FLAT] * 6, [0, 0, 0, 0, 1, 1]); t = one(df, 1, max_hold_bars=5)[0]
    assert t.exit_reason == "session_end" and t.exit_time == df.index[3]
    assert one(df, 3) == []      # signal on last bar of a session -> next open is a new session -> no trade

def test_gap_through_stop_fills_at_open():
    df = mk([FLAT, FLAT, [100, 100.5, 99.5, 100], [97, 97.5, 96.5, 97]], [0] * 4); t = one(df, 1)[0]
    assert t.exit_reason == "stop_gap" and t.exit == 97.0 and t.net_r < -1.5

def test_costs_reduce_net_result():
    df = mk([FLAT, FLAT, [100, 101, 99.5, 100.5], [100.5, 102.5, 100, 102]], [0] * 4)
    assert one(df, 1, cost=CostModel(1, 1, 1, 1))[0].net_r < one(df, 1)[0].net_r

def test_position_sizing_risk_based():
    df = mk([FLAT, FLAT, [100, 101, 99.5, 100.5], [100.5, 102.5, 100, 102]], [0] * 4); t = one(df, 1)[0]
    assert abs(t.units - 1_000_000 * 0.005 / 1.0) < 1e-6      # 0.5% of equity / 1.0 risk per unit

def test_daily_loss_halt_lifts_next_session():
    r = RiskEngine(RiskConfig(daily_loss_limit_pct=2.0), 100000); r.record_trade_result(-2500)
    assert not r.check_circuit_breakers(); r.reset_day(); assert r.check_circuit_breakers()

def test_consecutive_loss_halt_persists_across_days():
    r = RiskEngine(RiskConfig(max_consecutive_losses=2), 100000); r.record_trade_result(-1); r.record_trade_result(-1)
    assert not r.check_circuit_breakers(); r.reset_day(); assert not r.check_circuit_breakers()

def test_cap_is_chronological_not_best():
    T = namedtuple("T", "entry_time net_r"); ts = [T(pd.Timestamp(2024, 1, d), -1.0 if d < 4 else 5.0) for d in (9, 8, 3, 2, 1, 7)]
    assert [t.entry_time.day for t in cap_first_n(ts, 3)] == [1, 2, 3]

def test_walk_forward_is_chronological_and_non_overlapping():
    T = namedtuple("T", "session_idx net_r"); cfg = (1, 1.0, 1.0, None)
    grid = {c: [] for c in [(h, r, s, t) for h in study.HOLD for r in study.RR for s in study.STOP for t in study.TRAIL]}
    grid[cfg] = [T(i, 0.5) for i in range(100)]
    folds, oos = study.walk_forward(grid, 100, init_frac=0.4, test_len=20)
    assert len(folds) == 3 and [f["train_end"] for f in folds] == [40, 60, 80]
    assert [t.session_idx for t in oos] == list(range(40, 100))

def test_registry_versioning_and_approval_gate():
    with tempfile.TemporaryDirectory() as d:
        r = Registry(d + "/r.json"); cfg = dict(CANDIDATES[0]); k = r.register(cfg, "p", ["f"])
        assert r.register(cfg, "p", ["f"]) == k
        bad = dict(cfg); bad["entry"] = dict(cfg["entry"], adx_min=99)
        try: r.register(bad, "p", ["f"]); assert False
        except ValueError: pass
        r.set_status(k, "VALIDATED", "x")
        try: r.set_status(k, "PAPER_TRADING", "x"); assert False
        except PermissionError: pass
        try: r.set_status(k, "EXPERIMENTAL", "x"); assert False
        except ValueError: pass

def test_data_quality_detects_problems():
    assert validate_intraday(RAW)["ok"]
    dup = pd.concat([RAW, RAW.iloc[[10]]]).sort_index(); assert any("DUPLICATE" in i for i in validate_intraday(dup)["issues"])
    gap = RAW.drop(RAW.index[100:130]); assert any("MISSING" in i for i in validate_intraday(gap)["issues"])
    assert any("TIMEZONE" in i for i in validate_intraday(RAW.tz_localize(None))["issues"])
    pre = RAW.copy(); pre.index = pre.index - pd.Timedelta(hours=1); assert any("outside regular session" in i for i in validate_intraday(pre)["issues"])

def test_library_generation():
    from research.docs import write_library
    with tempfile.TemporaryDirectory() as d:
        r = Registry(d + "/r.json"); r.register(CANDIDATES[0], "p", ["f"]); write_library(r, d + "/L.md")
        assert "TREND_VWAP_001" in open(d + "/L.md").read()
