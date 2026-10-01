"""
Verify BacktestResult computes metrics correctly on hand-constructed trades
with known answers — this is the check that matters most: if the metric
math is wrong, every downstream decision (Section 8, 9, 14) is built on sand.
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from backtest.engine import Trade, BacktestResult


def make_trade(net_pnl, r_multiple, strategy="s1", regime="strong_bull"):
    return Trade(
        strategy=strategy, direction="long",
        entry_time=pd.Timestamp("2024-01-01"), entry_price=100,
        stop_loss=98, target=104,
        net_pnl=net_pnl, r_multiple=r_multiple, regime_at_entry=regime,
    )


def test_win_rate():
    trades = [make_trade(10, 5), make_trade(-5, -2.5), make_trade(10, 5), make_trade(-5, -2.5)]
    r = BacktestResult(trades)
    assert r.win_rate == 50.0, f"expected 50.0, got {r.win_rate}"


def test_profit_factor():
    trades = [make_trade(20, 10), make_trade(-10, -5)]
    r = BacktestResult(trades)
    assert r.profit_factor == 2.0, f"expected 2.0, got {r.profit_factor}"


def test_expectancy():
    trades = [make_trade(10, 5), make_trade(-10, -5)]
    r = BacktestResult(trades)
    assert r.expectancy == 0.0, f"expected 0.0, got {r.expectancy}"


def test_expectancy_known_positive():
    # 60% win rate, avg win 10, avg loss -5 -> expectancy = 0.6*10 - 0.4*5 = 4.0
    trades = [make_trade(10, 2), make_trade(10, 2), make_trade(10, 2), make_trade(-5, -1), make_trade(-5, -1)]
    r = BacktestResult(trades)
    assert abs(r.expectancy - 4.0) < 1e-9, f"expected 4.0, got {r.expectancy}"


def test_max_drawdown():
    # equity path: +10, +10, -25, +5  -> cumsum: 10, 20, -5, 0 ; peak: 10,20,20,20 ; dd: 0,0,-25,-20
    trades = [make_trade(10, 1), make_trade(10, 1), make_trade(-25, -2), make_trade(5, 0.5)]
    r = BacktestResult(trades)
    assert r.max_drawdown_pct == -25.0, f"expected -25.0, got {r.max_drawdown_pct}"


def test_by_strategy_splits_correctly():
    trades = [make_trade(10, 1, strategy="a"), make_trade(-10, -1, strategy="a"), make_trade(10, 1, strategy="b")]
    r = BacktestResult(trades)
    by_strat = r.by_strategy()
    assert by_strat["a"]["n_trades"] == 2
    assert by_strat["b"]["n_trades"] == 1
    assert by_strat["b"]["win_rate_pct"] == 100.0


def test_empty_result_no_crash():
    r = BacktestResult([])
    assert r.win_rate == 0.0
    assert r.profit_factor == 0.0
    assert r.expectancy == 0.0
    assert r.max_drawdown_pct == 0.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
