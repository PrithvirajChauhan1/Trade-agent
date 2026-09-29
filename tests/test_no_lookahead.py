"""
Critical correctness check: values computed at bar i must not change if we
truncate the DataFrame to i+1 rows and recompute. If they do, there's
lookahead bias — the single most dangerous class of bug in a backtester,
because it silently inflates win rate/expectancy in exactly the direction
that would falsely "confirm" the 80% hypothesis.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from data.synthetic import default_mixed_regime_series
from features.indicators import compute_all
from features.regime import classify_regime


def test_indicators_no_lookahead():
    df = default_mixed_regime_series(seed=1)
    full = compute_all(df)

    check_at = 800  # arbitrary bar well past all warmups
    truncated = compute_all(df.iloc[:check_at + 1])

    cols_to_check = ["ema_21", "ema_50", "rsi_14", "macd", "atr_14", "bb_mid", "adx_14", "donchian_upper"]
    for col in cols_to_check:
        full_val = full[col].iloc[check_at]
        trunc_val = truncated[col].iloc[check_at]
        # allow tiny float drift, not structural difference
        assert np.isclose(full_val, trunc_val, rtol=1e-6, equal_nan=True), \
            f"{col} differs: full={full_val} truncated={trunc_val} — LOOKAHEAD BUG"


def test_regime_no_lookahead():
    df = default_mixed_regime_series(seed=1)
    full = compute_all(df)
    full["regime"] = classify_regime(full)

    check_at = 800
    truncated = compute_all(df.iloc[:check_at + 1])
    truncated["regime"] = classify_regime(truncated)

    assert full["regime"].iloc[check_at] == truncated["regime"].iloc[check_at], \
        f"regime differs: full={full['regime'].iloc[check_at]} truncated={truncated['regime'].iloc[check_at]}"


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
