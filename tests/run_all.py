"""Fallback runner when pytest is unavailable: python tests/run_all.py   (pytest also works: `pytest -q`)"""
import sys, importlib, traceback; sys.path.insert(0, ".")
ok = bad = 0
for m in ("test_backtest_metrics", "test_no_lookahead", "test_decision", "test_intraday", "test_extended"):
    mod = importlib.import_module(f"tests.{m}")
    for k, v in list(vars(mod).items()):
        if k.startswith("test_") and callable(v):
            try: v(); ok += 1
            except Exception: bad += 1; print("FAIL", m, k); traceback.print_exc()
print(f"{ok} passed, {bad} failed"); sys.exit(1 if bad else 0)
