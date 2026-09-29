import sys
sys.path.insert(0, ".")
from data.synthetic import default_mixed_regime_series
from features.indicators import compute_all
from features.regime import classify_regime
from strategy.library import ALL_STRATEGIES
from backtest.engine import run_backtest, walk_forward_split
from risk.engine import RiskConfig
import json

raw = default_mixed_regime_series(bar_minutes=5, seed=42)
print(f"Generated {len(raw)} bars")

feat = compute_all(raw)
train, val, test = walk_forward_split(feat, train_frac=0.6, val_frac=0.2)
print(f"Train: {len(train)}  Val: {len(val)}  Test (out-of-sample): {len(test)}")

risk_cfg = RiskConfig(max_risk_per_trade_pct=0.5, min_risk_reward=1.5)

print("\n=== TRAIN SET ===")
train_result = run_backtest(train, ALL_STRATEGIES, risk_cfg)
print(json.dumps(train_result.summary(), indent=2))
print("By strategy:")
print(json.dumps(train_result.by_strategy(), indent=2))

print("\n=== VALIDATION SET ===")
val_result = run_backtest(val, ALL_STRATEGIES, risk_cfg)
print(json.dumps(val_result.summary(), indent=2))

print("\n=== TEST SET (OUT-OF-SAMPLE, held out) ===")
test_result = run_backtest(test, ALL_STRATEGIES, risk_cfg)
print(json.dumps(test_result.summary(), indent=2))
print("By strategy:")
print(json.dumps(test_result.by_strategy(), indent=2))
print("By regime:")
print(json.dumps(test_result.by_regime(), indent=2))

print("\n=== 80% WIN RATE HYPOTHESIS CHECK ===")
oos_wr = test_result.win_rate
if oos_wr >= 80 and test_result.n_trades >= 30:
    print(f"Out-of-sample win rate {oos_wr}% on {test_result.n_trades} trades — meets 80% threshold with adequate sample.")
else:
    print(f"80% win-rate target NOT statistically demonstrated. "
          f"Out-of-sample: {oos_wr}% win rate over {test_result.n_trades} trades. "
          f"Expectancy (R): {test_result.expectancy_r}")
