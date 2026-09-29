"""
Full pipeline run:
  1. Generate 2000-2025 synthetic OHLCV
  2. Compute standard indicators on training set
  3. Optimise custom indicator parameters on 2000-2020 data
  4. Apply frozen params to 2020-2025 data
  5. Run fixed-exit backtest (12% target / 7% stop)
  6. Print comprehensive report
"""
import sys, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data.longterm import generate_long_ohlcv
from features.indicators import compute_all
from features.custom_indicators import train_params, apply_custom_indicators
from strategy.custom_strategies import get_custom_strategies
from backtest.fixed_exit import run_fixed_exit_backtest, FullBacktestResult
from risk.engine import RiskConfig

pd.set_option("display.float_format", "{:.4f}".format)


def divider(title=""):
    print("\n" + "="*68)
    if title:
        print(f"  {title}")
        print("="*68)


def print_summary(label: str, result: FullBacktestResult):
    s = result.summary()
    divider(label)
    print(f"  Trades         : {s['n_trades']}")
    print(f"  Win Rate       : {s['win_rate_pct']}%  (min. needed for +EV: {s['required_win_rate_pct']}%)")
    print(f"  Exits          : +{s['profit_target_pct']}% target / -{s['stop_loss_pct']}% stop  [R:R = {s['risk_reward_ratio']}:1]")
    print(f"  Avg Win        : {s['avg_win_pct']}%")
    print(f"  Avg Loss       : {s['avg_loss_pct']}%")
    print(f"  Profit Factor  : {s['profit_factor']}")
    print(f"  Expectancy/trade: {s['expectancy_per_trade_pct']}%")
    print(f"  Max Drawdown   : {s['max_drawdown_pct']}%")
    print(f"  CAGR           : {s['cagr_pct']}%")
    print(f"  Max Consec W   : {s['max_consecutive_wins']}")
    print(f"  Max Consec L   : {s['max_consecutive_losses']}")


def print_strategy_breakdown(result: FullBacktestResult):
    print("\n--- By Strategy ---")
    for name, s in result.by_strategy().items():
        print(f"\n  {name}")
        print(f"    Trades     : {s['n_trades']}")
        print(f"    Win Rate   : {s['win_rate_pct']}%")
        print(f"    Prof Factor: {s['profit_factor']}")
        print(f"    Expectancy : {s['expectancy_per_trade_pct']}%")
        print(f"    Drawdown   : {s['max_drawdown_pct']}%")


def print_regime_breakdown(result: FullBacktestResult):
    print("\n--- By Market Regime ---")
    for r, s in result.by_regime().items():
        print(f"  {r:<20} | {s['n_trades']:>4} trades | WR {s['win_rate_pct']:>5.1f}% | PF {s['profit_factor']:>5.2f}")


def print_epoch_breakdown(result: FullBacktestResult):
    bd = result.by_epoch()
    if not bd:
        return
    print("\n--- By Market Epoch (synthetic labels) ---")
    for ep, s in bd.items():
        print(f"  {ep:<25} | {s['n_trades']:>4} trades | WR {s['win_rate_pct']:>5.1f}% | PF {s['profit_factor']:>5.2f}")


def print_equity_curve_ascii(result: FullBacktestResult, width=60, height=12):
    eq = result.equity_curve.values
    if len(eq) < 2:
        return
    lo, hi = eq.min(), eq.max()
    span = hi - lo if hi > lo else 1.0
    print("\n--- Equity Curve (x=trades, y=equity) ---")
    for row in range(height - 1, -1, -1):
        threshold = lo + span * row / (height - 1)
        line = ""
        for col in range(width):
            idx = int(col * (len(eq) - 1) / (width - 1))
            line += "█" if eq[idx] >= threshold else " "
        label = f"{threshold:.3f}" if row == height - 1 or row == 0 else "      "
        print(f"  {label} |{line}|")
    print("         " + "-" * width)
    print(f"         0{' ' * (width//2 - 4)}trades→{result.n - 1}")


def print_trade_log(result: FullBacktestResult, max_rows: int = 30):
    print(f"\n--- Trade Log (showing {min(max_rows, result.n)} of {result.n}) ---")
    print(f"  {'#':>4}  {'Date':<12} {'Dir':<6} {'Strategy':<22} {'Entry':>8} {'Exit':>8} {'Net%':>7} {'Result':<8} {'Regime'}")
    print(f"  {'-'*4}  {'-'*12} {'-'*6} {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*15}")
    for idx, t in enumerate(result.trades[:max_rows], 1):
        result_str = "WIN ✓" if t.win else "LOSS ✗"
        print(f"  {idx:>4}  {str(t.entry_time.date()):<12} {t.direction:<6} "
              f"{t.strategy:<22} {t.entry_price:>8.2f} {t.exit_price:>8.2f} "
              f"{t.net_pct*100:>+6.2f}% {result_str:<8} {t.regime}")
    if result.n > max_rows:
        print(f"  ... [{result.n - max_rows} more trades — full log in reports/trade_log.json]")


def hypothesis_check(oos: FullBacktestResult):
    divider("80% WIN-RATE HYPOTHESIS CHECK")
    wr = oos.win_rate
    n  = oos.n
    ev = oos.expectancy_pct

    from scipy.stats import binom
    # one-sided binomial test: H0 = win rate <= 50%, H1 = win rate > 50%
    p_value = 1 - binom.cdf(int(wr * n / 100) - 1, n, 0.50) if n > 0 else 1.0

    print(f"\n  Target win rate     : 80.0%")
    print(f"  Actual OOS win rate : {wr}%  (n={n} trades)")
    print(f"  Min for +EV         : 36.8%  [7/(12+7)]")
    print(f"  Expectancy          : {ev}% per trade")
    print(f"  p-value vs 50%      : {p_value:.4f}  {'✓ significant' if p_value < 0.05 else '✗ not significant'}")

    print()
    if wr >= 80 and n >= 30 and p_value < 0.05:
        print("  RESULT: 80% win-rate DEMONSTRATED on OOS data (adequate sample).")
    else:
        print("  RESULT: 80% win-rate target NOT demonstrated on OOS data.")
        if wr >= 36.8 and ev > 0:
            print(f"  NOTE:   System has POSITIVE EXPECTANCY ({ev}%/trade) even below 80% win rate.")
            print("          Per spec Section 8: optimize for expectancy, not win rate alone.")
        elif ev <= 0:
            print(f"  WARNING: Negative expectancy ({ev}%/trade) — strategy refinement needed")
            print("           before paper trading. Do NOT deploy.")

    if n < 30:
        print(f"\n  ⚠ SAMPLE SIZE WARNING: {n} trades is below the 30-trade minimum for")
        print("    any statistical conclusion. Widen date range or lower signal")
        print("    thresholds to generate more trades, then re-test.")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    divider("GENERATING 2000-2025 SYNTHETIC OHLCV")
    train_raw, test_raw = generate_long_ohlcv(seed=42)
    print(f"  Train (2000-2019): {len(train_raw)} trading days")
    print(f"  Test  (2020-2024): {len(test_raw)} trading days")

    divider("COMPUTING BASE INDICATORS — TRAINING SET")
    train_feat = compute_all(train_raw)
    print(f"  Indicators computed. Columns: {len(train_feat.columns)}")

    divider("TRAINING CUSTOM INDICATORS ON 2000-2020 DATA")
    learned = train_params(train_feat)

    divider("APPLYING FROZEN PARAMS TO 2020-2025 TEST DATA")
    test_feat = compute_all(test_raw)
    test_feat = apply_custom_indicators(test_feat, learned)
    print(f"  Custom indicators applied. Total columns: {len(test_feat.columns)}")

    # Also apply to training set for in-sample comparison
    train_feat = apply_custom_indicators(train_feat, learned)

    strategies = get_custom_strategies()
    risk_cfg = RiskConfig(
        max_risk_per_trade_pct=1.0,
        daily_loss_limit_pct=5.0,
        max_consecutive_losses=6,
        max_open_positions=1,
        min_risk_reward=1.5,
    )

    divider("RUNNING IN-SAMPLE BACKTEST (2000-2019)")
    train_result = run_fixed_exit_backtest(
        train_feat, strategies, risk_cfg, period_label="2000-2019 (in-sample)"
    )
    print_summary("IN-SAMPLE RESULTS: 2000-2019", train_result)
    print_strategy_breakdown(train_result)
    print_regime_breakdown(train_result)
    print_epoch_breakdown(train_result)
    print_equity_curve_ascii(train_result)
    print_trade_log(train_result, max_rows=20)

    divider("RUNNING OOS BACKTEST (2020-2024) — 500-trade target")
    oos_result = run_fixed_exit_backtest(
        test_feat, strategies, risk_cfg, period_label="2020-2024 (out-of-sample)"
    )
    print_summary("OUT-OF-SAMPLE RESULTS: 2020-2024", oos_result)
    print_strategy_breakdown(oos_result)
    print_regime_breakdown(oos_result)
    print_epoch_breakdown(oos_result)
    print_equity_curve_ascii(oos_result)
    print_trade_log(oos_result, max_rows=30)

    hypothesis_check(oos_result)

    # Save full trade log as JSON
    import os, json as _json
    os.makedirs("reports", exist_ok=True)
    log = []
    for t in oos_result.trades:
        log.append({
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time),
            "strategy": t.strategy,
            "direction": t.direction,
            "entry_price": round(t.entry_price, 4),
            "exit_price": round(t.exit_price, 4) if t.exit_price else None,
            "stop_loss": round(t.stop_loss, 4),
            "target": round(t.target, 4),
            "net_pct": round(t.net_pct * 100, 4),
            "win": t.win,
            "exit_reason": t.exit_reason,
            "regime": t.regime,
            "epoch": t.epoch,
            "confidence": t.confidence,
            "evidence": t.signal_evidence,
        })
    with open("reports/oos_trade_log.json", "w") as f:
        _json.dump(log, f, indent=2)

    # OOS summary JSON
    with open("reports/oos_summary.json", "w") as f:
        _json.dump({
            "overall": oos_result.summary(),
            "by_strategy": oos_result.by_strategy(),
            "by_regime": oos_result.by_regime(),
            "by_epoch": oos_result.by_epoch(),
        }, f, indent=2)

    divider("DONE")
    print(f"  Full trade log → reports/oos_trade_log.json")
    print(f"  Summary        → reports/oos_summary.json")
    print(f"\n  ⚠ SYNTHETIC DATA DISCLAIMER")
    print("  All results above used procedurally-generated OHLCV (no network")
    print("  access in this environment). Pipeline architecture and metric math")
    print("  are validated. For real conclusions, run on yfinance data locally.")
