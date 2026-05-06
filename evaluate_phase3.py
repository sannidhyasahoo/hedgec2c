"""
evaluate_phase3.py
------------------
Phase 3 evaluation: Signal Engine + Portfolio Simulator.

Runs the Risk-Aware Signal Engine over the dataset and evaluates
its performance against the Buy-and-Hold benchmark.

Usage:
    venv\\Scripts\\python evaluate_phase3.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion   import load_master
from preprocessing import preprocess
from risk        import compute_all_risk_metrics
from portfolio   import Portfolio
from signals     import RiskAwareSignalEngine

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def run_strategy(df: pd.DataFrame) -> Portfolio:
    port = Portfolio(
        initial_capital  = 100_000,
        transaction_cost = 0.001,
        slippage_pct     = 0.0005,
        max_position_pct = 0.95,
    )
    
    engine = RiskAwareSignalEngine()
    
    # We rebalance monthly (every 21 trading days) to save transaction costs,
    # unless a critical risk threshold is crossed (sentiment drops suddenly).
    last_rebalance_idx = -999
    
    for i, row in df.iterrows():
        date = row["Date"]
        prices = {
            "Equity": row["Equity_Price"],
            "Gold"  : row["MA_Gold_Price"],
            "Oil"   : row["Oil_Price"],
        }
        
        # Always record snapshot to track daily NAV
        port.record_snapshot(date=date, prices=prices)
        
        # Generate target weights
        signal = engine.generate_signal(row)
        
        # Current portfolio value and allocations
        nav = port.compute_nav(prices)
        current_weights = {asset: (port.positions.get(asset, 0) * prices.get(asset, 0)) / nav if nav > 0 else 0 
                           for asset in prices}
        
        # Rebalance if 21 days have passed OR it's a "Risk-Off" emergency signal
        is_risk_off = "Risk-Off" in signal.reason or "De-risk" in signal.reason
        
        # Calculate max deviation from target weights
        max_deviation = 0
        for asset, target_w in signal.target_weights.items():
            curr_w = current_weights.get(asset, 0.0)
            max_deviation = max(max_deviation, abs(target_w - curr_w))
        
        # Only rebalance if it's been 21 days OR (it's risk-off AND we are off-target by > 5%)
        # This prevents trading every single day during a prolonged risk-off regime.
        should_rebalance = (i - last_rebalance_idx >= 21) or (is_risk_off and max_deviation > 0.05)
        
        if should_rebalance:
            port.rebalance(
                target_weights=signal.target_weights,
                prices=prices,
                date=date,
                reason=signal.reason
            )
            # update last rebalance index, but if we emergency traded, we still wait 21 days
            # before regular rebalancing to avoid whipsawing.
            if i - last_rebalance_idx >= 21:
                last_rebalance_idx = i

    return port

def run_buy_and_hold(df: pd.DataFrame) -> Portfolio:
    """Run pure Buy and Hold for benchmark comparison"""
    port = Portfolio(initial_capital=100_000)
    first_price = df.iloc[0]["Equity_Price"]
    quantity = (100_000 * 0.90) // first_price
    port.buy("Equity", quantity, first_price, df.iloc[0]["Date"], "Buy and Hold")
    
    for _, row in df.iterrows():
        port.record_snapshot(
            date=row["Date"],
            prices={"Equity": row["Equity_Price"], "Gold": row["MA_Gold_Price"], "Oil": row["Oil_Price"]}
        )
    return port

def evaluate():
    section("LOADING & PREPROCESSING")
    raw = load_master()
    df, _ = preprocess(raw)
    print(f"  Dataset: {len(df):,} rows")

    section("STEP 1 — Run Strategy")
    strat_port = run_strategy(df)
    strat_ret = strat_port.get_returns()
    
    bh_port = run_buy_and_hold(df)
    bh_ret = bh_port.get_returns()

    section("STEP 2 — Risk Metrics Comparison")
    
    equity_ret = df["Equity_Returns_clean"]
    
    strat_metrics = compute_all_risk_metrics(strat_ret, benchmark_returns=equity_ret, risk_free_annual=0.02)
    bh_metrics = compute_all_risk_metrics(bh_ret, benchmark_returns=equity_ret, risk_free_annual=0.02)
    
    print("\n  [Active Strategy Portfolio]")
    print(strat_metrics.summary())
    
    print("\n  [Buy & Hold Portfolio]")
    print(bh_metrics.summary())
    
    section("STEP 3 — Extracting Trade Logs")
    tlog = strat_port.trade_log
    print(f"\n  Total trades executed by Strategy: {len(tlog)}")
    print("  Showing first 10 trades:")
    if len(tlog) > 0:
        print(tlog.head(10).to_string(index=False, columns=["date", "asset", "action", "quantity", "price", "reason"]))

    section("STEP 4 — Generating Phase 3 Charts")
    _save_charts(df, strat_port.nav_history, bh_port.nav_history, strat_ret, bh_ret)
    
    section("PHASE 3 COMPLETE")

def _save_charts(df, strat_nav, bh_nav, strat_ret, bh_ret):
    plt.rcParams.update({"figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d2e", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white", "grid.color": "#2a2d3e"})

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(strat_nav["date"], strat_nav["nav"], color="#34d399", lw=1.5, label="Active Strategy")
    ax.plot(bh_nav["date"], bh_nav["nav"], color="#ff6b35", lw=1.0, alpha=0.7, label="Buy & Hold ($100K)")
    ax.axhline(100_000, color="white", lw=0.5, linestyle=":")
    ax.set_title("Strategy vs Buy & Hold NAV", fontsize=13)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "12_strategy_vs_bh.png", dpi=120)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(16, 5))
    cum_s = (1 + strat_ret).cumprod()
    cum_b = (1 + bh_ret).cumprod()
    ax.fill_between(strat_nav["date"], (cum_s - cum_s.cummax())/cum_s.cummax(), color="#34d399", alpha=0.5, label="Active Strategy DD")
    ax.fill_between(bh_nav["date"], (cum_b - cum_b.cummax())/cum_b.cummax(), color="#f87171", alpha=0.4, label="Buy & Hold DD")
    ax.set_title("Drawdown Comparison", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "13_strategy_drawdown.png", dpi=120)
    plt.close()
    print("  Saved: 12_strategy_vs_bh.png and 13_strategy_drawdown.png")

if __name__ == "__main__":
    evaluate()
