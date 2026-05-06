"""
evaluate_ml.py
--------------
Trains the Machine Learning model and backtests the ML Signal Engine.

Usage:
    venv\\Scripts\\python evaluate_ml.py
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion import load_master
from preprocessing import preprocess
from ml_model import train_model
from portfolio import Portfolio
from signals import MLSignalEngine
from risk import compute_all_risk_metrics

def evaluate():
    print("============================================================")
    print("  LOADING & PREPROCESSING")
    print("============================================================")
    raw = load_master()
    df, _ = preprocess(raw)
    
    print("\n============================================================")
    print("  TRAINING ML MODEL")
    print("============================================================")
    # Train and save the model
    train_model(df)
    
    print("\n============================================================")
    print("  BACKTESTING ML STRATEGY")
    print("============================================================")
    
    # Initialize the new ML Signal Engine
    engine = MLSignalEngine()
    
    port = Portfolio(initial_capital=100_000, max_position_pct=0.95)
    last_rebalance_idx = -999
    
    for i, row in df.iterrows():
        date = row["Date"]
        prices = {
            "Equity": row["Equity_Price"],
            "Gold": row["MA_Gold_Price"],
            "Oil": row["Oil_Price"],
        }
        
        port.record_snapshot(date, prices)
        
        # ML Engine requires all features to be present. If missing, it will throw an error or predict badly.
        # But we dropped warmup NaNs in preprocess, so we're good.
        try:
            signal = engine.generate_signal(row)
        except Exception as e:
            # First few rows might lack rolling features if we didn't drop them all
            continue
            
        nav = port.compute_nav(prices)
        current_weights = {asset: (port.positions.get(asset, 0) * prices.get(asset, 0)) / nav if nav > 0 else 0 
                           for asset in prices}
                           
        # Check max deviation
        max_deviation = 0
        for asset, target_w in signal.target_weights.items():
            curr_w = current_weights.get(asset, 0.0)
            max_deviation = max(max_deviation, abs(target_w - curr_w))
            
        # Rebalance every 21 days OR if deviation > 10% (more lenient than rule-based to avoid overtrading)
        if (i - last_rebalance_idx >= 21) or (max_deviation > 0.10):
            port.rebalance(signal.target_weights, prices, date, signal.reason)
            if i - last_rebalance_idx >= 21:
                last_rebalance_idx = i

    strat_ret = port.get_returns()
    equity_ret = df["Equity_Returns_clean"].values[-len(strat_ret):] if len(strat_ret) > 0 else df["Equity_Returns_clean"].values
    
    metrics = compute_all_risk_metrics(strat_ret.reset_index(drop=True), benchmark_returns=pd.Series(equity_ret), risk_free_annual=0.02)
    
    print("\n  [ML Strategy Portfolio]")
    print(metrics.summary())
    
    print(f"\n  Total trades executed: {len(port.trade_log)}")
    print("  First 10 Trades:")
    if not port.trade_log.empty:
        print(port.trade_log.head(10)[["date", "asset", "action", "quantity", "reason"]].to_string())

if __name__ == "__main__":
    evaluate()
