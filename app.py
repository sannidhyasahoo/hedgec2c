import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Setup Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion import load_master
from preprocessing import preprocess
from risk import compute_all_risk_metrics
from portfolio import Portfolio
from signals import MLSignalEngine

# Use full page width
st.set_page_config(layout="wide", page_title="Hedge Fund Risk Dashboard")

st.title("📈 Hedge Fund Risk & Trading Dashboard")
st.markdown("Interactive backtesting dashboard evaluating a **Risk-Aware Trend Following** strategy against a standard **Buy-and-Hold** approach.")

# --- Data Loading (Cached) ---
@st.cache_data
def get_data():
    raw = load_master()
    df, scaler = preprocess(raw)
    return raw, df

raw_df, df = get_data()

# --- Simulation Runner (Cached) ---
@st.cache_data
def run_simulations(df):
    # Strategy
    strat_port = Portfolio(initial_capital=100_000, max_position_pct=0.95)
    engine = MLSignalEngine()
    
    last_rebalance_idx = -999
    
    for i, row in df.iterrows():
        date = row["Date"]
        prices = {"Equity": row["Equity_Price"], "Gold": row["MA_Gold_Price"], "Oil": row["Oil_Price"]}
        strat_port.record_snapshot(date, prices)
        
        signal = engine.generate_signal(row)
        nav = strat_port.compute_nav(prices)
        current_weights = {asset: (strat_port.positions.get(asset, 0) * prices.get(asset, 0)) / nav if nav > 0 else 0 
                           for asset in prices}
                           
        is_risk_off = "Risk-Off" in signal.reason or "De-risk" in signal.reason
        max_deviation = max([abs(target_w - current_weights.get(asset, 0.0)) for asset, target_w in signal.target_weights.items()] + [0])
        
        if (i - last_rebalance_idx >= 21) or (is_risk_off and max_deviation > 0.05):
            strat_port.rebalance(signal.target_weights, prices, date, signal.reason)
            if i - last_rebalance_idx >= 21:
                last_rebalance_idx = i

    # Buy and hold
    bh_port = Portfolio(initial_capital=100_000)
    first_price = df.iloc[0]["Equity_Price"]
    bh_port.buy("Equity", (100_000 * 0.90) // first_price, first_price, df.iloc[0]["Date"], "Initial")
    
    for _, row in df.iterrows():
        bh_port.record_snapshot(row["Date"], {"Equity": row["Equity_Price"], "Gold": row["MA_Gold_Price"], "Oil": row["Oil_Price"]})

    return strat_port, bh_port

st.sidebar.header("Running Simulation...")
with st.spinner('Running Backtest Simulation...'):
    strat_port, bh_port = run_simulations(df)
st.sidebar.success("Simulation Complete!")

strat_nav = strat_port.nav_history
bh_nav = bh_port.nav_history
strat_ret = strat_port.get_returns()
bh_ret = bh_port.get_returns()
equity_ret = df["Equity_Returns_clean"]

strat_metrics = compute_all_risk_metrics(strat_ret, benchmark_returns=equity_ret)
bh_metrics = compute_all_risk_metrics(bh_ret, benchmark_returns=equity_ret)

# --- Display KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategy Total Return", f"{strat_metrics.total_return:.2%}", delta_color="normal")
col2.metric("Buy & Hold Return", f"{bh_metrics.total_return:.2%}", delta_color="normal")
col3.metric("Strategy Max Drawdown", f"{strat_metrics.drawdown.max_drawdown:.2%}")
col4.metric("Strategy Sharpe", f"{strat_metrics.sharpe_ratio:.2f}")

st.divider()

# --- Interactive Charts ---
st.subheader("Performance Comparison (NAV)")
chart_data = pd.DataFrame({
    'Date': strat_nav['date'],
    'Active Strategy': strat_nav['nav'],
    'Buy & Hold': bh_nav['nav']
}).set_index('Date')

st.line_chart(chart_data)

st.subheader("Drawdown Comparison")
cum_s = (1 + strat_ret).cumprod()
cum_b = (1 + bh_ret).cumprod()
dd_s = (cum_s - cum_s.cummax()) / cum_s.cummax()
dd_b = (cum_b - cum_b.cummax()) / cum_b.cummax()

dd_data = pd.DataFrame({
    'Date': strat_nav['date'],
    'Strategy Drawdown': dd_s.values,
    'B&H Drawdown': dd_b.values
}).set_index('Date')

st.line_chart(dd_data)

# --- Risk Table ---
st.subheader("Detailed Risk Metrics")
metrics_df = pd.DataFrame({
    "Metric": ["Annualised Return", "Annualised Volatility", "Sharpe Ratio", "Sortino Ratio", "VaR 95%", "Max Drawdown"],
    "Active Strategy": [
        f"{strat_metrics.ann_return:.2%}", f"{strat_metrics.ann_volatility:.2%}", 
        f"{strat_metrics.sharpe_ratio:.2f}", f"{strat_metrics.sortino_ratio:.2f}", 
        f"{strat_metrics.var_95.historical:.2%}", f"{strat_metrics.drawdown.max_drawdown:.2%}"
    ],
    "Buy & Hold": [
        f"{bh_metrics.ann_return:.2%}", f"{bh_metrics.ann_volatility:.2%}", 
        f"{bh_metrics.sharpe_ratio:.2f}", f"{bh_metrics.sortino_ratio:.2f}", 
        f"{bh_metrics.var_95.historical:.2%}", f"{bh_metrics.drawdown.max_drawdown:.2%}"
    ]
})
st.table(metrics_df)

# --- Trade Logs ---
st.subheader("Trade Audit Log (Last 100 Trades)")
trades = strat_port.trade_log
if not trades.empty:
    st.dataframe(trades.tail(100).sort_values(by="date", ascending=False), use_container_width=True)
else:
    st.write("No trades executed.")
