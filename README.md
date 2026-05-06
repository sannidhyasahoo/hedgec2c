# Hedge Fund Risk Modeling & Semi-Automated Trading System

## Team Information
- **Team Name**: Paneer_Lovers
- **Year**: 2
- **All-Female Team**: No
## Architecture Overview

#### Describe your approach here. Keep it short and clear.

    - How does your system ingest and preprocess the varying data sources (market, macro, sentiment)?
      The system uses `ingestion.py` to merge four datasets (Equity, Macro, Oil, Multi-Asset) via an inner join on the 'Date' index. `preprocessing.py` standardizes scales using Z-score Normalization and applies Winsorization (±3 sigma) to clip extreme outliers, preventing bias. We engineer features like Rolling Volatility, Momentum, and Cross-Asset Correlation.

    - What risk modeling techniques were selected, and how are they integrated into the trading decision pipeline?
      We implemented Historical VaR, Parametric VaR, Conditional VaR (Expected Shortfall), and Maximum Drawdown. These are integrated as "Safety Overlays." If 20-day rolling volatility breaches 25%, the system overrides our Machine Learning signals and executes an emergency "Risk-Off" shift into Gold and Cash.

    - How does your semi-automated strategy generate signals while respecting portfolio constraints and handling realistic conditions like slippage?
      An `MLSignalEngine` generates portfolio targets using a Random Forest model tuned via TimeSeriesSplit to prevent look-ahead bias. The Portfolio manager executes targets while enforcing `max_position_pct` constraints. It simulates 0.1% Commission and 0.05% Slippage, and utilizes a "Significance Filter" (>5% deviation) to avoid excessive trading fees.

    - How is the dashboard designed to provide explainable insights and key metrics (Sharpe, drawdown) to stakeholders?
      The interactive Streamlit dashboard provides live NAV Line Charts and Drawdown Visualizations compared to a Buy-and-Hold benchmark. It computes live risk-adjusted metrics (Sharpe, Sortino, Calmar, Alpha, Beta). Most importantly, it features an immutable Trade Audit Log displaying the exact algorithmic probability or rule that triggered every transaction.

**Note:** Please do not change the format or spelling of anything in this README. The fields are extracted using a script, so any changes to the structure or formatting may break the extraction process.
