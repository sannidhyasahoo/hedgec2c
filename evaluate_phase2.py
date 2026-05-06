"""
evaluate_phase2.py
------------------
Phase 2 evaluation: Risk Model + Portfolio State Manager.

Runs a simple Buy-and-Hold simulation to validate the full
risk / portfolio pipeline before building the signal engine.

Usage:
    venv\\Scripts\\python evaluate_phase2.py
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
from risk        import (
    compute_all_risk_metrics,
    rolling_var,
    rolling_sharpe,
    rolling_volatility,
)
from portfolio   import Portfolio

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation: Buy-and-Hold Equity
# ─────────────────────────────────────────────────────────────────────────────

def run_buy_and_hold(df: pd.DataFrame) -> Portfolio:
    """
    Simulate a simple buy-and-hold strategy on Equity.
    Buys on Day 1 with 90% of capital, holds to the end.
    This validates the Portfolio class before any signal logic.
    """
    port = Portfolio(
        initial_capital  = 100_000,
        transaction_cost = 0.001,    # 0.1% commission
        slippage_pct     = 0.0005,   # 0.05% slippage
        max_position_pct = 0.95,     # allow up to 95% in one asset
    )

    first_row  = df.iloc[0]
    first_price = first_row["Equity_Price"]
    first_date  = first_row["Date"]

    # Allocate 90% of capital to equity on Day 1
    budget   = port.initial_capital * 0.90
    quantity = budget // first_price          # whole shares only
    port.buy("Equity", quantity, first_price, first_date,
             reason="Buy-and-hold initialisation")

    # Step through every day
    for _, row in df.iterrows():
        port.record_snapshot(
            date   = row["Date"],
            prices = {
                "Equity": row["Equity_Price"],
                "Gold"  : row["MA_Gold_Price"],
                "Oil"   : row["Oil_Price"],
            }
        )

    return port


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate():

    # ── Load + preprocess ─────────────────────────────────────────────────────
    section("LOADING & PREPROCESSING")
    raw    = load_master()
    df, _  = preprocess(raw)
    print(f"  Dataset: {len(df):,} rows | {df['Date'].min().date()} -> {df['Date'].max().date()}")

    # ── Portfolio simulation ───────────────────────────────────────────────────
    section("STEP 1 — Buy-and-Hold Portfolio Simulation")
    port = run_buy_and_hold(df)

    final_prices = {
        "Equity": df.iloc[-1]["Equity_Price"],
        "Gold"  : df.iloc[-1]["MA_Gold_Price"],
        "Oil"   : df.iloc[-1]["Oil_Price"],
    }
    print(port.summary(prices=final_prices))

    nav_df    = port.nav_history
    port_ret  = port.get_returns()

    # ── Risk Metrics ──────────────────────────────────────────────────────────
    section("STEP 2 — Risk Metrics")

    equity_ret   = df["Equity_Returns_clean"]
    oil_ret      = df["Oil_Returns_clean"]
    gold_ret     = df["MA_Gold_Returns_clean"]

    # Portfolio risk (using NAV returns)
    port_metrics = compute_all_risk_metrics(
        returns           = port_ret,
        benchmark_returns = equity_ret,   # equity index as benchmark
        dates             = nav_df["date"],
        risk_free_annual  = 0.02,
    )
    print("\n  [Buy-and-Hold Portfolio]")
    print(port_metrics.summary())

    # Raw equity risk (as baseline)
    eq_metrics = compute_all_risk_metrics(
        returns          = equity_ret,
        dates            = df["Date"],
        risk_free_annual = 0.02,
    )
    print("\n  [Raw Equity Benchmark]")
    print(eq_metrics.summary())

    # ── Multi-asset Risk Table ────────────────────────────────────────────────
    section("STEP 3 — Multi-Asset Risk Comparison")

    assets = {
        "Portfolio" : port_ret,
        "Equity"    : equity_ret,
        "Oil"       : oil_ret,
        "Gold"      : gold_ret,
    }

    print(f"\n  {'Asset':<12} {'Ann.Ret':>9} {'Ann.Vol':>9} {'Sharpe':>8} "
          f"{'Sortino':>9} {'Calmar':>8} {'VaR95%':>8} {'MaxDD':>9}")
    print(f"  {'-' * 72}")

    for name, ret in assets.items():
        m = compute_all_risk_metrics(ret, risk_free_annual=0.02)
        print(
            f"  {name:<12} {m.ann_return:>9.3%} {m.ann_volatility:>9.3%} "
            f"{m.sharpe_ratio:>8.4f} {m.sortino_ratio:>9.4f} "
            f"{m.calmar_ratio:>8.4f} {m.var_95.historical:>8.3%} "
            f"{m.drawdown.max_drawdown:>9.3%}"
        )

    # ── Trade Log Audit ───────────────────────────────────────────────────────
    section("STEP 4 — Trade Log (Audit Trail)")
    tlog = port.trade_log
    print(f"\n  Total trades executed: {len(tlog)}")
    print(tlog.to_string(index=False))

    # ── Charts ────────────────────────────────────────────────────────────────
    section("STEP 5 — Generating Phase 2 Charts")
    _save_charts(df, nav_df, port_ret, equity_ret)

    # ── Done ──────────────────────────────────────────────────────────────────
    section("PHASE 2 COMPLETE")
    print("  Portfolio Manager : OK")
    print("  Risk Metrics      : OK")
    print("  VaR / CVaR        : OK")
    print("  Drawdown          : OK")
    print("  Alpha / Beta      : OK")
    print("  Trade Audit Log   : OK")
    print("  Charts saved      : reports/")
    print("\n  Next -> Phase 3: Signal Engine (buy/sell/hold)")


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _save_charts(df, nav_df, port_ret, equity_ret):
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor"  : "#1a1d2e",
        "axes.edgecolor"  : "#3a3d4e",
        "text.color"      : "white",
        "axes.labelcolor" : "white",
        "xtick.color"     : "white",
        "ytick.color"     : "white",
        "grid.color"      : "#2a2d3e",
        "grid.linestyle"  : "--",
        "grid.alpha"      : 0.5,
    })

    # ── Chart 8: NAV over time ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    cum_eq = (1 + equity_ret).cumprod() * 100_000
    axes[0].plot(nav_df["date"], nav_df["nav"],  color="#00d4ff", lw=1.5, label="Portfolio NAV")
    axes[0].plot(df["Date"],     cum_eq.values,  color="#ff6b35", lw=1.0, alpha=0.7, linestyle="--", label="Equity Buy&Hold ($100K)")
    axes[0].axhline(100_000, color="white", lw=0.5, linestyle=":", alpha=0.4, label="Initial Capital")
    axes[0].set_title("Portfolio NAV vs Equity Benchmark", fontsize=13)
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True)

    # Daily returns
    axes[1].bar(nav_df["date"], nav_df["returns"], color=nav_df["returns"].apply(
        lambda x: "#10b981" if x >= 0 else "#f87171"), alpha=0.7, width=1)
    axes[1].axhline(0, color="white", lw=0.5)
    axes[1].set_title("Daily Portfolio Returns", fontsize=13)
    axes[1].set_ylabel("Return")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "08_portfolio_nav.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_portfolio_nav.png")

    # ── Chart 9: Drawdown comparison ──
    fig, ax = plt.subplots(figsize=(16, 6))

    # Portfolio drawdown
    cum_p  = (1 + port_ret).cumprod()
    dd_p   = (cum_p - cum_p.cummax()) / cum_p.cummax()

    # Equity drawdown
    cum_e  = (1 + equity_ret).cumprod()
    dd_e   = (cum_e - cum_e.cummax()) / cum_e.cummax()

    ax.fill_between(nav_df["date"], dd_p.values,           color="#7c3aed", alpha=0.6, label="Portfolio DD")
    ax.fill_between(df["Date"],     dd_e.values, color="#f87171", alpha=0.4, label="Equity DD")
    ax.axhline(0, color="white", lw=0.5)
    ax.set_title("Drawdown: Portfolio vs Equity", fontsize=13)
    ax.set_ylabel("Drawdown %")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "09_drawdown_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 09_drawdown_comparison.png")

    # ── Chart 10: Rolling Risk Metrics ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    roll_vol   = rolling_volatility(equity_ret, window=20)
    roll_sh    = rolling_sharpe(equity_ret, window=252)
    roll_var95 = rolling_var(equity_ret, window=252, confidence=0.95)

    axes[0].plot(df["Date"], roll_vol,   color="#a78bfa", lw=1)
    axes[0].set_title("Rolling 20-day Annualised Volatility", fontsize=11)
    axes[0].set_ylabel("Volatility")
    axes[0].grid(True)

    axes[1].plot(df["Date"], roll_sh, color="#34d399", lw=1)
    axes[1].axhline(0, color="white", lw=0.5, linestyle="--")
    axes[1].axhline(1, color="#10b981", lw=0.5, linestyle=":", alpha=0.6, label="Sharpe=1 target")
    axes[1].set_title("Rolling 252-day Sharpe Ratio", fontsize=11)
    axes[1].set_ylabel("Sharpe")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    axes[2].fill_between(df["Date"], roll_var95, 0, color="#f87171", alpha=0.7)
    axes[2].set_title("Rolling 252-day VaR (95%)", fontsize=11)
    axes[2].set_ylabel("Daily VaR")
    axes[2].grid(True)

    plt.suptitle("Rolling Risk Metrics — Equity", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "10_rolling_risk.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 10_rolling_risk.png")

    # ── Chart 11: Return Distribution with VaR lines ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (label, ret) in zip(axes, [("Portfolio", port_ret), ("Equity", equity_ret)]):
        ax.hist(ret.dropna(), bins=100, color="#00d4ff", alpha=0.7, edgecolor="none", density=True)
        v95 = ret.quantile(0.05)
        v99 = ret.quantile(0.01)
        ax.axvline(v95, color="orange", lw=2, label=f"VaR 95%: {v95:.3%}")
        ax.axvline(v99, color="red",    lw=2, label=f"VaR 99%: {v99:.3%}")
        ax.axvline(ret.mean(), color="#10b981", lw=1.5, linestyle="--", label=f"Mean: {ret.mean():.4%}")
        ax.set_title(f"{label} Return Distribution", fontsize=12)
        ax.set_xlabel("Daily Return")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "11_return_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 11_return_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()
