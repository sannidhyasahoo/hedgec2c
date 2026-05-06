"""
evaluate_phase1.py
------------------
Phase 1 evaluation script.

Runs the full ingestion + preprocessing pipeline and prints a
comprehensive data quality & feature report to the console.

Usage:
    python evaluate_phase1.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ── ensure src/ is on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion    import load_master
from preprocessing import preprocess

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


def subsection(title: str):
    print(f"\n--- {title} ---")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate():

    # ── STEP 1: LOAD ──────────────────────────────────────────────────────────
    section("STEP 1 — Data Ingestion")
    raw = load_master()

    subsection("Raw dataset overview")
    print(f"  Rows            : {len(raw):,}")
    print(f"  Columns         : {len(raw.columns)}")
    print(f"  Date range      : {raw['Date'].min().date()} -> {raw['Date'].max().date()}")
    print(f"  Years covered   : {raw['Date'].dt.year.nunique()}")
    print(f"  NaN totals      :")
    for col, n in raw.isnull().sum().items():
        if n:
            print(f"    {col:<30} : {n}")

    # ── STEP 2: PREPROCESS ────────────────────────────────────────────────────
    section("STEP 2 — Preprocessing")
    df, scaler = preprocess(raw)

    subsection("Post-processing overview")
    print(f"  Rows after clean : {len(df):,}")
    print(f"  Total NaNs       : {df.isnull().sum().sum()}")
    print(f"  Columns          : {len(df.columns)}")

    # ── STEP 3: EQUITY RETURN STATS ───────────────────────────────────────────
    section("STEP 3 — Equity Return Statistics")
    ret = df["Equity_Returns_clean"]

    ann_ret = ret.mean() * 252
    ann_vol = ret.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol else 0
    skew    = ret.skew()
    kurt    = ret.kurtosis()
    var_95  = ret.quantile(0.05)
    var_99  = ret.quantile(0.01)

    # Max drawdown
    cum = (1 + ret).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd   = drawdown.min()

    print(f"\n  Annualised Return : {ann_ret:>10.4%}")
    print(f"  Annualised Vol    : {ann_vol:>10.4%}")
    print(f"  Sharpe Ratio      : {sharpe:>10.4f}")
    print(f"  Skewness          : {skew:>10.4f}")
    print(f"  Excess Kurtosis   : {kurt:>10.4f}")
    print(f"  VaR 95%  (daily)  : {var_95:>10.4%}")
    print(f"  VaR 99%  (daily)  : {var_99:>10.4%}")
    print(f"  Max Drawdown      : {max_dd:>10.4%}")
    print(f"  Total +ve days    : {(ret > 0).sum():>10,} / {len(ret):,}")

    # ── STEP 4: MULTI-ASSET STATS ─────────────────────────────────────────────
    section("STEP 4 — Multi-Asset Summary")
    asset_map = {
        "Equity"   : "Equity_Returns_clean",
        "Oil"      : "Oil_Returns_clean",
        "Gold"     : "MA_Gold_Returns_clean",
    }

    print(f"\n  {'Asset':<10} {'Ann.Ret':>10} {'Ann.Vol':>10} {'Sharpe':>10} {'VaR95%':>10}")
    print(f"  {'-'*52}")
    for name, col in asset_map.items():
        r_  = df[col]
        ar  = r_.mean() * 252
        av  = r_.std()  * np.sqrt(252)
        sh  = ar / av if av else 0
        v95 = r_.quantile(0.05)
        print(f"  {name:<10} {ar:>10.4%} {av:>10.4%} {sh:>10.4f} {v95:>10.4%}")

    # ── STEP 5: FEATURE QUALITY ───────────────────────────────────────────────
    section("STEP 5 — Engineered Features")
    feat_cols = [c for c in df.columns if any(x in c for x in
                 ["Vol", "Momentum", "RoC", "Corr", "norm"])]

    print(f"\n  {'Feature':<35} {'Mean':>10} {'Std':>10} {'NaN':>6}")
    print(f"  {'-'*65}")
    for c in feat_cols:
        print(f"  {c:<35} {df[c].mean():>10.4f} {df[c].std():>10.4f} {df[c].isna().sum():>6}")

    # ── STEP 6: CORRELATION MATRIX ────────────────────────────────────────────
    section("STEP 6 — Return Correlations")
    corr_cols = [
        "Equity_Returns_clean", "Oil_Returns_clean",
        "MA_Gold_Returns_clean",
        "Inflation_norm", "Interest_Rate_norm",
        "USD_Index_norm",  "Sentiment_norm"
    ]
    corr = df[corr_cols].corr().round(3)

    label_map = {
        "Equity_Returns_clean"  : "Equity",
        "Oil_Returns_clean"     : "Oil",
        "MA_Gold_Returns_clean" : "Gold",
        "Inflation_norm"        : "Inflation",
        "Interest_Rate_norm"    : "IntRate",
        "USD_Index_norm"        : "USD",
        "Sentiment_norm"        : "Sentiment",
    }
    corr.rename(columns=label_map, index=label_map, inplace=True)
    print(f"\n{corr.to_string()}")

    # ── STEP 7: CHARTS ────────────────────────────────────────────────────────
    section("STEP 7 — Generating Charts -> reports/")
    _save_charts(df)

    # ── DONE ──────────────────────────────────────────────────────────────────
    section("PHASE 1 COMPLETE")
    print("  Ingestion    : OK")
    print("  Preprocessing: OK")
    print("  Features     : OK")
    print("  Charts saved : reports/")
    print("\n  Next -> Phase 2: Risk Model (VaR, Drawdown, Sharpe)")


# ─────────────────────────────────────────────────────────────────────────────
# Chart Generator
# ─────────────────────────────────────────────────────────────────────────────

def _save_charts(df: pd.DataFrame):
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

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

    # ── Chart 1: Equity Price + SMA ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(df["Date"], df["Equity_Price"],   color="#00d4ff", lw=0.8, label="Price")
    axes[0].plot(df["Date"], df["Equity_SMA10"],   color="#ff6b35", lw=1.0, alpha=0.8, label="SMA10")
    axes[0].set_title("Equity Price & SMA10", fontsize=13)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].bar(df["Date"], df["Equity_Volume"], color="#7c3aed", alpha=0.5, width=1)
    axes[1].set_title("Daily Volume", fontsize=13)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "01_equity_price.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_equity_price.png")

    # ── Chart 2: Cumulative Returns ──
    cum_equity = (1 + df["Equity_Returns_clean"]).cumprod()
    cum_oil    = (1 + df["Oil_Returns_clean"]).cumprod()
    cum_gold   = (1 + df["MA_Gold_Returns_clean"]).cumprod()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["Date"], cum_equity, color="#00d4ff",  lw=1.2, label="Equity")
    ax.plot(df["Date"], cum_oil,    color="#f59e0b",  lw=1.2, label="Oil")
    ax.plot(df["Date"], cum_gold,   color="#fbbf24",  lw=1.2, label="Gold")
    ax.axhline(1, color="white", lw=0.5, linestyle="--", alpha=0.4)
    ax.set_title("Cumulative Returns — Equity vs Oil vs Gold", fontsize=13)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "02_cumulative_returns.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 02_cumulative_returns.png")

    # ── Chart 3: Drawdown ──
    cum  = (1 + df["Equity_Returns_clean"]).cumprod()
    roll = cum.cummax()
    dd   = (cum - roll) / roll

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(df["Date"], dd, 0, color="#f87171", alpha=0.7)
    ax.set_title("Equity Drawdown", fontsize=13)
    ax.set_ylabel("Drawdown %")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "03_drawdown.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_drawdown.png")

    # ── Chart 4: Rolling Volatility ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["Date"], df["Equity_RollingVol20"], color="#a78bfa", lw=1, label="Equity Vol (20d)")
    ax.plot(df["Date"], df["Oil_Volatility"],       color="#f59e0b", lw=1, alpha=0.7, label="Oil Vol")
    ax.set_title("Rolling Volatility", fontsize=13)
    ax.set_ylabel("Annualised Volatility")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "04_rolling_volatility.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_rolling_volatility.png")

    # ── Chart 5: Return Distribution ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ret = df["Equity_Returns_clean"]
    ax.hist(ret, bins=120, color="#00d4ff", alpha=0.8, edgecolor="none")
    ax.axvline(ret.quantile(0.05), color="red",   lw=2, linestyle="--", label=f"VaR 95%: {ret.quantile(0.05):.3%}")
    ax.axvline(ret.quantile(0.01), color="#f87171", lw=2, linestyle="--", label=f"VaR 99%: {ret.quantile(0.01):.3%}")
    ax.axvline(ret.mean(),         color="#10b981", lw=2, label=f"Mean: {ret.mean():.4%}")
    ax.set_title("Equity Return Distribution", fontsize=13)
    ax.set_xlabel("Daily Return")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "05_return_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_return_distribution.png")

    # ── Chart 6: Macro Signals ──
    macro_cols  = ["Inflation", "Interest_Rate", "USD_Index", "Sentiment"]
    mac_colors  = ["#f87171", "#fb923c", "#a78bfa", "#34d399"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    for ax, col, color in zip(axes, macro_cols, mac_colors):
        ax.plot(df["Date"], df[col], color=color, lw=0.7, alpha=0.8)
        ax.plot(df["Date"], df[col].rolling(30).mean(),
                color="white", lw=1.2, linestyle="--", alpha=0.6, label="30d avg")
        ax.set_title(col, fontsize=11)
        ax.grid(True)
        ax.legend(fontsize=8)
    plt.suptitle("Macro Signals Over Time", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "06_macro_signals.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_macro_signals.png")

    # ── Chart 7: Correlation heatmap ──
    corr_cols = [
        "Equity_Returns_clean", "Oil_Returns_clean",
        "MA_Gold_Returns_clean",
        "Inflation", "Interest_Rate", "USD_Index", "Sentiment",
    ]
    label_map = {
        "Equity_Returns_clean"  : "Equity",
        "Oil_Returns_clean"     : "Oil",
        "MA_Gold_Returns_clean" : "Gold",
        "Inflation"             : "Inflation",
        "Interest_Rate"         : "IntRate",
        "USD_Index"             : "USD",
        "Sentiment"             : "Sentiment",
    }
    corr = df[corr_cols].corr()
    corr.rename(columns=label_map, index=label_map, inplace=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(corr.iloc[i,j]) > 0.4 else "black")
    ax.set_title("Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "07_correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()
