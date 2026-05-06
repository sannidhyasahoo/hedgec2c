"""
ingestion.py
------------
Loads, validates, and merges all 4 raw datasets into a single master DataFrame.

Datasets:
    - equity_dataset.csv     : Daily equity price, volume, returns, SMA_10
    - macro_dataset.csv      : Inflation, Interest_Rate, USD_Index, Sentiment
    - multi_asset_dataset.csv: Oil, Gold, Bonds prices + returns
    - oil_dataset.csv        : Oil price, volume, returns, volatility
"""

import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

EQUITY_PATH       = DATA_DIR / "equity_dataset.csv"
MACRO_PATH        = DATA_DIR / "macro_dataset.csv"
MULTI_ASSET_PATH  = DATA_DIR / "multi_asset_dataset.csv"
OIL_PATH          = DATA_DIR / "oil_dataset.csv"


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_equity() -> pd.DataFrame:
    """Load equity dataset. Returns: Date, Price, Volume, Returns, SMA_10"""
    df = pd.read_csv(EQUITY_PATH, parse_dates=["Date"])
    df = df.rename(columns={
        "Price":   "Equity_Price",
        "Volume":  "Equity_Volume",
        "Returns": "Equity_Returns",
        "SMA_10":  "Equity_SMA10"
    })
    return df.sort_values("Date").reset_index(drop=True)


def load_macro() -> pd.DataFrame:
    """Load macro dataset. Returns: Date, Inflation, Interest_Rate, USD_Index, Sentiment"""
    df = pd.read_csv(MACRO_PATH, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_multi_asset() -> pd.DataFrame:
    """Load multi-asset dataset. Returns: Date, Oil, Gold, Bonds, Oil_Returns, Gold_Returns"""
    df = pd.read_csv(MULTI_ASSET_PATH, parse_dates=["Date"])
    df = df.rename(columns={
        "Oil":         "MA_Oil_Price",
        "Gold":        "MA_Gold_Price",
        "Bonds":       "MA_Bonds_Price",
        "Oil_Returns": "MA_Oil_Returns",
        "Gold_Returns":"MA_Gold_Returns",
    })
    return df.sort_values("Date").reset_index(drop=True)


def load_oil() -> pd.DataFrame:
    """Load oil-specific dataset. Returns: Date, Price, Volume, Returns, Volatility"""
    df = pd.read_csv(OIL_PATH, parse_dates=["Date"])
    df = df.rename(columns={
        "Price":      "Oil_Price",
        "Volume":     "Oil_Volume",
        "Returns":    "Oil_Returns",
        "Volatility": "Oil_Volatility"
    })
    return df.sort_values("Date").reset_index(drop=True)


# ── Master Merge ──────────────────────────────────────────────────────────────

def load_master() -> pd.DataFrame:
    """
    Merge all 4 datasets on Date into a single master DataFrame.
    All datasets share the same daily cadence so a simple inner join is safe.

    Returns
    -------
    pd.DataFrame
        Master DataFrame with all features aligned by Date.
    """
    equity      = load_equity()
    macro       = load_macro()
    multi_asset = load_multi_asset()
    oil         = load_oil()

    # Drop redundant oil columns from multi_asset (we have them in oil_dataset)
    multi_asset_cols = ["Date", "MA_Gold_Price", "MA_Bonds_Price", "MA_Gold_Returns"]

    master = (
        equity
        .merge(macro,                  on="Date", how="inner")
        .merge(multi_asset[multi_asset_cols], on="Date", how="inner")
        .merge(oil[["Date", "Oil_Price", "Oil_Volume", "Oil_Returns", "Oil_Volatility"]],
               on="Date", how="inner")
    )

    print(f"[ingestion] Master shape : {master.shape}")
    print(f"[ingestion] Date range   : {master['Date'].min().date()} -> {master['Date'].max().date()}")
    print(f"[ingestion] Columns      : {list(master.columns)}")

    return master


# ── Quick Validation ──────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """Print a quick health check of the master DataFrame."""
    print("\n-- NaN counts --")
    nan_counts = df.isnull().sum()
    print(nan_counts[nan_counts > 0].to_string() if nan_counts.any() else "No NaNs found OK")

    print("\n-- Duplicate dates --")
    dupes = df['Date'].duplicated().sum()
    print(f"{dupes} duplicate date rows" if dupes else "No duplicates OK")

    print("\n-- Basic stats --")
    print(df.describe().round(4).to_string())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    master = load_master()
    validate(master)
