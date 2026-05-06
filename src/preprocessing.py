"""
preprocessing.py
----------------
Cleans the master DataFrame and engineers features needed for
risk modelling and signal generation.

Steps:
    1. Drop / impute warmup-period NaNs
    2. Winsorize extreme returns  (+/-3 sigma)
    3. Normalize macro columns (Z-score)
    4. Engineer rolling features:
       - Equity rolling volatility (20-day)
       - Equity momentum          (10-day cumulative return)
       - Macro rate-of-change
       - Cross-asset rolling correlation (equity vs gold/oil)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── 1. NaN Handling ───────────────────────────────────────────────────────────

def drop_warmup_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the first rows that have NaN due to rolling-window warmup.
    The largest warmup window is 10 days (SMA_10, Oil_Volatility).
    """
    before = len(df)
    df = df.dropna(subset=["Equity_SMA10", "Oil_Volatility"]).copy()
    after = len(df)
    print(f"[preprocess] Dropped {before - after} warmup rows  ({before} -> {after})")
    return df.reset_index(drop=True)


def impute_remaining_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill any remaining NaNs (e.g. mid-series gaps).
    Returns columns filled are zero for returns, ffill for prices.
    """
    return_cols  = [c for c in df.columns if "Returns" in c]
    price_cols   = [c for c in df.columns if "Price" in c or "SMA" in c
                    or "Bonds" in c or "Index" in c]

    df[return_cols] = df[return_cols].fillna(0)
    df[price_cols]  = df[price_cols].ffill()
    df = df.ffill()  # catch anything remaining
    return df


# ── 2. Outlier Handling ───────────────────────────────────────────────────────

def winsorize_returns(df: pd.DataFrame, sigma: float = 3.0) -> pd.DataFrame:
    """
    Clip extreme daily returns to +/- `sigma` standard deviations.
    Adds  _clean  suffix columns so originals are preserved.
    """
    return_cols = [c for c in df.columns if "Returns" in c]
    for col in return_cols:
        series = df[col].dropna()
        lo = series.mean() - sigma * series.std()
        hi = series.mean() + sigma * series.std()
        df[f"{col}_clean"] = df[col].clip(lo, hi)
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_clipped:
            print(f"[preprocess] Winsorized {n_clipped:>4} rows in {col}")
    return df


# ── 3. Normalisation ──────────────────────────────────────────────────────────

MACRO_COLS = ["Inflation", "Interest_Rate", "USD_Index", "Sentiment"]

def normalize_macro(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Z-score normalize macro columns.
    Returns the modified DataFrame AND the fitted scaler (for inverse-transform later).
    """
    scaler = StandardScaler()
    df[[f"{c}_norm" for c in MACRO_COLS]] = scaler.fit_transform(df[MACRO_COLS])
    print(f"[preprocess] Normalized {MACRO_COLS}")
    return df, scaler


# ── 4. Feature Engineering ────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features on top of the raw data:

    Equity:
        - Equity_RollingVol20   : 20-day rolling std of returns
        - Equity_Momentum10     : 10-day cumulative return
        - Equity_Momentum30     : 30-day cumulative return

    Macro:
        - Rate_of_Change columns for Inflation & Interest_Rate

    Cross-asset:
        - Corr_Equity_Gold30    : 30-day rolling correlation
        - Corr_Equity_Oil30     : 30-day rolling correlation
    """
    r = "Equity_Returns_clean"

    # -- Equity volatility --
    df["Equity_RollingVol20"] = df[r].rolling(20).std() * np.sqrt(252)  # annualised

    # -- Momentum (sum of returns over window) --
    df["Equity_Momentum10"] = df[r].rolling(10).sum()
    df["Equity_Momentum30"] = df[r].rolling(30).sum()

    # -- Macro rate of change --
    df["Inflation_RoC"]      = df["Inflation"].pct_change()
    df["InterestRate_RoC"]   = df["Interest_Rate"].pct_change()

    # -- Cross-asset correlations --
    df["Corr_Equity_Gold30"] = (
        df["Equity_Returns_clean"]
        .rolling(30)
        .corr(df["MA_Gold_Returns_clean"])
    )
    df["Corr_Equity_Oil30"] = (
        df["Equity_Returns_clean"]
        .rolling(30)
        .corr(df["Oil_Returns_clean"])
    )

    # Drop the short rolling-window warmup NaNs from new features
    df = df.dropna(subset=["Equity_RollingVol20", "Equity_Momentum30",
                            "Corr_Equity_Gold30"]).reset_index(drop=True)

    print(f"[preprocess] Feature engineering done. Shape: {df.shape}")
    return df


# ── Master Preprocess Pipeline ────────────────────────────────────────────────

def preprocess(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Full pipeline:
        raw_df  ->  cleaned + normalised + feature-engineered DataFrame

    Returns
    -------
    df     : processed DataFrame
    scaler : fitted StandardScaler (for macro cols)
    """
    df = drop_warmup_nans(raw_df)
    df = impute_remaining_nans(df)
    df = winsorize_returns(df)
    df, scaler = normalize_macro(df)
    df = add_rolling_features(df)
    return df, scaler


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from ingestion import load_master
    raw = load_master()

    df, scaler = preprocess(raw)

    print(f"\nFinal dataset shape : {df.shape}")
    print(f"Date range          : {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"\nNew feature columns :")
    new_cols = [c for c in df.columns if any(x in c for x in
                ["Vol", "Momentum", "RoC", "Corr", "clean", "norm"])]
    for c in new_cols:
        print(f"  {c}")

    remaining_nans = df.isnull().sum().sum()
    print(f"\nRemaining NaNs: {remaining_nans}")
