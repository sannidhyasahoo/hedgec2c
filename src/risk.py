"""
risk.py
-------
Risk metrics for the hedge fund system.

Covers Issues 6, 7, 12, 13 from ISSUES.md:
    - Issue  6 : Value at Risk (VaR)  -- Historical + Parametric + CVaR
    - Issue  7 : Max Drawdown & Portfolio Volatility
    - Issue 12 : Sharpe Ratio (risk-adjusted return)
    - Issue 13 : Alpha & Beta vs a benchmark

All functions operate on a pandas Series of daily returns
(or a DataFrame for multi-asset calculations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VaRResult:
    """Value at Risk and Conditional VaR at a given confidence level."""
    confidence   : float          # e.g. 0.95 for 95%
    historical   : float          # historical simulation VaR
    parametric   : float          # Gaussian parametric VaR
    cvar         : float          # Conditional VaR (Expected Shortfall)
    period_days  : int            # number of daily return observations used

    def __str__(self) -> str:
        return (
            f"VaR ({self.confidence:.0%} conf, {self.period_days}d)\n"
            f"  Historical : {self.historical:.4%}\n"
            f"  Parametric : {self.parametric:.4%}\n"
            f"  CVaR (ES)  : {self.cvar:.4%}"
        )


@dataclass
class DrawdownResult:
    """Drawdown statistics for a returns series."""
    max_drawdown       : float   # worst peak-to-trough drop
    max_dd_start       : object  # date when drawdown started
    max_dd_end         : object  # date of trough
    avg_drawdown       : float   # average of all drawdown periods
    recovery_days      : int     # calendar days to recover from worst drawdown
    drawdown_series    : pd.Series = field(repr=False)   # full daily drawdown values

    def __str__(self) -> str:
        return (
            f"Drawdown Analysis\n"
            f"  Max Drawdown  : {self.max_drawdown:.4%}\n"
            f"  DD Start      : {self.max_dd_start}\n"
            f"  DD Trough     : {self.max_dd_end}\n"
            f"  Recovery Days : {self.recovery_days}\n"
            f"  Avg Drawdown  : {self.avg_drawdown:.4%}"
        )


@dataclass
class RiskMetrics:
    """Complete risk/return profile for a returns series."""
    # Return metrics
    total_return       : float
    ann_return         : float
    ann_volatility     : float

    # Risk-adjusted
    sharpe_ratio       : float
    sortino_ratio      : float
    calmar_ratio       : float

    # Risk
    var_95             : VaRResult
    var_99             : VaRResult
    drawdown           : DrawdownResult

    # Alpha / Beta vs benchmark
    alpha              : Optional[float] = None
    beta               : Optional[float] = None
    r_squared          : Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  RISK METRICS SUMMARY",
            "=" * 50,
            f"  Total Return     : {self.total_return:.4%}",
            f"  Annualised Ret   : {self.ann_return:.4%}",
            f"  Annualised Vol   : {self.ann_volatility:.4%}",
            f"  Sharpe Ratio     : {self.sharpe_ratio:.4f}",
            f"  Sortino Ratio    : {self.sortino_ratio:.4f}",
            f"  Calmar Ratio     : {self.calmar_ratio:.4f}",
            f"  VaR 95% (hist)   : {self.var_95.historical:.4%}",
            f"  VaR 99% (hist)   : {self.var_99.historical:.4%}",
            f"  CVaR 95%         : {self.var_95.cvar:.4%}",
            f"  Max Drawdown     : {self.drawdown.max_drawdown:.4%}",
        ]
        if self.alpha is not None:
            lines += [
                f"  Alpha            : {self.alpha:.6f}",
                f"  Beta             : {self.beta:.4f}",
                f"  R-squared        : {self.r_squared:.4f}",
            ]
        lines.append("=" * 50)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Issue 6: Value at Risk
# ─────────────────────────────────────────────────────────────────────────────

def calculate_var(
    returns  : pd.Series,
    confidence: float = 0.95,
    dates    : Optional[pd.Series] = None,
) -> VaRResult:
    """
    Compute VaR and CVaR using two methods:

    Historical Simulation
        Sort the observed returns and take the appropriate quantile.
        No distributional assumption — uses actual fat tails.

    Parametric (Gaussian)
        VaR = mu - z * sigma  where z is the inverse-normal quantile.
        Fast but underestimates tail risk for fat-tailed distributions.

    CVaR / Expected Shortfall
        Average of all returns that breach the VaR threshold.
        A coherent risk measure — always used alongside VaR.

    Parameters
    ----------
    returns    : daily returns Series (already cleaned / winsorized)
    confidence : e.g. 0.95 or 0.99
    dates      : optional Date Series (same index as returns)

    Returns
    -------
    VaRResult dataclass
    """
    r = returns.dropna()
    alpha = 1.0 - confidence            # left-tail probability

    # -- Historical --
    hist_var = float(r.quantile(alpha))

    # -- Parametric --
    from scipy.stats import norm
    mu, sigma = r.mean(), r.std()
    param_var = float(mu + norm.ppf(alpha) * sigma)

    # -- CVaR (Expected Shortfall) --
    cvar = float(r[r <= hist_var].mean())

    return VaRResult(
        confidence  = confidence,
        historical  = hist_var,
        parametric  = param_var,
        cvar        = cvar,
        period_days = len(r),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue 7: Drawdown & Volatility
# ─────────────────────────────────────────────────────────────────────────────

def calculate_drawdown(
    returns : pd.Series,
    dates   : Optional[pd.Series] = None,
) -> DrawdownResult:
    """
    Compute the drawdown series and extract key drawdown statistics.

    The drawdown at time t is:
        DD(t) = (NAV(t) - peak NAV up to t) / peak NAV up to t

    Parameters
    ----------
    returns : daily returns Series
    dates   : optional Date Series for labelling start/end

    Returns
    -------
    DrawdownResult dataclass
    """
    r = returns.dropna().reset_index(drop=True)
    idx = dates.reset_index(drop=True) if dates is not None else pd.RangeIndex(len(r))

    # Net Asset Value (NAV) from $1
    nav     = (1 + r).cumprod()
    peak    = nav.cummax()
    dd_ser  = (nav - peak) / peak

    max_dd_loc  = dd_ser.idxmin()
    max_dd_val  = float(dd_ser.iloc[max_dd_loc])

    # Find the start of the worst drawdown (last time NAV was at peak before trough)
    peak_before_trough = nav.iloc[:max_dd_loc + 1].idxmax()

    # Recovery: first time NAV >= peak after trough
    nav_after = nav.iloc[max_dd_loc:]
    peak_val  = float(peak.iloc[max_dd_loc])
    recovery_mask = nav_after >= peak_val
    recovery_loc  = recovery_mask.idxmax() if recovery_mask.any() else None

    if recovery_loc is not None and recovery_mask.any():
        rec_days = int(recovery_loc - max_dd_loc)
    else:
        rec_days = -1   # still in drawdown at end of dataset

    # Average drawdown (only negative periods)
    avg_dd = float(dd_ser[dd_ser < 0].mean()) if (dd_ser < 0).any() else 0.0

    return DrawdownResult(
        max_drawdown    = max_dd_val,
        max_dd_start    = idx.iloc[peak_before_trough] if hasattr(idx, 'iloc') else peak_before_trough,
        max_dd_end      = idx.iloc[max_dd_loc]         if hasattr(idx, 'iloc') else max_dd_loc,
        avg_drawdown    = avg_dd,
        recovery_days   = rec_days,
        drawdown_series = dd_ser,
    )


def portfolio_volatility(returns: pd.Series, ann_factor: int = 252) -> float:
    """
    Annualised portfolio volatility = daily_std * sqrt(ann_factor).

    ann_factor = 252  for daily returns (trading days per year)
    """
    return float(returns.dropna().std() * np.sqrt(ann_factor))


# ─────────────────────────────────────────────────────────────────────────────
# Issue 12: Sharpe & Sortino & Calmar
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(
    returns    : pd.Series,
    risk_free  : float = 0.0,
    ann_factor : int   = 252,
) -> float:
    """
    Annualised Sharpe Ratio.

    Sharpe = (E[R] - Rf) / sigma(R)  * sqrt(ann_factor)

    Parameters
    ----------
    returns    : daily returns
    risk_free  : daily risk-free rate (default 0 %)
    ann_factor : trading days per year
    """
    r       = returns.dropna()
    excess  = r - risk_free
    vol     = excess.std()
    if vol == 0:
        return 0.0
    return float((excess.mean() / vol) * np.sqrt(ann_factor))


def sortino_ratio(
    returns    : pd.Series,
    risk_free  : float = 0.0,
    ann_factor : int   = 252,
) -> float:
    """
    Annualised Sortino Ratio.

    Like Sharpe but only penalises DOWNSIDE volatility — better for
    strategies with skewed return distributions.

    Sortino = (E[R] - Rf) / downside_std  * sqrt(ann_factor)
    """
    r           = returns.dropna()
    excess      = r - risk_free
    downside    = excess[excess < 0]
    down_vol    = downside.std() if len(downside) > 1 else 1e-9
    return float((excess.mean() / down_vol) * np.sqrt(ann_factor))


def calmar_ratio(
    returns    : pd.Series,
    ann_factor : int = 252,
) -> float:
    """
    Calmar Ratio = Annualised Return / |Max Drawdown|

    Good for evaluating strategies that may have infrequent but
    severe drawdowns (e.g. trend-following).
    """
    ann_ret = returns.dropna().mean() * ann_factor
    dd      = calculate_drawdown(returns)
    if dd.max_drawdown == 0:
        return 0.0
    return float(ann_ret / abs(dd.max_drawdown))


# ─────────────────────────────────────────────────────────────────────────────
# Issue 13: Alpha & Beta
# ─────────────────────────────────────────────────────────────────────────────

def alpha_beta(
    portfolio_returns  : pd.Series,
    benchmark_returns  : pd.Series,
    risk_free          : float = 0.0,
    ann_factor         : int   = 252,
) -> tuple[float, float, float]:
    """
    Compute CAPM Alpha, Beta, and R-squared.

    Model:
        R_p - Rf = alpha + beta * (R_b - Rf) + epsilon

    Parameters
    ----------
    portfolio_returns : daily portfolio returns
    benchmark_returns : daily benchmark returns (e.g. equity index)
    risk_free         : daily risk-free rate
    ann_factor        : for annualising alpha

    Returns
    -------
    (alpha, beta, r_squared)

    Interpretation
    --------------
    beta  > 1  : portfolio moves more than the market
    beta  < 1  : portfolio is less volatile than the market
    beta  < 0  : portfolio moves inversely to the market
    alpha > 0  : positive excess return above market risk-adjusted expectation
    """
    # Align and drop NaN
    df = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    ep = df["p"] - risk_free
    eb = df["b"] - risk_free

    # OLS: ep = alpha + beta * eb
    cov_matrix = np.cov(ep, eb)
    beta_val   = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] != 0 else 0.0
    alpha_daily = float(ep.mean() - beta_val * eb.mean())
    alpha_ann   = alpha_daily * ann_factor    # annualised

    # R-squared
    corr = float(ep.corr(eb))
    r_sq = corr ** 2

    return alpha_ann, beta_val, r_sq


# ─────────────────────────────────────────────────────────────────────────────
# Master: compute all risk metrics in one call
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_risk_metrics(
    returns           : pd.Series,
    benchmark_returns : Optional[pd.Series] = None,
    dates             : Optional[pd.Series] = None,
    risk_free_annual  : float = 0.02,
    ann_factor        : int   = 252,
) -> RiskMetrics:
    """
    Convenience wrapper: computes every risk metric in one call.

    Parameters
    ----------
    returns           : daily portfolio returns (winsorized / cleaned)
    benchmark_returns : daily benchmark returns for alpha/beta (optional)
    dates             : Date series aligned with returns
    risk_free_annual  : annual risk-free rate (default 2%)
    ann_factor        : trading days per year (252)

    Returns
    -------
    RiskMetrics dataclass with .summary() method
    """
    rf_daily = risk_free_annual / ann_factor

    r = returns.dropna()

    # Returns
    total_ret = float((1 + r).prod() - 1)
    ann_ret   = float(r.mean() * ann_factor)
    ann_vol   = portfolio_volatility(r, ann_factor)

    # Risk-adjusted
    sh  = sharpe_ratio (r, rf_daily, ann_factor)
    so  = sortino_ratio(r, rf_daily, ann_factor)
    cal = calmar_ratio (r, ann_factor)

    # VaR
    v95 = calculate_var(r, 0.95, dates)
    v99 = calculate_var(r, 0.99, dates)

    # Drawdown
    dd  = calculate_drawdown(r, dates)

    # Alpha / Beta
    a, b, r2 = None, None, None
    if benchmark_returns is not None:
        a, b, r2 = alpha_beta(r, benchmark_returns, rf_daily, ann_factor)

    return RiskMetrics(
        total_return   = total_ret,
        ann_return     = ann_ret,
        ann_volatility = ann_vol,
        sharpe_ratio   = sh,
        sortino_ratio  = so,
        calmar_ratio   = cal,
        var_95         = v95,
        var_99         = v99,
        drawdown       = dd,
        alpha          = a,
        beta           = b,
        r_squared      = r2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rolling risk  (for dashboard / live monitoring)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_var(
    returns    : pd.Series,
    window     : int   = 252,
    confidence : float = 0.95,
) -> pd.Series:
    """Rolling Historical VaR over a sliding window."""
    alpha = 1.0 - confidence
    return returns.rolling(window).quantile(alpha)


def rolling_sharpe(
    returns    : pd.Series,
    window     : int   = 252,
    risk_free  : float = 0.0,
    ann_factor : int   = 252,
) -> pd.Series:
    """Rolling annualised Sharpe Ratio."""
    excess = returns - risk_free
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(ann_factor)


def rolling_volatility(
    returns    : pd.Series,
    window     : int = 20,
    ann_factor : int = 252,
) -> pd.Series:
    """Rolling annualised volatility."""
    return returns.rolling(window).std() * np.sqrt(ann_factor)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from ingestion     import load_master
    from preprocessing import preprocess

    raw = load_master()
    df, _  = preprocess(raw)

    ret       = df["Equity_Returns_clean"]
    benchmark = df["Oil_Returns_clean"]      # use oil as benchmark proxy
    dates     = df["Date"]

    metrics = compute_all_risk_metrics(
        returns           = ret,
        benchmark_returns = benchmark,
        dates             = dates,
        risk_free_annual  = 0.02,
    )

    print(metrics.summary())
    print()
    print(metrics.var_95)
    print()
    print(metrics.drawdown)
