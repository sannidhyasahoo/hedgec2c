"""
portfolio.py
------------
Portfolio state manager for the hedge fund backtesting system.

Covers Issue 5 from ISSUES.md:
    - Track cash, positions (shares held), PnL per asset
    - Execute buy/sell orders with transaction costs and slippage (Issue 10)
    - Record a full trade log with timestamps (Issue 14 - audit trail)
    - Compute daily NAV (Net Asset Value) from positions + cash

Design Principles:
    - No forward-looking bias: prices are consumed one row at a time
    - All state mutations go through explicit methods (no direct dict edits)
    - Trade log is append-only — full audit trail preserved
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import date


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Immutable record of a single executed trade."""
    date        : object       # trade date
    asset       : str          # asset identifier
    action      : str          # "BUY" | "SELL"
    quantity    : float        # number of shares / units
    price       : float        # execution price (post-slippage)
    raw_price   : float        # price before slippage
    slippage    : float        # slippage cost (absolute $)
    commission  : float        # transaction cost (absolute $)
    cash_before : float
    cash_after  : float
    reason      : str = ""     # signal / rule that triggered the trade


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of portfolio state."""
    date        : object
    cash        : float
    positions   : dict          # {asset: shares}
    prices      : dict          # {asset: current price}
    nav         : float         # total portfolio value
    returns     : float         # day-over-day return


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Manager
# ─────────────────────────────────────────────────────────────────────────────

class Portfolio:
    """
    Manages cash, multi-asset positions, trade execution, and PnL.

    Parameters
    ----------
    initial_capital   : starting cash (default $100,000)
    transaction_cost  : % of trade value charged as commission (default 0.1%)
    slippage_pct      : % of price added/subtracted for market impact (default 0.05%)
    max_position_pct  : max % of NAV in a single asset (default 20%)

    Example
    -------
    port = Portfolio(initial_capital=100_000)
    port.buy("Equity", quantity=10, price=150.0, date="2020-01-05")
    port.sell("Equity", quantity=5,  price=155.0, date="2020-01-10")
    print(port.summary())
    """

    def __init__(
        self,
        initial_capital  : float = 100_000.0,
        transaction_cost : float = 0.001,    # 0.1%
        slippage_pct     : float = 0.0005,   # 0.05%
        max_position_pct : float = 0.20,     # 20% per asset
    ):
        self.initial_capital   = initial_capital
        self.cash              = initial_capital
        self.transaction_cost  = transaction_cost
        self.slippage_pct      = slippage_pct
        self.max_position_pct  = max_position_pct

        # {asset_name: shares_held}
        self._positions : dict[str, float] = {}
        # {asset_name: average_cost_basis}
        self._cost_basis: dict[str, float] = {}

        # Audit trail
        self._trade_log   : list[TradeRecord]       = []
        self._nav_history : list[PortfolioSnapshot] = []

        self._prev_nav = initial_capital

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    @property
    def trade_log(self) -> pd.DataFrame:
        """Return full trade history as a DataFrame."""
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self._trade_log])

    @property
    def nav_history(self) -> pd.DataFrame:
        """Return NAV history as a DataFrame."""
        if not self._nav_history:
            return pd.DataFrame()
        records = []
        for s in self._nav_history:
            records.append({
                "date"    : s.date,
                "cash"    : s.cash,
                "nav"     : s.nav,
                "returns" : s.returns,
                **{f"pos_{k}": v for k, v in s.positions.items()},
            })
        return pd.DataFrame(records)

    # ── NAV Calculation ───────────────────────────────────────────────────────

    def compute_nav(self, prices: dict[str, float]) -> float:
        """
        NAV = cash + sum(shares_held[asset] * price[asset])

        Parameters
        ----------
        prices : {asset_name: current_price}  — snapshot for today
        """
        position_value = sum(
            self._positions.get(asset, 0) * price
            for asset, price in prices.items()
        )
        return self.cash + position_value

    def record_snapshot(
        self,
        date  : object,
        prices: dict[str, float],
    ) -> float:
        """
        Record today's NAV snapshot. Call this once per simulation day.

        Returns today's NAV.
        """
        nav     = self.compute_nav(prices)
        ret     = (nav - self._prev_nav) / self._prev_nav if self._prev_nav else 0.0
        self._prev_nav = nav

        snap = PortfolioSnapshot(
            date      = date,
            cash      = self.cash,
            positions = dict(self._positions),
            prices    = dict(prices),
            nav       = nav,
            returns   = ret,
        )
        self._nav_history.append(snap)
        return nav

    # ── Order Execution ───────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, action: str) -> float:
        """
        Add slippage to execution price.
        BUY  orders pay slightly more  (market moves against you).
        SELL orders receive slightly less.
        """
        direction = 1 if action == "BUY" else -1
        return price * (1 + direction * self.slippage_pct)

    def _apply_commission(self, trade_value: float) -> float:
        """Commission = trade_value * transaction_cost_rate"""
        return abs(trade_value) * self.transaction_cost

    def _check_capital_shortfall(self, required: float, label: str):
        """Raise an informative error if insufficient cash (Issue 15)."""
        if self.cash < required:
            raise ValueError(
                f"[portfolio] Capital shortfall on {label}: "
                f"need ${required:,.2f}, have ${self.cash:,.2f}"
            )

    def _check_position_limit(self, asset: str, cost: float, prices: dict[str, float]):
        """Raise if new position would exceed max_position_pct of NAV (Issue 9)."""
        if not prices:
            return
        nav = self.compute_nav(prices)
        existing_val = self._positions.get(asset, 0) * prices.get(asset, 0)
        if (existing_val + cost) / nav > self.max_position_pct:
            raise ValueError(
                f"[portfolio] Position limit breach for {asset}: "
                f"would exceed {self.max_position_pct:.0%} of NAV=${nav:,.2f}"
            )

    def buy(
        self,
        asset    : str,
        quantity : float,
        price    : float,
        date     : object,
        reason   : str = "",
        prices   : Optional[dict] = None,
    ) -> TradeRecord:
        """
        Execute a BUY order.

        Parameters
        ----------
        asset    : asset identifier (e.g. "Equity", "Gold")
        quantity : number of shares / units to buy
        price    : raw market price
        date     : trade date (for audit log)
        reason   : human-readable reason (signal, rule)
        prices   : full {asset:price} dict for position limit check
        """
        exec_price  = self._apply_slippage(price, "BUY")
        trade_value = exec_price * quantity
        commission  = self._apply_commission(trade_value)
        total_cost  = trade_value + commission

        # Guard rails
        self._check_capital_shortfall(total_cost, f"BUY {quantity} {asset}")
        if prices:
            self._check_position_limit(asset, trade_value, prices)

        # Update state
        cash_before     = self.cash
        self.cash      -= total_cost
        self._positions[asset] = self._positions.get(asset, 0) + quantity

        # Update cost basis (weighted average)
        prev_shares = self._positions.get(asset, 0) - quantity
        prev_cost   = self._cost_basis.get(asset, exec_price)
        if prev_shares > 0:
            self._cost_basis[asset] = (
                (prev_shares * prev_cost + quantity * exec_price)
                / self._positions[asset]
            )
        else:
            self._cost_basis[asset] = exec_price

        record = TradeRecord(
            date        = date,
            asset       = asset,
            action      = "BUY",
            quantity    = quantity,
            price       = exec_price,
            raw_price   = price,
            slippage    = (exec_price - price) * quantity,
            commission  = commission,
            cash_before = cash_before,
            cash_after  = self.cash,
            reason      = reason,
        )
        self._trade_log.append(record)
        return record

    def sell(
        self,
        asset    : str,
        quantity : float,
        price    : float,
        date     : object,
        reason   : str = "",
    ) -> TradeRecord:
        """
        Execute a SELL order.

        Parameters
        ----------
        asset    : asset identifier
        quantity : number of shares / units to sell
        price    : raw market price
        date     : trade date (for audit log)
        reason   : human-readable reason (signal, rule)
        """
        held = self._positions.get(asset, 0)
        if quantity > held:
            raise ValueError(
                f"[portfolio] Cannot sell {quantity} {asset} — only {held} held"
            )

        exec_price  = self._apply_slippage(price, "SELL")
        trade_value = exec_price * quantity
        commission  = self._apply_commission(trade_value)
        proceeds    = trade_value - commission

        cash_before  = self.cash
        self.cash   += proceeds
        self._positions[asset] = held - quantity
        if self._positions[asset] == 0:
            del self._positions[asset]
            del self._cost_basis[asset]

        record = TradeRecord(
            date        = date,
            asset       = asset,
            action      = "SELL",
            quantity    = quantity,
            price       = exec_price,
            raw_price   = price,
            slippage    = (price - exec_price) * quantity,
            commission  = commission,
            cash_before = cash_before,
            cash_after  = self.cash,
            reason      = reason,
        )
        self._trade_log.append(record)
        return record

    def rebalance(
        self,
        target_weights : dict[str, float],
        prices         : dict[str, float],
        date           : object,
        reason         : str = "rebalance",
    ):
        """
        Rebalance to target weight allocation (Issue 11).

        Parameters
        ----------
        target_weights : {asset: target_fraction_of_NAV}  — must sum to <= 1.0
        prices         : {asset: current_price}
        date           : rebalance date

        Example
        -------
        port.rebalance(
            target_weights = {"Equity": 0.60, "Gold": 0.20, "Bonds": 0.10},
            prices         = {"Equity": 150, "Gold": 1800, "Bonds": 100},
            date           = "2020-06-30"
        )
        """
        assert sum(target_weights.values()) <= 1.001, "Weights must sum to <= 1"

        nav = self.compute_nav(prices)
        
        # Calculate target shares and deltas
        deltas = {}
        for asset, weight in target_weights.items():
            target_value  = nav * weight
            current_price = prices.get(asset)
            if current_price is None or current_price <= 0:
                continue

            target_shares  = target_value / current_price
            current_shares = self._positions.get(asset, 0)
            deltas[asset] = target_shares - current_shares

        # Sell first to free up cash
        for asset, delta in deltas.items():
            if delta < -0.001:
                self.sell(asset, abs(delta), prices[asset], date, reason)
                
        # Then buy with available cash
        for asset, delta in deltas.items():
            if delta > 0.001:
                # Adjust delta if we don't have enough cash (due to slippage/commission buffer)
                required_cash = delta * prices[asset] * (1 + self.slippage_pct + self.transaction_cost)
                if required_cash > self.cash:
                    delta = (self.cash * 0.99) / (prices[asset] * (1 + self.slippage_pct + self.transaction_cost))
                
                if delta > 0.001:
                    self.buy(asset, delta, prices[asset], date, reason, prices)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, prices: Optional[dict] = None) -> str:
        nav = self.compute_nav(prices) if prices else None
        lines = [
            "=" * 50,
            "  PORTFOLIO SUMMARY",
            "=" * 50,
            f"  Initial Capital : ${self.initial_capital:>12,.2f}",
            f"  Cash            : ${self.cash:>12,.2f}",
        ]
        for asset, shares in self._positions.items():
            cb = self._cost_basis.get(asset, 0)
            lines.append(f"  {asset:<14} : {shares:>8.2f} shares @ avg ${cb:.2f}")
        if nav:
            pnl = nav - self.initial_capital
            pnl_pct = pnl / self.initial_capital
            lines += [
                f"  NAV             : ${nav:>12,.2f}",
                f"  PnL             : ${pnl:>12,.2f}  ({pnl_pct:.2%})",
            ]
        lines += [
            f"  Total Trades    : {len(self._trade_log)}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def get_returns(self) -> pd.Series:
        """Extract daily NAV returns from nav_history."""
        hist = self.nav_history
        if hist.empty:
            return pd.Series(dtype=float)
        return hist.set_index("date")["returns"]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = Portfolio(initial_capital=100_000, transaction_cost=0.001, slippage_pct=0.0005)

    # Day 1: buy 100 shares of Equity at $100
    port.buy("Equity", 100, 100.0, "2020-01-02", reason="Initial allocation")
    port.record_snapshot("2020-01-02", {"Equity": 100.0})

    # Day 2: price rises to $105
    port.record_snapshot("2020-01-03", {"Equity": 105.0})

    # Day 3: buy 50 more shares
    port.buy("Equity", 50, 105.0, "2020-01-04", reason="Momentum signal")
    port.record_snapshot("2020-01-04", {"Equity": 105.0})

    # Day 4: sell 80 shares at $108
    port.sell("Equity", 80, 108.0, "2020-01-07", reason="Take profit")
    port.record_snapshot("2020-01-07", {"Equity": 108.0})

    print(port.summary(prices={"Equity": 108.0}))
    print()
    print("Trade Log:")
    print(port.trade_log.to_string())
    print()
    print("NAV History:")
    print(port.nav_history[["date", "cash", "nav", "returns"]].to_string())
