"""
signals.py
----------
Signal Generation Engine for the semi-automated trading system.

Implements a Risk-Aware Trend Following strategy (Issue 8 & 16) that:
    1. Detects market momentum (trend)
    2. Overlays macro sentiment & volatility for risk mitigation
    3. Outputs target portfolio weights for rebalancing

This fulfills the requirement to build an "explainable" strategy
that aims to minimize drawdowns while capturing upside.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class SignalResult:
    """The output of the signal engine for a single day."""
    date: object
    target_weights: Dict[str, float]
    reason: str


class RiskAwareSignalEngine:
    """
    A rule-based, explainable signal generator.
    
    Logic:
        - Base: Follow Equity momentum (30-day). If positive, go LONG Equity.
        - Risk Overlay 1: High Volatility. If rolling vol > threshold, reduce Equity, buy Gold.
        - Risk Overlay 2: Macro Sentiment. If sentiment < threshold, move to Cash/Bonds.
    """
    
    def __init__(
        self,
        vol_threshold: float = 0.20,       # 20% annualized vol limit
        sentiment_threshold: float = -0.5, # negative sentiment threshold (normalized)
        max_equity_weight: float = 0.90,   # Never go 100% equity
    ):
        self.vol_threshold = vol_threshold
        self.sentiment_threshold = sentiment_threshold
        self.max_equity_weight = max_equity_weight

    def generate_signal(self, row: pd.Series) -> SignalResult:
        """
        Evaluate a single day's data and return target portfolio weights.
        
        Parameters
        ----------
        row : pd.Series representing one row of the preprocessed DataFrame
        """
        # Read current state
        momentum = row.get("Equity_Momentum30", 0.0)
        vol      = row.get("Equity_RollingVol20", 0.0)
        sent     = row.get("Sentiment_norm", 0.0)
        
        date = row["Date"]

        # Default: All Cash
        weights = {"Equity": 0.0, "Gold": 0.0}
        reasons = []

        # 1. Risk Overlay: Extreme Macro Fear
        if sent < self.sentiment_threshold:
            weights["Gold"] = 0.50
            reasons.append(f"Risk-Off: Sentiment ({sent:.2f}) < {self.sentiment_threshold}")
            
        # 2. Risk Overlay: High Volatility
        elif vol > self.vol_threshold:
            # Reduce equity exposure, hedge with gold
            weights["Equity"] = 0.40
            weights["Gold"]   = 0.40
            reasons.append(f"De-risk: Volatility ({vol:.1%}) > {self.vol_threshold:.1%}")
            
        # 3. Base Strategy: Trend Following
        else:
            if momentum > 0:
                # Up-trend: Max allocation to Equity
                weights["Equity"] = self.max_equity_weight
                weights["Gold"]   = 0.0
                reasons.append(f"Trend-On: Momentum ({momentum:.2%}) > 0")
            else:
                # Down-trend: Move to Cash
                weights["Equity"] = 0.0
                weights["Gold"]   = 0.0
                reasons.append(f"Trend-Off: Momentum ({momentum:.2%}) <= 0")

        # Clean up reason string
        final_reason = " | ".join(reasons) if reasons else "No clear signal, holding cash."
        
        return SignalResult(
            date=date,
            target_weights=weights,
            reason=final_reason
        )

class MLSignalEngine:
    """
    An ML-driven signal generator using the Random Forest classifier.
    
    Logic:
        - If ML predicts UP (prob > threshold), allocate strongly to Equity.
        - If ML predicts DOWN, allocate to Gold.
        - Also retains the extreme Volatility Risk Overlay for safety.
    """
    
    def __init__(
        self,
        vol_threshold: float = 0.25,
        prob_threshold: float = 0.55,
        max_equity_weight: float = 0.90,
    ):
        import joblib
        from pathlib import Path
        
        self.vol_threshold = vol_threshold
        self.prob_threshold = prob_threshold
        self.max_equity_weight = max_equity_weight
        
        # Load the model
        model_path = Path(__file__).resolve().parent.parent / "models" / "rf_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"ML Model not found at {model_path}. Please run evaluate_ml.py first to train it.")
        
        self.model = joblib.load(model_path)
        
    def generate_signal(self, row: pd.Series) -> SignalResult:
        from ml_model import get_prediction_probability
        
        date = row["Date"]
        vol = row.get("Equity_RollingVol20", 0.0)
        
        # Get ML prediction probability for UP (1)
        up_prob = get_prediction_probability(row, self.model)
        
        weights = {"Equity": 0.0, "Gold": 0.0}
        reasons = []
        
        # 1. Extreme Risk Overlay (Trumps ML)
        if vol > self.vol_threshold:
            weights["Equity"] = 0.20
            weights["Gold"]   = 0.50
            reasons.append(f"De-risk Override: Volatility ({vol:.1%}) > {self.vol_threshold:.1%}")
            
        # 2. ML Prediction
        else:
            if up_prob >= self.prob_threshold:
                # Strong conviction UP
                weights["Equity"] = self.max_equity_weight
                weights["Gold"]   = 0.0
                reasons.append(f"ML Long: Up probability {up_prob:.1%} >= {self.prob_threshold:.1%}")
            elif up_prob <= (1 - self.prob_threshold):
                # Strong conviction DOWN
                weights["Equity"] = 0.0
                weights["Gold"]   = 0.50
                reasons.append(f"ML Short: Down probability {(1-up_prob):.1%} >= {self.prob_threshold:.1%}")
            else:
                # Low conviction -> Hold balanced/cash
                weights["Equity"] = 0.40
                weights["Gold"]   = 0.20
                reasons.append(f"ML Neutral: Prob {up_prob:.1%} (Low conviction)")
                
        final_reason = " | ".join(reasons)
        
        return SignalResult(
            date=date,
            target_weights=weights,
            reason=final_reason
        )


