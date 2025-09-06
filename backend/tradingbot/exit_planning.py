"""
Exit Planning and Scenario Analysis Tools
Implements systematic profit-taking and loss management from the successful playbook.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import math

from .options_calculator import BlackScholesCalculator
from .risk_management import Position, PositionStatus


class ExitReason(Enum):
    """Reasons for position exit"""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_DECAY = "time_decay"
    DELTA_THRESHOLD = "delta_threshold"
    VOLATILITY_CRUSH = "volatility_crush"
    TREND_BREAK = "trend_break"
    MANUAL = "manual"
    EXPIRATION = "expiration"


class ExitSignalStrength(Enum):
    """Strength of exit signal"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    URGENT = "urgent"


@dataclass
class ExitLevel:
    """Definition of an exit level with conditions"""
    name: str
    trigger_condition: str              # Human readable condition
    target_roi: float                   # Target return on investment
    position_fraction: float            # Fraction of position to close (0.0 to 1.0)
    priority: int                       # Lower number = higher priority
    delta_threshold: Optional[float] = None
    days_threshold: Optional[int] = None
    iv_threshold: Optional[float] = None


@dataclass
class ExitSignal:
    """Exit signal with reasoning and urgency"""
    reason: ExitReason
    strength: ExitSignalStrength
    position_fraction: float            # How much of position to close
    estimated_exit_price: float         # Estimated exit premium per contract
    expected_pnl: float                # Expected P&L from this exit
    reasoning: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"{self.reason.value.upper()} - {self.strength.value} ({self.position_fraction:.1%} of position)"


@dataclass
class ScenarioResult:
    """Result of a scenario analysis"""
    scenario_name: str
    spot_price: float
    spot_change_pct: float
    days_passed: int
    implied_volatility: float
    
    # Option pricing results
    estimated_premium: float
    position_value: float
    pnl_per_contract: float
    total_pnl: float
    roi: float
    
    # Greeks and risk metrics
    delta: float
    time_decay_impact: float
    vega_impact: float
    
    # Exit recommendations
    recommended_action: str
    exit_signals: List[ExitSignal] = field(default_factory=list)


class ExitStrategy:
    """Systematic exit strategy based on successful playbook"""
    
    def __init__(self):
        self.bs_calc = BlackScholesCalculator()
        
        # Exit levels from successful trade (100%, 200%, 250% gains)
        self.default_exit_levels = [
            ExitLevel(
                name="First Take-Profit",
                trigger_condition="100% profit (2x premium)",
                target_roi=1.0,
                position_fraction=0.33,  # Close 1/3
                priority=1
            ),
            ExitLevel(
                name="Second Take-Profit", 
                trigger_condition="200% profit (3x premium)",
                target_roi=2.0,
                position_fraction=0.33,  # Close another 1/3
                priority=2
            ),
            ExitLevel(
                name="Final Take-Profit",
                trigger_condition="250% profit or delta ≥ 0.60",
                target_roi=2.5,
                position_fraction=1.0,   # Close remaining
                priority=3,
                delta_threshold=0.60
            ),
            ExitLevel(
                name="Stop Loss",
                trigger_condition="45% loss or below 50-DMA",
                target_roi=-0.45,
                position_fraction=1.0,   # Close all
                priority=0  # Highest priority
            ),
            ExitLevel(
                name="Time Stop",
                trigger_condition="5 trading days with no progress",
                target_roi=-0.20,  # Assume small loss
                position_fraction=1.0,
                priority=0,
                days_threshold=5
            )
        ]
    
    def analyze_exit_conditions(
        self,
        position: Position,
        current_spot: float,
        current_iv: float,
        trend_broken: bool = False,
        days_since_entry: int = 0,
        risk_free_rate: float = 0.04,
        dividend_yield: float = 0.0
    ) -> List[ExitSignal]:
        """
        Analyze current exit conditions for a position
        
        Returns:
            List of exit signals sorted by priority
        """
        signals = []
        
        # Calculate current option metrics
        time_to_expiry = max(1, position.days_to_expiry) / 365.0
        
        try:
            current_theoretical_price = self.bs_calc.call_price(
                spot=current_spot,
                strike=position.strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=current_iv
            ) * 100  # Convert to per-contract
            
            current_delta = self.bs_calc.delta(
                spot=current_spot,
                strike=position.strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=current_iv
            )
            
        except (ValueError, ZeroDivisionError):
            current_theoretical_price = 0.01
            current_delta = 0.0
        
        current_roi = (current_theoretical_price - position.entry_premium) / position.entry_premium
        
        # Check each exit level
        for exit_level in self.default_exit_levels:
            signal = self._check_exit_level(
                exit_level, position, current_theoretical_price, current_roi,
                current_delta, trend_broken, days_since_entry
            )
            if signal:
                signals.append(signal)
        
        # Check for special conditions
        
        # Volatility crush detection
        if current_iv < 0.15 and current_roi < 0.5:  # Low IV + modest gains
            signals.append(ExitSignal(
                reason=ExitReason.VOLATILITY_CRUSH,
                strength=ExitSignalStrength.MODERATE,
                position_fraction=0.5,
                estimated_exit_price=current_theoretical_price,
                expected_pnl=(current_theoretical_price - position.entry_premium) * position.contracts * 0.5,
                reasoning=["Low implied volatility may limit upside", "Consider reducing exposure"]
            ))
        
        # Trend break
        if trend_broken and current_roi < 1.0:  # Trend broken before significant profit
            signals.append(ExitSignal(
                reason=ExitReason.TREND_BREAK,
                strength=ExitSignalStrength.STRONG,
                position_fraction=1.0,
                estimated_exit_price=current_theoretical_price,
                expected_pnl=(current_theoretical_price - position.entry_premium) * position.contracts,
                reasoning=["Bull trend broken", "Exit to preserve capital"]
            ))
        
        # Time decay urgency (within 1 week of expiry)
        if position.days_to_expiry <= 7 and current_roi < 0.5:
            signals.append(ExitSignal(
                reason=ExitReason.TIME_DECAY,
                strength=ExitSignalStrength.URGENT,
                position_fraction=1.0,
                estimated_exit_price=current_theoretical_price,
                expected_pnl=(current_theoretical_price - position.entry_premium) * position.contracts,
                reasoning=["Less than 1 week to expiry", "Time decay accelerating", "Exit immediately"]
            ))
        
        # Sort by priority (lower number = higher priority)
        signals.sort(key=lambda x: (
            0 if x.strength == ExitSignalStrength.URGENT else
            1 if x.strength == ExitSignalStrength.STRONG else
            2 if x.strength == ExitSignalStrength.MODERATE else 3
        ))
        
        return signals
    
    def _check_exit_level(
        self,
        exit_level: ExitLevel,
        position: Position,
        current_price: float,
        current_roi: float,
        current_delta: float,
        trend_broken: bool,
        days_since_entry: int
    ) -> Optional[ExitSignal]:
        """Check if a specific exit level is triggered"""
        
        reasoning = []
        triggered = False
        strength = ExitSignalStrength.WEAK
        
        # Check ROI trigger
        if exit_level.target_roi >= 0:  # Profit target
            if current_roi >= exit_level.target_roi:
                triggered = True
                reasoning.append(f"Profit target hit: {current_roi:.1%} ≥ {exit_level.target_roi:.1%}")
                strength = ExitSignalStrength.MODERATE
        else:  # Stop loss
            if current_roi <= exit_level.target_roi:
                triggered = True
                reasoning.append(f"Stop loss triggered: {current_roi:.1%} ≤ {exit_level.target_roi:.1%}")
                strength = ExitSignalStrength.STRONG
        
        # Check delta threshold
        if exit_level.delta_threshold and current_delta >= exit_level.delta_threshold:
            triggered = True
            reasoning.append(f"Delta threshold reached: {current_delta:.3f} ≥ {exit_level.delta_threshold:.3f}")
            strength = ExitSignalStrength.STRONG
        
        # Check time threshold
        if exit_level.days_threshold and days_since_entry >= exit_level.days_threshold:
            triggered = True
            reasoning.append(f"Time threshold exceeded: {days_since_entry} days ≥ {exit_level.days_threshold} days")
            strength = ExitSignalStrength.MODERATE
        
        # Check trend break for stop loss
        if exit_level.target_roi < 0 and trend_broken:  # Stop loss level
            triggered = True
            reasoning.append("Trend broken below 50-EMA")
            strength = ExitSignalStrength.STRONG
        
        if triggered:
            expected_pnl = (current_price - position.entry_premium) * position.contracts * exit_level.position_fraction
            
            return ExitSignal(
                reason=ExitReason.PROFIT_TARGET if exit_level.target_roi >= 0 else ExitReason.STOP_LOSS,
                strength=strength,
                position_fraction=exit_level.position_fraction,
                estimated_exit_price=current_price,
                expected_pnl=expected_pnl,
                reasoning=reasoning
            )
        
        return None


class ScenarioAnalyzer:
    """Advanced scenario analysis for exit planning"""
    
    def __init__(self):
        self.bs_calc = BlackScholesCalculator()
        self.exit_strategy = ExitStrategy()
    
    def run_comprehensive_analysis(
        self,
        position: Position,
        current_spot: float,
        current_iv: float,
        scenarios: Dict[str, Dict] = None,
        risk_free_rate: float = 0.04,
        dividend_yield: float = 0.0
    ) -> List[ScenarioResult]:
        """
        Run comprehensive scenario analysis
        
        Args:
            position: Current position to analyze
            current_spot: Current spot price
            current_iv: Current implied volatility
            scenarios: Custom scenarios or None for default
            
        Returns:
            List of scenario results
        """
        if scenarios is None:
            scenarios = self._get_default_scenarios()
        
        results = []
        
        for scenario_name, params in scenarios.items():
            result = self._analyze_scenario(
                position, current_spot, current_iv, scenario_name, params,
                risk_free_rate, dividend_yield
            )
            results.append(result)
        
        return results
    
    def _get_default_scenarios(self) -> Dict[str, Dict]:
        """Get default scenario parameters"""
        return {
            "Bear Case (-5%)": {
                "spot_change": -0.05,
                "days_forward": 3,
                "iv_change": 0.10,  # IV often increases in selloffs
            },
            "Mild Pullback (-3%)": {
                "spot_change": -0.03,
                "days_forward": 2,
                "iv_change": 0.05,
            },
            "Current (No Change)": {
                "spot_change": 0.00,
                "days_forward": 1,
                "iv_change": 0.00,
            },
            "Modest Rally (+3%)": {
                "spot_change": 0.03,
                "days_forward": 2,
                "iv_change": -0.02,  # IV often decreases in rallies
            },
            "Strong Rally (+5%)": {
                "spot_change": 0.05,
                "days_forward": 3,
                "iv_change": -0.05,
            },
            "Breakout (+8%)": {
                "spot_change": 0.08,
                "days_forward": 5,
                "iv_change": -0.03,
            },
            "Time Decay (1 Week)": {
                "spot_change": 0.00,
                "days_forward": 7,
                "iv_change": 0.00,
            },
            "Time Decay (2 Weeks)": {
                "spot_change": 0.01,
                "days_forward": 14,
                "iv_change": -0.02,
            }
        }
    
    def _analyze_scenario(
        self,
        position: Position,
        current_spot: float,
        current_iv: float,
        scenario_name: str,
        params: Dict,
        risk_free_rate: float,
        dividend_yield: float
    ) -> ScenarioResult:
        """Analyze a single scenario"""
        
        # Calculate scenario parameters
        new_spot = current_spot * (1 + params["spot_change"])
        new_iv = max(0.01, current_iv + params.get("iv_change", 0))
        days_forward = params.get("days_forward", 1)
        
        # Calculate new time to expiry
        new_dte = max(1, position.days_to_expiry - days_forward)
        time_to_expiry = new_dte / 365.0
        
        # Calculate option value under scenario
        try:
            new_premium = self.bs_calc.call_price(
                spot=new_spot,
                strike=position.strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=new_iv
            ) * 100
            
            new_delta = self.bs_calc.delta(
                spot=new_spot,
                strike=position.strike,
                time_to_expiry_years=time_to_expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                implied_volatility=new_iv
            )
            
        except (ValueError, ZeroDivisionError):
            new_premium = max(0, new_spot - position.strike) if new_spot > position.strike else 0.01
            new_delta = 1.0 if new_spot > position.strike else 0.0
        
        # Calculate P&L metrics
        position_value = new_premium * position.contracts
        pnl_per_contract = new_premium - position.entry_premium
        total_pnl = pnl_per_contract * position.contracts
        roi = pnl_per_contract / position.entry_premium if position.entry_premium > 0 else 0
        
        # Estimate impact components
        time_decay_impact = self._estimate_time_decay_impact(
            position, current_spot, current_iv, days_forward, risk_free_rate, dividend_yield
        )
        
        vega_impact = self._estimate_vega_impact(
            position, current_spot, current_iv, params.get("iv_change", 0), 
            new_dte, risk_free_rate, dividend_yield
        )
        
        # Generate exit signals for this scenario
        exit_signals = self.exit_strategy.analyze_exit_conditions(
            position, new_spot, new_iv, days_since_entry=days_forward
        )
        
        # Determine recommended action
        if exit_signals:
            strongest_signal = max(exit_signals, key=lambda x: x.strength.value)
            recommended_action = f"{strongest_signal.reason.value}: {strongest_signal.position_fraction:.0%}"
        else:
            if roi > 0.5:
                recommended_action = "HOLD - profitable position"
            elif roi < -0.3:
                recommended_action = "CONSIDER EXIT - significant loss"
            else:
                recommended_action = "MONITOR - neutral position"
        
        return ScenarioResult(
            scenario_name=scenario_name,
            spot_price=new_spot,
            spot_change_pct=params["spot_change"],
            days_passed=days_forward,
            implied_volatility=new_iv,
            estimated_premium=new_premium,
            position_value=position_value,
            pnl_per_contract=pnl_per_contract,
            total_pnl=total_pnl,
            roi=roi,
            delta=new_delta,
            time_decay_impact=time_decay_impact,
            vega_impact=vega_impact,
            recommended_action=recommended_action,
            exit_signals=exit_signals
        )
    
    def _estimate_time_decay_impact(
        self, position: Position, spot: float, iv: float, days_forward: int,
        risk_free_rate: float, dividend_yield: float
    ) -> float:
        """Estimate pure time decay impact"""
        try:
            # Current premium
            current_dte = position.days_to_expiry / 365.0
            current_premium = self.bs_calc.call_price(
                spot, position.strike, current_dte, risk_free_rate, dividend_yield, iv
            ) * 100
            
            # Premium after time decay (same spot, same IV)
            future_dte = max(1, position.days_to_expiry - days_forward) / 365.0
            future_premium = self.bs_calc.call_price(
                spot, position.strike, future_dte, risk_free_rate, dividend_yield, iv
            ) * 100
            
            return future_premium - current_premium
            
        except (ValueError, ZeroDivisionError):
            return -days_forward * position.entry_premium * 0.02  # Rough 2% per day estimate
    
    def _estimate_vega_impact(
        self, position: Position, spot: float, current_iv: float, iv_change: float,
        days_to_expiry: int, risk_free_rate: float, dividend_yield: float
    ) -> float:
        """Estimate impact of volatility change"""
        if abs(iv_change) < 0.001:
            return 0.0
        
        try:
            time_to_expiry = days_to_expiry / 365.0
            
            # Premium at current IV
            current_premium = self.bs_calc.call_price(
                spot, position.strike, time_to_expiry, risk_free_rate, dividend_yield, current_iv
            ) * 100
            
            # Premium at new IV
            new_iv = max(0.01, current_iv + iv_change)
            new_premium = self.bs_calc.call_price(
                spot, position.strike, time_to_expiry, risk_free_rate, dividend_yield, new_iv
            ) * 100
            
            return new_premium - current_premium
            
        except (ValueError, ZeroDivisionError):
            # Rough estimate: each 1% IV change = ~0.5% premium change
            return position.entry_premium * iv_change * 50
    
    def generate_exit_plan(self, scenarios: List[ScenarioResult]) -> Dict:
        """Generate actionable exit plan based on scenario analysis"""
        
        # Count scenarios by recommended action
        action_counts = {}
        for scenario in scenarios:
            action = scenario.recommended_action.split(':')[0]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Find most likely profitable scenarios
        profitable_scenarios = [s for s in scenarios if s.roi > 0]
        loss_scenarios = [s for s in scenarios if s.roi < -0.2]
        
        # Calculate expected value
        total_scenarios = len(scenarios)
        expected_roi = sum(s.roi for s in scenarios) / total_scenarios if scenarios else 0
        
        plan = {
            'summary': {
                'total_scenarios': total_scenarios,
                'profitable_scenarios': len(profitable_scenarios),
                'loss_scenarios': len(loss_scenarios),
                'expected_roi': expected_roi,
                'win_rate': len(profitable_scenarios) / total_scenarios if total_scenarios > 0 else 0
            },
            
            'action_distribution': action_counts,
            
            'recommendations': [],
            
            'risk_assessment': {
                'upside_potential': max(s.roi for s in scenarios) if scenarios else 0,
                'downside_risk': min(s.roi for s in scenarios) if scenarios else 0,
                'volatility': np.std([s.roi for s in scenarios]) if scenarios else 0
            },
            
            'key_scenarios': {
                'best_case': max(scenarios, key=lambda x: x.roi) if scenarios else None,
                'worst_case': min(scenarios, key=lambda x: x.roi) if scenarios else None,
                'base_case': next((s for s in scenarios if 'Current' in s.scenario_name), scenarios[0] if scenarios else None)
            }
        }
        
        # Generate recommendations
        if expected_roi > 0.5:
            plan['recommendations'].append("Strong hold - positive expected value")
        elif expected_roi > 0.1:
            plan['recommendations'].append("Moderate hold - slight positive edge")
        elif expected_roi < -0.2:
            plan['recommendations'].append("Consider exit - negative expected value")
        else:
            plan['recommendations'].append("Neutral - monitor closely")
        
        # Risk-based recommendations
        if plan['risk_assessment']['downside_risk'] < -0.4:
            plan['recommendations'].append("High downside risk - consider stop loss")
        
        if plan['risk_assessment']['upside_potential'] > 1.0:
            plan['recommendations'].append("Significant upside potential - consider holding")
        
        return plan


if __name__ == "__main__":
    # Test the exit planning system
    print("=== EXIT PLANNING SYSTEM TEST ===")
    
    # Create sample position (similar to the successful trade)
    sample_position = Position(
        ticker="GOOGL",
        position_type="call",
        entry_date=datetime.now() - timedelta(days=3),
        expiry_date=datetime.now() + timedelta(days=27),
        strike=220.0,
        contracts=100,  # Smaller size for testing
        entry_premium=4.70,
        current_premium=8.50,  # Currently profitable
        total_cost=470,
        current_value=850,
        stop_loss_level=2.35,
        profit_targets=[1.0, 2.0, 2.5]
    )
    
    # Test exit strategy
    exit_strategy = ExitStrategy()
    
    exit_signals = exit_strategy.analyze_exit_conditions(
        position=sample_position,
        current_spot=215.0,
        current_iv=0.28,
        days_since_entry=3
    )
    
    print("Current Exit Signals:")
    for signal in exit_signals:
        print(f"  {signal}")
        for reason in signal.reasoning:
            print(f"    - {reason}")
    
    # Test scenario analysis
    print("\n=== SCENARIO ANALYSIS ===")
    analyzer = ScenarioAnalyzer()
    
    scenarios = analyzer.run_comprehensive_analysis(
        position=sample_position,
        current_spot=215.0,
        current_iv=0.28
    )
    
    print(f"\n{'Scenario':<20} {'Spot':<8} {'Premium':<8} {'P&L':<10} {'ROI':<8} {'Action':<20}")
    print("-" * 80)
    
    for scenario in scenarios:
        print(f"{scenario.scenario_name:<20} "
              f"${scenario.spot_price:<7.0f} "
              f"${scenario.estimated_premium:<7.2f} "
              f"${scenario.total_pnl:<9.0f} "
              f"{scenario.roi:<7.1%} "
              f"{scenario.recommended_action:<20}")
    
    # Generate exit plan
    exit_plan = analyzer.generate_exit_plan(scenarios)
    
    print(f"\n=== EXIT PLAN ===")
    print(f"Expected ROI: {exit_plan['summary']['expected_roi']:+.1%}")
    print(f"Win Rate: {exit_plan['summary']['win_rate']:.1%}")
    print(f"Upside Potential: {exit_plan['risk_assessment']['upside_potential']:+.1%}")
    print(f"Downside Risk: {exit_plan['risk_assessment']['downside_risk']:+.1%}")
    
    print("\nRecommendations:")
    for rec in exit_plan['recommendations']:
        print(f"  • {rec}")