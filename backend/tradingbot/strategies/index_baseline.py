#!/usr / bin/env python3
"""
WSB Strategy: Index Fund Baseline Comparison Tracker
Track performance vs. SPY / VTI baseline to validate if active trading beats "just buy index"
Most WSB strategies should beat buying and holding index funds, otherwise they're not worth it
"""

import argparse
import json
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

try: 
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e: 
    print(f"Missing required package: {e}")
    print("Run: pip install -r wsb_requirements.txt")
    exit(1)


@dataclass
class PerformanceComparison: 
    start_date: date
    end_date: date
    period_days: int
    
    # Strategy performance
    strategy_name: str
    strategy_return: float
    strategy_sharpe: float
    strategy_max_drawdown: float
    strategy_win_rate: float
    strategy_total_trades: int
    
    # Baseline performance (SPY / VTI)
    spy_return: float
    vti_return: float
    qqq_return: float  # NASDAQ baseline
    
    # Comparison metrics
    alpha_vs_spy: float      # Excess return over SPY
    alpha_vs_vti: float      # Excess return over VTI
    alpha_vs_qqq: float      # Excess return over QQQ
    
    # Risk metrics
    strategy_volatility: float
    spy_volatility: float
    information_ratio_spy: float  # Alpha / tracking error
    
    # Risk - adjusted comparison
    beats_spy: bool
    beats_vti: bool
    beats_qqq: bool
    risk_adjusted_winner: str
    
    # Trading costs impact
    trading_costs_drag: float  # Estimated cost drag from trading
    net_alpha_after_costs: float


@dataclass
class BaselineTracker: 
    """Track key index fund benchmarks"""
    spy_current: float
    vti_current: float
    qqq_current: float
    
    spy_ytd: float
    vti_ytd: float
    qqq_ytd: float
    
    spy_1y: float
    vti_1y: float
    qqq_1y: float
    
    last_updated: datetime


class IndexBaselineScanner: 
    def __init__(self): 
        self.benchmarks=["SPY", "VTI", "QQQ", "IWM", "DIA"]  # Key index ETFs
        
        # WSB strategy performance tracking (simplified mock data)
        self.wsb_strategies={
            "wheel_strategy": {
                "return_6m": 0.18,
                "volatility": 0.12,
                "max_drawdown": 0.08,
                "win_rate": 0.78,
                "trades": 24
            },
            "spx_credit_spreads": {
                "return_6m": 0.24,
                "volatility": 0.15,
                "max_drawdown": 0.12,
                "win_rate": 0.72,
                "trades": 48
            },
            "swing_trading": {
                "return_6m": 0.32,
                "volatility": 0.28,
                "max_drawdown": 0.18,
                "win_rate": 0.64,
                "trades": 36
            },
            "leaps_strategy": {
                "return_6m": 0.28,
                "volatility": 0.22,
                "max_drawdown": 0.15,
                "win_rate": 0.68,
                "trades": 12
            }
        }
    
    def get_baseline_performance(self, period_months: int=6) -> BaselineTracker:
        """Get current baseline index performance"""
        end_date=datetime.now()
        start_date=end_date - timedelta(days=period_months * 30)
        
        baselines={}
        ytd_start=datetime(end_date.year, 1, 1)
        one_year_ago=end_date - timedelta(days=365)
        
        for ticker in ["SPY", "VTI", "QQQ"]: 
            try: 
                stock=yf.Ticker(ticker)
                
                # Get current price
                current_data=stock.history(period="1d")
                if current_data.empty: 
                    continue
                current_price=current_data['Close'].iloc[-1]
                
                # Get historical prices
                hist_data=stock.history(period="1y")
                if len(hist_data) < 50: 
                    continue
                
                # Calculate returns
                period_start_price=hist_data.loc[hist_data.index >= start_date.replace(tzinfo=hist_data.index.tz)]['Close'].iloc[0]
                ytd_start_price=hist_data.loc[hist_data.index >= ytd_start.replace(tzinfo=hist_data.index.tz)]['Close'].iloc[0] 
                one_year_price=hist_data.loc[hist_data.index >= one_year_ago.replace(tzinfo=hist_data.index.tz)]['Close'].iloc[0]
                
                period_return=(current_price / period_start_price - 1.0)
                ytd_return=(current_price / ytd_start_price - 1.0) 
                one_year_return=(current_price / one_year_price - 1.0)
                
                baselines[ticker]={
                    'current': current_price,
                    'period_return': period_return,
                    'ytd_return': ytd_return,
                    'one_year_return': one_year_return
                }
                
            except Exception as e: 
                print(f"Error fetching {ticker}: {e}")
                # Use default values
                baselines[ticker]={
                    'current': 500.0 if ticker== "SPY" else 250.0 if ticker == "VTI" else 400.0,
                    'period_return': 0.10,
                    'ytd_return': 0.12,
                    'one_year_return': 0.15
                }
        
        return BaselineTracker(
            spy_current=baselines["SPY"]['current'],
            vti_current=baselines["VTI"]['current'],
            qqq_current=baselines["QQQ"]['current'],
            spy_ytd=baselines["SPY"]['ytd_return'],
            vti_ytd=baselines["VTI"]['ytd_return'], 
            qqq_ytd=baselines["QQQ"]['ytd_return'],
            spy_1y=baselines["SPY"]['one_year_return'],
            vti_1y=baselines["VTI"]['one_year_return'],
            qqq_1y=baselines["QQQ"]['one_year_return'],
            last_updated=datetime.now()
        )
    
    def calculate_trading_costs(self, total_trades: int, avg_position_size: float=5000) -> float:
        """Estimate trading cost drag"""
        # Assume $1 commission + 0.02% spread per trade
        commission_per_trade=1.0
        spread_cost_per_trade=avg_position_size * 0.0002  # 2 bps spread
        
        total_costs=total_trades * (commission_per_trade + spread_cost_per_trade)
        cost_drag=total_costs / avg_position_size  # As percentage of capital
        
        return min(cost_drag, 0.05)  # Cap at 5% drag
    
    def compare_strategy_performance(self, strategy_name: str, period_months: int=6) -> PerformanceComparison:
        """Compare strategy performance vs baselines"""
        
        # Get baseline performance
        baselines=self.get_baseline_performance(period_months)
        
        # Get strategy performance (mock data - would be real P & L in production)
        if strategy_name not in self.wsb_strategies: 
            raise ValueError(f"Strategy {strategy_name} not found")
        
        strategy=self.wsb_strategies[strategy_name]
        
        # Calculate performance metrics
        start_date_obj=date.today() - timedelta(days=period_months * 30)
        end_date_obj=date.today()
        
        strategy_return=strategy["return_6m"] * (period_months / 6.0)  # Scale by period
        strategy_vol=strategy["volatility"]
        max_dd=strategy["max_drawdown"] 
        win_rate=strategy["win_rate"]
        total_trades=strategy["trades"] * (period_months / 6.0)
        
        # Risk - free rate (approximate)
        risk_free_rate=0.04 * (period_months / 12.0)  # 4% annual
        
        # Calculate Sharpe ratio
        sharpe_ratio=(strategy_return - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        
        # Get baseline returns for the period
        spy_return=baselines.spy_1y * (period_months / 12.0) if period_months >= 12 else baselines.spy_ytd
        vti_return=baselines.vti_1y * (period_months / 12.0) if period_months >= 12 else baselines.vti_ytd
        qqq_return=baselines.qqq_1y * (period_months / 12.0) if period_months >= 12 else baselines.qqq_ytd
        
        # Estimate baseline volatility (historical approximation)
        spy_vol=0.16 * np.sqrt(period_months / 12.0)  # ~16% annual vol
        
        # Calculate alpha (excess returns)
        alpha_spy=strategy_return - spy_return
        alpha_vti=strategy_return - vti_return
        alpha_qqq=strategy_return - qqq_return
        
        # Information ratio (alpha / tracking error)
        tracking_error=max(0.01, abs(strategy_vol - spy_vol))  # Rough estimate
        info_ratio=alpha_spy / tracking_error
        
        # Trading cost impact
        trading_costs=self.calculate_trading_costs(int(total_trades))
        net_alpha_spy=alpha_spy - trading_costs
        
        # Determine winners
        beats_spy=strategy_return > spy_return
        beats_vti=strategy_return > vti_return
        beats_qqq=strategy_return > qqq_return
        
        # Risk - adjusted winner (Sharpe comparison)
        spy_sharpe=(spy_return - risk_free_rate) / spy_vol
        risk_adj_winner="Strategy" if sharpe_ratio > spy_sharpe else "SPY"
        
        return PerformanceComparison(
            start_date=start_date_obj,
            end_date=end_date_obj,
            period_days=period_months * 30,
            strategy_name=strategy_name,
            strategy_return=strategy_return,
            strategy_sharpe=sharpe_ratio,
            strategy_max_drawdown=max_dd,
            strategy_win_rate=win_rate,
            strategy_total_trades=int(total_trades),
            spy_return=spy_return,
            vti_return=vti_return,
            qqq_return=qqq_return,
            alpha_vs_spy=alpha_spy,
            alpha_vs_vti=alpha_vti,
            alpha_vs_qqq=alpha_qqq,
            strategy_volatility=strategy_vol,
            spy_volatility=spy_vol,
            information_ratio_spy=info_ratio,
            beats_spy=beats_spy,
            beats_vti=beats_vti,
            beats_qqq=beats_qqq,
            risk_adjusted_winner=risk_adj_winner,
            trading_costs_drag=trading_costs,
            net_alpha_after_costs=net_alpha_spy
        )
    
    def scan_all_strategies(self, period_months: int=6) -> List[PerformanceComparison]:
        """Compare all WSB strategies vs baselines"""
        comparisons=[]
        
        print(f"📊 Comparing WSB strategies vs index baselines ({period_months} month period)")
        
        for strategy_name in self.wsb_strategies.keys(): 
            try: 
                comparison=self.compare_strategy_performance(strategy_name, period_months)
                comparisons.append(comparison)
                print(f"✅ Analyzed {strategy_name}")
            except Exception as e: 
                print(f"❌ Error analyzing {strategy_name}: {e}")
        
        # Sort by net alpha (after costs)
        comparisons.sort(key=lambda x: x.net_alpha_after_costs, reverse=True)
        
        return comparisons
    
    def format_comparison_report(self, comparisons: List[PerformanceComparison]) -> str:
        """Format baseline comparison report"""
        if not comparisons: 
            return "📊 No strategy comparisons available."
        
        output=f"\n📈 WSB STRATEGY vs INDEX BASELINE COMPARISON\n"
        output += "=" * 80 + "\n"
        
        # Summary table header
        output += f"\n{'Strategy': <20} {'Return': <8} {'Alpha': <8} {'Sharpe': <8} {'Beats SPY': <10} {'Winner': <12}\n"
        output += "-" * 78 + "\n"
        
        for comp in comparisons: 
            strategy_display=comp.strategy_name.replace('_', ' ').title()[: 18]
            return_pct=f"{comp.strategy_return: .1%}"
            alpha_pct=f"{comp.net_alpha_after_costs:+.1%}"
            sharpe_str=f"{comp.strategy_sharpe: .2f}"
            beats_spy="YES" if comp.beats_spy else "NO"
            winner=comp.risk_adjusted_winner
            
            output += f"{strategy_display: <20} {return_pct: <8} {alpha_pct: <8} {sharpe_str: <8} {beats_spy: <10} {winner: <12}\n"
        
        # Detailed breakdown
        output += "\n" + "=" * 80 + "\n"
        output += "📊 DETAILED PERFORMANCE BREAKDOWN: \n"
        
        for i, comp in enumerate(comparisons, 1): 
            output += f"\n{i}. {comp.strategy_name.replace('_', ' ').upper()}\n"
            output += f"   Period: {comp.start_date} to {comp.end_date} ({comp.period_days} days)\n"
            output += f"   Strategy Return: {comp.strategy_return:.2%} | Sharpe: {comp.strategy_sharpe:.2f}\n"
            output += f"   Max Drawdown: {comp.strategy_max_drawdown:.2%} | Win Rate: {comp.strategy_win_rate:.1%}\n"
            output += f"   Total Trades: {comp.strategy_total_trades} | Trading Cost Drag: {comp.trading_costs_drag:.2%}\n"
            
            output += f"\n   📈 BASELINE COMPARISON: \n"
            output += f"   SPY Return:     {comp.spy_return:.2%} | Alpha: {comp.alpha_vs_spy:+.2%}\n"
            output += f"   VTI Return:     {comp.vti_return:.2%} | Alpha: {comp.alpha_vs_vti:+.2%}\n"
            output += f"   QQQ Return:     {comp.qqq_return:.2%} | Alpha: {comp.alpha_vs_qqq:+.2%}\n"
            
            output += f"\n   🎯 FINAL VERDICT: \n"
            output += f"   Beats SPY: {'✅ YES' if comp.beats_spy else '❌ NO'} | "
            output += f"Beats VTI: {'✅ YES' if comp.beats_vti else '❌ NO'} | "
            output += f"Beats QQQ: {'✅ YES' if comp.beats_qqq else '❌ NO'}\n"
            output += f"   Risk - Adjusted Winner: {comp.risk_adjusted_winner}\n"
            output += f"   Net Alpha After Costs: {comp.net_alpha_after_costs:+.2%}\n"
        
        # Summary insights
        output += "\n" + "=" * 80
        output += "\n🎯 WSB STRATEGY INSIGHTS: \n"
        
        total_strategies=len(comparisons)
        beating_spy=len([c for c in comparisons if c.beats_spy])
        beating_vti=len([c for c in comparisons if c.beats_vti])
        positive_alpha=len([c for c in comparisons if c.net_alpha_after_costs > 0])
        
        output += f"• {beating_spy}/{total_strategies} strategies beat SPY ({beating_spy / total_strategies: .0%})\n"
        output += f"• {beating_vti}/{total_strategies} strategies beat VTI ({beating_vti / total_strategies: .0%})\n"
        output += f"• {positive_alpha}/{total_strategies} have positive alpha after costs ({positive_alpha / total_strategies: .0%})\n"
        
        if positive_alpha > 0: 
            best_strategy=comparisons[0]
            output += f"• Best strategy: {best_strategy.strategy_name.replace('_', ' ').title()} "
            output += f"(+{best_strategy.net_alpha_after_costs: .1%} alpha)\n"
        
        output += "\n💡 THE WSB REALITY CHECK: \n"
        output += "• If you can't beat SPY, just buy SPY\n"
        output += "• Trading costs and taxes eat into alpha\n"
        output += "• Higher Sharpe ratios matter more than raw returns\n"
        output += "• Consistency (win rate) is key for long - term success\n"
        output += "• Most active strategies fail to beat index funds long - term\n"
        
        output += "\n⚠️  IMPORTANT DISCLAIMERS: \n"
        output += "• Past performance doesn't predict future results\n"
        output += "• These are simplified backtests, not live trading\n"
        output += "• Real trading has slippage, commissions, and taxes\n"
        output += "• Market conditions change - strategies may stop working\n"
        output += "• Only trade with money you can afford to lose\n"
        
        return output


def main(): 
    parser=argparse.ArgumentParser(description="Index Fund Baseline Comparison Scanner")
    parser.add_argument('--period - months', type=int, default=6,
                       help='Period in months for comparison (default: 6)')
    parser.add_argument('--strategy', type=str,
                       help='Specific strategy to analyze (default: all)')
    parser.add_argument('--output', choices=['json', 'text'], default='text',
                       help='Output format')
    
    args=parser.parse_args()
    
    scanner=IndexBaselineScanner()
    
    if args.strategy: 
        if args.strategy not in scanner.wsb_strategies: 
            print(f"❌ Strategy '{args.strategy}' not found")
            print(f"Available strategies: {list(scanner.wsb_strategies.keys())}")
            return
        
        comparison=scanner.compare_strategy_performance(args.strategy, args.period_months)
        comparisons=[comparison]
    else: 
        comparisons=scanner.scan_all_strategies(args.period_months)
    
    if args.output== 'json': print(json.dumps([asdict(c) for c in comparisons], indent=2, default=str))
    else: 
        print(scanner.format_comparison_report(comparisons))
    
    # Summary stats
    if comparisons: 
        avg_alpha=np.mean([c.net_alpha_after_costs for c in comparisons])
        print(f"\n📊 Average net alpha across all strategies: {avg_alpha:+.2%}")
        
        if avg_alpha > 0: 
            print("✅ WSB strategies are beating the market on average!")
        else: 
            print("❌ Just buy index funds - WSB strategies underperforming")
    else: 
        print("\n❌ No strategy comparisons available")


if __name__== "__main__": main()