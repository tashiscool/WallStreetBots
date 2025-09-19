#!/usr/bin/env python3
"""WSB Dip Bot Configuration Examples

This file shows how to configure the WSB Dip Bot for different trading modes:
1. Conservative mode (default)
2. WSB mode (full account reinvestment)
3. Custom parameters

Usage:
    # Conservative mode (default)
    python -c "from examples.wsb_dip_bot_config import conservative_config; print(conservative_config)"
    
    # WSB mode (all-in)
    python -c "from examples.wsb_dip_bot_config import wsb_config; print(wsb_config)"
"""

# Conservative Configuration (Default)
conservative_config = {
    "run_lookback_days": 10,
    "run_threshold": 0.10,  # 10% minimum run
    "dip_threshold": -0.03,  # -3% minimum dip
    "target_dte_days": 30,   # 30 days to expiry
    "otm_percentage": 0.05,  # 5% out of the money
    "max_position_size": 0.20,  # 20% max position size
    "target_multiplier": 3.0,  # 3x profit target
    "delta_target": 0.60,  # Delta exit threshold
    "wsb_mode": False,  # Use risk-based sizing
    "universe": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AVGO", 
        "TSLA", "AMZN", "NFLX", "CRM", "COST", "ADBE", "V", "MA", "LIN"
    ]
}

# WSB Mode Configuration (Full Account Reinvestment)
wsb_config = {
    "run_lookback_days": 10,
    "run_threshold": 0.10,  # 10% minimum run
    "dip_threshold": -0.03,  # -3% minimum dip
    "target_dte_days": 30,   # 30 days to expiry
    "otm_percentage": 0.05,  # 5% out of the money
    "max_position_size": 1.0,  # Ignored in WSB mode
    "target_multiplier": 3.0,  # Fixed 3x profit target
    "delta_target": 0.60,  # Delta exit threshold
    "wsb_mode": True,  # Use full account cash
    "universe": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AVGO", 
        "TSLA", "AMZN", "NFLX", "CRM", "COST", "ADBE", "V", "MA", "LIN"
    ]
}

# Aggressive Configuration (Higher thresholds)
aggressive_config = {
    "run_lookback_days": 5,   # Shorter lookback
    "run_threshold": 0.15,    # 15% minimum run
    "dip_threshold": -0.05,   # -5% minimum dip
    "target_dte_days": 21,    # 21 days to expiry
    "otm_percentage": 0.03,   # 3% out of the money (closer to ATM)
    "max_position_size": 0.30,  # 30% max position size
    "target_multiplier": 2.5,  # 2.5x profit target
    "delta_target": 0.70,  # Higher delta threshold
    "wsb_mode": False,
    "universe": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA", "AMZN"
    ]
}

# Example usage in production
if __name__ == "__main__":
    print("WSB Dip Bot Configuration Examples:")
    print("\n1. Conservative Mode:")
    print(f"   - Max position size: {conservative_config['max_position_size']:.0%}")
    print(f"   - WSB mode: {conservative_config['wsb_mode']}")
    print(f"   - Profit target: {conservative_config['target_multiplier']:.1f}x")
    
    print("\n2. WSB Mode (All-in):")
    print(f"   - Max position size: {wsb_config['max_position_size']:.0%} (ignored)")
    print(f"   - WSB mode: {wsb_config['wsb_mode']}")
    print(f"   - Profit target: {wsb_config['target_multiplier']:.1f}x")
    print("   - ⚠️  Uses FULL ACCOUNT CASH for each trade!")
    
    print("\n3. Aggressive Mode:")
    print(f"   - Max position size: {aggressive_config['max_position_size']:.0%}")
    print(f"   - WSB mode: {aggressive_config['wsb_mode']}")
    print(f"   - Profit target: {aggressive_config['target_multiplier']:.1f}x")
    print(f"   - Higher thresholds: {aggressive_config['run_threshold']:.0%} run, {aggressive_config['dip_threshold']:.0%} dip")
