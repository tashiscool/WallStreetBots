"""
Database Schema for Risk Management
SQLite implementation matching the PostgreSQL schema from the bundle
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

class RiskDatabase:
    """SQLite database for risk management data"""
    
    def __init__(self, db_path: str = "risk_management.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Prices & returns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ts TIMESTAMP NOT NULL,
                    px REAL NOT NULL,
                    UNIQUE(symbol, ts)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS returns_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ts TIMESTAMP NOT NULL,
                    ret REAL NOT NULL,
                    UNIQUE(symbol, ts)
                )
            """)
            
            # Positions (live snapshot)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    qty REAL NOT NULL,
                    value REAL NOT NULL,
                    delta REAL DEFAULT 0,
                    gamma REAL DEFAULT 0,
                    vega REAL DEFAULT 0,
                    theta REAL DEFAULT 0,
                    strategy TEXT,
                    ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Risk limits
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    max_total_var REAL NOT NULL,
                    max_total_cvar REAL NOT NULL,
                    per_strategy TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Risk results (audit)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TIMESTAMP NOT NULL,
                    account_id TEXT NOT NULL,
                    alpha REAL NOT NULL,
                    var REAL NOT NULL,
                    cvar REAL NOT NULL,
                    lvar REAL,
                    exceptions_250 INTEGER,
                    kupiec_lr REAL,
                    details TEXT
                )
            """)
            
            # Strategy performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TIMESTAMP NOT NULL,
                    strategy_name TEXT NOT NULL,
                    var REAL NOT NULL,
                    cvar REAL NOT NULL,
                    utilization REAL NOT NULL,
                    pnl REAL NOT NULL,
                    details TEXT
                )
            """)
            
            # Risk alerts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TIMESTAMP NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    portfolio_impact REAL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    details TEXT
                )
            """)
            
            conn.commit()
    
    def insert_price_data(self, symbol: str, prices: pd.DataFrame):
        """Insert price data"""
        with sqlite3.connect(self.db_path) as conn:
            for _, row in prices.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO price_series (symbol, ts, px)
                    VALUES (?, ?, ?)
                """, (symbol, row['timestamp'], row['price']))
            conn.commit()
    
    def insert_returns_data(self, symbol: str, returns: pd.DataFrame):
        """Insert returns data"""
        with sqlite3.connect(self.db_path) as conn:
            for _, row in returns.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO returns_series (symbol, ts, ret)
                    VALUES (?, ?, ?)
                """, (symbol, row['timestamp'], row['return']))
            conn.commit()
    
    def get_returns_for_var(self, symbol: str, window: int = 250) -> pd.Series:
        """Get returns vector for VaR calculation"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT ret FROM returns_series
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, window))
            return pd.Series(df['ret'].values)
    
    def insert_risk_result(
        self,
        account_id: str,
        alpha: float,
        var: float,
        cvar: float,
        lvar: Optional[float] = None,
        exceptions_250: Optional[int] = None,
        kupiec_lr: Optional[float] = None,
        details: Optional[Dict] = None
    ):
        """Insert risk calculation result"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO risk_results 
                (ts, account_id, alpha, var, cvar, lvar, exceptions_250, kupiec_lr, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                account_id,
                alpha,
                var,
                cvar,
                lvar,
                exceptions_250,
                kupiec_lr,
                json.dumps(details) if details else None
            ))
            conn.commit()
    
    def get_latest_risk_result(self, account_id: str) -> Optional[Dict]:
        """Get latest risk result for account"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM risk_results
                WHERE account_id = ?
                ORDER BY ts DESC
                LIMIT 1
            """, (account_id,))
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def check_risk_limits(self, account_id: str, var: float, cvar: float) -> Dict[str, bool]:
        """Check if risk metrics are within limits"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT max_total_var, max_total_cvar FROM risk_limits
                WHERE account_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (account_id,))
            row = cursor.fetchone()
            
            if row:
                max_var, max_cvar = row
                return {
                    "var_within_limit": var <= max_var,
                    "cvar_within_limit": cvar <= max_cvar,
                    "var_utilization": var / max_var if max_var > 0 else 0,
                    "cvar_utilization": cvar / max_cvar if max_cvar > 0 else 0
                }
            else:
                # Default limits if none set
                return {
                    "var_within_limit": True,
                    "cvar_within_limit": True,
                    "var_utilization": 0,
                    "cvar_utilization": 0
                }
    
    def set_risk_limits(
        self,
        account_id: str,
        max_total_var: float,
        max_total_cvar: float,
        per_strategy: Dict[str, float]
    ):
        """Set risk limits for account"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO risk_limits (account_id, max_total_var, max_total_cvar, per_strategy)
                VALUES (?, ?, ?, ?)
            """, (account_id, max_total_var, max_total_cvar, json.dumps(per_strategy)))
            conn.commit()
    
    def insert_positions(self, account_id: str, positions: List[Dict]):
        """Insert current positions"""
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing positions for this account
            conn.execute("DELETE FROM positions WHERE account_id = ?", (account_id,))
            
            # Insert new positions
            for pos in positions:
                conn.execute("""
                    INSERT INTO positions 
                    (account_id, symbol, qty, value, delta, gamma, vega, theta, strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    account_id,
                    pos.get('symbol', ''),
                    pos.get('qty', 0),
                    pos.get('value', 0),
                    pos.get('delta', 0),
                    pos.get('gamma', 0),
                    pos.get('vega', 0),
                    pos.get('theta', 0),
                    pos.get('strategy', '')
                ))
            conn.commit()
    
    def get_positions(self, account_id: str) -> List[Dict]:
        """Get current positions for account"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM positions
                WHERE account_id = ?
                ORDER BY ts DESC
            """, (account_id,))
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    def insert_risk_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        portfolio_impact: float = 0.0,
        details: Optional[Dict] = None
    ):
        """Insert risk alert"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO risk_alerts 
                (ts, alert_type, severity, message, portfolio_impact, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                alert_type,
                severity,
                message,
                portfolio_impact,
                json.dumps(details) if details else None
            ))
            conn.commit()
    
    def get_active_alerts(self) -> List[Dict]:
        """Get unacknowledged alerts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM risk_alerts
                WHERE acknowledged = FALSE
                ORDER BY ts DESC
            """)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    def acknowledge_alert(self, alert_id: int):
        """Acknowledge an alert"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE risk_alerts
                SET acknowledged = TRUE
                WHERE id = ?
            """, (alert_id,))
            conn.commit()
    
    def get_risk_history(self, account_id: str, days: int = 30) -> pd.DataFrame:
        """Get risk history for account"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM risk_results
                WHERE account_id = ? AND ts >= datetime('now', '-{} days')
                ORDER BY ts DESC
            """.format(days)
            return pd.read_sql_query(query, conn, params=(account_id,))
    
    def compute_returns_from_prices(self, symbol: str):
        """Compute log returns from price data"""
        with sqlite3.connect(self.db_path) as conn:
            # Get price data
            prices_df = pd.read_sql_query("""
                SELECT ts, px FROM price_series
                WHERE symbol = ?
                ORDER BY ts
            """, conn, params=(symbol,))
            
            if len(prices_df) < 2:
                return
            
            # Compute log returns
            prices_df['return'] = np.log(prices_df['px'] / prices_df['px'].shift(1))
            returns_df = prices_df[['ts', 'return']].dropna()
            
            # Insert returns
            for _, row in returns_df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO returns_series (symbol, ts, ret)
                    VALUES (?, ?, ?)
                """, (symbol, row['ts'], row['return']))
            
            conn.commit()

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = RiskDatabase("test_risk.db")
    
    # Set up risk limits
    db.set_risk_limits(
        account_id="test_account",
        max_total_var=5000.0,  # $5K max VaR
        max_total_cvar=8000.0,  # $8K max CVaR
        per_strategy={
            "wsb_dip_bot": 2000.0,
            "index_baseline": 3000.0,
            "momentum_weeklies": 1500.0
        }
    )
    
    # Insert sample positions
    positions = [
        {
            'symbol': 'AAPL',
            'qty': 100,
            'value': 15000,
            'delta': 0.5,
            'gamma': 0.1,
            'vega': 0.2,
            'strategy': 'wsb_dip_bot'
        },
        {
            'symbol': 'SPY',
            'qty': 200,
            'value': 80000,
            'delta': 0.8,
            'gamma': 0.05,
            'vega': 0.15,
            'strategy': 'index_baseline'
        }
    ]
    db.insert_positions("test_account", positions)
    
    # Insert risk result
    db.insert_risk_result(
        account_id="test_account",
        alpha=0.99,
        var=3500.0,
        cvar=4200.0,
        lvar=3800.0,
        exceptions_250=3,
        kupiec_lr=1.2
    )
    
    # Check limits
    limits = db.check_risk_limits("test_account", 3500.0, 4200.0)
    print("Risk Limits Check:", limits)
    
    # Get positions
    positions = db.get_positions("test_account")
    print("Current Positions:", len(positions))
    
    print("Database initialized successfully!")
