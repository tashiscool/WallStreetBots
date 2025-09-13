"""Production Database Models and Management
Implements PostgreSQL database schema for production trading system

This replaces the JSON file system with proper relational database for:
- Trade tracking and history
- Position management
- Risk monitoring
- Performance analytics
- Audit trails
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import asyncpg


@dataclass
class Trade:
    """Production trade record"""

    id: int | None = None
    strategy_name: str = ""
    ticker: str = ""
    trade_type: str = ""  # 'stock', 'option', 'spread'
    action: str = ""  # 'buy', 'sell', 'open', 'close'
    quantity: int = 0
    entry_price: Decimal = Decimal("0.00")
    exit_price: Decimal | None = None
    pnl: Decimal | None = None
    commission: Decimal = Decimal("0.00")
    slippage: Decimal = Decimal("0.00")
    order_id: str = ""
    fill_timestamp: datetime | None = None
    exit_timestamp: datetime | None = None
    risk_amount: Decimal = Decimal("0.00")
    expected_return: Decimal = Decimal("0.00")
    actual_return: Decimal | None = None
    win: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Production position record"""

    id: int | None = None
    strategy_name: str = ""
    ticker: str = ""
    position_type: str = ""  # 'long', 'short', 'spread'
    quantity: int = 0
    avg_cost_basis: Decimal = Decimal("0.00")
    current_price: Decimal = Decimal("0.00")
    market_value: Decimal = Decimal("0.00")
    unrealized_pnl: Decimal = Decimal("0.00")
    realized_pnl: Decimal = Decimal("0.00")
    total_pnl: Decimal = Decimal("0.00")
    risk_amount: Decimal = Decimal("0.00")
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    entry_timestamp: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""

    id: int | None = None
    strategy_name: str = ""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0.00")
    avg_win: Decimal = Decimal("0.00")
    avg_loss: Decimal = Decimal("0.00")
    largest_win: Decimal = Decimal("0.00")
    largest_loss: Decimal = Decimal("0.00")
    total_pnl: Decimal = Decimal("0.00")
    gross_profit: Decimal = Decimal("0.00")
    gross_loss: Decimal = Decimal("0.00")
    profit_factor: Decimal = Decimal("0.00")
    sharpe_ratio: Decimal = Decimal("0.00")
    max_drawdown: Decimal = Decimal("0.00")
    kelly_fraction: Decimal = Decimal("0.00")
    var_95: Decimal = Decimal("0.00")
    expected_shortfall: Decimal = Decimal("0.00")
    calmar_ratio: Decimal = Decimal("0.00")
    sortino_ratio: Decimal = Decimal("0.00")
    last_calculated: datetime = field(default_factory=datetime.now)


@dataclass
class RiskMetrics:
    """Risk metrics record"""

    id: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    total_account_value: Decimal = Decimal("0.00")
    total_risk_exposure: Decimal = Decimal("0.00")
    risk_percentage: Decimal = Decimal("0.00")
    position_count: int = 0
    max_single_position_risk: Decimal = Decimal("0.00")
    correlation_risk: Decimal = Decimal("0.00")
    sector_concentration: dict[str, Decimal] = field(default_factory=dict)
    var_95_portfolio: Decimal = Decimal("0.00")
    beta_portfolio: Decimal = Decimal("1.00")
    daily_pnl: Decimal = Decimal("0.00")
    mtd_pnl: Decimal = Decimal("0.00")
    ytd_pnl: Decimal = Decimal("0.00")
    alerts: list[str] = field(default_factory=list)


class ProductionDatabaseManager:
    """Production PostgreSQL database manager"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pool: asyncpg.Pool | None = None

        # Database connection parameters
        self.db_host = config.get("db_host", "localhost")
        self.db_port = config.get("db_port", 5432)
        self.db_name = config.get("db_name", "wallstreetbots")
        self.db_user = config.get("db_user", "postgres")
        self.db_password = config.get("db_password", "")

        self.logger.info("Production Database Manager initialized")

    async def initialize(self) -> bool:
        """Initialize database connection pool"""
        try:
            self.logger.info("Initializing database connection pool")

            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                min_size=5,
                max_size=20,
                command_timeout=30,
            )

            # Create tables if they don't exist
            await self._create_tables()

            self.logger.info("Database initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connection pool closed")

    async def _create_tables(self):
        """Create production database tables"""
        try:
            async with self.pool.acquire() as conn:
                # Create strategies table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(50) UNIQUE NOT NULL,
                        description TEXT,
                        risk_level VARCHAR(20),
                        status VARCHAR(20) DEFAULT 'active',
                        max_position_risk DECIMAL(5,4) DEFAULT 0.02,
                        max_account_risk DECIMAL(5,4) DEFAULT 0.10,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Create trades table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        strategy_name VARCHAR(50),
                        ticker VARCHAR(20),
                        trade_type VARCHAR(20),
                        action VARCHAR(10),
                        quantity INTEGER,
                        entry_price DECIMAL(12,4),
                        exit_price DECIMAL(12,4),
                        pnl DECIMAL(12,4),
                        commission DECIMAL(8,4) DEFAULT 0.00,
                        slippage DECIMAL(8,4) DEFAULT 0.00,
                        order_id VARCHAR(100),
                        fill_timestamp TIMESTAMP,
                        exit_timestamp TIMESTAMP,
                        risk_amount DECIMAL(12,4),
                        expected_return DECIMAL(8,4),
                        actual_return DECIMAL(8,4),
                        win BOOLEAN,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Create positions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id SERIAL PRIMARY KEY,
                        strategy_name VARCHAR(50),
                        ticker VARCHAR(20),
                        position_type VARCHAR(20),
                        quantity INTEGER,
                        avg_cost_basis DECIMAL(12,4),
                        current_price DECIMAL(12,4),
                        market_value DECIMAL(12,4),
                        unrealized_pnl DECIMAL(12,4),
                        realized_pnl DECIMAL(12,4),
                        total_pnl DECIMAL(12,4),
                        risk_amount DECIMAL(12,4),
                        stop_loss_price DECIMAL(12,4),
                        take_profit_price DECIMAL(12,4),
                        entry_timestamp TIMESTAMP,
                        last_update TIMESTAMP DEFAULT NOW(),
                        is_active BOOLEAN DEFAULT TRUE,
                        metadata JSONB
                    )
                """)

                # Create strategy_performance table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id SERIAL PRIMARY KEY,
                        strategy_name VARCHAR(50) UNIQUE,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        win_rate DECIMAL(5,4) DEFAULT 0.00,
                        avg_win DECIMAL(8,4) DEFAULT 0.00,
                        avg_loss DECIMAL(8,4) DEFAULT 0.00,
                        largest_win DECIMAL(8,4) DEFAULT 0.00,
                        largest_loss DECIMAL(8,4) DEFAULT 0.00,
                        total_pnl DECIMAL(12,4) DEFAULT 0.00,
                        gross_profit DECIMAL(12,4) DEFAULT 0.00,
                        gross_loss DECIMAL(12,4) DEFAULT 0.00,
                        profit_factor DECIMAL(8,4) DEFAULT 0.00,
                        sharpe_ratio DECIMAL(6,4) DEFAULT 0.00,
                        max_drawdown DECIMAL(8,4) DEFAULT 0.00,
                        kelly_fraction DECIMAL(6,4) DEFAULT 0.00,
                        var_95 DECIMAL(8,4) DEFAULT 0.00,
                        expected_shortfall DECIMAL(8,4) DEFAULT 0.00,
                        calmar_ratio DECIMAL(6,4) DEFAULT 0.00,
                        sortino_ratio DECIMAL(6,4) DEFAULT 0.00,
                        last_calculated TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Create risk_metrics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS risk_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        total_account_value DECIMAL(12,4),
                        total_risk_exposure DECIMAL(12,4),
                        risk_percentage DECIMAL(6,4),
                        position_count INTEGER,
                        max_single_position_risk DECIMAL(8,4),
                        correlation_risk DECIMAL(6,4),
                        sector_concentration JSONB,
                        var_95_portfolio DECIMAL(8,4),
                        beta_portfolio DECIMAL(6,4) DEFAULT 1.00,
                        daily_pnl DECIMAL(12,4),
                        mtd_pnl DECIMAL(12,4),
                        ytd_pnl DECIMAL(12,4),
                        alerts JSONB
                    )
                """)

                # Create indexes for performance
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_strategy_ticker ON trades(strategy_name, ticker)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(created_at)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_name)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_positions_active ON positions(is_active)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON risk_metrics(timestamp)"
                )

                self.logger.info("Database tables created successfully")

        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise

    async def create_trade(self, trade: Trade) -> int:
        """Create new trade record"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO trades (
                        strategy_name, ticker, trade_type, action, quantity,
                        entry_price, commission, slippage, order_id, fill_timestamp,
                        risk_amount, expected_return, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                """

                trade_id = await conn.fetchval(
                    query,
                    trade.strategy_name,
                    trade.ticker,
                    trade.trade_type,
                    trade.action,
                    trade.quantity,
                    trade.entry_price,
                    trade.commission,
                    trade.slippage,
                    trade.order_id,
                    trade.fill_timestamp,
                    trade.risk_amount,
                    trade.expected_return,
                    json.dumps(trade.metadata),
                )

                self.logger.debug(f"Created trade record: {trade_id}")
                return trade_id

        except Exception as e:
            self.logger.error(f"Error creating trade record: {e}")
            raise

    async def update_trade(self, trade_id: int, updates: dict[str, Any]) -> bool:
        """Update trade record"""
        try:
            async with self.pool.acquire() as conn:
                # Build dynamic update query
                set_clauses = []
                values = []
                param_count = 1

                for field_name, value in updates.items():
                    set_clauses.append(f"{field_name} = ${param_count}")
                    values.append(value)
                    param_count += 1

                set_clauses.append(f"updated_at=${param_count}")
                values.append(datetime.now())
                values.append(trade_id)  # For WHERE clause

                query = f"""
                    UPDATE trades 
                    SET {", ".join(set_clauses)}
                    WHERE id = ${param_count + 1}
                """

                await conn.execute(query, *values)
                self.logger.debug(f"Updated trade: {trade_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error updating trade {trade_id}: {e}")
            return False

    async def create_position(self, position: Position) -> int:
        """Create new position record"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO positions (
                        strategy_name, ticker, position_type, quantity,
                        avg_cost_basis, current_price, market_value, unrealized_pnl,
                        risk_amount, stop_loss_price, take_profit_price,
                        entry_timestamp, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                """

                position_id = await conn.fetchval(
                    query,
                    position.strategy_name,
                    position.ticker,
                    position.position_type,
                    position.quantity,
                    position.avg_cost_basis,
                    position.current_price,
                    position.market_value,
                    position.unrealized_pnl,
                    position.risk_amount,
                    position.stop_loss_price,
                    position.take_profit_price,
                    position.entry_timestamp,
                    json.dumps(position.metadata),
                )

                self.logger.debug(f"Created position record: {position_id}")
                return position_id

        except Exception as e:
            self.logger.error(f"Error creating position record: {e}")
            raise

    async def get_active_positions(self, strategy_name: str | None = None) -> list[Position]:
        """Get active positions"""
        try:
            async with self.pool.acquire() as conn:
                if strategy_name:
                    query = "SELECT * FROM positions WHERE is_active = TRUE AND strategy_name = $1"
                    rows = await conn.fetch(query, strategy_name)
                else:
                    query = "SELECT * FROM positions WHERE is_active = TRUE"
                    rows = await conn.fetch(query)

                positions = []
                for row in rows:
                    position = Position(
                        id=row["id"],
                        strategy_name=row["strategy_name"],
                        ticker=row["ticker"],
                        position_type=row["position_type"],
                        quantity=row["quantity"],
                        avg_cost_basis=row["avg_cost_basis"],
                        current_price=row["current_price"],
                        market_value=row["market_value"],
                        unrealized_pnl=row["unrealized_pnl"],
                        realized_pnl=row["realized_pnl"],
                        total_pnl=row["total_pnl"],
                        risk_amount=row["risk_amount"],
                        stop_loss_price=row["stop_loss_price"],
                        take_profit_price=row["take_profit_price"],
                        entry_timestamp=row["entry_timestamp"],
                        last_update=row["last_update"],
                        is_active=row["is_active"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                    positions.append(position)

                return positions

        except Exception as e:
            self.logger.error(f"Error getting active positions: {e}")
            return []

    async def calculate_strategy_performance(
        self, strategy_name: str, period_days: int = 30
    ) -> StrategyPerformance:
        """Calculate strategy performance metrics"""
        try:
            async with self.pool.acquire() as conn:
                # Get trades for the period
                since_date = datetime.now() - timedelta(days=period_days)

                trades_query = """
                    SELECT * FROM trades 
                    WHERE strategy_name = $1 
                    AND created_at  >=  $2 
                    AND exit_price IS NOT NULL
                """

                trades = await conn.fetch(trades_query, strategy_name, since_date)

                if not trades:
                    return StrategyPerformance(strategy_name=strategy_name)

                # Calculate metrics
                total_trades = len(trades)
                winning_trades = sum(1 for t in trades if t["pnl"] and t["pnl"] > 0)
                losing_trades = total_trades - winning_trades

                win_rate = (
                    Decimal(winning_trades) / Decimal(total_trades)
                    if total_trades > 0
                    else Decimal("0")
                )

                wins = [t["pnl"] for t in trades if t["pnl"] and t["pnl"] > 0]
                losses = [t["pnl"] for t in trades if t["pnl"] and t["pnl"] < 0]

                avg_win = Decimal(sum(wins)) / Decimal(len(wins)) if wins else Decimal("0")
                avg_loss = Decimal(sum(losses)) / Decimal(len(losses)) if losses else Decimal("0")

                largest_win = Decimal(max(wins)) if wins else Decimal("0")
                largest_loss = Decimal(min(losses)) if losses else Decimal("0")

                total_pnl = sum(t["pnl"] for t in trades if t["pnl"])
                gross_profit = sum(wins) if wins else Decimal("0")
                gross_loss = abs(sum(losses)) if losses else Decimal("0")

                profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("0")

                # Kelly Criterion calculation
                if avg_loss != 0:
                    b = avg_win / abs(avg_loss)
                    p = win_rate
                    q = Decimal("1") - p
                    kelly_fraction = (b * p - q) / b if b > 0 else Decimal("0")
                else:
                    kelly_fraction = Decimal("0")

                performance = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    largest_win=largest_win,
                    largest_loss=largest_loss,
                    total_pnl=Decimal(total_pnl),
                    gross_profit=gross_profit,
                    gross_loss=gross_loss,
                    profit_factor=profit_factor,
                    kelly_fraction=kelly_fraction,
                    last_calculated=datetime.now(),
                )

                # Save to database
                await self._save_strategy_performance(performance)

                return performance

        except Exception as e:
            self.logger.error(f"Error calculating strategy performance: {e}")
            return StrategyPerformance(strategy_name=strategy_name)

    async def _save_strategy_performance(self, performance: StrategyPerformance):
        """Save strategy performance to database"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO strategy_performance (
                        strategy_name, total_trades, winning_trades, losing_trades,
                        win_rate, avg_win, avg_loss, largest_win, largest_loss,
                        total_pnl, gross_profit, gross_loss, profit_factor,
                        kelly_fraction, last_calculated
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (strategy_name) DO UPDATE SET
                        total_trades = EXCLUDED.total_trades,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        win_rate = EXCLUDED.win_rate,
                        avg_win = EXCLUDED.avg_win,
                        avg_loss = EXCLUDED.avg_loss,
                        largest_win = EXCLUDED.largest_win,
                        largest_loss = EXCLUDED.largest_loss,
                        total_pnl = EXCLUDED.total_pnl,
                        gross_profit = EXCLUDED.gross_profit,
                        gross_loss = EXCLUDED.gross_loss,
                        profit_factor = EXCLUDED.profit_factor,
                        kelly_fraction = EXCLUDED.kelly_fraction,
                        last_calculated = EXCLUDED.last_calculated
                """

                await conn.execute(
                    query,
                    performance.strategy_name,
                    performance.total_trades,
                    performance.winning_trades,
                    performance.losing_trades,
                    performance.win_rate,
                    performance.avg_win,
                    performance.avg_loss,
                    performance.largest_win,
                    performance.largest_loss,
                    performance.total_pnl,
                    performance.gross_profit,
                    performance.gross_loss,
                    performance.profit_factor,
                    performance.kelly_fraction,
                    performance.last_calculated,
                )

        except Exception as e:
            self.logger.error(f"Error saving strategy performance: {e}")


# Factory function for easy initialization
def create_database_manager(config: dict[str, Any]) -> ProductionDatabaseManager:
    """Create production database manager"""
    return ProductionDatabaseManager(config)


# Standalone testing
async def main():
    """Test database functionality"""
    logging.basicConfig(level=logging.INFO)

    # Test configuration
    config = {
        "db_host": "localhost",
        "db_port": 5432,
        "db_name": "wallstreetbots_test",
        "db_user": "postgres",
        "db_password": "password",
    }

    db = create_database_manager(config)

    try:
        # Initialize database
        await db.initialize()

        # Test trade creation
        trade = Trade(
            strategy_name="test_strategy",
            ticker="AAPL",
            trade_type="stock",
            action="buy",
            quantity=100,
            entry_price=Decimal("150.00"),
            risk_amount=Decimal("1000.00"),
            expected_return=Decimal("0.08"),
        )

        trade_id = await db.create_trade(trade)
        print(f"Created trade: {trade_id}")

        # Test performance calculation
        performance = await db.calculate_strategy_performance("test_strategy")
        print(f"Strategy performance: {performance}")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
