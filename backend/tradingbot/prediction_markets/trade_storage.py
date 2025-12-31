"""
Trade Storage Service for Prediction Market Arbitrage.

Synthesized from:
- kalshi-polymarket-arbitrage-bot: Supabase integration, batch storage
- Trade audit trail and historical analysis

Persistent storage for trades and opportunities.
"""

import asyncio
import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageTradeRecord:
    """
    Complete record of an arbitrage trade.

    From kalshi-polymarket-arbitrage-bot: Full audit trail.
    """
    # Identifiers
    trade_id: str
    opportunity_id: str

    # Market info
    polymarket_market_id: str
    kalshi_market_id: str

    # Strategy
    strategy: str
    direction: str  # "poly_yes_kalshi_no" or "kalshi_yes_poly_no"

    # Order details
    polymarket_order_id: Optional[str] = None
    kalshi_order_id: Optional[str] = None

    # Prices
    poly_price: Decimal = Decimal("0")
    kalshi_price: Decimal = Decimal("0")
    combined_cost: Decimal = Decimal("0")

    # Size
    size: int = 0

    # Profit
    expected_profit: Decimal = Decimal("0")
    actual_profit: Optional[Decimal] = None

    # Fees
    poly_fee: Decimal = Decimal("0")
    kalshi_fee: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")

    # Status
    status: str = "pending"  # pending, success, partial, failed
    poly_status: str = "pending"
    kalshi_status: str = "pending"

    # Timestamps
    detected_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error info
    error_message: Optional[str] = None

    # Dry run flag
    is_dry_run: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "trade_id": self.trade_id,
            "opportunity_id": self.opportunity_id,
            "polymarket_market_id": self.polymarket_market_id,
            "kalshi_market_id": self.kalshi_market_id,
            "strategy": self.strategy,
            "direction": self.direction,
            "polymarket_order_id": self.polymarket_order_id,
            "kalshi_order_id": self.kalshi_order_id,
            "poly_price": float(self.poly_price),
            "kalshi_price": float(self.kalshi_price),
            "combined_cost": float(self.combined_cost),
            "size": self.size,
            "expected_profit": float(self.expected_profit),
            "actual_profit": float(self.actual_profit) if self.actual_profit else None,
            "poly_fee": float(self.poly_fee),
            "kalshi_fee": float(self.kalshi_fee),
            "total_fees": float(self.total_fees),
            "status": self.status,
            "poly_status": self.poly_status,
            "kalshi_status": self.kalshi_status,
            "detected_at": self.detected_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "is_dry_run": self.is_dry_run,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArbitrageTradeRecord":
        """Create from dictionary."""
        return cls(
            trade_id=data["trade_id"],
            opportunity_id=data["opportunity_id"],
            polymarket_market_id=data["polymarket_market_id"],
            kalshi_market_id=data["kalshi_market_id"],
            strategy=data["strategy"],
            direction=data["direction"],
            polymarket_order_id=data.get("polymarket_order_id"),
            kalshi_order_id=data.get("kalshi_order_id"),
            poly_price=Decimal(str(data.get("poly_price", 0))),
            kalshi_price=Decimal(str(data.get("kalshi_price", 0))),
            combined_cost=Decimal(str(data.get("combined_cost", 0))),
            size=data.get("size", 0),
            expected_profit=Decimal(str(data.get("expected_profit", 0))),
            actual_profit=Decimal(str(data["actual_profit"])) if data.get("actual_profit") else None,
            poly_fee=Decimal(str(data.get("poly_fee", 0))),
            kalshi_fee=Decimal(str(data.get("kalshi_fee", 0))),
            total_fees=Decimal(str(data.get("total_fees", 0))),
            status=data.get("status", "pending"),
            poly_status=data.get("poly_status", "pending"),
            kalshi_status=data.get("kalshi_status", "pending"),
            detected_at=datetime.fromisoformat(data["detected_at"]),
            executed_at=datetime.fromisoformat(data["executed_at"]) if data.get("executed_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error_message=data.get("error_message"),
            is_dry_run=data.get("is_dry_run", True),
        )


class TradeStorageBackend(ABC):
    """Abstract base class for trade storage backends."""

    @abstractmethod
    async def store_trade(self, record: ArbitrageTradeRecord) -> None:
        """Store a single trade record."""
        pass

    @abstractmethod
    async def store_trades_batch(self, records: List[ArbitrageTradeRecord]) -> None:
        """Store multiple trade records."""
        pass

    @abstractmethod
    async def get_trade(self, trade_id: str) -> Optional[ArbitrageTradeRecord]:
        """Get trade by ID."""
        pass

    @abstractmethod
    async def get_trades(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        is_dry_run: Optional[bool] = None,
    ) -> List[ArbitrageTradeRecord]:
        """Get trades with optional filters."""
        pass

    @abstractmethod
    async def get_trades_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        pass


class SQLiteStorageBackend(TradeStorageBackend):
    """
    SQLite-based trade storage.

    Good for local development and single-instance deployments.
    """

    def __init__(self, db_path: str = "data/arbitrage_trades.db"):
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS arbitrage_trades (
                    trade_id TEXT PRIMARY KEY,
                    opportunity_id TEXT NOT NULL,
                    polymarket_market_id TEXT NOT NULL,
                    kalshi_market_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    polymarket_order_id TEXT,
                    kalshi_order_id TEXT,
                    poly_price REAL NOT NULL,
                    kalshi_price REAL NOT NULL,
                    combined_cost REAL NOT NULL,
                    size INTEGER NOT NULL,
                    expected_profit REAL NOT NULL,
                    actual_profit REAL,
                    poly_fee REAL NOT NULL,
                    kalshi_fee REAL NOT NULL,
                    total_fees REAL NOT NULL,
                    status TEXT NOT NULL,
                    poly_status TEXT NOT NULL,
                    kalshi_status TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    executed_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    is_dry_run INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status
                ON arbitrage_trades(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_dry_run
                ON arbitrage_trades(is_dry_run)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_detected
                ON arbitrage_trades(detected_at)
            """)

            conn.commit()

    async def store_trade(self, record: ArbitrageTradeRecord) -> None:
        """Store a single trade record."""
        await self.store_trades_batch([record])

    async def store_trades_batch(self, records: List[ArbitrageTradeRecord]) -> None:
        """Store multiple trade records."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                for record in records:
                    data = record.to_dict()
                    conn.execute("""
                        INSERT OR REPLACE INTO arbitrage_trades (
                            trade_id, opportunity_id, polymarket_market_id,
                            kalshi_market_id, strategy, direction,
                            polymarket_order_id, kalshi_order_id,
                            poly_price, kalshi_price, combined_cost,
                            size, expected_profit, actual_profit,
                            poly_fee, kalshi_fee, total_fees,
                            status, poly_status, kalshi_status,
                            detected_at, executed_at, completed_at,
                            error_message, is_dry_run
                        ) VALUES (
                            :trade_id, :opportunity_id, :polymarket_market_id,
                            :kalshi_market_id, :strategy, :direction,
                            :polymarket_order_id, :kalshi_order_id,
                            :poly_price, :kalshi_price, :combined_cost,
                            :size, :expected_profit, :actual_profit,
                            :poly_fee, :kalshi_fee, :total_fees,
                            :status, :poly_status, :kalshi_status,
                            :detected_at, :executed_at, :completed_at,
                            :error_message, :is_dry_run
                        )
                    """, data)
                conn.commit()

        logger.debug(f"Stored {len(records)} trade records")

    async def get_trade(self, trade_id: str) -> Optional[ArbitrageTradeRecord]:
        """Get trade by ID."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM arbitrage_trades WHERE trade_id = ?",
                (trade_id,)
            )
            row = cursor.fetchone()

            if row:
                return ArbitrageTradeRecord.from_dict(dict(row))
            return None

    async def get_trades(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        is_dry_run: Optional[bool] = None,
    ) -> List[ArbitrageTradeRecord]:
        """Get trades with optional filters."""
        query = "SELECT * FROM arbitrage_trades WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if is_dry_run is not None:
            query += " AND is_dry_run = ?"
            params.append(1 if is_dry_run else 0)

        query += " ORDER BY detected_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [ArbitrageTradeRecord.from_dict(dict(row)) for row in rows]

    async def get_trades_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with sqlite3.connect(self._db_path) as conn:
            # Total counts
            total = conn.execute(
                "SELECT COUNT(*) FROM arbitrage_trades"
            ).fetchone()[0]

            # By status
            by_status = {}
            for status in ["pending", "success", "partial", "failed"]:
                count = conn.execute(
                    "SELECT COUNT(*) FROM arbitrage_trades WHERE status = ?",
                    (status,)
                ).fetchone()[0]
                by_status[status] = count

            # Profits (successful only)
            profit_result = conn.execute("""
                SELECT
                    SUM(actual_profit) as total_profit,
                    AVG(actual_profit) as avg_profit,
                    SUM(total_fees) as total_fees,
                    SUM(size) as total_volume
                FROM arbitrage_trades
                WHERE status = 'success' AND is_dry_run = 0
            """).fetchone()

            # Dry run stats
            dry_run_count = conn.execute(
                "SELECT COUNT(*) FROM arbitrage_trades WHERE is_dry_run = 1"
            ).fetchone()[0]

            live_count = conn.execute(
                "SELECT COUNT(*) FROM arbitrage_trades WHERE is_dry_run = 0"
            ).fetchone()[0]

            return {
                "total_trades": total,
                "by_status": by_status,
                "total_profit": profit_result[0] or 0,
                "avg_profit": profit_result[1] or 0,
                "total_fees": profit_result[2] or 0,
                "total_volume": profit_result[3] or 0,
                "dry_run_trades": dry_run_count,
                "live_trades": live_count,
                "success_rate": (
                    by_status.get("success", 0) / total * 100
                    if total > 0 else 0
                ),
            }


class TradeStorageService:
    """
    Trade storage service with batch accumulation.

    From kalshi-polymarket-arbitrage-bot: Batch storage with periodic flush.
    """

    def __init__(
        self,
        backend: Optional[TradeStorageBackend] = None,
        batch_size: int = 10,
        flush_interval: float = 300.0,  # 5 minutes
    ):
        """
        Initialize storage service.

        Args:
            backend: Storage backend (defaults to SQLite)
            batch_size: Flush when batch reaches this size
            flush_interval: Flush every N seconds
        """
        self._backend = backend or SQLiteStorageBackend()
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._batch: List[ArbitrageTradeRecord] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the storage service."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Trade storage service started")

    async def stop(self) -> None:
        """Stop the service and flush remaining records."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()
        logger.info("Trade storage service stopped")

    async def store(self, record: ArbitrageTradeRecord) -> None:
        """
        Add record to batch.

        Flushes automatically when batch size is reached.
        """
        async with self._lock:
            self._batch.append(record)

            if len(self._batch) >= self._batch_size:
                await self._flush_locked()

    async def _flush(self) -> None:
        """Flush batch to backend."""
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Flush batch (must hold lock)."""
        if not self._batch:
            return

        batch = self._batch.copy()
        self._batch.clear()

        try:
            await self._backend.store_trades_batch(batch)
            logger.info(f"Flushed {len(batch)} trade records to storage")
        except Exception as e:
            logger.error(f"Failed to flush trades: {e}")
            # Re-add to batch for retry
            self._batch.extend(batch)

    async def _periodic_flush(self) -> None:
        """Periodically flush batch."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    async def get_trade(self, trade_id: str) -> Optional[ArbitrageTradeRecord]:
        """Get trade by ID."""
        return await self._backend.get_trade(trade_id)

    async def get_trades(
        self,
        limit: int = 100,
        **kwargs,
    ) -> List[ArbitrageTradeRecord]:
        """Get trades with filters."""
        return await self._backend.get_trades(limit=limit, **kwargs)

    async def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return await self._backend.get_trades_summary()


# =============================================================================
# Singleton Instance
# =============================================================================

_trade_storage: Optional[TradeStorageService] = None


def get_trade_storage() -> TradeStorageService:
    """Get or create the global trade storage service."""
    global _trade_storage
    if _trade_storage is None:
        _trade_storage = TradeStorageService()
    return _trade_storage


async def init_trade_storage(
    db_path: str = "data/arbitrage_trades.db",
    batch_size: int = 10,
    flush_interval: float = 300.0,
) -> TradeStorageService:
    """
    Initialize and start the trade storage service.

    Args:
        db_path: Path to SQLite database
        batch_size: Batch size for storage
        flush_interval: Flush interval in seconds

    Returns:
        Started TradeStorageService
    """
    global _trade_storage
    backend = SQLiteStorageBackend(db_path)
    _trade_storage = TradeStorageService(
        backend=backend,
        batch_size=batch_size,
        flush_interval=flush_interval,
    )
    await _trade_storage.start()
    return _trade_storage
