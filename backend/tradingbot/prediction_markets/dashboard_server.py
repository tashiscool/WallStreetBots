"""
Real-Time WebSocket Dashboard Server for Prediction Market Arbitrage.

Synthesized from:
- polymarket-arbitrage: FastAPI + WebSocket state broadcasting
- kalshi-polymarket-arbitrage-bot: REST API for bot control

Provides live visibility into arbitrage operations.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)

# Try to import FastAPI (optional dependency)
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Dashboard server disabled.")


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


@dataclass
class DashboardState:
    """
    Complete state for dashboard display.

    From polymarket-arbitrage: Tracks all relevant data.
    """
    # Market data
    polymarket_markets: int = 0
    kalshi_markets: int = 0
    matched_pairs: int = 0

    # Order books (last 50)
    orderbook_snapshots: deque = field(default_factory=lambda: deque(maxlen=50))

    # Opportunities (last 100)
    opportunities: deque = field(default_factory=lambda: deque(maxlen=100))
    active_opportunities: List[Dict] = field(default_factory=list)

    # Trades (last 100)
    trades: deque = field(default_factory=lambda: deque(maxlen=100))

    # Portfolio state
    portfolio: Dict[str, Any] = field(default_factory=dict)

    # Risk metrics
    risk_metrics: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "total_opportunities": 0,
        "opportunities_executed": 0,
        "total_profit": 0.0,
        "avg_opportunity_duration_ms": 0.0,
        "uptime_seconds": 0,
    })

    # Timing statistics
    timing_stats: Dict[str, Any] = field(default_factory=dict)

    # Matching progress
    matching_status: str = "idle"  # idle, loading, matching, complete, error
    matching_progress: float = 0.0

    # System status
    is_running: bool = False
    dry_run: bool = True
    started_at: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "polymarket_markets": self.polymarket_markets,
            "kalshi_markets": self.kalshi_markets,
            "matched_pairs": self.matched_pairs,
            "orderbook_snapshots": list(self.orderbook_snapshots),
            "opportunities": list(self.opportunities),
            "active_opportunities": self.active_opportunities,
            "trades": list(self.trades),
            "portfolio": self.portfolio,
            "risk_metrics": self.risk_metrics,
            "stats": self.stats,
            "timing_stats": self.timing_stats,
            "matching_status": self.matching_status,
            "matching_progress": self.matching_progress,
            "is_running": self.is_running,
            "dry_run": self.dry_run,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_update": self.last_update.isoformat(),
            "uptime_seconds": (datetime.now() - self.started_at).total_seconds() if self.started_at else 0,
        }


class ConnectionManager:
    """
    Manages WebSocket connections.

    From polymarket-arbitrage: Broadcasts to all connected clients.
    """

    def __init__(self):
        self._active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and store WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self._active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        async with self._lock:
            self._active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self._active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        if not self._active_connections:
            return

        json_message = json.dumps(message, cls=DecimalEncoder)

        disconnected = set()
        async with self._lock:
            for connection in self._active_connections:
                try:
                    await connection.send_text(json_message)
                except Exception as e:
                    logger.warning(f"Failed to send to client: {e}")
                    disconnected.add(connection)

        # Clean up disconnected
        for conn in disconnected:
            await self.disconnect(conn)

    @property
    def connection_count(self) -> int:
        return len(self._active_connections)


def create_dashboard_app(
    arbitrage_engine=None,
    cors_origins: List[str] = None,
) -> "FastAPI":
    """
    Create FastAPI dashboard application.

    Args:
        arbitrage_engine: Optional reference to ArbitrageEngine
        cors_origins: Allowed CORS origins

    Returns:
        FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title="Prediction Market Arbitrage Dashboard",
        description="Real-time dashboard for cross-platform arbitrage",
        version="1.0.0",
    )

    # CORS for development
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    app.state.dashboard_state = DashboardState()
    app.state.connection_manager = ConnectionManager()
    app.state.arbitrage_engine = arbitrage_engine

    # ---------------------------------------------------------------------------
    # REST Endpoints
    # ---------------------------------------------------------------------------

    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "service": "arbitrage-dashboard"}

    @app.get("/state")
    async def get_state():
        """Get complete dashboard state."""
        return JSONResponse(
            content=app.state.dashboard_state.to_dict(),
            media_type="application/json",
        )

    @app.get("/markets")
    async def get_markets():
        """Get market counts."""
        state = app.state.dashboard_state
        return {
            "polymarket": state.polymarket_markets,
            "kalshi": state.kalshi_markets,
            "matched": state.matched_pairs,
        }

    @app.get("/opportunities")
    async def get_opportunities():
        """Get recent opportunities."""
        return {
            "opportunities": list(app.state.dashboard_state.opportunities),
            "active": app.state.dashboard_state.active_opportunities,
        }

    @app.get("/trades")
    async def get_trades():
        """Get recent trades."""
        return {"trades": list(app.state.dashboard_state.trades)}

    @app.get("/stats")
    async def get_stats():
        """Get statistics."""
        return app.state.dashboard_state.stats

    @app.get("/risk")
    async def get_risk():
        """Get risk metrics."""
        return app.state.dashboard_state.risk_metrics

    @app.get("/portfolio")
    async def get_portfolio():
        """Get portfolio state."""
        return app.state.dashboard_state.portfolio

    @app.get("/matching")
    async def get_matching_progress():
        """Get market matching progress."""
        state = app.state.dashboard_state
        return {
            "status": state.matching_status,
            "progress": state.matching_progress,
            "polymarket_markets": state.polymarket_markets,
            "kalshi_markets": state.kalshi_markets,
            "matched_pairs": state.matched_pairs,
        }

    # ---------------------------------------------------------------------------
    # Control Endpoints
    # ---------------------------------------------------------------------------

    @app.post("/start")
    async def start_engine():
        """Start the arbitrage engine."""
        engine = app.state.arbitrage_engine
        if engine is None:
            raise HTTPException(status_code=400, detail="Engine not configured")

        try:
            await engine.start()
            app.state.dashboard_state.is_running = True
            app.state.dashboard_state.started_at = datetime.now()
            return {"status": "started"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/stop")
    async def stop_engine():
        """Stop the arbitrage engine."""
        engine = app.state.arbitrage_engine
        if engine is None:
            raise HTTPException(status_code=400, detail="Engine not configured")

        try:
            await engine.stop()
            app.state.dashboard_state.is_running = False
            return {"status": "stopped"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/reset")
    async def reset_stats():
        """Reset statistics."""
        app.state.dashboard_state.stats = {
            "total_opportunities": 0,
            "opportunities_executed": 0,
            "total_profit": 0.0,
            "avg_opportunity_duration_ms": 0.0,
            "uptime_seconds": 0,
        }
        return {"status": "reset"}

    # ---------------------------------------------------------------------------
    # WebSocket Endpoint
    # ---------------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time updates.

        Clients receive state updates automatically.
        """
        manager = app.state.connection_manager
        await manager.connect(websocket)

        try:
            # Send initial state
            await websocket.send_text(
                json.dumps({
                    "type": "initial_state",
                    "data": app.state.dashboard_state.to_dict(),
                }, cls=DecimalEncoder)
            )

            # Keep connection alive, listen for messages
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0
                    )
                    # Handle incoming messages (heartbeat, commands, etc.)
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))

        except WebSocketDisconnect:
            await manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await manager.disconnect(websocket)

    return app


class DashboardIntegration:
    """
    Integration helper for connecting ArbitrageEngine to Dashboard.

    From polymarket-arbitrage: Periodic state updates.
    """

    def __init__(
        self,
        app: "FastAPI",
        update_interval: float = 1.0,
    ):
        self._app = app
        self._update_interval = update_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start periodic updates."""
        self._running = True
        self._task = asyncio.create_task(self._update_loop())
        logger.info("Dashboard integration started")

    async def stop(self) -> None:
        """Stop periodic updates."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Dashboard integration stopped")

    async def _update_loop(self) -> None:
        """Periodically broadcast state updates."""
        while self._running:
            try:
                state = self._app.state.dashboard_state
                state.last_update = datetime.now()

                # Broadcast to all connected clients
                await self._app.state.connection_manager.broadcast({
                    "type": "state_update",
                    "data": state.to_dict(),
                })

                await asyncio.sleep(self._update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1.0)

    # ---------------------------------------------------------------------------
    # State Update Methods (call from ArbitrageEngine)
    # ---------------------------------------------------------------------------

    def update_market_counts(
        self,
        polymarket: int,
        kalshi: int,
        matched: int,
    ) -> None:
        """Update market counts."""
        state = self._app.state.dashboard_state
        state.polymarket_markets = polymarket
        state.kalshi_markets = kalshi
        state.matched_pairs = matched

    def update_matching_progress(
        self,
        status: str,
        progress: float,
    ) -> None:
        """Update matching progress."""
        state = self._app.state.dashboard_state
        state.matching_status = status
        state.matching_progress = progress

    def add_opportunity(self, opportunity: Dict[str, Any]) -> None:
        """Add detected opportunity."""
        state = self._app.state.dashboard_state
        opportunity["detected_at"] = datetime.now().isoformat()
        state.opportunities.append(opportunity)
        state.stats["total_opportunities"] += 1

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add executed trade."""
        state = self._app.state.dashboard_state
        trade["executed_at"] = datetime.now().isoformat()
        state.trades.append(trade)
        state.stats["opportunities_executed"] += 1
        state.stats["total_profit"] += trade.get("profit", 0)

    def add_orderbook_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Add order book snapshot."""
        state = self._app.state.dashboard_state
        snapshot["timestamp"] = datetime.now().isoformat()
        state.orderbook_snapshots.append(snapshot)

    def update_portfolio(self, portfolio: Dict[str, Any]) -> None:
        """Update portfolio state."""
        self._app.state.dashboard_state.portfolio = portfolio

    def update_risk_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update risk metrics."""
        self._app.state.dashboard_state.risk_metrics = metrics

    def update_timing_stats(self, stats: Dict[str, Any]) -> None:
        """Update timing statistics."""
        self._app.state.dashboard_state.timing_stats = stats


async def run_dashboard_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    arbitrage_engine=None,
) -> None:
    """
    Run the dashboard server.

    Args:
        host: Host to bind to
        port: Port to listen on
        arbitrage_engine: Optional ArbitrageEngine reference
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required. Install: pip install fastapi uvicorn")

    import uvicorn

    app = create_dashboard_app(arbitrage_engine=arbitrage_engine)

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    logger.info(f"Starting dashboard server on {host}:{port}")
    await server.serve()
