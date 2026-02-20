"""
REST API Client Library for WallStreetBots.

Ported from freqtrade's REST client.
Provides a clean Python interface for interacting with the trading API.
"""

import asyncio
import aiohttp
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TypeVar, Generic
from urllib.parse import urljoin
import time

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ApiError(Exception):
    """Base API error."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(ApiError):
    """Authentication failed."""
    pass


class RateLimitError(ApiError):
    """Rate limit exceeded."""
    pass


class NotFoundError(ApiError):
    """Resource not found."""
    pass


class ValidationError(ApiError):
    """Request validation failed."""
    pass


@dataclass
class ApiResponse(Generic[T]):
    """Wrapper for API responses."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    status_code: int = 200
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_response(cls, data: Any, status_code: int = 200) -> "ApiResponse":
        return cls(success=True, data=data, status_code=status_code)

    @classmethod
    def from_error(cls, error: str, status_code: int = 400) -> "ApiResponse":
        return cls(success=False, error=error, status_code=status_code)


@dataclass
class ClientConfig:
    """Configuration for the API client."""
    base_url: str = "http://localhost:8000/api/v1"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True


class HttpMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class TradingApiClient:
    """
    REST API client for WallStreetBots.

    Provides methods for all trading operations including:
    - Account management
    - Order execution
    - Position management
    - Strategy control
    - Market data
    - Backtest operations
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        """Initialize the API client."""
        self.config = config or ClientConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    async def __aenter__(self) -> "TradingApiClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Create the HTTP session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(ssl=self.config.verify_ssl)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        elif self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        return headers

    async def _request(
        self,
        method: HttpMethod,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> ApiResponse:
        """Make an API request."""
        if self._session is None:
            await self.connect()

        url = urljoin(self.config.base_url, endpoint)
        headers = self._get_headers()

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.request(
                    method.value,
                    url,
                    json=data,
                    params=params,
                    headers=headers,
                ) as response:
                    response_data = await response.json() if response.content_length else None

                    if response.status == 200:
                        return ApiResponse.from_response(response_data, response.status)
                    elif response.status == 201:
                        return ApiResponse.from_response(response_data, response.status)
                    elif response.status == 204:
                        return ApiResponse.from_response(None, response.status)
                    elif response.status == 401:
                        raise AuthenticationError(
                            "Authentication failed",
                            status_code=response.status,
                            response=response_data,
                        )
                    elif response.status == 404:
                        raise NotFoundError(
                            "Resource not found",
                            status_code=response.status,
                            response=response_data,
                        )
                    elif response.status == 422:
                        raise ValidationError(
                            "Validation error",
                            status_code=response.status,
                            response=response_data,
                        )
                    elif response.status == 429:
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                            continue
                        raise RateLimitError(
                            "Rate limit exceeded",
                            status_code=response.status,
                            response=response_data,
                        )
                    else:
                        error_msg = response_data.get("detail", "Unknown error") if response_data else "Unknown error"
                        return ApiResponse.from_error(error_msg, response.status)

            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                raise ApiError(f"Request failed: {e}") from e

        return ApiResponse.from_error("Max retries exceeded")

    # Authentication endpoints
    async def login(self, username: str, password: str) -> ApiResponse[dict]:
        """Authenticate and get access token."""
        response = await self._request(
            HttpMethod.POST,
            "/auth/login",
            data={"username": username, "password": password},
        )
        if response.success and response.data:
            self._token = response.data.get("access_token")
            if "expires_at" in response.data:
                self._token_expiry = datetime.fromisoformat(response.data["expires_at"])
        return response

    async def logout(self) -> ApiResponse:
        """Logout and invalidate token."""
        response = await self._request(HttpMethod.POST, "/auth/logout")
        self._token = None
        self._token_expiry = None
        return response

    async def refresh_token(self) -> ApiResponse[dict]:
        """Refresh access token."""
        response = await self._request(HttpMethod.POST, "/auth/refresh")
        if response.success and response.data:
            self._token = response.data.get("access_token")
        return response

    # Account endpoints
    async def get_account(self) -> ApiResponse[dict]:
        """Get account information."""
        return await self._request(HttpMethod.GET, "/account")

    async def get_balance(self) -> ApiResponse[dict]:
        """Get account balance."""
        return await self._request(HttpMethod.GET, "/account/balance")

    async def get_positions(self) -> ApiResponse[list]:
        """Get all open positions."""
        return await self._request(HttpMethod.GET, "/positions")

    async def get_position(self, symbol: str) -> ApiResponse[dict]:
        """Get position for a specific symbol."""
        return await self._request(HttpMethod.GET, f"/positions/{symbol}")

    async def close_position(self, symbol: str) -> ApiResponse[dict]:
        """Close a position."""
        return await self._request(HttpMethod.POST, f"/positions/{symbol}/close")

    # Order endpoints
    async def get_orders(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> ApiResponse[list]:
        """Get orders with optional filters."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        if symbol:
            params["symbol"] = symbol
        return await self._request(HttpMethod.GET, "/orders", params=params)

    async def get_order(self, order_id: str) -> ApiResponse[dict]:
        """Get a specific order."""
        return await self._request(HttpMethod.GET, f"/orders/{order_id}")

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        **kwargs,
    ) -> ApiResponse[dict]:
        """Create a new order."""
        data = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "time_in_force": time_in_force,
        }
        if price is not None:
            data["price"] = price
        if stop_price is not None:
            data["stop_price"] = stop_price
        data.update(kwargs)
        return await self._request(HttpMethod.POST, "/orders", data=data)

    async def cancel_order(self, order_id: str) -> ApiResponse[dict]:
        """Cancel an order."""
        return await self._request(HttpMethod.DELETE, f"/orders/{order_id}")

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> ApiResponse[dict]:
        """Cancel all orders, optionally filtered by symbol."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request(HttpMethod.DELETE, "/orders", params=params)

    # Trade history endpoints
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> ApiResponse[list]:
        """Get trade history."""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        return await self._request(HttpMethod.GET, "/trades", params=params)

    async def get_trade(self, trade_id: str) -> ApiResponse[dict]:
        """Get a specific trade."""
        return await self._request(HttpMethod.GET, f"/trades/{trade_id}")

    # Strategy endpoints
    async def get_strategies(self) -> ApiResponse[list]:
        """Get all strategies."""
        return await self._request(HttpMethod.GET, "/strategies")

    async def get_strategy(self, strategy_id: str) -> ApiResponse[dict]:
        """Get a specific strategy."""
        return await self._request(HttpMethod.GET, f"/strategies/{strategy_id}")

    async def start_strategy(self, strategy_id: str) -> ApiResponse[dict]:
        """Start a strategy."""
        return await self._request(HttpMethod.POST, f"/strategies/{strategy_id}/start")

    async def stop_strategy(self, strategy_id: str) -> ApiResponse[dict]:
        """Stop a strategy."""
        return await self._request(HttpMethod.POST, f"/strategies/{strategy_id}/stop")

    async def get_strategy_performance(self, strategy_id: str) -> ApiResponse[dict]:
        """Get strategy performance metrics."""
        return await self._request(HttpMethod.GET, f"/strategies/{strategy_id}/performance")

    async def update_strategy_config(
        self,
        strategy_id: str,
        config: dict,
    ) -> ApiResponse[dict]:
        """Update strategy configuration."""
        return await self._request(
            HttpMethod.PATCH,
            f"/strategies/{strategy_id}/config",
            data=config,
        )

    # Market data endpoints
    async def get_ticker(self, symbol: str) -> ApiResponse[dict]:
        """Get current ticker data."""
        return await self._request(HttpMethod.GET, f"/market/ticker/{symbol}")

    async def get_orderbook(self, symbol: str, depth: int = 20) -> ApiResponse[dict]:
        """Get order book."""
        return await self._request(
            HttpMethod.GET,
            f"/market/orderbook/{symbol}",
            params={"depth": depth},
        )

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ApiResponse[list]:
        """Get OHLCV candles."""
        params = {"interval": interval, "limit": limit}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        return await self._request(
            HttpMethod.GET,
            f"/market/candles/{symbol}",
            params=params,
        )

    # Backtest endpoints
    async def run_backtest(
        self,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000,
        symbols: Optional[list[str]] = None,
        config: Optional[dict] = None,
    ) -> ApiResponse[dict]:
        """Run a backtest."""
        data = {
            "strategy_id": strategy_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": initial_capital,
        }
        if symbols:
            data["symbols"] = symbols
        if config:
            data["config"] = config
        return await self._request(HttpMethod.POST, "/backtest/run", data=data)

    async def get_backtest_status(self, backtest_id: str) -> ApiResponse[dict]:
        """Get backtest status."""
        return await self._request(HttpMethod.GET, f"/backtest/{backtest_id}/status")

    async def get_backtest_results(self, backtest_id: str) -> ApiResponse[dict]:
        """Get backtest results."""
        return await self._request(HttpMethod.GET, f"/backtest/{backtest_id}/results")

    async def cancel_backtest(self, backtest_id: str) -> ApiResponse[dict]:
        """Cancel a running backtest."""
        return await self._request(HttpMethod.POST, f"/backtest/{backtest_id}/cancel")

    # Alert endpoints
    async def get_alerts(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> ApiResponse[list]:
        """Get alerts."""
        return await self._request(
            HttpMethod.GET,
            "/alerts",
            params={"active_only": active_only, "limit": limit},
        )

    async def create_alert(
        self,
        symbol: str,
        condition: str,
        value: float,
        message: Optional[str] = None,
        notify_channels: Optional[list[str]] = None,
    ) -> ApiResponse[dict]:
        """Create a price alert."""
        data = {
            "symbol": symbol,
            "condition": condition,
            "value": value,
        }
        if message:
            data["message"] = message
        if notify_channels:
            data["notify_channels"] = notify_channels
        return await self._request(HttpMethod.POST, "/alerts", data=data)

    async def delete_alert(self, alert_id: str) -> ApiResponse:
        """Delete an alert."""
        return await self._request(HttpMethod.DELETE, f"/alerts/{alert_id}")

    # Risk management endpoints
    async def get_risk_metrics(self) -> ApiResponse[dict]:
        """Get risk metrics."""
        return await self._request(HttpMethod.GET, "/risk/metrics")

    async def get_risk_limits(self) -> ApiResponse[dict]:
        """Get risk limits."""
        return await self._request(HttpMethod.GET, "/risk/limits")

    async def update_risk_limits(self, limits: dict) -> ApiResponse[dict]:
        """Update risk limits."""
        return await self._request(HttpMethod.PUT, "/risk/limits", data=limits)

    # System endpoints
    async def get_status(self) -> ApiResponse[dict]:
        """Get system status."""
        return await self._request(HttpMethod.GET, "/status")

    async def get_version(self) -> ApiResponse[dict]:
        """Get API version."""
        return await self._request(HttpMethod.GET, "/version")

    async def health_check(self) -> ApiResponse[dict]:
        """Health check endpoint."""
        return await self._request(HttpMethod.GET, "/health")


# Synchronous wrapper for non-async contexts
class SyncTradingApiClient:
    """Synchronous wrapper for TradingApiClient."""

    def __init__(self, config: Optional[ClientConfig] = None):
        self._async_client = TradingApiClient(config)

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    def __enter__(self) -> "SyncTradingApiClient":
        self._run_async(self._async_client.connect())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._run_async(self._async_client.close())

    # Auth methods
    def login(self, username: str, password: str) -> ApiResponse:
        return self._run_async(self._async_client.login(username, password))

    def logout(self) -> ApiResponse:
        return self._run_async(self._async_client.logout())

    # Account methods
    def get_account(self) -> ApiResponse:
        return self._run_async(self._async_client.get_account())

    def get_balance(self) -> ApiResponse:
        return self._run_async(self._async_client.get_balance())

    def get_positions(self) -> ApiResponse:
        return self._run_async(self._async_client.get_positions())

    def get_position(self, symbol: str) -> ApiResponse:
        return self._run_async(self._async_client.get_position(symbol))

    def close_position(self, symbol: str) -> ApiResponse:
        return self._run_async(self._async_client.close_position(symbol))

    # Order methods
    def get_orders(self, **kwargs) -> ApiResponse:
        return self._run_async(self._async_client.get_orders(**kwargs))

    def get_order(self, order_id: str) -> ApiResponse:
        return self._run_async(self._async_client.get_order(order_id))

    def create_order(self, **kwargs) -> ApiResponse:
        return self._run_async(self._async_client.create_order(**kwargs))

    def cancel_order(self, order_id: str) -> ApiResponse:
        return self._run_async(self._async_client.cancel_order(order_id))

    def cancel_all_orders(self, symbol: Optional[str] = None) -> ApiResponse:
        return self._run_async(self._async_client.cancel_all_orders(symbol))

    # Trade methods
    def get_trades(self, **kwargs) -> ApiResponse:
        return self._run_async(self._async_client.get_trades(**kwargs))

    def get_trade(self, trade_id: str) -> ApiResponse:
        return self._run_async(self._async_client.get_trade(trade_id))

    # Strategy methods
    def get_strategies(self) -> ApiResponse:
        return self._run_async(self._async_client.get_strategies())

    def get_strategy(self, strategy_id: str) -> ApiResponse:
        return self._run_async(self._async_client.get_strategy(strategy_id))

    def start_strategy(self, strategy_id: str) -> ApiResponse:
        return self._run_async(self._async_client.start_strategy(strategy_id))

    def stop_strategy(self, strategy_id: str) -> ApiResponse:
        return self._run_async(self._async_client.stop_strategy(strategy_id))

    def get_strategy_performance(self, strategy_id: str) -> ApiResponse:
        return self._run_async(self._async_client.get_strategy_performance(strategy_id))

    def update_strategy_config(self, strategy_id: str, config: dict) -> ApiResponse:
        return self._run_async(self._async_client.update_strategy_config(strategy_id, config))

    # Market data methods
    def get_ticker(self, symbol: str) -> ApiResponse:
        return self._run_async(self._async_client.get_ticker(symbol))

    def get_orderbook(self, symbol: str, depth: int = 20) -> ApiResponse:
        return self._run_async(self._async_client.get_orderbook(symbol, depth))

    def get_candles(self, symbol: str, **kwargs) -> ApiResponse:
        return self._run_async(self._async_client.get_candles(symbol, **kwargs))

    # Backtest methods
    def run_backtest(self, **kwargs) -> ApiResponse:
        return self._run_async(self._async_client.run_backtest(**kwargs))

    def get_backtest_status(self, backtest_id: str) -> ApiResponse:
        return self._run_async(self._async_client.get_backtest_status(backtest_id))

    def get_backtest_results(self, backtest_id: str) -> ApiResponse:
        return self._run_async(self._async_client.get_backtest_results(backtest_id))

    def cancel_backtest(self, backtest_id: str) -> ApiResponse:
        return self._run_async(self._async_client.cancel_backtest(backtest_id))

    # Alert methods
    def get_alerts(self, **kwargs) -> ApiResponse:
        return self._run_async(self._async_client.get_alerts(**kwargs))

    def create_alert(self, **kwargs) -> ApiResponse:
        return self._run_async(self._async_client.create_alert(**kwargs))

    def delete_alert(self, alert_id: str) -> ApiResponse:
        return self._run_async(self._async_client.delete_alert(alert_id))

    # Risk methods
    def get_risk_metrics(self) -> ApiResponse:
        return self._run_async(self._async_client.get_risk_metrics())

    def get_risk_limits(self) -> ApiResponse:
        return self._run_async(self._async_client.get_risk_limits())

    def update_risk_limits(self, limits: dict) -> ApiResponse:
        return self._run_async(self._async_client.update_risk_limits(limits))

    # System methods
    def get_status(self) -> ApiResponse:
        return self._run_async(self._async_client.get_status())

    def get_version(self) -> ApiResponse:
        return self._run_async(self._async_client.get_version())

    def health_check(self) -> ApiResponse:
        return self._run_async(self._async_client.health_check())


def create_client(
    base_url: str = "http://localhost:8000/api/v1",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    async_mode: bool = False,
) -> TradingApiClient | SyncTradingApiClient:
    """Factory function to create an API client."""
    config = ClientConfig(
        base_url=base_url,
        api_key=api_key,
        api_secret=api_secret,
    )
    if async_mode:
        return TradingApiClient(config)
    return SyncTradingApiClient(config)
