"""OpenAPI schema generation for WallStreetBots API.

Generates OpenAPI 3.0 specification for all API endpoints.
"""

import os
from typing import Any, Dict


def get_openapi_schema() -> Dict[str, Any]:
    """Generate OpenAPI 3.0 schema for the WallStreetBots API."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "WallStreetBots Trading API",
            "description": """
## WallStreetBots Trading Platform API

A comprehensive trading platform API providing:
- **Automated Trading Strategies**: 10+ pre-built strategies including WSB Dip Bot, Wheel Strategy, and more
- **Options Trading**: Full support for options pricing, Greeks calculation, and exotic spreads
- **Risk Management**: Position sizing, stop-loss management, and portfolio analytics
- **Real-time Data**: Integration with Alpaca, Polygon, and other market data providers

### Authentication
All API endpoints (except health checks) require authentication. Use Auth0 for OAuth2.0 authentication.

### Rate Limiting
- Anonymous: 60 requests/minute
- Authenticated: 120 requests/minute

### Correlation IDs
All responses include `X-Correlation-ID` header for distributed tracing.
            """.strip(),
            "version": "1.0.0",
            "contact": {
                "name": "WallStreetBots Team",
                "url": "https://github.com/wallstreetbots",
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT",
            },
        },
        "servers": [
            {
                "url": os.environ.get("API_BASE_URL", "http://localhost:8000"),
                "description": "Current environment",
            },
        ],
        "tags": [
            {"name": "Health", "description": "Health check endpoints for monitoring"},
            {"name": "Authentication", "description": "Login and logout endpoints"},
            {"name": "Dashboard", "description": "Trading dashboard and overview"},
            {"name": "Positions", "description": "Position management"},
            {"name": "Orders", "description": "Order management"},
            {"name": "Strategies", "description": "Trading strategy management"},
            {"name": "Backtesting", "description": "Strategy backtesting"},
            {"name": "Risk", "description": "Risk management and analytics"},
            {"name": "Alerts", "description": "Alert configuration"},
            {"name": "System", "description": "System status and configuration"},
        ],
        "paths": {
            # Health Check Endpoints
            "/health/": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Comprehensive health check",
                    "description": "Returns health status of all system components including database, Redis, and trading systems.",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "System is healthy or degraded but operational",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthCheckResponse"},
                                }
                            },
                        },
                        "503": {
                            "description": "System is unhealthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthCheckResponse"},
                                }
                            },
                        },
                    },
                }
            },
            "/health/live/": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Kubernetes liveness probe",
                    "description": "Simple check to verify the application is running.",
                    "operationId": "livenessCheck",
                    "responses": {
                        "200": {"description": "Application is alive"},
                        "503": {"description": "Application is dead"},
                    },
                }
            },
            "/health/ready/": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Kubernetes readiness probe",
                    "description": "Checks if the application is ready to receive traffic.",
                    "operationId": "readinessCheck",
                    "responses": {
                        "200": {"description": "Application is ready"},
                        "503": {"description": "Application is not ready"},
                    },
                }
            },
            "/metrics/": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Prometheus metrics",
                    "description": "Returns Prometheus-format metrics for monitoring.",
                    "operationId": "prometheusMetrics",
                    "responses": {
                        "200": {
                            "description": "Metrics in Prometheus format",
                            "content": {"text/plain": {"schema": {"type": "string"}}},
                        },
                    },
                }
            },
            # Authentication
            "/": {
                "get": {
                    "tags": ["Authentication"],
                    "summary": "Login page",
                    "description": "Renders the login page. Redirects to Auth0 for authentication.",
                    "operationId": "login",
                    "responses": {
                        "200": {"description": "Login page HTML"},
                        "302": {"description": "Redirect to Auth0"},
                    },
                }
            },
            "/logout": {
                "get": {
                    "tags": ["Authentication"],
                    "summary": "Logout",
                    "description": "Logs out the user and clears session.",
                    "operationId": "logout",
                    "responses": {
                        "302": {"description": "Redirect to login page"},
                    },
                }
            },
            # Dashboard
            "/dashboard": {
                "get": {
                    "tags": ["Dashboard"],
                    "summary": "Trading dashboard",
                    "description": "Main trading dashboard with portfolio overview, positions, and market data.",
                    "operationId": "dashboard",
                    "security": [{"auth0": []}],
                    "responses": {
                        "200": {"description": "Dashboard HTML page"},
                        "401": {"description": "Authentication required"},
                    },
                }
            },
            # Positions
            "/positions": {
                "get": {
                    "tags": ["Positions"],
                    "summary": "View positions",
                    "description": "List all current positions with P/L calculations.",
                    "operationId": "getPositions",
                    "security": [{"auth0": []}],
                    "responses": {
                        "200": {"description": "Positions page"},
                        "401": {"description": "Authentication required"},
                    },
                }
            },
            # Orders
            "/orders": {
                "get": {
                    "tags": ["Orders"],
                    "summary": "View orders",
                    "description": "List all orders (open, filled, cancelled).",
                    "operationId": "getOrders",
                    "security": [{"auth0": []}],
                    "responses": {
                        "200": {"description": "Orders page"},
                        "401": {"description": "Authentication required"},
                    },
                }
            },
            # Strategies
            "/strategies": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Strategy overview",
                    "description": "List all available trading strategies with status.",
                    "operationId": "getStrategies",
                    "security": [{"auth0": []}],
                    "responses": {
                        "200": {"description": "Strategies overview page"},
                        "401": {"description": "Authentication required"},
                    },
                }
            },
            "/strategies/wsb-dip-bot": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "WSB Dip Bot strategy",
                    "description": "Reddit-inspired dip buying strategy for popular stocks.",
                    "operationId": "strategyWsbDipBot",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/wheel": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Wheel strategy",
                    "description": "Options wheel strategy selling puts and covered calls.",
                    "operationId": "strategyWheel",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/momentum-weeklies": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Momentum weeklies",
                    "description": "Weekly options momentum strategy.",
                    "operationId": "strategyMomentumWeeklies",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/earnings-protection": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Earnings protection",
                    "description": "Hedging strategy around earnings announcements.",
                    "operationId": "strategyEarningsProtection",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/debit-spreads": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Debit spreads",
                    "description": "Directional debit spread strategy.",
                    "operationId": "strategyDebitSpreads",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/leaps-tracker": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "LEAPS tracker",
                    "description": "Long-term equity anticipation securities tracking.",
                    "operationId": "strategyLeapsTracker",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/lotto-scanner": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Lotto scanner",
                    "description": "High-risk/high-reward options scanner.",
                    "operationId": "strategyLottoScanner",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/swing-trading": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Swing trading",
                    "description": "Multi-day swing trading strategy.",
                    "operationId": "strategySwingTrading",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/spx-credit-spreads": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "SPX credit spreads",
                    "description": "S&P 500 index credit spread strategy.",
                    "operationId": "strategySpxCreditSpreads",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            "/strategies/index-baseline": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Index baseline",
                    "description": "Index-based baseline comparison strategy.",
                    "operationId": "strategyIndexBaseline",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Strategy configuration page"}},
                }
            },
            # Backtesting
            "/backtesting": {
                "get": {
                    "tags": ["Backtesting"],
                    "summary": "Backtesting interface",
                    "description": "Run historical backtests on trading strategies.",
                    "operationId": "backtesting",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Backtesting page"}},
                }
            },
            # Risk Management
            "/risk": {
                "get": {
                    "tags": ["Risk"],
                    "summary": "Risk management",
                    "description": "Portfolio risk metrics and management interface.",
                    "operationId": "riskManagement",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Risk management page"}},
                }
            },
            "/analytics": {
                "get": {
                    "tags": ["Risk"],
                    "summary": "Analytics dashboard",
                    "description": "Trading analytics and performance metrics.",
                    "operationId": "analytics",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Analytics page"}},
                }
            },
            # Alerts
            "/alerts": {
                "get": {
                    "tags": ["Alerts"],
                    "summary": "Alert configuration",
                    "description": "Configure price, volume, and trading alerts.",
                    "operationId": "alerts",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Alerts page"}},
                }
            },
            # System
            "/system-status": {
                "get": {
                    "tags": ["System"],
                    "summary": "System status",
                    "description": "View system status and component health.",
                    "operationId": "systemStatus",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "System status page"}},
                }
            },
            "/settings": {
                "get": {
                    "tags": ["System"],
                    "summary": "User settings",
                    "description": "Configure user preferences and API keys.",
                    "operationId": "userSettings",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Settings page"}},
                }
            },
            "/setup": {
                "get": {
                    "tags": ["System"],
                    "summary": "Setup wizard",
                    "description": "Initial setup wizard for new users.",
                    "operationId": "setupWizard",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Setup wizard page"}},
                }
            },
            # Advanced Features
            "/crypto": {
                "get": {
                    "tags": ["Dashboard"],
                    "summary": "Crypto trading",
                    "description": "Cryptocurrency trading interface (via Alpaca).",
                    "operationId": "cryptoTrading",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Crypto trading page"}},
                }
            },
            "/extended-hours": {
                "get": {
                    "tags": ["Dashboard"],
                    "summary": "Extended hours",
                    "description": "Pre-market and after-hours trading interface.",
                    "operationId": "extendedHours",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Extended hours page"}},
                }
            },
            "/exotic-spreads": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "Exotic spreads",
                    "description": "Iron condors, butterflies, and other exotic option spreads.",
                    "operationId": "exoticSpreads",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "Exotic spreads page"}},
                }
            },
            "/machine-learning": {
                "get": {
                    "tags": ["Dashboard"],
                    "summary": "Machine learning",
                    "description": "ML-based price prediction and analysis.",
                    "operationId": "machineLearning",
                    "security": [{"auth0": []}],
                    "responses": {"200": {"description": "ML dashboard page"}},
                }
            },
        },
        "components": {
            "schemas": {
                "HealthCheckResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "degraded", "unhealthy"],
                            "description": "Overall health status",
                        },
                        "timestamp": {
                            "type": "number",
                            "description": "Unix timestamp of the check",
                        },
                        "components": {
                            "type": "object",
                            "properties": {
                                "database": {"$ref": "#/components/schemas/ComponentHealth"},
                                "redis": {"$ref": "#/components/schemas/ComponentHealth"},
                                "system": {"$ref": "#/components/schemas/SystemHealth"},
                                "trading": {"$ref": "#/components/schemas/ComponentHealth"},
                            },
                        },
                    },
                },
                "ComponentHealth": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "unhealthy", "not_configured", "unknown"],
                        },
                        "latency_ms": {"type": "number", "description": "Response latency in milliseconds"},
                        "error": {"type": "string", "description": "Error message if unhealthy"},
                    },
                },
                "SystemHealth": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["healthy", "degraded", "unknown"]},
                        "memory_percent": {"type": "number"},
                        "memory_available_mb": {"type": "number"},
                        "cpu_percent": {"type": "number"},
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "nullable": True,
                        },
                    },
                },
                "StockOrderRequest": {
                    "type": "object",
                    "required": ["symbol", "side", "quantity"],
                    "properties": {
                        "symbol": {"type": "string", "minLength": 1, "maxLength": 5, "pattern": "^[A-Z]{1,5}$"},
                        "side": {"type": "string", "enum": ["buy", "sell"]},
                        "quantity": {"type": "integer", "minimum": 1, "maximum": 10000},
                        "order_type": {"type": "string", "enum": ["market", "limit", "stop", "stop_limit"], "default": "market"},
                        "limit_price": {"type": "number", "minimum": 0, "nullable": True},
                        "stop_price": {"type": "number", "minimum": 0, "nullable": True},
                        "time_in_force": {"type": "string", "enum": ["day", "gtc", "ioc", "fok"], "default": "day"},
                        "extended_hours": {"type": "boolean", "default": False},
                    },
                },
                "OptionOrderRequest": {
                    "type": "object",
                    "required": ["symbol", "side", "quantity"],
                    "properties": {
                        "symbol": {"type": "string", "description": "OCC option symbol"},
                        "side": {"type": "string", "enum": ["buy", "sell"]},
                        "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
                        "order_type": {"type": "string", "enum": ["market", "limit", "stop", "stop_limit"], "default": "limit"},
                        "limit_price": {"type": "number", "minimum": 0, "nullable": True},
                        "time_in_force": {"type": "string", "enum": ["day", "gtc", "ioc", "fok"], "default": "day"},
                    },
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"},
                    },
                },
            },
            "securitySchemes": {
                "auth0": {
                    "type": "oauth2",
                    "description": "Auth0 OAuth2.0 authentication",
                    "flows": {
                        "authorizationCode": {
                            "authorizationUrl": "https://YOUR_AUTH0_DOMAIN/authorize",
                            "tokenUrl": "https://YOUR_AUTH0_DOMAIN/oauth/token",
                            "scopes": {
                                "openid": "OpenID Connect scope",
                                "profile": "User profile",
                                "email": "User email",
                            },
                        }
                    },
                }
            },
            "responses": {
                "Unauthorized": {
                    "description": "Authentication required",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"},
                        }
                    },
                },
                "RateLimited": {
                    "description": "Rate limit exceeded",
                    "headers": {
                        "Retry-After": {
                            "description": "Seconds until rate limit resets",
                            "schema": {"type": "integer"},
                        },
                        "X-RateLimit-Limit": {
                            "description": "Rate limit ceiling",
                            "schema": {"type": "integer"},
                        },
                        "X-RateLimit-Remaining": {
                            "description": "Remaining requests",
                            "schema": {"type": "integer"},
                        },
                    },
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"},
                        }
                    },
                },
            },
        },
    }
