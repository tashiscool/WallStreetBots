"""
Production Configuration Management
Environment-based configuration with validation
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading .env file
    pass


@dataclass
class DataProviderConfig:
    """Data provider configuration"""
    iex_api_key: str = ""
    polygon_api_key: str = ""
    fmp_api_key: str = ""
    news_api_key: str = ""
    alpha_vantage_api_key: str = ""
    
    def validate(self) -> List[str]:
        """Validate data provider configuration"""
        errors = []
        
        if not self.iex_api_key:
            errors.append("IEX API key is required")
        
        if not self.polygon_api_key:
            errors.append("Polygon API key is required")
        
        return errors


@dataclass
class BrokerConfig:
    """Broker configuration"""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    ibkr_host: str = ""
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    
    def validate(self) -> List[str]:
        """Validate broker configuration"""
        errors = []
        
        if not self.alpaca_api_key:
            errors.append("Alpaca API key is required")
        
        if not self.alpaca_secret_key:
            errors.append("Alpaca secret key is required")
        
        return errors


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_risk: float = 0.10  # 10% per position
    max_total_risk: float = 0.30      # 30% total portfolio risk
    max_drawdown: float = 0.20       # 20% max drawdown
    max_correlation: float = 0.25    # 25% max correlation
    account_size: float = 100000.0   # Default account size
    default_commission: float = 1.0  # Default commission per trade
    default_slippage: float = 0.002   # Default slippage (0.2%)
    
    def validate(self) -> List[str]:
        """Validate risk configuration"""
        errors = []
        
        if not 0 < self.max_position_risk <= 1:
            errors.append("max_position_risk must be between 0 and 1")
        
        if not 0 < self.max_total_risk <= 1:
            errors.append("max_total_risk must be between 0 and 1")
        
        if not 0 < self.max_drawdown <= 1:
            errors.append("max_drawdown must be between 0 and 1")
        
        if self.account_size <= 0:
            errors.append("account_size must be positive")
        
        return errors


@dataclass
class TradingConfig:
    """Trading system configuration"""
    universe: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'AMD', 'TSLA'
    ])
    scan_interval: int = 300  # 5 minutes
    max_concurrent_trades: int = 10
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    
    def validate(self) -> List[str]:
        """Validate trading configuration"""
        errors = []
        
        if not self.universe:
            errors.append("Trading universe cannot be empty")
        
        if self.scan_interval < 60:
            errors.append("scan_interval must be at least 60 seconds")
        
        if self.max_concurrent_trades <= 0:
            errors.append("max_concurrent_trades must be positive")
        
        return errors


@dataclass
class AlertConfig:
    """Alert system configuration"""
    enable_slack: bool = False
    slack_webhook_url: str = ""
    enable_email: bool = False
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate alert configuration"""
        errors = []
        
        if self.enable_slack and not self.slack_webhook_url:
            errors.append("Slack webhook URL is required when Slack alerts are enabled")
        
        if self.enable_email:
            if not self.email_smtp_server:
                errors.append("Email SMTP server is required when email alerts are enabled")
            if not self.email_username:
                errors.append("Email username is required when email alerts are enabled")
            if not self.email_password:
                errors.append("Email password is required when email alerts are enabled")
            if not self.email_recipients:
                errors.append("Email recipients list cannot be empty when email alerts are enabled")
        
        return errors


@dataclass
class DatabaseConfig:
    """Database configuration"""
    engine: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    name: str = "wallstreetbots"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    
    def validate(self) -> List[str]:
        """Validate database configuration"""
        errors = []
        
        if not self.name:
            errors.append("Database name is required")
        
        if not self.username:
            errors.append("Database username is required")
        
        return errors


@dataclass
class ProductionConfig:
    """Complete production configuration"""
    data_providers: DataProviderConfig = field(default_factory=DataProviderConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    def validate(self) -> List[str]:
        """Validate entire configuration"""
        errors = []
        
        errors.extend(self.data_providers.validate())
        errors.extend(self.broker.validate())
        errors.extend(self.risk.validate())
        errors.extend(self.trading.validate())
        errors.extend(self.alerts.validate())
        errors.extend(self.database.validate())
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'data_providers': {
                'iex_api_key': self.data_providers.iex_api_key,
                'polygon_api_key': self.data_providers.polygon_api_key,
                'fmp_api_key': self.data_providers.fmp_api_key,
                'news_api_key': self.data_providers.news_api_key,
                'alpha_vantage_api_key': self.data_providers.alpha_vantage_api_key,
            },
            'broker': {
                'alpaca_api_key': self.broker.alpaca_api_key,
                'alpaca_secret_key': self.broker.alpaca_secret_key,
                'alpaca_base_url': self.broker.alpaca_base_url,
                'ibkr_host': self.broker.ibkr_host,
                'ibkr_port': self.broker.ibkr_port,
                'ibkr_client_id': self.broker.ibkr_client_id,
            },
            'risk': {
                'max_position_risk': self.risk.max_position_risk,
                'max_total_risk': self.risk.max_total_risk,
                'max_drawdown': self.risk.max_drawdown,
                'max_correlation': self.risk.max_correlation,
                'account_size': self.risk.account_size,
                'default_commission': self.risk.default_commission,
                'default_slippage': self.risk.default_slippage,
            },
            'trading': {
                'universe': self.trading.universe,
                'scan_interval': self.trading.scan_interval,
                'max_concurrent_trades': self.trading.max_concurrent_trades,
                'enable_paper_trading': self.trading.enable_paper_trading,
                'enable_live_trading': self.trading.enable_live_trading,
            },
            'alerts': {
                'enable_slack': self.alerts.enable_slack,
                'slack_webhook_url': self.alerts.slack_webhook_url,
                'enable_email': self.alerts.enable_email,
                'email_smtp_server': self.alerts.email_smtp_server,
                'email_smtp_port': self.alerts.email_smtp_port,
                'email_username': self.alerts.email_username,
                'email_password': self.alerts.email_password,
                'email_recipients': self.alerts.email_recipients,
            },
            'database': {
                'engine': self.database.engine,
                'host': self.database.host,
                'port': self.database.port,
                'name': self.database.name,
                'username': self.database.username,
                'password': self.database.password,
                'ssl_mode': self.database.ssl_mode,
            }
        }


class ConfigManager:
    """Configuration manager with environment variable support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/production.json"
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> ProductionConfig:
        """Load configuration from file and environment variables"""
        # Start with defaults
        config = ProductionConfig()
        
        # Load from file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                config = self._merge_config(config, file_config)
            except Exception as e:
                self.logger.warning(f"Could not load config file {self.config_file}: {e}")
        
        # Override with environment variables - THIS IS WHERE REAL CONFIG HAPPENS
        config = self._load_from_env(config)
        
        return config
    
    def _merge_config(self, base_config: ProductionConfig, file_config: Dict[str, Any]) -> ProductionConfig:
        """Merge file configuration into base configuration"""
        # Data providers
        if 'data_providers' in file_config:
            dp_config = file_config['data_providers']
            base_config.data_providers.iex_api_key = dp_config.get('iex_api_key', base_config.data_providers.iex_api_key)
            base_config.data_providers.polygon_api_key = dp_config.get('polygon_api_key', base_config.data_providers.polygon_api_key)
            base_config.data_providers.fmp_api_key = dp_config.get('fmp_api_key', base_config.data_providers.fmp_api_key)
            base_config.data_providers.news_api_key = dp_config.get('news_api_key', base_config.data_providers.news_api_key)
            base_config.data_providers.alpha_vantage_api_key = dp_config.get('alpha_vantage_api_key', base_config.data_providers.alpha_vantage_api_key)
        
        # Broker
        if 'broker' in file_config:
            broker_config = file_config['broker']
            base_config.broker.alpaca_api_key = broker_config.get('alpaca_api_key', base_config.broker.alpaca_api_key)
            base_config.broker.alpaca_secret_key = broker_config.get('alpaca_secret_key', base_config.broker.alpaca_secret_key)
            base_config.broker.alpaca_base_url = broker_config.get('alpaca_base_url', base_config.broker.alpaca_base_url)
            base_config.broker.ibkr_host = broker_config.get('ibkr_host', base_config.broker.ibkr_host)
            base_config.broker.ibkr_port = broker_config.get('ibkr_port', base_config.broker.ibkr_port)
            base_config.broker.ibkr_client_id = broker_config.get('ibkr_client_id', base_config.broker.ibkr_client_id)
        
        # Risk
        if 'risk' in file_config:
            risk_config = file_config['risk']
            base_config.risk.max_position_risk = risk_config.get('max_position_risk', base_config.risk.max_position_risk)
            base_config.risk.max_total_risk = risk_config.get('max_total_risk', base_config.risk.max_total_risk)
            base_config.risk.max_drawdown = risk_config.get('max_drawdown', base_config.risk.max_drawdown)
            base_config.risk.max_correlation = risk_config.get('max_correlation', base_config.risk.max_correlation)
            base_config.risk.account_size = risk_config.get('account_size', base_config.risk.account_size)
            base_config.risk.default_commission = risk_config.get('default_commission', base_config.risk.default_commission)
            base_config.risk.default_slippage = risk_config.get('default_slippage', base_config.risk.default_slippage)
        
        # Trading
        if 'trading' in file_config:
            trading_config = file_config['trading']
            base_config.trading.universe = trading_config.get('universe', base_config.trading.universe)
            base_config.trading.scan_interval = trading_config.get('scan_interval', base_config.trading.scan_interval)
            base_config.trading.max_concurrent_trades = trading_config.get('max_concurrent_trades', base_config.trading.max_concurrent_trades)
            base_config.trading.enable_paper_trading = trading_config.get('enable_paper_trading', base_config.trading.enable_paper_trading)
            base_config.trading.enable_live_trading = trading_config.get('enable_live_trading', base_config.trading.enable_live_trading)
        
        # Alerts
        if 'alerts' in file_config:
            alert_config = file_config['alerts']
            base_config.alerts.enable_slack = alert_config.get('enable_slack', base_config.alerts.enable_slack)
            base_config.alerts.slack_webhook_url = alert_config.get('slack_webhook_url', base_config.alerts.slack_webhook_url)
            base_config.alerts.enable_email = alert_config.get('enable_email', base_config.alerts.enable_email)
            base_config.alerts.email_smtp_server = alert_config.get('email_smtp_server', base_config.alerts.email_smtp_server)
            base_config.alerts.email_smtp_port = alert_config.get('email_smtp_port', base_config.alerts.email_smtp_port)
            base_config.alerts.email_username = alert_config.get('email_username', base_config.alerts.email_username)
            base_config.alerts.email_password = alert_config.get('email_password', base_config.alerts.email_password)
            base_config.alerts.email_recipients = alert_config.get('email_recipients', base_config.alerts.email_recipients)
        
        # Database
        if 'database' in file_config:
            db_config = file_config['database']
            base_config.database.engine = db_config.get('engine', base_config.database.engine)
            base_config.database.host = db_config.get('host', base_config.database.host)
            base_config.database.port = db_config.get('port', base_config.database.port)
            base_config.database.name = db_config.get('name', base_config.database.name)
            base_config.database.username = db_config.get('username', base_config.database.username)
            base_config.database.password = db_config.get('password', base_config.database.password)
            base_config.database.ssl_mode = db_config.get('ssl_mode', base_config.database.ssl_mode)
        
        return base_config
    
    def _load_from_env(self, config: ProductionConfig) -> ProductionConfig:
        """Load configuration from environment variables"""
        # Data providers
        config.data_providers.iex_api_key = os.getenv('IEX_API_KEY', config.data_providers.iex_api_key)
        config.data_providers.polygon_api_key = os.getenv('POLYGON_API_KEY', config.data_providers.polygon_api_key)
        config.data_providers.fmp_api_key = os.getenv('FMP_API_KEY', config.data_providers.fmp_api_key)
        config.data_providers.news_api_key = os.getenv('NEWS_API_KEY', config.data_providers.news_api_key)
        config.data_providers.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY', config.data_providers.alpha_vantage_api_key)
        
        # Broker
        config.broker.alpaca_api_key = os.getenv('ALPACA_API_KEY', config.broker.alpaca_api_key)
        config.broker.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY', config.broker.alpaca_secret_key)
        config.broker.alpaca_base_url = os.getenv('ALPACA_BASE_URL', config.broker.alpaca_base_url)
        config.broker.ibkr_host = os.getenv('IBKR_HOST', config.broker.ibkr_host)
        config.broker.ibkr_port = int(os.getenv('IBKR_PORT', config.broker.ibkr_port))
        config.broker.ibkr_client_id = int(os.getenv('IBKR_CLIENT_ID', config.broker.ibkr_client_id))
        
        # Risk
        config.risk.max_position_risk = float(os.getenv('MAX_POSITION_RISK', config.risk.max_position_risk))
        config.risk.max_total_risk = float(os.getenv('MAX_TOTAL_RISK', config.risk.max_total_risk))
        config.risk.max_drawdown = float(os.getenv('MAX_DRAWDOWN', config.risk.max_drawdown))
        config.risk.max_correlation = float(os.getenv('MAX_CORRELATION', config.risk.max_correlation))
        config.risk.account_size = float(os.getenv('ACCOUNT_SIZE', config.risk.account_size))
        config.risk.default_commission = float(os.getenv('DEFAULT_COMMISSION', config.risk.default_commission))
        config.risk.default_slippage = float(os.getenv('DEFAULT_SLIPPAGE', config.risk.default_slippage))
        
        # Trading
        universe_str = os.getenv('TRADING_UNIVERSE', '')
        if universe_str:
            config.trading.universe = [ticker.strip() for ticker in universe_str.split(',')]
        config.trading.scan_interval = int(os.getenv('SCAN_INTERVAL', config.trading.scan_interval))
        config.trading.max_concurrent_trades = int(os.getenv('MAX_CONCURRENT_TRADES', config.trading.max_concurrent_trades))
        config.trading.enable_paper_trading = os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true'
        config.trading.enable_live_trading = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
        
        # Alerts
        config.alerts.enable_slack = os.getenv('ENABLE_SLACK', 'false').lower() == 'true'
        config.alerts.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', config.alerts.slack_webhook_url)
        config.alerts.enable_email = os.getenv('ENABLE_EMAIL', 'false').lower() == 'true'
        config.alerts.email_smtp_server = os.getenv('EMAIL_SMTP_SERVER', config.alerts.email_smtp_server)
        config.alerts.email_smtp_port = int(os.getenv('EMAIL_SMTP_PORT', config.alerts.email_smtp_port))
        config.alerts.email_username = os.getenv('EMAIL_USERNAME', config.alerts.email_username)
        config.alerts.email_password = os.getenv('EMAIL_PASSWORD', config.alerts.email_password)
        email_recipients_str = os.getenv('EMAIL_RECIPIENTS', '')
        if email_recipients_str:
            config.alerts.email_recipients = [email.strip() for email in email_recipients_str.split(',')]
        
        # Database
        config.database.engine = os.getenv('DB_ENGINE', config.database.engine)
        config.database.host = os.getenv('DB_HOST', config.database.host)
        config.database.port = int(os.getenv('DB_PORT', config.database.port))
        config.database.name = os.getenv('DB_NAME', config.database.name)
        config.database.username = os.getenv('DB_USERNAME', config.database.username)
        config.database.password = os.getenv('DB_PASSWORD', config.database.password)
        config.database.ssl_mode = os.getenv('DB_SSL_MODE', config.database.ssl_mode)
        
        return config
    
    def save_config(self, config: ProductionConfig) -> bool:
        """Save configuration to file"""
        try:
            # Ensure config directory exists
            config_dir = Path(self.config_file).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Could not save configuration: {e}")
            return False
    
    def create_env_template(self, output_file: str = ".env.template") -> bool:
        """Create environment variable template file"""
        try:
            template = """# WallStreetBots Production Configuration
# Copy this file to .env and fill in your actual values

# Data Provider API Keys
IEX_API_KEY=your_iex_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here
FMP_API_KEY=your_fmp_api_key_here
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Broker Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
IBKR_HOST=localhost
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Risk Management
MAX_POSITION_RISK=0.10
MAX_TOTAL_RISK=0.30
MAX_DRAWDOWN=0.20
MAX_CORRELATION=0.25
ACCOUNT_SIZE=100000.0
DEFAULT_COMMISSION=1.0
DEFAULT_SLIPPAGE=0.002

# Trading Configuration
TRADING_UNIVERSE=AAPL,MSFT,GOOGL,GOOG,META,NVDA,AVGO,AMD,TSLA
SCAN_INTERVAL=300
MAX_CONCURRENT_TRADES=10
ENABLE_PAPER_TRADING=true
ENABLE_LIVE_TRADING=false

# Alert Configuration
ENABLE_SLACK=false
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
ENABLE_EMAIL=false
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email_here
EMAIL_PASSWORD=your_email_password_here
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com

# Database Configuration
DB_ENGINE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=wallstreetbots
DB_USERNAME=postgres
DB_PASSWORD=your_db_password_here
DB_SSL_MODE=prefer
"""
            
            with open(output_file, 'w') as f:
                f.write(template)
            
            self.logger.info(f"Environment template created: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Could not create environment template: {e}")
            return False


# Factory function for easy initialization
def create_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Create configuration manager"""
    return ConfigManager(config_file)
