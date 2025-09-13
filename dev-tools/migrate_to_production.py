"""Database Migration Script
Migrate from JSON files to PostgreSQL database.
"""

import asyncio
import json
import os
from datetime import datetime

from django.conf import settings
from django.core.management.base import BaseCommand

from .production_config import ConfigManager, ProductionConfig
from .production_logging import ProductionLogger
from .production_models import (
    Configuration,
    DatabaseMigration,
    Position,
    RiskLimit,
    Strategy,
    Trade,
)


class ProductionMigration:
    """Production database migration."""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = ProductionLogger("migration")
        self.migration_stats = {
            "strategies_created": 0,
            "positions_migrated": 0,
            "trades_migrated": 0,
            "errors": 0,
        }

    async def run_full_migration(self):
        """Run complete migration from JSON to database."""
        self.logger.info("Starting production migration")

        try:
            # 1. Create strategies
            await self.create_strategies()

            # 2. Migrate portfolios
            await self.migrate_portfolios()

            # 3. Create risk limits
            await self.create_risk_limits()

            # 4. Create default configurations
            await self.create_default_configurations()

            # 5. Generate migration report
            await self.generate_migration_report()

            self.logger.info("Migration completed successfully", **self.migration_stats)

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise

    async def create_strategies(self):
        """Create strategy records."""
        strategies_data = [
            {
                "name": "WSB Dip Bot",
                "description": "High - risk momentum continuation strategy",
                "risk_level": "high",
                "status": "testing",
                "max_position_risk": 0.10,
                "max_total_risk": 0.30,
            },
            {
                "name": "Momentum Weeklies",
                "description": "High - risk intraday reversal scanner",
                "risk_level": "high",
                "status": "testing",
                "max_position_risk": 0.05,
                "max_total_risk": 0.20,
            },
            {
                "name": "Debit Call Spreads",
                "description": "Medium - risk defined - risk bulls",
                "risk_level": "medium",
                "status": "active",
                "max_position_risk": 0.10,
                "max_total_risk": 0.25,
            },
            {
                "name": "LEAPS Tracker",
                "description": "Medium - risk long - term growth positions",
                "risk_level": "medium",
                "status": "active",
                "max_position_risk": 0.15,
                "max_total_risk": 0.30,
            },
            {
                "name": "Lotto Scanner",
                "description": "Extreme-risk lottery plays",
                "risk_level": "extreme",
                "status": "testing",
                "max_position_risk": 0.01,
                "max_total_risk": 0.05,
            },
            {
                "name": "Wheel Strategy",
                "description": "Lower - risk income generation",
                "risk_level": "low",
                "status": "active",
                "max_position_risk": 0.20,
                "max_total_risk": 0.40,
            },
            {
                "name": "Swing Trading",
                "description": "High - risk breakout trades",
                "risk_level": "high",
                "status": "testing",
                "max_position_risk": 0.08,
                "max_total_risk": 0.25,
            },
            {
                "name": "SPX Credit Spreads",
                "description": "Medium - risk defined - risk spreads",
                "risk_level": "medium",
                "status": "active",
                "max_position_risk": 0.12,
                "max_total_risk": 0.30,
            },
            {
                "name": "Earnings Protection",
                "description": "Medium - risk IV - resistant strategies",
                "risk_level": "medium",
                "status": "active",
                "max_position_risk": 0.10,
                "max_total_risk": 0.25,
            },
            {
                "name": "Index Baseline",
                "description": "Low - risk benchmarking",
                "risk_level": "low",
                "status": "active",
                "max_position_risk": 0.05,
                "max_total_risk": 0.15,
            },
        ]

        for strategy_data in strategies_data:
            try:
                strategy, created = Strategy.objects.get_or_create(
                    name=strategy_data["name"], defaults=strategy_data
                )

                if created:
                    self.migration_stats["strategies_created"] += 1
                    self.logger.info(f"Created strategy: {strategy.name}")
                else:
                    self.logger.info(f"Strategy already exists: {strategy.name}")

            except Exception as e:
                self.migration_stats["errors"] += 1
                self.logger.error(f"Error creating strategy {strategy_data['name']}: {e}")

    async def migrate_portfolios(self):
        """Migrate portfolio data from JSON files."""
        # Migrate LEAPS portfolio
        leaps_file = "leaps_portfolio.json"
        if os.path.exists(leaps_file):
            try:
                DatabaseMigration.migrate_leaps_portfolio(leaps_file)
                self.logger.info("Migrated LEAPS portfolio")
            except Exception as e:
                self.migration_stats["errors"] += 1
                self.logger.error(f"Error migrating LEAPS portfolio: {e}")

        # Migrate Wheel portfolio
        wheel_file = "wheel_portfolio.json"
        if os.path.exists(wheel_file):
            try:
                DatabaseMigration.migrate_wheel_portfolio(wheel_file)
                self.logger.info("Migrated Wheel portfolio")
            except Exception as e:
                self.migration_stats["errors"] += 1
                self.logger.error(f"Error migrating Wheel portfolio: {e}")

        # Count migrated positions
        self.migration_stats["positions_migrated"] = Position.objects.count()

    async def create_risk_limits(self):
        """Create risk limits for each strategy."""
        strategies = Strategy.objects.all()

        for strategy in strategies:
            try:
                _risk_limit, created = RiskLimit.objects.get_or_create(
                    strategy=strategy,
                    defaults={
                        "max_position_risk": strategy.max_position_risk,
                        "max_total_risk": strategy.max_total_risk,
                        "max_drawdown": 0.20,
                        "max_correlation": 0.25,
                    },
                )

                if created:
                    self.logger.info(f"Created risk limits for {strategy.name}")

            except Exception as e:
                self.migration_stats["errors"] += 1
                self.logger.error(f"Error creating risk limits for {strategy.name}: {e}")

    async def create_default_configurations(self):
        """Create default system configurations."""
        default_configs = [
            {
                "key": "system_version",
                "value": "1.0.0",
                "description": "Current system version",
                "data_type": "string",
            },
            {
                "key": "migration_date",
                "value": datetime.now().isoformat(),
                "description": "Date of migration to production",
                "data_type": "string",
            },
            {
                "key": "default_account_size",
                "value": str(self.config.risk.account_size),
                "description": "Default account size",
                "data_type": "float",
            },
            {
                "key": "max_position_risk",
                "value": str(self.config.risk.max_position_risk),
                "description": "Maximum position risk",
                "data_type": "float",
            },
            {
                "key": "max_total_risk",
                "value": str(self.config.risk.max_total_risk),
                "description": "Maximum total portfolio risk",
                "data_type": "float",
            },
            {
                "key": "enable_paper_trading",
                "value": str(self.config.trading.enable_paper_trading).lower(),
                "description": "Enable paper trading mode",
                "data_type": "boolean",
            },
            {
                "key": "enable_live_trading",
                "value": str(self.config.trading.enable_live_trading).lower(),
                "description": "Enable live trading mode",
                "data_type": "boolean",
            },
            {
                "key": "trading_universe",
                "value": ",".join(self.config.trading.universe),
                "description": "Trading universe tickers",
                "data_type": "string",
            },
            {
                "key": "scan_interval",
                "value": str(self.config.trading.scan_interval),
                "description": "Scan interval in seconds",
                "data_type": "integer",
            },
            {
                "key": "max_concurrent_trades",
                "value": str(self.config.trading.max_concurrent_trades),
                "description": "Maximum concurrent trades",
                "data_type": "integer",
            },
        ]

        for config_data in default_configs:
            try:
                config, created = Configuration.objects.get_or_create(
                    key=config_data["key"], defaults=config_data
                )

                if created:
                    self.logger.info(f"Created configuration: {config.key}")

            except Exception as e:
                self.migration_stats["errors"] += 1
                self.logger.error(f"Error creating configuration {config_data['key']}: {e}")

    async def generate_migration_report(self):
        """Generate migration report."""
        report = {
            "migration_date": datetime.now().isoformat(),
            "statistics": self.migration_stats,
            "strategies": list(Strategy.objects.values("name", "risk_level", "status")),
            "positions_count": Position.objects.count(),
            "trades_count": Trade.objects.count(),
            "configurations_count": Configuration.objects.count(),
            "risk_limits_count": RiskLimit.objects.count(),
        }

        # Save report to file
        report_file = f"migration_report_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Migration report saved to {report_file}")

        # Log summary
        self.logger.info("Migration Summary", **self.migration_stats)


class Command(BaseCommand):
    """Django management command for migration."""

    help = "Migrate from JSON files to production database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--config - file",
            type=str,
            default="config / production.json",
            help="Configuration file path",
        )
        parser.add_argument(
            "--dry - run", action="store_true", help="Run migration in dry - run mode"
        )

    def handle(self, *args, **options):
        """Handle migration command."""
        # Load configuration
        config_manager = ConfigManager(options["config_file"])
        config = config_manager.load_config()

        # Validate configuration
        errors = config.validate()
        if errors:
            self.stdout.write(self.style.ERROR(f"Configuration errors: {', '.join(errors)}"))
            return

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("Dry run mode-no changes will be made"))
            return

        # Run migration
        migration = ProductionMigration(config)

        try:
            asyncio.run(migration.run_full_migration())
            self.stdout.write(self.style.SUCCESS("Migration completed successfully"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Migration failed: {e}"))


# Standalone migration script
async def main():
    """Standalone migration script."""
    import sys

    sys.path.append(".")

    # Setup Django
    import django

    if not settings.configured:
        settings.configure(
            DATABASES={
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": "db.sqlite3",
                }
            },
            INSTALLED_APPS=[
                "django.contrib.auth",
                "django.contrib.contenttypes",
                "backend.tradingbot",
            ],
        )
        django.setup()

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()

    # Run migration
    migration = ProductionMigration(config)
    await migration.run_full_migration()


if __name__ == "__main__":
    asyncio.run(main())
