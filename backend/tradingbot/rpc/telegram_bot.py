"""
Telegram Bot Interface

Provides remote control of trading system via Telegram.
Inspired by freqtrade's telegram integration.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import telegram library
try:
    from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        CallbackContext,
        CallbackQueryHandler,
        MessageHandler,
        filters,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram features disabled.")

from .message_types import RPCMessage, RPCMessageType


@dataclass
class TelegramConfig:
    """Configuration for Telegram bot."""
    token: str
    chat_id: str  # Authorized chat ID
    admin_ids: List[str] = field(default_factory=list)  # Admin user IDs
    notification_settings: Dict[str, bool] = field(default_factory=lambda: {
        'trades': True,
        'warnings': True,
        'errors': True,
        'daily_report': True,
    })
    rate_limit_messages: int = 30  # Max messages per minute
    silent_hours: Optional[tuple] = None  # (start_hour, end_hour) for silent mode


class TradingTelegramBot:
    """
    Telegram bot for trading system control.

    Commands:
        /start - Initialize bot
        /status - Show current positions and P&L
        /balance - Show account balance
        /daily - Show daily performance
        /trades - List recent trades
        /performance - Show strategy performance
        /forcebuy <symbol> - Force buy entry
        /forcesell <symbol> - Force exit position
        /stopbuy - Disable new entries
        /startbuy - Re-enable entries
        /reload - Reload configuration
        /help - Show help
    """

    def __init__(
        self,
        config: TelegramConfig,
        trading_system=None,
    ):
        """
        Initialize Telegram bot.

        Args:
            config: Telegram configuration
            trading_system: Reference to trading system for control
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. Install with: pip install python-telegram-bot"
            )

        self.config = config
        self.trading_system = trading_system
        self._app: Optional[Application] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._rate_limiter: List[datetime] = []
        self._is_running = False

    async def start(self) -> None:
        """Start the Telegram bot."""
        self._app = (
            Application.builder()
            .token(self.config.token)
            .build()
        )

        # Register command handlers
        self._register_handlers()

        # Start the bot
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        self._is_running = True
        logger.info("Telegram bot started")

        # Send startup message
        await self._send_message("Trading bot started and ready!")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            await self._send_message("Trading bot shutting down...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        self._is_running = False
        logger.info("Telegram bot stopped")

    def _register_handlers(self) -> None:
        """Register command handlers."""
        handlers = [
            ('start', self._cmd_start),
            ('help', self._cmd_help),
            ('status', self._cmd_status),
            ('balance', self._cmd_balance),
            ('daily', self._cmd_daily),
            ('weekly', self._cmd_weekly),
            ('trades', self._cmd_trades),
            ('performance', self._cmd_performance),
            ('profit', self._cmd_profit),
            ('forcebuy', self._cmd_forcebuy),
            ('forcesell', self._cmd_forcesell),
            ('stopbuy', self._cmd_stopbuy),
            ('startbuy', self._cmd_startbuy),
            ('reload', self._cmd_reload),
            ('version', self._cmd_version),
        ]

        for command, handler in handlers:
            self._app.add_handler(CommandHandler(command, handler))

        # Callback query handler for inline buttons
        self._app.add_handler(CallbackQueryHandler(self._callback_handler))

    async def _check_authorized(self, update: Update) -> bool:
        """Check if user is authorized."""
        chat_id = str(update.effective_chat.id)
        user_id = str(update.effective_user.id)

        if chat_id != self.config.chat_id:
            await update.message.reply_text("Unauthorized chat.")
            return False

        return True

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Clean old entries
        self._rate_limiter = [t for t in self._rate_limiter if t > cutoff]

        if len(self._rate_limiter) >= self.config.rate_limit_messages:
            return False

        self._rate_limiter.append(now)
        return True

    async def _send_message(
        self,
        message: str,
        parse_mode: str = 'Markdown',
        keyboard: Optional[List[List[InlineKeyboardButton]]] = None,
    ) -> None:
        """Send message to configured chat."""
        if not await self._check_rate_limit():
            logger.warning("Rate limit exceeded, message not sent")
            return

        # Check silent hours
        if self.config.silent_hours:
            hour = datetime.now().hour
            start, end = self.config.silent_hours
            if start <= hour < end:
                logger.debug("Silent hours, message not sent")
                return

        reply_markup = None
        if keyboard:
            reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            bot = Bot(token=self.config.token)
            await bot.send_message(
                chat_id=self.config.chat_id,
                text=message,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def send_notification(self, rpc_message: RPCMessage) -> None:
        """Send RPC notification via Telegram."""
        # Check notification settings
        msg_type = rpc_message.msg_type.value
        category = self._get_notification_category(rpc_message.msg_type)

        if not self.config.notification_settings.get(category, True):
            return

        # Format message
        emoji = self._get_emoji(rpc_message.msg_type)
        message = f"{emoji} *{msg_type.upper()}*\n\n{rpc_message.message}"

        # Add data details
        if rpc_message.data:
            for key, value in rpc_message.data.items():
                if value is not None:
                    message += f"\n{key}: `{value}`"

        await self._send_message(message)

    def _get_notification_category(self, msg_type: RPCMessageType) -> str:
        """Map message type to notification category."""
        if msg_type in (RPCMessageType.ENTRY, RPCMessageType.EXIT,
                        RPCMessageType.ENTRY_FILL, RPCMessageType.EXIT_FILL):
            return 'trades'
        elif msg_type in (RPCMessageType.WARNING, RPCMessageType.RISK_ALERT):
            return 'warnings'
        elif msg_type == RPCMessageType.ERROR:
            return 'errors'
        elif msg_type == RPCMessageType.DAILY_REPORT:
            return 'daily_report'
        return 'trades'

    def _get_emoji(self, msg_type: RPCMessageType) -> str:
        """Get emoji for message type."""
        emoji_map = {
            RPCMessageType.ENTRY: "",
            RPCMessageType.EXIT: "",
            RPCMessageType.FORCE_ENTRY: "",
            RPCMessageType.FORCE_EXIT: "",
            RPCMessageType.WARNING: "",
            RPCMessageType.ERROR: "",
            RPCMessageType.RISK_ALERT: "",
            RPCMessageType.DAILY_REPORT: "",
            RPCMessageType.STARTUP: "",
            RPCMessageType.SHUTDOWN: "",
        }
        return emoji_map.get(msg_type, "")

    # Command handlers

    async def _cmd_start(self, update: Update, context: CallbackContext) -> None:
        """Handle /start command."""
        if not await self._check_authorized(update):
            return

        await update.message.reply_text(
            " *WallStreetBots Trading Bot*\n\n"
            "Use /help to see available commands.",
            parse_mode='Markdown',
        )

    async def _cmd_help(self, update: Update, context: CallbackContext) -> None:
        """Handle /help command."""
        if not await self._check_authorized(update):
            return

        help_text = """
 *Available Commands*

*Status*
/status - Current positions & P&L
/balance - Account balance
/daily - Today's performance
/weekly - Week's performance
/profit - Total profit summary

*Trading*
/trades - Recent trade history
/performance - Strategy stats
/forcebuy <symbol> - Force entry
/forcesell <symbol> - Force exit

*Control*
/stopbuy - Disable new entries
/startbuy - Enable entries
/reload - Reload config

*Info*
/version - Bot version
/help - This message
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def _cmd_status(self, update: Update, context: CallbackContext) -> None:
        """Handle /status command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            positions = self.trading_system.get_positions()
            total_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)

            message = " *Current Positions*\n\n"

            if not positions:
                message += "_No open positions_"
            else:
                for pos in positions:
                    emoji = ""
                    message += (
                        f"{emoji} *{pos['symbol']}*\n"
                        f"  Qty: {pos['quantity']} @ ${pos['avg_price']:.2f}\n"
                        f"  P&L: ${pos['unrealized_pnl']:.2f}\n\n"
                    )

                message += f"\n*Total P&L:* ${total_pnl:.2f}"

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error getting status: {e}")

    async def _cmd_balance(self, update: Update, context: CallbackContext) -> None:
        """Handle /balance command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            account = self.trading_system.get_account()

            message = (
                " *Account Balance*\n\n"
                f"*Cash:* ${account.get('cash', 0):.2f}\n"
                f"*Equity:* ${account.get('equity', 0):.2f}\n"
                f"*Buying Power:* ${account.get('buying_power', 0):.2f}\n"
                f"*Day P&L:* ${account.get('daily_pnl', 0):.2f}"
            )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error getting balance: {e}")

    async def _cmd_daily(self, update: Update, context: CallbackContext) -> None:
        """Handle /daily command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            stats = self.trading_system.get_daily_stats()

            message = (
                " *Daily Performance*\n\n"
                f"*Profit:* ${stats.get('profit', 0):.2f}\n"
                f"*Trades:* {stats.get('trades', 0)}\n"
                f"*Win Rate:* {stats.get('win_rate', 0):.1f}%\n"
                f"*Best Trade:* ${stats.get('best_trade', 0):.2f}\n"
                f"*Worst Trade:* ${stats.get('worst_trade', 0):.2f}"
            )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error getting daily stats: {e}")

    async def _cmd_weekly(self, update: Update, context: CallbackContext) -> None:
        """Handle /weekly command."""
        if not await self._check_authorized(update):
            return

        await update.message.reply_text("Weekly stats coming soon...")

    async def _cmd_trades(self, update: Update, context: CallbackContext) -> None:
        """Handle /trades command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            trades = self.trading_system.get_recent_trades(limit=10)

            message = " *Recent Trades*\n\n"

            if not trades:
                message += "_No recent trades_"
            else:
                for trade in trades:
                    emoji = ""
                    message += (
                        f"{emoji} {trade['symbol']} "
                        f"({trade['side']}) "
                        f"P&L: ${trade.get('pnl', 0):.2f}\n"
                    )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error getting trades: {e}")

    async def _cmd_performance(self, update: Update, context: CallbackContext) -> None:
        """Handle /performance command."""
        if not await self._check_authorized(update):
            return

        await update.message.reply_text("Performance stats coming soon...")

    async def _cmd_profit(self, update: Update, context: CallbackContext) -> None:
        """Handle /profit command."""
        if not await self._check_authorized(update):
            return

        await update.message.reply_text("Profit summary coming soon...")

    async def _cmd_forcebuy(self, update: Update, context: CallbackContext) -> None:
        """Handle /forcebuy command."""
        if not await self._check_authorized(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /forcebuy <SYMBOL>")
            return

        symbol = context.args[0].upper()

        # Confirmation buttons
        keyboard = [
            [
                InlineKeyboardButton("Confirm", callback_data=f"forcebuy_{symbol}"),
                InlineKeyboardButton("Cancel", callback_data="cancel"),
            ]
        ]

        await update.message.reply_text(
            f"Force buy *{symbol}*?",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _cmd_forcesell(self, update: Update, context: CallbackContext) -> None:
        """Handle /forcesell command."""
        if not await self._check_authorized(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /forcesell <SYMBOL>")
            return

        symbol = context.args[0].upper()

        keyboard = [
            [
                InlineKeyboardButton("Confirm", callback_data=f"forcesell_{symbol}"),
                InlineKeyboardButton("Cancel", callback_data="cancel"),
            ]
        ]

        await update.message.reply_text(
            f"Force sell *{symbol}*?",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _cmd_stopbuy(self, update: Update, context: CallbackContext) -> None:
        """Handle /stopbuy command."""
        if not await self._check_authorized(update):
            return

        if self.trading_system:
            self.trading_system.disable_buying()
            await update.message.reply_text(" Buying disabled. Use /startbuy to re-enable.")
        else:
            await update.message.reply_text("Trading system not connected.")

    async def _cmd_startbuy(self, update: Update, context: CallbackContext) -> None:
        """Handle /startbuy command."""
        if not await self._check_authorized(update):
            return

        if self.trading_system:
            self.trading_system.enable_buying()
            await update.message.reply_text(" Buying enabled.")
        else:
            await update.message.reply_text("Trading system not connected.")

    async def _cmd_reload(self, update: Update, context: CallbackContext) -> None:
        """Handle /reload command."""
        if not await self._check_authorized(update):
            return

        if self.trading_system:
            self.trading_system.reload_config()
            await update.message.reply_text(" Configuration reloaded.")
        else:
            await update.message.reply_text("Trading system not connected.")

    async def _cmd_version(self, update: Update, context: CallbackContext) -> None:
        """Handle /version command."""
        if not await self._check_authorized(update):
            return

        await update.message.reply_text(
            " *WallStreetBots*\n"
            "Version: 1.0.0\n"
            "Framework: Algorithm Framework v1",
            parse_mode='Markdown',
        )

    async def _callback_handler(self, update: Update, context: CallbackContext) -> None:
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()

        data = query.data

        if data == "cancel":
            await query.edit_message_text("Cancelled.")
            return

        if data.startswith("forcebuy_"):
            symbol = data.replace("forcebuy_", "")
            if self.trading_system:
                try:
                    result = self.trading_system.force_buy(symbol)
                    await query.edit_message_text(f" Bought {symbol}: {result}")
                except Exception as e:
                    await query.edit_message_text(f" Error: {e}")
            else:
                await query.edit_message_text("Trading system not connected.")

        elif data.startswith("forcesell_"):
            symbol = data.replace("forcesell_", "")
            if self.trading_system:
                try:
                    result = self.trading_system.force_sell(symbol)
                    await query.edit_message_text(f" Sold {symbol}: {result}")
                except Exception as e:
                    await query.edit_message_text(f" Error: {e}")
            else:
                await query.edit_message_text("Trading system not connected.")

        elif data.startswith("panic_confirm"):
            if self.trading_system:
                try:
                    result = await self.trading_system.emergency_sell_all()
                    await query.edit_message_text(f"Emergency sell complete: {result}")
                except Exception as e:
                    await query.edit_message_text(f"Error: {e}")
            else:
                await query.edit_message_text("Trading system not connected.")


# Extended RPC command handlers for full freqtrade-style functionality
class ExtendedTelegramRPC(TradingTelegramBot):
    """
    Extended Telegram RPC with 30+ commands.

    Additional commands beyond base implementation:
    - /whitelist, /blacklist - Manage trading pairs
    - /stats - Detailed trading statistics
    - /monthly - Monthly performance
    - /count - Trade count info
    - /locks - Show locked pairs
    - /edge - Edge positioning info
    - /show_config - Show current config
    - /logs - View recent logs
    - /greeks - Portfolio Greeks (options)
    - /spreads - Open option spreads
    - /iv - IV rank/percentile
    - /expiry - Positions by expiry
    - /roll - Roll options position
    - /health - System health check
    - /panic - Emergency sell all
    """

    def __init__(self, config: TelegramConfig, trading_system=None):
        """Initialize extended Telegram RPC."""
        super().__init__(config, trading_system)

    def _register_handlers(self) -> None:
        """Register all command handlers including extended commands."""
        # Base handlers
        super()._register_handlers()

        # Extended handlers
        extended_handlers = [
            ('whitelist', self._cmd_whitelist),
            ('blacklist', self._cmd_blacklist),
            ('stats', self._cmd_stats),
            ('monthly', self._cmd_monthly),
            ('count', self._cmd_count),
            ('locks', self._cmd_locks),
            ('unlock', self._cmd_unlock),
            ('edge', self._cmd_edge),
            ('show_config', self._cmd_show_config),
            ('logs', self._cmd_logs),
            ('health', self._cmd_health),
            ('panic', self._cmd_panic),
            # Options-specific commands
            ('greeks', self._cmd_greeks),
            ('spreads', self._cmd_spreads),
            ('iv', self._cmd_iv),
            ('expiry', self._cmd_expiry),
            ('roll', self._cmd_roll),
        ]

        for command, handler in extended_handlers:
            self._app.add_handler(CommandHandler(command, handler))

    async def _cmd_help(self, update: Update, context: CallbackContext) -> None:
        """Extended help command with all commands."""
        if not await self._check_authorized(update):
            return

        help_text = """
*Available Commands*

*Status & Info*
/status - Current positions & P&L
/balance - Account balance
/daily - Today's performance
/weekly - Week's performance
/monthly - Monthly performance
/profit - Total profit summary
/stats - Detailed statistics
/count - Trade count info
/health - System health check

*Trading Control*
/trades - Recent trade history
/performance - Strategy stats
/forcebuy <symbol> - Force entry
/forcesell <symbol> - Force exit
/stopbuy - Disable new entries
/startbuy - Enable entries
/panic - Emergency sell all

*Configuration*
/whitelist - Show trading whitelist
/blacklist - Manage blacklist
/locks - Show locked pairs
/unlock <pair> - Unlock a pair
/edge - Edge positioning info
/show_config - Show current config
/reload - Reload config
/logs - View recent logs

*Options Trading*
/greeks - Portfolio Greeks
/spreads - Open option spreads
/iv <symbol> - IV rank/percentile
/expiry - Positions by expiry
/roll <symbol> - Roll recommendation

*General*
/version - Bot version
/help - This message
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def _cmd_whitelist(self, update: Update, context: CallbackContext) -> None:
        """Handle /whitelist command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            whitelist = self.trading_system.get_whitelist() if hasattr(
                self.trading_system, 'get_whitelist'
            ) else []

            if not whitelist:
                await update.message.reply_text("Whitelist is empty.")
                return

            message = f"*Trading Whitelist ({len(whitelist)} pairs)*\n\n"
            message += ", ".join([f"`{s}`" for s in whitelist])

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_blacklist(self, update: Update, context: CallbackContext) -> None:
        """Handle /blacklist command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            # Parse arguments: /blacklist [add|remove] [symbol]
            if context.args and len(context.args) >= 2:
                action = context.args[0].lower()
                symbol = context.args[1].upper()

                if action == "add":
                    self.trading_system.add_to_blacklist([symbol])
                    await update.message.reply_text(f"Added `{symbol}` to blacklist.", parse_mode='Markdown')
                    return
                elif action == "remove":
                    self.trading_system.remove_from_blacklist([symbol])
                    await update.message.reply_text(f"Removed `{symbol}` from blacklist.", parse_mode='Markdown')
                    return

            # Show current blacklist
            blacklist = self.trading_system.get_blacklist() if hasattr(
                self.trading_system, 'get_blacklist'
            ) else []

            if not blacklist:
                await update.message.reply_text("Blacklist is empty.")
                return

            message = f"*Trading Blacklist ({len(blacklist)} pairs)*\n\n"
            message += ", ".join([f"`{s}`" for s in blacklist])
            message += "\n\nUse `/blacklist add SYMBOL` or `/blacklist remove SYMBOL`"

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_stats(self, update: Update, context: CallbackContext) -> None:
        """Handle /stats command - detailed trading statistics."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            stats = self.trading_system.get_trading_stats() if hasattr(
                self.trading_system, 'get_trading_stats'
            ) else {}

            message = (
                "*Detailed Trading Statistics*\n\n"
                f"*Total Trades:* {stats.get('total_trades', 0)}\n"
                f"*Winning Trades:* {stats.get('winning_trades', 0)}\n"
                f"*Losing Trades:* {stats.get('losing_trades', 0)}\n"
                f"*Win Rate:* {stats.get('win_rate', 0):.1f}%\n\n"
                f"*Profit Factor:* {stats.get('profit_factor', 0):.2f}\n"
                f"*Sharpe Ratio:* {stats.get('sharpe_ratio', 0):.2f}\n"
                f"*Max Drawdown:* {stats.get('max_drawdown', 0):.1f}%\n\n"
                f"*Avg Win:* ${stats.get('avg_win', 0):.2f}\n"
                f"*Avg Loss:* ${stats.get('avg_loss', 0):.2f}\n"
                f"*Best Trade:* ${stats.get('best_trade', 0):.2f}\n"
                f"*Worst Trade:* ${stats.get('worst_trade', 0):.2f}\n\n"
                f"*Win Streak:* {stats.get('max_win_streak', 0)}\n"
                f"*Loss Streak:* {stats.get('max_loss_streak', 0)}\n"
                f"*Avg Duration:* {stats.get('avg_trade_duration', 'N/A')}"
            )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error getting stats: {e}")

    async def _cmd_monthly(self, update: Update, context: CallbackContext) -> None:
        """Handle /monthly command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            monthly = self.trading_system.get_monthly_stats() if hasattr(
                self.trading_system, 'get_monthly_stats'
            ) else {}

            message = "*Monthly Performance*\n\n"
            for month, data in monthly.items():
                profit = data.get('profit', 0)
                emoji = ""
                message += f"{emoji} {month}: ${profit:+,.2f}\n"

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_count(self, update: Update, context: CallbackContext) -> None:
        """Handle /count command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            positions = self.trading_system.get_positions() if hasattr(
                self.trading_system, 'get_positions'
            ) else []

            message = (
                "*Trade Count*\n\n"
                f"Open Positions: `{len(positions)}`\n"
                f"Max Positions: `{getattr(self.trading_system, 'max_positions', 'N/A')}`"
            )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_locks(self, update: Update, context: CallbackContext) -> None:
        """Handle /locks command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            locks = self.trading_system.get_locked_pairs() if hasattr(
                self.trading_system, 'get_locked_pairs'
            ) else []

            if not locks:
                await update.message.reply_text("No pairs currently locked.")
                return

            message = "*Locked Pairs*\n\n"
            for lock in locks:
                message += f"`{lock['pair']}` until {lock.get('until', 'N/A')}\n"
                if lock.get('reason'):
                    message += f"  Reason: {lock['reason']}\n"

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_unlock(self, update: Update, context: CallbackContext) -> None:
        """Handle /unlock command."""
        if not await self._check_authorized(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /unlock <SYMBOL>")
            return

        symbol = context.args[0].upper()

        if self.trading_system and hasattr(self.trading_system, 'unlock_pair'):
            try:
                self.trading_system.unlock_pair(symbol)
                await update.message.reply_text(f"Unlocked `{symbol}`", parse_mode='Markdown')
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")
        else:
            await update.message.reply_text("Trading system not connected or method not available.")

    async def _cmd_edge(self, update: Update, context: CallbackContext) -> None:
        """Handle /edge command."""
        if not await self._check_authorized(update):
            return

        await update.message.reply_text(
            "*Edge Analysis*\n\nEdge positioning data not yet available.",
            parse_mode='Markdown'
        )

    async def _cmd_show_config(self, update: Update, context: CallbackContext) -> None:
        """Handle /show_config command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            config = self.trading_system.get_config() if hasattr(
                self.trading_system, 'get_config'
            ) else {}

            # Filter sensitive data
            safe_keys = ['strategy', 'max_positions', 'stake_amount', 'dry_run']
            safe_config = {k: v for k, v in config.items() if k in safe_keys}

            message = "*Current Configuration*\n\n```\n"
            for key, value in safe_config.items():
                message += f"{key}: {value}\n"
            message += "```"

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_logs(self, update: Update, context: CallbackContext) -> None:
        """Handle /logs command."""
        if not await self._check_authorized(update):
            return

        await update.message.reply_text(
            "*Recent Logs*\n\nLog viewing not yet implemented.",
            parse_mode='Markdown'
        )

    async def _cmd_health(self, update: Update, context: CallbackContext) -> None:
        """Handle /health command."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            health = self.trading_system.get_health() if hasattr(
                self.trading_system, 'get_health'
            ) else {'status': 'unknown'}

            status = health.get('status', 'unknown')
            emoji = ""

            message = (
                f"{emoji} *System Health*\n\n"
                f"Status: `{status}`\n"
                f"Uptime: `{health.get('uptime', 'N/A')}`\n"
                f"Memory: `{health.get('memory_mb', 0):.0f} MB`\n"
                f"CPU: `{health.get('cpu_percent', 0):.1f}%`\n"
                f"API Status: `{health.get('api_status', 'N/A')}`"
            )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_panic(self, update: Update, context: CallbackContext) -> None:
        """Handle /panic command - emergency sell all."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        positions = self.trading_system.get_positions() if hasattr(
            self.trading_system, 'get_positions'
        ) else []

        if not positions:
            await update.message.reply_text("No positions to sell.")
            return

        keyboard = [
            [
                InlineKeyboardButton(
                    f"SELL ALL {len(positions)} POSITIONS",
                    callback_data="panic_confirm"
                ),
            ],
            [
                InlineKeyboardButton("Cancel", callback_data="cancel"),
            ]
        ]

        await update.message.reply_text(
            f"*EMERGENCY SELL*\n\n"
            f"This will close all {len(positions)} positions immediately.\n"
            f"Are you sure?",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    # Options-specific commands

    async def _cmd_greeks(self, update: Update, context: CallbackContext) -> None:
        """Handle /greeks command - show portfolio Greeks."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            greeks = self.trading_system.get_portfolio_greeks() if hasattr(
                self.trading_system, 'get_portfolio_greeks'
            ) else {}

            message = (
                "*Portfolio Greeks*\n\n"
                f"Delta: `{greeks.get('delta', 0):+.0f}`\n"
                f"Gamma: `{greeks.get('gamma', 0):+.0f}`\n"
                f"Theta: `${greeks.get('theta', 0):+.0f}/day`\n"
                f"Vega: `{greeks.get('vega', 0):+.0f}`\n"
                f"Beta-Weighted Delta: `{greeks.get('beta_weighted_delta', 0):+.0f}`\n\n"
                f"*Risk Metrics*\n"
                f"VaR (95%): `${greeks.get('var_95', 0):.0f}`\n"
                f"VaR (99%): `${greeks.get('var_99', 0):.0f}`"
            )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_spreads(self, update: Update, context: CallbackContext) -> None:
        """Handle /spreads command - show open option spreads."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            spreads = self.trading_system.get_option_spreads() if hasattr(
                self.trading_system, 'get_option_spreads'
            ) else []

            if not spreads:
                await update.message.reply_text("No open option spreads.")
                return

            message = "*Open Option Spreads*\n\n"
            for spread in spreads:
                pnl = spread.get('unrealized_pnl', 0)
                emoji = ""
                message += (
                    f"{emoji} *{spread['underlying']}* - {spread['type']}\n"
                    f"  Expiry: {spread['expiry']}\n"
                    f"  P&L: ${pnl:+.2f}\n\n"
                )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_iv(self, update: Update, context: CallbackContext) -> None:
        """Handle /iv command - show IV rank/percentile."""
        if not await self._check_authorized(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /iv <SYMBOL>")
            return

        symbol = context.args[0].upper()

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            iv_data = self.trading_system.get_iv_data(symbol) if hasattr(
                self.trading_system, 'get_iv_data'
            ) else {}

            message = (
                f"*{symbol} IV Analysis*\n\n"
                f"Current IV: `{iv_data.get('iv', 0):.1f}%`\n"
                f"IV Rank: `{iv_data.get('iv_rank', 0):.1f}%`\n"
                f"IV Percentile: `{iv_data.get('iv_percentile', 0):.1f}%`\n"
                f"52-Week High IV: `{iv_data.get('iv_high', 0):.1f}%`\n"
                f"52-Week Low IV: `{iv_data.get('iv_low', 0):.1f}%`"
            )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_expiry(self, update: Update, context: CallbackContext) -> None:
        """Handle /expiry command - show positions by expiration."""
        if not await self._check_authorized(update):
            return

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            by_expiry = self.trading_system.get_positions_by_expiry() if hasattr(
                self.trading_system, 'get_positions_by_expiry'
            ) else {}

            if not by_expiry:
                await update.message.reply_text("No options positions.")
                return

            message = "*Positions by Expiration*\n\n"
            for expiry, positions in sorted(by_expiry.items()):
                message += f"*{expiry}*\n"
                for pos in positions:
                    message += f"  {pos['symbol']}: {pos['quantity']}\n"
                message += "\n"

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_roll(self, update: Update, context: CallbackContext) -> None:
        """Handle /roll command - get roll recommendation."""
        if not await self._check_authorized(update):
            return

        if not context.args:
            await update.message.reply_text("Usage: /roll <SYMBOL>")
            return

        symbol = context.args[0].upper()

        if not self.trading_system:
            await update.message.reply_text("Trading system not connected.")
            return

        try:
            recommendation = self.trading_system.get_roll_recommendation(symbol) if hasattr(
                self.trading_system, 'get_roll_recommendation'
            ) else None

            if not recommendation:
                await update.message.reply_text(f"No roll recommendation available for {symbol}")
                return

            message = (
                f"*Roll Recommendation for {symbol}*\n\n"
                f"Action: `{recommendation.get('action', 'N/A')}`\n"
                f"New Expiry: `{recommendation.get('new_expiry', 'N/A')}`\n"
                f"New Strike: `${recommendation.get('new_strike', 'N/A')}`\n"
                f"Est. Credit/Debit: `${recommendation.get('credit_debit', 0):+.2f}`\n"
                f"Reason: {recommendation.get('reason', 'N/A')}"
            )

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
