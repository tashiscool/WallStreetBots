"""
CLI Diagnostic Printer for Order Book Visualization.

Synthesized from:
- kalshi-polymarket-arbitrage-bot: DiagnosticPrinter service
- ASCII table formatting for terminal display

Real-time terminal-based monitoring tool.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable
import threading

from .platform_client import OrderBook, PriceLevel, Platform, Outcome, MarketState

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic printer."""
    # Print interval in seconds
    interval: float = 5.0

    # Order book depth to display
    depth: int = 5

    # Column widths
    price_width: int = 10
    size_width: int = 10

    # Enable/disable sections
    show_orderbooks: bool = True
    show_spreads: bool = True
    show_opportunities: bool = True
    show_stats: bool = True

    # Clear screen before each print
    clear_screen: bool = False

    # Use colors (ANSI)
    use_colors: bool = True


class Colors:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"


class DiagnosticPrinter:
    """
    Terminal-based diagnostic printer for order book visualization.

    From kalshi-polymarket-arbitrage-bot: Periodic ASCII snapshots.
    """

    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """
        Initialize diagnostic printer.

        Args:
            config: Printer configuration
        """
        self._config = config or DiagnosticConfig()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Data sources (set by integration)
        self._get_polymarket_book: Optional[Callable] = None
        self._get_kalshi_book: Optional[Callable] = None
        self._get_opportunities: Optional[Callable] = None
        self._get_stats: Optional[Callable] = None

        # State cache
        self._markets: Dict[str, MarketState] = {}
        self._lock = threading.RLock()

        # Detect if colors are supported
        self._use_colors = self._config.use_colors and sys.stdout.isatty()

    def set_data_sources(
        self,
        get_polymarket_book: Optional[Callable] = None,
        get_kalshi_book: Optional[Callable] = None,
        get_opportunities: Optional[Callable] = None,
        get_stats: Optional[Callable] = None,
    ) -> None:
        """Set data source callbacks."""
        self._get_polymarket_book = get_polymarket_book
        self._get_kalshi_book = get_kalshi_book
        self._get_opportunities = get_opportunities
        self._get_stats = get_stats

    def update_market(self, market_id: str, state: MarketState) -> None:
        """Update cached market state."""
        with self._lock:
            self._markets[market_id] = state

    async def start(self) -> None:
        """Start periodic printing."""
        self._running = True
        self._task = asyncio.create_task(self._run_printer_loop())
        logger.info(f"Diagnostic printer started (interval={self._config.interval}s)")

    async def stop(self) -> None:
        """Stop periodic printing."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Diagnostic printer stopped")

    async def _run_printer_loop(self) -> None:
        """Main printer loop."""
        while self._running:
            try:
                self._print_snapshot()
                await asyncio.sleep(self._config.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in printer loop: {e}")
                await asyncio.sleep(1.0)

    def _print_snapshot(self) -> None:
        """Print complete diagnostic snapshot."""
        if self._config.clear_screen:
            print("\033[2J\033[H", end="")  # Clear screen

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._print_header(f"ARBITRAGE DIAGNOSTICS - {timestamp}")

        with self._lock:
            # Print order books for each market
            if self._config.show_orderbooks:
                for market_id, state in self._markets.items():
                    self._print_market_books(market_id, state)

            # Print opportunities
            if self._config.show_opportunities and self._get_opportunities:
                opportunities = self._get_opportunities()
                self._print_opportunities(opportunities)

            # Print stats
            if self._config.show_stats and self._get_stats:
                stats = self._get_stats()
                self._print_stats(stats)

        print()  # Final newline

    def _print_header(self, title: str) -> None:
        """Print section header."""
        width = 80
        if self._use_colors:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'=' * width}{Colors.RESET}\n")
        else:
            print(f"\n{'=' * width}")
            print(f"{title.center(width)}")
            print(f"{'=' * width}\n")

    def _print_subheader(self, title: str) -> None:
        """Print subsection header."""
        if self._use_colors:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}--- {title} ---{Colors.RESET}")
        else:
            print(f"\n--- {title} ---")

    def _print_market_books(self, market_id: str, state: MarketState) -> None:
        """Print order books for a market."""
        self._print_subheader(f"Market: {market_id}")

        # Print Polymarket books
        print("\n  POLYMARKET:")
        if state.yes_book:
            self._print_order_book(state.yes_book, "YES", Platform.POLYMARKET)
        if state.no_book:
            self._print_order_book(state.no_book, "NO", Platform.POLYMARKET)

        # Print spread info
        if state.yes_price and state.no_price:
            combined = state.yes_price + state.no_price
            edge = Decimal("1.0") - combined
            edge_color = Colors.GREEN if edge > 0 else Colors.RED

            if self._use_colors:
                print(f"\n  Combined: ${combined:.4f}  |  "
                      f"Edge: {edge_color}${edge:.4f}{Colors.RESET}")
            else:
                print(f"\n  Combined: ${combined:.4f}  |  Edge: ${edge:.4f}")

    def _print_order_book(
        self,
        book: OrderBook,
        outcome: str,
        platform: Platform,
    ) -> None:
        """Print formatted order book."""
        depth = self._config.depth
        pw = self._config.price_width
        sw = self._config.size_width

        # Header
        header = f"    {outcome} Book (Last Update: {book.last_update.strftime('%H:%M:%S')})"
        print(header)

        # Column headers
        bid_header = f"{'Bid Price':>{pw}} {'Size':>{sw}}"
        ask_header = f"{'Ask Price':>{pw}} {'Size':>{sw}}"
        print(f"    {bid_header}  |  {ask_header}")
        print(f"    {'-' * (pw + sw + 1)}  |  {'-' * (pw + sw + 1)}")

        # Get levels
        bids = sorted(book.bids, key=lambda x: x.price, reverse=True)[:depth]
        asks = sorted(book.asks, key=lambda x: x.price)[:depth]

        # Pad to same length
        max_len = max(len(bids), len(asks), 1)

        for i in range(max_len):
            bid_str = self._format_level(bids[i] if i < len(bids) else None, pw, sw)
            ask_str = self._format_level(asks[i] if i < len(asks) else None, pw, sw)

            if self._use_colors:
                bid_colored = f"{Colors.GREEN}{bid_str}{Colors.RESET}" if i < len(bids) else bid_str
                ask_colored = f"{Colors.RED}{ask_str}{Colors.RESET}" if i < len(asks) else ask_str
                print(f"    {bid_colored}  |  {ask_colored}")
            else:
                print(f"    {bid_str}  |  {ask_str}")

        # Spread
        if book.best_bid and book.best_ask:
            spread = book.best_ask.price - book.best_bid.price
            mid = (book.best_ask.price + book.best_bid.price) / 2
            print(f"    Spread: ${spread:.4f} | Mid: ${mid:.4f}")

    def _format_level(
        self,
        level: Optional[PriceLevel],
        price_width: int,
        size_width: int,
    ) -> str:
        """Format a price level."""
        if level is None:
            return f"{'-':>{price_width}} {'-':>{size_width}}"
        return f"${level.price:>{price_width - 1}.4f} {level.size:>{size_width}}"

    def _print_opportunities(self, opportunities: List[Dict[str, Any]]) -> None:
        """Print active opportunities."""
        self._print_subheader("Active Opportunities")

        if not opportunities:
            print("  No active opportunities")
            return

        for opp in opportunities[:10]:  # Limit to 10
            strategy = opp.get("strategy", "unknown")
            edge = opp.get("net_profit", 0)
            size = opp.get("max_size", 0)

            if self._use_colors:
                edge_color = Colors.GREEN if edge > 0 else Colors.RED
                print(f"  {Colors.BOLD}{strategy}{Colors.RESET}: "
                      f"Edge={edge_color}${edge:.4f}{Colors.RESET} "
                      f"Size={size}")
            else:
                print(f"  {strategy}: Edge=${edge:.4f} Size={size}")

    def _print_stats(self, stats: Dict[str, Any]) -> None:
        """Print statistics."""
        self._print_subheader("Statistics")

        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    def print_comparison(
        self,
        poly_book: OrderBook,
        kalshi_book: OrderBook,
        market_name: str = "Market",
    ) -> None:
        """
        Print side-by-side comparison of two order books.

        From kalshi-polymarket-arbitrage-bot: Cross-platform comparison.
        """
        depth = self._config.depth
        pw = self._config.price_width
        sw = self._config.size_width

        self._print_subheader(f"Cross-Platform Comparison: {market_name}")

        # Headers
        poly_header = "POLYMARKET"
        kalshi_header = "KALSHI"
        col_width = pw + sw + 5

        print(f"    {poly_header:^{col_width * 2}}  |  {kalshi_header:^{col_width * 2}}")
        print(f"    {'Bid':^{col_width}}{'Ask':^{col_width}}  |  "
              f"{'Bid':^{col_width}}{'Ask':^{col_width}}")
        print(f"    {'-' * col_width * 2}  |  {'-' * col_width * 2}")

        # Get levels
        poly_bids = sorted(poly_book.bids, key=lambda x: x.price, reverse=True)[:depth]
        poly_asks = sorted(poly_book.asks, key=lambda x: x.price)[:depth]
        kalshi_bids = sorted(kalshi_book.bids, key=lambda x: x.price, reverse=True)[:depth]
        kalshi_asks = sorted(kalshi_book.asks, key=lambda x: x.price)[:depth]

        max_len = max(len(poly_bids), len(poly_asks), len(kalshi_bids), len(kalshi_asks), 1)

        for i in range(max_len):
            pb = self._format_level(poly_bids[i] if i < len(poly_bids) else None, pw, sw)
            pa = self._format_level(poly_asks[i] if i < len(poly_asks) else None, pw, sw)
            kb = self._format_level(kalshi_bids[i] if i < len(kalshi_bids) else None, pw, sw)
            ka = self._format_level(kalshi_asks[i] if i < len(kalshi_asks) else None, pw, sw)

            print(f"    {pb} {pa}  |  {kb} {ka}")

        # Arbitrage analysis
        print()
        if poly_book.best_ask and kalshi_book.best_ask:
            poly_yes_price = poly_book.best_ask.price
            kalshi_no_price = kalshi_book.best_ask.price if kalshi_book.best_ask else Decimal("1")

            combined = poly_yes_price + kalshi_no_price
            edge = Decimal("1.0") - combined

            if self._use_colors:
                if edge > 0:
                    print(f"    {Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD}"
                          f" ARBITRAGE OPPORTUNITY: Edge = ${edge:.4f} "
                          f"{Colors.RESET}")
                else:
                    print(f"    Combined Cost: ${combined:.4f} (No Arb)")
            else:
                if edge > 0:
                    print(f"    *** ARBITRAGE OPPORTUNITY: Edge = ${edge:.4f} ***")
                else:
                    print(f"    Combined Cost: ${combined:.4f} (No Arb)")


def run_diagnostic_printer(
    config: Optional[DiagnosticConfig] = None,
) -> DiagnosticPrinter:
    """
    Create and return a diagnostic printer instance.

    Usage:
        printer = run_diagnostic_printer()
        printer.update_market("market_123", market_state)
        await printer.start()
        # ... later ...
        await printer.stop()
    """
    printer = DiagnosticPrinter(config)
    return printer


# =============================================================================
# Quick Terminal Output Functions
# =============================================================================

def print_order_book_snapshot(book: OrderBook, title: str = "Order Book") -> None:
    """Quick print of a single order book."""
    printer = DiagnosticPrinter()
    print(f"\n{title}")
    print("-" * 40)
    printer._print_order_book(book, "YES", Platform.POLYMARKET)


def print_arbitrage_summary(
    poly_yes: Decimal,
    poly_no: Decimal,
    kalshi_yes: Decimal,
    kalshi_no: Decimal,
) -> None:
    """Print quick arbitrage summary."""
    # Strategy 1: Buy Poly YES + Kalshi NO
    strat1_cost = poly_yes + kalshi_no
    strat1_edge = Decimal("1.0") - strat1_cost

    # Strategy 2: Buy Kalshi YES + Poly NO
    strat2_cost = kalshi_yes + poly_no
    strat2_edge = Decimal("1.0") - strat2_cost

    print("\n" + "=" * 50)
    print("ARBITRAGE SUMMARY".center(50))
    print("=" * 50)
    print(f"\nPolymarket YES: ${poly_yes:.4f}  NO: ${poly_no:.4f}")
    print(f"Kalshi     YES: ${kalshi_yes:.4f}  NO: ${kalshi_no:.4f}")
    print(f"\nStrategy 1 (Poly YES + Kalshi NO):")
    print(f"  Cost: ${strat1_cost:.4f}  Edge: ${strat1_edge:.4f}")
    print(f"\nStrategy 2 (Kalshi YES + Poly NO):")
    print(f"  Cost: ${strat2_cost:.4f}  Edge: ${strat2_edge:.4f}")

    best = "Strategy 1" if strat1_edge > strat2_edge else "Strategy 2"
    best_edge = max(strat1_edge, strat2_edge)

    if best_edge > 0:
        print(f"\n*** BEST: {best} with ${best_edge:.4f} edge ***")
    else:
        print(f"\n(No arbitrage opportunity)")
    print("=" * 50 + "\n")
