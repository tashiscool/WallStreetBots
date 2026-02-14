"""Exactly-once order replay guard.

This module prevents duplicate order submission on system restart by
persisting client_order_id -> terminal_state mappings.
"""
from __future__ import annotations
import json
import pathlib
import logging
import threading
from typing import Dict, Optional

log = logging.getLogger("wsb.replay_guard")


class ReplayGuard:
    """Persists client_order_id -> terminal_state. Never re-send acknowledged ids.
    
    This prevents duplicate order submission on system restart by maintaining
    a persistent record of all order states.
    """

    def __init__(self, path: str = "./.state/replay_guard.json"):
        """Initialize replay guard.
        
        Args:
            path: Path to persistent state file
        """
        self.p = pathlib.Path(path)
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, str] = {}
        self._lock = threading.Lock()  # Thread safety
        
        # Create the file if it doesn't exist
        if not self.p.exists():
            self.p.touch()
        self._load_state()

    def _load_state(self) -> None:
        """Load state from persistent storage."""
        if self.p.exists():
            try:
                self.state = json.loads(self.p.read_text())
                log.info(f"Loaded {len(self.state)} order states from replay guard")
            except Exception as e:
                log.error(f"Failed to load replay guard state: {e}")
                self.state = {}
        else:
            # File doesn't exist, clear in-memory state
            self.state = {}

    def _save_state(self) -> None:
        """Save state to persistent storage."""
        try:
            self.p.write_text(json.dumps(self.state, separators=(",", ":")))
        except Exception as e:
            log.error(f"Failed to save replay guard state: {e}")

    def seen(self, client_order_id: str) -> bool:
        """Check if order ID has been seen before.

        Args:
            client_order_id: Client order identifier

        Returns:
            True if order has been seen before
        """
        with self._lock:
            return client_order_id in self.state

    def record(self, client_order_id: str, state: str) -> None:
        """Record order state.

        Args:
            client_order_id: Client order identifier
            state: Terminal state (e.g., 'acknowledged', 'filled', 'rejected')
        """
        with self._lock:
            self.state[client_order_id] = state
            self._save_state()
            log.debug(f"Recorded order {client_order_id} in state {state}")

    def get_state(self, client_order_id: str) -> Optional[str]:
        """Get recorded state for order.

        Args:
            client_order_id: Client order identifier

        Returns:
            Recorded state or None if not found
        """
        with self._lock:
            return self.state.get(client_order_id)

    def clear(self) -> None:
        """Clear all recorded states (use with caution)."""
        with self._lock:
            self.state.clear()
            self._save_state()
            log.warning("Cleared all replay guard states")

    def cleanup_old_orders(self, max_age_days: int = 30) -> int:
        """Clean up old order records.
        
        Args:
            max_age_days: Maximum age in days for order records
            
        Returns:
            Number of orders cleaned up
        """
        # This is a placeholder - in production you'd want to track timestamps
        # and clean up based on actual age
        log.info(f"Cleanup requested for orders older than {max_age_days} days")
        return 0

    def should_process_order(self, order) -> bool:
        """Atomically check if an order should be processed and record it.

        This combines the check and the record into a single atomic operation
        under the lock to prevent TOCTOU race conditions.

        Args:
            order: Order object with client_order_id attribute

        Returns:
            True if order should be processed (not seen before)
        """
        with self._lock:
            # Reload state to get latest changes from other processes
            self._load_state()

            client_order_id = None
            if hasattr(order, 'client_order_id'):
                client_order_id = order.client_order_id
            elif hasattr(order, 'id'):
                client_order_id = order.id

            if client_order_id is None:
                # If no client_order_id, assume it should be processed
                return True

            # Atomic check-and-record: both under the same lock acquisition
            if client_order_id in self.state:
                return False

            self.state[client_order_id] = "processed"
            self._save_state()
            log.debug(f"Recorded order {client_order_id} in state processed")
            return True

    def status(self) -> dict:
        """Get replay guard status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "total_orders": len(self.state),
            "state_file": str(self.p),
            "file_exists": self.p.exists(),
        }

