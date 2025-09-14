"""Build and version information for reproducibility.

This module provides build identification and version stamping
for orders, logs, and journal entries.
"""
from __future__ import annotations
import subprocess
import os
import logging
from typing import Optional

log = logging.getLogger("wsb.build_info")


def build_id() -> str:
    """Get build identifier.
    
    Returns:
        Git SHA or environment variable or 'unknown'
    """
    # Check environment variable first (for CI/CD)
    sha = os.getenv("WSB_GIT_SHA")
    if sha:
        return sha
    
    # Try to get from git
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            text=True, 
            stderr=subprocess.DEVNULL
        )
        return result.strip()
    except Exception:
        pass
    
    # Fallback to full SHA
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            text=True, 
            stderr=subprocess.DEVNULL
        )
        return result.strip()
    except Exception:
        pass
    
    return "unknown"


def build_timestamp() -> str:
    """Get build timestamp.
    
    Returns:
        ISO timestamp string
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def version_info() -> dict:
    """Get comprehensive version information.
    
    Returns:
        Dictionary with version details
    """
    return {
        "build_id": build_id(),
        "build_timestamp": build_timestamp(),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "platform": os.name,
    }


def get_strategy_version(strategy_name: str) -> str:
    """Get version for specific strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Version string combining build_id and strategy
    """
    return f"{strategy_name}@{build_id()}"


def stamp_order(order_data: dict) -> dict:
    """Add build stamp to order data.
    
    Args:
        order_data: Order dictionary to stamp
        
    Returns:
        Order data with build information added
    """
    order_data["build_id"] = build_id()
    order_data["build_timestamp"] = build_timestamp()
    return order_data


def stamp_log_entry(log_data: dict) -> dict:
    """Add build stamp to log entry.
    
    Args:
        log_data: Log dictionary to stamp
        
    Returns:
        Log data with build information added
    """
    log_data["build_id"] = build_id()
    log_data["build_timestamp"] = build_timestamp()
    return log_data


class BuildStamper:
    """Context manager for adding build stamps to operations."""
    
    def __init__(self, operation_type: str):
        """Initialize stamper.
        
        Args:
            operation_type: Type of operation being stamped
        """
        self.operation_type = operation_type
        self.build_id = build_id()
        self.timestamp = build_timestamp()
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass
    
    def stamp(self, data: dict) -> dict:
        """Stamp data with build info.
        
        Args:
            data: Data to stamp
            
        Returns:
            Stamped data
        """
        data.update({
            "build_id": self.build_id,
            "build_timestamp": self.timestamp,
            "operation_type": self.operation_type,
        })
        return data
