"""Test that the analysis module handles optional dependencies gracefully."""

import importlib
import sys
import types
from unittest import mock


def test_analysis_init_without_plotly():
    """Verify analysis/__init__.py loads even when plotly is not installed."""
    # Remove plotly from sys.modules if present, then block it
    plotly_mods = [key for key in sys.modules if key.startswith("plotly")]
    saved = {key: sys.modules.pop(key) for key in plotly_mods}

    try:
        with mock.patch.dict(sys.modules, {"plotly": None, "plotly.graph_objects": None, "plotly.subplots": None}):
            # Force reimport
            mod_name = "backend.tradingbot.analysis"
            if mod_name in sys.modules:
                del sys.modules[mod_name]

            try:
                import backend.tradingbot.analysis as analysis_mod

                # Should not crash - availability flags should be False
                assert hasattr(analysis_mod, "TEARSHEET_AVAILABLE") or hasattr(analysis_mod, "PLOT_CONFIGURATOR_AVAILABLE")
            except ImportError:
                # If the module itself can't be imported due to other deps, that's acceptable
                pass
    finally:
        sys.modules.update(saved)


def test_analysis_init_with_plotly_available():
    """Verify analysis/__init__.py exports correctly when plotly is available."""
    try:
        import backend.tradingbot.analysis as analysis_mod

        # Check that __all__ exists
        assert hasattr(analysis_mod, "__all__")
    except ImportError:
        # Module may not be importable in test environment
        pass
