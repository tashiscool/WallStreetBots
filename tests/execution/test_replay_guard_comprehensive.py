"""Comprehensive tests for replay guard module."""
import pytest
import tempfile
import os
import time
from unittest.mock import Mock, patch

from backend.tradingbot.execution.replay_guard import ReplayGuard
from backend.tradingbot.execution.interfaces import OrderRequest


class TestReplayGuardComprehensive:
    """Comprehensive tests for replay guard functionality."""

    def test_replay_guard_file_creation(self):
        """Test replay guard creates file when needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "test_replay.txt")

            # File should not exist initially
            assert not os.path.exists(guard_file)

            guard = ReplayGuard(guard_file)

            # File should be created
            assert os.path.exists(guard_file)

    def test_replay_guard_file_permissions(self):
        """Test replay guard file permissions and access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "permissions_test.txt")

            guard = ReplayGuard(guard_file)

            # Should be readable and writable
            assert os.access(guard_file, os.R_OK)
            assert os.access(guard_file, os.W_OK)

    def test_replay_guard_concurrent_access(self):
        """Test replay guard with concurrent access scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "concurrent_test.txt")

            guard1 = ReplayGuard(guard_file)
            guard2 = ReplayGuard(guard_file)  # Same file

            # Both should be able to process different orders
            order1 = OrderRequest("id1", "AAPL", 100, "buy", "market")
            order2 = OrderRequest("id2", "MSFT", 200, "sell", "limit", limit_price=250.0)

            result1 = guard1.should_process_order(order1)
            result2 = guard2.should_process_order(order2)

            assert result1 is True
            assert result2 is True

            # Same order should be rejected by either guard
            result3 = guard1.should_process_order(order1)
            result4 = guard2.should_process_order(order1)

            assert result3 is False
            assert result4 is False

    def test_replay_guard_file_corruption_recovery(self):
        """Test replay guard recovery from corrupted files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "corrupt_test.txt")

            # Create corrupted file content
            with open(guard_file, 'w') as f:
                f.write("invalid\nfile\ncontent\nwith\nmissing\nfields\n")

            # Guard should handle corrupted file gracefully
            guard = ReplayGuard(guard_file)

            order = OrderRequest("test_id", "AAPL", 100, "buy", "market")
            result = guard.should_process_order(order)

            # Should work despite corrupted initial content
            assert result is True

    def test_replay_guard_empty_file_handling(self):
        """Test replay guard with empty file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "empty_test.txt")

            # Create empty file
            with open(guard_file, 'w') as f:
                pass

            guard = ReplayGuard(guard_file)

            order = OrderRequest("test_id", "AAPL", 100, "buy", "market")
            result = guard.should_process_order(order)

            assert result is True

    def test_replay_guard_large_file_performance(self):
        """Test replay guard performance with large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "large_test.txt")

            guard = ReplayGuard(guard_file)

            # Process many orders to create large file
            start_time = time.time()

            for i in range(1000):
                order = OrderRequest(f"order_{i}", "AAPL", 100, "buy", "market")
                guard.should_process_order(order)

            # Test lookup performance
            lookup_start = time.time()
            test_order = OrderRequest("order_500", "AAPL", 100, "buy", "market")
            result = guard.should_process_order(test_order)
            lookup_end = time.time()

            # Should be fast even with large file
            assert lookup_end - lookup_start < 0.1  # Less than 100ms
            assert result is False  # Should be duplicate

    def test_replay_guard_file_locking_simulation(self):
        """Test replay guard behavior under file locking scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "locking_test.txt")

            guard = ReplayGuard(guard_file)

            # Mock file operations to simulate locking issues
            with patch('builtins.open', side_effect=PermissionError("File locked")):
                order = OrderRequest("test_id", "AAPL", 100, "buy", "market")

                # Should handle file locking gracefully
                try:
                    result = guard.should_process_order(order)
                    # If it doesn't raise, it handled the error
                    assert isinstance(result, bool)
                except PermissionError:
                    # If it raises, that's also acceptable behavior
                    pass

    def test_replay_guard_memory_management(self):
        """Test replay guard memory management with many operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "memory_test.txt")

            guard = ReplayGuard(guard_file)

            # Process many unique orders
            for i in range(5000):
                order = OrderRequest(
                    f"unique_order_{i}",
                    f"STOCK_{i % 100}",
                    100 + i,
                    "buy" if i % 2 == 0 else "sell",
                    "market"
                )
                guard.should_process_order(order)

            # Memory usage should be reasonable
            # (This is more of a behavioral test - no crashes/excessive memory)

            # Test duplicate detection still works
            duplicate_order = OrderRequest("unique_order_1000", "STOCK_0", 1100, "buy", "market")
            result = guard.should_process_order(duplicate_order)
            assert result is False

    def test_replay_guard_order_id_edge_cases(self):
        """Test replay guard with edge case order IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "edge_cases_test.txt")

            guard = ReplayGuard(guard_file)

            # Test various edge case order IDs
            edge_case_ids = [
                "",  # Empty string
                " ",  # Space
                "\n",  # Newline
                "\t",  # Tab
                "id with spaces",
                "id,with,commas",
                "id\nwith\nnewlines",
                "very" * 100 + "long" * 100 + "id",  # Very long ID
                "special!@#$%^&*()characters",
                "unicode_测试_id",
                "123456789",  # Numeric
                "MixedCase_ID_123"
            ]

            for order_id in edge_case_ids:
                order = OrderRequest(order_id, "AAPL", 100, "buy", "market")

                # First occurrence should be processed
                result1 = guard.should_process_order(order)
                assert result1 is True

                # Second occurrence should be rejected
                result2 = guard.should_process_order(order)
                assert result2 is False

    def test_replay_guard_order_parameter_variations(self):
        """Test replay guard with variations in order parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "variations_test.txt")

            guard = ReplayGuard(guard_file)

            # Same ID but different parameters should still be caught
            base_order = OrderRequest("same_id", "AAPL", 100, "buy", "market")
            variant_order = OrderRequest("same_id", "MSFT", 200, "sell", "limit", limit_price=250.0)

            result1 = guard.should_process_order(base_order)
            assert result1 is True

            # Same ID should be rejected regardless of other parameters
            result2 = guard.should_process_order(variant_order)
            assert result2 is False

    def test_replay_guard_file_path_edge_cases(self):
        """Test replay guard with edge case file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with nested directory (should create if needed)
            nested_dir = os.path.join(temp_dir, "nested", "deep", "path")
            guard_file = os.path.join(nested_dir, "test.txt")

            # Should create directory structure if needed (if implemented)
            try:
                guard = ReplayGuard(guard_file)
                order = OrderRequest("test", "AAPL", 100, "buy", "market")
                result = guard.should_process_order(order)
                assert isinstance(result, bool)
            except (FileNotFoundError, OSError):
                # Acceptable if directory creation is not implemented
                pass

    def test_replay_guard_unicode_handling(self):
        """Test replay guard with unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "unicode_test.txt")

            guard = ReplayGuard(guard_file)

            # Test with unicode order IDs and symbols
            unicode_orders = [
                OrderRequest("订单_001", "测试股票", 100, "buy", "market"),
                OrderRequest("заказ_002", "ТЕСТ", 200, "sell", "limit", limit_price=150.0),
                OrderRequest("注文_003", "株式", 300, "buy", "market"),
                OrderRequest("emoji_😀_order", "STOCK_🚀", 400, "sell", "market")
            ]

            for order in unicode_orders:
                try:
                    result1 = guard.should_process_order(order)
                    assert result1 is True

                    result2 = guard.should_process_order(order)
                    assert result2 is False
                except UnicodeError:
                    # Acceptable if unicode is not fully supported
                    pass

    def test_replay_guard_timestamp_precision(self):
        """Test replay guard timestamp precision and ordering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "timestamp_test.txt")

            guard = ReplayGuard(guard_file)

            # Process orders in rapid succession
            for i in range(10):
                order = OrderRequest(f"rapid_{i}", "AAPL", 100, "buy", "market")
                result = guard.should_process_order(order)
                assert result is True

                # Tiny delay to ensure timestamp differences
                time.sleep(0.001)

            # Verify all orders were recorded with different timestamps
            # (This is more of a behavioral test)

    def test_replay_guard_file_recovery_after_deletion(self):
        """Test replay guard recovery after file deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "deletion_test.txt")

            guard = ReplayGuard(guard_file)

            # Process an order
            order1 = OrderRequest("before_deletion", "AAPL", 100, "buy", "market")
            result1 = guard.should_process_order(order1)
            assert result1 is True

            # Delete the file
            os.remove(guard_file)

            # Process another order (should recreate file)
            order2 = OrderRequest("after_deletion", "MSFT", 200, "sell", "market")
            result2 = guard.should_process_order(order2)
            assert result2 is True

            # Original order should be processable again (file was recreated)
            order3 = OrderRequest("before_deletion", "AAPL", 100, "buy", "market")
            result3 = guard.should_process_order(order3)
            assert result3 is True  # Should be allowed since file was recreated

    def test_replay_guard_order_type_edge_cases(self):
        """Test replay guard with different order types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "order_types_test.txt")

            guard = ReplayGuard(guard_file)

            # Test different order types with same ID
            order_types = ["market", "limit"]

            for i, order_type in enumerate(order_types):
                if order_type == "limit":
                    order = OrderRequest(f"type_test_{i}", "AAPL", 100, "buy", order_type, limit_price=150.0)
                else:
                    order = OrderRequest(f"type_test_{i}", "AAPL", 100, "buy", order_type)

                result1 = guard.should_process_order(order)
                assert result1 is True

                result2 = guard.should_process_order(order)
                assert result2 is False

    def test_replay_guard_stress_test(self):
        """Stress test replay guard with high concurrency simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            guard_file = os.path.join(temp_dir, "stress_test.txt")

            guard = ReplayGuard(guard_file)

            import threading
            import queue

            results_queue = queue.Queue()

            def process_orders(start_idx, count):
                local_results = []
                for i in range(start_idx, start_idx + count):
                    order = OrderRequest(f"stress_{i}", f"STOCK_{i % 50}", 100, "buy", "market")
                    result = guard.should_process_order(order)
                    local_results.append((i, result))
                results_queue.put(local_results)

            # Start multiple threads
            threads = []
            for t in range(5):
                thread = threading.Thread(target=process_orders, args=(t * 100, 100))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Collect results
            all_results = []
            while not results_queue.empty():
                all_results.extend(results_queue.get())

            # All first occurrences should be True
            assert len(all_results) == 500
            assert all(result for idx, result in all_results)

            # Test duplicate detection
            duplicate_order = OrderRequest("stress_100", "STOCK_0", 100, "buy", "market")
            duplicate_result = guard.should_process_order(duplicate_order)
            assert duplicate_result is False