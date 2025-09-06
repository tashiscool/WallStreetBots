#!/usr/bin/env python3
"""
Comprehensive Test Suite for Alert System
Tests alert generation, delivery, and execution checklists
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.tradingbot.  # noqa: E402alert_system import (
    Alert, AlertType, AlertPriority, AlertChannel, ChecklistItem, ExecutionChecklist,
    TradingAlertSystem, AlertHandler, DesktopAlertHandler, EmailAlertHandler,
    ExecutionChecklistManager, send_slack, send_email
)


class TestAlertDataClasses(unittest.TestCase):
    """Test alert data classes and structures"""
    
    def setUp(self):
        self.sample_alert = Alert(
            alert_type=AlertType.ENTRY_SIGNAL,
            priority=AlertPriority.HIGH,
            ticker="AAPL",
            title="Entry Signal Detected",
            message="Bull market pullback setup confirmed for AAPL",
            data={"price": 150.0, "confidence": 0.85}
        )
    
    def test_alert_creation(self):
        """Test alert object creation and properties"""
        self.assertEqual(self.sample_alert.alert_type, AlertType.ENTRY_SIGNAL)
        self.assertEqual(self.sample_alert.priority, AlertPriority.HIGH)
        self.assertEqual(self.sample_alert.ticker, "AAPL")
        self.assertEqual(self.sample_alert.title, "Entry Signal Detected")
        self.assertIn("AAPL", self.sample_alert.message)
        self.assertEqual(self.sample_alert.data["price"], 150.0)
        self.assertFalse(self.sample_alert.acknowledged)
        self.assertIsInstance(self.sample_alert.timestamp, datetime)
    
    def test_alert_json_serialization(self):
        """Test alert JSON serialization"""
        json_str = self.sample_alert.to_json()
        self.assertIsInstance(json_str, str)
        
        # Should be valid JSON
        data = json.loads(json_str)
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(data["alert_type"], "entry_signal")
        self.assertEqual(data["priority"], 3)  # HIGH priority value
    
    def test_alert_enums(self):
        """Test alert enum types"""
        # Test AlertType enum
        self.assertEqual(AlertType.SETUP_DETECTED.value, "setup_detected")
        self.assertEqual(AlertType.ENTRY_SIGNAL.value, "entry_signal")
        self.assertEqual(AlertType.PROFIT_TARGET.value, "profit_target")
        
        # Test AlertPriority enum
        self.assertEqual(AlertPriority.LOW.value, 1)
        self.assertEqual(AlertPriority.URGENT.value, 4)
        
        # Test AlertChannel enum
        self.assertEqual(AlertChannel.EMAIL.value, "email")
        self.assertEqual(AlertChannel.SLACK.value, "slack")
    
    def test_checklist_item_creation(self):
        """Test checklist item functionality"""
        item = ChecklistItem(
            step=1,
            description="Verify market regime is bull"
        )
        
        self.assertEqual(item.step, 1)
        self.assertEqual(item.description, "Verify market regime is bull")
        self.assertFalse(item.completed)
        self.assertIsNone(item.timestamp)
        self.assertEqual(item.notes, "")
        
        # Test completion
        item.complete("Regime verified as bull market")
        self.assertTrue(item.completed)
        self.assertIsInstance(item.timestamp, datetime)
        self.assertEqual(item.notes, "Regime verified as bull market")
    
    def test_execution_checklist(self):
        """Test execution checklist functionality"""
        checklist = ExecutionChecklist(
            trade_id="AAPL_20241206_001",
            ticker="AAPL",
            checklist_type="entry"
        )
        
        # Add items
        items = [
            ChecklistItem(1, "Verify market regime"),
            ChecklistItem(2, "Check technical setup"),
            ChecklistItem(3, "Confirm risk parameters"),
            ChecklistItem(4, "Execute trade")
        ]
        checklist.items = items
        
        # Initially incomplete
        self.assertEqual(checklist.completion_percentage, 0.0)
        self.assertFalse(checklist.is_complete)
        
        # Complete some items
        checklist.complete_step(1, "Bull market confirmed")
        checklist.complete_step(2, "Pullback setup verified")
        
        self.assertEqual(checklist.completion_percentage, 50.0)
        self.assertFalse(checklist.is_complete)
        
        # Complete all items
        checklist.complete_step(3, "Risk within limits")
        checklist.complete_step(4, "Trade executed")
        
        self.assertEqual(checklist.completion_percentage, 100.0)
        self.assertTrue(checklist.is_complete)
        self.assertIsInstance(checklist.completed_at, datetime)


class TestAlertHandlers(unittest.TestCase):
    """Test alert handler implementations"""
    
    def test_desktop_alert_handler(self):
        """Test desktop alert handler"""
        handler = DesktopAlertHandler()
        
        alert = Alert(
            alert_type=AlertType.ENTRY_SIGNAL,
            priority=AlertPriority.HIGH,
            ticker="AAPL",
            title="Entry Signal",
            message="Test message"
        )
        
        # Mock the desktop notification
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            result = handler.send_alert(alert)
            self.assertTrue(result)
            mock_run.assert_called_once()
    
    @patch('backend.tradingbot.alert_system.send_email')
    def test_email_alert_handler(self, mock_send_email):
        """Test email alert handler"""
        mock_send_email.return_value = True
        
        handler = EmailAlertHandler()
        alert = Alert(
            alert_type=AlertType.PROFIT_TARGET,
            priority=AlertPriority.MEDIUM,
            ticker="GOOGL",
            title="Profit Target Hit",
            message="First profit target reached"
        )
        
        result = handler.send_alert(alert)
        self.assertTrue(result)
        mock_send_email.assert_called_once()
    
    def test_alert_handler_interface(self):
        """Test alert handler abstract interface"""
        from backend.tradingbot.alert_system import AlertHandler
        
        # Should not be able to instantiate abstract class
        with self.assertRaises(TypeError):
            AlertHandler()


class TestTradingAlertSystem(unittest.TestCase):
    """Test main trading alert system"""
    
    def setUp(self):
        self.alert_system = TradingAlertSystem()
    
    def test_handler_registration(self):
        """Test alert handler registration"""
        handler = DesktopAlertHandler()
        
        self.alert_system.register_handler(AlertChannel.DESKTOP, handler)
        self.assertIn(AlertChannel.DESKTOP, self.alert_system.handlers)
        self.assertEqual(self.alert_system.handlers[AlertChannel.DESKTOP], handler)
    
    def test_alert_sending(self):
        """Test alert sending through system"""
        # Register mock handler
        mock_handler = Mock()
        mock_handler.send_alert.return_value = True
        self.alert_system.register_handler(AlertChannel.EMAIL, mock_handler)
        
        alert = Alert(
            alert_type=AlertType.SETUP_DETECTED,
            priority=AlertPriority.MEDIUM,
            ticker="TSLA",
            title="Setup Detected",
            message="Pullback setup detected",
            channels=[AlertChannel.EMAIL]
        )
        
        results = self.alert_system.send_alert(alert)
        
        self.assertIn(AlertChannel.EMAIL, results)
        self.assertTrue(results[AlertChannel.EMAIL])
        mock_handler.send_alert.assert_called_once_with(alert)
        self.assertIn(alert, self.alert_system.alert_history)
    
    def test_alert_filtering_by_priority(self):
        """Test alert filtering by priority levels"""
        self.alert_system.min_priority = AlertPriority.HIGH
        
        # Register mock handler
        mock_handler = Mock()
        mock_handler.send_alert.return_value = True
        self.alert_system.register_handler(AlertChannel.DESKTOP, mock_handler)
        
        # Low priority alert should be filtered
        low_alert = Alert(
            alert_type=AlertType.SETUP_DETECTED,
            priority=AlertPriority.LOW,
            ticker="SPY",
            title="Low Priority",
            message="Low priority message",
            channels=[AlertChannel.DESKTOP]
        )
        
        results = self.alert_system.send_alert(low_alert)
        self.assertEqual(results, {})  # No results due to filtering
        mock_handler.send_alert.assert_not_called()
        
        # High priority alert should pass
        high_alert = Alert(
            alert_type=AlertType.ENTRY_SIGNAL,
            priority=AlertPriority.HIGH,
            ticker="SPY",
            title="High Priority",
            message="High priority message",
            channels=[AlertChannel.DESKTOP]
        )
        
        results = self.alert_system.send_alert(high_alert)
        self.assertTrue(results[AlertChannel.DESKTOP])
        mock_handler.send_alert.assert_called_once_with(high_alert)
    
    def test_alert_history_management(self):
        """Test alert history tracking"""
        # Initial state
        self.assertEqual(len(self.alert_system.alert_history), 0)
        
        alert = Alert(
            alert_type=AlertType.TIME_WARNING,
            priority=AlertPriority.MEDIUM,
            ticker="QQQ",
            title="Time Warning",
            message="Options expiring soon"
        )
        
        self.alert_system.send_alert(alert)
        self.assertEqual(len(self.alert_system.alert_history), 1)
        self.assertEqual(self.alert_system.alert_history[0], alert)
        
        # Test history limit
        self.alert_system.max_history = 2
        
        for i in range(3):
            new_alert = Alert(
                alert_type=AlertType.SYSTEM_ERROR,
                priority=AlertPriority.URGENT,
                ticker="TEST",
                title=f"Test Alert {i}",
                message=f"Test message {i}"
            )
            self.alert_system.send_alert(new_alert)
        
        # Should only keep last 2 alerts
        self.assertEqual(len(self.alert_system.alert_history), 2)


class TestExecutionChecklistManager(unittest.TestCase):
    """Test execution checklist management"""
    
    def setUp(self):
        self.checklist_manager = ExecutionChecklistManager()
    
    def test_entry_checklist_creation(self):
        """Test creation of entry checklist"""
        from backend.tradingbot.options_calculator import TradeCalculation
        
        # Mock trade calculation
        trade_calc = Mock(spec=TradeCalculation)
        trade_calc.ticker = "AAPL"
        trade_calc.recommended_contracts = 10
        trade_calc.total_cost = 5000
        trade_calc.account_risk_pct = 2.5
        
        checklist_id = self.checklist_manager.create_entry_checklist("AAPL", trade_calc)
        
        self.assertIsInstance(checklist_id, str)
        self.assertIn(checklist_id, self.checklist_manager.checklists)
        
        checklist = self.checklist_manager.checklists[checklist_id]
        self.assertEqual(checklist.ticker, "AAPL")
        self.assertEqual(checklist.checklist_type, "entry")
        self.assertGreater(len(checklist.items), 5)  # Should have multiple items
    
    def test_monitoring_checklist_creation(self):
        """Test creation of monitoring checklist"""
        checklist_id = self.checklist_manager.create_monitoring_checklist("GOOGL")
        
        checklist = self.checklist_manager.checklists[checklist_id]
        self.assertEqual(checklist.ticker, "GOOGL")
        self.assertEqual(checklist.checklist_type, "monitoring")
        self.assertGreater(len(checklist.items), 3)
    
    def test_exit_checklist_creation(self):
        """Test creation of exit checklist"""
        checklist_id = self.checklist_manager.create_exit_checklist("TSLA", "profit_target")
        
        checklist = self.checklist_manager.checklists[checklist_id]
        self.assertEqual(checklist.ticker, "TSLA")
        self.assertEqual(checklist.checklist_type, "exit")
        self.assertGreater(len(checklist.items), 3)
    
    def test_checklist_completion(self):
        """Test checklist item completion"""
        checklist_id = self.checklist_manager.create_monitoring_checklist("SPY")
        
        # Complete an item
        result = self.checklist_manager.complete_item(checklist_id, 1, "Market regime verified")
        self.assertTrue(result)
        
        checklist = self.checklist_manager.checklists[checklist_id]
        completed_items = [item for item in checklist.items if item.completed]
        self.assertEqual(len(completed_items), 1)
        self.assertEqual(completed_items[0].step, 1)
        self.assertEqual(completed_items[0].notes, "Market regime verified")
    
    def test_checklist_retrieval(self):
        """Test checklist retrieval methods"""
        # Create test checklists
        entry_id = self.checklist_manager.create_entry_checklist("AAPL", Mock())
        monitoring_id = self.checklist_manager.create_monitoring_checklist("AAPL")
        
        # Test get_checklist
        entry_checklist = self.checklist_manager.get_checklist(entry_id)
        self.assertIsNotNone(entry_checklist)
        self.assertEqual(entry_checklist.ticker, "AAPL")
        
        # Test get_checklists_for_ticker
        aapl_checklists = self.checklist_manager.get_checklists_for_ticker("AAPL")
        self.assertEqual(len(aapl_checklists), 2)
        
        # Test get_active_checklists
        active_checklists = self.checklist_manager.get_active_checklists()
        self.assertEqual(len(active_checklists), 2)  # Both are active (incomplete)


class TestAlertUtilities(unittest.TestCase):
    """Test alert utility functions"""
    
    @patch('requests.post')
    def test_send_slack_success(self, mock_post):
        """Test successful Slack message sending"""
        mock_response = Mock()
        mock_response.ok = True
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {'ALERT_SLACK_WEBHOOK': 'https://hooks.slack.com/test'}):
            result = send_slack("Test message")
            self.assertTrue(result)
            mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_send_slack_failure(self, mock_post):
        """Test Slack message sending failure"""
        mock_post.side_effect = Exception("Network error")
        
        with patch.dict(os.environ, {'ALERT_SLACK_WEBHOOK': 'https://hooks.slack.com/test'}):
            result = send_slack("Test message")
            self.assertFalse(result)
    
    def test_send_slack_no_webhook(self):
        """Test Slack sending with no webhook configured"""
        with patch.dict(os.environ, {}, clear=True):
            result = send_slack("Test message")
            self.assertFalse(result)
    
    @patch('smtplib.SMTP')
    def test_send_email_success(self, mock_smtp):
        """Test successful email sending"""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        env_vars = {
            'ALERT_EMAIL_SMTP_HOST': 'smtp.gmail.com',
            'ALERT_EMAIL_SMTP_PORT': '587',
            'ALERT_EMAIL_FROM': 'test@example.com',
            'ALERT_EMAIL_TO': 'recipient@example.com',
            'ALERT_EMAIL_USER': 'user',
            'ALERT_EMAIL_PASS': 'pass'
        }
        
        with patch.dict(os.environ, env_vars):
            result = send_email("Test Subject", "Test body")
            self.assertTrue(result)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.sendmail.assert_called_once()
    
    def test_send_email_missing_config(self):
        """Test email sending with missing configuration"""
        with patch.dict(os.environ, {}, clear=True):
            result = send_email("Test Subject", "Test body")
            self.assertFalse(result)


class TestAlertIntegration(unittest.TestCase):
    """Test alert system integration scenarios"""
    
    def setUp(self):
        self.alert_system = TradingAlertSystem()
        self.checklist_manager = ExecutionChecklistManager()
    
    def test_complete_trading_workflow(self):
        """Test complete trading workflow with alerts and checklists"""
        # 1. Setup detection alert
        setup_alert = Alert(
            alert_type=AlertType.SETUP_DETECTED,
            priority=AlertPriority.MEDIUM,
            ticker="AAPL",
            title="Bull Pullback Setup Detected",
            message="AAPL showing bull market pullback pattern"
        )
        
        # Register mock handler
        mock_handler = Mock()
        mock_handler.send_alert.return_value = True
        self.alert_system.register_handler(AlertChannel.DESKTOP, mock_handler)
        
        # Send setup alert
        results = self.alert_system.send_alert(setup_alert)
        self.assertTrue(results.get(AlertChannel.DESKTOP, False))
        
        # 2. Create entry checklist
        trade_calc = Mock()
        trade_calc.ticker = "AAPL"
        trade_calc.recommended_contracts = 5
        trade_calc.total_cost = 2500
        
        checklist_id = self.checklist_manager.create_entry_checklist("AAPL", trade_calc)
        
        # 3. Complete checklist items
        self.checklist_manager.complete_item(checklist_id, 1, "Bull regime confirmed")
        self.checklist_manager.complete_item(checklist_id, 2, "Pullback setup verified")
        
        # 4. Entry signal alert
        entry_alert = Alert(
            alert_type=AlertType.ENTRY_SIGNAL,
            priority=AlertPriority.HIGH,
            ticker="AAPL",
            title="Entry Signal Confirmed",
            message="All conditions met for AAPL entry"
        )
        
        results = self.alert_system.send_alert(entry_alert)
        self.assertTrue(results.get(AlertChannel.DESKTOP, False))
        
        # Verify workflow state
        self.assertEqual(len(self.alert_system.alert_history), 2)
        checklist = self.checklist_manager.get_checklist(checklist_id)
        self.assertGreater(checklist.completion_percentage, 0)


def run_alert_system_tests():
    """Run all alert system tests"""
    print("=" * 60)
    print("ALERT SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAlertDataClasses,
        TestAlertHandlers,
        TestTradingAlertSystem,
        TestExecutionChecklistManager,
        TestAlertUtilities,
        TestAlertIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ALERT SYSTEM TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    
    return result


if __name__ == "__main__":
    run_alert_system_tests()