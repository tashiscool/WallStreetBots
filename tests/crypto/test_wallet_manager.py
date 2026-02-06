"""Tests for wallet manager."""

import pytest
from unittest.mock import MagicMock, patch

from backend.tradingbot.crypto.wallet_manager import (
    WalletManager,
    WalletInfo,
    HAS_FERNET,
    HAS_WEB3,
)


class TestWalletManager:
    """Test WalletManager."""

    def test_init(self):
        """Should initialize without errors."""
        manager = WalletManager()
        assert manager.list_wallets() == []

    @pytest.mark.skipif(not HAS_WEB3, reason="web3 not installed")
    def test_create_wallet(self):
        """Should create wallet with valid address."""
        manager = WalletManager()
        info = manager.create_wallet(label="Test", chain="ethereum")
        assert info is not None
        assert info.address.startswith("0x")
        assert len(info.address) == 42
        assert info.label == "Test"
        assert info.is_default is True

    @pytest.mark.skipif(not HAS_WEB3, reason="web3 not installed")
    def test_create_multiple_wallets(self):
        """First wallet should be default."""
        manager = WalletManager()
        w1 = manager.create_wallet(label="First")
        w2 = manager.create_wallet(label="Second")
        assert w1.is_default is True
        assert w2.is_default is False
        assert len(manager.list_wallets()) == 2

    @pytest.mark.skipif(not HAS_WEB3, reason="web3 not installed")
    def test_import_wallet(self):
        """Should import wallet from private key."""
        manager = WalletManager()
        # Generate a test key
        from web3 import Web3
        account = Web3().eth.account.create()

        info = manager.import_wallet(
            private_key=account.key.hex(),
            label="Imported",
        )
        assert info is not None
        assert info.address == account.address

    @pytest.mark.skipif(not HAS_WEB3, reason="web3 not installed")
    def test_get_private_key(self):
        """Should retrieve and decrypt private key."""
        manager = WalletManager()
        info = manager.create_wallet(label="Test")
        key = manager.get_private_key(info.address)
        assert key is not None
        assert key.startswith("0x")

    @pytest.mark.skipif(not HAS_WEB3, reason="web3 not installed")
    def test_encryption_roundtrip(self):
        """Encrypt then decrypt should return original."""
        manager = WalletManager()
        original = "0x" + "a" * 64
        encrypted = manager._encrypt(original)
        decrypted = manager._decrypt(encrypted)
        assert decrypted == original
        if HAS_FERNET:
            assert encrypted != original  # Should actually be encrypted

    @pytest.mark.skipif(not HAS_WEB3, reason="web3 not installed")
    def test_remove_wallet(self):
        """Should remove wallet."""
        manager = WalletManager()
        info = manager.create_wallet(label="ToRemove")
        assert len(manager.list_wallets()) == 1
        assert manager.remove_wallet(info.address) is True
        assert len(manager.list_wallets()) == 0

    @pytest.mark.skipif(not HAS_WEB3, reason="web3 not installed")
    def test_set_default_wallet(self):
        """Should change default wallet."""
        manager = WalletManager()
        w1 = manager.create_wallet(label="First")
        w2 = manager.create_wallet(label="Second")

        assert manager.set_default_wallet(w2.address) is True
        default = manager.get_default_wallet()
        assert default.address == w2.address

    def test_remove_nonexistent(self):
        """Should return False for nonexistent wallet."""
        manager = WalletManager()
        assert manager.remove_wallet("0x" + "0" * 40) is False

    def test_get_private_key_nonexistent(self):
        """Should return None for nonexistent wallet."""
        manager = WalletManager()
        assert manager.get_private_key("0x" + "0" * 40) is None
