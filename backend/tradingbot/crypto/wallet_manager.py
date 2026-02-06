"""
Encrypted Wallet Manager.

Manages cryptocurrency wallets with encrypted private key storage.
Reuses Fernet encryption pattern from credential_encryption.py.
"""

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except ImportError:
    HAS_FERNET = False

try:
    from web3 import Web3
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False


@dataclass
class WalletInfo:
    """Wallet information (public data only)."""
    address: str
    chain: str
    label: str = ""
    created_at: str = ""
    is_default: bool = False


class WalletManager:
    """
    Manages encrypted wallet storage and retrieval.

    Private keys are encrypted at rest using Fernet symmetric encryption.
    The encryption key should be derived from a user-specific secret.
    """

    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize wallet manager.

        Args:
            encryption_key: Fernet-compatible encryption key (32 bytes, base64-encoded).
                          If not provided, a new key is generated.
        """
        if not HAS_FERNET:
            logger.warning("cryptography not available. Wallet encryption disabled.")
            self._fernet = None
        else:
            if encryption_key is None:
                encryption_key = Fernet.generate_key()
            self._fernet = Fernet(encryption_key)

        self._wallets: Dict[str, Dict[str, Any]] = {}  # address -> wallet data

    def create_wallet(self, label: str = "", chain: str = "ethereum") -> Optional[WalletInfo]:
        """
        Create a new wallet with a random private key.

        Returns:
            WalletInfo with the new wallet address, or None if web3 unavailable.
        """
        if not HAS_WEB3:
            logger.error("web3 required to create wallets")
            return None

        account = Web3().eth.account.create()
        address = account.address
        private_key = account.key.hex()

        # Encrypt private key
        encrypted_key = self._encrypt(private_key)

        from datetime import datetime
        wallet_data = {
            'address': address,
            'encrypted_key': encrypted_key,
            'chain': chain,
            'label': label or f"Wallet {len(self._wallets) + 1}",
            'created_at': datetime.now().isoformat(),
            'is_default': len(self._wallets) == 0,
        }

        self._wallets[address.lower()] = wallet_data
        logger.info(f"Created wallet: {address} ({label})")

        return WalletInfo(
            address=address,
            chain=chain,
            label=wallet_data['label'],
            created_at=wallet_data['created_at'],
            is_default=wallet_data['is_default'],
        )

    def import_wallet(
        self,
        private_key: str,
        label: str = "",
        chain: str = "ethereum",
    ) -> Optional[WalletInfo]:
        """
        Import an existing wallet from private key.

        Args:
            private_key: Hex-encoded private key (with or without 0x prefix)
            label: Display label for the wallet
            chain: Blockchain network

        Returns:
            WalletInfo or None if import fails
        """
        if not HAS_WEB3:
            logger.error("web3 required to import wallets")
            return None

        try:
            # Normalize private key
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key

            account = Web3().eth.account.from_key(private_key)
            address = account.address

            # Encrypt private key
            encrypted_key = self._encrypt(private_key)

            from datetime import datetime
            wallet_data = {
                'address': address,
                'encrypted_key': encrypted_key,
                'chain': chain,
                'label': label or f"Imported Wallet {len(self._wallets) + 1}",
                'created_at': datetime.now().isoformat(),
                'is_default': len(self._wallets) == 0,
            }

            self._wallets[address.lower()] = wallet_data
            logger.info(f"Imported wallet: {address}")

            return WalletInfo(
                address=address,
                chain=chain,
                label=wallet_data['label'],
                created_at=wallet_data['created_at'],
                is_default=wallet_data['is_default'],
            )
        except Exception as e:
            logger.error(f"Failed to import wallet: {e}")
            return None

    def get_private_key(self, address: str) -> Optional[str]:
        """
        Retrieve and decrypt private key for a wallet.

        Args:
            address: Wallet address

        Returns:
            Decrypted private key string, or None if not found
        """
        wallet_data = self._wallets.get(address.lower())
        if not wallet_data:
            return None

        encrypted_key = wallet_data.get('encrypted_key')
        if not encrypted_key:
            return None

        return self._decrypt(encrypted_key)

    def list_wallets(self) -> List[WalletInfo]:
        """List all wallets (public info only)."""
        return [
            WalletInfo(
                address=data['address'],
                chain=data['chain'],
                label=data['label'],
                created_at=data['created_at'],
                is_default=data['is_default'],
            )
            for data in self._wallets.values()
        ]

    def get_default_wallet(self) -> Optional[WalletInfo]:
        """Get the default wallet."""
        for data in self._wallets.values():
            if data['is_default']:
                return WalletInfo(
                    address=data['address'],
                    chain=data['chain'],
                    label=data['label'],
                    created_at=data['created_at'],
                    is_default=True,
                )
        return None

    def set_default_wallet(self, address: str) -> bool:
        """Set a wallet as default."""
        address_lower = address.lower()
        if address_lower not in self._wallets:
            return False

        for addr, data in self._wallets.items():
            data['is_default'] = (addr == address_lower)
        return True

    def remove_wallet(self, address: str) -> bool:
        """Remove a wallet."""
        address_lower = address.lower()
        if address_lower in self._wallets:
            del self._wallets[address_lower]
            logger.info(f"Removed wallet: {address}")
            return True
        return False

    def _encrypt(self, data: str) -> str:
        """Encrypt a string using Fernet."""
        if self._fernet:
            return self._fernet.encrypt(data.encode()).decode()
        # Fallback: store as-is (not recommended for production)
        logger.warning("Storing private key without encryption (cryptography not installed)")
        return data

    def _decrypt(self, encrypted_data: str) -> str:
        """Decrypt a string using Fernet."""
        if self._fernet:
            return self._fernet.decrypt(encrypted_data.encode()).decode()
        return encrypted_data
