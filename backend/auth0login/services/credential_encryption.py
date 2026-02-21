"""Credential encryption service using Fernet symmetric encryption.

This service provides secure encryption and decryption for sensitive credentials
like API keys. It uses the cryptography library's Fernet implementation which
provides authenticated encryption (AES-128 in CBC mode with HMAC).

The encryption key is derived from Django's SECRET_KEY using PBKDF2.
"""

import base64
import logging
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from django.conf import settings

logger = logging.getLogger(__name__)


class CredentialEncryptionService:
    """Handles encryption/decryption of sensitive credentials.

    This is a singleton service that maintains a single Fernet instance
    derived from Django's SECRET_KEY. All credential encryption in the
    application should use this service.

    Usage:
        service = get_encryption_service()
        encrypted = service.encrypt("my-api-key")
        decrypted = service.decrypt(encrypted)
    """

    _instance: Optional['CredentialEncryptionService'] = None
    _fernet: Optional[Fernet] = None

    def __new__(cls):
        """Singleton pattern for encryption service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_key()
        return cls._instance

    def _initialize_key(self):
        """Initialize encryption key from Django settings.

        Uses PBKDF2 with SHA-256 to derive a 32-byte key from SECRET_KEY.
        The salt can be customized via CREDENTIAL_ENCRYPTION_SALT setting.
        """
        # Use a dedicated salt for credential encryption
        # This can be customized in settings for additional security
        salt = getattr(
            settings,
            'CREDENTIAL_ENCRYPTION_SALT',
            b'wallstreetbots_credential_encryption_salt_v1'
        )

        # Ensure salt is bytes
        if isinstance(salt, str):
            salt = salt.encode('utf-8')

        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # High iteration count for security
        )

        # Get the secret key
        secret_key = settings.SECRET_KEY
        if isinstance(secret_key, str):
            secret_key = secret_key.encode('utf-8')

        # Derive the encryption key
        key = kdf.derive(secret_key)
        self._fernet = Fernet(base64.urlsafe_b64encode(key))

        logger.info("Credential encryption service initialized")

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext value.

        Args:
            plaintext: The value to encrypt (e.g., an API key)

        Returns:
            Base64-encoded encrypted value that can be safely stored

        Raises:
            ValueError: If encryption fails
        """
        if not plaintext:
            return ''

        try:
            # Encode plaintext to bytes and encrypt
            encrypted_bytes = self._fernet.encrypt(plaintext.encode('utf-8'))
            # Return as base64 string for safe storage
            return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError("Failed to encrypt credential") from e

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt an encrypted value.

        Args:
            ciphertext: Base64-encoded encrypted value

        Returns:
            Decrypted plaintext

        Raises:
            ValueError: If decryption fails (invalid key or corrupted data)
        """
        if not ciphertext:
            return ''

        try:
            # Decode from base64 and decrypt
            encrypted_bytes = base64.urlsafe_b64decode(ciphertext.encode('utf-8'))
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except InvalidToken as e:
            logger.error("Decryption failed: Invalid token (key mismatch or corruption)")
            raise ValueError(
                "Failed to decrypt credential - invalid key or corrupted data. "
                "This may happen if SECRET_KEY has changed."
            ) from e
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt credential") from e

    def is_encrypted(self, value: str) -> bool:
        """Check if a value appears to be encrypted.

        This is a heuristic check - encrypted values are longer and have
        specific patterns. Used for migration from plaintext to encrypted.

        Args:
            value: The value to check

        Returns:
            True if the value appears to be encrypted
        """
        if not value or len(value) < 100:
            # Encrypted values are longer due to base64 encoding and overhead
            return False

        try:
            # Try to decode as base64
            decoded = base64.urlsafe_b64decode(value.encode('utf-8'))
            # Fernet tokens have a specific structure
            # Version byte + timestamp (8 bytes) + IV (16 bytes) + ciphertext + HMAC
            return len(decoded) >= 57  # Minimum Fernet token length
        except Exception:
            return False

    def rotate_key(self, old_ciphertext: str, new_service: 'CredentialEncryptionService') -> str:
        """Re-encrypt a value with a new key.

        Used when rotating encryption keys. Decrypts with current key
        and re-encrypts with a new service instance.

        Args:
            old_ciphertext: Value encrypted with current key
            new_service: New encryption service with rotated key

        Returns:
            Value encrypted with new key
        """
        plaintext = self.decrypt(old_ciphertext)
        return new_service.encrypt(plaintext)


# Singleton accessor
_service_instance: Optional[CredentialEncryptionService] = None


def get_encryption_service() -> CredentialEncryptionService:
    """Get the credential encryption service instance.

    This function provides access to the singleton encryption service.
    The service is lazily initialized on first call.

    Returns:
        The credential encryption service instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = CredentialEncryptionService()
    return _service_instance


def encrypt_credential(value: str) -> str:
    """Convenience function to encrypt a credential.

    Args:
        value: The plaintext credential to encrypt

    Returns:
        Encrypted credential
    """
    return get_encryption_service().encrypt(value)


def decrypt_credential(value: str) -> str:
    """Convenience function to decrypt a credential.

    Args:
        value: The encrypted credential

    Returns:
        Decrypted plaintext credential
    """
    return get_encryption_service().decrypt(value)
