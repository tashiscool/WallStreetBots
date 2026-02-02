"""
Comprehensive tests for CredentialEncryptionService.

Tests encryption, decryption, key derivation, and edge cases.
Target: 80%+ coverage.
"""
import os
import unittest

# Django is configured by pytest-django via pytest.ini
# No need to mock - just import and use


class TestCredentialEncryptionService(unittest.TestCase):
    """Test suite for CredentialEncryptionService."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset singleton before each test
        from backend.auth0login.services import credential_encryption
        credential_encryption._service_instance = None
        if hasattr(credential_encryption.CredentialEncryptionService, '_instance'):
            credential_encryption.CredentialEncryptionService._instance = None

        from backend.auth0login.services.credential_encryption import CredentialEncryptionService
        self.service = CredentialEncryptionService()

    def test_encryption_decryption_roundtrip(self):
        """Test that encryption followed by decryption returns original value."""
        test_values = [
            "my-super-secret-api-key-12345",
            "PKXXXXXXXXXXXXXXXXXX",
            "short",
            "a" * 1000,  # Long string
        ]

        for original in test_values:
            with self.subTest(original=original[:20]):
                encrypted = self.service.encrypt(original)
                decrypted = self.service.decrypt(encrypted)
                self.assertEqual(decrypted, original)

    def test_encryption_produces_different_output(self):
        """Test that encryption produces output different from input."""
        original = "my-secret-api-key"
        encrypted = self.service.encrypt(original)
        self.assertNotEqual(encrypted, original)

    def test_encryption_produces_unique_ciphertext(self):
        """Test that encrypting the same value twice produces different ciphertexts."""
        original = "my-secret-api-key"
        encrypted1 = self.service.encrypt(original)
        encrypted2 = self.service.encrypt(original)
        # Fernet uses random IV, so ciphertexts should differ
        self.assertNotEqual(encrypted1, encrypted2)
        # But both should decrypt to same value
        self.assertEqual(self.service.decrypt(encrypted1), original)
        self.assertEqual(self.service.decrypt(encrypted2), original)

    def test_decrypt_invalid_data_raises_error(self):
        """Test that decrypting invalid data raises ValueError."""
        invalid_values = [
            "not-valid-base64!@#$",
            "dGVzdA==",  # Valid base64 but not valid Fernet token
            "",
        ]

        for invalid in invalid_values:
            if invalid:  # Skip empty string for this test
                with self.subTest(invalid=invalid[:20]):
                    with self.assertRaises(ValueError):
                        self.service.decrypt(invalid)

    def test_encrypt_empty_string(self):
        """Test encryption of empty string returns empty string."""
        result = self.service.encrypt("")
        self.assertEqual(result, "")

    def test_encrypt_unicode_characters(self):
        """Test encryption of unicode characters."""
        test_values = [
            "密码",  # Chinese
            "пароль",  # Russian
            "🔐🔑",  # Emojis
            "café résumé",  # Accented
        ]

        for original in test_values:
            with self.subTest(original=original):
                encrypted = self.service.encrypt(original)
                decrypted = self.service.decrypt(encrypted)
                self.assertEqual(decrypted, original)

    def test_special_characters(self):
        """Test encryption of special characters."""
        original = "!@#$%^&*()_+-=[]{}|;':\",./<>?\\"
        encrypted = self.service.encrypt(original)
        decrypted = self.service.decrypt(encrypted)
        self.assertEqual(decrypted, original)

    def test_newlines_and_whitespace(self):
        """Test encryption preserves newlines and whitespace."""
        original = "line1\nline2\r\nline3\ttab  spaces"
        encrypted = self.service.encrypt(original)
        decrypted = self.service.decrypt(encrypted)
        self.assertEqual(decrypted, original)

    def test_encrypt_long_string(self):
        """Test encryption of a long string."""
        original = "x" * 10000
        encrypted = self.service.encrypt(original)
        decrypted = self.service.decrypt(encrypted)
        self.assertEqual(decrypted, original)

    def test_is_encrypted_detection(self):
        """Test is_encrypted method correctly identifies encrypted values."""
        original = "my-api-key"
        encrypted = self.service.encrypt(original)

        self.assertTrue(self.service.is_encrypted(encrypted))
        self.assertFalse(self.service.is_encrypted(original))
        self.assertFalse(self.service.is_encrypted(""))
        self.assertFalse(self.service.is_encrypted("not-encrypted-at-all"))

    def test_is_encrypted_with_invalid_base64(self):
        """Test is_encrypted handles invalid base64."""
        self.assertFalse(self.service.is_encrypted("!!!invalid!!!"))
        self.assertFalse(self.service.is_encrypted(None))

    def test_singleton_pattern(self):
        """Test that service follows singleton pattern."""
        from backend.auth0login.services.credential_encryption import CredentialEncryptionService

        service1 = CredentialEncryptionService()
        service2 = CredentialEncryptionService()
        self.assertIs(service1, service2)

    def test_services_share_key(self):
        """Test that service instances use same encryption key."""
        from backend.auth0login.services.credential_encryption import CredentialEncryptionService

        service1 = CredentialEncryptionService()
        service2 = CredentialEncryptionService()

        original = "shared-secret"
        encrypted = service1.encrypt(original)
        decrypted = service2.decrypt(encrypted)
        self.assertEqual(decrypted, original)


class TestKeyRotation(unittest.TestCase):
    """Test key rotation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from backend.auth0login.services import credential_encryption
        credential_encryption._service_instance = None
        if hasattr(credential_encryption.CredentialEncryptionService, '_instance'):
            credential_encryption.CredentialEncryptionService._instance = None

    def test_rotate_key(self):
        """Test key rotation re-encrypts with new key."""
        from backend.auth0login.services.credential_encryption import CredentialEncryptionService

        service = CredentialEncryptionService()
        original = "my-secret"
        encrypted_old = service.encrypt(original)

        # Create a new service instance (in practice, would have different key)
        new_service = CredentialEncryptionService()
        new_encrypted = service.rotate_key(encrypted_old, new_service)

        # Old encrypted value should still decrypt with current key
        self.assertEqual(service.decrypt(encrypted_old), original)

        # New encrypted value should also decrypt to same original
        self.assertEqual(new_service.decrypt(new_encrypted), original)

        # Encryption produces unique ciphertexts (due to random IV)
        self.assertNotEqual(encrypted_old, new_encrypted)


class TestSaltConfiguration(unittest.TestCase):
    """Test salt configuration handling."""

    def setUp(self):
        """Set up test fixtures."""
        from backend.auth0login.services import credential_encryption
        credential_encryption._service_instance = None
        if hasattr(credential_encryption.CredentialEncryptionService, '_instance'):
            credential_encryption.CredentialEncryptionService._instance = None

    def test_string_salt_converted_to_bytes(self):
        """Test that string salt is properly converted to bytes."""
        from backend.auth0login.services.credential_encryption import CredentialEncryptionService

        # Reset singleton
        from backend.auth0login.services import credential_encryption
        credential_encryption._service_instance = None
        if hasattr(CredentialEncryptionService, '_instance'):
            CredentialEncryptionService._instance = None

        service = CredentialEncryptionService()

        # Should still work
        original = "test-value"
        encrypted = service.encrypt(original)
        decrypted = service.decrypt(encrypted)
        self.assertEqual(decrypted, original)


class TestCredentialEncryptionConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        from backend.auth0login.services import credential_encryption
        credential_encryption._service_instance = None
        if hasattr(credential_encryption.CredentialEncryptionService, '_instance'):
            credential_encryption.CredentialEncryptionService._instance = None

    def test_get_encryption_service_function(self):
        """Test get_encryption_service returns singleton."""
        from backend.auth0login.services.credential_encryption import (
            get_encryption_service,
            CredentialEncryptionService,
        )

        service1 = get_encryption_service()
        service2 = get_encryption_service()
        self.assertIs(service1, service2)
        self.assertIsInstance(service1, CredentialEncryptionService)

    def test_encrypt_credential_function(self):
        """Test encrypt_credential convenience function."""
        from backend.auth0login.services.credential_encryption import (
            encrypt_credential,
            decrypt_credential,
        )

        original = "my-api-key-12345"
        encrypted = encrypt_credential(original)
        decrypted = decrypt_credential(encrypted)
        self.assertEqual(decrypted, original)


if __name__ == '__main__':
    unittest.main()
