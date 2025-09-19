"""
Secrets Management & IAM Hardening for Trading Systems
Implements production-grade secrets management and identity/access controls.
"""

import os
import hashlib
import hmac
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from pathlib import Path
import boto3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt


class SecretType(Enum):
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"  # noqa: S105
    BROKER_CREDENTIALS = "broker_credentials"
    ENCRYPTION_KEY = "encryption_key"
    WEBHOOK_SECRET = "webhook_secret"  # noqa: S105
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"


class AccessLevel(Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SERVICE = "service"


@dataclass
class Secret:
    """Represents a managed secret."""
    name: str
    secret_type: SecretType
    value: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    rotation_required: bool = False


@dataclass
class AccessPolicy:
    """Defines access control policy."""
    name: str
    description: str
    allowed_secrets: Set[str] = field(default_factory=set)
    allowed_operations: Set[str] = field(default_factory=set)
    conditions: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None


@dataclass
class Principal:
    """Represents an identity (user, service, etc.)."""
    id: str
    name: str
    principal_type: str  # user, service, system
    access_level: AccessLevel
    policies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    mfa_enabled: bool = False
    api_key_hash: Optional[str] = None


class SecretEncryption:
    """Handles encryption/decryption of secrets."""

    def __init__(self, master_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        if master_key:
            self.key = master_key.encode()
        else:
            self.key = self._generate_key_from_environment()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.key[:32]))

    def _generate_key_from_environment(self) -> bytes:
        """Generate encryption key from environment variables."""
        # Use multiple environment variables for key derivation
        sources = [
            os.environ.get('WALLSTREETBOTS_MASTER_KEY', ''),
            os.environ.get('HOSTNAME', ''),
            os.environ.get('USER', ''),
            'wallstreetbots_default_salt'
        ]

        combined = ''.join(sources).encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'wallstreetbots_salt_2024',
            iterations=100000,
        )
        return kdf.derive(combined)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext value."""
        try:
            encrypted = self.fernet.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext value."""
        try:
            encrypted_data = base64.urlsafe_b64decode(ciphertext.encode())
            decrypted = self.fernet.decrypt(encrypted_data)
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise


class SecretsManager:
    """Manages secrets with encryption and access control."""

    def __init__(self, storage_path: str | None = None):
        self.logger = logging.getLogger(__name__)
        self.encryption = SecretEncryption()
        self.storage_path = Path(storage_path or os.environ.get('SECRETS_STORAGE_PATH', '/tmp/wallstreetbots_secrets'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.secrets: Dict[str, Secret] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self._load_secrets()

    def _load_secrets(self):
        """Load encrypted secrets from storage."""
        secrets_file = self.storage_path / 'secrets.enc'
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    encrypted_data = f.read()

                decrypted_data = self.encryption.decrypt(encrypted_data)
                secrets_dict = json.loads(decrypted_data)

                for name, secret_data in secrets_dict.items():
                    secret = Secret(
                        name=secret_data['name'],
                        secret_type=SecretType(secret_data['secret_type']),
                        value=secret_data['value'],
                        description=secret_data.get('description', ''),
                        created_at=datetime.fromisoformat(secret_data['created_at']),
                        expires_at=datetime.fromisoformat(secret_data['expires_at']) if secret_data.get('expires_at') else None,
                        last_accessed=datetime.fromisoformat(secret_data['last_accessed']) if secret_data.get('last_accessed') else None,
                        access_count=secret_data.get('access_count', 0),
                        tags=secret_data.get('tags', {}),
                        rotation_required=secret_data.get('rotation_required', False)
                    )
                    self.secrets[name] = secret

                self.logger.info(f"Loaded {len(self.secrets)} secrets from storage")

            except Exception as e:
                self.logger.error(f"Failed to load secrets: {e}")

    def _save_secrets(self):
        """Save encrypted secrets to storage."""
        try:
            secrets_dict = {}
            for name, secret in self.secrets.items():
                secrets_dict[name] = {
                    'name': secret.name,
                    'secret_type': secret.secret_type.value,
                    'value': secret.value,
                    'description': secret.description,
                    'created_at': secret.created_at.isoformat(),
                    'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
                    'last_accessed': secret.last_accessed.isoformat() if secret.last_accessed else None,
                    'access_count': secret.access_count,
                    'tags': secret.tags,
                    'rotation_required': secret.rotation_required
                }

            plaintext_data = json.dumps(secrets_dict, indent=2)
            encrypted_data = self.encryption.encrypt(plaintext_data)

            secrets_file = self.storage_path / 'secrets.enc'
            with open(secrets_file, 'w') as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            os.chmod(secrets_file, 0o600)

            self.logger.info(f"Saved {len(self.secrets)} secrets to storage")

        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
            raise

    def store_secret(self, name: str, value: str, secret_type: SecretType,
                    description: str = "", expires_in_days: Optional[int] = None,
                    tags: Optional[Dict[str, str]] = None) -> bool:
        """Store a new secret."""
        try:
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)

            secret = Secret(
                name=name,
                secret_type=secret_type,
                value=value,
                description=description,
                expires_at=expires_at,
                tags=tags or {}
            )

            self.secrets[name] = secret
            self._save_secrets()

            self._audit_log('STORE_SECRET', {'secret_name': name, 'secret_type': secret_type.value})
            self.logger.info(f"Stored secret: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store secret {name}: {e}")
            return False

    def get_secret(self, name: str, principal_id: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value."""
        try:
            if name not in self.secrets:
                self._audit_log('GET_SECRET_FAILED', {'secret_name': name, 'reason': 'not_found', 'principal_id': principal_id})
                return None

            secret = self.secrets[name]

            # Check expiration
            if secret.expires_at and datetime.now() > secret.expires_at:
                self._audit_log('GET_SECRET_FAILED', {'secret_name': name, 'reason': 'expired', 'principal_id': principal_id})
                self.logger.warning(f"Secret {name} has expired")
                return None

            # Update access tracking
            secret.last_accessed = datetime.now()
            secret.access_count += 1
            self._save_secrets()

            self._audit_log('GET_SECRET', {'secret_name': name, 'principal_id': principal_id})
            return secret.value

        except Exception as e:
            self.logger.error(f"Failed to get secret {name}: {e}")
            self._audit_log('GET_SECRET_ERROR', {'secret_name': name, 'error': str(e), 'principal_id': principal_id})
            return None

    def rotate_secret(self, name: str, new_value: str) -> bool:
        """Rotate a secret value."""
        try:
            if name not in self.secrets:
                return False

            secret = self.secrets[name]
            old_value_hash = hashlib.sha256(secret.value.encode()).hexdigest()[:8]

            secret.value = new_value
            secret.rotation_required = False
            secret.created_at = datetime.now()  # Reset creation time

            self._save_secrets()

            self._audit_log('ROTATE_SECRET', {
                'secret_name': name,
                'old_value_hash': old_value_hash
            })

            self.logger.info(f"Rotated secret: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to rotate secret {name}: {e}")
            return False

    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        try:
            if name in self.secrets:
                del self.secrets[name]
                self._save_secrets()

                self._audit_log('DELETE_SECRET', {'secret_name': name})
                self.logger.info(f"Deleted secret: {name}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to delete secret {name}: {e}")
            return False

    def list_secrets(self) -> List[Dict[str, Any]]:
        """List all secrets (metadata only)."""
        result = []
        for name, secret in self.secrets.items():
            result.append({
                'name': name,
                'secret_type': secret.secret_type.value,
                'description': secret.description,
                'created_at': secret.created_at.isoformat(),
                'expires_at': secret.expires_at.isoformat() if secret.expires_at else None,
                'last_accessed': secret.last_accessed.isoformat() if secret.last_accessed else None,
                'access_count': secret.access_count,
                'rotation_required': secret.rotation_required,
                'tags': secret.tags
            })
        return result

    def check_expiring_secrets(self, days_ahead: int = 30) -> List[str]:
        """Check for secrets expiring within specified days."""
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        expiring = []

        for name, secret in self.secrets.items():
            if secret.expires_at and secret.expires_at <= cutoff_date:
                expiring.append(name)

        return expiring

    def mark_for_rotation(self, name: str) -> bool:
        """Mark a secret for rotation."""
        if name in self.secrets:
            self.secrets[name].rotation_required = True
            self._save_secrets()
            self._audit_log('MARK_FOR_ROTATION', {'secret_name': name})
            return True
        return False

    def _audit_log(self, action: str, details: Dict[str, Any]):
        """Log audit events."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        self.audit_log.append(log_entry)

        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]


class IAMManager:
    """Identity and Access Management for trading systems."""

    def __init__(self, secrets_manager: SecretsManager):
        self.secrets_manager = secrets_manager
        self.logger = logging.getLogger(__name__)
        self.principals: Dict[str, Principal] = {}
        self.policies: Dict[str, AccessPolicy] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._setup_default_policies()

    def _setup_default_policies(self):
        """Setup default access policies."""

        # Trading system service policy
        self.policies['trading_service'] = AccessPolicy(
            name='trading_service',
            description='Full access for trading service',
            allowed_secrets={'broker_api_key', 'database_password', 'risk_engine_key'},
            allowed_operations={'read', 'rotate'},
            conditions={
                'source_ip_ranges': ['10.0.0.0/8', '172.16.0.0/12'],
                'time_restrictions': None
            }
        )

        # Read-only policy for monitoring
        self.policies['monitoring_readonly'] = AccessPolicy(
            name='monitoring_readonly',
            description='Read-only access for monitoring systems',
            allowed_secrets={'monitoring_api_key'},
            allowed_operations={'read'},
            conditions={
                'source_ip_ranges': ['10.0.0.0/8'],
                'rate_limit_per_hour': 100
            }
        )

        # Admin policy
        self.policies['admin'] = AccessPolicy(
            name='admin',
            description='Full administrative access',
            allowed_secrets=set(),  # Empty means all secrets
            allowed_operations={'read', 'write', 'delete', 'rotate'},
            conditions={
                'mfa_required': True,
                'session_timeout_minutes': 30
            }
        )

    def create_principal(self, principal_id: str, name: str, principal_type: str,
                        access_level: AccessLevel, policies: List[str] | None = None) -> bool:
        """Create a new principal (user/service)."""
        try:
            # Validate policies exist
            if policies:
                for policy_name in policies:
                    if policy_name not in self.policies:
                        raise ValueError(f"Policy not found: {policy_name}")

            principal = Principal(
                id=principal_id,
                name=name,
                principal_type=principal_type,
                access_level=access_level,
                policies=policies or []
            )

            self.principals[principal_id] = principal
            self.logger.info(f"Created principal: {principal_id} ({name})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create principal {principal_id}: {e}")
            return False

    def generate_api_key(self, principal_id: str) -> Optional[str]:
        """Generate API key for a principal."""
        if principal_id not in self.principals:
            return None

        # Generate secure API key
        api_key = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip('=')
        api_key = f"wsb_{principal_id}_{api_key}"

        # Store hash of API key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.principals[principal_id].api_key_hash = key_hash

        self.logger.info(f"Generated API key for principal: {principal_id}")
        return api_key

    def validate_api_key(self, api_key: str) -> Optional[Principal]:
        """Validate API key and return principal."""
        try:
            if not api_key.startswith('wsb_'):
                return None

            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            for principal in self.principals.values():
                if principal.api_key_hash == key_hash:
                    principal.last_login = datetime.now()
                    return principal

            return None

        except Exception as e:
            self.logger.error(f"API key validation error: {e}")
            return None

    def check_access(self, principal_id: str, secret_name: str, operation: str,
                    context: Dict[str, Any] | None = None) -> Tuple[bool, str]:
        """Check if principal has access to perform operation on secret."""
        try:
            if principal_id not in self.principals:
                return False, "Principal not found"

            principal = self.principals[principal_id]
            context = context or {}

            # Check each policy attached to principal
            for policy_name in principal.policies:
                if policy_name not in self.policies:
                    continue

                policy = self.policies[policy_name]

                # Check if policy allows this secret (empty set means all secrets)
                if policy.allowed_secrets and secret_name not in policy.allowed_secrets:
                    continue

                # Check if policy allows this operation
                if operation not in policy.allowed_operations:
                    continue

                # Check policy conditions
                access_allowed, reason = self._check_policy_conditions(policy, context)
                if access_allowed:
                    return True, "Access granted"

            return False, "No policy grants access"

        except Exception as e:
            self.logger.error(f"Access check error: {e}")
            return False, f"Error: {e}"

    def _check_policy_conditions(self, policy: AccessPolicy, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if policy conditions are met."""
        conditions = policy.conditions

        # Check IP restrictions
        if 'source_ip_ranges' in conditions and 'source_ip' in context:
            source_ip = context['source_ip']
            allowed_ranges = conditions['source_ip_ranges']
            # Simplified IP range check (would use ipaddress module in production)
            ip_allowed = any(source_ip.startswith(range_prefix.split('/')[0][:3])
                           for range_prefix in allowed_ranges)
            if not ip_allowed:
                return False, "IP address not in allowed ranges"

        # Check MFA requirement
        if conditions.get('mfa_required', False) and not context.get('mfa_verified', False):
            return False, "MFA required but not verified"

        # Check time restrictions
        if conditions.get('time_restrictions'):
            current_hour = datetime.now().hour
            allowed_hours = conditions['time_restrictions']
            if current_hour not in allowed_hours:
                return False, "Access not allowed at this time"

        # Check rate limiting
        if 'rate_limit_per_hour' in conditions:
            # Would implement rate limiting logic here
            pass

        return True, "All conditions met"

    def create_session(self, principal_id: str, context: Dict[str, Any] | None = None) -> Optional[str]:
        """Create authenticated session for principal."""
        if principal_id not in self.principals:
            return None

        session_id = base64.urlsafe_b64encode(os.urandom(24)).decode().rstrip('=')

        session_data = {
            'principal_id': principal_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'context': context or {}
        }

        self.active_sessions[session_id] = session_data
        self.logger.info(f"Created session for principal: {principal_id}")
        return session_id

    def validate_session(self, session_id: str) -> Optional[Principal]:
        """Validate session and return principal."""
        if session_id not in self.active_sessions:
            return None

        session_data = self.active_sessions[session_id]
        principal_id = session_data['principal_id']

        if principal_id not in self.principals:
            del self.active_sessions[session_id]
            return None

        # Check session timeout
        session_timeout = timedelta(minutes=30)  # Default timeout
        if datetime.now() - session_data['last_activity'] > session_timeout:
            del self.active_sessions[session_id]
            return None

        # Update last activity
        session_data['last_activity'] = datetime.now()

        return self.principals[principal_id]

    def revoke_session(self, session_id: str) -> bool:
        """Revoke an active session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Revoked session: {session_id}")
            return True
        return False

    def get_principal_permissions(self, principal_id: str) -> Dict[str, Any]:
        """Get detailed permissions for a principal."""
        if principal_id not in self.principals:
            return {}

        principal = self.principals[principal_id]
        permissions = {
            'principal_id': principal_id,
            'name': principal.name,
            'access_level': principal.access_level.value,
            'policies': [],
            'effective_permissions': {
                'secrets': set(),
                'operations': set()
            }
        }

        for policy_name in principal.policies:
            if policy_name in self.policies:
                policy = self.policies[policy_name]
                permissions['policies'].append({
                    'name': policy.name,
                    'description': policy.description,
                    'allowed_secrets': list(policy.allowed_secrets),
                    'allowed_operations': list(policy.allowed_operations),
                    'conditions': policy.conditions
                })

                # Aggregate effective permissions
                if not policy.allowed_secrets:  # Empty means all secrets
                    permissions['effective_permissions']['secrets'] = {'*'}
                else:
                    permissions['effective_permissions']['secrets'].update(policy.allowed_secrets)

                permissions['effective_permissions']['operations'].update(policy.allowed_operations)

        # Convert sets to lists for JSON serialization
        permissions['effective_permissions']['secrets'] = list(permissions['effective_permissions']['secrets'])
        permissions['effective_permissions']['operations'] = list(permissions['effective_permissions']['operations'])

        return permissions


class TradingSecretsHardening:
    """Complete secrets and IAM hardening for trading systems."""

    def __init__(self):
        self.secrets_manager = SecretsManager()
        self.iam_manager = IAMManager(self.secrets_manager)
        self.logger = logging.getLogger(__name__)
        self._setup_trading_secrets()
        self._setup_trading_principals()

    def _setup_trading_secrets(self):
        """Setup standard trading system secrets."""

        # Only setup if not already exist
        existing_secrets = {s['name'] for s in self.secrets_manager.list_secrets()}

        if 'broker_api_key' not in existing_secrets:
            self.secrets_manager.store_secret(
                'broker_api_key',
                'demo_broker_key_' + base64.urlsafe_b64encode(os.urandom(16)).decode(),
                SecretType.BROKER_CREDENTIALS,
                'Broker API access key',
                expires_in_days=90,
                tags={'environment': 'production', 'rotation_frequency': 'quarterly'}
            )

        if 'database_password' not in existing_secrets:
            self.secrets_manager.store_secret(
                'database_password',
                base64.urlsafe_b64encode(os.urandom(24)).decode(),
                SecretType.DATABASE_PASSWORD,
                'Trading database password',
                expires_in_days=30,
                tags={'environment': 'production', 'rotation_frequency': 'monthly'}
            )

        if 'risk_engine_key' not in existing_secrets:
            self.secrets_manager.store_secret(
                'risk_engine_key',
                base64.urlsafe_b64encode(os.urandom(32)).decode(),
                SecretType.ENCRYPTION_KEY,
                'Risk engine encryption key',
                expires_in_days=365,
                tags={'environment': 'production', 'rotation_frequency': 'yearly'}
            )

        if 'webhook_secret' not in existing_secrets:
            self.secrets_manager.store_secret(
                'webhook_secret',
                base64.urlsafe_b64encode(os.urandom(32)).decode(),
                SecretType.WEBHOOK_SECRET,
                'Webhook verification secret',
                expires_in_days=180,
                tags={'environment': 'production', 'rotation_frequency': 'biannually'}
            )

    def _setup_trading_principals(self):
        """Setup standard trading system principals."""

        # Trading service principal
        self.iam_manager.create_principal(
            'trading_service',
            'Trading Service',
            'service',
            AccessLevel.SERVICE,
            ['trading_service']
        )

        # Risk monitoring service
        self.iam_manager.create_principal(
            'risk_monitor',
            'Risk Monitoring Service',
            'service',
            AccessLevel.READ_ONLY,
            ['monitoring_readonly']
        )

        # Admin user
        self.iam_manager.create_principal(
            'admin_user',
            'System Administrator',
            'user',
            AccessLevel.ADMIN,
            ['admin']
        )

    def setup_production_secrets(self) -> Dict[str, str]:
        """Setup production secrets and return API keys."""
        api_keys = {}

        # Generate API keys for services
        for principal_id in ['trading_service', 'risk_monitor']:
            api_key = self.iam_manager.generate_api_key(principal_id)
            if api_key:
                api_keys[principal_id] = api_key

        return api_keys

    def perform_security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit."""
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'secrets_audit': self._audit_secrets(),
            'principals_audit': self._audit_principals(),
            'sessions_audit': self._audit_sessions(),
            'recommendations': []
        }

        # Generate recommendations
        recommendations = self._generate_security_recommendations(audit_results)
        audit_results['recommendations'] = recommendations

        return audit_results

    def _audit_secrets(self) -> Dict[str, Any]:
        """Audit secrets management."""
        secrets = self.secrets_manager.list_secrets()

        return {
            'total_secrets': len(secrets),
            'expiring_soon': len(self.secrets_manager.check_expiring_secrets(30)),
            'requiring_rotation': len([s for s in secrets if s['rotation_required']]),
            'never_accessed': len([s for s in secrets if not s['last_accessed']]),
            'high_access_count': len([s for s in secrets if s['access_count'] > 1000])
        }

    def _audit_principals(self) -> Dict[str, Any]:
        """Audit principals and permissions."""
        principals = list(self.iam_manager.principals.values())

        return {
            'total_principals': len(principals),
            'service_accounts': len([p for p in principals if p.principal_type == 'service']),
            'user_accounts': len([p for p in principals if p.principal_type == 'user']),
            'mfa_enabled': len([p for p in principals if p.mfa_enabled]),
            'admin_access': len([p for p in principals if p.access_level == AccessLevel.ADMIN])
        }

    def _audit_sessions(self) -> Dict[str, Any]:
        """Audit active sessions."""
        sessions = list(self.iam_manager.active_sessions.values())
        now = datetime.now()

        return {
            'active_sessions': len(sessions),
            'old_sessions': len([s for s in sessions if (now - s['last_activity']).total_seconds() > 3600])
        }

    def _generate_security_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on audit."""
        recommendations = []

        # Secrets recommendations
        secrets_audit = audit_results['secrets_audit']
        if secrets_audit['expiring_soon'] > 0:
            recommendations.append(f"Rotate {secrets_audit['expiring_soon']} secrets expiring within 30 days")

        if secrets_audit['requiring_rotation'] > 0:
            recommendations.append(f"Complete rotation for {secrets_audit['requiring_rotation']} marked secrets")

        if secrets_audit['never_accessed'] > 0:
            recommendations.append(f"Review {secrets_audit['never_accessed']} unused secrets for cleanup")

        # Principals recommendations
        principals_audit = audit_results['principals_audit']
        if principals_audit['mfa_enabled'] < principals_audit['user_accounts']:
            recommendations.append("Enable MFA for all user accounts")

        if principals_audit['admin_access'] > 2:
            recommendations.append("Review admin access - consider principle of least privilege")

        # Sessions recommendations
        sessions_audit = audit_results['sessions_audit']
        if sessions_audit['old_sessions'] > 0:
            recommendations.append(f"Cleanup {sessions_audit['old_sessions']} stale sessions")

        return recommendations

    def rotate_all_expiring_secrets(self) -> Dict[str, bool]:
        """Rotate all secrets expiring within 7 days."""
        expiring_secrets = self.secrets_manager.check_expiring_secrets(7)
        results = {}

        for secret_name in expiring_secrets:
            # Generate new secret value based on type
            secret_list = self.secrets_manager.list_secrets()
            secret_info = next((s for s in secret_list if s['name'] == secret_name), None)

            if secret_info:
                secret_type = SecretType(secret_info['secret_type'])
                new_value = self._generate_secret_value(secret_type)
                results[secret_name] = self.secrets_manager.rotate_secret(secret_name, new_value)

        return results

    def _generate_secret_value(self, secret_type: SecretType) -> str:
        """Generate new secret value based on type."""
        if secret_type == SecretType.API_KEY:
            return 'demo_key_' + base64.urlsafe_b64encode(os.urandom(24)).decode()
        elif secret_type == SecretType.DATABASE_PASSWORD:
            return base64.urlsafe_b64encode(os.urandom(24)).decode()
        elif secret_type in [SecretType.ENCRYPTION_KEY, SecretType.WEBHOOK_SECRET]:
            return base64.urlsafe_b64encode(os.urandom(32)).decode()
        else:
            return base64.urlsafe_b64encode(os.urandom(24)).decode()


# Example usage and testing
if __name__ == "__main__":
    def demo_secrets_and_iam():
        print("=== Trading Secrets & IAM Hardening Demo ===")

        # Initialize system
        hardening = TradingSecretsHardening()

        # Setup production secrets
        api_keys = hardening.setup_production_secrets()
        print(f"Generated API keys for {len(api_keys)} services")

        # Test secret access
        trading_key = hardening.secrets_manager.get_secret('broker_api_key', 'trading_service')
        print(f"Retrieved broker API key: {trading_key[:10]}...")

        # Test IAM access control
        api_key = api_keys.get('trading_service')
        if api_key:
            principal = hardening.iam_manager.validate_api_key(api_key)
            if principal:
                can_access, reason = hardening.iam_manager.check_access(
                    principal.id, 'broker_api_key', 'read'
                )
                print(f"Access check: {can_access} - {reason}")

        # Perform security audit
        audit = hardening.perform_security_audit()
        print("\nSecurity Audit Results:")
        print(f"- Total secrets: {audit['secrets_audit']['total_secrets']}")
        print(f"- Active principals: {audit['principals_audit']['total_principals']}")
        print(f"- Active sessions: {audit['sessions_audit']['active_sessions']}")
        print(f"- Recommendations: {len(audit['recommendations'])}")

        for rec in audit['recommendations'][:3]:  # Show first 3
            print(f"  • {rec}")

    demo_secrets_and_iam()