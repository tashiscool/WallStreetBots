"""
Signals for auto-assigning platform roles to new users.

When a new user is created (via Auth0 OAuth or Django admin), they are
automatically assigned the default platform role ('trader').
"""

import logging

from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

from .permissions import DEFAULT_ROLE, PlatformRoles, assign_role

logger = logging.getLogger(__name__)


@receiver(post_save, sender=User)
def assign_default_role(sender, instance, created, **kwargs):
    """Assign the default platform role to newly created users."""
    if not created:
        return

    # Superusers get admin role
    if instance.is_superuser:
        assign_role(instance, PlatformRoles.ADMIN)
        logger.info("Assigned admin role to superuser %s", instance.username)
    else:
        assign_role(instance, DEFAULT_ROLE)
        logger.info("Assigned default role '%s' to new user %s", DEFAULT_ROLE, instance.username)
