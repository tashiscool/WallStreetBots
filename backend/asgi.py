"""ASGI config for backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https: //docs.djangoproject.com / en / 3.2 / howto / deployment / asgi/
"""

import os

from django.core.asgi import get_asgi_application


def _default_settings_module() -> str:
    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    if environment in {"prod", "production"}:
        return "backend.production_settings"
    return "backend.settings"


os.environ.setdefault("DJANGO_SETTINGS_MODULE", _default_settings_module())

application = get_asgi_application()
