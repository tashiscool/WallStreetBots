"""WSGI config for backend project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https: //docs.djangoproject.com / en / 3.2 / howto / deployment / wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application


def _default_settings_module() -> str:
    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    if environment in {"prod", "production"}:
        return "backend.production_settings"
    return "backend.settings"


os.environ.setdefault("DJANGO_SETTINGS_MODULE", _default_settings_module())

application = get_wsgi_application()
