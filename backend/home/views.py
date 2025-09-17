from django.db import connection
from django.http import HttpResponse
from django.utils import timezone


def index(request):
    return HttpResponse(
        "Hello World, welcome to homepage of UTMIST's tradingbot! Checkout /tradingbot as well"
    )


def get_time(request):
    # Use Django's timezone-aware current time instead of raw SQL
    # This works across all database backends (SQLite, PostgreSQL, MySQL, etc.)
    current_datetime = timezone.now()
    return HttpResponse(f"Current time: {current_datetime}")
