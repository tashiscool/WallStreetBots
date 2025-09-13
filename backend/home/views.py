from django.db import connection
from django.http import HttpResponse


def index(request):
    return HttpResponse(
        "Hello World, welcome to homepage of UTMIST's tradingbot! Checkout /tradingbot as well"
    )


def get_time(request):
    cursor = connection.cursor()
    # Use SQLite-compatible datetime function
    cursor.execute("""SELECT datetime('now', 'localtime')""")
    current_datetime = cursor.fetchone()[0]
    return HttpResponse(f"Current time: {current_datetime}")
