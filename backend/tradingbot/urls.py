from django.urls import path

from . import views

# TODO: add route for creating, patching company
urlpatterns = [
    path("", views.index, name="tradingbot_welcome"),
    path("stock_trade", views.stock_trade, name="stock_trade"),
    path("close_position", views.close_position, name="close_position"),
    path("position/<str:symbol>", views.get_position_details, name="position_details"),
]
