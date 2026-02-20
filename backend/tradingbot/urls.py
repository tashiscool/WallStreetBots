from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="tradingbot_welcome"),
    path("stock_trade", views.stock_trade, name="stock_trade"),
    path("companies", views.create_company, name="company_create"),
    path("companies/<str:ticker>", views.patch_company, name="company_patch"),
    path("close_position", views.close_position, name="close_position"),
    path("position/<str:symbol>", views.get_position_details, name="position_details"),
]
