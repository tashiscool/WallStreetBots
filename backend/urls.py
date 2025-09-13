"""URL configuration for WallStreetBots backend."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("backend.home.urls")),
]
