from django.contrib import admin
from django.urls import path

from incidents.views import (
    detail,
    UpdateIncidentsView,
    alexa,
    stp,
    stats,
    api_incidents,
    api_stations,
)
from pages.views import FAQPageView, PrivacyPageView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", detail, name="home"),
    path("api/incidents/", api_incidents, name="incidents"),
    path("api/stations/", api_stations, name="stations"),
    path("problems.txt", alexa),
    path("functions/update_incidents", UpdateIncidentsView.as_view()),
    path("faq/", FAQPageView.as_view(), name="faq"),
    path("privacy/", PrivacyPageView.as_view(), name="privacy"),
    path("stp/", stp, name="stp"),
    path("stats/", stats, name="stats"),
]
