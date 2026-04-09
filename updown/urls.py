from django.contrib import admin
from django.urls import path

from incidents.views import (
    detail,
    UpdateIncidentsView,
    UploadModelView,
    alexa,
    stp,
    stats,
    api_incidents,
    api_stations,
    api_training_data,
)
from pages.views import FAQPageView, PrivacyPageView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", detail, name="home"),
    path("api/incidents/", api_incidents, name="incidents"),
    path("api/stations/", api_stations, name="stations"),
    path("problems.txt", alexa),
    path("functions/update_incidents", UpdateIncidentsView.as_view()),
    path("functions/upload_model", UploadModelView.as_view()),
    path("api/training-data/", api_training_data, name="training_data"),
    path("faq/", FAQPageView.as_view(), name="faq"),
    path("privacy/", PrivacyPageView.as_view(), name="privacy"),
    path("stp/", stp, name="stp"),
    path("stats/", stats, name="stats"),
]
