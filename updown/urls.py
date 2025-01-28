from django.contrib import admin
from django.urls import path

from incidents.views import detail, UpdateIncidentsView, alexa, stp
from pages.views import FAQPageView, PressPageView, PrivacyPageView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", detail, name="home"),
    path("problems.txt", alexa),
    path("functions/update_incidents", UpdateIncidentsView.as_view()),
    path("faq/", FAQPageView.as_view(), name="faq"),
    path("press/", PressPageView.as_view(), name="press"),
    path("privacy/", PrivacyPageView.as_view(), name="privacy"),
    path("stp/", stp, name="stp"),
]
