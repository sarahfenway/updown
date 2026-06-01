from django.views.generic import TemplateView


class FAQPageView(TemplateView):
    template_name = "faq.html"


class PrivacyPageView(TemplateView):
    template_name = "privacy.html"
