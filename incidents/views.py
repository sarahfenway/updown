from datetime import timedelta

from django.conf import settings
from django.core.management import call_command
from django.http import HttpResponse, HttpResponseNotFound
from django.shortcuts import render
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt

from incidents.models import Incident
from incidents.utils import get_last_updated


@never_cache
def detail(request):
    issues = Incident.objects.filter(resolved=False, information=False).order_by(
        "-start_time", "station__parent_station"
    )
    resolved = Incident.objects.filter(
        resolved=True, end_time__gte=timezone.now() - timedelta(hours=12)
    ).order_by("-start_time", "station__parent_station")
    information = Incident.objects.filter(resolved=False, information=True).order_by(
        "-start_time", "station__parent_station"
    )

    return render(
        request,
        "home.html",
        {
            "issues": issues,
            "resolved": resolved,
            "information": information,
            "last_updated": get_last_updated(),
        },
    )


def alexa(request):
    issues = Incident.objects.filter(resolved=False, information=False).order_by(
        "station__parent_station"
    )

    if issues.count() == 0:
        alexa_string = "There are currently no reported step free access issues on the \
            Transport for London network."
    else:
        alexa_string = "There are step free access issues at: "
        alexa_string += ", ".join(
            sorted(issues.values_list("station__parent_station__name", flat=True))[0:-1]
        )

        if issues.count() > 1:
            alexa_string += " and "

        alexa_string += sorted(
            issues.values_list("station__parent_station__name", flat=True)
        )[-1]

    alexa_string = alexa_string.replace("&", "and")

    return HttpResponse(alexa_string)


@method_decorator(csrf_exempt, name="dispatch")
class UpdateIncidentsView(View):
    def post(self, request, *args, **kwargs):
        if request.POST.get("key") == settings.FUNCTIONS_SECRET_KEY:
            call_command("update_incidents")
            return HttpResponse(status=204)
        return HttpResponseNotFound()
