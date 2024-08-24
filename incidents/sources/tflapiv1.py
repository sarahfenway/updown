import requests
from django.conf import settings
from django.utils import timezone

from incidents.models import Report
from incidents.utils import (
    find_dates,
    fix_additional_info_grammar,
    remove_tfl_specifics,
)
from stations.utils import find_station_from_naptan


def check():
    StatusPageURI = (
        "https://api.tfl.gov.uk/StopPoint/Mode/tube,cable-car,dlr,national-rail,overground,river-bus,elizabeth-line,tram/Disruption?includeRouteBlockedStops=True&app_id=%s&app_key=%s"
        % (settings.TFL_API_ID, settings.TFL_API_KEY)
    )

    cleared_disruption = list(
        Report.objects.filter(resolved=False, source=Report.SOURCE_TFLAPI_V1)
    )

    try:
        r = requests.get(StatusPageURI)

        if r.status_code == 200 and len(r.text) > 0:
            disruption = r.json()

            for issue in disruption:
                # All issues should now be caught with this text
                description = issue["description"].lower().replace("-", " ")
                if (
                    "step free access is not available" in description
                    or "there will be no step free access" in description
                    or "no step free access to" in description
                    or "no step free access" in description
                ):
                    try:
                        first_colon = issue["description"].find(":")

                        station = find_station_from_naptan(issue["atcoCode"])
                        status_details = issue["description"][first_colon + 1 :].strip()
                        status_details = status_details.replace(
                            "No Step Free Access - ", ""
                        )
                        status_details = remove_tfl_specifics(status_details)

                        if issue.get("additionalInformation"):
                            additional_info = fix_additional_info_grammar(
                                issue["additionalInformation"]
                            )
                            status_details += "<p><i>%s</i></p>" % additional_info

                        report, created = Report.objects.get_or_create(
                            station=station,
                            text=status_details,
                            source=Report.SOURCE_TFLAPI_V1,
                            information=issue["appearance"] != "RealTime",
                            resolved=False,
                        )

                        if created:
                            if issue["appearance"] == "PlannedWork":
                                start_date, end_date = find_dates(status_details)
                                report.start_time = start_date
                                report.end_time = end_date
                                if " changes " not in status_details:
                                    if (start_date and start_date < timezone.now()) or (
                                        end_date and end_date > timezone.now()
                                    ):
                                        report.information = False
                            report.save()
                        else:
                            if report in cleared_disruption:
                                cleared_disruption.remove(report)
                    except ValueError:
                        pass

            for report in cleared_disruption:
                report.resolved = True
                report.end_time = timezone.now()
                report.save()

    except requests.exceptions.ConnectionError:
        pass
