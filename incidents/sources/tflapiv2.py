import requests
from django.conf import settings
from django.utils import timezone

from incidents.models import Report
from incidents.utils import remove_tfl_specifics
from stations.utils import find_station_from_naptan


def check():
    StatusPageURI = f"https://api.tfl.gov.uk/Disruptions/Lifts/?app_id={settings.TFL_API_ID}&app_key={settings.TFL_API_KEY}"

    cleared_disruption = {
        report.pk: report
        for report in Report.objects.filter(resolved=False, source=Report.SOURCE_TFLAPI_V2)
    }

    try:
        # Hard cap (connect, read) so a hung TfL endpoint can't block the
        # update_incidents run indefinitely.
        r = requests.get(StatusPageURI, timeout=(5, 20))

        if r.status_code == 200 and len(r.text) > 0:
            disruption = r.json()

            for issue in disruption:
                try:
                    station = find_station_from_naptan(issue["stationUniqueId"])

                    if not station:
                        continue

                    first_colon = issue["message"].find(":")
                    status_details = issue["message"][first_colon + 1 :].strip()
                    status_details = status_details.replace(
                        "No Step Free Access - ", ""
                    )
                    status_details = remove_tfl_specifics(status_details)

                    report, created = Report.objects.get_or_create(
                        station=station,
                        text=status_details,
                        source=Report.SOURCE_TFLAPI_V2,
                        information=False,
                    )

                    if not created:
                        cleared_disruption.pop(report.pk, None)

                except ValueError:
                    pass

            now = timezone.now()
            for report in cleared_disruption.values():
                report.resolved = True
                report.end_time = now
                report.save()

    except requests.exceptions.RequestException:
        # Covers connection errors and timeouts alike — a flaky or slow
        # TfL endpoint just means we skip this run, not crash or hang.
        pass
