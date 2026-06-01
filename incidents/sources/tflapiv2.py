import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from incidents.models import Report
from incidents.sources.helpers import existing_report_lookup
from incidents.utils import remove_tfl_specifics
from stations.utils import find_station_from_naptan


def check():
    StatusPageURI = f"https://api.tfl.gov.uk/Disruptions/Lifts/?app_id={settings.TFL_API_ID}&app_key={settings.TFL_API_KEY}"

    reports_by_key, cleared_disruption = existing_report_lookup(
        Report.SOURCE_TFLAPI_V2, resolved=None
    )

    try:
        # Hard cap (connect, read) so a hung TfL endpoint can't block the
        # update_incidents run indefinitely.
        r = requests.get(StatusPageURI, timeout=(5, 20))

        if r.status_code == 200 and len(r.text) > 0:
            disruption = r.json()

            with transaction.atomic():
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

                        key = (
                            station.pk,
                            status_details,
                            Report.SOURCE_TFLAPI_V2,
                            False,
                        )
                        existing_reports = reports_by_key.get(key)

                        if existing_reports:
                            if len(existing_reports) > 1:
                                raise Report.MultipleObjectsReturned
                            for report in existing_reports:
                                cleared_disruption.pop(report.pk, None)
                        else:
                            report = Report.objects.create(
                                station=station,
                                text=status_details,
                                source=Report.SOURCE_TFLAPI_V2,
                                information=False,
                            )
                            reports_by_key[key] = [report]

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
