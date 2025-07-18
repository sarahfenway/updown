from datetime import datetime

import requests
from django.conf import settings
from django.utils import timezone

from incidents.models import Report
from incidents.utils import remove_tfl_specifics
from stations.utils import find_station_from_naptan


def check():
    StatusPageURI = f"https://api.tfl.gov.uk/Disruptions/Lifts/?app_id={settings.TFL_API_ID}&app_key={settings.TFL_API_KEY}"

    cleared_disruption = list(
        Report.objects.filter(resolved=False, source=Report.SOURCE_TFLAPI_V2)
    )

    try:
        r = requests.get(StatusPageURI)

        if r.status_code == 200 and len(r.text) > 0:
            disruption = r.json()

            for issue in disruption:
                try:
                    station = find_station_from_naptan(issue["stationUniqueId"])

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
