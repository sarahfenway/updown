import requests
from django.conf import settings
from django.db import transaction
from django.utils.dateparse import parse_datetime
from django.utils import timezone

from incidents.models import Report
from incidents.utils import (
    find_dates,
    fix_additional_info_grammar,
    remove_tfl_specifics,
    strip_step_free_prefix,
)
from incidents.sources.helpers import existing_report_lookup
from stations.utils import find_station_from_naptan


def _is_step_free_disruption(issue):
    description = issue["description"].lower().replace("-", " ")
    return (
        "step free access is not available" in description
        or "there will be no step free access" in description
        or "no step free access to" in description
        or "no step free access" in description
    )


def _parse_tfl_datetime(value):
    if not value:
        return None

    parsed = parse_datetime(value)
    if parsed is None:
        return None
    if timezone.is_naive(parsed):
        return timezone.make_aware(parsed)
    return parsed


def _report_fields(issue, station):
    first_colon = issue["description"].find(":")
    status_details = issue["description"][first_colon + 1 :].strip()
    status_details = strip_step_free_prefix(status_details)
    status_details = remove_tfl_specifics(status_details)

    if issue.get("additionalInformation"):
        additional_info = fix_additional_info_grammar(issue["additionalInformation"])
        status_details += "<p><i>%s</i></p>" % additional_info

    lookup_information = issue["appearance"] != "RealTime"
    information = lookup_information
    timing_fields = {}
    source_start_date = _parse_tfl_datetime(issue.get("fromDate"))

    if issue["appearance"] == "PlannedWork":
        start_date, end_date = find_dates(status_details)
        timing_fields["start_time"] = source_start_date or start_date or timezone.now()
        timing_fields["end_time"] = end_date
        if " changes " not in status_details:
            now = timezone.now()
            if (
                (source_start_date and source_start_date < now)
                or (start_date and start_date < now)
                or (end_date and end_date > now)
            ):
                information = False
    elif source_start_date:
        timing_fields["start_time"] = source_start_date

    return (
        {
            "station": station,
            "text": status_details,
            "source": Report.SOURCE_TFLAPI_V1,
            "information": information,
            "resolved": False,
            **timing_fields,
        },
        lookup_information,
    )


def check():
    StatusPageURI = (
        "https://api.tfl.gov.uk/StopPoint/Mode/tube,cable-car,dlr,national-rail,overground,river-bus,elizabeth-line,tram/Disruption?includeRouteBlockedStops=True&app_id=%s&app_key=%s"
        % (settings.TFL_API_ID, settings.TFL_API_KEY)
    )

    reports_by_key, cleared_disruption = existing_report_lookup(
        Report.SOURCE_TFLAPI_V1
    )

    try:
        # Hard cap (connect, read). Without this a hung TfL endpoint
        # blocks the whole update_incidents run indefinitely — there is
        # no other timeout in the call path.
        r = requests.get(StatusPageURI, timeout=(5, 20))

        if r.status_code == 200 and len(r.text) > 0:
            disruption = r.json()

            with transaction.atomic():
                for issue in disruption:
                    if _is_step_free_disruption(issue):
                        try:
                            station = find_station_from_naptan(issue["atcoCode"])

                            if not station:
                                # We don't have this station in our database
                                continue

                            fields, lookup_information = _report_fields(issue, station)
                            key = (
                                station.pk,
                                fields["text"],
                                fields["source"],
                                lookup_information,
                            )

                            existing_reports = reports_by_key.get(key)
                            if existing_reports:
                                if len(existing_reports) > 1:
                                    raise Report.MultipleObjectsReturned
                                for report in existing_reports:
                                    cleared_disruption.pop(report.pk, None)
                            else:
                                report = Report.objects.create(**fields)
                                if fields["information"] == lookup_information:
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
