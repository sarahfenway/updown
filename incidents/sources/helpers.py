from incidents.models import Report


def existing_report_lookup(source, *, resolved=False):
    reports_by_key = {}
    cleared_disruption = {}

    reports = Report.objects.filter(source=source)
    if resolved is not None:
        reports = reports.filter(resolved=resolved)
    reports = reports.order_by().only(
        "id", "station_id", "text", "source", "information", "resolved", "end_time"
    )

    for report in reports:
        key = (report.station_id, report.text, report.source, report.information)
        reports_by_key.setdefault(key, []).append(report)
        if resolved is False:
            cleared_disruption[report.pk] = report

    if resolved is False:
        return reports_by_key, cleared_disruption

    cleared_reports = (
        Report.objects.filter(resolved=False, source=source)
        .order_by()
        .only(
            "id",
            "station_id",
            "text",
            "source",
            "information",
            "resolved",
            "end_time",
        )
    )
    for report in cleared_reports:
        cleared_disruption[report.pk] = report

    return reports_by_key, cleared_disruption
