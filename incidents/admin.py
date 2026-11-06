import hashlib

from django.contrib import admin
from django.core.cache import cache
from django.core.exceptions import EmptyResultSet
from django.core.paginator import Paginator
from django.urls import reverse
from django.utils.functional import cached_property
from django.utils.html import format_html, format_html_join

from incidents.models import Incident, Report


class CachedCountPaginator(Paginator):
    """Paginator that caches the (possibly filtered) row COUNT briefly.

    Django's admin evaluates ``paginator.count`` on every changelist page to
    build the pagination links, and ``show_full_result_count = False`` does
    not change that — it only suppresses the extra *unfiltered* total. On
    the ~2M-row Report table a ``COUNT(*)`` is a full index scan that
    dominates the page load on shared-cpu-1x SQLite, and it runs a second
    time whenever a filter is applied. The real count moves by only a
    handful of rows per 5-minute tick, so a short cache is invisible to the
    user and turns almost every changelist load into a cache hit. Keyed on
    the exact SQL so each distinct filter combination caches separately.
    """

    count_cache_seconds = 300

    @cached_property
    def count(self):
        object_list = self.object_list
        query = getattr(object_list, "query", None)
        if query is None:
            return super().count
        try:
            sql = str(query)
        except EmptyResultSet:
            # The query provably matches nothing (e.g. an out-of-range id
            # filter or .none()); the stock Paginator counts that as 0.
            return 0
        key = "admin_count:" + hashlib.md5(sql.encode("utf-8")).hexdigest()
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = object_list.count()
        cache.set(key, result, self.count_cache_seconds)
        return result


@admin.register(Incident)
class IncidentAdmin(admin.ModelAdmin):
    exclude = ["reports"]
    search_fields = ["station__name", "text"]
    # ``station__parent_station__name`` builds the filter sidebar via a
    # DISTINCT join, which on the report-heavy joined queryset costs
    # more time than gunicorn's worker timeout. Removed; use the search
    # box on station name instead.
    list_filter = ["resolved", "start_time"]
    list_display = (
        "station",
        "start_time",
        "end_time",
        "information",
        "resolved",
        "report_count",
    )
    ordering = ("-start_time",)
    raw_id_fields = ["station"]
    list_select_related = ["station", "station__parent_station"]
    show_full_result_count = False
    list_per_page = 50
    paginator = CachedCountPaginator
    readonly_fields = ["linked_reports"]

    # A TabularInline over the reports through-table rendered one raw-id row
    # per linked report with no pagination, and each row fired ~2 queries
    # (ForeignKeyRawIdWidget.label_and_url_for_value + Report.__str__ lazily
    # loading the station) — hundreds to thousands of queries on a busy
    # incident. Instead we show a bounded, self-fetched list of links.
    @admin.display(description="Reports")
    def linked_reports(self, obj):
        if obj is None or not obj.pk:
            return "—"
        reports = list(
            obj.reports.only("id", "source", "start_time").order_by("-start_time")[:100]
        )
        total = (
            obj.report_count
            if obj.report_count is not None
            else obj.reports.count()
        )
        links = format_html_join(
            "",
            '<div><a href="{}">#{} · {} · {}</a></div>',
            (
                (
                    reverse("admin:incidents_report_change", args=[report.pk]),
                    report.pk,
                    report.get_source_display(),
                    report.start_time.strftime("%Y-%m-%d %H:%M"),
                )
                for report in reports
            ),
        )
        suffix = (
            format_html("<div>… and {} more</div>", total - len(reports))
            if total and total > len(reports)
            else ""
        )
        return format_html(
            "<div><strong>{} report(s)</strong></div>{}{}", total, links, suffix
        )


@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    # "text" is deliberately NOT searchable here: on the ~2M-row Report
    # table a search term becomes an unindexable leading-wildcard
    # ``text LIKE '%term%'`` OR'd across the station join, run twice per page
    # (once for the paginator count), which blew past the worker timeout.
    # Station-name search is a small indexed join; search Incident (≈19k
    # rows) when you need to match on text.
    search_fields = ["station__name"]
    # Same reasoning as IncidentAdmin: the parent-station filter scans the
    # whole 2M-row Reports table to populate its sidebar dropdown.
    list_filter = ["resolved", "source"]
    list_display = (
        "station",
        "start_time",
        "end_time",
        "information",
        "resolved",
        "source",
    )
    ordering = ("-start_time",)
    raw_id_fields = ["station"]
    list_select_related = ["station", "station__parent_station"]
    show_full_result_count = False
    list_per_page = 50
    paginator = CachedCountPaginator


admin.site.site_header = "Up Down London Administration"
