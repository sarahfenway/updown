from django.contrib import admin

from incidents.models import Incident, Report


class ReportInline(admin.TabularInline):
    model = Incident.reports.through
    raw_id_fields = ["report"]
    extra = 0


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
    )
    ordering = ("-start_time",)
    raw_id_fields = ["station"]
    list_select_related = ["station", "station__parent_station"]
    show_full_result_count = False
    list_per_page = 50

    inlines = [
        ReportInline,
    ]


@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    search_fields = ["station__name", "text"]
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


admin.site.site_header = "Up Down London Administration"
