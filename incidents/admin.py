from django.contrib import admin

from incidents.models import Incident, Report


class ReportInline(admin.TabularInline):
    model = Incident.reports.through
    raw_id_fields = ["report"]


@admin.register(Incident)
class IncidentAdmin(admin.ModelAdmin):
    exclude = ["reports"]
    search_fields = ["station__name", "text"]
    list_filter = ["resolved", "start_time", "station__parent_station__name"]
    list_display = (
        "station",
        "start_time",
        "end_time",
        "information",
        "resolved",
    )
    ordering = ("-start_time",)
    raw_id_fields = ["station"]

    inlines = [
        ReportInline,
    ]


@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    search_fields = ["station__name", "text"]
    list_filter = ["resolved", "station__parent_station__name", "source"]
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


admin.site.site_header = "Up Down London Administration"
