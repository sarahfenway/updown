from django.contrib import admin
from stations.models import LiftOutage, Station


class StationAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "naptan_id",
        "hub_naptan_id",
        "tube",
        "dlr",
        "national_rail",
        "crossrail",
        "overground",
    )
    list_filter = ["tube", "dlr", "national_rail", "crossrail", "overground"]
    ordering = ("name",)
    search_fields = ["name", "naptan_id", "hub_naptan_id"]
    raw_id_fields = ["parent_station"]
    show_full_result_count = False
    list_per_page = 100


admin.site.register(Station, StationAdmin)


@admin.register(LiftOutage)
class LiftOutageAdmin(admin.ModelAdmin):
    """Background record only; see stations/step_free/broken_lifts.py."""

    list_display = ("lift_id", "station", "first_seen", "cleared_at")
    search_fields = ("lift_id", "tfl_station_id", "station__name")
    raw_id_fields = ("station",)
