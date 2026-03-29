from django.contrib import admin
from stations.models import Station


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
