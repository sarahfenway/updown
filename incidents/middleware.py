import os

from django.conf import settings
from django.http import HttpResponse


MAINTENANCE_ENV = "MAINTENANCE_SNAPSHOT"


def _snapshot_path():
    """Where the frozen HTML snapshot lives.

    Defaults to sitting next to the SQLite database (on the Fly volume, so
    it survives a restart mid-maintenance and never needs regenerating
    from a database that's currently being operated on). Overridable via
    the MAINTENANCE_SNAPSHOT_PATH setting.
    """
    explicit = getattr(settings, "MAINTENANCE_SNAPSHOT_PATH", None)
    if explicit:
        return explicit
    sqlite_path = os.getenv("SQLITE_PATH")
    base = os.path.dirname(sqlite_path) if sqlite_path else settings.BASE_DIR
    return os.path.join(base, "maintenance_snapshot.html")


class MaintenanceSnapshotMiddleware:
    """Freeze-frame mode for database maintenance.

    When the MAINTENANCE_SNAPSHOT env var is set, the first request renders
    the current homepage to a static HTML file (a single database read) and
    every request thereafter is served that file verbatim — no further
    database access of any kind. This lets the database be taken offline
    for surgery (dump / repair / VACUUM / restore) while the site keeps
    showing real, recent content instead of an error page.

    Because it intercepts *every* request, the cron's
    /functions/update_incidents POST is short-circuited too, so nothing
    writes to the database during the window.

    This middleware sits immediately after WhiteNoise, which serves
    /static/ before we see the request — so the frozen page's CSS/JS still
    load and it renders correctly.

    Removing the env var and restarting deletes the snapshot and returns to
    normal serving.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if os.environ.get(MAINTENANCE_ENV):
            return self._serve_snapshot(request)

        # Not in maintenance — remove any leftover snapshot, then proceed
        # to the normal view stack.
        path = _snapshot_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return self.get_response(request)

    def _serve_snapshot(self, request):
        path = _snapshot_path()
        if not os.path.exists(path):
            html = self._render_snapshot(request)
            # Atomic write so a second concurrent request never reads a
            # half-written file.
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(html)
            os.replace(tmp, path)
        with open(path, encoding="utf-8") as fh:
            return HttpResponse(fh.read())

    def _render_snapshot(self, request):
        # Imported lazily so the middleware module stays import-light and
        # we don't create an import cycle with views.
        from django.template.loader import render_to_string

        from incidents.views import _status_page_context

        return render_to_string(
            "home.html", _status_page_context(), request=request
        )
