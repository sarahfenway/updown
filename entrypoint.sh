#!/bin/sh
#
# Container entrypoint. Runs briefly as root to ensure /data is owned by
# the app user (volumes can come up root-owned after creation or after a
# manual sftp upload of the SQLite file), then drops privileges via
# gosu for everything else.
#
# Order:
#   1. chown /data so SQLite + Litestream can write.
#   2. If the SQLite file is missing but a Litestream replica exists in
#      B2, restore it. Recovers from a destroyed volume automatically.
#   3. Apply Django migrations (idempotent — creates the schema on a
#      truly fresh boot, no-op afterwards).
#   4. Hand off to Litestream's replicate-and-exec, which keeps the WAL
#      streaming to B2 and runs gunicorn as its child process.

set -e

DB_PATH="${SQLITE_PATH:-/data/db.sqlite3}"

# As root: fix ownership of /data. Idempotent and fast even on a populated
# volume — chown only touches inodes whose ownership differs.
if [ "$(id -u)" = "0" ]; then
    chown -R app:app /data
fi

# Helper: run a command as the app user, regardless of whether we
# started as root (use gosu) or already as app (run direct).
run_as_app() {
    if [ "$(id -u)" = "0" ]; then
        gosu app "$@"
    else
        "$@"
    fi
}

if [ "${DB_ENGINE:-postgres}" = "sqlite" ] && [ ! -f "${DB_PATH}" ]; then
    echo "[entrypoint] Local DB missing at ${DB_PATH} — attempting Litestream restore"
    run_as_app litestream restore -v -if-replica-exists -config /app/litestream.yml "${DB_PATH}" \
        || echo "[entrypoint] No B2 replica found; will start from a fresh DB"
fi

# Gunicorn config: one sync worker. The cron's ``update_incidents`` may
# block traffic briefly while it runs, but the steady-state cost is
# minimal and we don't pay GIL-on-shared-CPU overhead. Timeout bumped to
# 180s so the cron has room to finish without SIGKILL.
GUNICORN_ARGS="updown.wsgi:application --bind 0.0.0.0:8080 \
    --workers 1 --timeout 180 \
    --access-logfile - --error-logfile -"

# Background cron scheduler. supercronic runs as PID > 1, logs to stdout
# (captured by Fly), and spawns each job as a separate child process so
# the gunicorn worker is never blocked by a tick. We launch it before
# Litestream/gunicorn so cron is up the moment the machine is.
if [ -f /app/crontab ]; then
    echo "[entrypoint] Starting supercronic (in-container cron)"
    if [ "$(id -u)" = "0" ]; then
        gosu app supercronic /app/crontab &
    else
        supercronic /app/crontab &
    fi
fi

if [ "${DB_ENGINE:-postgres}" = "sqlite" ]; then
    echo "[entrypoint] Applying migrations"
    run_as_app python -m django migrate --noinput --settings=updown.settings
    echo "[entrypoint] Launching Litestream + gunicorn"
    if [ "$(id -u)" = "0" ]; then
        exec gosu app litestream replicate -config /app/litestream.yml \
            -exec "gunicorn ${GUNICORN_ARGS}"
    else
        exec litestream replicate -config /app/litestream.yml \
            -exec "gunicorn ${GUNICORN_ARGS}"
    fi
else
    echo "[entrypoint] DB_ENGINE=${DB_ENGINE:-postgres}; skipping Litestream and migrations"
    if [ "$(id -u)" = "0" ]; then
        exec gosu app gunicorn ${GUNICORN_ARGS}
    else
        exec gunicorn ${GUNICORN_ARGS}
    fi
fi
