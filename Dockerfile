# syntax=docker/dockerfile:1.6
#
# Two-stage build:
#   1. ``builder`` installs Python dependencies into a virtualenv so the
#      final image doesn't carry pip's cache or apt build tools.
#   2. ``runtime`` is a slim base with just the venv + the source tree.
#
# We use python:3.10-slim to match the local dev environment. Switch in
# lockstep when bumping Python locally.

ARG PYTHON_VERSION=3.10

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# psycopg2-binary ships its own libpq, so we don't need libpq-dev. But
# scipy/numpy wheels need ld + libgomp at runtime; the slim image already
# has libgomp via the manylinux wheels — no apt-get install required at
# build time. If a future dep needs a build toolchain we'll add it here.

WORKDIR /app

# Install Python deps into a self-contained venv at /opt/venv so we can
# COPY just that directory into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PORT=8080 \
    DJANGO_SETTINGS_MODULE=updown.settings

# libgomp1 for sklearn; ca-certificates + curl so we can install Litestream
# (Linux .deb) and so the runtime can TLS-talk to B2; gosu so the
# entrypoint can fix /data ownership as root then drop to the app user;
# supercronic for in-container scheduled jobs (replaces NAS cron).
ARG LITESTREAM_VERSION=0.3.13
ARG SUPERCRONIC_VERSION=0.2.29
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates curl gosu \
    && curl -fsSL "https://github.com/benbjohnson/litestream/releases/download/v${LITESTREAM_VERSION}/litestream-v${LITESTREAM_VERSION}-linux-amd64.deb" \
       -o /tmp/litestream.deb \
    && dpkg -i /tmp/litestream.deb \
    && rm /tmp/litestream.deb \
    && curl -fsSL "https://github.com/aptible/supercronic/releases/download/v${SUPERCRONIC_VERSION}/supercronic-linux-amd64" \
       -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

# Collect static files into STATIC_ROOT. WhiteNoise serves them in
# production. We use ``python -m django`` rather than ``manage.py``
# because the project's manage.py lives under updown/ (non-standard
# Django layout) which doesn't put the project root on sys.path. A dummy
# SECRET_KEY is fine for collectstatic — Django only uses it for
# signing, which isn't exercised here.
RUN SECRET_KEY=build-time-dummy DEBUG=false \
    python -m django collectstatic --noinput --settings=updown.settings

# Make the entrypoint executable while we still have root.
RUN chmod +x /app/entrypoint.sh

# Create the unprivileged ``app`` user but DON'T switch to it here. The
# entrypoint enters as root, chowns the mounted /data volume to ``app``
# (volumes can come up owned by root after creation or after an sftp
# upload), and then drops to ``app`` via ``gosu`` for the actual app
# processes. That way root is gone before gunicorn ever sees a request.
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app \
    && mkdir -p /data \
    && chown app:app /data

EXPOSE 8080

# entrypoint.sh:
#   - restores the DB from B2 if the volume is empty,
#   - applies migrations (idempotent),
#   - starts Litestream which exec()s gunicorn as its child.
CMD ["/app/entrypoint.sh"]
