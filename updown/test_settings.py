from .settings import *  # noqa: F401,F403


SECRET_KEY = SECRET_KEY or "test-secret-key"
DEBUG = True
ALLOWED_HOSTS = ["testserver", *ALLOWED_HOSTS]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR + "/test.sqlite3",
    }
}

MIGRATION_MODULES = {
    "incidents": None,
    "stations": None,
    "pages": None,
}

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.MD5PasswordHasher",
]

STATICFILES_DIRS = [f"{BASE_DIR}/static"]

# Dummy cache so per-test state never leaks via the prediction-policy
# cache we use in production.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.dummy.DummyCache",
    }
}
