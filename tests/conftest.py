import os
import sys

import django

from django.test.utils import (
    setup_databases,
    setup_test_environment,
    teardown_databases,
    teardown_test_environment,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "updown.test_settings")
django.setup()


def pytest_sessionstart(session):
    setup_test_environment()
    session.config._django_db_state = setup_databases(verbosity=1, interactive=False)


def pytest_sessionfinish(session, exitstatus):
    teardown_databases(session.config._django_db_state, verbosity=1)
    teardown_test_environment()
