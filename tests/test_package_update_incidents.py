import importlib.util

from pathlib import Path
from unittest.mock import MagicMock, patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "updown"
    / "update_incidents"
    / "__main__.py"
)

SPEC = importlib.util.spec_from_file_location("update_incidents_main", MODULE_PATH)
update_incidents_main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update_incidents_main)


class TestPackageEntrypoint:
    def test_main_posts_to_default_function_endpoint(self):
        response = MagicMock()

        with patch.object(
            update_incidents_main.requests, "post", return_value=response
        ) as post_mock, patch.dict(update_incidents_main.os.environ, {}, clear=True):
            update_incidents_main.main([])

        post_mock.assert_called_once_with(
            "http://127.0.0.1:8000/functions/update_incidents",
            data={"key": "verysecret"},
        )
        response.raise_for_status.assert_called_once_with()

    def test_main_uses_environment_overrides(self):
        response = MagicMock()

        with patch.object(
            update_incidents_main.requests, "post", return_value=response
        ) as post_mock, patch.dict(
            update_incidents_main.os.environ,
            {
                "FUNCTIONS_URL_BASE": "https://example.com/functions/",
                "FUNCTIONS_SECRET_KEY": "super-secret",
            },
            clear=True,
        ):
            update_incidents_main.main([])

        post_mock.assert_called_once_with(
            "https://example.com/functions/update_incidents",
            data={"key": "super-secret"},
        )
        response.raise_for_status.assert_called_once_with()
