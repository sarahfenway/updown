import os
from urllib.parse import urljoin

import requests


def main(args):
    url_base = os.getenv("FUNCTIONS_URL_BASE", "http://127.0.0.1:8000/functions/")
    secret_key = os.getenv("FUNCTIONS_SECRET_KEY", "verysecret")
    function_name = "update_incidents"  # os.path.basename(os.path.dirname(os.path.realpath(__file__)))

    r = requests.post(urljoin(url_base, function_name), data={"key": secret_key})

    r.raise_for_status()
