import sys
import urllib.request
from http.client import HTTPException

from serving.config import ConfigError, load_settings


def check() -> int:
    try:
        settings = load_settings()
    except ConfigError:
        return 1

    url = f"http://127.0.0.1:{settings.port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 0 if response.status == 200 else 1
    except (OSError, HTTPException):
        return 1


if __name__ == "__main__":
    sys.exit(check())
