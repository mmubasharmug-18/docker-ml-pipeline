import sys

import uvicorn

from serving.config import ConfigError, load_settings


def run() -> None:
    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Bad settings: {error}", file=sys.stderr)
        raise SystemExit(2)

    uvicorn.run(
        "serving.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
