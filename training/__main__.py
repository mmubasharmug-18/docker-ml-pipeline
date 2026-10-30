import sys

from training.config import ConfigError, load_settings
from training.data import DataError
from training.pipeline import format_report, run


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Bad settings: {error}", file=sys.stderr)
        return 2

    try:
        metadata = run(settings, progress=lambda text: print(text, flush=True))
    except DataError as error:
        print(f"Data problem: {error}", file=sys.stderr)
        return 2

    print()
    print(format_report(metadata))
    return 0


if __name__ == "__main__":
    sys.exit(main())
