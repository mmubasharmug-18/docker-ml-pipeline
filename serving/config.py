import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    model_dir: Path = Path("models")
    model_version: Optional[str] = None


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    env = os.environ if env is None else env
    problems = []

    raw_level = env.get("LOG_LEVEL", "").strip()
    log_level = (raw_level or "INFO").upper()
    if log_level not in LOG_LEVELS:
        problems.append(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)} (got {raw_level!r})")

    port = 8000
    raw_port = env.get("APP_PORT", "").strip()
    if raw_port:
        if raw_port.isdigit() and 1 <= int(raw_port) <= 65535:
            port = int(raw_port)
        else:
            problems.append(f"APP_PORT must be a whole number from 1 to 65535 (got {raw_port!r})")

    version = env.get("MODEL_VERSION", "").strip() or None
    if version and not re.fullmatch(r"v\d+", version):
        problems.append(f"MODEL_VERSION must look like v1, v2, v3 (got {version!r})")

    if problems:
        raise ConfigError("; ".join(problems))

    return Settings(
        host=env.get("APP_HOST", "").strip() or "127.0.0.1",
        port=port,
        log_level=log_level,
        model_dir=Path(env.get("MODEL_DIR", "").strip() or "models"),
        model_version=version,
    )
