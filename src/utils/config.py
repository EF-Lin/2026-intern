import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.toml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)
