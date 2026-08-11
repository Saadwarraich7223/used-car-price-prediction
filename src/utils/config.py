from pathlib import Path

import yaml


def load_yaml_config(path: str = "configs/model_config.yaml") -> dict:
    """Load a YAML config file, raising a clear error if it is missing."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Run this command from the project root directory."
        )
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create the parent directory of a file path if it does not exist."""
    directory = Path(path)
    directory.parent.mkdir(parents=True, exist_ok=True)
    return directory
