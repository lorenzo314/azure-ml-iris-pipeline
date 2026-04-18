"""Shared configuration loader.

Loads configs/config.yaml and exposes a simple helper used by all scripts.
Sensitive values (subscription ID, resource group, workspace name) are always
read from environment variables — never from the config file.
"""

import os
from pathlib import Path
import yaml

# Repo root is two levels up from this file (src/utils/config.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | None = None) -> dict:
    """Load and return the YAML configuration file as a dictionary.

    Resolves the config path relative to the repo root, so this works
    regardless of the working directory — including inside Azure ML pipeline
    steps where the working directory is the src/ folder.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_azure_credentials() -> tuple[str, str, str]:
    """Read Azure workspace credentials from environment variables.

    Returns:
        (subscription_id, resource_group, workspace_name)

    Raises:
        EnvironmentError: if any required variable is missing.
    """
    required = ["SUBSCRIPTION_ID", "RESOURCE_GROUP", "WORKSPACE_NAME"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your values."
        )
    return (
        os.environ["SUBSCRIPTION_ID"],
        os.environ["RESOURCE_GROUP"],
        os.environ["WORKSPACE_NAME"],
    )
