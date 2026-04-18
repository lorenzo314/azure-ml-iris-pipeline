"""Unit tests for the config loader — no Azure connection required."""

from src.utils.config import load_config


def test_config_loads():
    """config.yaml loads without errors."""
    cfg = load_config()
    assert cfg is not None


def test_config_azure_ml_section():
    """config.yaml contains required azure_ml keys."""
    cfg = load_config()
    assert "azure_ml" in cfg
    assert "environment_name" in cfg["azure_ml"]
    assert "compute_name" in cfg["azure_ml"]
    assert "experiment_name" in cfg["azure_ml"]


def test_config_model_section():
    """config.yaml contains required model keys."""
    cfg = load_config()
    assert "model" in cfg
    assert "name" in cfg["model"]
    assert "version" in cfg["model"]


def test_config_deployment_section():
    """config.yaml contains required deployment keys."""
    cfg = load_config()
    assert "deployment" in cfg
    assert "instance_type" in cfg["deployment"]
    assert "deployment_name" in cfg["deployment"]

