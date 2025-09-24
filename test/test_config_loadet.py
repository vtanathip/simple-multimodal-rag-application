# test/test_config_loader.py

import pytest
from src.utilities.config_loader import ConfigLoader

TEST_CONFIG_PATH = "test/temp_config.yaml"


def test_get_top_level_values():
    """
    Tests retrieving various top-level values.
    """
    config = ConfigLoader(TEST_CONFIG_PATH)

    # Assertions for the new top-level keys
    assert config.get_value("app_name") == "Test App"
    assert config.get_value("version") == "1.0-test"
    assert config.get_value("debug_mode") is False


def test_get_nested_database_values():
    """
    Tests retrieving nested values from the 'database' section.
    """
    config = ConfigLoader(TEST_CONFIG_PATH)

    assert config.get_value("database.host") == "test-db-host"
    assert config.get_value("database.user") == "test_user"
    assert config.get_value("database.port") == 1234


def test_get_list_value():
    """
    Tests retrieving a list of values.
    """
    # This line MUST point to your test file.
    config = ConfigLoader(TEST_CONFIG_PATH)

    keys = config.get_value("api_keys")

    # This assertion will now pass because it's loading the correct file.
    assert isinstance(keys, list)
    assert keys[0] == "test-key-1"


def test_get_non_existent_key_with_default():
    """
    This test remains the same, as it tests the loader's core logic.
    """
    config = ConfigLoader(TEST_CONFIG_PATH)
    value = config.get_value("server.timeout", default=60)
    assert value == 60
