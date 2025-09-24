import yaml
from typing import Any, Optional


class ConfigLoader:
    """
    A simple utility class to load YAML configuration files
    and retrieve values using dot notation for nested keys.
    """
    _config: dict = {}

    def __init__(self, config_path: str):
        """
        Initializes the loader by loading the YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            if self._config is None:
                self._config = {}
        except FileNotFoundError:
            print(f"Error: Configuration file not found at '{config_path}'")
            # Or raise a custom exception
            # raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            # Or raise a custom exception
            # raise ValueError(f"Error parsing YAML file: {e}")

    def get_value(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a value from the loaded configuration using a key.
        Supports nested keys using dot notation (e.g., 'database.host').

        Args:
            key (str): The key of the value to retrieve.
            default (Optional[Any]): The default value to return if the key is not found.
                                     Defaults to None.

        Returns:
            Any: The configuration value or the default value.
        """
        # Start with the full configuration dictionary
        value = self._config

        # Split the key by dots to navigate through nested dictionaries
        keys = key.split('.')

        for k in keys:
            # Check if the current value is a dictionary and contains the next key
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # If the key is not found at any level, return the default value
                return default

        return value
