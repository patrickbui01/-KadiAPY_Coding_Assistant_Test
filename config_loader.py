import json
import os

class ConfigLoader:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.config_path = config_path

    def load(self):
        with open(self.config_path, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSON: {e}")