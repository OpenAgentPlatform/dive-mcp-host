from os import getenv
from pathlib import Path

RESOURCE_DIR = Path(getenv("RESOURCE_DIR", Path.cwd()))
DIVE_CONFIG_DIR = Path(getenv("DIVE_CONFIG_DIR", Path.cwd()))
