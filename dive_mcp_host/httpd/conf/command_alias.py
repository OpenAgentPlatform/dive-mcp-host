import logging
import os
from pathlib import Path

from pydantic import Field, RootModel

from dive_mcp_host.env import DIVE_CONFIG_DIR

logger = logging.getLogger(__name__)


class CommandAliasConfig(RootModel[dict[str, str]]):
    """Configuration model."""

    root: dict[str, str] = Field(default_factory=dict)


class CommandAliasManager:
    """Command Alias Manager for configuration handling."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the CommandAliasManager.

        Args:
            config_path: Optional path to the configuration file.
                If not provided, it will be set to "config.json" in current
                working directory.
        """
        self._config_path: str = config_path or str(
            DIVE_CONFIG_DIR / "command_alias.json"
        )
        self._current_config: dict[str, str] | None = None

    @property
    def config_path(self) -> str:
        """Get the configuration path."""
        return self._config_path

    @property
    def current_config(self) -> dict[str, str] | None:
        """Get the current configuration."""
        return self._current_config

    def initialize(self) -> None:
        """Initialize the CommandAliasManager.

        Returns:
            True if successful, False otherwise.
        """
        logger.info("Initializing CommandAliasManager from %s", self._config_path)
        env_config = os.environ.get("DIVE_COMMAND_ALIAS_CONTENT")

        if env_config:
            config_content = env_config
        else:
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()

        config_dict = CommandAliasConfig.model_validate_json(config_content)
        self._current_config = config_dict.root
