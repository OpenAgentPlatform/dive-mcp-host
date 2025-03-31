import json
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field


# Define necessary types for configuration
class ServerConfig(BaseModel):
    """Server configuration model."""

    transport: Annotated[
        Literal["stdio", "sse", "websocket"],
        BeforeValidator(lambda v: "stdio" if v == "command" else v),
    ]
    enabled: bool = True
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None


class Config(BaseModel):
    """Configuration model."""

    mcp_servers: dict[str, ServerConfig] = Field(alias="mcpServers")


# Logger setup
logger = logging.getLogger(__name__)


class MCPServerManager:
    """MCP Server Manager for configuration handling."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the MCPServerManager.

        Args:
            config_path: Optional path to the configuration file.
                If not provided, it will be set to "config.json" in current
                working directory.
        """
        self._config_path: str = config_path or str(Path.cwd() / "config.json")
        self._current_config: Config | None = None

    @property
    def config_path(self) -> str:
        """Get the configuration path."""
        return self._config_path

    @property
    def current_config(self) -> Config | None:
        """Get the current configuration."""
        return self._current_config

    def initialize(self) -> None:
        """Initialize the MCPServerManager.

        Returns:
            True if successful, False otherwise.
        """
        env_config = os.environ.get("DIVE_MCP_CONFIG_CONTENT")

        if env_config:
            config_content = env_config
        else:
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()

        config_dict = json.loads(config_content)
        self._current_config = Config(**config_dict)

    def get_enabled_servers(self) -> dict[str, ServerConfig]:
        """Get list of enabled server names.

        Returns:
            Dictionary of enabled server names and their configurations.
        """
        if not self._current_config:
            return {}

        return {
            server_name: config
            for server_name, config in self._current_config.mcp_servers.items()
            if config.enabled
        }

    def update_all_configs(self, new_config: Config) -> bool:
        """Replace all configurations.

        Args:
            new_config: New configuration.

        Returns:
            True if successful, False otherwise.
        """
        with Path(self._config_path).open("w", encoding="utf-8") as f:
            json_dict = new_config.model_dump(by_alias=True)
            json.dump(json_dict, f, indent=2)

        self._current_config = new_config
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = MCPServerManager()
    manager.initialize()
    if manager.current_config:
        logger.info(
            "Available server configurations: %s",
            list(manager.current_config.mcp_servers.keys()),
        )

    enabled_servers = manager.get_enabled_servers()
    logger.info("Enabled servers: %s", enabled_servers)
