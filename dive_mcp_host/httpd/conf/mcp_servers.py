import json
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, SecretStr, field_serializer

from dive_mcp_host.httpd.conf.misc import DIVE_CONFIG_DIR, write_then_replace


# Define necessary types for configuration
class MCPServerConfig(BaseModel):
    """MCP Server configuration model."""

    transport: (
        Annotated[
            Literal["stdio", "sse", "websocket"],
            BeforeValidator(lambda v: "stdio" if v == "command" else v),
        ]
        | None
    ) = "stdio"
    enabled: bool = True
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, SecretStr] | None = None

    @field_serializer("headers", when_used="json")
    def dump_headers(
        self, headers: dict[str, SecretStr] | None
    ) -> dict[str, str] | None:
        """Serialize the headers field to plain text."""
        if not headers:
            return None
        return {k: v.get_secret_value() for k, v in headers.items()}


class Config(BaseModel):
    """Model of mcp_config.json."""

    mcp_servers: dict[str, MCPServerConfig] = Field(alias="mcpServers")


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
        self._config_path: str = config_path or str(DIVE_CONFIG_DIR / "mcp_config.json")
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
        logger.info("Initializing MCPServerManager from %s", self._config_path)
        env_config = os.environ.get("DIVE_MCP_CONFIG_CONTENT")

        if env_config:
            config_content = env_config
        elif Path(self._config_path).exists():
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()
        else:
            logger.warning("MCP server configuration not found")
            return

        config_dict = json.loads(config_content)
        self._current_config = Config(**config_dict)

    def get_enabled_servers(self) -> dict[str, MCPServerConfig]:
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
        write_then_replace(
            Path(self._config_path),
            new_config.model_dump_json(by_alias=True),
        )

        self._current_config = new_config
        return True
