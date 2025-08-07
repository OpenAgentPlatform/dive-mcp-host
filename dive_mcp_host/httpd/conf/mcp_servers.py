import json
import logging
import os
from asyncio import iscoroutine
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, SecretStr, field_serializer

from dive_mcp_host.env import DIVE_CONFIG_DIR
from dive_mcp_host.host.conf import ProxyUrl
from dive_mcp_host.httpd.conf.misc import write_then_replace
from dive_mcp_host.plugins.registry import HookInfo, PluginManager


# Define necessary types for configuration
class MCPServerConfig(BaseModel):
    """MCP Server configuration model."""

    transport: (
        Annotated[
            Literal["stdio", "sse", "websocket", "streamable"],
            BeforeValidator(lambda v: "stdio" if v == "command" else v),
        ]
        | None
    ) = "stdio"
    enabled: bool = True
    command: str | None = None
    args: list[str] | None = Field(default_factory=list)
    env: dict[str, str] | None = Field(default_factory=dict)
    url: str | None = None
    extra_data: dict[str, Any] | None = Field(default=None, alias="extraData")
    proxy: ProxyUrl | None = None
    headers: dict[str, SecretStr] | None = Field(default_factory=dict)
    exclude_tools: list[str] = Field(default_factory=list)
    initial_timeout: float = Field(default=10, ge=10)

    def model_post_init(self, _: Any) -> None:
        """Post-initialization hook."""
        if self.transport in ["sse", "websocket"]:
            if self.url is None:
                raise ValueError("url is required for sse and websocket transport")
        elif self.transport == "stdio" and self.command is None:
            raise ValueError("command is required for stdio transport")

    @field_serializer("headers", when_used="json")
    def dump_headers(self, v: dict[str, SecretStr] | None) -> dict[str, str] | None:
        """Serialize the headers field to plain text."""
        return {k: v.get_secret_value() for k, v in v.items()} if v else None


class Config(BaseModel):
    """Model of mcp_config.json."""

    mcp_servers: dict[str, MCPServerConfig] = Field(
        alias="mcpServers", default_factory=dict
    )


type McpServerConfigCallback = Callable[[Config], Config | Coroutine[Any, Any, Config]]
UpdateAllConfigsHookName = "httpd.config.mcp_servers.update_all_configs"
CurrentConfigHookName = "httpd.config.mcp_servers.current_config"

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

        self._update_config_callbacks: list[tuple[McpServerConfigCallback, str]] = []
        self._current_config_callbacks: list[tuple[McpServerConfigCallback, str]] = []

    @property
    def config_path(self) -> str:
        """Get the configuration path."""
        return self._config_path

    async def get_current_config(self) -> Config | None:
        """Get the current configuration."""
        if self._current_config is None:
            return None
        if self._current_config_callbacks:
            config = self._current_config.model_copy(deep=True)
            for item in self._current_config_callbacks:
                callback, plugin_name = item
                try:
                    _ret = callback(config)
                    if iscoroutine(_ret):
                        config = await _ret
                    else:
                        assert isinstance(_ret, Config), "Must be Config type"
                        config = _ret
                except Exception:
                    logger.exception(
                        "current config callback errer, plugin: %s", plugin_name
                    )
            return config
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

    async def get_enabled_servers(self) -> dict[str, MCPServerConfig]:
        """Get list of enabled server names.

        Returns:
            Dictionary of enabled server names and their configurations.
        """
        if config := await self.get_current_config():
            return {
                server_name: config
                for server_name, config in config.mcp_servers.items()
                if config.enabled
            }
        return {}

    async def update_all_configs(self, new_config: Config) -> bool:
        """Replace all configurations.

        Args:
            new_config: New configuration.

        Returns:
            True if successful, False otherwise.
        """
        if self._update_config_callbacks:
            new_config = new_config.model_copy(deep=True)
            for item in self._update_config_callbacks:
                callback, plugin_name = item
                try:
                    _ret = callback(new_config)
                    if iscoroutine(_ret):
                        new_config = await _ret
                    else:
                        assert isinstance(_ret, Config), "Must be Config type"
                        new_config = _ret
                except Exception:
                    logger.exception(
                        "update config callback errer, plugin: %s", plugin_name
                    )

        write_then_replace(
            Path(self._config_path),
            new_config.model_dump_json(by_alias=True),
        )

        self._current_config = new_config
        return True

    def register_plugin(
        self,
        callback: McpServerConfigCallback,
        hook_name: str,
        plugin_name: str,
    ) -> bool:
        """Register the static plugin."""
        if hook_name == CurrentConfigHookName:
            self._current_config_callbacks.append((callback, plugin_name))
        elif hook_name == UpdateAllConfigsHookName:
            self._update_config_callbacks.append((callback, plugin_name))
        else:
            return False
        return True

    def register_hook(self, manager: PluginManager) -> None:
        """Register the hook."""
        manager.register_hookable(
            HookInfo(
                hook_name=CurrentConfigHookName,
                static_register=self.register_plugin,
            )
        )

        manager.register_hookable(
            HookInfo(
                hook_name=UpdateAllConfigsHookName,
                static_register=self.register_plugin,
            )
        )
