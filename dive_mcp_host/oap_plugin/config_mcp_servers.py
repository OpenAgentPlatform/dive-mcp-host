"""MCP Server configuration management in OAP Plugin."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import SecretStr, ValidationError

from dive_mcp_host.env import DIVE_CONFIG_DIR
from dive_mcp_host.httpd.conf.mcp_servers import (
    Config,
    ExtraDataKey,
    MCPServerConfig,
    MCPServerManager,
)
from dive_mcp_host.oap_plugin.models import (
    BaseResponse,
    OAPConfig,
    UserMcpConfig,
)

CONFIG_FILE = Path(DIVE_CONFIG_DIR, "oap_config.json")
logger = logging.getLogger(__name__)

MIN_REFRESH_INTERVAL = 60
AUTH_HEADER_KEY = "Authorization"


TOKEN_PLACEHOLDER = "{{AccessToken}}"  # noqa: S105


class MCPServerManagerPlugin:
    """Manage MCP Server configurations in OAP Plugin."""

    def __init__(self, device_token: str | None, oap_root_url: str) -> None:
        """Initialize the MCPServerConfigs from OAP."""
        self.device_token: str | None = device_token
        self._user_mcp_configs: list[UserMcpConfig] | None = []
        self._refresh_ts: float = 0
        self._http_client = httpx.AsyncClient(
            base_url=oap_root_url,
            headers={AUTH_HEADER_KEY: f"Bearer {self.device_token}"}
            if self.device_token
            else None,
        )

    def _replace_token_holder(self, inpt: str) -> str:
        """Replace {{AccessToken}} by device_token."""
        if self.device_token:
            return inpt.replace(TOKEN_PLACEHOLDER, self.device_token)
        return inpt

    async def update_device_token(
        self, device_token: str | None, mcp_server_manager: MCPServerManager
    ) -> None:
        """Update the device token and refresh the configs."""
        logout = device_token is None
        prev_token = self.device_token
        self.device_token = device_token
        self._http_client.headers = {"Authorization": f"Bearer {self.device_token}"}
        update_oap_token(self.device_token)
        await self.refresh(mcp_server_manager, logout, prev_token)

    def _logout_handler(self, config: Config, prev_token: str) -> Config:
        """Find current OAP MCPs (ones with `extraData['oap']`).
        Replace device token back to placeholder.
        """  # noqa: D205

        def _token_to_placeholder(inpt: str) -> str:
            return inpt.replace(prev_token, TOKEN_PLACEHOLDER)

        for server in config.mcp_servers.values():
            if server.extra_data and server.extra_data.get("oap"):
                if server.url:
                    server.url = _token_to_placeholder(server.url)
                if server.env:
                    server.env = {
                        _token_to_placeholder(k): _token_to_placeholder(v)
                        for k, v in server.env.items()
                    }
                if server.headers:
                    server.headers = {
                        _token_to_placeholder(k): SecretStr(
                            _token_to_placeholder(v.get_secret_value())
                        )
                        for k, v in server.headers.items()
                    }
        return config

    async def refresh(
        self,
        mcp_server_manager: MCPServerManager,
        logout: bool = False,
        prev_token: str | None = None,
    ) -> None:
        """Refresh the MCP server configs."""
        cfg = await mcp_server_manager.get_current_config()
        # we already merged the configuration in callback function
        assert cfg is not None
        if logout and prev_token:
            self._logout_handler(cfg, prev_token)
        await mcp_server_manager.update_all_configs(cfg)

    def update_all_config_callback(self, new_config: Config) -> Config:
        """Callback function for updating all configs."""
        return new_config

    def builtin_mcp(self, config: Config) -> Config:
        """Load builtin MCP."""
        headers = {}
        extra_url_query = "?access=guest"
        if self.device_token:
            headers["Authorization"] = SecretStr(f"Bearer {self.device_token}")
            extra_url_query = ""

        config.mcp_servers["Search MCP"] = MCPServerConfig(
            headers=headers,
            url="https://proxy.oaphub.ai/v1/mcp/246152813338427392" + extra_url_query,
            transport="streamable",
            enabled=True,
            extraData={ExtraDataKey.HIDE: True},
        )

        env = {}
        if self.device_token:
            env["OAP_CLIENT_KEY"] = self.device_token
        config.mcp_servers["File Uploader"] = MCPServerConfig(
            transport="stdio",
            command=get_npx_path(),
            args=["@oaphub/file-uploader-mcp"],
            enabled=True,
            env=env,
            extraData={ExtraDataKey.HIDE: True},
        )

        return config

    async def current_config_callback(self, config: Config) -> Config:
        """Find current OAP MCPs (ones with `extraData['oap']`).
        Update the ones that uses OAP client key to the current self.device_token.
        """  # noqa: D205
        for server in config.mcp_servers.values():
            if server.extra_data and server.extra_data.get("oap"):
                if server.url:
                    server.url = self._replace_token_holder(server.url)
                if server.env:
                    server.env = {
                        self._replace_token_holder(k): self._replace_token_holder(v)
                        for k, v in server.env.items()
                    }
                if server.headers:
                    server.headers = {
                        self._replace_token_holder(k): SecretStr(
                            self._replace_token_holder(v.get_secret_value())
                        )
                        for k, v in server.headers.items()
                    }
        return config

    async def _send_api_request[T](
        self,
        url: str,
        method: Literal["get", "post", "put", "delete"] = "get",
        model: type[T] | Any = Any,
    ) -> tuple[T | None, int]:
        """Send a request to the API and return the response."""
        response = await self._http_client.request(method, url)
        try:
            return (
                BaseResponse[model].model_validate_json(response.text).data,
                response.status_code,
            )
        except ValidationError:
            logger.exception(
                "Failed to validate response: %s, url: %s, method: %s",
                response.text,
                url,
                method,
            )
            return None, response.status_code

    async def revoke_device_token(self) -> None:
        """Revoke the device token."""
        await self._send_api_request("/api/v1/user/devices/self", "delete")

    async def _get_user_mcp_configs(
        self, refresh: bool = False
    ) -> list[UserMcpConfig] | None:
        """Get the user MCP configs."""
        url = "/api/v1/user/mcp/configs"
        if (
            refresh
            or not self._user_mcp_configs
            or time.time() - self._refresh_ts > MIN_REFRESH_INTERVAL
        ):
            r, code = await self._send_api_request(url, "get", list[UserMcpConfig])
            # nothing to update
            if code != httpx.codes.OK and code not in [
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
            ]:
                r = []
            self._refresh_ts = time.time()
            self._user_mcp_configs = r
        return self._user_mcp_configs


def read_oap_config() -> OAPConfig:
    """Read the OAP config."""
    if not CONFIG_FILE.exists():
        return OAPConfig()

    with CONFIG_FILE.open("r") as f:
        return OAPConfig.model_validate_json(f.read())


def update_oap_token(token: str | None) -> None:
    """Update the OAP token."""
    config = read_oap_config()
    config.auth_key = token
    with CONFIG_FILE.open("w") as f:
        f.write(config.model_dump_json())


def get_npx_path() -> str:
    """Get the npx executable path from environment variable or default to 'npx'."""
    return os.environ.get("TOOL_NPX_PATH", "npx")


def get_uvx_path() -> str:
    """Get the uvx executable path from environment variable or default to 'uvx'."""
    return os.environ.get("TOOL_UVX_PATH", "uvx")
