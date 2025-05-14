"""MCP Server configuration management in OAP Plugin."""

import logging
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from dive_mcp_host.env import DIVE_CONFIG_DIR
from dive_mcp_host.httpd.conf.mcp_servers import (
    Config,
    MCPServerConfig,
    MCPServerManager,
)
from dive_mcp_host.oap_plugin.models import BaseResponse, OAPConfig, UserMcpConfig

OAP_ROOT_URL = "https://oap-hub.biggo.dev"
CONFIG_FILE = Path(DIVE_CONFIG_DIR, "oap_config.json")
logger = logging.getLogger("OAP_PLUGIN")

MIN_REFRESH_INTERVAL = 60


class MCPServerManagerPlugin:
    """Manage MCP Server configurations in OAP Plugin."""

    def __init__(self, device_token: str = "") -> None:
        """Initialize the MCPServerConfigs from OAP."""
        self.device_token: str = device_token
        self._user_mcp_configs: list[UserMcpConfig] | None = []
        self._refresh_ts: float = 0
        self._http_client = httpx.Client(
            base_url=OAP_ROOT_URL,
            headers={"Authorization": f"bearer {self.device_token}"}
            if self.device_token
            else None,
        )

    def update_device_token(
        self, device_token: str, mcp_server_manager: MCPServerManager
    ) -> None:
        """Update the device token and refresh the configs."""
        self.device_token = device_token
        self._http_client.headers = {"Authorization": f"bearer {self.device_token}"}
        update_oap_token(self.device_token)
        self.refresh(mcp_server_manager)

    def refresh(self, mcp_server_manager: MCPServerManager) -> None:
        """Refresh the MCP server configs."""
        self._get_user_mcp_configs(refresh=True)
        cfg = mcp_server_manager.current_config
        # we already merged the configuration in callback function
        assert cfg is not None
        mcp_server_manager.update_all_configs(cfg)

    def update_all_config_callback(self, new_config: Config) -> Config:
        """Callback function for updating all configs."""
        return new_config

    def current_config_callback(self, config: Config) -> Config:
        """Callback function for getting current config."""
        mcp_servers = self._get_user_mcp_configs()
        # remove oap mcp servers
        if mcp_servers is None:
            for key in config.mcp_servers.copy():
                value = config.mcp_servers[key]
                if value.extra_data and value.extra_data.get("oap"):
                    config.mcp_servers.pop(key)
            return config

        for server in mcp_servers:
            old_server = config.mcp_servers.get(server.name)
            config.mcp_servers[server.name] = MCPServerConfig(
                enabled=old_server.enabled if old_server else True,
                url=server.url,
                transport=server.transport,
                headers=server.headers,  # type: ignore
                extraData={
                    "oap": {
                        "planTag": server.plan.lower(),
                        "description": server.description,
                    }
                },
            )
        return config

    def _send_api_request[T](
        self,
        url: str,
        method: Literal["get", "post", "put", "delete"] = "get",
        model: type[T] | Any = Any,
    ) -> tuple[T | None, int]:
        """Send a request to the API and return the response."""
        response = self._http_client.request(method, url)
        try:
            return (
                BaseResponse[model].model_validate_json(response.text).data,
                response.status_code,
            )
        except ValidationError:
            logger.exception("Failed to validate response: %s", response.text)
            return None, response.status_code

    def _get_user_mcp_configs(
        self, refresh: bool = False
    ) -> list[UserMcpConfig] | None:
        """Get the user MCP configs."""
        url = "/api/v1/user/mcp/configs"
        if (
            refresh
            or not self._user_mcp_configs
            or time.time() - self._refresh_ts > MIN_REFRESH_INTERVAL
        ):
            r, code = self._send_api_request(url, "get", list[UserMcpConfig])
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
        return OAPConfig(auth_key="", store_url="")

    with CONFIG_FILE.open("r") as f:
        return OAPConfig.model_validate_json(f.read())


def update_oap_token(token: str) -> None:
    """Update the OAP token."""
    config = read_oap_config()
    config.auth_key = token
    with CONFIG_FILE.open("w") as f:
        f.write(config.model_dump_json())
