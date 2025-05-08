"""MCP Server configuration management in OAP Plugin."""

import logging
from contextlib import suppress
from typing import Any, Literal

import httpx

from dive_mcp_host.httpd.conf.mcp_servers import Config, MCPServerConfig
from dive_mcp_host.oap_plugin.models import BaseResponse, UserMcpConfig

OAP_ROOT_URL = "https://oap-hub.biggo.dev"
logger = logging.getLogger("OAP_PLUGIN")


class MCPServerManagerPlugin:
    """Manage MCP Server configurations in OAP Plugin."""

    def __init__(self, device_token: str = "") -> None:
        """Initialize the MCPServerConfigs from OAP."""
        self.device_token: str = device_token
        self._configs: Config
        self._refresh_ts: float = 0
        self._http_client = httpx.Client(
            base_url=OAP_ROOT_URL,
            headers={"Authorization": f"bearer {self.device_token}"},
        )

    def update_device_token(self, device_token: str) -> None:
        """Update the device token and refresh the configs."""
        self.device_token = device_token
        self._http_client.headers = {"Authorization": f"bearer {self.device_token}"}
        self.refresh()

    def refresh(self) -> None:
        """Refresh the MCP server configs."""

    def update_all_config_callback(self, new_config: Config) -> Config:
        """Callback function for updating all configs."""
        return new_config

    def current_config_callback(self, config: Config) -> Config:
        """Callback function for getting current config."""

        mcp_servers = self._get_user_mcp_configs()
        for server in mcp_servers:
            config.mcp_servers[server.name] = MCPServerConfig(
                enabled=True,
                url=server.url,
                transport=server.transport,
                headers=server.headers,
                extraData={
                    "oap": {
                        "planTag": server.plan.lower(),
                    }
                },
            )
        return config

    def _send_api_request[T](
        self,
        url: str,
        method: Literal["get", "post", "put", "delete"] = "get",
        model: type[T] | Any = Any,
    ) -> T | None:
        """Send a request to the API and return the response."""
        response = self._http_client.request(method, url)
        return BaseResponse[model].model_validate_json(response.text).data

    def _get_user_mcp_configs(self) -> list[UserMcpConfig]:
        """Get the user MCP configs."""
        url = "/api/v1/user/mcp/configs"
        return self._send_api_request(url, "get", list[UserMcpConfig]) or []
