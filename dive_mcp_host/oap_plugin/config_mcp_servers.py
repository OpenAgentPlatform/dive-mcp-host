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

CONFIG_FILE = Path(DIVE_CONFIG_DIR, "oap_config.json")
logger = logging.getLogger("OAP_PLUGIN")

MIN_REFRESH_INTERVAL = 60


class MCPServerManagerPlugin:
    """Manage MCP Server configurations in OAP Plugin."""

    def __init__(self, device_token: str | None, oap_root_url: str) -> None:
        """Initialize the MCPServerConfigs from OAP."""
        self.device_token: str | None = device_token
        self._user_mcp_configs: list[UserMcpConfig] | None = []
        self._refresh_ts: float = 0
        self._http_client = httpx.Client(
            base_url=oap_root_url,
            headers={"Authorization": f"bearer {self.device_token}"}
            if self.device_token
            else None,
        )

    def update_device_token(
        self, device_token: str | None, mcp_server_manager: MCPServerManager
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

        # oap id and is enable or not
        mcp_enabled = {}
        for server in config.mcp_servers.values():
            if oap := (server.extra_data or {}).get("oap"):
                mcp_enabled[oap["id"]] = server.enabled

        # remove oap mcp servers
        if mcp_servers is None or len(mcp_servers) > 0:
            for key in config.mcp_servers.copy():
                value = config.mcp_servers[key]
                if value.extra_data and value.extra_data.get("oap"):
                    config.mcp_servers.pop(key)

        if mcp_servers is None:
            return config

        for server in mcp_servers:
            config.mcp_servers[server.name] = MCPServerConfig(
                enabled=mcp_enabled.get(server.id, True),
                url=server.url,
                transport=server.transport,
                headers={"Authorization": f"Bearer {self.device_token}", **server.headers}, # type: ignore
                extraData={
                    "oap": {
                        "id": server.id,
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

    def revoke_device_token(self) -> None:
        """Revoke the device token."""
        self._send_api_request("/api/v1/user/devices/self", "delete")

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
        return OAPConfig()

    with CONFIG_FILE.open("r") as f:
        _tmp = OAPConfig.model_validate_json(f.read())
        return OAPConfig(auth_key=_tmp.auth_key)


def update_oap_token(token: str | None) -> None:
    """Update the OAP token."""
    config = read_oap_config()
    config.auth_key = token
    with CONFIG_FILE.open("w") as f:
        f.write(config.model_dump_json())
