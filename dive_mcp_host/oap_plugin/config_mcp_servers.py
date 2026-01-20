"""MCP Server configuration management in OAP Plugin."""

import logging
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
    ExternalEndpointCMD,
    ExternalEndpointHttp,
    OAPConfig,
    UserMcpConfig,
)

CONFIG_FILE = Path(DIVE_CONFIG_DIR, "oap_config.json")
logger = logging.getLogger(__name__)

MIN_REFRESH_INTERVAL = 60


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
            headers={"Authorization": f"Bearer {self.device_token}"}
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
        self.device_token = device_token
        self._http_client.headers = {"Authorization": f"Bearer {self.device_token}"}
        update_oap_token(self.device_token)
        await self.refresh(mcp_server_manager)

    async def refresh(self, mcp_server_manager: MCPServerManager) -> None:
        """Refresh the MCP server configs."""
        await self._get_user_mcp_configs(refresh=True)
        cfg = await mcp_server_manager.get_current_config()
        # we already merged the configuration in callback function
        assert cfg is not None
        await mcp_server_manager.update_all_configs(cfg)

    def update_all_config_callback(self, new_config: Config) -> Config:
        """Callback function for updating all configs."""
        return new_config

    def builtin_mcp(self, config: Config) -> Config:
        """Load builtin MCP."""
        headers = {}
        if self.device_token:
            headers["Authorization"] = SecretStr(f"Bearer {self.device_token}")
        config.mcp_servers["Search_MCP"] = MCPServerConfig(
            headers=headers,
            url="https://proxy.oaphub.ai/v1/mcp/246152813338427392",
            transport="streamable",
            enabled=True,
            extraData={ExtraDataKey.HIDE: True},
        )

        env = {}
        if self.device_token:
            env["OAP_CLIENT_KEY"] = self.device_token
        config.mcp_servers["File_Uploader"] = MCPServerConfig(
            transport="stdio",
            command="npx",
            args=["@oaphub/file-uploader-mcp"],
            enabled=True,
            env=env,
            extraData={ExtraDataKey.HIDE: True},
        )

        return config

    async def current_config_callback(self, config: Config) -> Config:
        """Callback function for getting current config."""
        mcp_servers = await self._get_user_mcp_configs()

        # oap id and is enable or not
        mcp_enabled = {}
        for server in config.mcp_servers.values():
            if oap := (server.extra_data or {}).get("oap"):
                mcp_enabled[oap["id"]] = server.enabled

        # remove oap mcp servers
        # preserve old config, might be used to construct final output
        old_oap: dict[str, MCPServerConfig] = {}
        if mcp_servers is None or len(mcp_servers) > 0:
            for key in config.mcp_servers.copy():
                value = config.mcp_servers[key]
                if value.extra_data and value.extra_data.get("oap"):
                    old_oap[key] = config.mcp_servers.pop(key)

        if mcp_servers is None:
            return config

        for server in mcp_servers:
            old_config = old_oap.get(server.name)
            enabled = mcp_enabled.get(server.id, True)
            exclude_tools = old_config.exclude_tools if old_config else []
            extra_data = {
                "oap": {
                    "id": server.id,
                    "planTag": server.plan.lower(),
                    "description": server.description,
                }
            }
            if (
                isinstance(server.external_endpoint, dict)
                or server.external_endpoint is None
            ):
                headers = {k: SecretStr(v) for k, v in server.headers.items()}
                if server.auth_type == "header":
                    headers["Authorization"] = SecretStr(f"Bearer {self.device_token}")
                new_config = MCPServerConfig(
                    headers=headers,
                    url=server.url,
                    transport=server.transport,
                    enabled=enabled,
                    exclude_tools=exclude_tools,
                    extraData=extra_data,
                )
            elif isinstance(server.external_endpoint, ExternalEndpointHttp):
                headers = {
                    k: SecretStr(self._replace_token_holder(v))
                    for k, v in server.external_endpoint.headers.items()
                }
                new_config = MCPServerConfig(
                    url=self._replace_token_holder(server.external_endpoint.url),
                    transport=server.external_endpoint.protocol,
                    headers=headers if headers else None,
                    enabled=enabled,
                    exclude_tools=exclude_tools,
                    extraData=extra_data,
                )
            elif isinstance(server.external_endpoint, ExternalEndpointCMD):
                new_config = MCPServerConfig(
                    command=server.external_endpoint.command,
                    args=server.external_endpoint.args.copy(),
                    env={
                        self._replace_token_holder(k): self._replace_token_holder(v)
                        for k, v in server.external_endpoint.env.items()
                    },
                    transport="stdio",
                    enabled=enabled,
                    exclude_tools=exclude_tools,
                    extraData=extra_data,
                )
            else:
                logger.warning("unknown external endpoint type: %s", server.name)
                new_config = None
            if new_config:
                config.mcp_servers[server.name] = new_config
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
