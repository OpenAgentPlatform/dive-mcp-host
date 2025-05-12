from fastapi import APIRouter, Depends

from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.server import DiveHostAPI

from .config_mcp_servers import MCPServerManagerPlugin


class OAPHttpHandlers:
    """OAP Plugin."""

    def __init__(self, mcp_server_manager: MCPServerManagerPlugin) -> None:
        """Initialize the OAP Plugin."""
        self._mcp_server_manager = mcp_server_manager
        self._router = APIRouter(tags=["oap_plugin"])
        self._router.post("/auth")(self.auth_handler)
        self._router.post("/config/refresh")(self.refresh_config_handler)

    async def auth_handler(
        self, token: str, app: DiveHostAPI = Depends(get_app)
    ) -> None:
        """Update the device token."""
        self._mcp_server_manager.update_device_token(
            token, app.mcp_server_config_manager
        )

    async def refresh_config_handler(self, app: DiveHostAPI = Depends(get_app)) -> None:
        """Refresh the config."""
        self._mcp_server_manager.refresh(app.mcp_server_config_manager)

    def get_router(self) -> APIRouter:
        """Get the router."""
        return self._router
