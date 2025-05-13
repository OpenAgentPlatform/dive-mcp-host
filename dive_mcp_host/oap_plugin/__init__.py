from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

from dive_mcp_host.httpd.conf.mcp_servers import (
    CurrentConfigHookName,
    UpdateAllConfigsHookName,
)
from dive_mcp_host.httpd.store.manager import StoreHookName
from dive_mcp_host.oap_plugin.config_mcp_servers import (
    MCPServerManagerPlugin,
    read_oap_config,
)
from dive_mcp_host.oap_plugin.http_handlers import OAPHttpHandlers
from dive_mcp_host.plugins.registry import PluginCallbackDef

from .store import oap_store


def get_static_callbacks() -> dict[str, tuple[Callable[..., Any], PluginCallbackDef]]:
    """Get the static callbacks."""
    oap_config = read_oap_config()

    mcp_plugin = MCPServerManagerPlugin(oap_config.auth_key)
    handlers = OAPHttpHandlers(mcp_plugin, oap_store)

    return {
        "get_mcp_configs": (
            mcp_plugin.current_config_callback,
            PluginCallbackDef(
                hook_point=CurrentConfigHookName, callback="get_mcp_configs"
            ),
        ),
        "update_all_configs": (
            mcp_plugin.update_all_config_callback,
            PluginCallbackDef(
                hook_point=UpdateAllConfigsHookName, callback="update_all_configs"
            ),
        ),
        "http_routes": (
            handlers.get_router,
            PluginCallbackDef(hook_point="httpd.routers", callback="http_routes"),
        ),
    }


class OAPPlugin:
    """OAP Plugin."""

    def __init__(self, _: dict[str, Any]) -> None:
        """Initialize the OAP Plugin."""

    async def __aenter__(self) -> Self:
        """Enter the OAP Plugin."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> bool:
        """Exit the OAP Plugin."""
        return True

    def callbacks(self) -> dict[str, tuple[Callable[..., Any], PluginCallbackDef]]:
        """Get the callbacks."""
        return {
            "oap_store": (
                lambda: oap_store,
                PluginCallbackDef(hook_point=StoreHookName, callback="oap_store"),
            ),
        }
