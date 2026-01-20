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

from .store import OAPStore, oap_store


def get_static_callbacks() -> dict[str, tuple[Callable[..., Any], str]]:
    """Get the static callbacks."""
    oap_config = read_oap_config()

    oap_store.update_store_url(oap_config.store_url, oap_config.verify_ssl)
    oap_store.update_token(oap_config.auth_key)
    mcp_plugin = MCPServerManagerPlugin(oap_config.auth_key, oap_config.oap_root_url)
    handlers = OAPHttpHandlers(mcp_plugin, oap_store)

    return {
        "builtin_mcp": (mcp_plugin.builtin_mcp, CurrentConfigHookName),
        "get_mcp_configs": (mcp_plugin.current_config_callback, CurrentConfigHookName),
        "update_all_configs": (
            mcp_plugin.update_all_config_callback,
            UpdateAllConfigsHookName,
        ),
        "http_routes": (
            handlers.get_router,
            "httpd.routers",
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

    def callbacks(self) -> dict[str, tuple[Callable[..., Any], str]]:
        """Get the callbacks."""

        async def _get_oap_store() -> OAPStore:
            return oap_store

        return {
            "oap_store": (
                _get_oap_store,
                StoreHookName,
            ),
        }
