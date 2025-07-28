from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter

from dive_mcp_host.plugins.registry import HookInfo, PluginManager


class RouterPlugin:
    """The plugin that registers a router."""

    _routers: list[tuple[APIRouter, str]]

    def __init__(self, api_router: APIRouter) -> None:
        """Initialize the router plugin."""
        self._routers = []
        self._api_router = api_router

    async def register_plugin(
        self,
        _callback: Callable[..., Coroutine[Any, Any, Any]],
        _hook_name: str,
        _plugin_name: str,
    ) -> bool:
        """Callback used to register plugin."""
        return True

    def static_register_plugin(
        self,
        callback: Callable[..., Any],
        _hook_name: str,
        plugin_name: str,
    ) -> bool:
        """Callback used to register plugin."""
        self._api_router.include_router(callback(), prefix=f"/{plugin_name}")
        self._routers.append((callback(), plugin_name))
        return True

    def register_hook(self, manager: PluginManager) -> None:
        """Register the hook."""
        manager.register_hookable(
            HookInfo(
                hook_name="httpd.routers",
                register=self.register_plugin,
                static_register=self.static_register_plugin,
            )
        )
