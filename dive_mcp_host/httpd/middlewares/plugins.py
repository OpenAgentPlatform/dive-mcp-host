from collections.abc import Callable, Coroutine
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from dive_mcp_host.plugins.registry import HookInfo, PluginCallbackDef, PluginManager

type MiddlewareCallback = Callable[
    [Request, Callable[[Request], Coroutine[Any, Any, Response]]],
    Coroutine[Any, Any, Response],
]
type MiddlewareHook = HookInfo[
    [Request, MiddlewareCallback],
    Response,
]


class PluginMiddlewaresManager:
    """The middleware that chains the plugins."""

    def __init__(self) -> None:
        """Initialize the middleware."""
        self.plugins: list[tuple[MiddlewareCallback, PluginCallbackDef, str]] = []

    async def dispatch(
        self, request: Request, call_next: Callable, idx: int = 0
    ) -> Response:
        """The middleware entry point."""
        if idx >= len(self.plugins):
            return await call_next(request)
        callback, _, _ = self.plugins[idx]
        return await callback(request, lambda r: self.dispatch(r, call_next, idx + 1))

    async def register_plugin(
        self,
        callback: MiddlewareCallback,
        callback_def: PluginCallbackDef,
        plugin_name: str,
    ) -> bool:
        """Callback used to register plugin."""
        self.plugins.append((callback, callback_def, plugin_name))
        return True

    def register_hook(self, manager: PluginManager) -> None:
        """Register the hook."""
        manager.register_hookable(
            HookInfo(
                hook_name="httpd.middlewares",
                register=self.register_plugin,
            )
        )
