import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from importlib import import_module
from types import ModuleType, TracebackType
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from dive_mcp_host.plugins.error import (
    PluginAlreadyRegisteredError,
    PluginError,
    PluginHookNameAlreadyRegisteredError,
    PluginHookNotFoundError,
    PluginLoadError,
)

logger = logging.getLogger(__name__)

type PlugInName = str
type HookPoint = str
type CallbackName = str


class PluginCallbackDef(BaseModel):
    """Plugin callback info.

    callback: str  # 掛載的 callback 名稱
    """

    hook_point: HookPoint
    callback: CallbackName

    model_config = ConfigDict(extra="allow")


class PluginDef(BaseModel):
    """Plugin 的定義.

    ex:
    name: "test"
    module: "this.is.module.name"
    config: {"key": "value"}
    callbacks: [
        {
            "hook_point": "on_start",
            "callback": "this.is.module.name",
            "configs": {"key": "value"}
        }
    ]
    """

    name: str
    module: str
    config: dict[str, Any]
    ctx_manager: CallbackName | None = None
    callbacks: list[PluginCallbackDef]


@dataclass
class HookInfo[**HOOK_PARAMS, HOOK_RET]:
    """Hook 的資訊."""

    hook_name: HookPoint
    register: Callable[
        [
            Callable[HOOK_PARAMS, Coroutine[Any, Any, HOOK_RET]],
            PluginCallbackDef,
            PlugInName,
        ],
        Coroutine[Any, Any, bool],
    ]


@dataclass
class LoadedPlugin[**HOOK_PARAMS, HOOK_RET]:
    """已經載入的 plugin."""

    name: str
    module: ModuleType
    config: dict[str, Any]
    callbacks: dict[
        HookPoint,
        tuple[Callable[HOOK_PARAMS, Coroutine[Any, Any, HOOK_RET]], PluginCallbackDef],
    ]
    info: PluginDef
    ctx_manager: Callable[[dict[str, Any]], AbstractAsyncContextManager[Any]] | None


@dataclass
class _RegistedHook:
    """儲存 Hook 的資訊."""

    hook_info: HookInfo[Any, Any]
    hooked_plugins: dict[str, LoadedPlugin] = field(default_factory=dict)


class PluginManager:
    """Plugin registry."""

    def __init__(self) -> None:
        """初始化."""
        self._hooks: dict[str, _RegistedHook] = {}
        self._plugins: dict[str, LoadedPlugin] = {}
        self._plugin_used: defaultdict[str, list[str]] = defaultdict(list)
        self._ctx_stack: AsyncExitStack = AsyncExitStack()

    def register_hookable[**P, R](self, hook_info: HookInfo[P, R]) -> None:
        """Register a hookable.

        Args:
            hookable_name: The name of the hookable.
            hook_info: The hook info.

        Registers a hookable point that plugins can attach to.

        When a plugin is loaded, it will automatically register its hooks to the
        corresponding hookable points if they exist.

        The hook_register function takes a hook function as a parameter and returns a
        boolean indicating whether the registration was successful.

        Type parameters:
            P: The parameters that the hook function accepts
            R: The return type of the hook function
        """
        if hook_info.hook_name in self._hooks:
            raise PluginHookNameAlreadyRegisteredError(
                f"Hook {hook_info.hook_name} already registered"
            )
        self._hooks[hook_info.hook_name] = _RegistedHook(hook_info=hook_info)

    async def __aenter__(self) -> Self:
        """Enter the context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> bool:
        """Exit the context."""
        await self._ctx_stack.aclose()
        return True

    async def register_plugin[T](
        self, plugin: PluginDef
    ) -> list[tuple[str, PluginError | None]]:
        """Register a plugin module."""
        plugin_name = plugin.name
        ret: list[tuple[str, PluginError | None]] = []
        if plugin_name in self._plugins:
            raise PluginAlreadyRegisteredError(
                f"Plugin {plugin_name} already registered"
            )

        loaded_plugin = _load_plugin(plugin)
        self._plugins[plugin_name] = loaded_plugin
        if loaded_plugin.ctx_manager:
            await self._ctx_stack.enter_async_context(
                loaded_plugin.ctx_manager(loaded_plugin.config)
            )

        for hook_point, (callback_func, hook_info) in loaded_plugin.callbacks.items():
            try:
                registered_hook = self._hooks[hook_point]
            except KeyError:
                logger.warning(
                    "Hook point %s not registered for plugin %s",
                    hook_point,
                    plugin_name,
                )
                ret.append(
                    (
                        hook_point,
                        PluginHookNotFoundError(
                            f"Hook point {hook_point} not registered"
                        ),
                    )
                )
                continue


            if await registered_hook.hook_info.register(
                callback_func, hook_info, plugin_name
            ):
                self._plugin_used[hook_point].append(plugin_name)
                registered_hook.hooked_plugins[plugin_name] = loaded_plugin
                ret.append((hook_point, None))
        return ret


def _load_plugin(plugin_info: PluginDef) -> LoadedPlugin:
    """Load a plugin module.

    Args:
        plugin_info: The plugin information.

    Returns:
        The loaded plugin.

    Raises:
        PluginLoadError: If the plugin cannot be loaded.
    """
    try:
        # Import the plugin module
        module = import_module(plugin_info.module)

        callbacks: dict[HookPoint, tuple[Callable[..., Any], PluginCallbackDef]] = {}
        for callback_def in plugin_info.callbacks:
            module_path, func_name = callback_def.callback.rsplit(".", 1)
            try:
                callback_module = import_module(module_path)
                callback_func = getattr(callback_module, func_name)
            except Exception as e:
                raise PluginLoadError(
                    f"Failed to load callback {callback_def.callback}: {e}"
                ) from e

            callbacks[callback_def.hook_point] = (callback_func, callback_def)
        ctx_manager = None
        if plugin_info.ctx_manager:
            module_path, func_name = plugin_info.ctx_manager.rsplit(".", 1)
            ctx_manager = getattr(import_module(module_path), func_name)

        return LoadedPlugin(
            name=plugin_info.name,
            module=module,
            config=plugin_info.config,
            callbacks=callbacks,
            info=plugin_info,
            ctx_manager=ctx_manager,
        )
    except Exception as e:
        raise PluginLoadError(f"Failed to load plugin {plugin_info.name}: {e}") from e
