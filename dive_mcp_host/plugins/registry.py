import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, RootModel

from dive_mcp_host.host.helpers.context import ContextProtocol
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
    """Plugin callback definition.

    Attributes:
        callback: The name of the callback function to be hooked
        hook_point: The hook point where this callback should be registered
    """

    hook_point: HookPoint
    callback: CallbackName

    model_config = ConfigDict(extra="allow")


class PluginDef(BaseModel):
    """Plugin definition.

    ex:
    name: "test"
    module: "this.is.module.name"
    config: {"key": "value"}
    ctx_manager: "this.is.module.name.ctx_manager"
    static_callbacks: "this.is.module.name.get_static_callbacks"

    Attributes:
        name: The name of the plugin.
        module: The module name of the plugin.
        config: The configuration of the plugin.
        ctx_manager: The context manager of the plugin.
        static_callbacks: The function name to retrieve static callbacks.
            The return format is the same as LoadedPlugin.static_callbacks.
    """

    name: str
    module: str
    config: dict[str, Any]
    ctx_manager: CallbackName
    static_callbacks: CallbackName | None = None


@dataclass
class HookInfo[**HOOK_PARAMS, HOOK_RET]:
    """Information about a hook.

    Defines the hook name and registration functions for both async and sync callbacks.
    """

    hook_name: HookPoint
    register: Callable[
        [
            Callable[HOOK_PARAMS, Coroutine[Any, Any, HOOK_RET]],
            PluginCallbackDef,
            PlugInName,
        ],
        Coroutine[Any, Any, bool],
    ]
    static_register: (
        Callable[
            [
                Callable[HOOK_PARAMS, HOOK_RET],
                PluginCallbackDef,
                PlugInName,
            ],
            bool,
        ]
        | None
    ) = None


type Callbacks = dict[
    HookPoint,
    tuple[Callable[..., Coroutine[Any, Any, Any]], PluginCallbackDef],
]


class CtxManager(ContextProtocol, Protocol):
    """Context manager for plugins.

    Overwrite _run_in_context to control the context of the plugin.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize method."""
        ...

    def callbacks(self) -> Callbacks:
        """Get the callbacks."""
        ...


@dataclass
class LoadedPlugin:
    """Loaded plugin information.

    Contains the loaded plugin module, configuration, and callback information.
    Stores both the context manager factory and any static callbacks registered
    by the plugin.
    """

    name: str
    module: ModuleType
    config: dict[str, Any]
    info: PluginDef
    ctx_manager: Callable[[dict[str, Any]], CtxManager] | None
    static_callbacks: dict[str, tuple[Callable[..., Any], PluginCallbackDef]] = field(
        default_factory=dict
    )


@dataclass
class _RegistedHook:
    """Stores information about a registered hook.

    Contains the hook information and a mapping of plugin names to loaded plugin
    instances that have registered with this hook.
    """

    hook_info: HookInfo[Any, Any]
    hooked_plugins: dict[str, LoadedPlugin] = field(default_factory=dict)


class PluginManager:
    """Plugin registry."""

    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self._hooks: dict[str, _RegistedHook] = {}
        self._plugins: dict[str, LoadedPlugin] = {}
        self._plugin_used: defaultdict[str, list[str]] = defaultdict(list)
        self._ctx_stack: AsyncExitStack = AsyncExitStack()
        self._plugin_state: list[tuple[str, PluginError | None]] = []

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
        """Enter the context.

        It will enter the context of the plugin and register the callbacks.
        """
        for plugin_name, loaded_plugin in self._plugins.items():
            if loaded_plugin.ctx_manager:
                ctx_manager = loaded_plugin.ctx_manager(loaded_plugin.config)
                await self._ctx_stack.enter_async_context(ctx_manager)

            callbacks = ctx_manager.callbacks()
            for _, (
                callback_func,
                hook_info,
            ) in callbacks.items():
                hook_point = hook_info.hook_point
                try:
                    registered_hook = self._hooks[hook_point]
                except KeyError:
                    logger.warning(
                        "Hook point %s not registered for plugin %s",
                        hook_point,
                        plugin_name,
                    )
                    self._plugin_state.append(
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
                    self._plugin_state.append((hook_point, None))
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

    def register_plugin[T](self, plugin: PluginDef) -> None:
        """Register a plugin module.

        It will load the plugin module and register the static callbacks.
        Only static callbacks are registered into hooks at this point. Other callbacks
        will be registered when entering the PluginManager context.

        Args:
            plugin: The plugin information.

        Raises:
            PluginAlreadyRegisteredError: If the plugin is already registered.
        """
        plugin_name = plugin.name
        if plugin_name in self._plugins:
            raise PluginAlreadyRegisteredError(
                f"Plugin {plugin_name} already registered"
            )

        loaded_plugin = _load_plugin(plugin)
        self._plugins[plugin_name] = loaded_plugin

        if loaded_plugin.static_callbacks is None:
            return

        for _, (
            callback_func,
            hook_info,
        ) in loaded_plugin.static_callbacks.items():
            registered_hook = self._hooks.get(hook_info.hook_point)
            if registered_hook and registered_hook.hook_info.static_register:
                registered_hook.hook_info.static_register(
                    callback_func, hook_info, plugin_name
                )


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

        module_path, func_name = plugin_info.ctx_manager.rsplit(".", 1)
        ctx_manager = getattr(import_module(module_path), func_name)

        if plugin_info.static_callbacks:
            module_path, func_name = plugin_info.static_callbacks.rsplit(".", 1)
            static_callbacks = getattr(import_module(module_path), func_name)()
        else:
            static_callbacks = None

        return LoadedPlugin(
            name=plugin_info.name,
            module=module,
            config=plugin_info.config,
            info=plugin_info,
            ctx_manager=ctx_manager,
            static_callbacks=static_callbacks,  # type: ignore
        )
    except Exception as e:
        raise PluginLoadError(f"Failed to load plugin {plugin_info.name}: {e}") from e


def load_plugins_config(path: str | None) -> list[PluginDef]:
    """Load the plugins config from the given path.

    Args:
        path: The path to the plugins config.

    Returns:
        The plugins config.
    """
    if path is None:
        return []

    class PluginDefList(RootModel):
        root: list[PluginDef]

    with Path(path).open(encoding="utf-8") as f:
        try:
            return PluginDefList.model_validate_json(f.read()).root
        except:
            logger.exception("Failed to load plugins config from %s", path)
            raise
