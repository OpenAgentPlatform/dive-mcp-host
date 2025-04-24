from contextlib import AsyncExitStack
from typing import Any

import pytest

from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.plugins.registry import (
    Callbacks,
    HookInfo,
    PluginCallbackDef,
    PluginDef,
    PluginError,
    PluginHookNameAlreadyRegisteredError,
    PluginManager,
)


# Test hook functions
async def callback_func1(arg1: str) -> str:
    """Test callback function 1."""
    return f"callback1: {arg1}"


async def callback_func2(arg1: str) -> str:
    """Test callback function 2."""
    return f"callback2: {arg1}"


class PluginA(ContextProtocol):
    """Test plugin A."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize method."""
        self.config = config

    def callbacks(self) -> Callbacks:
        """Test callbacks."""
        return {
            "hook1": (
                callback_func1,
                PluginCallbackDef(hook_point="hook1", callback="callback_func1"),
            ),
            "hook2": (
                callback_func2,
                PluginCallbackDef(hook_point="hook2", callback="callback_func2"),
            ),
        }


class PluginB(PluginA):
    """Test plugin B."""

    def callbacks(self) -> Callbacks:
        """Test callbacks."""
        return {
            "hookx": (
                callback_func2,
                PluginCallbackDef(hook_point="hookx", callback="callback_func2"),
            ),
        }


@pytest.mark.asyncio
async def test_plugin_manager():
    """Create plugin manager."""
    # Create a registry to track registered hookers
    callback_registry = {}
    hooks = []

    async def register_hook(
        callback_func, callback_info: PluginCallbackDef, plugin_name
    ):
        callback_registry[f"{plugin_name}.{callback_info.hook_point}"] = callback_func
        return True

    # Test hook registration
    hook_info1 = HookInfo(
        hook_name="hook1",
        register=register_hook,
    )
    hook_info2 = HookInfo(
        hook_name="hook2",
        register=register_hook,
    )
    hooks.append(hook_info1)
    hooks.append(hook_info2)

    exit_stack = AsyncExitStack()
    plugin_manager_ctx = PluginManager()

    # Create plugin definitions
    plugin1 = PluginDef(
        name="test_plugin1",
        module="tests.plugin.test_registry",
        config={"key1": "value1"},
        ctx_manager="tests.plugin.test_registry.PluginA",
    )
    plugin_manager_ctx.register_plugin(plugin1)

    # Register hooks
    plugin_manager_ctx.register_hookable(hook_info1)
    plugin_manager_ctx.register_hookable(hook_info2)

    # Test duplicate hook registration
    with pytest.raises(PluginHookNameAlreadyRegisteredError):
        plugin_manager_ctx.register_hookable(hook_info1)

    await exit_stack.enter_async_context(plugin_manager_ctx)

    # Test hooker execution
    test_input = "test_input"
    assert (
        await callback_registry["test_plugin1.hook1"](test_input)
        == f"callback1: {test_input}"
    )
    assert (
        await callback_registry["test_plugin1.hook2"](test_input)
        == f"callback2: {test_input}"
    )
    # Test duplicate plugin registration
    with pytest.raises(PluginError):
        plugin_manager_ctx.register_plugin(plugin1)

    # Test plugin with non-existent hook
    plugin2 = PluginDef(
        name="test_plugin2",
        module="tests.plugin.test_registry",
        config={"key2": "value2"},
        ctx_manager="tests.plugin.test_registry.PluginB",
    )
    plugin_manager_ctx.register_plugin(plugin2)
