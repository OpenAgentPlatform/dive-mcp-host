import pytest
import pytest_asyncio

from dive_mcp_host.plugins.registry import (
    HookInfo,
    PluginCallbackDef,
    PluginDef,
    PluginError,
    PluginHookNameAlreadyRegisteredError,
    PluginHookNotFoundError,
    PluginManager,
)


# Test hook functions
async def callback_func1(arg1: str) -> str:
    """Test callback function 1."""
    return f"callback1: {arg1}"


async def callback_func2(arg1: str) -> str:
    """Test callback function 2."""
    return f"callback2: {arg1}"


@pytest_asyncio.fixture
async def plugin_manager_ctx():
    """Plugin manager context."""
    async with PluginManager() as manager:
        yield manager


@pytest.mark.asyncio
async def test_plugin_manager(plugin_manager_ctx: PluginManager):
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

    # Register hooks
    plugin_manager_ctx.register_hookable(hook_info1)
    plugin_manager_ctx.register_hookable(hook_info2)

    # Test duplicate hook registration
    with pytest.raises(PluginHookNameAlreadyRegisteredError):
        plugin_manager_ctx.register_hookable(hook_info1)

    # Create plugin definitions
    plugin1 = PluginDef(
        name="test_plugin1",
        module="tests.plugin.test_registry",
        config={"key1": "value1"},
        callbacks=[
            PluginCallbackDef(
                hook_point="hook1", callback="tests.plugin.test_registry.callback_func1"
            ),
            PluginCallbackDef(
                hook_point="hook2", callback="tests.plugin.test_registry.callback_func2"
            ),
        ],
    )

    # Test plugin registration
    results1 = await plugin_manager_ctx.register_plugin(plugin1)
    assert len(results1) == 2
    assert all(result[1] is None for result in results1)  # No errors

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
        await plugin_manager_ctx.register_plugin(plugin1)

    # Test plugin with non-existent hook
    plugin2 = PluginDef(
        name="test_plugin2",
        module="tests.plugin.test_registry",
        config={"key2": "value2"},
        callbacks=[
            PluginCallbackDef(
                hook_point="hookx", callback="tests.plugin.test_registry.callback_func2"
            ),
        ],
    )
    results2 = await plugin_manager_ctx.register_plugin(plugin2)
    assert len(results2) == 1
    assert results2[0][0] == "hookx"
    assert isinstance(results2[0][1], PluginHookNotFoundError)
