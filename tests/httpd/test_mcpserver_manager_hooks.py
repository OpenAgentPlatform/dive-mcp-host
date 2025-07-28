import json
import tempfile
from collections.abc import Generator

import pytest

from dive_mcp_host.httpd.conf.mcp_servers import (
    Config,
    CurrentConfigHookName,
    MCPServerManager,
    UpdateAllConfigsHookName,
)
from dive_mcp_host.plugins.registry import PluginManager


@pytest.fixture
def mock_config_file() -> Generator[str, None, None]:
    """Create a mock configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        json.dump(
            {
                "mcpServers": {
                    "test_server": {
                        "transport": "stdio",
                        "enabled": True,
                        "command": "test_command",
                        "args": ["--test"],
                        "env": {"TEST_ENV": "test_value"},
                    }
                }
            },
            f,
        )
        f.flush()
        config_path = f.name
        yield config_path


@pytest.fixture
def plugin_manager() -> PluginManager:
    """Create a mock plugin manager."""
    return PluginManager()


@pytest.mark.asyncio
async def test_current_config_hook(
    mock_config_file: str, plugin_manager: PluginManager
) -> None:
    """Test the current_config hook functionality."""
    manager = MCPServerManager(mock_config_file)
    manager.initialize()

    # Register hooks
    manager.register_hook(plugin_manager)

    # Test current_config hook - modify existing server
    def modify_current_config(config: Config) -> Config:
        config.mcp_servers["test_server"].transport = "websocket"
        return config

    assert (
        manager.register_plugin(
            modify_current_config, CurrentConfigHookName, "test_plugin"
        )
        is True
    )

    # Verify current_config hook is applied
    current_config = await manager.get_current_config()
    assert current_config is not None
    assert current_config.mcp_servers["test_server"].transport == "websocket"

    # Verify self._current_config is not modified
    assert manager._current_config is not None
    assert manager._current_config.mcp_servers["test_server"].transport == "stdio"

    # Test current_config hook - add new server
    def add_new_server(config: Config) -> Config:
        config.mcp_servers["new_server"] = config.mcp_servers[
            "test_server"
        ].model_copy()
        config.mcp_servers["new_server"].transport = "sse"
        return config

    assert (
        manager.register_plugin(add_new_server, CurrentConfigHookName, "test_plugin")
        is True
    )

    # Verify new server is added in current_config
    current_config = await manager.get_current_config()
    assert current_config is not None
    assert "new_server" in current_config.mcp_servers
    assert current_config.mcp_servers["new_server"].transport == "sse"

    # Verify self._current_config is not modified
    assert manager._current_config is not None
    assert "new_server" not in manager._current_config.mcp_servers

    # Test current_config hook - remove server
    def remove_server(config: Config) -> Config:
        del config.mcp_servers["test_server"]
        return config

    assert (
        manager.register_plugin(remove_server, CurrentConfigHookName, "test_plugin")
        is True
    )

    # Verify server is removed in current_config
    current_config = await manager.get_current_config()
    assert current_config is not None
    assert "test_server" not in current_config.mcp_servers
    assert "new_server" in current_config.mcp_servers

    # Verify self._current_config is not modified
    assert manager._current_config is not None
    assert "test_server" in manager._current_config.mcp_servers


@pytest.mark.asyncio
async def test_update_all_configs_hook(
    mock_config_file: str, plugin_manager: PluginManager
) -> None:
    """Test the update_all_configs hook functionality."""
    manager = MCPServerManager(mock_config_file)
    manager.initialize()

    # Register hooks
    manager.register_hook(plugin_manager)

    # Test update_all_configs hook - modify server
    def modify_update_config(config: Config) -> Config:
        config.mcp_servers["new_server"].enabled = False
        return config

    assert (
        manager.register_plugin(
            modify_update_config, UpdateAllConfigsHookName, "test_plugin"
        )
        is True
    )

    # Create new config and add new server before updating
    new_config = await manager.get_current_config()
    assert new_config is not None
    new_config.mcp_servers["new_server"] = new_config.mcp_servers[
        "test_server"
    ].model_copy()
    new_config.mcp_servers["new_server"].transport = "sse"

    original_config = new_config.model_copy()
    assert await manager.update_all_configs(new_config) is True

    # Verify update_all_configs hook is applied
    updated_config = await manager.get_current_config()
    assert updated_config is not None
    assert updated_config.mcp_servers["new_server"].enabled is False

    # Verify original config is not modified
    assert original_config.mcp_servers["new_server"].enabled is True

    # Test update_all_configs hook - add and remove servers
    def modify_servers(config: Config) -> Config:
        # Add a new server
        config.mcp_servers["another_server"] = config.mcp_servers[
            "new_server"
        ].model_copy()
        config.mcp_servers["another_server"].transport = "websocket"
        # Remove existing server
        del config.mcp_servers["new_server"]
        return config

    assert (
        manager.register_plugin(modify_servers, UpdateAllConfigsHookName, "test_plugin")
        is True
    )

    # Update config and verify changes
    new_config = await manager.get_current_config()
    assert new_config is not None
    original_config = new_config.model_copy()
    assert await manager.update_all_configs(new_config) is True

    updated_config = await manager.get_current_config()
    assert updated_config is not None
    assert "new_server" not in updated_config.mcp_servers
    assert "another_server" in updated_config.mcp_servers
    assert updated_config.mcp_servers["another_server"].transport == "websocket"

    # Verify original config is not modified
    assert "new_server" in original_config.mcp_servers
    assert "another_server" not in original_config.mcp_servers

    # Test invalid hook name
    assert (
        manager.register_plugin(modify_update_config, "invalid_hook", "test_plugin")
        is False
    )
