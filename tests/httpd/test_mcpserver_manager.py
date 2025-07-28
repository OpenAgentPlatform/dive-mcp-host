import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from pydantic import SecretStr

from dive_mcp_host.httpd.conf.mcp_servers import (
    Config,
    MCPServerConfig,
    MCPServerManager,
)

# Register custom mark
integration = pytest.mark.integration


# Unit tests
class TestMCPServerManager:
    """Unit tests for MCPServerManager class's basic functionality."""

    @pytest.fixture
    def mock_config_file(self):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "mcpServers": {
                        "test_server": {
                            # Although the transport setting has been changed to stdio,
                            # we keep "command" here for compatibility.
                            "transport": "command",
                            "enabled": True,
                            "command": "test_command",
                            "args": ["--test"],
                            "env": {"TEST_ENV": "test_value"},
                        },
                        "disabled_server": {
                            "transport": "sse",
                            "enabled": False,
                            "url": "http://test.url",
                            "headers": {"Authorization": "bearer token"},
                        },
                    },
                },
                f,
            )
            config_path = f.name
        yield config_path
        # Clean up after test
        Path(config_path).unlink()

    def test_config_path_setting(self):
        """Test if the configuration path is set correctly."""
        test_path = "/test/path.json"
        manager = MCPServerManager(test_path)
        assert manager.config_path == test_path

    @pytest.mark.asyncio
    async def test_get_config(self, mock_config_file):
        """Test retrieving MCP server configuration."""
        manager = MCPServerManager(mock_config_file)
        manager.initialize()
        config = await manager.get_current_config()
        assert config is not None
        assert "test_server" in config.mcp_servers
        assert config.mcp_servers["test_server"].transport == "stdio"

    @pytest.mark.asyncio
    async def test_initialize(self, mock_config_file):
        """Test initializing the manager."""
        manager = MCPServerManager(mock_config_file)
        manager.initialize()

        config = await manager.get_current_config()
        assert config is not None
        assert "test_server" in config.mcp_servers
        assert config.mcp_servers["test_server"].enabled is True
        assert config.mcp_servers["disabled_server"].enabled is False

    @pytest.mark.asyncio
    async def test_get_enabled_servers(self, mock_config_file):
        """Test getting enabled servers."""
        manager = MCPServerManager(mock_config_file)
        manager.initialize()
        enabled_servers = await manager.get_enabled_servers()

        assert enabled_servers is not None
        assert len(enabled_servers) == 1
        assert "test_server" in enabled_servers
        assert "disabled_server" not in enabled_servers

    @pytest.mark.asyncio
    async def test_multiple_instances(self, mock_config_file):
        """Test that multiple instances can be created with different configs."""
        manager1 = MCPServerManager(mock_config_file)
        manager1.initialize()

        # Create a second config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "mcpServers": {
                        "second_server": {
                            "transport": "sse",
                            "enabled": True,
                            "url": "http://second.url",
                        },
                    },
                },
                f,
            )
            second_config_path = f.name

        try:
            manager2 = MCPServerManager(second_config_path)
            manager2.initialize()

            # Verify managers have different configs
            config1 = await manager1.get_current_config()
            config2 = await manager2.get_current_config()
            assert config1 is not None
            assert config2 is not None
            assert "test_server" in config1.mcp_servers
            assert "second_server" in config2.mcp_servers
            assert "second_server" not in config1.mcp_servers
            assert "test_server" not in config2.mcp_servers

            # Verify they are different instances
            assert manager1 is not manager2
        finally:
            # Clean up
            Path(second_config_path).unlink()

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.mcp_servers.json.dump")
    async def test_update_all_configs(self, mock_json_dump, mock_config_file):
        """Test updating all configurations."""
        # Mock json.dump to avoid writing to file
        mock_json_dump.return_value = None

        manager = MCPServerManager(mock_config_file)
        manager.initialize()

        # Create new configuration
        new_config = Config(
            mcpServers={
                "new_server": MCPServerConfig(
                    transport="websocket",
                    enabled=True,
                    url="ws://new.url",
                    headers={"Authorization": SecretStr("bearer token2")},
                ),
            },
        )

        # Update all configurations
        result = await manager.update_all_configs(new_config)
        assert result is True

        # Manager's current_config should be updated
        config = await manager.get_current_config()
        assert config is not None
        assert len(config.mcp_servers) == 1
        assert "new_server" in config.mcp_servers
        assert config.mcp_servers["new_server"].transport == "websocket"
        assert config.mcp_servers["new_server"].url == "ws://new.url"
        assert config.mcp_servers["new_server"].headers
        assert config.mcp_servers["new_server"].headers.get(
            "Authorization"
        ) == SecretStr("bearer token2")


# Integration tests
@pytest.mark.integration
class TestMCPServerManagerIntegration:
    """Integration tests for the complete functionality of MCPServerManager class."""

    @pytest_asyncio.fixture
    async def test_config_path(self):
        """Set up the test configuration file path."""
        # Use a temporary directory to create the config file
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            test_config = {
                "mcpServers": {
                    "test_server": {
                        "transport": "command",
                        "enabled": True,
                        "command": "echo",
                        "args": ["Hello, MCP!"],
                        "env": {},
                    },
                },
            }
            with config_path.open("w") as f:
                json.dump(test_config, f)
            yield str(config_path)

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.mcp_servers.json.dump")
    async def test_full_config_workflow(self, mock_json_dump, test_config_path):
        """Test the complete server configuration workflow."""
        # Mock json.dump to avoid serialization issues
        mock_json_dump.return_value = None

        # Set up MCPServerManager instance
        manager = MCPServerManager(test_config_path)

        # Initialize the manager
        result = manager.initialize()
        config = await manager.get_current_config()
        assert config is not None
        assert config is not None
        assert "test_server" in config.mcp_servers

        # Get enabled servers
        enabled_servers = await manager.get_enabled_servers()
        assert "test_server" in enabled_servers

        # Create a new configuration with additional server
        current_servers = config.mcp_servers.copy()
        current_servers["new_server"] = MCPServerConfig(
            transport="sse",
            enabled=True,
            url="http://new.server",
            headers={"Authorization": SecretStr("bearer token2")},
        )

        # Update config with new server
        new_config = Config(mcpServers=current_servers)
        result = await manager.update_all_configs(new_config)
        assert result is True

        # Verify new server was added in current_config
        config = await manager.get_current_config()
        assert config is not None
        assert "new_server" in config.mcp_servers
        assert config.mcp_servers["new_server"].transport == "sse"
        assert config.mcp_servers["new_server"].url == "http://new.server"
        assert config.mcp_servers["new_server"].headers
        assert config.mcp_servers["new_server"].headers.get(
            "Authorization"
        ) == SecretStr("bearer token2")

        # Test environment variable config
        # Setting in environment variable has higher priority than file config
        with patch.dict(
            os.environ, {"DIVE_MCP_CONFIG_CONTENT": json.dumps({"mcpServers": {}})}
        ):
            env_manager = MCPServerManager()
            env_manager.initialize()
            config = await env_manager.get_current_config()
            assert config is not None
            assert len(config.mcp_servers) == 0


def test_mcp_server_config_validation():
    """Test the validation of MCP server configuration."""
    content = """{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/tmp"
            ],
            "enabled": true
        },
        "yt-dlp": {
            "command": "np",
            "transport": "sse",
            "headers": {
                "Authorization": "bearer token"
            },
            "url": "http://localhost:8080/sse",
            "args": [
                "-y",
                "@someone/some-package"
            ],
            "enabled": false,
            "env": {
                "NODE_ENV": "production"
            }
        }
    }
}
"""
    config = Config(**json.loads(content))
    # The default transport is stdio
    assert config.mcp_servers["filesystem"].transport == "stdio"
    assert config.mcp_servers["filesystem"].enabled is True
    assert config.mcp_servers["filesystem"].command == "npx"
    assert config.mcp_servers["filesystem"].args == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/tmp",  # noqa: S108
    ]
    assert config.mcp_servers["filesystem"].env == {}
    assert config.mcp_servers["yt-dlp"].transport == "sse"
    assert config.mcp_servers["yt-dlp"].headers == {
        "Authorization": SecretStr("bearer token")
    }
    assert config.mcp_servers["yt-dlp"].enabled is False
    assert config.mcp_servers["yt-dlp"].command == "np"
    assert config.mcp_servers["yt-dlp"].url == "http://localhost:8080/sse"
    assert config.mcp_servers["yt-dlp"].args == [
        "-y",
        "@someone/some-package",
    ]
    assert config.mcp_servers["yt-dlp"].env == {"NODE_ENV": "production"}
