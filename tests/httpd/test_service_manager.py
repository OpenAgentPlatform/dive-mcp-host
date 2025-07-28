import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from pydantic import AnyUrl

from dive_mcp_host.env import RESOURCE_DIR
from dive_mcp_host.httpd.conf.httpd_service import ServiceManager

# Register custom mark
integration = pytest.mark.integration


# Unit tests
class TestServiceManager:
    """Unit tests for ServiceManager class's basic functionality."""

    @pytest.fixture
    def mock_config_file(self):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "db": {
                        "uri": "sqlite:///test.sqlite",
                        "async_uri": "sqlite+aiosqlite:///test.sqlite",
                        "pool_size": 3,
                        "echo": True,
                    },
                    "checkpointer": {
                        "uri": "sqlite:///checkpoints.sqlite",
                    },
                    "local_file_cache_prefix": "test_prefix",
                    "resource_dir": "/test/resource",
                    "config_location": {
                        "mcp_server_config_path": "/test/mcp_config.json",
                        "model_config_path": "/test/model_config.json",
                        "prompt_config_path": "/test/prompt_config.json",
                        "command_alias_config_path": "/test/command_alias.json",
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
        manager = ServiceManager(test_path)
        assert manager.config_path == test_path

    def test_initialize(self, mock_config_file):
        """Test initializing the manager."""
        manager = ServiceManager(mock_config_file)
        result = manager.initialize()

        assert result is True
        assert manager.current_setting is not None
        assert manager.current_setting.db.uri == "sqlite:///test.sqlite"
        assert manager.current_setting.db.pool_size == 3
        assert manager.current_setting.db.echo is True
        assert manager.current_setting.checkpointer.uri == AnyUrl(
            "sqlite:///checkpoints.sqlite"
        )
        assert manager.current_setting.resource_dir == Path("/test/resource")
        assert manager.current_setting.local_file_cache_prefix == "test_prefix"
        assert (
            manager.current_setting.config_location.mcp_server_config_path
            == "/test/mcp_config.json"
        )

    def test_default_config_path(self):
        """Test the default configuration path."""
        manager = ServiceManager()
        assert manager.config_path == str(Path.cwd() / "dive_httpd.json")

    def test_initialize_with_missing_config_file(self):
        """Test initializing with a missing configuration file."""
        manager = ServiceManager("/nonexistent/file.json")

        with pytest.raises(FileNotFoundError):
            manager.initialize()


# Integration tests
@pytest.mark.integration
class TestServiceManagerIntegration:
    """Integration tests for the complete functionality of ServiceManager class."""

    @pytest_asyncio.fixture
    async def test_config_path(self):
        """Set up the test configuration file path."""
        # Use a temporary directory to create the config file
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            test_config = {
                "db": {
                    "uri": "sqlite:///integration_test.sqlite",
                    "async_uri": "sqlite+aiosqlite:///integration_test.sqlite",
                    "pool_size": 10,
                },
                "checkpointer": {
                    "uri": "sqlite:///checkpoints.sqlite",
                },
                "local_file_cache_prefix": "integration_test",
                "resource_dir": "/test/resource",
                "config_location": {
                    "mcp_server_config_path": str(Path(tmp_dir) / "mcp_config.json"),
                },
            }
            with config_path.open("w") as f:
                json.dump(test_config, f)
            yield str(config_path)

    @pytest.mark.asyncio
    async def test_full_service_manager_workflow(self, test_config_path):
        """Test the complete service configuration workflow."""
        # Set up ServiceManager instance
        manager = ServiceManager(test_config_path)

        # Initialize the manager
        result = manager.initialize()
        assert result is True
        assert manager.current_setting is not None

        # Verify the configuration details
        assert manager.current_setting.db.uri == "sqlite:///integration_test.sqlite"
        assert manager.current_setting.db.pool_size == 10
        assert manager.current_setting.checkpointer.uri == AnyUrl(
            "sqlite:///checkpoints.sqlite"
        )
        assert manager.current_setting.resource_dir == Path("/test/resource")

        # Test that default values are properly applied for unspecified options
        assert manager.current_setting.db.echo is False
        assert manager.current_setting.db.pool_recycle == 60

    @pytest.mark.asyncio
    async def test_environment_variable_config(self):
        """Test that environment variable configuration works."""
        test_config = {
            "db": {
                "uri": "sqlite:///env_var_test.sqlite",
                "async_uri": "sqlite+aiosqlite:///env_var_test.sqlite",
            },
            "checkpointer": {
                "uri": "sqlite:///checkpoints.sqlite",
            },
            "resource_dir": str(RESOURCE_DIR),
        }

        # Setting environment variable config
        with patch.dict(
            os.environ, {"DIVE_SERVICE_CONFIG_CONTENT": json.dumps(test_config)}
        ):
            env_manager = ServiceManager()
            env_manager.initialize()

            assert env_manager.current_setting is not None
            assert env_manager.current_setting.db.uri == "sqlite:///env_var_test.sqlite"
            assert env_manager.current_setting.resource_dir == RESOURCE_DIR

    @pytest.mark.asyncio
    async def test_environment_config_precedence(self, test_config_path):
        """Test that environment variable config takes precedence over file config."""
        env_config = {
            "db": {
                "uri": "sqlite:///env_precedence.sqlite",
                "async_uri": "sqlite+aiosqlite:///env_precedence.sqlite",
            },
            "checkpointer": {
                "uri": "sqlite:///checkpoints.sqlite",
            },
            "resource_dir": str(RESOURCE_DIR),
        }

        # Setting environment variable config
        with patch.dict(
            os.environ, {"DIVE_SERVICE_CONFIG_CONTENT": json.dumps(env_config)}
        ):
            # Create manager with file path, but env var should take precedence
            manager = ServiceManager(test_config_path)
            manager.initialize()

            assert manager.current_setting is not None
            assert manager.current_setting.db.uri == "sqlite:///env_precedence.sqlite"
            assert manager.current_setting.resource_dir == RESOURCE_DIR
