import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from pydantic import SecretStr

from dive_mcp_host.host.conf.llm import LLMAnthropicConfig
from dive_mcp_host.httpd.conf.models import ModelManager

# Register custom mark
integration = pytest.mark.integration


# Unit tests
class TestModelManager:
    """Unit tests for ModelManager class's basic functionality."""

    @pytest.fixture
    def mock_config_file(self):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "activeProvider": "test_provider",
                    "enableTools": True,
                    "configs": {
                        "test_provider": {
                            "modelProvider": "test_provider",
                            "model": "test_model",
                            "apiKey": "test_key",
                            "configuration": {"baseURL": "http://test.url"},
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
        manager = ModelManager(test_path)
        assert manager.config_path == test_path

    def test_multiple_instances(self):
        """Test that multiple instances can be created with different configurations."""
        path1 = "/test/path1.json"
        path2 = "/test/path2.json"

        manager1 = ModelManager(path1)
        manager2 = ModelManager(path2)

        assert manager1 is not manager2
        assert manager1.config_path == path1
        assert manager2.config_path == path2

    @pytest.mark.asyncio
    async def test_initialize(self, mock_config_file):
        """Test initializing the manager."""
        manager = ModelManager(mock_config_file)
        result = manager.initialize()

        assert result is True
        assert manager.current_setting is not None
        assert manager.current_setting.model == "test_model"
        assert manager.current_setting.model_provider == "test_provider"

    @pytest.mark.asyncio
    async def test_get_active_settings(self, mock_config_file):
        """Test getting the active model settings."""
        manager = ModelManager(mock_config_file)
        manager.initialize()
        settings = manager.current_setting

        assert settings is not None
        assert settings.model == "test_model"
        assert settings.model_provider == "test_provider"

    @pytest.mark.asyncio
    async def test_parse_settings(self, mock_config_file):
        """Test parsing model settings."""
        manager = ModelManager(mock_config_file)
        r = manager.initialize()
        if not r:
            pytest.skip("Configuration not available")

        settings = manager.get_settings_by_provider("test_provider")

        assert settings is not None
        assert settings.model == "test_model"  # type: ignore
        assert settings.model_provider == "test_provider"  # type: ignore
        assert settings.api_key == SecretStr("test_key")  # type: ignore
        assert settings.configuration is not None  # type: ignore
        assert settings.configuration.base_url == "http://test.url"  # type: ignore

    @pytest.mark.asyncio
    async def test_get_settings_by_provider(self, mock_config_file):
        """Test getting model settings by specific provider."""
        manager = ModelManager(mock_config_file)
        manager.initialize()
        # Test existing provider
        settings = manager.get_settings_by_provider("test_provider")
        assert settings is not None
        assert settings.model == "test_model"  # type: ignore
        assert settings.model_provider == "test_provider"  # type: ignore
        assert settings.api_key == SecretStr("test_key")  # type: ignore
        assert settings.configuration is not None  # type: ignore
        assert settings.configuration.base_url == "http://test.url"  # type: ignore

        # Test non-existing provider
        non_existing_settings = manager.get_settings_by_provider(
            "non_existing_provider"
        )
        assert non_existing_settings is None


# Integration tests
@pytest.mark.integration
class TestModelManagerIntegration:
    """Integration tests for the complete functionality of ModelManager class."""

    @pytest_asyncio.fixture
    async def test_config_path(self):
        """Set up the test configuration file path."""
        # Use a temporary directory to create the config file
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "modelConfig.json"
            test_config = {
                "activeProvider": "fake",
                "enableTools": True,
                "configs": {
                    "fake": {
                        "modelProvider": "fake",
                        "model": "fake-model",
                        "apiKey": "",
                        "configuration": {"baseURL": ""},
                    },
                },
            }
            with config_path.open("w") as f:
                json.dump(test_config, f)
            yield str(config_path)

    @pytest.mark.asyncio
    @patch("dive_mcp_host.httpd.conf.models.json.dump")
    async def test_full_model_workflow(self, mock_json_dump, test_config_path):
        """Test the complete model configuration, initialization, and usage workflow."""
        # Mock json.dump to avoid serialization issues
        mock_json_dump.return_value = None

        # Set up ModelManager instance
        manager = ModelManager(test_config_path)

        # Initialize the manager
        result = manager.initialize()
        assert result is True
        assert manager.current_setting is not None
        assert manager.current_setting.model == "fake-model"
        assert manager.current_setting.model_provider == "fake"


def test_anthropic_model():
    """Test max tokens setting for Anthropic Model."""

    @contextmanager
    def tmp_config(model_name):
        """Create a mock configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "activeProvider": "anthropic",
                    "enableTools": True,
                    "configs": {
                        "anthropic": {
                            "modelProvider": "anthropic",
                            "model": model_name,
                            "apiKey": "test_key",
                        },
                    },
                },
                f,
            )
            config_path = f.name
        try:
            yield config_path
        finally:
            # Clean up after test
            Path(config_path).unlink()

    with tmp_config("claude-3-7-sonnet-20250219") as anthropic_config:
        manager = ModelManager(anthropic_config)

        # Initialize the manager
        result = manager.initialize()
        assert result is True
        assert manager.current_setting
        assert isinstance(manager.current_setting, LLMAnthropicConfig)
        assert manager.current_setting.max_tokens == 128000  # type: ignore
        assert manager.current_setting.default_headers
        assert (
            manager.current_setting.default_headers.get("anthropic-beta")
            == "output-128k-2025-02-19"
        )
    with tmp_config("claude-3-5-sonnet-20241022") as anthropic_config:
        manager = ModelManager(anthropic_config)

        # Initialize the manager
        result = manager.initialize()
        assert result is True
        assert manager.current_setting
        assert isinstance(manager.current_setting, LLMAnthropicConfig)
        assert manager.current_setting.max_tokens == 8129  # type: ignore
        assert (
            manager.current_setting.default_headers is None
        )  # No beta headers for small max_tokens
