import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from dive_mcp_host.httpd.conf.prompt import PromptKey, PromptManager

# Register custom mark
integration = pytest.mark.integration


# Unit tests
class TestPromptManager:
    """Unit tests for PromptManager class's basic functionality."""

    @pytest.fixture
    def mock_custom_rules_file(self):
        """Create a mock custom rules file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="custom_rules", delete=False
        ) as f:
            f.write("Test custom rules content")
            rules_path = f.name
        yield rules_path
        # Clean up after test
        Path(rules_path).unlink()

    def test_multiple_instances(self):
        """Test that multiple instances can be created with different configurations."""
        path1 = "/test/path1/custom_rules"
        path2 = "/test/path2/custom_rules"

        manager1 = PromptManager(path1)
        manager2 = PromptManager(path2)

        assert manager1 is not manager2
        assert manager1.custom_rules_path == path1
        assert manager2.custom_rules_path == path2

    def test_custom_rules_path_setting(self):
        """Test if the custom rules path is set correctly."""
        test_path = "/test/path/custom_rules"
        manager = PromptManager(test_path)
        assert manager.custom_rules_path == test_path

    def test_get_prompt(self):
        """Test retrieving prompt."""
        manager = PromptManager()
        # Set a test prompt
        test_prompt = "This is a test prompt"
        manager.initialize()
        manager.set_prompt("test_key", test_prompt)

        # Get the prompt
        prompt = manager.get_prompt("test_key")
        assert prompt == test_prompt

        # Get a non-existing prompt
        prompt = manager.get_prompt("non_existing_key")
        assert prompt is None

    def test_set_prompt(self):
        """Test setting prompt."""
        manager = PromptManager()
        test_prompt = "This is a test prompt"
        manager.set_prompt("test_key", test_prompt)

        # Verify the prompt has been set
        assert manager.prompts.get("test_key") == test_prompt

    def test_load_custom_rules_from_file(self, mock_custom_rules_file):
        """Test loading custom rules from file."""
        manager = PromptManager(mock_custom_rules_file)
        custom_rules = manager.load_custom_rules()
        assert custom_rules == "Test custom rules content"

    def test_load_custom_rules_from_env(self):
        """Test loading custom rules from environment variable."""
        with patch.dict(os.environ, {"DIVE_CUSTOM_RULES_CONTENT": "Env rules content"}):
            manager = PromptManager()
            custom_rules = manager.load_custom_rules()
            assert custom_rules == "Env rules content"

    def test_load_custom_rules_file_not_found(self):
        """Test loading custom rules when file not found."""
        manager = PromptManager("/non/existent/path/custom_rules")
        custom_rules = manager.load_custom_rules()
        assert custom_rules == ""

    def test_update_prompts(self, mock_custom_rules_file):
        """Test updating prompts."""
        manager = PromptManager(mock_custom_rules_file)
        # Update the system prompt
        manager.update_prompts()

        # Verify the system prompt has been updated
        system_prompt_text = manager.get_prompt(PromptKey.SYSTEM)
        assert system_prompt_text is not None
        assert "Test custom rules content" in system_prompt_text

        custom_prompt_text = manager.get_prompt(PromptKey.CUSTOM)
        assert custom_prompt_text is not None
        assert custom_prompt_text == "Test custom rules content"

    def test_system_prompt_initialization(self, mock_custom_rules_file):
        """Test system prompt initialization during PromptManager initialization."""
        # Initialize with custom rules path
        manager = PromptManager(mock_custom_rules_file)
        manager.initialize()

        # Verify system prompt contains custom rules
        system_prompt_text = manager.get_prompt(PromptKey.SYSTEM)
        assert system_prompt_text is not None
        assert "Test custom rules content" in system_prompt_text

        custom_prompt_text = manager.get_prompt(PromptKey.CUSTOM)
        assert custom_prompt_text is not None
        assert custom_prompt_text == "Test custom rules content"

    def test_env_variable_precedence(self, mock_custom_rules_file):
        """Test that environment variable takes precedence over file."""
        # Set environment variable
        with patch.dict(
            os.environ, {"DIVE_CUSTOM_RULES_CONTENT": "Env rules have precedence"}
        ):
            # Initialize with custom rules path
            manager = PromptManager(mock_custom_rules_file)
            manager.initialize()

            # Load custom rules
            custom_rules = manager.load_custom_rules()

            # Verify environment variable has precedence
            assert custom_rules == "Env rules have precedence"

            # Verify system prompt contains env rules
            system_prompt_text = manager.get_prompt("system")
            assert system_prompt_text is not None
            assert "Env rules have precedence" in system_prompt_text


# Integration tests
@pytest.mark.integration
class TestPromptManagerIntegration:
    """Integration tests for the complete functionality of PromptManager class."""

    @pytest_asyncio.fixture
    async def test_custom_rules_path(self):
        """Set up the test custom rules file path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rules_path = Path(tmp_dir) / "custom_rules"
            with rules_path.open("w") as f:
                f.write("Integration test custom rules content")
            yield str(rules_path)

    def test_full_prompt_workflow(self, test_custom_rules_path):
        """Test the complete prompt management workflow."""
        # Initialize the PromptManager
        manager = PromptManager(test_custom_rules_path)
        manager.initialize()

        # Verify system prompt contains custom rules
        system_prompt_text = manager.get_prompt(PromptKey.SYSTEM)
        assert system_prompt_text is not None
        assert "Integration test custom rules content" in system_prompt_text

        custom_prompt_text = manager.get_prompt(PromptKey.CUSTOM)
        assert custom_prompt_text is not None
        assert custom_prompt_text == "Integration test custom rules content"

        # Update custom rules file
        with Path(test_custom_rules_path).open("w") as f:
            f.write("Updated custom rules content")

        # Update the system prompt
        manager.update_prompts()

        # Verify system prompt has been updated
        updated_system_prompt = manager.get_prompt(PromptKey.SYSTEM)
        assert updated_system_prompt is not None
        assert "Updated custom rules content" in updated_system_prompt

        # Set a new prompt
        manager.set_prompt("custom_key", "Custom prompt content")

        # Get the prompt
        custom_prompt = manager.get_prompt("custom_key")
        assert custom_prompt == "Custom prompt content"
