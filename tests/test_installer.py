"""Tests for the MCP Server Installer Agent."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dive_mcp_host.host.tools.elicitation_manager import ElicitationManager
from dive_mcp_host.mcp_installer_plugin import (
    InstallerAgent,
    InstallerBashTool,
    InstallerFetchTool,
    InstallerReadFileTool,
    InstallerWriteFileTool,
    get_installer_tools,
    install_mcp_server_tool,
)


class TestInstallerTools:
    """Tests for installer tools."""

    def test_get_installer_tools(self) -> None:
        """Test that get_installer_tools returns empty (installer agent removed)."""
        tools = get_installer_tools()
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_read_file_tool(self) -> None:
        """Test the read_file tool."""
        tool = InstallerReadFileTool()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name

        try:
            config = {
                "configurable": {
                    "stream_writer": lambda _: None,
                }
            }

            # Read the file (no elicitation needed - handled by graph)
            result = await tool._arun(path=temp_path, config=config)
            assert "Hello, World!" in result
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_write_file_tool(self) -> None:
        """Test the write_file tool."""
        tool = InstallerWriteFileTool()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.txt"

            config = {
                "configurable": {
                    "stream_writer": lambda _: None,
                }
            }

            # Write the file (no elicitation needed - handled by graph)
            result = await tool._arun(
                path=str(temp_path),
                content="Test content",
                config=config,
            )

            assert "Successfully wrote" in result
            assert temp_path.exists()
            assert temp_path.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_bash_tool(self) -> None:
        """Test the bash tool."""
        tool = InstallerBashTool()

        config = {
            "configurable": {
                "stream_writer": lambda _: None,
            }
        }

        # Execute command (no elicitation needed - handled by graph)
        result = await tool._arun(command="echo 'hello'", config=config)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_bash_tool_dry_run(self) -> None:
        """Test the bash tool in dry_run mode."""
        tool = InstallerBashTool()

        config = {
            "configurable": {
                "stream_writer": lambda _: None,
                "dry_run": True,
            }
        }

        # In dry_run mode, command should NOT be executed
        result = await tool._arun(command="echo 'should not run'", config=config)
        assert "[DRY RUN]" in result
        assert "would be executed" in result
        assert "Simulated success" in result

    @pytest.mark.asyncio
    async def test_bash_tool_high_risk_detection(self) -> None:
        """Test that the bash tool detects high-risk commands."""
        from dive_mcp_host.mcp_installer_plugin.tools import _detect_high_risk_command

        # Test sudo detection
        is_high_risk, reasons = _detect_high_risk_command("sudo apt install package")
        assert is_high_risk is True
        assert any("sudo" in r.lower() for r in reasons)

        # Test rm -rf detection
        is_high_risk, reasons = _detect_high_risk_command("rm -rf /some/path")
        assert is_high_risk is True
        assert any("rm -rf" in r.lower() for r in reasons)

        # Test safe command
        is_high_risk, reasons = _detect_high_risk_command("echo 'hello'")
        assert is_high_risk is False
        assert len(reasons) == 0

    @pytest.mark.asyncio
    async def test_bash_tool_sudo_requires_password(self) -> None:
        """Test that sudo commands require password (elicitation)."""
        tool = InstallerBashTool()

        # Without elicitation manager, should return error for sudo commands
        config = {
            "configurable": {
                "stream_writer": lambda _: None,
            }
        }

        result = await tool._arun(command="sudo echo 'test'", config=config)
        assert "Error" in result
        assert "elicitation manager" in result.lower()

    @pytest.mark.asyncio
    async def test_bash_tool_custom_timeout(self) -> None:
        """Test that the bash tool respects custom timeout."""
        tool = InstallerBashTool()

        config = {
            "configurable": {
                "stream_writer": lambda _: None,
            }
        }

        # Short timeout should be respected
        result = await tool._arun(
            command="sleep 0.1 && echo 'done'",
            timeout=5,
            config=config,
        )
        assert "done" in result

        # Timeout capped at 600 seconds
        tool_log_details = {}

        def capture_writer(data: Any) -> None:
            if hasattr(data[1], "details"):
                tool_log_details.update(data[1].details)

        config_with_capture = {
            "configurable": {
                "stream_writer": capture_writer,
            }
        }

        await tool._arun(
            command="echo 'test'",
            timeout=1000,  # Should be capped to 600
            config=config_with_capture,
        )
        assert tool_log_details.get("timeout") == 600

    @pytest.mark.asyncio
    async def test_fetch_tool_with_mock(self) -> None:
        """Test the fetch tool with mocked HTTP."""
        tool = InstallerFetchTool()

        config = {
            "configurable": {
                "stream_writer": lambda _: None,
            }
        }

        # Mock httpx
        with patch(
            "dive_mcp_host.mcp_installer_plugin.tools.fetch.httpx.AsyncClient"
        ) as mock_client:
            mock_response = MagicMock()
            mock_response.text = '{"name": "test-package"}'
            mock_response.headers = {"content-type": "application/json"}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.request = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            # Fetch URL (no elicitation needed - handled by graph)
            result = await tool._arun(
                url="https://github.com/api/packages/test",
                config=config,
            )
            assert "test-package" in result


class TestInstallerAgent:
    """Tests for InstallerAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self) -> None:
        """Test that the agent initializes correctly."""
        # Mock model
        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)

        elicitation_manager = ElicitationManager()
        agent = InstallerAgent(
            model=mock_model, elicitation_manager=elicitation_manager
        )
        assert agent._graph is not None
        # Installer agent tools are now empty (installer agent removed)
        assert len(agent.tools) == 0

    @pytest.mark.asyncio
    async def test_agent_graph_structure(self) -> None:
        """Test that the agent graph has the expected nodes."""
        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)

        elicitation_manager = ElicitationManager()
        agent = InstallerAgent(
            model=mock_model, elicitation_manager=elicitation_manager
        )

        # Check graph structure (confirm_install replaced by request_confirmation tool)
        node_names = list(agent._graph.nodes.keys())
        assert "call_model" in node_names
        assert "tools" in node_names
        # confirm_install removed, confirmation now via request_confirmation tool


class TestInstallMcpServerTool:
    """Tests for InstallMcpServerTool."""

    def test_tool_creation(self) -> None:
        """Test creating the install_mcp_server tool."""
        tool = install_mcp_server_tool()
        assert tool.name == "install_mcp_server"
        assert tool.model is None

    def test_tool_with_model(self) -> None:
        """Test creating the tool with a model."""
        mock_model = MagicMock()
        tool = install_mcp_server_tool(mock_model)
        assert tool.model is mock_model

    def test_bind_model(self) -> None:
        """Test binding a model to the tool."""
        mock_model = MagicMock()
        tool = install_mcp_server_tool()
        tool.bind_model(mock_model)
        assert tool.model is mock_model

    @pytest.mark.asyncio
    async def test_tool_without_model_returns_error(self) -> None:
        """Test that the tool returns an error if no model is bound."""
        tool = install_mcp_server_tool()
        result = await tool._arun(
            server_description="test-server",
            config={},
        )
        assert "Error: No model configured" in result

    @pytest.mark.asyncio
    async def test_tool_without_elicitation_manager_returns_error(self) -> None:
        """Test that the tool returns an error if no elicitation manager."""
        mock_model = MagicMock()
        tool = install_mcp_server_tool(mock_model)
        result = await tool._arun(
            server_description="test-server",
            config={"configurable": {}},
        )
        assert "Error: No elicitation manager configured" in result
