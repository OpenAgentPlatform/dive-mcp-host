"""Tests for the MCP Server Installer Tools."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph.state import RunnableConfig  # noqa: TC002

from dive_mcp_host.internal_tools.tools import bash, fetch
from dive_mcp_host.internal_tools.tools.file_ops import read_file, write_file
from dive_mcp_host.internal_tools.tools.patterns import _detect_high_risk_command


class TestInstallerTools:
    """Tests for installer tools."""

    @pytest.mark.asyncio
    async def test_read_file_tool(self) -> None:
        """Test the read_file tool."""
        tool = read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name

        try:
            config: RunnableConfig = {
                "configurable": {
                    "stream_writer": lambda _: None,
                }
            }

            # Read the file (no elicitation needed - handled by graph)
            result = await tool.arun(tool_input={"path": temp_path}, config=config)
            assert "Hello, World!" in result
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_write_file_tool(self) -> None:
        """Test the write_file tool."""
        tool = write_file

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.txt"

            config: RunnableConfig = {
                "configurable": {
                    "stream_writer": lambda _: None,
                }
            }

            # Write the file (no elicitation needed - handled by graph)
            result = await tool.arun(
                tool_input={
                    "path": str(temp_path),
                    "content": "Test content",
                },
                config=config,
            )

            assert "Successfully wrote" in result
            assert temp_path.exists()
            assert temp_path.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_bash_tool(self) -> None:
        """Test the bash tool."""
        tool = bash

        config: RunnableConfig = {
            "configurable": {
                "stream_writer": lambda _: None,
            }
        }

        # Execute command (no elicitation needed - handled by graph)
        result = await tool.arun(tool_input={"command": "echo 'hello'"}, config=config)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_bash_tool_dry_run(self) -> None:
        """Test the bash tool in dry_run mode."""
        tool = bash

        config: RunnableConfig = {
            "configurable": {
                "stream_writer": lambda _: None,
                "dry_run": True,
            }
        }

        # In dry_run mode, command should NOT be executed
        result = await tool.arun(
            tool_input={"command": "echo 'should not run'"}, config=config
        )
        assert "[DRY RUN]" in result
        assert "would be executed" in result
        assert "Simulated success" in result

    @pytest.mark.asyncio
    async def test_bash_tool_high_risk_detection(self) -> None:
        """Test that the bash tool detects high-risk commands."""
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
        tool = bash

        # Without elicitation manager, should return error for sudo commands
        config: RunnableConfig = {
            "configurable": {
                "stream_writer": lambda _: None,
            }
        }

        result = await tool.arun(
            tool_input={"command": "sudo echo 'test'"}, config=config
        )
        assert "Error" in result
        assert "elicitation manager" in result.lower()

    @pytest.mark.asyncio
    async def test_bash_tool_custom_timeout(self) -> None:
        """Test that the bash tool respects custom timeout."""
        tool = bash

        config: RunnableConfig = {
            "configurable": {
                "stream_writer": lambda _: None,
            }
        }

        # Short timeout should be respected
        result = await tool.arun(
            tool_input={
                "command": "sleep 0.1 && echo 'done'",
                "timeout": 5,
            },
            config=config,
        )
        assert "done" in result

        # Timeout capped at 600 seconds
        tool_log_details = {}

        def capture_writer(data: Any) -> None:
            if hasattr(data[1], "details"):
                tool_log_details.update(data[1].details)

        config_with_capture: RunnableConfig = {
            "configurable": {
                "stream_writer": capture_writer,
            }
        }

        await tool.arun(
            tool_input={
                "command": "echo 'test'",
                "timeout": 1000,  # Should be capped to 600
            },
            config=config_with_capture,
        )
        assert tool_log_details.get("timeout") == 600

    @pytest.mark.asyncio
    async def test_fetch_tool_with_mock(self) -> None:
        """Test the fetch tool with mocked HTTP."""
        tool = fetch

        config: RunnableConfig = {
            "configurable": {
                "stream_writer": lambda _: None,
            }
        }

        # Mock httpx
        with patch(
            "dive_mcp_host.internal_tools.tools.fetch.httpx.AsyncClient"
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
            result = await tool.arun(
                tool_input={
                    "url": "https://github.com/api/packages/test",
                },
                config=config,
            )
            assert "test-package" in result
