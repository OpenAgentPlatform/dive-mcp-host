"""MCP Server Installation Tool.

This module provides a LangChain tool that wraps the InstallerAgent,
allowing it to be used as a tool by the main chat agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: TC002
from langchain_core.tools import BaseTool, InjectedToolArg
from langgraph.config import get_config, get_stream_writer
from pydantic import BaseModel, Field

from dive_mcp_host.mcp_installer_plugin.agent import InstallerAgent
from dive_mcp_host.mcp_installer_plugin.config import InstallerConfigManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class InstallMcpServerInput(BaseModel):
    """Input schema for the install_mcp_server tool."""

    server_description: Annotated[
        str,
        Field(
            description="""Description of the MCP server to install.
This can be:
- A package name (e.g., "mcp-server-fetch", "@anthropic/claude-mcp")
- A description (e.g., "fetch tool for making HTTP requests")
- A URL to documentation or repository
- A specific installation command
"""
        ),
    ]
    additional_instructions: Annotated[
        str | None,
        Field(
            default=None,
            description="Additional instructions or preferences for the installation.",
        ),
    ] = None
    locale: Annotated[
        str,
        Field(
            default="en",
            description="Locale/language for user-facing messages "
            "(e.g., 'en', 'zh-TW', 'ja'). Use the same language as the user's input.",
        ),
    ] = "en"


class InstallMcpServerTool(BaseTool):
    """Tool for installing MCP servers.

    This tool wraps an InstallerAgent that uses LLM reasoning to
    automatically install MCP servers. It handles:
    - Fetching documentation and package info
    - Running installation commands
    - Creating configuration files
    - User approval for potentially dangerous operations (via elicitation)
    """

    name: str = "install_mcp_server"
    description: str = """Install an MCP (Model Context Protocol) server.

Use this tool when the user wants to install a new MCP server or tool.
The tool will:
1. Research the server and its installation requirements
2. Execute necessary installation commands (with user approval)
3. Create or update the MCP configuration

Provide either a package name, description, or URL for the server to install.
The tool will figure out the best installation method.

Examples:
- "mcp-server-fetch" - Install by package name
- "A server for making HTTP requests" - Install by description
- "https://github.com/example/mcp-server" - Install from URL
"""
    args_schema: type[BaseModel] = InstallMcpServerInput
    return_direct: bool = False

    # Internal state
    model: BaseChatModel | None = None
    """The LLM model to use for the installer agent."""

    _active_agent: InstallerAgent | None = None

    def bind_model(self, model: BaseChatModel) -> InstallMcpServerTool:
        """Bind a model to the tool.

        Args:
            model: The LLM model to use for the installer agent.

        Returns:
            Self for chaining.
        """
        self.model = model
        return self

    async def _arun(
        self,
        server_description: str,
        additional_instructions: str | None = None,
        locale: str = "en",
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> str:
        """Run the MCP server installation.

        Args:
            server_description: Description of the server to install.
            additional_instructions: Additional instructions.
            locale: Locale/language for user-facing messages.
            config: Runnable config with stream writer and other settings.

        Returns:
            Result of the installation as a string.
        """
        if self.model is None:
            return (
                "Error: No model configured for the installer tool. "
                "Please bind a model first."
            )

        # Get config from LangGraph context if not provided via InjectedToolArg
        if config is None or not config.get("configurable"):
            try:
                config = get_config()
            except (RuntimeError, LookupError):
                config = {}

        # Get stream writer and elicitation manager from config
        stream_writer = self._get_stream_writer(config)
        abort_signal = config.get("configurable", {}).get("abort_signal")
        elicitation_manager = config.get("configurable", {}).get("elicitation_manager")

        if elicitation_manager is None:
            return (
                "Error: No elicitation manager configured. "
                "Cannot request user approval."
            )

        # Build the query for the installer agent
        query = f"Install MCP server: {server_description}"
        if additional_instructions:
            query += f"\n\nAdditional instructions: {additional_instructions}"

        # Emit initial progress
        if stream_writer:
            from dive_mcp_host.mcp_installer_plugin.events import InstallerProgress

            stream_writer(
                (
                    InstallerProgress.NAME,
                    InstallerProgress(
                        phase="analyzing",
                        message=f"Starting installation: {server_description[:50]}...",
                    ),
                )
            )

        # Get dry_run setting from host.json
        config_manager = InstallerConfigManager.get_instance()
        dry_run = config_manager.get_installer_settings().dry_run

        # Get MCP reload callback from config (deprecated fallback)
        mcp_reload_callback = config.get("configurable", {}).get("mcp_reload_callback")

        # Create the installer agent
        agent = InstallerAgent(
            model=self.model,
            elicitation_manager=elicitation_manager,
            dry_run=dry_run,
            mcp_reload_callback=mcp_reload_callback,
            locale=locale,
        )
        self._active_agent = agent

        try:
            # Collect results from the agent
            final_messages: list[Any] = []

            async for chunk in agent.run(
                query=query,
                stream_writer=stream_writer,
                abort_signal=abort_signal,
            ):
                # Collect messages from the agent's state updates
                if "call_model" in chunk:
                    messages = chunk["call_model"].get("messages", [])
                    final_messages.extend(messages)
                elif "tools" in chunk:
                    messages = chunk["tools"].get("messages", [])
                    final_messages.extend(messages)

            # Extract the final result
            if final_messages:
                # Get the last AI message as the result
                for msg in reversed(final_messages):
                    if hasattr(msg, "content") and msg.content:
                        return str(msg.content)

            return "Installation completed, but no summary was provided."

        except Exception as e:
            logger.exception("Error during MCP server installation")
            return f"Error during installation: {e}"
        finally:
            self._active_agent = None

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")

    def _get_stream_writer(
        self, config: RunnableConfig
    ) -> Callable[[tuple[str, Any]], None]:
        """Get stream writer from LangGraph context.

        Forwards events to parent's custom event stream.
        """
        # Use LangGraph's get_stream_writer to get the actual stream writer from context
        try:
            parent_writer = get_stream_writer()
        except (RuntimeError, LookupError):
            # Fallback to config if not in LangGraph context
            parent_writer = config.get("configurable", {}).get("stream_writer")

        def stream_writer(event: tuple[str, Any]) -> None:
            """Forward events with tool context."""
            if parent_writer:
                parent_writer(event)

        return stream_writer


def install_mcp_server_tool(model: BaseChatModel | None = None) -> InstallMcpServerTool:
    """Create an install_mcp_server tool.

    Args:
        model: The LLM model to use for the installer agent.
               If not provided, must be bound later with bind_model().

    Returns:
        The configured tool.
    """
    tool = InstallMcpServerTool()
    if model:
        tool.bind_model(model)
    return tool


class InstallMcpServerToolManager:
    """Manager for install_mcp_server tools.

    This manager handles the lifecycle of installation tools.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Initialize the manager.

        Args:
            model: The LLM model to use for installer agents.
        """
        self._model = model

    def get_tool(self) -> InstallMcpServerTool:
        """Get an installation tool.

        Returns:
            The configured tool.
        """
        return install_mcp_server_tool(self._model)
