"""Tool Manager Plugin for managing non-MCP tools.

This module provides a plugin system for registering additional tools
that are not MCP servers, such as built-in tools or custom tools.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Hook names for tool manager plugin
ToolManagerGetToolsHookName = "host.tools.tool_manager.get_tools"


class ToolManagerPlugin:
    """Plugin for managing non-MCP tools.

    This plugin allows registration of additional tools that will be
    included in the ToolManager's langchain_tools() output.

    It also manages:
    - Installer tool for MCP server installation

    Example:
        plugin = ToolManagerPlugin()

        # Register tools directly
        plugin.register_tools([my_tool1, my_tool2])

        # Or use callbacks for dynamic tool generation
        plugin.register_callback(lambda: [create_dynamic_tool()])

        # Setup installer tool with model
        plugin.setup_installer_tool(model)
    """

    def __init__(self) -> None:
        """Initialize the ToolManagerPlugin."""
        self._tools: list[BaseTool] = []
        self._callbacks: list[tuple[Callable[[], list[BaseTool]], str]] = []
        self._installer_tool: BaseTool | None = None
        self._local_tools: list[BaseTool] | None = None
        self._locale: str = "en"
        # Deprecated: kept for backwards compatibility
        self._mcp_reload_callback: Callable[[], Awaitable[None]] | None = None

    def register_tools(self, tools: list[BaseTool]) -> None:
        """Register static tools.

        Args:
            tools: List of tools to register.
        """
        self._tools.extend(tools)
        logger.info("Registered %d static tools", len(tools))

    def register_callback(
        self,
        callback: Callable[[], list[BaseTool]],
        plugin_name: str = "default",
    ) -> bool:
        """Register a callback that returns tools.

        Args:
            callback: A function that returns a list of tools.
            plugin_name: Name of the plugin registering this callback.

        Returns:
            True if registration was successful.
        """
        self._callbacks.append((callback, plugin_name))
        logger.info("Registered tool callback from plugin: %s", plugin_name)
        return True

    def setup_installer_tool(self, model: BaseChatModel) -> None:
        """Setup the installer tool with the given model.

        Args:
            model: The LLM model to use for the installer agent.
        """
        # from dive_mcp_host.mcp_installer_plugin import install_mcp_server_tool
        #
        # self._installer_tool = install_mcp_server_tool(model)
        # logger.info("Installer tool initialized")

    def setup_local_tools(self) -> None:
        """Setup local tools (fetch, bash, read_file, write_file).

        These tools can be exposed to external LLMs directly without going
        through the installer agent. They include built-in safety mechanisms
        like user confirmation for potentially dangerous operations.
        """
        from dive_mcp_host.mcp_installer_plugin import get_local_tools

        self._local_tools = get_local_tools()
        logger.info("Local tools initialized: %d tools", len(self._local_tools))

    @property
    def local_tools(self) -> list[BaseTool] | None:
        """Get the local tools."""
        return self._local_tools

    def set_locale(self, locale: str) -> None:
        """Set the locale for user-facing messages.

        Args:
            locale: The locale code (e.g., 'en', 'zh-TW', 'ja').
        """
        self._locale = locale
        logger.info("Locale set: %s", locale)

    @property
    def locale(self) -> str:
        """Get the current locale."""
        return self._locale

    def set_mcp_reload_callback(
        self, callback: Callable[[], Awaitable[None]] | None
    ) -> None:
        """Set the MCP reload callback (deprecated).

        Deprecated: Use mcp_installer_plugin.set_httpd_base_url() for HTTP API reload.

        Args:
            callback: An async callback function that triggers MCP server reload.
        """
        self._mcp_reload_callback = callback
        logger.info("MCP reload callback set (deprecated): %s", callback is not None)

    @property
    def mcp_reload_callback(self) -> Callable[[], Awaitable[None]] | None:
        """Get the MCP reload callback (deprecated)."""
        return self._mcp_reload_callback

    @property
    def installer_tool(self) -> BaseTool | None:
        """Get the installer tool."""
        return self._installer_tool

    def get_tools(
        self,
        include_installer: bool = True,
        include_local_tools: bool = False,
    ) -> list[BaseTool]:
        """Get all registered tools.

        Args:
            include_installer: Whether to include the installer tool.
            include_local_tools: Whether to include local tools (fetch, bash, etc.).

        Returns:
            List of all tools from static registration, callbacks, installer,
            and optionally local tools.
        """
        tools = list(self._tools)

        # Add callback tools
        for callback, plugin_name in self._callbacks:
            try:
                callback_tools = callback()
                tools.extend(callback_tools)
                logger.debug(
                    "Got %d tools from callback plugin: %s",
                    len(callback_tools),
                    plugin_name,
                )
            except Exception:
                logger.exception(
                    "Error getting tools from callback plugin: %s", plugin_name
                )

        # Add installer tool if available and requested
        # if include_installer and self._installer_tool is not None:
        #     tools.append(self._installer_tool)

        # Add local tools if available and requested
        if include_local_tools and self._local_tools is not None:
            tools.extend(self._local_tools)

        return tools

    def clear(self) -> None:
        """Clear all registered tools and callbacks."""
        self._tools.clear()
        self._callbacks.clear()

    @property
    def tool_count(self) -> int:
        """Get the total number of registered tools."""
        count = len(self._tools) + sum(len(cb()) for cb, _ in self._callbacks)
        if self._installer_tool is not None:
            count += 1
        return count

    def register_plugin(
        self,
        callback: Callable[[], list[BaseTool]],
        hook_name: str,
        plugin_name: str,
    ) -> bool:
        """Register a plugin callback (for use with PluginManager).

        This method is compatible with the PluginManager's hook system.

        Args:
            callback: A function that returns a list of tools.
            hook_name: The hook name (should be ToolManagerGetToolsHookName).
            plugin_name: Name of the plugin.

        Returns:
            True if registration was successful.
        """
        if hook_name != ToolManagerGetToolsHookName:
            logger.warning("Unknown hook name: %s", hook_name)
            return False

        return self.register_callback(callback, plugin_name)
