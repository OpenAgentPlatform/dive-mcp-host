"""MCP server tools for the MCP Server Installer Agent.

This module provides tools for managing MCP server configuration including
get_mcp_config, add_mcp_server, reload_mcp_server, and install_mcp_instructions.
"""

# ruff: noqa: E501, PLR0911, PLR2004
# E501: Line too long - tool descriptions require specific formatting
# PLR0911: Many return statements needed for complex control flow
# PLR2004: Magic values are intentional (HTTP status codes, limits)

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

import httpx
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from pydantic import BaseModel, Field

from dive_mcp_host.mcp_installer_plugin.events import InstallerToolLog
from dive_mcp_host.mcp_installer_plugin.prompt import get_installer_system_prompt
from dive_mcp_host.mcp_installer_plugin.tools.common import (
    _check_aborted,
    _ensure_config,
    _get_abort_signal,
    _get_httpd_base_url,
    _get_mcp_reload_callback,
    _get_stream_writer,
)

logger = logging.getLogger(__name__)


@tool(
    description="""
Tool for getting the MCP installation instructions.
MUST call this tool before any MCP installation related stuff.
"""
)
async def install_mcp_instructions(
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Get the current MCP configuration."""
    config = _ensure_config(config)

    stream_writer = _get_stream_writer(config)
    abort_signal = _get_abort_signal(config)

    # Check if already aborted
    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    stream_writer(
        (
            InstallerToolLog.NAME,
            InstallerToolLog(
                tool="install_mcp_instructions",
                action="Reading MCP installer system prompt",
                details={},
            ),
        )
    )

    try:
        return get_installer_system_prompt()
    except Exception as e:
        logger.exception("Error reading MCP installer system prompt")
        result = f"Error reading MCP installer system prompt: {e}"

    return result


class InstallMCPInstructions(BaseTool):
    """Tool for getting the MCP installation prompt."""

    name: str = "install_mcp_instructions"
    description: str = """
    Tool for getting the MCP installation instructions.
    MUST call this tool before any MCP installation related stuff.
    """
    args_schema: type[BaseModel] | None = None

    async def _arun(
        self,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Get the current MCP configuration."""
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="install_mcp_instructions",
                    action="Reading MCP installer system prompt",
                    details={},
                ),
            )
        )

        try:
            return get_installer_system_prompt()
        except Exception as e:
            logger.exception("Error reading MCP installer system prompt")
            result = f"Error reading MCP installer system prompt: {e}"

        return result

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


@tool(
    description="""Get the current MCP server configuration.

Use this tool to check what MCP servers are already configured before adding new ones.
This helps avoid duplicate installations and understand the current setup.

Returns a JSON object with all configured MCP servers, including:
- Server names
- Transport type (stdio, sse, websocket, streamable)
- Command and arguments (for stdio transport)
- URL (for sse/websocket transport)
- Enabled status
"""
)
async def get_mcp_config(
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Get the current MCP configuration."""
    import json

    config = _ensure_config(config)

    stream_writer = _get_stream_writer(config)
    abort_signal = _get_abort_signal(config)

    # Check if already aborted
    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    stream_writer(
        (
            InstallerToolLog.NAME,
            InstallerToolLog(
                tool="get_mcp_config",
                action="Reading MCP configuration",
                details={},
            ),
        )
    )

    try:
        from dive_mcp_host.httpd.conf.mcp_servers import MCPServerManager

        manager = MCPServerManager()
        manager.initialize()

        current_config = manager._current_config  # noqa: SLF001
        if current_config is None:
            result = json.dumps({"mcpServers": {}}, indent=2)
        else:
            result = current_config.model_dump_json(
                by_alias=True, exclude_unset=True, indent=2
            )

    except Exception as e:
        logger.exception("Error reading MCP config")
        result = f"Error reading MCP configuration: {e}"

    return result


class InstallerGetMcpConfigTool(BaseTool):
    """Tool for getting the current MCP server configuration.

    This tool reads the current MCP configuration file and returns
    the list of configured servers.
    """

    name: str = "get_mcp_config"
    description: str = """Get the current MCP server configuration.

Use this tool to check what MCP servers are already configured before adding new ones.
This helps avoid duplicate installations and understand the current setup.

Returns a JSON object with all configured MCP servers, including:
- Server names
- Transport type (stdio, sse, websocket, streamable)
- Command and arguments (for stdio transport)
- URL (for sse/websocket transport)
- Enabled status
"""
    args_schema: type[BaseModel] | None = None

    async def _arun(
        self,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Get the current MCP configuration."""
        import json

        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="get_mcp_config",
                    action="Reading MCP configuration",
                    details={},
                ),
            )
        )

        try:
            from dive_mcp_host.httpd.conf.mcp_servers import MCPServerManager

            manager = MCPServerManager()
            manager.initialize()

            current_config = manager._current_config  # noqa: SLF001
            if current_config is None:
                result = json.dumps({"mcpServers": {}}, indent=2)
            else:
                result = current_config.model_dump_json(
                    by_alias=True, exclude_unset=True, indent=2
                )

        except Exception as e:
            logger.exception("Error reading MCP config")
            result = f"Error reading MCP configuration: {e}"

        return result

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class AddMcpServerInput(BaseModel):
    """Input schema for the add_mcp_server tool."""

    server_name: Annotated[
        str,
        Field(description="Unique name for the MCP server (e.g., 'yt-dlp', 'fetch')."),
    ]
    command: Annotated[
        str | None,
        Field(
            default=None,
            description="Command to run for stdio transport (e.g., 'npx', 'uvx', 'python').",
        ),
    ] = None
    args: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Arguments for the command (e.g., ['-y', 'yt-dlp-mcp']).",
        ),
    ] = None
    env: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description="Environment variables for the server.",
        ),
    ] = None
    url: Annotated[
        str | None,
        Field(
            default=None,
            description="URL for sse/websocket transport.",
        ),
    ] = None
    transport: Annotated[
        str,
        Field(
            default="stdio",
            description="Transport type: 'stdio', 'sse', 'websocket', or 'streamable'.",
        ),
    ] = "stdio"
    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the server should be enabled.",
        ),
    ] = True


@tool(
    description="""Add a new MCP server configuration.

Use this tool to register a newly installed MCP server with the system.
This will add the server to mcp_config.json and reload the host.

For stdio transport (most common), provide:
- server_name: A unique identifier for the server
- command: The command to run (e.g., 'npx', 'uvx', 'python')
- args: List of arguments for the command

For sse/websocket/streamable transport, provide:
- server_name: A unique identifier
- url: The server URL
- transport: 'sse', 'websocket', or 'streamable'

Example for npx-based server:
  server_name="yt-dlp"
  command="npx"
  args=["-y", "yt-dlp-mcp"]

Example for uvx-based server:
  server_name="mcp-server-fetch"
  command="uvx"
  args=["mcp-server-fetch"]

"""
)
async def add_mcp_server(
    server_name: Annotated[
        str,
        Field(description="Unique name for the MCP server (e.g., 'yt-dlp', 'fetch')."),
    ],
    command: Annotated[
        str | None,
        Field(
            default=None,
            description="Command to run for stdio transport (e.g., 'npx', 'uvx', 'python').",
        ),
    ] = None,
    arguments: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Arguments for the command (e.g., ['-y', 'yt-dlp-mcp']).",
        ),
    ] = None,
    env: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description="Environment variables for the server.",
        ),
    ] = None,
    url: Annotated[
        str | None,
        Field(
            default=None,
            description="URL for sse/websocket transport.",
        ),
    ] = None,
    transport: Annotated[
        str,
        Field(
            default="stdio",
            description="Transport type: 'stdio', 'sse', 'websocket', or 'streamable'.",
        ),
    ] = "stdio",
    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the server should be enabled.",
        ),
    ] = True,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Add an MCP server configuration.

    Note: User confirmation is handled by the confirm_install node in the graph,
    not by individual tools.
    """
    config = _ensure_config(config)

    stream_writer = _get_stream_writer(config)
    abort_signal = _get_abort_signal(config)

    # Check if already aborted
    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Validate input
    if transport == "stdio" and command is None:
        return "Error: 'command' is required for stdio transport."
    if transport in ["sse", "websocket", "streamable"] and url is None:
        return "Error: 'url' is required for sse/websocket/streamable transport."

    stream_writer(
        (
            InstallerToolLog.NAME,
            InstallerToolLog(
                tool="add_mcp_server",
                action=f"Adding MCP server: {server_name}",
                details={
                    "server_name": server_name,
                    "command": command,
                    "args": arguments,
                    "transport": transport,
                },
            ),
        )
    )

    try:
        # Import here to avoid circular imports
        from dive_mcp_host.httpd.conf.mcp_servers import (
            Config,
            MCPServerConfig,
            MCPServerManager,
        )

        # Load current config
        manager = MCPServerManager()
        manager.initialize()

        current_config = manager._current_config  # noqa: SLF001
        if current_config is None:
            current_config = Config()

        # Create new server config
        new_server = MCPServerConfig(
            transport=transport,  # type: ignore
            enabled=enabled,
            command=command,
            args=arguments or [],
            env=env or {},
            url=url,
        )

        # Add to config
        current_config.mcp_servers[server_name] = new_server

        # Trigger reload via HTTP API
        reload_status = await trigger_mcp_reload(config, current_config, server_name)

        return (
            f"Successfully added MCP server '{server_name}' to configuration.{reload_status} "
            f"Config: command={command}, args={arguments}, transport={transport}"
        )

    except Exception as e:
        logger.exception("Error adding MCP server")
        return f"Error adding MCP server '{server_name}': {e}"


async def trigger_mcp_reload(
    config: RunnableConfig,
    mcp_config: Any,
    server_name: str,
) -> str:
    """Trigger MCP server reload via HTTP API.

    Args:
        config: The runnable config (for deprecated mcp_reload_callback).
        mcp_config: The MCP server configuration to send.
        server_name: The name of the server being added (to check for errors).

    Returns:
        Status message for the reload operation, including any errors.
    """
    # First try HTTP API
    httpd_base_url = _get_httpd_base_url()
    if httpd_base_url:
        try:
            async with httpx.AsyncClient() as client:
                # Serialize config properly
                payload = mcp_config.model_dump(by_alias=True, exclude_unset=True)
                response = await client.post(
                    f"{httpd_base_url}/api/config/mcpserver",
                    json=payload,
                    timeout=30.0,
                )
                if response.status_code == 200:
                    result = response.json()
                    errors = result.get("errors", [])

                    # Check if the specific server we added has an error
                    server_error = None
                    for error in errors:
                        if error.get("serverName") == server_name:
                            server_error = error.get("error", "Unknown error")
                            break

                    if server_error:
                        logger.warning(
                            "MCP server '%s' failed to load: %s",
                            server_name,
                            server_error,
                        )
                        return (
                            f" ERROR: Server '{server_name}' failed to load: {server_error}. "
                            "You may need to install missing dependencies and then use "
                            "reload_mcp_server to retry loading."
                        )

                    logger.info("MCP reload via HTTP API succeeded")
                    return " The server has been loaded and is now available."

                logger.warning(
                    "MCP reload HTTP API returned %s: %s",
                    response.status_code,
                    response.text,
                )
                return " Note: Auto-reload failed, you may need to reload manually."
        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.warning("MCP reload via HTTP API failed: %s", e)
            return f" Note: Auto-reload failed ({e}), you may need to reload manually."

    # Fallback to callback (deprecated)
    reload_callback = _get_mcp_reload_callback(config)
    if reload_callback:
        try:
            callback_result = reload_callback()
            if asyncio.iscoroutine(callback_result):
                await callback_result
            logger.info("MCP reload callback executed successfully")
            return " The server has been loaded and is now available."
        except (OSError, RuntimeError) as e:
            logger.warning("MCP reload callback failed: %s", e)
            return f" Note: Auto-reload failed ({e}), you may need to reload manually."

    return " The server will be available after reloading."


class InstallerAddMcpServerTool(BaseTool):
    """Tool for adding an MCP server configuration.

    This tool adds a new MCP server to the configuration and reloads
    the host to apply the changes.
    """

    name: str = "add_mcp_server"
    description: str = """Add a new MCP server configuration.

Use this tool to register a newly installed MCP server with the system.
This will add the server to mcp_config.json and reload the host.

For stdio transport (most common), provide:
- server_name: A unique identifier for the server
- command: The command to run (e.g., 'npx', 'uvx', 'python')
- args: List of arguments for the command

For sse/websocket/streamable transport, provide:
- server_name: A unique identifier
- url: The server URL
- transport: 'sse', 'websocket', or 'streamable'

Example for npx-based server:
  server_name="yt-dlp"
  command="npx"
  args=["-y", "yt-dlp-mcp"]

Example for uvx-based server:
  server_name="mcp-server-fetch"
  command="uvx"
  args=["mcp-server-fetch"]
"""
    args_schema: type[BaseModel] | None = AddMcpServerInput

    async def _arun(
        self,
        server_name: str,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        transport: str = "stdio",
        enabled: bool = True,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Add an MCP server configuration.

        Note: User confirmation is handled by the confirm_install node in the graph,
        not by individual tools.
        """
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        # Validate input
        if transport == "stdio" and command is None:
            return "Error: 'command' is required for stdio transport."
        if transport in ["sse", "websocket", "streamable"] and url is None:
            return "Error: 'url' is required for sse/websocket/streamable transport."

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="add_mcp_server",
                    action=f"Adding MCP server: {server_name}",
                    details={
                        "server_name": server_name,
                        "command": command,
                        "args": args,
                        "transport": transport,
                    },
                ),
            )
        )

        try:
            # Import here to avoid circular imports
            from dive_mcp_host.httpd.conf.mcp_servers import (
                Config,
                MCPServerConfig,
                MCPServerManager,
            )

            # Load current config
            manager = MCPServerManager()
            manager.initialize()

            current_config = manager._current_config  # noqa: SLF001
            if current_config is None:
                current_config = Config()

            # Create new server config
            new_server = MCPServerConfig(
                transport=transport,  # type: ignore
                enabled=enabled,
                command=command,
                args=args or [],
                env=env or {},
                url=url,
            )

            # Add to config
            current_config.mcp_servers[server_name] = new_server

            # Trigger reload via HTTP API
            reload_status = await self._trigger_mcp_reload(
                config, current_config, server_name
            )

            return (
                f"Successfully added MCP server '{server_name}' to configuration.{reload_status} "
                f"Config: command={command}, args={args}, transport={transport}"
            )

        except Exception as e:
            logger.exception("Error adding MCP server")
            return f"Error adding MCP server '{server_name}': {e}"

    async def _trigger_mcp_reload(
        self,
        config: RunnableConfig,
        mcp_config: Any,
        server_name: str,
    ) -> str:
        """Trigger MCP server reload via HTTP API.

        Args:
            config: The runnable config (for deprecated mcp_reload_callback).
            mcp_config: The MCP server configuration to send.
            server_name: The name of the server being added (to check for errors).

        Returns:
            Status message for the reload operation, including any errors.
        """
        # First try HTTP API
        httpd_base_url = _get_httpd_base_url()
        if httpd_base_url:
            try:
                async with httpx.AsyncClient() as client:
                    # Serialize config properly
                    payload = mcp_config.model_dump(by_alias=True, exclude_unset=True)
                    response = await client.post(
                        f"{httpd_base_url}/api/config/mcpserver",
                        json=payload,
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        errors = result.get("errors", [])

                        # Check if the specific server we added has an error
                        server_error = None
                        for error in errors:
                            if error.get("serverName") == server_name:
                                server_error = error.get("error", "Unknown error")
                                break

                        if server_error:
                            logger.warning(
                                "MCP server '%s' failed to load: %s",
                                server_name,
                                server_error,
                            )
                            return (
                                f" ERROR: Server '{server_name}' failed to load: {server_error}. "
                                "You may need to install missing dependencies and then use "
                                "reload_mcp_server to retry loading."
                            )

                        logger.info("MCP reload via HTTP API succeeded")
                        return " The server has been loaded and is now available."

                    logger.warning(
                        "MCP reload HTTP API returned %s: %s",
                        response.status_code,
                        response.text,
                    )
                    return " Note: Auto-reload failed, you may need to reload manually."
            except (httpx.HTTPError, OSError, ValueError) as e:
                logger.warning("MCP reload via HTTP API failed: %s", e)
                return (
                    f" Note: Auto-reload failed ({e}), you may need to reload manually."
                )

        # Fallback to callback (deprecated)
        reload_callback = _get_mcp_reload_callback(config)
        if reload_callback:
            try:
                callback_result = reload_callback()
                if asyncio.iscoroutine(callback_result):
                    await callback_result
                logger.info("MCP reload callback executed successfully")
                return " The server has been loaded and is now available."
            except (OSError, RuntimeError) as e:
                logger.warning("MCP reload callback failed: %s", e)
                return (
                    f" Note: Auto-reload failed ({e}), you may need to reload manually."
                )

        return " The server will be available after reloading."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class ReloadMcpServerInput(BaseModel):
    """Input schema for the reload_mcp_server tool."""

    server_name: Annotated[
        str,
        Field(description="Name of the MCP server to reload."),
    ]


@tool(
    description="""Reload a specific MCP server.

Use this tool to retry loading an MCP server after:
- Installing missing dependencies
- Fixing configuration issues
- Resolving environment problems

This will reload the server configuration and attempt to start the server again.
Check the result for any errors.

Example:
  reload_mcp_server(server_name="my-server")
"""
)
async def reload_mcp_server(
    server_name: Annotated[
        str,
        Field(description="Name of the MCP server to reload."),
    ],
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Reload a specific MCP server."""
    config = _ensure_config(config)

    stream_writer = _get_stream_writer(config)
    abort_signal = _get_abort_signal(config)

    # Check if already aborted
    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    stream_writer(
        (
            InstallerToolLog.NAME,
            InstallerToolLog(
                tool="reload_mcp_server",
                action=f"Reloading MCP server: {server_name}",
                details={"server_name": server_name},
            ),
        )
    )

    httpd_base_url = _get_httpd_base_url()
    if not httpd_base_url:
        return "Error: Cannot reload - no HTTP API available."

    try:
        from dive_mcp_host.httpd.conf.mcp_servers import MCPServerManager

        # Load current config
        manager = MCPServerManager()
        manager.initialize()

        current_config = manager._current_config  # noqa: SLF001
        if current_config is None:
            return "Error: No MCP configuration found."

        if server_name not in current_config.mcp_servers:
            return f"Error: Server '{server_name}' not found in configuration."

        async with httpx.AsyncClient() as client:
            payload = current_config.model_dump(by_alias=True, exclude_unset=True)
            response = await client.post(
                f"{httpd_base_url}/api/config/mcpserver?force=true",
                json=payload,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                errors = result.get("errors", [])

                # Check if the specific server has an error
                server_error = None
                for error in errors:
                    if error.get("serverName") == server_name:
                        server_error = error.get("error", "Unknown error")
                        break

                if server_error:
                    return (
                        f"ERROR: Server '{server_name}' still failed to load: {server_error}. "
                        "Check if all dependencies are correctly installed."
                    )

                return (
                    f"Successfully reloaded MCP server '{server_name}'. "
                    "The server is now available."
                )

            return (
                f"Error: Reload request failed with status "
                f"{response.status_code}: {response.text}"
            )

    except (httpx.HTTPError, OSError, ValueError) as e:
        logger.exception("Error reloading MCP server")
        return f"Error reloading MCP server '{server_name}': {e}"


class InstallerReloadMcpServerTool(BaseTool):
    """Tool for reloading a specific MCP server.

    Use this tool after installing dependencies to retry loading a server
    that previously failed.
    """

    name: str = "reload_mcp_server"
    description: str = """Reload a specific MCP server.

Use this tool to retry loading an MCP server after:
- Installing missing dependencies
- Fixing configuration issues
- Resolving environment problems

This will reload the server configuration and attempt to start the server again.
Check the result for any errors.

Example:
  reload_mcp_server(server_name="my-server")
"""
    args_schema: type[BaseModel] | None = ReloadMcpServerInput

    async def _arun(
        self,
        server_name: str,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Reload a specific MCP server."""
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="reload_mcp_server",
                    action=f"Reloading MCP server: {server_name}",
                    details={"server_name": server_name},
                ),
            )
        )

        httpd_base_url = _get_httpd_base_url()
        if not httpd_base_url:
            return "Error: Cannot reload - no HTTP API available."

        try:
            from dive_mcp_host.httpd.conf.mcp_servers import MCPServerManager

            # Load current config
            manager = MCPServerManager()
            manager.initialize()

            current_config = manager._current_config  # noqa: SLF001
            if current_config is None:
                return "Error: No MCP configuration found."

            if server_name not in current_config.mcp_servers:
                return f"Error: Server '{server_name}' not found in configuration."

            async with httpx.AsyncClient() as client:
                payload = current_config.model_dump(by_alias=True, exclude_unset=True)
                response = await client.post(
                    f"{httpd_base_url}/api/config/mcpserver?force=true",
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    errors = result.get("errors", [])

                    # Check if the specific server has an error
                    server_error = None
                    for error in errors:
                        if error.get("serverName") == server_name:
                            server_error = error.get("error", "Unknown error")
                            break

                    if server_error:
                        return (
                            f"ERROR: Server '{server_name}' still failed to load: {server_error}. "
                            "Check if all dependencies are correctly installed."
                        )

                    return (
                        f"Successfully reloaded MCP server '{server_name}'. "
                        "The server is now available."
                    )

                return (
                    f"Error: Reload request failed with status "
                    f"{response.status_code}: {response.text}"
                )

        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.exception("Error reloading MCP server")
            return f"Error reloading MCP server '{server_name}': {e}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")
