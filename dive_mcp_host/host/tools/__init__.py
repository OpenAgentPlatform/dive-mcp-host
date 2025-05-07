"""Model for the MCP servers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from itertools import chain
from typing import TYPE_CHECKING, Self

from mcp import types

from dive_mcp_host.host.conf import (
    LogConfig,
    ServerConfig,
)
from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.host.tools.log import (
    LogManager,
)
from dive_mcp_host.host.tools.mcp_server import McpServer, McpServerInfo, McpTool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterable, Mapping

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


type ReadStreamType = MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
type WriteStreamType = MemoryObjectSendStream[types.JSONRPCMessage]
type StreamContextType = AbstractAsyncContextManager[
    tuple[ReadStreamType, WriteStreamType]
]


logger = logging.getLogger(__name__)


class ToolManager(ContextProtocol):
    """Manager for the MCP Servers.

    Example:
        config = HostConfig(...)
        async with ToolManager(config.mcp_servers) as tool_manager:
            tool_manager.tools()
    """

    def __init__(
        self,
        configs: dict[str, ServerConfig],
        log_config: LogConfig = LogConfig(),
    ) -> None:
        """Initialize the ToolManager."""
        self._configs = configs
        self._log_config = log_config
        self._log_manager = LogManager(
            log_dir=log_config.log_dir, rotation_files=log_config.rotation_files
        )
        self._mcp_servers = dict[str, McpServer]()
        self._mcp_servers_task = dict[str, tuple[asyncio.Task, asyncio.Event]]()
        self._lock = asyncio.Lock()
        self._initialized_event = asyncio.Event()

        self._mcp_servers = {
            name: McpServer(
                name=name,
                config=config,
                log_buffer_length=log_config.buffer_length,
            )
            for name, config in self._configs.items()
        }

    def langchain_tools(
        self,
        tool_filter: Callable[[McpServer], bool] = lambda _: True,
    ) -> list[McpTool]:
        """Get the langchain tools for the MCP servers."""
        return list(
            chain.from_iterable(
                [i.mcp_tools for i in self._mcp_servers.values() if tool_filter(i)],
            ),
        )

    async def _launch_tools(self, servers: Mapping[str, McpServer]) -> None:
        async def tool_process(
            server: McpServer, exit_signal: asyncio.Event, ready: asyncio.Event
        ) -> None:
            async with self._log_manager.register_buffer(server.log_buffer), server:
                ready.set()
                await exit_signal.wait()
            logger.debug("Tool process %s exited", server.name)

        async def _launch_task(name: str, server: McpServer) -> None:
            event = asyncio.Event()
            ready = asyncio.Event()
            task = asyncio.create_task(tool_process(server, event, ready))
            await ready.wait()
            self._mcp_servers_task[name] = (task, event)

        async with self._lock, asyncio.TaskGroup() as tg:
            for name, server in servers.items():
                tg.create_task(_launch_task(name, server))

        self._initialized_event.set()

    async def _shutdown_tools(self, servers: Iterable[str]) -> None:
        async def _shutdown_task(name: str) -> None:
            task, event = self._mcp_servers_task.pop(name, (None, None))
            if not (task and event):
                logger.warning(
                    "task or event not found for %s. %s %s", name, task, event
                )
                return
            event.set()
            logger.debug("ToolManager shutting down %s", name)
            await task
            del self._mcp_servers[name]

        async with self._lock, asyncio.TaskGroup() as tg:
            for name in servers:
                tg.create_task(_shutdown_task(name))
        logger.debug("ToolManager shutdown complete")

    async def reload(
        self, new_configs: dict[str, ServerConfig], force: bool = False
    ) -> None:
        """Reload the MCP servers.

        Args:
            new_configs: The new MCP server configurations.
            force: If True, reload all MCP servers even if they are not changed.
        """
        logger.debug("Reloading MCP servers, force: %s", force)

        if not force:
            to_shutdown = set(self._configs.keys()) - set(new_configs.keys())
            to_launch = set(new_configs.keys()) - set(self._configs.keys())

            # check if the config has changed
            for key in set(self._configs) - to_shutdown:
                if self._configs[key] != new_configs[key]:
                    to_shutdown.add(key)
                    to_launch.add(key)
        else:
            to_shutdown = set(self._configs.keys())
            to_launch = set(new_configs.keys())

        self._configs = new_configs

        await self._shutdown_tools(to_shutdown)

        launch_servers = {}
        for l_key in to_launch:
            new_server = McpServer(
                name=l_key,
                config=new_configs[l_key],
                log_buffer_length=self._log_config.buffer_length,
            )
            launch_servers[l_key] = new_server
            self._mcp_servers[l_key] = new_server
        await self._launch_tools(launch_servers)

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """Get the langchain tools for the MCP servers."""
        # we can manipulate the stack to add or remove tools
        launch_tools_task = asyncio.create_task(
            self._launch_tools(self._mcp_servers),
            name="init-launch-tools",
        )
        try:
            yield self
        finally:
            await self._shutdown_tools(list(self._mcp_servers.keys()))
            launch_tools_task.cancel()
            await launch_tools_task

    @property
    def mcp_server_info(self) -> dict[str, McpServerInfo]:
        """Get the MCP server capabilities and tools.

        Returns:
            A dictionary of MCP server name to server info.
            If the mcp server is not initialized, the value will be `None`.
        """
        return {name: i.server_info for name, i in self._mcp_servers.items()}

    @property
    def initialized_event(self) -> asyncio.Event:
        """Get the initialization event.

        Only useful on initial startup, not when reloading.
        """
        return self._initialized_event

    @property
    def log_manager(self) -> LogManager:
        """Get the log manager."""
        return self._log_manager
