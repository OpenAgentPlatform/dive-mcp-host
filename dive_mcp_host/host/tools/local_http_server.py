import asyncio
import logging
import signal
import sys
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from time import time
from typing import Any

import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client
from mcp.shared.message import SessionMessage
from mcp.types import InitializeResult, ListToolsResult
from pydantic import SecretStr

from dive_mcp_host.host.conf import ServerConfig
from dive_mcp_host.host.errors import InvalidMcpServerError
from dive_mcp_host.host.tools.log import LogProxy

logger = logging.getLogger(__name__)


@asynccontextmanager
async def local_http_server(
    config: ServerConfig,
    stderrlog: LogProxy,
    stdoutlog: LogProxy,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, Any] | None = None,
) -> AsyncGenerator[tuple[InitializeResult, ListToolsResult, int], None]:
    """Create a local MCP server client.

    Args:
        config: The configuration of the MCP server.
        command: The command to start the MCP server. if None, use config.command.
        args: The arguments to start the MCP server. if None, use config.args.
        env: The environment variables to start the MCP server. if None, use config.env.
        max_connection_retries: The maximum number of connection creaation.
        headers: The headers to send to the MCP server. if None, use config.headers.
        stderrlog: The log proxy to write the stderr of the subprocess.
        stdoutlog: The log proxy to write the stdout of the subprocess.
    """
    command = command or config.command
    args = args or config.args
    env = env or config.env
    headers = (headers or config.headers).copy()
    assert config.url is not None, "url is required"

    logger.error("env: %s", env)

    def _sse_client(
        url: str,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        for key in headers:
            value = headers[key]
            if isinstance(value, SecretStr):
                headers[key] = value.get_secret_value()
        logger.debug("Connecting sse client with timeout: %s", config.initial_timeout)
        return sse_client(
            url=url,
            headers=headers,
            sse_read_timeout=config.initial_timeout,
            timeout=config.initial_timeout,
        )

    get_client = _sse_client if config.transport != "websocket" else websocket_client
    logger.debug("Starting local MCP server %s with command: %s", config.name, command)
    if not (
        subprocess := await asyncio.create_subprocess_exec(
            command,
            *args,
            env=env,
            stderr=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
    ):
        logger.error("Failed to start subprocess for %s", config.name)
        raise RuntimeError("failed to start subprocess")

    # it tooks time to start the server, so we need to retry
    async def _read_stdout(
        stream: asyncio.StreamReader | None,
    ) -> None:
        """Read the stdout of the subprocess."""
        if not stream:
            return

        async for line in stream:
            await stdoutlog.write(line.decode())
            await stdoutlog.flush()

    async def _read_stderr(
        stream: asyncio.StreamReader | None,
    ) -> None:
        """Read the stderr of the subprocess."""
        if not stream:
            return

        async for line in stream:
            await stderrlog.write(line.decode())
            await stderrlog.flush()

    read_stderr_task = asyncio.create_task(
        _read_stderr(subprocess.stderr),
        name="read-stderr",
    )
    read_stdout_task = asyncio.create_task(
        _read_stdout(subprocess.stdout),
        name="read-stdout",
    )

    start_time = time()
    try:
        logger.debug(
            "Server %s initalizing with timeout: %s",
            config.name,
            config.initial_timeout,
        )
        while (time() - start_time) < config.initial_timeout:
            with suppress(TimeoutError, httpx.HTTPError):
                async with (
                    get_client(url=config.url) as streams,
                    ClientSession(*streams) as session,
                ):
                    async with asyncio.timeout(config.initial_timeout):
                        initialize_result = await session.initialize()
                        tools = await session.list_tools()
                        logger.info(
                            "Successfully connected to server %s, got tools: %s",
                            config.name,
                            tools,
                        )
                    logger.info("Connected to the server %s", config.name)
                    yield initialize_result, tools, subprocess.pid
                    break
            await asyncio.sleep(0.3)
        else:
            logger.warning(
                "Connected to the server %s failed after %s seconds",
                config.name,
                config.initial_timeout,
            )
            raise InvalidMcpServerError(config.name, reason="failed to initalize")
    finally:
        with suppress(TimeoutError, ProcessLookupError, asyncio.CancelledError):
            logger.debug("Terminating subprocess for %s", config.name)
            read_stderr_task.cancel()
            read_stdout_task.cancel()
            subprocess.terminate()
            if sys.platform != "win32":
                subprocess.send_signal(signal.SIGINT)
            await asyncio.wait_for(subprocess.wait(), timeout=10)
            await read_stderr_task
            await read_stdout_task
            subprocess = None
        if subprocess:
            logger.info("Timeout to terminate mcp-server %s. Kill it.", config.name)
            with suppress(TimeoutError, ProcessLookupError, asyncio.CancelledError):
                read_stderr_task.cancel()
                read_stdout_task.cancel()
                subprocess.kill()
                await asyncio.wait_for(subprocess.wait(), timeout=10)
                await read_stderr_task
                await read_stdout_task
