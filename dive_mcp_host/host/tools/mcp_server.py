"""Model for the MCP servers."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from json import JSONDecodeError
from json import loads as json_loads
from logging import getLogger
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Literal, Self

import anyio
import httpx
from langchain_core.runnables import (
    RunnableConfig,  # noqa: TC002 Langchain needs this to get type hits in runtime.
)
from langchain_core.tools import BaseTool, ToolException
from langgraph.config import get_stream_writer
from mcp import ClientSession, McpError, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json

from dive_mcp_host.host.errors import (
    InvalidMcpServerError,
    McpSessionClosedOrFailedError,
    McpSessionGroupError,
    McpSessionNotInitializedError,
)
from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.host.tools.local_http_server import local_http_server
from dive_mcp_host.host.tools.log import LogBuffer, LogProxy
from dive_mcp_host.host.tools.model_types import ClientState, ToolCallProgress
from dive_mcp_host.host.tools.stdio_server import stdio_client

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from mcp.shared.message import SessionMessage
    from mcp.shared.session import RequestResponder

    from dive_mcp_host.host.conf import ServerConfig

    type ReadStreamType = MemoryObjectReceiveStream[SessionMessage | Exception]
    type WriteStreamType = MemoryObjectSendStream[SessionMessage]
    type StreamContextType = AbstractAsyncContextManager[
        tuple[ReadStreamType, WriteStreamType]
    ]

logger = getLogger(__name__)


class McpServerInfo(BaseModel):
    """MCP server capability and tool list."""

    name: str
    """The name of the MCP server."""
    tools: list[types.Tool]
    """The tools provided by the MCP server."""
    initialize_result: types.InitializeResult | None
    """The result of the initialize method.

    initialize_result.capabilities: Server capabilities.
    initialize_result.instructions: Server instructions.
    """

    error: BaseException | BaseExceptionGroup | None
    """The error that occurred of the MCP server."""

    client_status: ClientState
    """The status of the client: RUNNING, CLOSED, RESTARTING, or INIT."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def error_str(self) -> str | None:
        """Print the entire error message."""
        if self.error is None:
            return None
        return "\n".join(format_exception(self.error))


class McpServer(ContextProtocol):
    """Base class for MCP servers."""

    RETRY_LIMIT: int = 3
    KEEP_ALIVE_INTERVAL: float = 60
    RESTART_INTERVAL: float = 3

    def __init__(
        self,
        name: str,
        config: ServerConfig,
        log_buffer_length: int = 1000,
    ) -> None:
        """Initialize the McpToolKit."""
        self.name = name
        self.config = config
        self._log_buffer = LogBuffer(name=name, size=log_buffer_length)
        self._stderr_log_proxy = LogProxy(
            callback=self._log_buffer.push_stderr,
            mcp_server_name=self.name,
            stdio=sys.stderr,
        )
        self._stdout_log_proxy = LogProxy(
            callback=self._log_buffer.push_stdout,
            mcp_server_name=self.name,
            stdio=sys.stdout,
        )
        self._cond = asyncio.Condition()
        """The condition variable to synchronize access to shared variables."""
        self._client_status: ClientState = ClientState.INIT
        self._tool_results: types.ListToolsResult | None = None
        self._initialize_result: types.InitializeResult | None = None
        self._session_count: int = 0
        self._exception: BaseException | BaseExceptionGroup | None = None
        self._mcp_tools: list[McpTool] = []
        self._retries: int = 0
        self._last_active: float = 0

        self._task: asyncio.Task | None = None

        self._session: ClientSession | None = None
        self._pid: int | None = None

        """Methods for different server types."""
        if self.config.command:
            if self.config.transport == "stdio":
                self._setup = self._stdio_setup
                self._teardown = self._stdio_teardown
                self._return_session = self._stdio_session
            if self.config.url:
                self._setup = self._local_http_setup
                self._teardown = self._local_http_teardown
                self._return_session = self._local_http_session
        elif self.config.url:
            self._setup = self._http_setup
            self._teardown = self._http_teardown
            self._return_session = self._http_session
        else:
            raise InvalidMcpServerError(self.config.name, "Invalid server config")

    async def _message_handler(
        self,
        message: RequestResponder[types.ServerRequest, types.ClientResult]
        | types.ServerNotification
        | Exception,
    ) -> None:
        """Used for handling mcp special responses.

        Such as:
        - Exception (Literal python exception)
        - ProgressResult (ServerNotification) ... etc
        """
        logger.info(
            "handling message for %s, type: %s, content: %s",
            self.name,
            type(message).__name__,
            message,
        )

        if isinstance(message, Exception):
            raise message

    async def _init_tool_info(self, session: ClientSession) -> None:
        """Initialize the session."""
        async with asyncio.timeout(10):
            # When using stdio, the initialize call may block indefinitely
            self._initialize_result = await session.initialize()
            logger.debug(
                "Client %s initializing, result: %s",
                self.name,
                self._initialize_result,
            )
        tool_results = await session.list_tools()
        self._last_active = time.time()
        mcp_tools = [McpTool.from_tool(tool, self) for tool in tool_results.tools]
        logger.debug(
            "Client %s initialized successfully with %d tools",
            self.name,
            len(mcp_tools),
        )
        async with self._cond:
            self._tool_results = tool_results
            self._mcp_tools = mcp_tools
            self._exception = None
            self._retries = 0
            await self.__change_state(ClientState.RUNNING, None, None)
            logger.debug(
                "Client %s initialized successfully with %d tools",
                self.name,
                len(mcp_tools),
            )

    @property
    def log_buffer(self) -> LogBuffer:
        """Get the log buffer."""
        return self._log_buffer

    @property
    def server_info(self) -> McpServerInfo:
        """Get the server info."""
        return McpServerInfo(
            name=self.name,
            initialize_result=self._initialize_result,
            tools=self._tool_results.tools if self._tool_results is not None else [],
            client_status=self._client_status,
            error=self._exception,
        )

    @property
    def mcp_tools(self) -> list[McpTool]:
        """Get the tools."""
        if self._client_status == ClientState.RUNNING:
            return self._mcp_tools
        return []

    def session(self) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """
        return self._return_session()

    async def wait(self, states: list[ClientState]) -> bool:
        """Wait until the client is in the given state or in the failed or closed state.

        Returns:
            True if the client is in the given state.
        """
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._client_status
                in [
                    *states,
                    ClientState.FAILED,
                    ClientState.CLOSED,
                ],
            )
            return self._client_status in states

    async def __change_state(
        self,
        new_state: ClientState,
        orig_state: list[ClientState] | None,
        e: BaseException | None | Literal[False],
    ) -> None:
        """Change the client state.

        The caller have to acquire self._cond before calling this function.
        It only notify the condition variable if the state is changed.

        Args:
            new_state: The new state.
            orig_state: The original state.
              Change to new_state if orig_state is None
              or self._client_status == orig_state.
            e: The exception that occurred.
              If e is not False, set self._exception to e.
        """
        if orig_state is None or self._client_status in orig_state:
            if e is not False:
                self._exception = e
            self._client_status = new_state
            log_msg = f"client status changed, {self.name} {new_state}, error: {e}"
            logger.debug(log_msg)
            await self._log_buffer.push_state_change(inpt=log_msg, state=new_state)
            self._cond.notify_all()

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """Get the langchain tools for the MCP servers."""
        await self._setup()
        try:
            yield self
        finally:
            async with self._cond:
                await self.__change_state(
                    ClientState.CLOSED,
                    [ClientState.INIT, ClientState.RUNNING],
                    False,
                )
                logger.debug(
                    "%s: wait all sessions to be closed. now is %s",
                    self.name,
                    self._session_count,
                )
            await self._teardown()

    @asynccontextmanager
    async def _session_wrapper(
        self,
        restart_client: Callable[[Exception], bool] = lambda _: False,
    ) -> AsyncGenerator[None, None]:
        """Wrap the session to suppress session errors and tracking session count."""
        try:
            self._session_count += 1
            yield
        except (ToolException, McpError) as e:
            logger.error("Tool exception for %s: %s", self.name, e)
            raise
        except (
            httpx.HTTPError,
            httpx.StreamError,
            httpx.TimeoutException,
            httpx.TooManyRedirects,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            anyio.EndOfStream,
            Exception,  # Before we know the exception type
        ) as e:
            if restart_client(e):
                async with self._cond:
                    await self.__change_state(
                        ClientState.RESTARTING, [ClientState.RUNNING], e
                    )
                logger.warning(
                    "mcp server %s failed, restarting, %s",
                    self.name,
                    e,
                    extra={
                        "mcp_server": self.name,
                        "client_status": self._client_status,
                    },
                )
            else:
                logger.warning(
                    "mcp server %s failed, %s",
                    self.name,
                    e,
                    extra={
                        "mcp_server": self.name,
                        "client_status": self._client_status,
                    },
                )
            raise
        finally:
            async with self._cond:
                self._session_count -= 1
                self._cond.notify_all()

    async def _stdio_client_watcher(self) -> None:  # noqa: C901, PLR0915
        """Client watcher task.

        Restart the client if need.
        Only this watcher can set the client status to RUNNING / FAILED.
        """
        env = os.environ.copy()
        env.update(self.config.env)
        while True:
            should_break = False
            try:
                logger.debug("Attempting to initialize client %s", self.name)
                async with (
                    stdio_client(
                        server=StdioServerParameters(
                            command=self.config.command,
                            args=self.config.args,
                            env=env,
                        ),
                        errlog=self._stderr_log_proxy,
                    ) as (stream_read, stream_send, pid),
                    ClientSession(
                        stream_read, stream_send, message_handler=self._message_handler
                    ) as session,
                ):
                    self._session = session
                    self._pid = pid
                    await self._init_tool_info(session)
                    async with self._cond:
                        await self._cond.wait_for(
                            lambda: self._client_status
                            in [ClientState.CLOSED, ClientState.RESTARTING],
                        )
                        logger.debug(
                            "client watcher %s exited. status: %s",
                            self.name,
                            self._client_status,
                        )
            except* ProcessLookupError as eg:
                # this raised when a stdio process is exited
                # and the initialize call is timeout
                err_msg = f"ProcessLookupError for {self.name}: {eg.exceptions}"
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(
                    err_msg,
                    eg.exceptions,
                )
                should_break = True
            except* (
                FileNotFoundError,
                PermissionError,
                McpError,
                httpx.ConnectError,
                httpx.InvalidURL,
                httpx.TooManyRedirects,
                httpx.ConnectTimeout,
            ) as eg:
                err_msg = (
                    f"Client initialization error for {self.name}: {eg.exceptions}"
                )
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(err_msg, eg.exceptions)
                should_break = True
            except* httpx.HTTPStatusError as eg:
                err_msg = f"Client http error for {self.name}: {eg.exceptions}"
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(err_msg, eg.exceptions)
                for e in eg.exceptions:
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code < 500  # noqa: PLR2004
                        and e.response.status_code != 429  # noqa: PLR2004
                    ):
                        should_break = True
                        break
            except* asyncio.CancelledError as e:
                should_break = True
                logger.debug("Client watcher cancelled for %s", self.name)
            except* BaseException as eg:
                err_msg = (
                    f"Client initialization error for {self.name}: {eg.exceptions}"
                )
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(err_msg, eg.exceptions)

            if self._exception:
                await self._log_buffer.push_session_error(self._exception)

            self._retries += 1
            self._session = None
            if self._client_status == ClientState.CLOSED:
                logger.info("Client %s closed, stopping watcher", self.name)
                return
            if self._retries >= self.RETRY_LIMIT or should_break:
                logger.warning(
                    "client for [%s] failed after %d retries %s",
                    self.name,
                    self._retries,
                    self._exception,
                )
                async with self._cond:
                    if self._client_status != ClientState.CLOSED:
                        await self.__change_state(ClientState.FAILED, None, False)
                return
            logger.debug(
                "Retrying client initialization for %s (attempt %d/%d)",
                self.name,
                self._retries,
                self.RETRY_LIMIT,
            )
            await asyncio.sleep(self.RESTART_INTERVAL)

    async def _stdio_setup(self) -> None:
        """Setup the stdio client."""
        self._task = asyncio.create_task(self._stdio_client_watcher())
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._client_status
                in [ClientState.RUNNING, ClientState.CLOSED, ClientState.FAILED]
            )

    async def _stdio_teardown(self) -> None:
        """Teardown the stdio client."""
        async with self._cond, asyncio.timeout(30):
            try:
                await self._cond.wait_for(
                    lambda: self._session_count == 0,
                )
            except TimeoutError:
                logger.warning(
                    "Timeout to wait %d sessions to be closed",
                    self._session_count,
                )
        if self._task:
            logger.debug("in stdio teardown %s", self._task)
            self._task.cancel()
            async with asyncio.timeout(30):
                with suppress(asyncio.CancelledError):
                    await self._task

    async def _stdio_wait_for_session(self) -> ClientSession:
        """Only called by the session context manager."""
        for retried in range(self.RETRY_LIMIT):
            await self.wait(
                [
                    ClientState.RUNNING,
                ]
            )
            if self._client_status in [ClientState.FAILED, ClientState.CLOSED]:
                logger.error(
                    "Session failed or closed for %s: %s",
                    self.name,
                    self._client_status,
                )
                raise McpSessionClosedOrFailedError(self.name, self._client_status.name)
            now = time.time()
            if (
                self._client_status == ClientState.RUNNING
                and self._session
                and (now - self._last_active > self.KEEP_ALIVE_INTERVAL)
            ):
                # check if the session is still active
                try:
                    logger.debug(
                        "Checking session health for %s (inactive for %.1f seconds)",
                        self.name,
                        now - self._last_active,
                    )
                    async with asyncio.timeout(10):
                        await self._session.send_ping()
                        self._last_active = time.time()
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "Keep-alive error for %s: %s",
                        self.name,
                        e,
                        extra={
                            "mcp_server": self.name,
                            "client_status": self._client_status,
                        },
                    )
                    async with self._cond:
                        await self.__change_state(
                            ClientState.RESTARTING, [ClientState.RUNNING], e
                        )
            if self._client_status == ClientState.RUNNING and self._session:
                return self._session
            if retried < self.RETRY_LIMIT - 1:
                logger.warning(
                    "Session not initialized, retrying, %s status: %s (attempt %d/%d)",
                    self.name,
                    self._client_status,
                    retried + 1,
                    self.RETRY_LIMIT,
                    extra={
                        "mcp_server": self.name,
                        "client_status": self._client_status,
                    },
                )
                await asyncio.sleep(self.RESTART_INTERVAL)
        logger.error(
            "Session not initialized after %d attempts, %s status: %s",
            self.RETRY_LIMIT,
            self.name,
            self._client_status,
            extra={"mcp_server": self.name, "client_status": self._client_status},
        )
        raise McpSessionNotInitializedError(self.name)

    def _stdio_session(self) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """

        @asynccontextmanager
        async def session_ctx() -> AsyncGenerator[ClientSession, None]:
            """Get the session.

            If the session is inactive for a long time, ping it first
            to check if it is still active.
            """
            session = await self._stdio_wait_for_session()
            async with self._session_wrapper(restart_client=lambda _: True):
                await session.initialize()
                yield session

        return session_ctx()

    def _http_get_client(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[ReadStreamType, WriteStreamType]
        | tuple[
            ReadStreamType,
            WriteStreamType,
            Callable[[], str | None],
        ]
    ]:
        assert self.config.url, "url is required"
        if self.config.transport in ("sse", None):
            return sse_client(
                url=self.config.url,
                headers={
                    key: value.get_secret_value()
                    for key, value in self.config.headers.items()
                },
            )
        if self.config.transport in ("streamable"):
            return streamablehttp_client(
                url=self.config.url,
                headers={
                    key: value.get_secret_value()
                    for key, value in self.config.headers.items()
                },
            )
        if self.config.transport == "websocket":
            return websocket_client(
                url=self.config.url,
            )
        raise InvalidMcpServerError(
            self.name, "Only sse and websocket are supported for url."
        )

    async def _http_init_client(self) -> None:
        """Initialize the HTTP client."""
        async with (
            self._http_get_client() as streams,
            ClientSession(*streams, message_handler=self._message_handler) as session,
        ):
            await self._init_tool_info(session)

    async def _http_setup(self) -> None:
        """Setup the http client."""
        self._retries = 0
        for _ in range(self.RETRY_LIMIT):
            should_break = False
            try:
                await self._http_init_client()
                async with self._cond:
                    await self.__change_state(ClientState.RUNNING, None, None)
                return
            except* (
                httpx.ConnectError,
                httpx.TooManyRedirects,
                httpx.ConnectTimeout,
            ) as eg:
                logger.error("http setup error %s", eg.exceptions)
                self._exception = McpSessionGroupError(
                    f"Client connection error for {self.name}: {eg.exceptions}",
                    eg.exceptions,
                )
            except* (
                McpError,
                httpx.InvalidURL,
            ) as eg:
                err_msg = (
                    f"Client initialization error for {self.name}: {eg.exceptions}"
                )
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(err_msg, eg.exceptions)
                should_break = True
            except* httpx.HTTPStatusError as eg:
                err_msg = f"Client http error for {self.name}: {eg.exceptions}"
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(err_msg, eg.exceptions)
                for e in eg.exceptions:
                    logger.error("http setup error %s", e)
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code < 500  # noqa: PLR2004
                        and e.response.status_code != 429  # noqa: PLR2004
                    ):
                        should_break = True
            if should_break:
                break
            self._retries += 1
            await asyncio.sleep(self.RESTART_INTERVAL)
        async with self._cond:
            logger.error("http setup failed %s", self._exception)
            await self.__change_state(ClientState.FAILED, None, self._exception)

    async def _http_teardown(self) -> None:
        """Teardown the http client. Do nothing."""
        logger.debug("http teardown")

    def _http_session(self) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """

        @asynccontextmanager
        async def session_ctx() -> AsyncGenerator[ClientSession, None]:
            """Get the session.

            If the session is inactive for a long time, ping it first
            to check if it is still active.
            """
            async with (
                self._http_get_client() as streams,
                ClientSession(
                    *streams, message_handler=self._message_handler
                ) as session,
                self._session_wrapper(),
            ):
                await session.initialize()
                yield session

        return session_ctx()

    async def _local_http_process_watcher(self) -> None:
        """Watcher the local http server process."""
        env = os.environ.copy()
        env.update(self.config.env)
        while True:
            should_break = False
            try:
                async with local_http_server(
                    config=self.config,
                    stderrlog=self._stderr_log_proxy,
                    stdoutlog=self._stdout_log_proxy,
                    env=env,
                ) as proc:
                    async with self._cond:
                        self._init_result, tool_results, self._pid = proc
                        self._mcp_tools = [
                            McpTool.from_tool(tool, self) for tool in tool_results.tools
                        ]
                        self._exception = None
                        self._retries = 0
                        await self.__change_state(ClientState.RUNNING, None, None)
                    logger.debug(
                        "Client %s initialized successfully with %d tools",
                        self.name,
                        len(self._mcp_tools),
                    )
                    async with self._cond:
                        await self._cond.wait_for(
                            lambda: self._client_status
                            in [ClientState.CLOSED, ClientState.RESTARTING],
                        )
                        logger.debug(
                            "client watcher %s exited. status: %s",
                            self.name,
                            self._client_status,
                        )
            except (
                InvalidMcpServerError,
                ProcessLookupError,
                FileNotFoundError,
                PermissionError,
                McpError,
                httpx.InvalidURL,
                httpx.TooManyRedirects,
            ) as e:
                logger.exception("Error initializing http server: %s", e)
                self._exception = McpSessionGroupError(
                    f"Error initializing http server: {e}", [e]
                )
                should_break = True
            except asyncio.CancelledError:
                should_break = True
            if self._exception:
                await self._log_buffer.push_session_error(self._exception)
            self._retries += 1

            if self._retries >= self.RETRY_LIMIT or should_break:
                logger.warning(
                    "Client for [%s] failed after %d retries %s",
                    self.name,
                    self._retries,
                    self._exception,
                )
                async with self._cond:
                    if self._client_status != ClientState.CLOSED:
                        await self.__change_state(ClientState.FAILED, None, False)
                return

    async def _local_http_setup(self) -> None:
        """Setup the local http server."""
        self._task = asyncio.create_task(self._local_http_process_watcher())
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._client_status
                in [ClientState.RUNNING, ClientState.CLOSED, ClientState.FAILED]
            )

    async def _local_http_teardown(self) -> None:
        """Teardown the local http server."""
        if self._task:
            self._task.cancel()
            async with asyncio.timeout(30):
                with suppress(asyncio.CancelledError):
                    await self._task

    def _local_http_session(self) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """

        @asynccontextmanager
        async def session_ctx() -> AsyncGenerator[ClientSession, None]:
            """Get the session.

            If the session is inactive for a long time, ping it first
            to check if it is still active.
            """
            async with (
                self._http_get_client() as streams,
                ClientSession(
                    *streams, message_handler=self._message_handler
                ) as session,
                self._session_wrapper(
                    restart_client=lambda e: isinstance(e, httpx.ConnectError)
                ),
            ):
                await session.initialize()
                yield session

        return session_ctx()


class McpTool(BaseTool):
    """A tool for the MCP."""

    toolkit_name: str
    description: str = ""
    mcp_server: McpServer
    kwargs_arg: bool = False

    def _run(
        self,
        _config: RunnableConfig,
        **kwargs: dict[str, Any],
    ) -> str:
        """Run the tool."""
        return asyncio.run(self._arun(_config, **kwargs))

    async def _arun(
        self,
        _config: RunnableConfig,
        **kwargs: dict[str, Any],
    ) -> str:
        """Run the tool."""

        async def progress_callback(
            progress: float, total: float | None, message: str | None
        ) -> None:
            """Progress callback."""
            get_stream_writer()(
                (
                    "tool_call_progress",
                    ToolCallProgress(
                        progress=progress,
                        total=total,
                        message=message,
                        tool_call_id=tool_call_id,
                    ),
                )
            )

        tool_call_id = _config.get("metadata", {}).get("tool_call_id", "")

        if not self.kwargs_arg and len(kwargs) == 1 and "kwargs" in kwargs:
            if isinstance(kwargs["kwargs"], str):
                with suppress(JSONDecodeError):
                    kwargs = json_loads(kwargs["kwargs"])
            else:
                kwargs = kwargs["kwargs"]
        logger.debug(
            "Executing tool %s.%s with args: %s", self.toolkit_name, self.name, kwargs
        )
        async with self.mcp_server.session() as session:
            result = await session.call_tool(
                self.name,
                arguments=kwargs,
                progress_callback=progress_callback,
            )
        content = to_json(result.content).decode()
        if result.isError:
            logger.error(
                "Tool execution failed for %s.%s: %s",
                self.toolkit_name,
                self.name,
                content,
            )
        logger.debug("Tool %s.%s executed successfully", self.toolkit_name, self.name)
        return content

    @classmethod
    def from_tool(cls, tool: types.Tool, mcp_server: McpServer) -> Self:
        """Create a McpTool from a langchain tool."""
        input_schema = tool.inputSchema.copy()
        if "properties" not in input_schema:
            input_schema["properties"] = {}
        return cls(
            toolkit_name=mcp_server.name,
            name=tool.name,
            description=tool.description or "",
            mcp_server=mcp_server,
            kwargs_arg="kwargs" in tool.inputSchema,
            args_schema=input_schema,
        )
