"""Model for the MCP servers."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
    suppress,
)
from datetime import timedelta
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
from mcp import McpError, StdioServerParameters, types
from mcp.client.auth import OAuthFlowError
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_core import to_json

from dive_mcp_host.host.agents.agent_factory import ConfigurableKey
from dive_mcp_host.host.custom_events import (
    ToolAuthenticationRequired,
    ToolCallProgress,
)
from dive_mcp_host.host.errors import (
    InvalidMcpServerError,
    McpSessionClosedOrFailedError,
    McpSessionGroupError,
    McpSessionNotInitializedError,
)
from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.host.tools.hack import (
    ClientSession,
    create_mcp_http_client_factory,
    stdio_client,
)
from dive_mcp_host.host.tools.local_http_server import local_http_server
from dive_mcp_host.host.tools.log import LogBuffer, LogProxy
from dive_mcp_host.host.tools.model_types import ChatID, ClientState
from dive_mcp_host.host.tools.server_session_store import ServerSessionStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from mcp.shared.message import SessionMessage
    from mcp.shared.session import RequestResponder

    from dive_mcp_host.host.conf import ServerConfig
    from dive_mcp_host.host.tools.oauth import AuthorizationProgress, OAuthManager

    type ReadStreamType = MemoryObjectReceiveStream[SessionMessage | Exception]
    type WriteStreamType = MemoryObjectSendStream[SessionMessage]
    type StreamContextType = AbstractAsyncContextManager[
        tuple[ReadStreamType, WriteStreamType]
    ]

logger = getLogger(__name__)


class ToolInfo(types.Tool):
    """Custom tool info with extra info."""

    enable: bool

    @classmethod
    def from_tool(cls, tool: types.Tool, enable: bool) -> Self:
        """Create from mcp Tool type."""
        return cls(**tool.model_dump(), enable=enable)


class McpServerInfo(BaseModel):
    """MCP server capability and tool list."""

    name: str
    """The name of the MCP server."""
    tools: list[ToolInfo]
    """The tools provided by the MCP server."""
    initialize_result: types.InitializeResult | None
    """The result of the initialize method.

    initialize_result.capabilities: Server capabilities.
    initialize_result.instructions: Server instructions.
    """

    url: str | None = None
    """URL for streamable http / sse."""

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
    RESTART_INTERVAL: float = 1

    def __init__(
        self,
        name: str,
        config: ServerConfig,
        auth_manager: OAuthManager,
        log_buffer_length: int = 1000,
    ) -> None:
        """Initialize the McpToolKit.

        Args:
            name: The name of the MCP server.
            config: The configuration of the MCP server.
            log_buffer_length: The length of the log buffer.
            auth_manager: The OAuth manager to use for the MCP server.
        """
        self.name = name
        self.config = config
        self._auth_manager: OAuthManager = auth_manager
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
        self._exception: BaseException | BaseExceptionGroup | None = None
        self._mcp_tools: list[McpTool] = []
        self._retries: int = 0
        self._last_active: float = 0

        # Background task for the server.
        # Used for stdio, and local http server.
        self._server_task: asyncio.Task | None = None

        # stdio can only have one session at a time
        self._stdio_client_session: ClientSession | None = None

        # Each session is mapped to a chat_id
        self._session_store: ServerSessionStore = ServerSessionStore(self.name)

        # The pid of the server process
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

        self._httpx_client_factory = create_mcp_http_client_factory(
            proxy=str(self.config.proxy) if self.config.proxy else None,
        )

    @property
    def session_count(self) -> int:
        """Retrive the session count."""
        return len(self._session_store)

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
        logger.debug(
            "Client %s initalizing with timeout: %s",
            self.name,
            self.config.initial_timeout,
        )
        async with asyncio.timeout(self.config.initial_timeout):
            # When using stdio, the initialize call may block indefinitely
            self._initialize_result = await session.initialize()
            logger.debug(
                "Client %s initialize result: %s",
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
        tools: list[ToolInfo] = []
        if self._tool_results:
            for tool in self._tool_results.tools:
                enable: bool = True
                if tool.name in self.config.exclude_tools:
                    enable = False
                tools.append(ToolInfo.from_tool(tool, enable))

        return McpServerInfo(
            name=self.name,
            initialize_result=self._initialize_result,
            tools=tools,
            client_status=self._client_status,
            url=self.config.url,
            error=self._exception,
        )

    def _get_enabled_tools(self) -> list[McpTool]:
        result: list[McpTool] = []
        for tool in self._mcp_tools:
            if tool.name in self.config.exclude_tools:
                continue
            result.append(tool)
        return result

    @property
    def mcp_tools(self) -> list[McpTool]:
        """Get the tools."""
        if self._client_status == ClientState.RUNNING:
            return self._get_enabled_tools()
        return []

    def session(
        self,
        chat_id: str = "default",
        auth_handler: Callable[[AuthorizationProgress], Awaitable[None]] | None = None,
    ) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """
        return self._return_session(chat_id, auth_handler)

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
                    self.session_count,
                )
            await self._session_store.cleanup()
            await self._teardown()

    @asynccontextmanager
    async def _session_ctx_mgr_wrapper(
        self,
        chat_id: str,
        session_creator: Callable[[], AbstractAsyncContextManager[ClientSession]],
        restart_client: Callable[[Exception], bool] = lambda _: False,
        auth_handler: Callable[[AuthorizationProgress], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """Get the session ctx mgr from the session store, and handle session errors.

        Args:
            chat_id: The chat id.
            session_creator: The session creator.
            restart_client: The function to determine if the client should be restarted.
                If the exception is not restartable, return False.
                If the exception is restartable, return True.
            auth_handler: The function to handle the authorization progress.

        This wrapper get the session from the session store, and handle the session
        errors.
        When error occurs, the exception will pass to restart_client to determine if
        the client should be restarted. If the client should be restarted, the state
        will be set to RESTARTING.
        """
        try:
            async with self._session_store.get_session_ctx_mgr(
                chat_id,
                session_creator,
                auth_handler,
            ) as session:
                yield session
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
                self._cond.notify_all()

    async def _stdio_client_watcher(self) -> None:
        """Client watcher task.

        Restart the client if need.
        Only this watcher can set the client status to RUNNING / FAILED.
        """
        env = os.environ.copy()
        env.update(self.config.env)
        start_time = time.time()
        while (
            self._retries == 0
            or (time.time() - start_time) < self.config.initial_timeout
        ):
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
                    self._stdio_client_session = session
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
                        if self._client_status == ClientState.RESTARTING:
                            self._retries = 0
                            start_time = time.time()
                            continue
                        return
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
                ValidationError,
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
            self._stdio_client_session = None
            if self._client_status == ClientState.CLOSED:
                logger.info("Client %s closed, stopping watcher", self.name)
                return
            if should_break:
                break
            logger.debug(
                "Retrying client initialization for %s (attempt %d)",
                self.name,
                self._retries,
            )
            await asyncio.sleep(self.RESTART_INTERVAL)

        logger.warning(
            "client for [%s] failed after %d retries %s",
            self.name,
            self._retries,
            self._exception,
        )
        async with self._cond:
            if self._client_status != ClientState.CLOSED:
                await self.__change_state(ClientState.FAILED, None, False)

    async def _stdio_setup(self) -> None:
        """Setup the stdio client."""
        self._server_task = asyncio.create_task(
            self._stdio_client_watcher(),
            name=f"stdio_client_watcher-{self.name}",
        )
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
                    lambda: self.session_count == 0,
                )
            except TimeoutError:
                logger.warning(
                    "Timeout to wait %d sessions to be closed",
                    self.session_count,
                )
        if self._server_task:
            logger.debug("in stdio teardown %s", self._server_task.get_name())
            self._server_task.cancel()
            async with asyncio.timeout(30):
                with suppress(asyncio.CancelledError):
                    await self._server_task

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
                and self._stdio_client_session
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
                        await self._stdio_client_session.send_ping()
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
            if (
                self._client_status == ClientState.RUNNING
                and self._stdio_client_session
            ):
                return self._stdio_client_session
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

    def _stdio_session(
        self,
        chat_id: ChatID,
        auth_handler: Callable[[AuthorizationProgress], Awaitable[None]] | None = None,  # noqa: ARG002
    ) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """

        @asynccontextmanager
        async def _create(**_kwargs: Any) -> AsyncGenerator[ClientSession, None]:
            """Create new session."""
            try:
                session = await self._stdio_wait_for_session()
                await session.initialize()
                yield session
            except Exception:
                logger.exception("stdio session error, chat_id: %s", chat_id)
                raise

        return self._session_ctx_mgr_wrapper("default", _create, lambda _: True)

    def _http_get_client(
        self,
        timeout: float = 30,
        sse_read_timeout: float = 60 * 5,
        auth: httpx.Auth | None = None,
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
                timeout=timeout,
                httpx_client_factory=self._httpx_client_factory,
                sse_read_timeout=sse_read_timeout,
                auth=auth,
            )
        if self.config.transport in ("streamable"):
            return streamablehttp_client(
                url=self.config.url,
                headers={
                    key: value.get_secret_value()
                    for key, value in self.config.headers.items()
                },
                timeout=timedelta(seconds=timeout),
                httpx_client_factory=self._httpx_client_factory,
                sse_read_timeout=timedelta(seconds=sse_read_timeout),
                auth=auth,
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
        async with AsyncExitStack() as stack:
            assert self.config.url
            auth = await stack.enter_async_context(
                self.auth_manager.with_client(
                    self.name,
                    server_url=self.config.url,
                )
            )
            streams = await stack.enter_async_context(
                self._http_get_client(
                    sse_read_timeout=self.config.initial_timeout,
                    timeout=self.config.initial_timeout,
                    auth=auth,
                )
            )
            session = await stack.enter_async_context(
                ClientSession(
                    *[streams[0], streams[1]], message_handler=self._message_handler
                )
            )
            await self._init_tool_info(session)

    async def _http_setup(self) -> None:
        """Setup the http client."""
        unauthenticated = False
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

        except* httpx.HTTPStatusError as eg:
            err_msg = f"Client http error for {self.name}: {eg.exceptions}"
            logger.exception(err_msg)
            self._exception = McpSessionGroupError(err_msg, eg.exceptions)
            for e in eg.exceptions:
                logger.error("http setup error %s", e)
                if (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code == httpx.codes.UNAUTHORIZED
                ):
                    unauthenticated = True

        except* (
            McpError,
            httpx.InvalidURL,
            OAuthFlowError,
            Exception,
            ValidationError,
        ) as eg:
            err_msg = f"Client initialization error for {self.name}: {eg.exceptions}"
            if any(isinstance(e, OAuthFlowError) for e in eg.exceptions):
                unauthenticated = True
            logger.exception(err_msg)
            self._exception = McpSessionGroupError(err_msg, eg.exceptions)

        async with self._cond:
            logger.error("http setup failed %s", self._exception)
            await self.__change_state(
                ClientState.UNAUTHORIZED if unauthenticated else ClientState.FAILED,
                None,
                self._exception,
            )

    async def _http_teardown(self) -> None:
        """Teardown the http client. Do nothing."""
        logger.debug("http teardown")

    def _http_session(
        self,
        chat_id: ChatID,
        auth_handler: Callable[[AuthorizationProgress], Awaitable[None]] | None = None,
    ) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """

        @asynccontextmanager
        async def _create(
            auth_handler: Callable[[AuthorizationProgress], Awaitable[None]]
            | None = None,
            **_kwargs: Any,
        ) -> AsyncGenerator[ClientSession, None]:
            """Create new session."""
            try:
                async with AsyncExitStack() as stack:
                    assert self.config.url
                    auth = await stack.enter_async_context(
                        self.auth_manager.with_client(
                            self.name,
                            auth_callback=auth_handler,
                            server_url=self.config.url,
                            wait_auth=True,
                        )
                    )
                    streams = await stack.enter_async_context(
                        self._http_get_client(auth=auth)
                    )
                    session = await stack.enter_async_context(
                        ClientSession(
                            *[streams[0], streams[1]],
                            message_handler=self._message_handler,
                        )
                    )
                    await session.initialize()
                    yield session
            except Exception:
                logger.exception("http session error, chat_id: %s", chat_id)
                raise

        return self._session_ctx_mgr_wrapper(
            chat_id, _create, auth_handler=auth_handler
        )

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
                ValidationError,
                Exception,
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
        self._server_task = asyncio.create_task(
            self._local_http_process_watcher(),
            name=f"local_http_process_watcher-{self.name}",
        )
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._client_status
                in [ClientState.RUNNING, ClientState.CLOSED, ClientState.FAILED]
            )

    async def _local_http_teardown(self) -> None:
        """Teardown the local http server."""
        if self._server_task:
            logger.debug("in local http teardown %s", self._server_task.get_name())
            self._server_task.cancel()
            async with asyncio.timeout(30):
                with suppress(asyncio.CancelledError):
                    await self._server_task

    def _local_http_session(
        self,
        chat_id: str,
        auth_handler: Callable[[AuthorizationProgress], Awaitable[None]] | None = None,  # noqa: ARG002
    ) -> AbstractAsyncContextManager[ClientSession]:
        """Get the session.

        Only one session can exist at a time for a McpStdioServer instance.

        Returns:
            The context manager for the session.
        """

        @asynccontextmanager
        async def _create(**_kwargs: Any) -> AsyncGenerator[ClientSession, None]:
            """Create new session."""
            try:
                async with (
                    self._http_get_client() as streams,
                    ClientSession(
                        *[streams[0], streams[1]], message_handler=self._message_handler
                    ) as session,
                ):
                    await session.initialize()
                    yield session
            except Exception:
                logger.exception("local http session error, chat_id: %s", chat_id)
                raise

        return self._session_ctx_mgr_wrapper(
            chat_id,
            _create,
            lambda e: isinstance(e, httpx.ConnectError),
        )

    @property
    def auth_manager(self) -> OAuthManager:
        """Get the auth manager."""
        return self._auth_manager

    async def create_oauth_authorization(self) -> AuthorizationProgress:
        """Authorize the OAuth client."""
        if self.config.transport not in ("streamable", "sse"):
            raise RuntimeError(
                "Only streamable and sse transport is supported for oauth"
            )

        if self.auth_manager is None:
            raise RuntimeError("OAuth manager is not initialized")

        assert self.config.url
        return await self.auth_manager.authorization_task(
            self.name,
            self._http_get_client,
            {
                "sse_read_timeout": self.config.initial_timeout,
                "timeout": self.config.initial_timeout,
            },
            self.config.url,
        )


async def _stream_writer_bridge(custom_event_queue: asyncio.Queue) -> None:
    """Stream writer."""
    # XXX: the stream writer sometimes is not available, so we need to handle the error.
    try:
        stream_writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        logger.debug("get_stream_writer error", exc_info=True)
        stream_writer = None

    while True:
        a = await custom_event_queue.get()
        logger.debug("custom_event_queue: %s", a)
        if a is None:
            return
        if stream_writer:
            try:
                stream_writer(a)
            except Exception as e:
                logger.exception("stream_writer error %s", e)
                stream_writer = None


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
        config: RunnableConfig,
        **kwargs: dict[str, Any],
    ) -> str:
        """Run the tool."""
        custom_event_queue = asyncio.Queue()

        async def progress_callback(
            progress: float, total: float | None, message: str | None
        ) -> None:
            """Progress callback."""
            try:
                await custom_event_queue.put(
                    (
                        ToolCallProgress.NAME,
                        ToolCallProgress(
                            progress=progress,
                            total=total,
                            message=message,
                            tool_call_id=tool_call_id,
                        ),
                    )
                )
            except Exception as e:
                logger.exception("progress_callback error %s", e)

        async def auth_callback(progress: AuthorizationProgress) -> None:
            """Auth callback."""
            logger.info("auth_callback: %s", progress)
            if progress.type == "wait_code" and progress.auth_url:
                await custom_event_queue.put(
                    (
                        ToolAuthenticationRequired.NAME,
                        ToolAuthenticationRequired(
                            server_name=self.mcp_server.name, auth_url=progress.auth_url
                        ),
                    )
                )

        tool_call_id = config.get("metadata", {}).get("tool_call_id", "")
        chat_id = config.get("configurable", {}).get(
            ConfigurableKey.THREAD_ID, "default"
        )
        abort_signal: asyncio.Event | None = config.get("configurable", {}).get(
            ConfigurableKey.ABORT_SIGNAL, None
        )

        if not self.kwargs_arg and len(kwargs) == 1 and "kwargs" in kwargs:
            if isinstance(kwargs["kwargs"], str):
                with suppress(JSONDecodeError):
                    kwargs = json_loads(kwargs["kwargs"])
            else:
                kwargs = kwargs["kwargs"]
        logger.debug(
            "Executing tool %s.%s with args: %s", self.toolkit_name, self.name, kwargs
        )

        async def _abort_task(tool_task: asyncio.Task) -> None:
            assert abort_signal
            await abort_signal.wait()
            tool_task.cancel()

        abort_task = None

        stream_writer_task = asyncio.create_task(
            _stream_writer_bridge(custom_event_queue),
            name=f"stream_writer-{self.name}",
        )

        try:
            async with self.mcp_server.session(chat_id, auth_callback) as session:
                try:
                    tool_task = asyncio.create_task(
                        session.call_tool(
                            self.name,
                            arguments=kwargs,
                            progress_callback=progress_callback,
                        )
                    )
                    if abort_signal:
                        abort_task = asyncio.create_task(_abort_task(tool_task))
                    result = await tool_task
                except asyncio.CancelledError as canceld:
                    logger.warning("tool call cancelled %s", canceld)
                    result = types.CallToolResult(
                        content=[types.TextContent(type="text", text="<user_aborted>")],
                    )
                finally:
                    if abort_task:
                        abort_task.cancel()
        finally:
            with suppress(Exception):
                custom_event_queue.put_nowait(None)
            await stream_writer_task

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
