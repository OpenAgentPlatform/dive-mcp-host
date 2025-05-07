"""Log management for MCP servers.

Each MCP server has a `LogBuffer` that collects important logs,
which is registered to a `LogManager` that writes logs to files and
provides a `listen_log` method that users can listen to log updates.

                      ┌────────────────────┐
                      │ API or other stuff │
                      └────────▲───────────┘
                               │
                           listen_log
                        ┌──────┼─────┐           ┌─────────────┐
                        │ LogManager ┼───────────►Write to file│
                        └──────▲─────┘           └─────────────┘
                               │
                               │
                               │  register_buffer
                               └──────────────────┐
                                                  │
┌─────────────────────────────────────────────────┼─────┐
│                          McpServer              │     │
│                                                 │     │
│                                                 │     │
│                                                 │     │
│┌─────────────────┐      ┌────────┐              │     │
││MCP Server stdio ├──────►LogProxy┼─────┐        │     │
│└─────────────────┘      └────────┘     │        │     │
│                                        │        │     │
│┌────────────────────────┐              │  ┌─────┼───┐ │
││MCP client session error┼──────────────┼──►LogBuffer│ │
│└────────────────────────┘              │  └─────────┘ │
│                                        │              │
│┌────────────────────────┐              │              │
││MCP client state change ┼──────────────┘              │
│└────────────────────────┘                             │
│                                                       │
└───────────────────────────────────────────────────────┘

# Drawn with https://asciiflow.com/
"""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum
from logging import INFO, getLogger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from traceback import format_exception
from typing import TextIO

from pydantic import BaseModel, Field

from dive_mcp_host.host.errors import LogBufferNotFoundError
from dive_mcp_host.host.tools.model_types import ClientState

logger = getLogger(__name__)


class LogEvent(StrEnum):
    """Log event type."""

    STATUS_CHANGE = "status_change"
    STDERR = "stderr"
    STDOUT = "stdout"
    SESSION_ERROR = "session_error"

    # extra events for api
    STREAMING_ERROR = "streaming_error"


class LogMsg(BaseModel):
    """Structure of logs."""

    event: LogEvent
    body: str
    mcp_server_name: str
    client_state: ClientState | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LogBuffer:
    """Log buffer with limited size, supports adding listeners to watch new logs.

    Add logs to the buffer:
        Use `push_logs` or other specific methods
        (e.g. `push_session_error`, `push_state_change`).

    Watch log updates:
        Use `add_listener` context manager to add a `listener` to the buffer.
        Listener is an async function that will be called whenever
        a new log is added to the buffer.
        When the the listener is first added, it will be called with all the
        logs in the buffer (one by one).

    Example:
        ```python
        # create a log buffer
        buffer = LogBuffer(size=1000, name="mcp_server")

        # push a log to the buffer
        msg = LogMsg(event=LogEvent.STDERR, body="hello", mcp_server_name="mcp_server")
        await buffer.push_log(msg)


        async def listener(log: LogMsg) -> None:
            print(log)


        # The listener is a context manager,
        # users decide how long it will listen to the buffer.
        async with buffer.add_listener(listener):
            await asyncio.sleep(10)
        ```
    """

    def __init__(self, size: int = 1000, name: str = "") -> None:
        """Initialize the log buffer."""
        self._size = size
        self._logs: list[LogMsg] = []
        self._listeners: list[Callable[[LogMsg], Coroutine[None, None, None]]] = []
        self._name = name
        self._client_state: ClientState = ClientState.INIT

    @property
    def name(self) -> str:
        """Get the name of the buffer."""
        return self._name

    def _listener_wrapper(
        self,
        listener: Callable[[LogMsg], Coroutine[None, None, None]],
    ) -> Callable[[LogMsg], Coroutine[None, None, None]]:
        """Wrap the listener to handle exceptions."""

        async def _wrapper(msg: LogMsg) -> None:
            try:
                await listener(msg)
            except Exception:
                logger.exception("listener exception")

        return _wrapper

    async def push_session_error(
        self,
        inpt: BaseExceptionGroup | BaseException,
    ) -> None:
        """Push the session error to the log buffer."""
        msg = LogMsg(
            event=LogEvent.SESSION_ERROR,
            body="".join(format_exception(inpt)),
            mcp_server_name=self.name,
            client_state=self._client_state,
        )
        await self.push_log(msg)

    async def push_state_change(
        self,
        inpt: str,
        state: ClientState,
    ) -> None:
        """Push the client status change to the log buffer."""
        self._client_state = state
        msg = LogMsg(
            event=LogEvent.STATUS_CHANGE,
            body=inpt,
            mcp_server_name=self.name,
            client_state=self._client_state,
        )
        await self.push_log(msg)

    async def push_stdout(
        self,
        inpt: str,
    ) -> None:
        """Push the stdout log to the log buffer."""
        msg = LogMsg(
            event=LogEvent.STDOUT,
            body=inpt,
            mcp_server_name=self.name,
            client_state=self._client_state,
        )
        await self.push_log(msg)

    async def push_stderr(
        self,
        inpt: str,
    ) -> None:
        """Push the stderr log to the log buffer."""
        msg = LogMsg(
            event=LogEvent.STDERR,
            body=inpt,
            mcp_server_name=self.name,
            client_state=self._client_state,
        )
        await self.push_log(msg)

    async def push_log(self, log: LogMsg) -> None:
        """Add a log to the buffer, all listener functions will be called."""
        self._logs.append(log)
        if len(self._logs) > self._size:
            self._logs.pop(0)

        async with asyncio.TaskGroup() as group:
            for listener in self._listeners:
                group.create_task(listener(log))

    def get_logs(self) -> list[LogMsg]:
        """Retrieve all logs."""
        return self._logs

    @asynccontextmanager
    async def add_listener(
        self, listener: Callable[[LogMsg], Coroutine[None, None, None]]
    ) -> AsyncIterator[None]:
        """Add a listener to the buffer.

        Reads all the logs in the buffer and listens for new logs.
        The listener is a context manager,
        user can decide how long it will listen to the buffer.

        Example:
            ```python
            async def listener(log: LogMsg) -> None:
                print(log)


            async with buffer.add_listener(listener):
                await asyncio.sleep(10)
            ```
        """
        for i in self._logs:
            await listener(i)
        _listener = self._listener_wrapper(listener)
        self._listeners.append(_listener)
        try:
            yield
        except Exception:
            logger.exception("add listener error")
        finally:
            self._listeners.remove(_listener)


class LogProxy:
    """Proxy stderr logs."""

    def __init__(
        self,
        callback: Callable[[str], Coroutine[None, None, None]],
        mcp_server_name: str,
        stdio: TextIO = sys.stderr,
    ) -> None:
        """Initialize the proxy."""
        self._stdio = stdio
        self._callback = callback
        self._mcp_server_name = mcp_server_name

    async def write(self, s: str) -> None:
        """Write logs."""
        await self._callback(s)
        self._stdio.write(s)

    async def flush(self) -> None:
        """Flush the logs."""
        self._stdio.flush()


class _LogFile:
    """Rotating log file by days."""

    def __init__(self, name: str, log_dir: Path, rotation_files: int = 5) -> None:
        self._name = f"{name}.log"
        self._path = log_dir / self._name
        self._logger = getLogger(self._name)
        self._logger.setLevel(INFO)
        self._logger.propagate = False
        handler = TimedRotatingFileHandler(
            self._path,
            when="D",
            interval=1,
            backupCount=rotation_files,
            encoding="utf-8",
        )
        self._logger.addHandler(handler)

    async def __call__(self, log: LogMsg) -> None:
        self._logger.info(log.model_dump_json())


class LogManager:
    """Log Manger for MCP Servers.

    `LogBuffers` that are registered to the log manager will have
    their logs written to files.

    Users can listen to log updates from specific MCP servers by
    calling `listen_log`.
    Which acts like `LogBuffer`'s `add_listener` method, a context
    manager that listens to log updates until user exit the context.

    Example:
        ```python
        log_dir = Path("/var/log/dive_mcp_host")
        dummy_log = LogMsg(
            event=LogEvent.STDERR,
            body="hello",
            mcp_server_name="mcp_server",
        )

        # create a log manager
        log_manager = LogManager(log_dir=log_dir)

        # create a log buffer
        buffer = LogBuffer(name="mcp_server")


        async def listener(log: LogMsg) -> None:
            print(log)


        # register the buffer and listen to log updates
        async with (
            log_manager.register_buffer(buffer),
            log_manager.listen_log(buffer.name, listener),
        ):
            await buffer.push_log(dummy_log)
            await buffer.push_log(dummy_log)

            await asyncio.sleep(10)
        ```
    """

    def __init__(self, log_dir: Path, rotation_files: int = 5) -> None:
        """Initialize the log manager.

        Args:
            log_dir: The directory to store the logs.
            rotation_files: The number of rotation files per mcp server.
        """
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[str, LogBuffer] = {}
        self._rotation_files = rotation_files

    @asynccontextmanager
    async def register_buffer(self, buffer: LogBuffer) -> AsyncGenerator[None, None]:
        """Register a buffer to the log manager.

        The manager will write log to files.
        """
        self._buffers[buffer.name] = buffer
        log_file = _LogFile(buffer.name, self._log_dir, self._rotation_files)
        async with buffer.add_listener(log_file):
            try:
                yield
            except Exception:
                logger.exception("register buffer error")
            finally:
                self._buffers.pop(buffer.name)

    @asynccontextmanager
    async def listen_log(
        self,
        name: str,
        listener: Callable[[LogMsg], Coroutine[None, None, None]],
    ) -> AsyncGenerator[None, None]:
        """Listen to log updates from a specific MCP server.

        The listener is a context manager,
        user can decide how long it will listen to the buffer.

        Only buffers that are registered to the log manager
        can be listened to. If the buffer is not registered,
        `LogBufferNotFoundError` will be raised.

        Example:
            ```python
            async def listener(log: LogMsg) -> None:
                print(log)


            async with log_manager.listen_log(buffer.name, listener):
                await asyncio.sleep(10)
            ```
        """
        buffer = self._buffers.get(name)
        if buffer is None:
            raise LogBufferNotFoundError(name)
        async with buffer.add_listener(listener):
            yield
