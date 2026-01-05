"""Copy of mcp.client.stdio.stdio_client."""

import asyncio
import logging
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Protocol

import anyio
import anyio.abc
import anyio.lowlevel
from anyio.abc import Process
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from mcp import types
from mcp.client.stdio import StdioServerParameters, get_default_environment
from mcp.os.posix.utilities import terminate_posix_process_tree
from mcp.os.win32.utilities import (
    FallbackProcess,
    get_windows_executable_command,
    terminate_windows_process_tree,
)
from mcp.shared.message import SessionMessage

from dive_mcp_host.host.tools.log import LogProxy

logger = logging.getLogger(__name__)

# Environment variables to inherit by default
DEFAULT_INHERITED_ENV_VARS = (
    [
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
    ]
    if sys.platform == "win32"
    else ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]
)

PROCESS_TERMINATION_TIMEOUT = 2.0


class CanceledFnT(Protocol):
    """Set cancled flag."""

    async def __call__(self, canceled: bool) -> None:
        """Set cancled flag."""


@asynccontextmanager
async def stdio_client(
    server: StdioServerParameters,
    errlog: LogProxy,
    is_canceled_callback: CanceledFnT | None = None,
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
        int,
    ],
    None,
]:
    """Copy of mcp.client.stdio.stdio_client."""
    if is_canceled_callback:
        await is_canceled_callback(False)
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]

    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    command = _get_executable_command(server.command)

    # Open process with stderr piped for capture
    process = await _create_platform_compatible_process(
        command=command,
        args=server.args,
        env=(
            {**get_default_environment(), **server.env}
            if server.env is not None
            else get_default_environment()
        ),
        cwd=server.cwd,
    )

    async def stderr_reader() -> None:
        try:
            assert process.stderr, "Opened process is missing stderr"
            async for line in TextReceiveStream(
                process.stderr,
                encoding=server.encoding,
                errors=server.encoding_error_handler,
            ):
                await errlog.write(line)
                await errlog.flush()
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        finally:
            logger.debug("stderr_pipe closed")

    async def stdout_reader() -> None:
        assert process.stdout, "Opened process is missing stdout"

        try:
            async with read_stream_writer:
                buffer = ""
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()
                    for line in lines:
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:  # noqa: BLE001
                            logger.error("Error validating message: %s, %s", exc, line)
                            await read_stream_writer.send(exc)
                            continue

                        session_message = SessionMessage(message)
                        await read_stream_writer.send(session_message)
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        finally:
            logger.debug("stdout_reader closed")

    async def stdin_writer() -> None:
        assert process.stdin, "Opened process is missing stdin"

        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    json = session_message.message.model_dump_json(
                        by_alias=True, exclude_none=True
                    )
                    await process.stdin.send(
                        (json + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        finally:
            logger.debug("stdin_writer closed")

    # All errors end up becoming anyio.BrockenResourceError after getting outside
    # of this 'async with' ...
    async with (
        anyio.create_task_group() as tg,
        process,
    ):
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        tg.start_soon(stderr_reader)
        try:
            yield read_stream, write_stream, process.pid
        except asyncio.CancelledError:
            logger.warning("process canceled: %s", process.pid)
            if is_canceled_callback:
                await is_canceled_callback(True)
            raise
        except Exception as exc:
            logger.error("Error, closing process %s: %s", process.pid, exc)
            raise
        finally:
            # MCP spec: stdio shutdown sequence
            # 1. Close input stream to server
            # 2. Wait for server to exit, or send SIGTERM if it doesn't exit in time
            # 3. Send SIGKILL if still not exited
            if process.stdin:
                with suppress(Exception):
                    await process.stdin.aclose()
            try:
                # Give the process time to exit gracefully after stdin closes
                with anyio.fail_after(PROCESS_TERMINATION_TIMEOUT):
                    await process.wait()
            except TimeoutError:
                # Process didn't exit from stdin closure, use platform-specific
                # termination
                # which handles SIGTERM -> SIGKILL escalation
                await _terminate_process_tree(process)
            except ProcessLookupError:
                # Process already exited, which is fine
                pass
            await read_stream.aclose()
            await write_stream.aclose()
            await read_stream_writer.aclose()
            await write_stream_reader.aclose()
    logger.error("Process %s closed", "xx")


def _get_executable_command(command: str) -> str:
    """Copy of mcp.client.stdio._get_executable_command."""
    if sys.platform == "win32":
        return get_windows_executable_command(command)
    return command


async def _create_platform_compatible_process(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> anyio.abc.Process:
    """Copy from mcp.client.stdio._create_platform_compatible_process."""
    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW
    else:
        creationflags = 0
    process = await anyio.open_process(
        [command, *args],
        creationflags=creationflags,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    logger.info("launched process: %s, pid: %s", command, process.pid)

    return process


async def _terminate_process_tree(
    process: Process | FallbackProcess, timeout_seconds: float = 2.0
) -> None:
    """Terminate a process and all its children using platform-specific methods.

    Unix: Uses os.killpg() for atomic process group termination
    Windows: Uses Job Objects via pywin32 for reliable child process cleanup

    Args:
        process: The process to terminate
        timeout_seconds: Timeout in seconds before force killing (default: 2.0)
    """
    if sys.platform == "win32":
        await terminate_windows_process_tree(process, timeout_seconds)
    else:
        # FallbackProcess should only be used for Windows compatibility
        assert isinstance(process, Process)
        await terminate_posix_process_tree(process, timeout_seconds)
