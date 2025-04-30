"""Copy of mcp.client.stdio.stdio_client."""

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import anyio.abc
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from mcp import types
from mcp.client.stdio import StdioServerParameters, get_default_environment
from mcp.client.stdio.win32 import (
    create_windows_process,
    get_windows_executable_command,
    terminate_windows_process,
)

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


@asynccontextmanager
async def stdio_client(  # noqa: C901, PLR0915
    server: StdioServerParameters,
    errlog: LogProxy,
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[types.JSONRPCMessage | Exception],
        MemoryObjectSendStream[types.JSONRPCMessage],
    ],
    None,
]:
    """Copy of mcp.client.stdio.stdio_client."""
    read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

    write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
    write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

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
        assert process.stderr, "Opened process is missing stderr"
        try:
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

                        await read_stream_writer.send(message)
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        finally:
            logger.debug("stdout_reader closed")

    async def stdin_writer() -> None:
        assert process.stdin, "Opened process is missing stdin"

        try:
            async with write_stream_reader:
                async for message in write_stream_reader:
                    json = message.model_dump_json(by_alias=True, exclude_none=True)
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

    async with (
        anyio.create_task_group() as tg,
        process,
    ):
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        tg.start_soon(stderr_reader)
        try:
            yield read_stream, write_stream
        except Exception as exc:  # noqa: BLE001
            logger.error("Error closing process %s: %s", process.pid, exc)
        finally:
            # Clean up process to prevent any dangling orphaned processes
            logger.info("Terminated process %s", process.pid)
            # Some process never terminates, so we need to kill it.
            await terminate_windows_process(process)
            status = await process.wait()
            logger.info("Process %s exited with status %s", process.pid, status)
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
    if sys.platform == "win32":
        process = await create_windows_process(command, args, env, cwd)
    else:
        process = await anyio.open_process([command, *args], env=env, cwd=cwd)
    logger.info("launched process: %s, pid: %s", command, process.pid)

    return process
