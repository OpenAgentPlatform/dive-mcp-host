import asyncio
import signal
import tempfile
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager

import httpx
import pytest
import pytest_asyncio

from dive_mcp_host.host.conf import LogConfig
from dive_mcp_host.host.tools import ServerConfig


@pytest.fixture
def sqlite_uri() -> Generator[str, None, None]:
    """Create a temporary SQLite URI."""
    with tempfile.NamedTemporaryFile(
        prefix="testServiceConfig_", suffix=".json"
    ) as service_config_file:
        yield f"sqlite:///{service_config_file.name}"


@pytest.fixture
def echo_tool_stdio_config() -> dict[str, ServerConfig]:  # noqa: D103
    return {
        "echo": ServerConfig(
            name="echo",
            command="python3",
            args=[
                "-m",
                "dive_mcp_host.host.tools.echo",
                "--transport=stdio",
            ],
            transport="stdio",
        ),
    }


@pytest.fixture
def echo_tool_local_sse_config(
    unused_tcp_port_factory: Callable[[], int],
) -> dict[str, ServerConfig]:
    """Echo Local SSE server configuration."""
    port = unused_tcp_port_factory()
    return {
        "echo": ServerConfig(
            name="echo",
            command="python3",
            args=[
                "-m",
                "dive_mcp_host.host.tools.echo",
                "--transport=sse",
                "--host=localhost",
                f"--port={port}",
            ],
            transport="sse",
            url=f"http://localhost:{port}/sse",
        ),
    }


@pytest_asyncio.fixture
@asynccontextmanager
async def echo_tool_sse_server(
    unused_tcp_port_factory: Callable[[], int],
) -> AsyncGenerator[tuple[int, dict[str, ServerConfig]], None]:
    """Start the echo tool SSE server."""
    port = unused_tcp_port_factory()
    proc = await asyncio.create_subprocess_exec(
        "python3",
        "-m",
        "dive_mcp_host.host.tools.echo",
        "--transport=sse",
        "--host=localhost",
        f"--port={port}",
    )
    while True:
        try:
            _ = await httpx.AsyncClient().get(f"http://localhost:{port}/xxxx")
            break
        except httpx.HTTPStatusError:
            break
        except:  # noqa: E722
            await asyncio.sleep(0.1)
    try:
        yield (
            port,
            {
                "echo": ServerConfig(
                    name="echo", url=f"http://localhost:{port}/sse", transport="sse"
                )
            },
        )
    finally:
        proc.send_signal(signal.SIGKILL)
        await proc.wait()


@pytest_asyncio.fixture
@asynccontextmanager
async def echo_tool_streamable_server(
    unused_tcp_port_factory: Callable[[], int],
) -> AsyncGenerator[tuple[int, dict[str, ServerConfig]], None]:
    """Start the echo tool SSE server."""
    port = unused_tcp_port_factory()
    proc = await asyncio.create_subprocess_exec(
        "python3",
        "-m",
        "dive_mcp_host.host.tools.echo",
        "--transport=streamable",
        "--host=localhost",
        f"--port={port}",
    )
    while True:
        try:
            _ = await httpx.AsyncClient().get(f"http://localhost:{port}/xxxx")
            break
        except httpx.HTTPStatusError:
            break
        except:  # noqa: E722
            await asyncio.sleep(0.1)
    try:
        yield (
            port,
            {
                "echo": ServerConfig(
                    name="echo",
                    url=f"http://localhost:{port}/mcp",
                    transport="streamable",
                )
            },
        )
    finally:
        proc.send_signal(signal.SIGKILL)
        await proc.wait()


@pytest.fixture
def log_config() -> LogConfig:
    """Fixture for log Config."""
    return LogConfig()


@pytest_asyncio.fixture
async def pproxy_server(
    unused_tcp_port_factory: Callable[[], int],
) -> AsyncGenerator[str, None]:
    """Fixture for proxy."""
    port = unused_tcp_port_factory()
    proc = await asyncio.create_subprocess_exec(
        "python3",
        "-m",
        "pproxy",
        "-l",
        f"http+socks4+socks5://:{port}",
    )
    try:
        for _ in range(20):
            try:
                _ = await httpx.AsyncClient().get(f"http://localhost:{port}/xxxx")
                break
            except httpx.RemoteProtocolError:
                break
            except:  # noqa: E722
                await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Failed to start pproxy server")
        yield f"localhost:{port}"
    finally:
        proc.send_signal(signal.SIGKILL)
        await proc.wait()
