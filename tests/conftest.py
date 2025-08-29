import asyncio
import signal
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlparse

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


@pytest_asyncio.fixture
@asynccontextmanager
async def echo_with_slash_tool_streamable_server(
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
                "echo/aaa/bbb/ccc": ServerConfig(
                    name="echo/aaa/bbb/ccc",
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


@pytest_asyncio.fixture
async def weather_tool_streamable_server(
    unused_tcp_port_factory: Callable[[], int],
) -> AsyncGenerator[
    tuple[int, dict[str, ServerConfig], Callable[[str], Awaitable[tuple[str, str]]]],
    None,
]:
    """Start the weather tool streamable server."""
    port = unused_tcp_port_factory()
    proc = await asyncio.create_subprocess_exec(
        "python3",
        "-m",
        "tests.mcp_servers.weather",
        "--host=localhost",
        f"--port={port}",
    )
    while True:
        try:
            _ = await httpx.AsyncClient().get(f"http://localhost:{port}/")
            break
        except httpx.HTTPStatusError:
            break
        except:  # noqa: E722
            await asyncio.sleep(0.1)

    async def get_auth_code(auth_url: str) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            response = await client.get(auth_url, follow_redirects=True)
            next_url = response.headers["x-redirect-link"]
            parsed = urlparse(next_url)
            query = parse_qs(parsed.query)
            return query.get("code")[0], query.get("state")[0]  # type: ignore

    try:
        yield (
            port,
            {
                "weather": ServerConfig(
                    name="weather",
                    url=f"http://localhost:{port}/weather/mcp",
                    transport="streamable",
                )
            },
            get_auth_code,
        )
    finally:
        proc.send_signal(signal.SIGKILL)
        await proc.wait()
