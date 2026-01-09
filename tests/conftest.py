import asyncio
import signal
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import pytest_asyncio

from dive_mcp_host.host.conf import LogConfig
from dive_mcp_host.host.tools import ServerConfig
from dive_mcp_host.host.tools.hack.httpx_wrapper import AsyncClient as WrappedAsyncClient


@pytest_asyncio.fixture(autouse=True)
async def cleanup_httpx_wrapper() -> AsyncGenerator[None, None]:
    """Clean up shared httpx clients after each test.

    The AsyncClient wrapper uses class-level client caching. When tests run
    sequentially with function-scoped event loops, cached clients become
    bound to closed event loops. This fixture ensures cleanup after each test.

    The wrapper is also event-loop aware and will automatically discard clients
    bound to different loops, so this cleanup is mostly for graceful shutdown.
    """
    yield
    try:
        await WrappedAsyncClient.close_all()
    except RuntimeError:
        # Event loop might be closing, just clear references
        # The wrapper's event-loop awareness will handle this case
        WrappedAsyncClient._default_client = None
        WrappedAsyncClient._default_client_loop = None
        WrappedAsyncClient._custom_clients.clear()
        WrappedAsyncClient._custom_clients_loops.clear()


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


@pytest.fixture
def ssl_pems() -> Generator[tuple[str, str], None, None]:
    """For SSL Cert."""
    key = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDKQgI8p31dhkZq
Xb14yIgNrZwbR41ZgAwnBssTvbPVKB7q+BdIZL/oUDElVgeCgPb/OVNylwZW2/mk
ASHKMoKbO4bQAcqHiMAcqnT4sjQn8Is1z9PsBPTe73AMBBoQWOVWvwgUNOzhfl9E
DZd97+nbIRPmPiiVEj1NgXN1OtHm6IcZiGViEGrB/XikYaLwZoHwvJDoot74rYwp
JCKsK+sXk6RjNJIJ9sgd62OudWBMpqtQ7eK9ObR7g5cA6YruVOsOMUO9X87dXRdh
vKncg49TgkZBLe+XS6/TFjPSlIQTkLzb77CP75eeR8cGEA0SFqpoS077/ORvHsyw
k7xzsHtTAgMBAAECggEAM+YBLfAqXoO0cNmx4xrNsnjQXD2dexA8zgzTsjKUwQbk
/BUB4G0dwy5Elhh/n9CY0H+0tmwjT8ltQtSZh4kzc0KwgpJbFjEUXqaskh4j/PWf
uwSjk+XnsYbOAP9ScsxkuMAXAhxchjYHmu5D56hWoQhZU1Sy1CGUgl1Ls5KOxsmC
Db/1tx0/0K6AkhnG85tXpcMwpIG7L5TdYL+kogoidAyhUpa4lMd718QLj9Mvxp7g
+3RG7BgPy8WvYFTRJ7IDYwTNEPth7imRFnIGsiKaeXivbaWoW4dhBLkZu5K3a7xf
6YUzHkuG4z7jCqDr3/OVy+3Cm3UUCrDpojysLZUfeQKBgQDktU/1qu7EHDLbVYhl
gCuIxYeVC6vgEI1/T3AeNGEvRYb82CAsdO9V/yGXfjPA7eJu2JRmoaGVsJj553aU
cAFnqssWooBTG6DAO6m+BQVfu7ltXcY8BhoCcSQRo14Ka93GNs6X4gDNxkQ9MbwE
a+9sFwbMxb0N5S9ktgTB/ZuPSwKBgQDiZK1MIVA8H+31cszkqplaWgsfI8jpjyM1
qXP/kieC4CMeQAGOOln39VZc9mt6H0DUFmpFK1+NgWd4vRpADS9s7wPsuscVAeqh
VLdEprDJNWE1pzBKA8CRj410uL8/ZZaMFaon5xQHcp2rGW8tNZgO8Vxmm+oQMT7v
PeDRvEdXGQKBgHeVYhCm5p89y02pxF6mDJ1AAytoZd8LaKDHjW0NoD8SUImGDBSS
s824T/0hLLhU6c9hiP68xns5UlNgtjzY77Mft37HSuepi3zX9WH4yB2NT2Ai5xLd
cd4TAHQEgphaXCRW5eC+eAMgBvAkQBXjxdUzmQea4MYSLpyvG3+/NgZPAoGBALAL
fb7fgNbgNVhSJwYBqzJZJqDWYNtN1SFbXbp02oio5mYVe6Avu9pXSwsLC9RxDSZf
L9ANxEXzRJTc59a66hAZZMHnE/w/+0Xs7H5T6NDt02O3WJOVi18lc+g56W3Q/1p7
Vk+lHywTcoukQZG0RUQJZ8Lapw5kXP4I7hPqBsexAoGBAIQrTUADeHguSvA9xzUc
twQ6ggYeBoWKWoOD/WiSvzl7s3QJGqKRz4UVKgs7+TxWQ0nYlOR8iMfOj8gkAx0Y
BeQg/E76NzU2TYx90+WUl1DibADHP9F8qSV2JxseLsmJWZdDiP8ckfZLFlqqbfnK
I30QXK2J1mx5Tl1/22Xb7arc
-----END PRIVATE KEY-----
"""
    cert = """-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIUVnoxPdIB4MtdUCiB+dVGPt7ElgIwDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI1MTIyNjA1MzIzOVoXDTI2MTIy
NjA1MzIzOVowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAykICPKd9XYZGal29eMiIDa2cG0eNWYAMJwbLE72z1Sge
6vgXSGS/6FAxJVYHgoD2/zlTcpcGVtv5pAEhyjKCmzuG0AHKh4jAHKp0+LI0J/CL
Nc/T7AT03u9wDAQaEFjlVr8IFDTs4X5fRA2Xfe/p2yET5j4olRI9TYFzdTrR5uiH
GYhlYhBqwf14pGGi8GaB8LyQ6KLe+K2MKSQirCvrF5OkYzSSCfbIHetjrnVgTKar
UO3ivTm0e4OXAOmK7lTrDjFDvV/O3V0XYbyp3IOPU4JGQS3vl0uv0xYz0pSEE5C8
2++wj++XnkfHBhANEhaqaEtO+/zkbx7MsJO8c7B7UwIDAQABo1MwUTAdBgNVHQ4E
FgQU0fn3bq0vZ6wtq6jrkb7rOaPQJmYwHwYDVR0jBBgwFoAU0fn3bq0vZ6wtq6jr
kb7rOaPQJmYwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAU7w+
/8GEOwA3sUed05zhPKDZ9q9KRq9+l4x3czLJTTkHxQlujquH38eqPZb9V8bv5bkY
7kKTN9e0YW0jddVC9VgSVWL5fdLG/t2EIRxb1VYNIEa20BNg/yWIpxpCW6OxwLKI
2Or1f/07gB/z8BJ2fVhNMKRvr0o2JK2qNQDAQ3HbDXLDIgT5kY5ye5ikFGfkhXJk
xB5XZknLt5JIA1X/ksTUiwdHpEP3UCmwoIEyg8ZSy6CugQwxUOkpUbqUApTMWOYH
PZJFLqPVBjrRflSPR+nlQ5jYgkgVVqMZAig7/2JOPQkCu0f7703DAAYZ1weHD9LI
AHkwGUvbXIK2kKl7BA==
-----END CERTIFICATE-----
"""
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=True) as key_file,
        tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=True) as cert_file,
    ):
        key_file.write(key)
        cert_file.write(cert)
        key_file.flush()
        cert_file.flush()

        yield key_file.name, cert_file.name


@pytest_asyncio.fixture
async def echo_https_server(
    unused_tcp_port_factory: Callable[[], int],
    ssl_pems: tuple[str, str],
) -> AsyncGenerator[tuple[int, dict[str, ServerConfig]], None]:
    """An Echo Https Server."""
    port = unused_tcp_port_factory()
    proc = await asyncio.create_subprocess_exec(
        "uvicorn",
        "--port",
        f"{port}",
        "--ssl-keyfile",
        ssl_pems[0],
        "--ssl-certfile",
        ssl_pems[1],
        "--factory",
        "dive_mcp_host.host.tools.echo:create_app",
    )
    while True:
        try:
            _ = await httpx.AsyncClient(verify=False).get(  # noqa: S501
                f"https://localhost:{port}/xxxx"
            )
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
                    url=f"https://localhost:{port}/mcp",
                    transport="streamable",
                )
            },
        )
    finally:
        proc.send_signal(signal.SIGKILL)
        await proc.wait()
