"""Tests for the httpx AsyncClient wrapper."""

import asyncio
import base64
import json

import httpx
import pytest

from dive_mcp_host.host.tools.hack.httpx_wrapper import AsyncClient

# =============================================================================
# Default Client Reuse Tests
# =============================================================================


@pytest.mark.asyncio
async def test_default_client_reuse(echo_http_server: str) -> None:
    """Test that multiple wrappers without proxy/verify=False share the same client."""
    wrapper1 = AsyncClient(key="test1")
    wrapper2 = AsyncClient(key="test2")
    wrapper3 = AsyncClient(key="test3")

    # All should use the same underlying default client
    assert wrapper1._client is wrapper2._client
    assert wrapper2._client is wrapper3._client
    assert AsyncClient._default_client is not None

    # Verify requests work
    response = await wrapper1.get(f"{echo_http_server}/echo")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_default_client_not_shared_with_custom(echo_http_server: str) -> None:
    """Test that default client is not shared when proxy or verify=False is used."""
    default_wrapper = AsyncClient(key="default")
    proxy_wrapper = AsyncClient(key="proxy", proxy="http://127.0.0.1:8888")
    verify_wrapper = AsyncClient(key="verify", verify=False)

    # Default should use _default_client
    assert default_wrapper._client is AsyncClient._default_client

    # Proxy and verify should use _custom_clients
    assert proxy_wrapper._client is not AsyncClient._default_client
    assert verify_wrapper._client is not AsyncClient._default_client
    assert "proxy" in AsyncClient._custom_clients
    assert "verify" in AsyncClient._custom_clients


# =============================================================================
# Custom Client Reuse Tests
# =============================================================================


@pytest.mark.asyncio
async def test_custom_client_reuse_same_key() -> None:
    """Test that custom clients are reused for the same key."""
    wrapper1 = AsyncClient(key="myproxy", proxy="http://127.0.0.1:8888")
    wrapper2 = AsyncClient(key="myproxy", proxy="http://127.0.0.1:8888")

    # Same key should share client
    assert wrapper1._client is wrapper2._client
    assert AsyncClient._custom_clients["myproxy"] is wrapper1._client


@pytest.mark.asyncio
async def test_custom_client_different_keys() -> None:
    """Test that different keys get different custom clients."""
    wrapper1 = AsyncClient(key="proxy1", proxy="http://127.0.0.1:8888")
    wrapper2 = AsyncClient(key="proxy2", proxy="http://127.0.0.1:8889")

    # Different keys should have different clients
    assert wrapper1._client is not wrapper2._client
    assert "proxy1" in AsyncClient._custom_clients
    assert "proxy2" in AsyncClient._custom_clients


@pytest.mark.asyncio
async def test_verify_false_uses_custom_client() -> None:
    """Test that verify=False triggers custom client creation."""
    wrapper = AsyncClient(key="noverify", verify=False)

    assert wrapper._client is not AsyncClient._default_client
    assert "noverify" in AsyncClient._custom_clients
    # Verify the wrapper tracks verify=False
    assert wrapper._verify is False


# =============================================================================
# Event Loop Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_event_loop_tracking() -> None:
    """Test that event loops are tracked for default and custom clients."""
    # Create clients to trigger loop tracking
    _ = AsyncClient(key="test")
    _ = AsyncClient(key="custom", proxy="http://127.0.0.1:8888")

    current_loop = asyncio.get_running_loop()

    # Default client loop tracking
    assert AsyncClient._default_client_loop is current_loop

    # Custom client loop tracking
    assert "custom" in AsyncClient._custom_clients_loops
    assert AsyncClient._custom_clients_loops["custom"] is current_loop


# =============================================================================
# HTTP Methods with Headers/Auth Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_method_with_headers_auth(echo_http_server: str) -> None:
    """Test GET method with headers and auth."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Custom-Header": "test-value"},
        auth=httpx.BasicAuth("user", "pass"),
    )

    response = await wrapper.get(f"{echo_http_server}/echo")
    data = response.json()

    assert data["method"] == "GET"
    assert data["headers"]["x-custom-header"] == "test-value"
    assert "authorization" in data["headers"]
    assert data["headers"]["authorization"].startswith("Basic ")


@pytest.mark.asyncio
async def test_post_method_with_headers_auth(echo_http_server: str) -> None:
    """Test POST method with headers, auth, and body."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Custom-Header": "post-test"},
        auth=httpx.BasicAuth("user", "pass"),
    )

    response = await wrapper.post(
        f"{echo_http_server}/echo",
        json={"key": "value"},
    )
    data = response.json()

    assert data["method"] == "POST"
    assert data["headers"]["x-custom-header"] == "post-test"
    assert data["body"] == {"key": "value"}
    assert "authorization" in data["headers"]


@pytest.mark.asyncio
async def test_all_http_methods(echo_http_server: str) -> None:
    """Test all HTTP methods: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Test": "all-methods"},
    )

    methods = [
        ("get", wrapper.get),
        ("post", wrapper.post),
        ("put", wrapper.put),
        ("patch", wrapper.patch),
        ("delete", wrapper.delete),
        ("head", wrapper.head),
        ("options", wrapper.options),
    ]

    for method_name, method_func in methods:
        response = await method_func(f"{echo_http_server}/echo")
        if method_name != "head":  # HEAD has no body
            data = response.json()
            assert data["method"] == method_name.upper()
            assert data["headers"]["x-test"] == "all-methods"
        else:
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_request_method(echo_http_server: str) -> None:
    """Test the generic request() method."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Request-Test": "value"},
    )

    response = await wrapper.request(
        "POST", f"{echo_http_server}/echo", json={"test": 1}
    )
    data = response.json()

    assert data["method"] == "POST"
    assert data["headers"]["x-request-test"] == "value"
    assert data["body"] == {"test": 1}


# =============================================================================
# Stream Method Tests
# =============================================================================


@pytest.mark.asyncio
async def test_stream_method_with_headers_auth(echo_http_server: str) -> None:
    """Test stream() method with headers and auth."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Stream-Header": "stream-value"},
        auth=httpx.BasicAuth("user", "pass"),
    )

    async with wrapper.stream("GET", f"{echo_http_server}/stream") as response:
        assert response.status_code == 200
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)

        data = json.loads(b"".join(chunks).decode())
        assert data["headers"]["x-stream-header"] == "stream-value"
        assert "authorization" in data["headers"]


@pytest.mark.asyncio
async def test_stream_post_method(echo_http_server: str) -> None:
    """Test stream() with POST method."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Stream-Post": "post-value"},
    )

    async with wrapper.stream("POST", f"{echo_http_server}/stream") as response:
        assert response.status_code == 200
        content = await response.aread()
        data = json.loads(content.decode())
        assert data["method"] == "POST"
        assert data["headers"]["x-stream-post"] == "post-value"


# =============================================================================
# Settings Merge Precedence Tests
# =============================================================================


@pytest.mark.asyncio
async def test_request_kwargs_override_instance_headers(echo_http_server: str) -> None:
    """Test that request kwargs override instance headers."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Instance": "instance-value", "X-Override": "old"},
    )

    response = await wrapper.get(
        f"{echo_http_server}/echo",
        headers={"X-Override": "new", "X-Request": "request-value"},
    )
    data = response.json()

    # Instance header preserved
    assert data["headers"]["x-instance"] == "instance-value"
    # Request header takes precedence
    assert data["headers"]["x-override"] == "new"
    # Request-only header present
    assert data["headers"]["x-request"] == "request-value"


@pytest.mark.asyncio
async def test_request_kwargs_override_auth(echo_http_server: str) -> None:
    """Test that request auth overrides instance auth."""
    wrapper = AsyncClient(
        key="test",
        auth=httpx.BasicAuth("instance-user", "instance-pass"),
    )

    # Override with different auth
    response = await wrapper.get(
        f"{echo_http_server}/echo",
        auth=httpx.BasicAuth("request-user", "request-pass"),
    )
    data = response.json()

    # Should use request auth, not instance auth
    expected_auth = base64.b64encode(b"request-user:request-pass").decode()
    assert f"Basic {expected_auth}" == data["headers"]["authorization"]


@pytest.mark.asyncio
async def test_request_kwargs_override_timeout(echo_http_server: str) -> None:
    """Test that request timeout overrides instance timeout."""
    wrapper = AsyncClient(
        key="test",
        timeout=httpx.Timeout(10.0),
    )

    # Per-request timeout should work without errors
    response = await wrapper.get(
        f"{echo_http_server}/echo",
        timeout=httpx.Timeout(5.0),
    )
    assert response.status_code == 200


# =============================================================================
# Build Request and Send Tests
# =============================================================================


@pytest.mark.asyncio
async def test_build_request_merges_settings(echo_http_server: str) -> None:
    """Test that build_request() merges instance settings."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Build": "build-value"},
    )

    request = wrapper.build_request("GET", f"{echo_http_server}/echo")

    assert "x-build" in request.headers
    assert request.headers["x-build"] == "build-value"


@pytest.mark.asyncio
async def test_send_request(echo_http_server: str) -> None:
    """Test send() with pre-built request."""
    wrapper = AsyncClient(
        key="test",
        headers={"X-Send": "send-value"},
    )

    request = wrapper.build_request("GET", f"{echo_http_server}/echo")
    response = await wrapper.send(request)
    data = response.json()

    assert data["headers"]["x-send"] == "send-value"


# =============================================================================
# Context Manager and Cleanup Tests
# =============================================================================


@pytest.mark.asyncio
async def test_context_manager() -> None:
    """Test async context manager protocol."""
    async with AsyncClient(key="test") as wrapper:
        assert wrapper is not None
        assert isinstance(wrapper, AsyncClient)
    # aclose is no-op for individual wrappers, so no assertion needed


@pytest.mark.asyncio
async def test_close_all() -> None:
    """Test close_all() class method."""
    AsyncClient(key="default")
    AsyncClient(key="custom", proxy="http://127.0.0.1:8888")

    assert AsyncClient._default_client is not None
    assert "custom" in AsyncClient._custom_clients

    await AsyncClient.close_all()

    assert AsyncClient._default_client is None
    assert len(AsyncClient._custom_clients) == 0
    assert len(AsyncClient._custom_clients_loops) == 0


@pytest.mark.asyncio
async def test_close_client_specific() -> None:
    """Test close_client() for specific key."""
    AsyncClient(key="keep", proxy="http://127.0.0.1:8888")
    AsyncClient(key="remove", proxy="http://127.0.0.1:8889")

    assert "keep" in AsyncClient._custom_clients
    assert "remove" in AsyncClient._custom_clients

    await AsyncClient.close_client("remove")

    assert "keep" in AsyncClient._custom_clients
    assert "remove" not in AsyncClient._custom_clients
    assert "remove" not in AsyncClient._custom_clients_loops


# =============================================================================
# Proxy Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_proxy_client_works(
    echo_http_server: str,
    pproxy_server: str,
) -> None:
    """Test that requests through proxy work correctly."""
    wrapper = AsyncClient(
        key="proxy-test",
        proxy=f"http://{pproxy_server}",
        headers={"X-Proxy-Test": "proxy-value"},
    )

    response = await wrapper.get(f"{echo_http_server}/echo")
    data = response.json()

    assert response.status_code == 200
    assert data["headers"]["x-proxy-test"] == "proxy-value"
    # Verify custom client was used
    assert "proxy-test" in AsyncClient._custom_clients
