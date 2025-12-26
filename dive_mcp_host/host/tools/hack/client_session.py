from copy import deepcopy
from typing import Any

import httpx
from mcp.shared._httpx_utils import McpHttpClientFactory, create_mcp_http_client


def create_mcp_http_client_factory(
    proxy: str | None = None, kwargs: dict[str, Any] | None = None
) -> McpHttpClientFactory:
    """Create the MCP HTTP client factory with custom settings.

    Args:
        proxy: Proxy URL to use for HTTP requests.
        kwargs: Additional configuration options to pass to httpx.AsyncClient.
    """
    _kwargs = deepcopy(kwargs) if kwargs else {}
    _kwargs["follow_redirects"] = True
    if proxy:
        _kwargs["proxy"] = proxy

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """Create a standardized httpx AsyncClient with MCP defaults."""
        kwargs: dict[str, Any] = _kwargs.copy()

        # Handle timeout
        if timeout is None:
            kwargs["timeout"] = httpx.Timeout(30.0)
        else:
            kwargs["timeout"] = timeout

        # Handle headers
        if headers is not None:
            kwargs["headers"] = headers

        # Handle authentication
        if auth is not None:
            kwargs["auth"] = auth

        return httpx.AsyncClient(**kwargs)

    if _kwargs:
        return factory
    return create_mcp_http_client
