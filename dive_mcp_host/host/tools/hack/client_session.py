from copy import deepcopy
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession as McpClientSession
from mcp import types
from mcp.shared._httpx_utils import McpHttpClientFactory, create_mcp_http_client
from mcp.shared.message import MessageMetadata
from mcp.shared.session import ProgressFnT


class ClientSession(McpClientSession):
    """Hacked MCP Client Session.

    This is a hack to allow for metadata to be passed to the call_tool method.
    """

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        metadata: MessageMetadata | None = None,
    ) -> types.CallToolResult:
        """Call a tool with optional metadata for resumption."""
        return await self.send_request(
            types.ClientRequest(
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(
                        name=name,
                        arguments=arguments,
                    ),
                )
            ),
            types.CallToolResult,
            request_read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
            metadata=metadata,
        )


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
