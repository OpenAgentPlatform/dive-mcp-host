from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx


class AsyncClient:
    """Wrapper. Uses global AsyncClient, created on first AsyncClient.

    Auth, header and timeout will be stored on creation and
    set on runtime when calling any method.
    """

    _global_client: httpx.AsyncClient | None = None

    def __init__(
        self,
        *,
        auth: httpx.Auth | None = None,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> None:
        """Initialize the wrapper with instance-level settings.

        Args:
            auth: Authentication to apply to requests.
            headers: Headers to merge into each request.
            timeout: Timeout to apply to requests.
            **kwargs: Additional kwargs passed to the global client on creation.
        """
        self._auth = auth
        self._headers = headers or {}
        self._timeout = timeout

        # Create global client on first instantiation
        if AsyncClient._global_client is None:
            AsyncClient._global_client = httpx.AsyncClient()

    @property
    def _client(self) -> httpx.AsyncClient:
        if AsyncClient._global_client is None:
            AsyncClient._global_client = httpx.AsyncClient()
        return AsyncClient._global_client

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the global client."""
        return getattr(self._client, name)

    def _merge_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge instance settings with request kwargs."""
        if self._auth is not None and "auth" not in kwargs:
            kwargs["auth"] = self._auth
        if self._timeout is not None and "timeout" not in kwargs:
            kwargs["timeout"] = self._timeout
        if self._headers:
            existing = kwargs.get("headers", {})
            kwargs["headers"] = {**self._headers, **existing}
        return kwargs

    def build_request(
        self,
        method: str,
        url: httpx.URL | str,
        **kwargs: Any,
    ) -> httpx.Request:
        """Build an HTTP request with merged settings."""
        return self._client.build_request(method, url, **self._merge_kwargs(kwargs))

    async def send(
        self,
        request: httpx.Request,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send a pre-built request."""
        return await self._client.send(request, **kwargs)

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: httpx.URL | str,
        **kwargs: Any,
    ) -> AsyncIterator[httpx.Response]:
        """Send a streaming HTTP request."""
        async with self._client.stream(
            method, url, **self._merge_kwargs(kwargs)
        ) as response:
            yield response

    async def request(
        self, method: str, url: httpx.URL | str, **kwargs: Any
    ) -> httpx.Response:
        """Send an HTTP request."""
        return await self._client.request(method, url, **self._merge_kwargs(kwargs))

    async def get(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        """Send a GET request."""
        return await self._client.get(url, **self._merge_kwargs(kwargs))

    async def post(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        """Send a POST request."""
        return await self._client.post(url, **self._merge_kwargs(kwargs))

    async def put(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        """Send a PUT request."""
        return await self._client.put(url, **self._merge_kwargs(kwargs))

    async def patch(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        """Send a PATCH request."""
        return await self._client.patch(url, **self._merge_kwargs(kwargs))

    async def delete(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        """Send a DELETE request."""
        return await self._client.delete(url, **self._merge_kwargs(kwargs))

    async def head(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        """Send a HEAD request."""
        return await self._client.head(url, **self._merge_kwargs(kwargs))

    async def options(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        """Send an OPTIONS request."""
        return await self._client.options(url, **self._merge_kwargs(kwargs))

    async def aclose(self) -> None:
        """No-op for individual wrappers - global client stays open."""

    @classmethod
    async def close_global(cls) -> None:
        """Close the global client. Call this on application shutdown."""
        if cls._global_client is not None:
            await cls._global_client.aclose()
            cls._global_client = None

    async def __aenter__(self) -> "AsyncClient":
        """Enter async context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context without closing global client."""
