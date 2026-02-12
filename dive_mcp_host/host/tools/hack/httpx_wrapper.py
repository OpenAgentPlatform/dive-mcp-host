"""Reason: Bug in httpx async support on windows.

Async requests become sync requests when each request uses their own AsyncClient.
This only happens on windows...

The workaround is to use the same AsyncClient when possible.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Self

import httpx


class AsyncClient:
    """Wrapper. Uses shared AsyncClients from a global pool.

    Auth, headers and timeout are stored per-instance and applied at request time.

    By default uses a shared global client. If proxy or verify=False is specified,
    a separate client is created and cached by the provided key (typically MCP name).

    The wrapper is event-loop aware: if the event loop changes (e.g., between test
    runs), cached clients bound to the old loop are discarded and new ones created.
    """

    _default_client: ClassVar[httpx.AsyncClient | None] = None
    _default_client_loop: ClassVar[asyncio.AbstractEventLoop | None] = None
    _custom_clients: ClassVar[dict[str, httpx.AsyncClient]] = {}
    _custom_clients_loops: ClassVar[dict[str, asyncio.AbstractEventLoop]] = {}

    @classmethod
    def _get_current_loop(cls) -> asyncio.AbstractEventLoop | None:
        """Get the current running event loop, or None if not running."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    @classmethod
    def _get_or_create_client(
        cls,
        key: str,
        proxy: str | None = None,
        verify: bool = True,
    ) -> httpx.AsyncClient:
        """Get or create an httpx client.

        Args:
            key: Cache key for custom clients (e.g., MCP name).
            proxy: Proxy URL for this client.
            verify: Whether to verify SSL certificates.

        Returns:
            The shared default client, or a custom client for the given key.
        """
        current_loop = cls._get_current_loop()

        # Use custom client if proxy or verify=False
        if proxy or not verify:
            # Check if existing client is bound to a different (possibly closed) loop
            if key in cls._custom_clients:
                stored_loop = cls._custom_clients_loops.get(key)
                if stored_loop is not current_loop:
                    # Discard old client (don't close - loop may be closed)
                    del cls._custom_clients[key]
                    if key in cls._custom_clients_loops:
                        del cls._custom_clients_loops[key]

            if key not in cls._custom_clients:
                kwargs: dict[str, Any] = {"follow_redirects": True, "verify": verify}
                if proxy:
                    kwargs["proxy"] = proxy
                cls._custom_clients[key] = httpx.AsyncClient(**kwargs)
                if current_loop is not None:
                    cls._custom_clients_loops[key] = current_loop
            return cls._custom_clients[key]

        # Use default client
        # Check if existing default client is bound to a different loop
        if (
            cls._default_client is not None
            and cls._default_client_loop is not current_loop
        ):
            # Discard old client (don't close - loop may be closed)
            cls._default_client = None
            cls._default_client_loop = None

        if cls._default_client is None:
            cls._default_client = httpx.AsyncClient(follow_redirects=True)
            cls._default_client_loop = current_loop
        return cls._default_client

    def __init__(
        self,
        *,
        key: str,
        auth: httpx.Auth | None = None,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | float | None = None,
        proxy: str | None = None,
        verify: bool = True,
    ) -> None:
        """Initialize the wrapper with instance-level settings.

        Args:
            auth: Authentication to apply to requests.
            headers: Headers to merge into each request.
            timeout: Timeout to apply to requests (default 30s).
            key: Cache key for custom client (e.g., MCP name). Required if
                proxy or verify=False.
            proxy: Proxy URL. If set, uses a separate cached client for this key.
            verify: Whether to verify SSL. If False, uses a separate cached client.
        """
        self._auth = auth
        self._headers = headers or {}
        self._timeout = timeout if timeout is not None else httpx.Timeout(30.0)
        self._key = key
        self._proxy = proxy
        self._verify = verify

        # Ensure client exists
        self._get_or_create_client(key=key, proxy=proxy, verify=verify)

    @property
    def _client(self) -> httpx.AsyncClient:
        return self._get_or_create_client(
            key=self._key, proxy=self._proxy, verify=self._verify
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying client."""
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
        """No-op for individual wrappers - clients are managed globally."""

    @classmethod
    async def close_all(cls) -> None:
        """Close all clients (default and custom). Call on application shutdown."""
        if cls._default_client is not None:
            await cls._default_client.aclose()
            cls._default_client = None
            cls._default_client_loop = None
        for client in cls._custom_clients.values():
            await client.aclose()
        cls._custom_clients.clear()
        cls._custom_clients_loops.clear()

    @classmethod
    async def close_client(cls, key: str) -> None:
        """Close and remove a specific custom client by key."""
        if key in cls._custom_clients:
            await cls._custom_clients[key].aclose()
            del cls._custom_clients[key]
            if key in cls._custom_clients_loops:
                del cls._custom_clients_loops[key]

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context without closing clients."""
