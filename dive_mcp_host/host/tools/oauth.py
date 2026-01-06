import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Hashable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Literal, Protocol, Self
from urllib.parse import parse_qs, urlparse

import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client.auth import OAuthFlowError, TokenStorage
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
)
from mcp.shared.message import SessionMessage
from pydantic import AnyUrl, Field, RootModel
from pydantic.dataclasses import dataclass

from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.host.tools.hack.oauth2 import OAuthClientProvider

type ReadStreamType = MemoryObjectReceiveStream[SessionMessage | Exception]
type WriteStreamType = MemoryObjectSendStream[SessionMessage]
type ClientFactory = Callable[
    ...,
    AbstractAsyncContextManager[
        tuple[ReadStreamType, WriteStreamType]
        | tuple[ReadStreamType, WriteStreamType, Callable[[], str | None]],
        bool | None,
    ],
]

logger = logging.getLogger(__name__)


@dataclass
class AuthorizationProgress:
    """A progress for the authorization task."""

    type: Literal[
        "auth_success",
        "auth_failed",
        "no_auth_required",
        "wait_code",
        "code_set",
        "canceled",
    ]
    server_name: str
    state: str | None = None
    auth_url: str | None = None
    code: str | None = None
    error: str | None = None

    def is_result(self) -> bool:
        """Check if the authorization is a result."""
        return self.type in ["auth_success", "auth_failed", "canceled"]

    def has_code(self) -> bool:
        """Check if the authorization has a code."""
        return self.code is not None

    def has_auth_url(self) -> bool:
        """Check if the authorization has an auth URL."""
        return self.auth_url is not None


class StateStore[K: Hashable, V]:
    """A state store that stores states for a given key."""

    def __init__(self) -> None:
        """Initialize the state store."""
        # update lock
        self._dict: dict[K, V] = {}
        self._cond: asyncio.Condition = asyncio.Condition()

    async def pop(self, key: K) -> V | None:
        """Pop the state for a given key."""
        async with self._cond:
            result = self._dict.pop(key, None)
            self._cond.notify_all()
            return result

    async def get(self, key: K) -> V | None:
        """Get the state for a given key."""
        async with self._cond:
            return self._dict.get(key)

    async def update(self, key: K, value: V) -> None:
        """Update the state for a given key."""
        async with self._cond:
            self._dict.update({key: value})
            self._cond.notify_all()

    async def wait_for(
        self, key: K, func: Callable[[V | None], bool], timeout: float | None = None
    ) -> V | None:
        """Wait for the state for a given key."""
        if func(self._dict.get(key)):
            return self._dict.get(key)
        is_timeout = asyncio.Event()
        tasks = set()
        results: list[V | None] = []
        if timeout is not None:

            async def time_out() -> None:
                await asyncio.sleep(timeout)
                async with self._cond:
                    is_timeout.set()
                    self._cond.notify_all()

            task = asyncio.create_task(time_out())
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        def wait() -> bool:
            value = self._dict.get(key)
            results.append(value)
            return is_timeout.is_set() or func(value)

        async with self._cond:
            await self._cond.wait_for(wait)
            if is_timeout.is_set():
                raise TimeoutError("wait_for timeout")
            return results.pop()


@dataclass
class TokenStore:
    """A token store that stores tokens for a single client."""

    tokens: OAuthToken | None = None
    client_info: OAuthClientInformationFull | None = None
    update_method: Callable[[Self], Awaitable[None]] | None = Field(
        default=None, exclude=True
    )
    oauth_metadata: OAuthMetadata | None = None
    token_expiry_time: int | None = None

    async def get_tokens(self) -> OAuthToken | None:
        """Get the tokens for the client."""
        if self.token_expiry_time is not None and self.tokens is not None:
            self.tokens.expires_in = int(self.token_expiry_time - time.time())
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Set the tokens for the client."""
        self.tokens = tokens
        if tokens.expires_in is not None:
            self.token_expiry_time = int(time.time()) + tokens.expires_in
        else:
            self.token_expiry_time = None
        if self.update_method:
            await self.update_method(self)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Get the client information for the client."""
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Set the client information for the client."""
        self.client_info = client_info
        if self.update_method:
            await self.update_method(self)

    async def set_oauth_metadata(self, metadata: OAuthMetadata) -> None:
        """Set the oauth metadata of the server."""
        self.oauth_metadata = metadata
        if self.update_method:
            await self.update_method(self)

    async def get_oauth_metadata(self) -> OAuthMetadata | None:
        """Get the oauth metadata of the server."""
        return self.oauth_metadata


class BaseTokenStore(Protocol):
    """Abstract UnionTokenStore."""

    async def get(self, name: str) -> TokenStore:
        """Get the token store for a given name."""
        ...

    async def delete(self, name: str) -> None:
        """Delete the token store for a given name."""
        ...

    async def list(self) -> list[str]:
        """List the token stores."""
        ...


class DictTokenStore(RootModel):
    """A root token store that stores token stores for multiple clients."""

    root: dict[str, TokenStore] = Field(default_factory=dict)


class LocalTokenStore(BaseTokenStore):
    """A union token store that stores tokens for multiple clients."""

    _root_store: DictTokenStore

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the union token store."""
        self._path = path or Path.cwd() / "oauth_store.json"
        self._load()

    def _load(self) -> None:
        try:
            with self._path.open() as file:
                root_store = DictTokenStore.model_validate_json(file.read())
        except FileNotFoundError:
            root_store = DictTokenStore()

        self._root_store = root_store

    def save(self) -> None:
        """Save the token store."""
        try:
            if not self._path.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.touch(mode=0o600)

            with self._path.open("w") as file:
                file.write(self._root_store.model_dump_json())
        except:
            raise

    async def _update(self, name: str, store: TokenStore) -> None:
        """Update the token store for a given name."""
        self._root_store.root[name] = store
        self.save()

    async def get(self, name: str) -> TokenStore:
        """Get the token store for a given name."""
        store = self._root_store.root.get(name, TokenStore())
        store.update_method = lambda s: self._update(name, s)
        return store

    async def delete(self, name: str) -> None:
        """Delete the token store for a given name."""
        self._root_store.root.pop(name, None)
        self.save()

    async def list(self) -> list[str]:
        """List the token stores."""
        return list(self._root_store.root.keys())


class OAuthManager(ContextProtocol):
    """A manager for OAuth providers."""

    def __init__(
        self,
        callback_url: str = "http://localhost:61990/api/tools/login/oauth/callback",
        store: BaseTokenStore | None = None,
    ) -> None:
        """Initialize the OAuth manager."""
        self._store = store or LocalTokenStore()
        self._callback_url = callback_url

        # authorization tasks
        self._authorization_tasks: set[asyncio.Task] = set()
        self._authorization_states: set[str] = set()

        # oauth results, state as key
        self._oauth_results: StateStore[str, AuthorizationProgress] = StateStore()

    @property
    def callback_url(self) -> str:
        """Get the callback URL."""
        return self._callback_url

    @property
    def store(self) -> BaseTokenStore:
        """Get the store."""
        return self._store

    @asynccontextmanager
    async def with_client(
        self,
        name: str,
        server_url: str,
        auth_callback: Callable[[AuthorizationProgress], Awaitable[None]] | None = None,
        wait_auth: bool = False,
    ) -> AsyncGenerator[OAuthClientProvider, None]:
        """With the MCP server."""

        async def callback(progress: AuthorizationProgress | None) -> None:
            if progress:
                if progress.state and not wait_auth:
                    await self.cancel_authorization(progress.state)
                if auth_callback:
                    await auth_callback(progress)

        try:
            auth = self.get_provider(
                name=name,
                server_url=server_url,
                storage=await self.store.get(name),
                auth_callback=callback,
            )
            yield auth
        finally:
            pass

    async def _authorization_task(
        self,
        name: str,
        factory: ClientFactory,
        factory_kwargs: dict,
        server_url: str,
        auth_callback: Callable[[AuthorizationProgress | None], Awaitable[None]],
    ) -> None:
        error: Exception | None = None
        state: str | None = None
        try:
            async with AsyncExitStack() as stack:

                async def callback(progress: AuthorizationProgress) -> None:
                    nonlocal state
                    if progress.state:
                        state = progress.state
                        self._authorization_states.add(state)
                    await auth_callback(progress)

                auth = await stack.enter_async_context(
                    self.with_client(
                        name,
                        wait_auth=True,
                        server_url=server_url,
                        auth_callback=callback,
                    )
                )

                factory_kwargs.pop("auth", None)
                client = factory(auth=auth, **factory_kwargs)
                streams = await stack.enter_async_context(client)
                session = await stack.enter_async_context(
                    ClientSession(
                        *[streams[0], streams[1]],
                    )
                )
                await session.initialize()
                progress = await self._oauth_results.get(state or "")
                if progress:
                    progress.type = "auth_success"
                else:
                    progress = AuthorizationProgress(
                        type="auth_success",
                        state=state,
                        server_name=name,
                    )
                await self._oauth_results.update(state or "", progress)
                await auth_callback(progress)
                return
        except* OAuthFlowError:
            logger.exception("oauth flow error")
            error = Exception("oauth flow error")
        except* (httpx.ConnectError, httpx.TimeoutException):
            logger.exception("network error")
            error = Exception("network error")
        except* Exception:
            logger.exception("unknown error")
            error = Exception("unknown error")

        progress = await self._oauth_results.get(state or "")
        if progress:
            progress.type = "auth_failed"
            progress.error = str(error) if error else None
        else:
            progress = AuthorizationProgress(
                type="auth_failed",
                state=state,
                server_name=name,
                error=str(error) if error else None,
            )
        await self._oauth_results.update(state or "", progress)
        await auth_callback(progress)

    async def authorization_task(
        self,
        name: str,
        factory: ClientFactory,
        factory_kwargs: dict,
        server_url: str,
    ) -> AuthorizationProgress:
        """Authorization task."""
        await self.store.delete(name)
        progress_queue = asyncio.Queue()

        task = asyncio.create_task(
            self._authorization_task(
                name,
                factory,
                factory_kwargs,
                server_url,
                progress_queue.put,
            )
        )
        self._authorization_tasks.add(task)
        task.add_done_callback(self._authorization_tasks.discard)
        return await progress_queue.get()

    async def cancel_authorization(self, state: str) -> None:
        """Cancel the authorization for a given name."""
        if progress := await self._oauth_results.get(state):
            progress.type = "canceled"
            await self._oauth_results.update(state, progress)

    async def wait_authorization(
        self,
        state: str,
        timeout: float | None = None,
    ) -> AuthorizationProgress:
        """Wait for the authorization to complete."""
        current = await self._oauth_results.get(state)
        if current and current.is_result():
            return current

        def wait(value: AuthorizationProgress | None) -> bool:
            return value is not None and value.is_result()

        if state not in self._authorization_states and (
            result := await self._oauth_results.get(state)
        ):
            return result

        result = await self._oauth_results.wait_for(
            state,
            wait,
            timeout,
        )
        assert result
        return result

    def get_provider(
        self,
        name: str,
        server_url: str,
        storage: TokenStorage,
        auth_callback: Callable[[AuthorizationProgress], Awaitable[None]],
    ) -> OAuthClientProvider:
        """Get the OAuth provider for a given name."""
        redirect_handler, callback_handler = self._generate_handler(name, auth_callback)

        return OAuthClientProvider(
            server_url=server_url,
            client_metadata=OAuthClientMetadata(
                client_name="Dive MCP Client",
                redirect_uris=[AnyUrl(self.callback_url)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
            ),
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

    async def set_oauth_code(self, code: str, state: str) -> str | None:
        """Set the OAuth code for a given name."""
        if result := await self._oauth_results.get(state):
            result.type = "code_set"
            result.code = code
            await self._oauth_results.update(state, result)
            return result.server_name
        return None

    async def _wait_oauth_code(
        self,
        name: str,
        auth_url: str,
        auth_callback: Callable[[AuthorizationProgress], Awaitable[None]],
    ) -> tuple[str, str | None]:
        """Wait for the OAuth code for a given name."""
        state = self.get_state(auth_url)
        assert state

        progress = AuthorizationProgress(
            type="wait_code",
            state=state,
            server_name=name,
            auth_url=auth_url,
        )
        await self._oauth_results.update(state, progress)
        await auth_callback(progress)

        def wait(value: AuthorizationProgress | None) -> bool:
            return value is not None and (value.has_code() or value.is_result())

        try:
            if result := await self._oauth_results.wait_for(state, wait, timeout=300):
                return result.code or "", state
        except TimeoutError:
            logger.warning("timeout to wait for oauth code")
            progress = await self._oauth_results.get(state)
            if not progress:
                progress = AuthorizationProgress(
                    type="auth_failed",
                    state=state,
                    server_name=name,
                )
            progress.type = "auth_failed"
            progress.error = "timeout to wait for oauth code"
            await self._oauth_results.update(state, progress)
        return "", state

    def get_state(self, url: str) -> str | None:
        """Get the state from the URL."""
        parsed_url = urlparse(url)
        query = parse_qs(parsed_url.query)
        return query.get("state", [None])[0]

    def _generate_handler(
        self,
        name: str,
        auth_callback: Callable[[AuthorizationProgress], Awaitable[None]],
    ) -> tuple[
        Callable[[str], Awaitable[None]],
        Callable[[], Awaitable[tuple[str, str | None]]],
    ]:
        """Register a handler for a given name."""
        url_queue = asyncio.Queue()

        async def callback_handler() -> tuple[str, str | None]:
            """Callback handler."""
            # empty callback
            url = await url_queue.get()
            code, state = await self._wait_oauth_code(name, url, auth_callback)
            return code, state

        return url_queue.put, callback_handler

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """Run the OAuth manager in a context."""
        try:
            yield self
        finally:
            for task in self._authorization_tasks:
                task.cancel()
            self._authorization_tasks.clear()
            self._authorization_states.clear()
