import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Literal, Self
from urllib.parse import parse_qs, urlparse

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from mcp.shared.message import SessionMessage
from pydantic import AnyUrl, Field, RootModel
from pydantic.dataclasses import dataclass

from dive_mcp_host.host.helpers.context import ContextProtocol

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

    type: Literal["auth_success", "auth_failed", "code_set"]
    error: str | None = None


@dataclass
class TokenStore:
    """A token store that stores tokens for a single client."""

    tokens: OAuthToken | None = None
    client_info: OAuthClientInformationFull | None = None
    update_method: Callable[[Self], None] = Field(default=lambda _: None, exclude=True)

    async def get_tokens(self) -> OAuthToken | None:
        """Get the tokens for the client."""
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Set the tokens for the client."""
        self.tokens = tokens
        self.update_method(self)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Get the client information for the client."""
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Set the client information for the client."""
        self.client_info = client_info
        self.update_method(self)


class RootTokenStore(RootModel):
    """A root token store that stores token stores for multiple clients."""

    root: dict[str, TokenStore] = {}


class UnionTokenStore:
    """A union token store that stores tokens for multiple clients."""

    _root_store: RootTokenStore

    def __init__(self, path: Path) -> None:
        """Initialize the union token store."""
        self._path = path
        self._load()

    def _load(self) -> None:
        try:
            with self._path.open() as file:
                root_store = RootTokenStore.model_validate_json(file.read())
        except FileNotFoundError:
            root_store = RootTokenStore()

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

    def _update(self, name: str, store: TokenStore) -> None:
        """Update the token store for a given name."""
        self._root_store.root[name] = store
        self.save()

    def get(self, name: str) -> TokenStore:
        """Get the token store for a given name."""
        store = self._root_store.root.get(name, TokenStore())
        store.update_method = lambda s: self._update(name, s)
        return store

    def delete(self, name: str) -> None:
        """Delete the token store for a given name."""
        self._root_store.root.pop(name, None)
        self.save()


class OAuthManager(ContextProtocol):
    """A manager for OAuth providers."""

    def __init__(self, path: Path, callback_url: str) -> None:
        """Initialize the OAuth manager."""
        self._store = UnionTokenStore(path)
        self._callback_url = callback_url
        self._terminated = asyncio.Event()

        # temporary storage for oauth code
        # store tuple[code, state], state as key
        self._oauth_code: dict[str, tuple[str, str]] = {}
        self._oauth_code_cond = asyncio.Condition()
        # store name, state as key
        self._oauth_code_wait: dict[str, str] = {}

        # authorization tasks
        self._authorization_tasks: set[asyncio.Task] = set()
        self._authorization_states: set[str] = set()

        # oauth results, state as key
        self._oauth_results: dict[str, AuthorizationProgress] = {}
        self._oauth_results_cond = asyncio.Condition()

    @property
    def callback_url(self) -> str:
        """Get the callback URL."""
        return self._callback_url

    @property
    def store(self) -> UnionTokenStore:
        """Get the store."""
        return self._store

    @asynccontextmanager
    async def with_client(
        self,
        name: str,
        server_url: str,
        auth_callback: Callable[[str], Awaitable[None]] | None = None,
        wait_auth: bool = False,
    ) -> AsyncGenerator[OAuthClientProvider, None]:
        """With the MCP server."""
        stopped = asyncio.Event()

        try:
            auth = self.get_provider(
                name=name,
                server_url=server_url,
                auth_callback=auth_callback,
                stopped=stopped if wait_auth else None,
            )
            yield auth
        finally:
            # stop waiting for code
            async with self._oauth_code_cond:
                stopped.set()
                self._oauth_code_cond.notify_all()

    async def _authorization_task(
        self,
        name: str,
        factory: ClientFactory,
        factory_kwargs: dict,
        server_url: str,
        auth_callback: Callable[[str], Awaitable[None]],
    ) -> None:
        error: Exception | None = None
        try:
            url_queue = asyncio.Queue()
            async with AsyncExitStack() as stack:

                async def callback(auth_url: str) -> None:
                    state = self.get_state(auth_url)
                    assert state
                    self._authorization_states.add(state)
                    await asyncio.gather(
                        url_queue.put(auth_url),
                        auth_callback(auth_url),
                    )

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
                url = await url_queue.get()
                assert url
                state = self.get_state(url)
                assert state
                async with self._oauth_results_cond:
                    self._oauth_results[state] = AuthorizationProgress(
                        type="auth_success",
                    )
                    self._oauth_results_cond.notify_all()
                return
        except Exception as e:
            logger.exception("authorization task error")
            error = e

        async with self._oauth_results_cond:
            self._oauth_results[name] = AuthorizationProgress(
                type="auth_failed",
                error=str(error) if error else None,
            )
            self._oauth_results_cond.notify_all()

    async def authorization_task(
        self,
        name: str,
        factory: ClientFactory,
        factory_kwargs: dict,
        server_url: str,
    ) -> str:
        """Authorization task."""
        self.store.delete(name)
        url_queue = asyncio.Queue()

        task = asyncio.create_task(
            self._authorization_task(
                name,
                factory,
                factory_kwargs,
                server_url,
                url_queue.put,
            )
        )
        self._authorization_tasks.add(task)
        task.add_done_callback(self._authorization_tasks.discard)
        return await url_queue.get()

    async def wait_authorization(self, state: str) -> AuthorizationProgress:
        """Wait for the authorization to complete."""
        if state not in self._authorization_states:
            return AuthorizationProgress(
                type="code_set",
            )

        async with self._oauth_results_cond:
            await self._oauth_results_cond.wait_for(
                lambda: self._oauth_results.get(state) is not None
            )
            return self._oauth_results[state]

    def get_provider(
        self,
        name: str,
        server_url: str,
        auth_callback: Callable[[str], Awaitable[None]] | None = None,
        stopped: asyncio.Event | None = None,
    ) -> OAuthClientProvider:
        """Get the OAuth provider for a given name."""
        redirect_handler, callback_handler = self._generate_handler(
            name, auth_callback, stopped
        )

        return OAuthClientProvider(
            server_url=server_url,
            client_metadata=OAuthClientMetadata(
                client_name="Dive MCP Client",
                redirect_uris=[AnyUrl(self.callback_url)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
            ),
            storage=self._store.get(name),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

    async def set_oauth_code(self, code: str, state: str) -> str | None:
        """Set the OAuth code for a given name."""
        async with self._oauth_code_cond:
            name = self._oauth_code_wait.get(state)
            self._oauth_code[state] = (code, state)
            self._oauth_code_cond.notify_all()
            logger.debug(
                "OAuth code set: name=%s, code=%s, state=%s", name, code, state
            )
            return name

    async def _wait_oauth_code(
        self, name: str, wait_id: str, stopped: asyncio.Event
    ) -> tuple[str, str | None]:
        """Wait for the OAuth code for a given name."""

        def wait() -> bool:
            return (
                self._terminated.is_set()
                or stopped.is_set()
                or wait_id in self._oauth_code
            )

        async with self._oauth_code_cond:
            self._oauth_code_wait[wait_id] = name
            try:
                await self._oauth_code_cond.wait_for(wait)
            finally:
                self._oauth_code_wait.pop(wait_id, None)
            if wait_id in self._oauth_code:
                return self._oauth_code.pop(wait_id)
            return "", None

    def get_state(self, url: str) -> str | None:
        """Get the state from the URL."""
        parsed_url = urlparse(url)
        query = parse_qs(parsed_url.query)
        return query.get("state", [None])[0]

    def _generate_handler(
        self,
        name: str,
        auth_callback: Callable[[str], Awaitable[None]] | None,
        stopped: asyncio.Event | None,
    ) -> tuple[
        Callable[[str], Awaitable[None]],
        Callable[[], Awaitable[tuple[str, str | None]]],
    ]:
        """Register a handler for a given name."""
        callback_queue = asyncio.Queue()

        async def redirect_handler(auth_url: str) -> None:
            """Redirect handler."""
            state = self.get_state(auth_url)
            if stopped:
                await callback_queue.put(state)
            if auth_callback:
                await auth_callback(auth_url)

        async def callback_handler() -> tuple[str, str | None]:
            """Callback handler."""
            # empty callback
            if stopped is None:
                return "", None
            wait_id = await callback_queue.get()
            code, state = await self._wait_oauth_code(name, wait_id, stopped)
            return code, state

        return redirect_handler, callback_handler

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """Run the OAuth manager in a context."""
        try:
            yield self
        finally:
            self._terminated.set()
            async with self._oauth_code_cond:
                self._oauth_code_cond.notify_all()
