import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

from dive_mcp_host.host.errors import McpSessionNotRunningError
from dive_mcp_host.host.tools.hack import ClientSession
from dive_mcp_host.host.tools.model_types import ChatID
from dive_mcp_host.host.tools.oauth import AuthorizationProgress

logger = getLogger(__name__)


MAX_IDLE_TIME = 300


@dataclass(slots=True)
class _SessionStoreItem:
    """Session store item.

    The session store item is created when a new session is created.
    - chat_id: The chat id of the session.
    - task: The task of the session watcher.
    - session: The session object.
    - initialized: An event that is set when the session is initialized.
      When the session goes into error state, the event will be set too.
    - cleared: Whether the resources of this session are cleared.
    - client_tasks: The tasks that use the session.
    - exec: The exception that occurred in the session.
    - active_ts: The timestamp when the session is active.
    """

    chat_id: ChatID
    task: asyncio.Task[None] | None = None
    session: ClientSession | None = None
    initialized: asyncio.Event = field(default_factory=asyncio.Event)
    cleared: bool = False
    client_tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    exec: BaseException | None = None
    active_ts: float = field(default_factory=time.time)

    async def waiting_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            if (
                time.time() - self.active_ts > MAX_IDLE_TIME
                and len(self.client_tasks) == 0
            ):
                return
            if self.session:
                await self.session.send_ping()

    def add_task(self, task: asyncio.Task[Any]) -> None:
        self.active_ts = time.time()
        self.client_tasks.append(task)


class ServerSessionStore:
    """Session Store for a running MCP server."""

    __slots__ = ("_auth_handlers", "_map", "_mcp_server_name")

    def __init__(self, mcp_server_name: str) -> None:
        """Initialize the session store."""
        self._map: dict[ChatID, _SessionStoreItem] = {}
        self._auth_handlers: dict[
            ChatID, Callable[[AuthorizationProgress], Awaitable[None]]
        ] = {}
        self._mcp_server_name = mcp_server_name

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, chat_id: ChatID) -> _SessionStoreItem:
        return self._map[chat_id]

    async def _session_watcher(
        self,
        session_ctx: Callable[[], AbstractAsyncContextManager[ClientSession]],
        chat_id: ChatID,
    ) -> None:
        stored_session = self._map[chat_id]
        try:
            async with session_ctx(
                auth_handler=self._wrapper_auth_handler(chat_id)
            ) as session:
                stored_session.session = session
                stored_session.initialized.set()
                stored_session.active_ts = time.time()
                await stored_session.waiting_loop()
        except Exception as e:
            logger.error(
                "Session error, chat_id: %s, name: %s, error: %s",
                chat_id,
                self._mcp_server_name,
                e,
            )
            if not isinstance(e, asyncio.CancelledError):
                stored_session.exec = e
            raise
        finally:
            stored_session.session = None
            self._error_cleanup(stored_session.task, stored_session, None)

    def _wrapper_auth_handler(self, chat_id: ChatID) -> None:
        """Wrapper the auth handler."""

        async def auth_handler(progress: AuthorizationProgress) -> None:
            """Handle the authorization progress."""
            if handler := self._auth_handlers.get(chat_id):
                await handler(progress)

        return auth_handler

    def _error_cleanup(
        self,
        self_task: asyncio.Task[Any] | None,
        stored_session: _SessionStoreItem,
        e: Exception | None,
    ) -> None:
        if stored_session.cleared:
            return
        if stored_session.chat_id in self._map:
            del self._map[stored_session.chat_id]
        if e and not isinstance(e, asyncio.CancelledError):
            stored_session.exec = e
        stored_session.session = None
        stored_session.initialized.set()
        if stored_session.task != self_task and stored_session.task:
            stored_session.task.cancel()
        for task in stored_session.client_tasks:
            if task != self_task:
                task.cancel()
        stored_session.cleared = True

    @asynccontextmanager
    async def get_session_ctx_mgr(
        self,
        chat_id: str,
        session_creator: Callable[[], AbstractAsyncContextManager[ClientSession]],
        auth_handler: Callable[[AuthorizationProgress], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """Create a new session or return the existing one.

        session_creator: The context manager that creates a new session.

        When no existing session is found in the store, a new session will be
        created in the session_watcher.
        Each chat's session is monitored by an independent session_watcher.
        The session watcher automatically closes idle sessions.

        When a session encounters issues, all tasks using that session will be
        cancelled.

        Sessions that have ended or encountered errors are immediately removed
        from the store and will not be used again.
        If the same chat_id is used again, a new session will be recreated.
        """
        stored_session = self._map.get(chat_id)
        self._auth_handlers[chat_id] = auth_handler

        if not stored_session:
            stored_session = _SessionStoreItem(chat_id=chat_id)
            self._map[chat_id] = stored_session
            logger.debug(
                "Create new session for chat_id: %s, name: %s",
                chat_id,
                self._mcp_server_name,
            )

            stored_session.task = asyncio.create_task(
                self._session_watcher(session_creator, chat_id),
                name=f"session_create_func-{self._mcp_server_name}-{chat_id}",
            )
        else:
            logger.debug(
                "Found prev session for chat_id: %s, name: %s",
                chat_id,
                self._mcp_server_name,
            )
        if current_task := asyncio.current_task():
            stored_session.add_task(current_task)
        else:
            raise RuntimeError("No current task")
        try:
            await stored_session.initialized.wait()
            if not stored_session.session:
                raise McpSessionNotRunningError(self._mcp_server_name, chat_id)
            yield stored_session.session
        except asyncio.CancelledError as e:
            if stored_session.exec:
                raise stored_session.exec from e
            raise
        except Exception as e:
            logger.error(
                "Session error, chat_id: %s, name: %s, error: %s",
                chat_id,
                self._mcp_server_name,
                e,
            )
            self._error_cleanup(current_task, stored_session, e)
            raise
        finally:
            stored_session.client_tasks.remove(current_task)

    async def cleanup(self) -> None:
        """Cleanup the session store."""
        for i in self._map.values():
            if i.task:
                i.task.cancel()
            for task in i.client_tasks:
                task.cancel()
            i.initialized.set()
            i.session = None
            i.exec = None
            i.client_tasks.clear()
        self._map.clear()
