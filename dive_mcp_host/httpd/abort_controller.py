from asyncio import Lock
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager


class AbortController:
    """AbortController is a class that allows you to abort a task."""

    def __init__(self) -> None:
        """Initialize the AbortController."""
        self._mapped_events: dict[str, Callable[[], None]] = {}
        self._locks: dict[str, Lock] = defaultdict(Lock)

    @asynccontextmanager
    async def abort_signal(
        self, key: str, func: Callable[[], None]
    ) -> AsyncGenerator[None, None]:
        """Get the abort signal for a given key."""
        async with self._locks[key]:
            self._mapped_events[key] = func
            try:
                yield
            finally:
                del self._mapped_events[key]

    async def abort(self, key: str) -> bool:
        """Abort the task for a given key."""
        if func := self._mapped_events.get(key):
            func()
            return True
        return False
