from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Protocol, Self


class ContextProtocol(Protocol):
    """A context for a conversation.

    The ContextProtocol defines an interface for managing asynchronous context
    managers that provide conversation contexts. It ensures proper resource
    management and prevents reentrant usage.

    This protocol is used by classes that need to provide a context for
    conversations, such as database connections, model instances, or server
    connections.

    Example:
        class DatabaseContext(ContextProtocol):
            async def _in_context(self) -> AsyncGenerator[Self, None]:
                # Set up database connection
                try:
                    await self.connect()
                    yield self
                finally:
                    await self.disconnect()

        async with DatabaseContext() as db:
            # Use db here
            pass
    """

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        yield self

    __gen: AsyncGenerator[Self, None] | None = None

    async def __aenter__(self) -> Self:
        if self.__gen is not None:
            raise RuntimeError("No Reentrant usage of ContextProtocol")
        self.__gen = self._run_in_context()
        return await anext(self.__gen)

    async def __aexit__(  # A copy of asynccontextmanager.__aexit__.
        self,
        typ: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self.__gen is None:
            raise RuntimeError("No context to exit")
        gen = self.__gen
        self.__gen = None
        if typ is None:
            try:
                await anext(gen)
            except StopAsyncIteration:
                return False
            else:
                try:
                    raise RuntimeError("generator didn't stop")
                finally:
                    await gen.aclose()
        else:
            if value is None:
                value = typ()
            try:
                await gen.athrow(value)
            except StopAsyncIteration as exc:
                return exc is not value
            except RuntimeError as exc:
                if exc is value:
                    exc.__traceback__ = traceback
                    return False
                if (
                    isinstance(value, StopIteration | StopAsyncIteration)
                    and exc.__cause__ is value
                ):
                    value.__traceback__ = traceback
                    return False
                raise
            except BaseException as exc:
                if exc is not value:
                    raise
                exc.__traceback__ = traceback
                return False
            try:
                raise RuntimeError("generator didn't stop after athrow()")
            finally:
                await gen.aclose()
