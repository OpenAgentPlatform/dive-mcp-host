import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any, Self, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import MessagesState
from langgraph.store.base import BaseStore
from langgraph.types import StreamMode

from dive_mcp_host.host.agents import AgentFactory, V
from dive_mcp_host.host.errors import (
    GraphNotCompiledError,
    MessageTypeError,
    ThreadNotFoundError,
)
from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.host.prompt import default_system_prompt

logger = logging.getLogger(__name__)


class Chat[STATE_TYPE: MessagesState](ContextProtocol):
    """A chat with a language model."""

    def __init__(  # noqa: PLR0913, too many arguments
        self,
        model: BaseChatModel,
        agent_factory: AgentFactory[STATE_TYPE],
        *,
        system_prompt: str | Callable[[STATE_TYPE], list[BaseMessage]] | None = None,
        chat_id: str | None = None,
        user_id: str = "default",
        store: BaseStore | None = None,
        checkpointer: BaseCheckpointSaver[V] | None = None,
        disable_default_system_prompt: bool = False,
    ) -> None:
        """Initialize the chat.

        Args:
            model: The language model to use for the chat.
            agent_factory: The agent factory to use for the chat.
            system_prompt: The system prompt to use for the chat.
            chat_id: The ID of the chat. (langgraph thread id)
            user_id: The user ID to use for the chat.
            store: The store to use for the chat.
            checkpointer: The langgraph checkpointer to use for the chat.
            disable_default_system_prompt: disable default system prompt

        The agent_factory is called only once to compile the agent.
        """
        self._chat_id: str = chat_id if chat_id else uuid.uuid4().hex
        self._user_id: str = user_id
        self._store = store
        self._checkpointer = checkpointer
        self._model = model
        self._system_prompt = system_prompt
        self._agent: CompiledGraph | None = None
        self._agent_factory: AgentFactory[STATE_TYPE] = agent_factory
        self._abort_signal: asyncio.Event | None = None
        self._disable_default_system_prompt = disable_default_system_prompt

    @property
    def active_agent(self) -> CompiledGraph:
        """The active agent of the chat."""
        if self._agent is None:
            raise GraphNotCompiledError(self._chat_id)
        return self._agent

    @property
    def chat_id(self) -> str:
        """The chat ID of the chat."""
        return self._chat_id

    def abort(self) -> None:
        """Abort the running query."""
        if self._abort_signal is None:
            return
        self._abort_signal.set()

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        if self._system_prompt is None and not self._disable_default_system_prompt:
            system_prompt = default_system_prompt()
        else:
            system_prompt = self._system_prompt

        if callable(system_prompt):
            prompt = system_prompt
        else:
            prompt = (
                self._agent_factory.create_prompt(system_prompt=system_prompt)
                if system_prompt
                else ""
            )

        # we can do something to the prompt here.
        self._agent = self._agent_factory.create_agent(
            prompt=prompt,
            checkpointer=self._checkpointer,
            store=self._store,
        )
        try:
            yield self
        finally:
            self._agent = None

    async def _get_updates_for_resend(
        self,
        resend: list[BaseMessage],
        update: list[BaseMessage],
    ) -> list[BaseMessage]:
        if not self._checkpointer:
            return update
        resend_map = {msg.id: msg for msg in resend}
        to_update = [i for i in update if i.id not in resend_map]
        if state := await self.active_agent.aget_state(
            RunnableConfig(
                configurable={
                    "thread_id": self._chat_id,
                    "user_id": self._user_id,
                },
            )
        ):
            drop_after = False
            for msg in cast(MessagesState, state.values)["messages"]:
                assert msg.id is not None  # all messages from the agent have an ID
                if msg.id in resend_map:
                    drop_after = True
                elif drop_after:
                    to_update.append(RemoveMessage(msg.id))
            return to_update
        raise ThreadNotFoundError(self._chat_id)

    def query(
        self,
        query: str | HumanMessage | list[BaseMessage] | None,
        *,
        stream_mode: list[StreamMode] | StreamMode | None = "messages",
        modify: list[BaseMessage] | None = None,
        is_resend: bool = False,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Query the chat.

        Args:
            query: The query to ask the chat. Can be a string, HumanMessage, or
                list of messages.
                For resending messages, pass the messages to resend here.
            stream_mode: The mode to stream the response.
            modify: Messages to modify in the chat state. Used for modifying
                messages without resending, e.g. when confirming tool call parameters.
            is_resend: If True, indicates that query contains messages to resend. The
                messages in query and all subsequent messages in the state will be
                removed. Any messages in modify that appear in query will be ignored.

        Returns:
            An async generator of the response.

        Raises:
            MessageTypeError: If the messages to modify are invalid.
        """

        async def _stream_response() -> AsyncGenerator[dict[str, Any] | Any, None]:
            query_msgs = _convert_query_to_messages(query)
            if is_resend and query_msgs:
                if len(query_msgs) == 0 or not all(
                    isinstance(msg, BaseMessage) and msg.id for msg in query_msgs
                ):
                    raise MessageTypeError("Resending messages must has an ID")
                query_msgs += await self._get_updates_for_resend(
                    query_msgs, modify or []
                )
            elif modify:
                query_msgs = [*query_msgs, *modify]
            signal = asyncio.Event()
            self._abort_signal = signal
            if query_msgs:
                init_state = self._agent_factory.create_initial_state(query=query_msgs)
            else:
                init_state = None
            logger.debug("init_state: %s", query_msgs)
            config = self._agent_factory.create_config(
                user_id=self._user_id,
                thread_id=self._chat_id,
            )
            async for response in self.active_agent.astream(
                input=init_state,
                stream_mode=stream_mode,
                config=config,
            ):
                if signal.is_set():
                    break
                yield response

        return _stream_response()


def _convert_query_to_messages(
    query: str | HumanMessage | list[BaseMessage] | None,
) -> list[BaseMessage]:
    if isinstance(query, BaseMessage):
        return [query]
    if isinstance(query, str):
        return [HumanMessage(content=query)]
    if isinstance(query, list):
        return [
            i if isinstance(i, BaseMessage) else HumanMessage(content=i) for i in query
        ]
    return []


class MessageChunkHolder:
    """Message chunk holder."""

    def __init__(self) -> None:
        """Initialize message chunk holder."""
        self._merged: dict[str, BaseMessage] = {}

    def feed[T: BaseMessage | BaseMessageChunk](self, chunk: T) -> T | None:
        """Feed a chunk, return a combined message if done."""
        if isinstance(chunk, BaseMessageChunk) and chunk.id:
            m = cast(T, s + chunk) if (s := self._merged.get(chunk.id)) else chunk
            self._merged[chunk.id] = m
            if m.response_metadata.keys() & {
                "finish_reason",
                "stop_reason",
                "done",
            }:
                return m
            return None
        return chunk

    def partial_merged[T: BaseMessage](self, chunk: T) -> T:
        """Return partial merged message."""
        if isinstance(chunk, BaseMessageChunk) and chunk.id:
            m = cast(T, s + chunk) if (s := self._merged.get(chunk.id)) else chunk
            self._merged[chunk.id] = m
            return m
        return chunk
