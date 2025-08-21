from enum import StrEnum
from typing import Literal, Protocol

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, V
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from dive_mcp_host.host.prompt import PromptType


class ConfigurableKey(StrEnum):
    """Enum for RunnableConfig.configurable keys."""

    # Thread id is also known as chat_id
    THREAD_ID = "thread_id"
    USER_ID = "user_id"
    MAX_INPUT_TOKENS = "max_input_tokens"
    OVERSIZE_POLICY = "oversize_policy"


# XXX is there any better way to do this?
class AgentFactory[T: MessagesState](Protocol):
    """A factory for creating agents.

    Implementing this protocol to create your own custom agent.
    Pass the factory to the host to create an agent for the chat.
    """

    def create_agent(
        self,
        *,
        prompt: PromptType | ChatPromptTemplate,
        checkpointer: BaseCheckpointSaver[V] | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
    ) -> CompiledStateGraph:
        """Create an agent.

        Args:
            prompt: The prompt to use for the agent.
            checkpointer: A langgraph checkpointer to keep the agent's state.
            store: A langgraph store for long-term memory.
            debug: Whether to enable debug mode for the agent.

        Returns:
            The compiled agent.
        """
        ...

    def create_config(
        self,
        *,
        user_id: str,
        thread_id: str,
        max_input_tokens: int | None = None,
        oversize_policy: Literal["window"] | None = None,
    ) -> RunnableConfig | None:
        """Create a config for the agent.

        Override this to customize the config for the agent.
        The default implementation returns this config:
        {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
            "recursion_limit": 100,
        }
        """
        return {
            "configurable": {
                ConfigurableKey.THREAD_ID: thread_id,
                ConfigurableKey.USER_ID: user_id,
                ConfigurableKey.MAX_INPUT_TOKENS: max_input_tokens,
                ConfigurableKey.OVERSIZE_POLICY: oversize_policy,
            },
            "recursion_limit": 102,
        }

    def create_initial_state(
        self,
        *,
        query: str | HumanMessage | list[BaseMessage],
    ) -> T:
        """Create an initial state for the query."""
        ...

    def state_type(
        self,
    ) -> type[T]:
        """Get type of the state."""
        ...

    def create_prompt(
        self,
        *,
        system_prompt: str,
    ) -> ChatPromptTemplate:
        """Create a prompt for the agent.

        Override this to customize the prompt for the agent.
        The default implementation returns a prompt with a placeholder for the messages.
        """
        return ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
            [
                ("system", system_prompt),
                ("placeholder", "{messages}"),
            ],
        )


def initial_messages(
    query: str | HumanMessage | list[AnyMessage | BaseMessage],
) -> list[AnyMessage]:
    """Create an initial message for your state.

    The state must contain a 'messages' key with type list[BaseMessage].
    This utility helps convert the query into list[BaseMessage], regardless of whether
    the query is a str or BaseMessage.

    Args:
        query: The query to create the initial message from.

    Returns:
        A list of HumanMessage objects.

    """
    if isinstance(query, list):
        messages = []
        for q in query:
            messages.append(
                q if isinstance(q, BaseMessage) else HumanMessage(content=q)
            )
        return messages
    return [query] if isinstance(query, BaseMessage) else [HumanMessage(content=query)]
