from asyncio import Event
from collections.abc import Callable
from enum import StrEnum
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal, Protocol

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, V
from langgraph.config import get_config
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from dive_mcp_host.host.prompt import PromptType

if TYPE_CHECKING:
    from dive_mcp_host.host.tools.elicitation_manager import ElicitationManager
    from dive_mcp_host.skills.manager import SkillManager

logger = getLogger(__name__)


class ConfigurableKey(StrEnum):
    """Enum for RunnableConfig.configurable keys."""

    # Thread id is also known as chat_id
    THREAD_ID = "thread_id"
    USER_ID = "user_id"
    MAX_INPUT_TOKENS = "max_input_tokens"
    OVERSIZE_POLICY = "oversize_policy"
    ABORT_SIGNAL = "abort_signal"
    ELICITATION_MANAGER = "elicitation_manager"
    STREAM_WRITER = "stream_writer"
    LOCALE = "locale"
    SKILL_MANAGER = "skill_manager"
    TOOL_CALL_ID = "tool_call_id"
    DRY_RUN = "dry_run"


def ensure_config(config: RunnableConfig | None) -> RunnableConfig:
    """Ensure config is available, falling back to LangGraph context if needed.

    InjectedToolArg may not work correctly with LangGraph's ToolNode,
    so we use get_config() as a fallback to get the config from context.
    """
    if config is not None and config.get("configurable"):
        return config

    try:
        ctx_config = get_config()
        if ctx_config and ctx_config.get("configurable"):
            logger.debug("Using config from LangGraph context")
            return ctx_config
    except (RuntimeError, LookupError):
        pass

    return config or {}


def get_stream_writer(
    config: RunnableConfig,
) -> Callable[[tuple[str, Any]], None]:
    """Extract stream writer from config or LangGraph context.

    Priority:
    1. Explicitly set stream_writer in config (used by InstallerAgent)
    2. LangGraph's get_stream_writer() (used when running in ToolNode)
    3. No-op lambda as fallback
    """
    # First check if stream_writer is explicitly set in config (InstallerAgent case)
    writer = config.get("configurable", {}).get(ConfigurableKey.STREAM_WRITER)
    if writer is not None:
        logger.debug("Using stream_writer from config")
        return writer

    # Try to get stream writer from LangGraph context (ToolNode case)
    try:
        from langgraph.config import get_stream_writer as lg_get_stream_writer

        writer = lg_get_stream_writer()
        if writer is not None:
            logger.debug("Using stream_writer from LangGraph context")
            return writer
        logger.debug("LangGraph get_stream_writer() returned None")
    except (ImportError, RuntimeError, LookupError) as e:
        logger.debug("Could not get stream writer from LangGraph context: %s", e)

    # Fallback to no-op
    logger.debug("Falling back to no-op stream_writer")
    return lambda _: None


def get_tool_call_id(config: RunnableConfig) -> str | None:
    """Extract tool_call_id from config metadata."""
    return config.get("metadata", {}).get(ConfigurableKey.TOOL_CALL_ID)


def get_thread_id(config: RunnableConfig) -> str | None:
    """Extract thread id from config metadata."""
    return config.get("metadata", {}).get(ConfigurableKey.THREAD_ID)


def get_dry_run(config: RunnableConfig) -> bool:
    """Extract dry_run setting from config."""
    return config.get("configurable", {}).get(ConfigurableKey.DRY_RUN, False)


def get_abort_signal(config: RunnableConfig) -> Event | None:
    """Extract abort signal from config."""
    return config.get("configurable", {}).get(ConfigurableKey.ABORT_SIGNAL)


def get_skill_manager(config: RunnableConfig) -> "SkillManager | None":
    """Get skill manager from config."""
    return config.get("configurable", {}).get(ConfigurableKey.SKILL_MANAGER)


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
        abort_signal: Event | None = None,
        elicitation_manager: "ElicitationManager | None" = None,
        stream_writer: "Any | None" = None,
        locale: str = "en",
        skill_manager: "SkillManager | None" = None,
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
                ConfigurableKey.ABORT_SIGNAL: abort_signal,
                ConfigurableKey.ELICITATION_MANAGER: elicitation_manager,
                ConfigurableKey.STREAM_WRITER: stream_writer,
                ConfigurableKey.LOCALE: locale,
                ConfigurableKey.SKILL_MANAGER: skill_manager,
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
