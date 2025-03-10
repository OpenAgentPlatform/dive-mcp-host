from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Self

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode

from dive_mcp.host.agents import AgentFactory, get_chat_agent_factory
from dive_mcp.host.conf import HostConfig
from dive_mcp.host.conversation import Conversation
from dive_mcp.host.helpers.context import ContextProtocol


class DiveMcpHost(ContextProtocol):
    """The Model Context Protocol (MCP) Host.

    The DiveMcpHost class provides an async context manager interface for managing
    and interacting with language models through the Model Context Protocol (MCP).
    It handles initialization and cleanup of model instances, manages server
    connections, and provides a unified interface for agent conversations.

    The MCP enables tools and models to communicate in a standardized way, allowing for
    consistent interaction patterns regardless of the underlying model implementation.

    Example:
        # Initialize host with configuration
        config = HostConfig(...)
        thread_id = ""
        async with DiveMcpHost(config) as host:
            # Send a message and get response
            async with host.conversation() as conversation:
                while query := input("Enter a message: "):
                    if query == "exit":
                        nonlocal thread_id
                        # save the thread_id for resume
                        thread_id = conversation.thread_id
                        break
                    async for response in await conversation.query(query):
                        print(response)
        ...
        # Resume conversation
        async with DiveMcpHost(config) as host:
            # pass the thread_id to resume the conversation
            async with host.conversation(thread_id=thread_id) as conversation:
                ...

    The host must be used as an async context manager to ensure proper resource
    management, including model initialization and cleanup.
    """

    def __init__(
        self,
        config: HostConfig,
    ) -> None:
        """Initialize the host.

        Args:
            config: The host configuration.
        """
        self._config = config
        self._model: BaseChatModel | None = None
        self._tools: Sequence[BaseTool] = []

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        try:
            # TODO: Add database context
            # async with database:
            #     yield self
            await self._init_models()
            yield self
        except Exception as e:
            raise e

    async def _init_tools(self) -> None:
        if self._tools:
            return
        raise NotImplementedError

    async def _init_models(self) -> None:
        if self._model:
            return
        raise NotImplementedError

    async def conversation[T](
        self,
        *,
        thread_id: str | None = None,
        user_id: str = "default",
        tools: Sequence[BaseTool] | None = None,
        get_agent_factory_method: Callable[
            [BaseChatModel, Sequence[BaseTool] | ToolNode],
            AgentFactory[T],
        ] = get_chat_agent_factory,
    ) -> Conversation[T]:
        """Start or resume a conversation.

        Args:
            thread_id: The thread ID to use for the conversation.
            user_id: The user ID to use for the conversation.
            tools: The tools to use for the conversation.
            get_agent_factory_method: The method to get the agent factory.

        If the thread ID is not provided, a new thread will be created.
        Customize the agent factory to use a different model or tools.
        If the tools are not provided, the host will use the tools initialized in the
        host.
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        if tools is None:
            tools = self._tools
        agent_factory = get_agent_factory_method(
            self._model,
            tools,
        )
        return Conversation(
            model=self._model,
            agent_factory=agent_factory,
            thread_id=thread_id,
            user_id=user_id,
        )

    async def reload(
        self,
        new_config: HostConfig,
        reloader: Callable[[], Awaitable[None]],
    ) -> None:
        """Reload the host with a new configuration.

        Args:
            new_config: The new configuration.
            reloader: The reloader function.

        The reloader function is called when the host is ready to reload. This means
        all ongoing conversations have completed and no new queries are being processed.
        The reloader should handle stopping and restarting services as needed.
        Conversations can be resumed after reload by using the same thread_id.
        """
        # NOTE: Do Not restart MCP Servers when there is on-going query.
        raise NotImplementedError

    @property
    def tools(self) -> Sequence[BaseTool]:
        """The tools available to the host.

        This property is read-only. Call `reload` to change the tools.
        """
        return self._tools
