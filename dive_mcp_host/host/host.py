import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from typing import TYPE_CHECKING, Self

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import MessagesState
from langgraph.prebuilt.tool_node import ToolNode

from dive_mcp_host.host.agents import AgentFactory, get_chat_agent_factory
from dive_mcp_host.host.chat import Chat
from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.errors import ThreadNotFoundError
from dive_mcp_host.host.helpers.checkpointer import get_checkpointer
from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.host.store.base import StoreManagerProtocol
from dive_mcp_host.host.tools import McpServerInfo, ToolManager
from dive_mcp_host.host.tools.log import LogManager
from dive_mcp_host.host.tools.mcp_server import McpServer
from dive_mcp_host.host.tools.oauth import BaseTokenStore, OAuthManager
from dive_mcp_host.models import load_model

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


logger = logging.getLogger(__name__)


class DiveMcpHost(ContextProtocol):
    """The Model Context Protocol (MCP) Host.

    The DiveMcpHost class provides an async context manager interface for managing
    and interacting with language models through the Model Context Protocol (MCP).
    It handles initialization and cleanup of model instances, manages server
    connections, and provides a unified interface for agent chats.

    The MCP enables tools and models to communicate in a standardized way, allowing for
    consistent interaction patterns regardless of the underlying model implementation.

    Example:
        # Initialize host with configuration
        config = HostConfig(...)
        chat_id = ""
        async with DiveMcpHost(config) as host:
            # Send a message and get response
            async with host.chat() as chat:
                while query := input("Enter a message: "):
                    if query == "exit":
                        nonlocal thread_id
                        # save the thread_id for resume
                        chat_id = chat.chat_id
                        break
                    async for response in await chat.query(query):
                        print(response)
        ...
        # Resume chat
        async with DiveMcpHost(config) as host:
            # pass the chat_id to resume the chat
            async with host.chat(chat_id=chat_id) as chat:
                ...

    The host must be used as an async context manager to ensure proper resource
    management, including model initialization and cleanup.
    """

    def __init__(
        self,
        config: HostConfig,
        store_manager: StoreManagerProtocol | None = None,
        oauth_store: BaseTokenStore | None = None,
    ) -> None:
        """Initialize the host.

        Args:
            config: The host configuration.
            store_manager: The store manager.
            oauth_store: The OAuth store.
        """
        self._config = config
        self._model: BaseChatModel | None = None
        self._checkpointer: BaseCheckpointSaver[str] | None = None
        # Build OAuthManager kwargs, only pass callback_url if redirect_uri is set
        oauth_kwargs = {"store": oauth_store}
        if self.config.oauth_config.redirect_uri:
            oauth_kwargs["callback_url"] = self.config.oauth_config.redirect_uri
        self._tool_manager: ToolManager = ToolManager(
            configs=self._config.mcp_servers,
            log_config=self.config.log_config,
            oauth_manager=OAuthManager(**oauth_kwargs),
        )
        self._store = store_manager
        self._exit_stack: AsyncExitStack | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        async with AsyncExitStack() as stack:
            self._exit_stack = stack
            await self._init_models()
            if self._config.checkpointer:
                checkpointer = get_checkpointer(str(self._config.checkpointer.uri))
                self._checkpointer = await stack.enter_async_context(checkpointer)
                await self._checkpointer.setup()
            await stack.enter_async_context(self._tool_manager)
            try:
                yield self
            except Exception as e:
                raise e

    async def _init_models(self) -> None:
        if self._model:
            return
        model = load_model(
            self._config.llm.model_provider,
            self._config.llm.model,
            **self._config.llm.to_load_model_kwargs(),
        )
        self._model = model

    def chat[T: MessagesState](
        self,
        *,
        chat_id: str | None = None,
        user_id: str = "default",
        tools: Sequence[BaseTool] | None = None,
        get_agent_factory_method: Callable[
            [
                BaseChatModel,
                Sequence[BaseTool] | ToolNode,
                bool,
                StoreManagerProtocol | None,
            ],
            AgentFactory[T],
        ] = get_chat_agent_factory,
        system_prompt: str | Callable[[T], list[BaseMessage]] | None = None,
        disable_default_system_prompt: bool = False,
        tools_in_prompt: bool | None = None,
        volatile: bool = False,
    ) -> Chat[T]:
        """Start or resume a chat.

        Args:
            chat_id: The chat ID to use for the chat.
            user_id: The user ID to use for the chat.
            tools: The tools to use for the chat.
            system_prompt: Use a custom system prompt for the chat.
            get_agent_factory_method: The method to get the agent factory.
            volatile: if True, the chat will not be saved.
            disable_default_system_prompt: disable default system prompt
            tools_in_prompt: if True, the tools will be passed in the prompt.

        If the chat ID is not provided, a new chat will be created.
        Customize the agent factory to use a different model or tools.
        If the tools are not provided, the host will use the tools initialized in the
        host.
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        if tools is None:
            tools = self._tool_manager.langchain_tools()
        if tools_in_prompt is None:
            tools_in_prompt = self._config.llm.tools_in_prompt
        agent_factory = get_agent_factory_method(
            self._model,
            tools,
            tools_in_prompt,
            self._store,
        )
        return Chat(
            model=self._model,
            agent_factory=agent_factory,
            system_prompt=system_prompt,
            disable_default_system_prompt=disable_default_system_prompt,
            chat_id=chat_id,
            user_id=user_id,
            checkpointer=None if volatile else self._checkpointer,
        )

    async def reload(
        self,
        new_config: HostConfig,
        reloader: Callable[[], Awaitable[None]] | None = None,
        force_mcp: bool = False,
    ) -> None:
        """Reload the host with a new configuration.

        Args:
            new_config: The new configuration.
            reloader: The reloader function.
            force_mcp: If True, reload all MCP servers even if they are not changed.

        The reloader function is called when the host is ready to reload. This means
        all ongoing chats have completed and no new queries are being processed.
        The reloader should handle stopping and restarting services as needed.
        Chats can be resumed after reload by using the same chat_id.
        """
        # NOTE: Do Not restart MCP Servers when there is on-going query.
        async with self._lock:
            if self._exit_stack is None:
                raise RuntimeError("Host not initialized")

            # Update config
            old_config = self._config
            self._config = new_config

            try:
                # Reload model if needed
                if old_config.llm != new_config.llm:
                    self._model = None
                    await self._init_models()

                await self._tool_manager.reload(
                    new_configs=new_config.mcp_servers, force=force_mcp
                )

                # Reload checkpointer if needed
                if old_config.checkpointer != new_config.checkpointer:
                    if self._checkpointer is not None:
                        await self._exit_stack.aclose()
                        self._checkpointer = None

                    if new_config.checkpointer:
                        checkpointer = get_checkpointer(
                            str(new_config.checkpointer.uri)
                        )
                        self._checkpointer = await self._exit_stack.enter_async_context(
                            checkpointer
                        )
                        await self._checkpointer.setup()

                # Call the reloader function to handle service restart
                if reloader:
                    await reloader()

            except Exception as e:
                # Restore old config if reload fails
                self._config = old_config
                logging.error("Failed to reload host: %s", e)
                raise

    @property
    def config(self) -> HostConfig:
        """A copy of the current host configuration.

        Note: Do not modify the returned config. Use `reload` to change the config.
        """
        return deepcopy(self._config)

    @property
    def tools_initialized_event(self) -> asyncio.Event:
        """Get tools initialized event.

        Only useful on initial startup, not when reloading.
        """
        return self._tool_manager.initialized_event

    @property
    def tools(self) -> Sequence[BaseTool]:
        """The ACTIVE tools to the host.

        This property is read-only. Call `reload` to change the tools.
        """
        return self._tool_manager.langchain_tools()

    @property
    def mcp_server_info(self) -> dict[str, McpServerInfo]:
        """Get information about active MCP servers.

        Returns:
            A dictionary mapping server names to their capabilities and tools.
            The value will be None for any server that has not completed initialization.
        """
        return self._tool_manager.mcp_server_info

    @property
    def model(self) -> BaseChatModel:
        """The model of the host."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model

    async def get_messages(self, thread_id: str, user_id: str) -> list[BaseMessage]:
        """Get messages of a specific thread.

        Args:
            thread_id: The thread ID to retrieve messages for.
            user_id: The user ID to retrieve messages for.

        Returns:
            A list of messages.

        Raises:
            ThreadNotFoundError: If the thread is not found.
        """
        if self._checkpointer is None:
            return []

        if ckp := await self._checkpointer.aget(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                }
            }
        ):
            return ckp["channel_values"].get("messages", [])
        raise ThreadNotFoundError(thread_id)

    async def delete_thread(self, thread_id: str) -> None:
        """Delete a thread.

        Args:
            thread_id: The thread ID to delete.
        """
        if self._checkpointer:
            await self._checkpointer.adelete_thread(thread_id)

    @property
    def log_manager(self) -> LogManager:
        """Get the log manager."""
        return self._tool_manager.log_manager

    @property
    def oauth_manager(self) -> OAuthManager:
        """Get the OAuth manager."""
        return self._tool_manager.oauth_manager

    def get_mcp_server(self, name: str) -> McpServer:
        """Get MCP server."""
        return self._tool_manager.mcp_servers[name]

    async def restart_mcp_server(self, name: str) -> McpServerInfo:
        """Restart MCP server."""
        return await self._tool_manager.restart_mcp_server(name)
