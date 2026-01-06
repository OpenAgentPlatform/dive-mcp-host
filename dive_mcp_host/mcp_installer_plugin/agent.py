"""LangGraph-based MCP Server Installer Agent.

This module implements the core agent logic for installing MCP servers.
The agent uses LangGraph to orchestrate the installation process with
tool calls for fetch, bash, and filesystem operations.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.managed import IsLastStep, RemainingSteps  # noqa: TC002
from langgraph.prebuilt.tool_node import ToolNode, ToolRuntime

from dive_mcp_host.mcp_installer_plugin.events import (
    InstallerProgress,
    InstallerResult,
)
from dive_mcp_host.mcp_installer_plugin.prompt import get_installer_system_prompt
from dive_mcp_host.mcp_installer_plugin.tools import get_installer_tools

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from dive_mcp_host.host.tools.elicitation_manager import ElicitationManager

logger = logging.getLogger(__name__)


class InstallerAgentState(MessagesState):
    """State for the installer agent."""

    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    current_phase: str
    server_name: str | None
    final_config: dict[str, Any] | None
    error: str | None
    install_confirmed: bool | None  # None = pending, True = approved, False = denied


MINIMUM_STEPS_FOR_TOOL_CALL = 2
MAX_ITERATIONS = 30


def _complete_tool_calls(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """Insert ToolMessage if not exists for incomplete tool calls."""
    tool_calls: list[tuple[int, ToolCall]] = []
    tool_messages: set[str] = set()

    for idx, message in enumerate(messages):
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_calls.append((idx, tool_call))
        elif isinstance(message, ToolMessage):
            tool_messages.add(message.tool_call_id)

    tool_calls.reverse()
    messages = list(messages)

    for idx, tool_call in tool_calls:
        if tool_call["id"] not in tool_messages:
            messages.insert(
                idx + 1,
                ToolMessage(content="canceled", tool_call_id=tool_call["id"]),
            )

    return messages


@dataclass
class InstallerAgent:
    """LangGraph-based agent for installing MCP servers.

    This agent orchestrates the installation process using LLM reasoning
    and tool calls for fetch, bash, and filesystem operations.
    """

    model: BaseChatModel
    """The LLM model to use for reasoning."""

    elicitation_manager: ElicitationManager
    """Elicitation manager for user approval requests."""

    tools: list[BaseTool] = field(default_factory=get_installer_tools)
    """Tools available to the agent."""

    max_iterations: int = MAX_ITERATIONS
    """Maximum number of agent iterations."""

    dry_run: bool = False
    """If True, bash commands will be simulated without actual execution."""

    mcp_reload_callback: Callable[[], Any] | None = None
    """Callback to reload MCP servers after adding a new server (deprecated)."""

    locale: str = "en"
    """Locale for user-facing messages (e.g., 'en', 'zh-TW', 'ja')."""

    _graph: CompiledStateGraph | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Build the agent graph after initialization."""
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the LangGraph for the installer agent.

        Graph structure:
            START → call_model → (has tool_calls?)
                    ├─ yes → tools → call_model
                    └─ no  → END

        Note: User confirmation is now handled by the request_confirmation tool,
        which the agent can call at any time to ask for user approval.
        """
        graph = StateGraph(InstallerAgentState)

        # Add nodes
        graph.add_node("call_model", self._call_model)
        graph.add_node("tools", self._create_tool_node())

        # Set entry point
        graph.set_entry_point("call_model")

        # Add conditional edges from model
        graph.add_conditional_edges(
            "call_model",
            self._should_continue,
            {"continue": "tools", "end": END},
        )

        # Add edge from tools back to model
        graph.add_edge("tools", "call_model")

        self._graph = graph.compile()

    def _create_tool_node(self) -> ToolNode:
        """Create the tool node with injected config."""

        class InstallerToolNode(ToolNode):
            """Custom tool node that injects tool_call_id and streams events."""

            async def _arun_one(
                self,
                call: ToolCall,
                input_type: Literal["list", "dict", "tool_calls"],
                tool_runtime: ToolRuntime,
            ) -> ToolMessage:
                if "metadata" in tool_runtime.config:
                    tool_runtime.config["metadata"]["tool_call_id"] = call["id"]
                else:
                    tool_runtime.config["metadata"] = {"tool_call_id": call["id"]}

                # Execute the tool (tools emit agent_tool_call/result events directly)
                result = await super()._arun_one(call, input_type, tool_runtime)
                return cast(ToolMessage, result)

        return InstallerToolNode(self.tools)

    async def _call_model(
        self, state: InstallerAgentState, config: RunnableConfig
    ) -> InstallerAgentState:
        """Call the LLM model."""
        logger.info("InstallerAgent._call_model() - START")
        stream_writer = config.get("configurable", {}).get(
            "stream_writer", lambda _: None
        )

        # Complete any incomplete tool calls
        messages = list(_complete_tool_calls(state["messages"]))
        logger.info("InstallerAgent._call_model() - messages count: %d", len(messages))

        # Check abort signal
        abort_signal = config.get("configurable", {}).get("abort_signal")
        if abort_signal and abort_signal.is_set():
            logger.info("InstallerAgent._call_model() - ABORTED")
            return cast(InstallerAgentState, {"messages": [], "error": "Aborted"})

        # Prepare model with tools
        model_with_tools = self.model.bind_tools(self.tools)

        # Add system prompt if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [
                SystemMessage(content=get_installer_system_prompt(self.locale)),
                *messages,
            ]

        # Emit progress before calling model
        stream_writer(
            (
                InstallerProgress.NAME,
                InstallerProgress(
                    phase="analyzing",
                    message="Calling LLM to analyze request...",
                ),
            )
        )

        # Call the model with abort support
        logger.info("InstallerAgent._call_model() - calling model.ainvoke()")
        try:
            model_task = asyncio.create_task(model_with_tools.ainvoke(messages, config))

            # If abort signal exists, wait for either model or abort
            if abort_signal is not None:
                abort_task = asyncio.create_task(abort_signal.wait())
                done, pending = await asyncio.wait(
                    [model_task, abort_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                # Check if aborted
                if abort_task in done:
                    logger.info("InstallerAgent._call_model() - ABORTED")
                    return cast(
                        InstallerAgentState, {"messages": [], "error": "Aborted"}
                    )

                response = model_task.result()
            else:
                response = await model_task

            logger.info(
                "InstallerAgent._call_model() - model response received, "
                "has_tool_calls: %s, content_length: %d",
                bool(isinstance(response, AIMessage) and response.tool_calls),
                len(str(response.content)) if response.content else 0,
            )
        except Exception as e:
            logger.exception("InstallerAgent._call_model() - model.ainvoke() FAILED")
            stream_writer(
                (
                    InstallerProgress.NAME,
                    InstallerProgress(
                        phase="failed",
                        message=f"Model call failed: {e}",
                    ),
                )
            )
            raise

        # Check if we need more steps
        if self._check_more_steps_needed(state, response):
            response = AIMessage(
                id=response.id,
                content="I've reached the maximum number of steps. "
                "Here's a summary of what was done.",
            )

        logger.info("InstallerAgent._call_model() - END")
        return cast(InstallerAgentState, {"messages": [response]})

    def _check_more_steps_needed(
        self, state: InstallerAgentState, response: BaseMessage
    ) -> bool:
        """Check if the model response would exceed step limits."""
        has_tool_calls = (
            isinstance(response, AIMessage)
            and response.tool_calls is not None
            and len(response.tool_calls) > 0
        )
        remaining_steps = state.get("remaining_steps", self.max_iterations)
        is_last_step = state.get("is_last_step", False)

        return (is_last_step and has_tool_calls) or (
            remaining_steps < MINIMUM_STEPS_FOR_TOOL_CALL and has_tool_calls
        )

    def _should_continue(self, state: InstallerAgentState) -> str:
        """Determine if the agent should continue or end."""
        messages = state["messages"]
        if not messages:
            return "end"

        last_message = messages[-1]

        # If there are tool calls, continue to execute tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"

        return "end"

    async def run(
        self,
        query: str,
        stream_writer: Callable[[tuple[str, Any]], None] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the installer agent.

        Args:
            query: The installation request (e.g., "Install the fetch MCP server").
            stream_writer: Callback to send events to the frontend.
            abort_signal: Signal to abort the agent.

        Yields:
            State updates from the agent.
        """
        logger.info("InstallerAgent.run() - START, query: %s", query[:100])
        if self._graph is None:
            raise RuntimeError("Agent graph not built")

        # Create initial state
        initial_state = InstallerAgentState(
            messages=[HumanMessage(content=query)],
            is_last_step=False,
            remaining_steps=self.max_iterations,
            current_phase="analyzing",
            server_name=None,
            final_config=None,
            error=None,
            install_confirmed=None,
        )

        # Emit initial progress
        if stream_writer:
            stream_writer(
                (
                    InstallerProgress.NAME,
                    InstallerProgress(
                        phase="analyzing",
                        message="Analyzing installation request...",
                    ),
                )
            )

        # Create config with stream writer, elicitation manager, and dry_run
        config = RunnableConfig(
            configurable={
                "stream_writer": stream_writer or (lambda _: None),
                "elicitation_manager": self.elicitation_manager,
                "abort_signal": abort_signal,
                "dry_run": self.dry_run,
                "mcp_reload_callback": self.mcp_reload_callback,  # deprecated
            },
            recursion_limit=self.max_iterations + 5,
        )

        try:
            # Run the agent
            logger.info("InstallerAgent.run() - starting graph.astream()")
            iteration = 0
            async for chunk in self._graph.astream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                # Check abort signal between iterations
                if abort_signal and abort_signal.is_set():
                    logger.info("InstallerAgent.run() - ABORTED at iter %d", iteration)
                    if stream_writer:
                        stream_writer(
                            (
                                InstallerProgress.NAME,
                                InstallerProgress(
                                    phase="aborted",
                                    message="Installation aborted by user.",
                                ),
                            )
                        )
                    return

                iteration += 1
                logger.info(
                    "InstallerAgent.run() - iteration %d, chunk keys: %s",
                    iteration,
                    list(chunk.keys()),
                )
                yield chunk

                # Update progress based on state changes
                if stream_writer and "call_model" in chunk:
                    messages = chunk["call_model"].get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            # Check what kind of tools are being called
                            tool_names = [tc["name"] for tc in last_msg.tool_calls]
                            if "fetch" in tool_names:
                                phase = "fetching_info"
                                message = "Fetching documentation and package info..."
                            elif "bash" in tool_names:
                                phase = "installing"
                                message = "Executing installation commands..."
                            elif "write_file" in tool_names:
                                phase = "configuring"
                                message = "Writing configuration..."
                            else:
                                phase = "preparing"
                                message = "Preparing installation..."

                            stream_writer(
                                (
                                    InstallerProgress.NAME,
                                    InstallerProgress(phase=phase, message=message),
                                )
                            )

            # Emit completion
            logger.info(
                "InstallerAgent.run() - graph completed after %d iterations", iteration
            )
            if stream_writer:
                stream_writer(
                    (
                        InstallerProgress.NAME,
                        InstallerProgress(
                            phase="completed",
                            message="Installation process completed.",
                        ),
                    )
                )

        except Exception as e:
            logger.exception("Installer agent error")
            if stream_writer:
                stream_writer(
                    (
                        InstallerProgress.NAME,
                        InstallerProgress(
                            phase="failed",
                            message=f"Installation failed: {e}",
                        ),
                    )
                )
                stream_writer(
                    (
                        InstallerResult.NAME,
                        InstallerResult(
                            success=False,
                            error_message=str(e),
                        ),
                    )
                )
            raise
