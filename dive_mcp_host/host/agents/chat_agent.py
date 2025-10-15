"""This module contains the ChatAgentFactory for the host.

It uses langgraph.prebuilt.create_react_agent to create the agent.
"""

import asyncio
import contextlib
from asyncio import Event
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver, V
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.utils.runnable import RunnableCallable  # type: ignore
from pydantic import BaseModel

from dive_mcp_host.host.agents.agent_factory import (
    AgentFactory,
    ConfigurableKey,
    initial_messages,
)
from dive_mcp_host.host.agents.file_in_additional_kwargs import FileMsgConverter
from dive_mcp_host.host.agents.message_order import tool_call_order
from dive_mcp_host.host.agents.tools_in_prompt import (
    convert_messages,
    extract_tool_calls,
)
from dive_mcp_host.host.helpers import today_datetime
from dive_mcp_host.host.prompt import PromptType, tools_prompt
from dive_mcp_host.host.store.base import StoreManagerProtocol

type StructuredResponse = dict | BaseModel
type StructuredResponseSchema = dict | type[BaseModel]


class AgentState(MessagesState):
    """The state of the agent."""

    is_last_step: IsLastStep
    today_datetime: str
    remaining_steps: RemainingSteps
    structured_response: StructuredResponse


MINIMUM_STEPS_TOOL_CALL_REQUIRED = 2

PROMPT_RUNNABLE_NAME = "Prompt"


# from langgraph.prebuilt
def get_prompt_runnable(prompt: PromptType | ChatPromptTemplate | None) -> Runnable:
    """Get the prompt runnable."""
    prompt_runnable: Runnable
    if prompt is None:
        prompt_runnable = RunnableCallable(
            lambda state: state.get("messages", None), name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, str):
        _system_message: BaseMessage = SystemMessage(content=prompt)

        def _func(state: AgentState | ChatPromptValue) -> list[BaseMessage]:
            if isinstance(state, ChatPromptValue):
                return [_system_message, *state.to_messages()]
            return [_system_message, *state.get("messages", None)]

        prompt_runnable = RunnableCallable(_func, name=PROMPT_RUNNABLE_NAME)
    elif isinstance(prompt, SystemMessage):
        prompt_runnable = RunnableCallable(
            lambda state: [prompt, *state.get("messages", None)],
            name=PROMPT_RUNNABLE_NAME,
        )
    elif callable(prompt):
        prompt_runnable = RunnableCallable(
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, Runnable):
        prompt_runnable = prompt
    else:
        raise ValueError(f"Got unexpected type for `prompt`: {type(prompt)}")

    return prompt_runnable


# from langgraph.prebuilt.chat_agent_executor
def complete_tool_calls(
    messages: Sequence[BaseMessage],
) -> Sequence[BaseMessage]:
    """Insert ToolMessage if not exists."""
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
                idx + 1, ToolMessage(content="canceled", tool_call_id=tool_call["id"])
            )
    return messages


class HackedToolNode(ToolNode):
    """hacked tool node to inject tool_call_id into the config.

    This is a hack. If langgraph support tool_call_id, we will remove this class.
    """

    async def _arun_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage:
        if "metadata" in config:
            config["metadata"]["tool_call_id"] = call["id"]
        else:
            config["metadata"] = {
                "tool_call_id": call["id"],
            }
        return await super()._arun_one(call, input_type, config)

    def _run_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage:
        if "metadata" in config:
            config["metadata"]["tool_call_id"] = call["id"]
        else:
            config["metadata"] = {
                "tool_call_id": call["id"],
            }
        return super()._run_one(call, input_type, config)


class InterruptableModel(BaseChatModel):
    """A model wrapper that supports abort signal."""

    model: BaseChatModel
    abort_signal: Event | None = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response."""
        return self.model._generate(messages, stop, run_manager, **kwargs)  # noqa: SLF001

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response with abort support (sync version)."""
        cid = str(uuid4())
        for chunk in self.model._stream(messages, stop, run_manager, **kwargs):  # noqa: SLF001
            if self.abort_signal and self.abort_signal.is_set():
                # Yield abort marker when aborted
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="<user_aborted>",
                        id=cid,
                    )
                )
                break
            cid = chunk.message.id
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream response with abort support (async version).

        This implementation can immediately respond to abort signals,
        even if the underlying model is slow to produce the next chunk.
        """
        # Create an async iterator from the model's stream
        stream = self.model._astream(messages, stop, run_manager, **kwargs)  # noqa: SLF001
        if not self.abort_signal:
            for chunk in stream:
                yield chunk
            return

        cid = str(uuid4())  # chunk id, before the stream starts
        try:
            while True:
                # Create a task for getting the next chunk
                chunk_task = asyncio.create_task(stream.__anext__())
                # Create a task for waiting on abort signal
                abort_task = asyncio.create_task(self.abort_signal.wait())

                # Wait for either the next chunk or abort signal
                done, pending = await asyncio.wait(
                    {chunk_task, abort_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                # Check if abort was triggered
                if abort_task in done:
                    # Abort signal was set, yield abort marker and stop
                    abort_chunk = ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="<user_aborted>",
                            id=cid,
                        )
                    )
                    yield abort_chunk
                    # Cancel the chunk task if it's still running
                    if chunk_task in pending or not chunk_task.done():
                        chunk_task.cancel()
                    break

                # Otherwise, we got a chunk
                if chunk_task in done:
                    chunk = await chunk_task
                    cid = chunk.message.id
                    yield chunk

        except StopAsyncIteration:
            # Stream ended naturally
            pass
        finally:
            # Always close the stream to release resources
            # Close stream if it has aclose method (async generators do)
            if hasattr(stream, "aclose"):
                await stream.aclose()

    @property
    def _llm_type(self) -> str:
        """Return the LLM type."""
        return self.model._llm_type  # noqa: SLF001

    def bind_tools(self, *args: Any, **kwargs: Any) -> Runnable:
        """Bind tools to the underlying model."""
        runnable = self.model.bind_tools(*args, **kwargs)
        return self.bind(tools=runnable.kwargs["tools"], **kwargs)


class ChatAgentFactory(AgentFactory[AgentState]):
    """A factory for ChatAgents."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | ToolNode,
        tools_in_prompt: bool = False,
        store: StoreManagerProtocol | None = None,
    ) -> None:
        """Initialize the chat agent factory."""
        self._model = model
        self._model_class = type(model).__name__
        self._tools = tools
        self._tools_in_prompt = tools_in_prompt
        self._response_format: (
            StructuredResponseSchema | tuple[str, StructuredResponseSchema] | None
        ) = None

        self._file_msg_converter = (
            FileMsgConverter(model_provider=self._model_class, store=store).runnable
            if store
            else RunnablePassthrough()
        )

        # changed when self._build_graph is called
        self._tool_classes: list[BaseTool] = []
        self._should_return_direct: set[str] = set()
        self._graph: StateGraph | None = None

        # changed when self.create_agent is called
        self._prompt: Runnable = get_prompt_runnable(None)
        self._tool_prompt: Runnable = get_prompt_runnable(None)

        # Initialize the tool prompt
        if self._tools_in_prompt:
            if isinstance(self._tools, ToolNode):
                tools = list(self._tools.tools_by_name.values())
                self._tool_prompt = get_prompt_runnable(tools_prompt(tools))
            else:
                self._tool_prompt = get_prompt_runnable(tools_prompt(self._tools))

        self._build_graph()

    def _check_more_steps_needed(
        self, state: AgentState, response: BaseMessage
    ) -> bool:
        has_tool_calls = (
            isinstance(response, AIMessage) and response.tool_calls is not None
        )
        all_tools_return_direct = (
            all(
                call["name"] in self._should_return_direct
                for call in response.tool_calls
            )
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = state.get("remaining_steps", None)
        is_last_step = state.get("is_last_step", False)

        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (
                remaining_steps is not None
                and remaining_steps < 1
                and all_tools_return_direct
            )
            or (
                remaining_steps is not None
                and remaining_steps < MINIMUM_STEPS_TOOL_CALL_REQUIRED
                and has_tool_calls
            )
        )

    async def _call_model(
        self, state: AgentState, config: RunnableConfig
    ) -> AgentState:
        state["messages"] = complete_tool_calls(state["messages"])
        # Get abort signal from config
        abort_signal: Event | None = config.get("configurable", {}).get(
            ConfigurableKey.ABORT_SIGNAL
        )
        if abort_signal and abort_signal.is_set():
            return cast(AgentState, {"messages": []})
        if not self._tools_in_prompt:
            # Wrap the model with abort functionality first
            wrapped_model = InterruptableModel(
                model=self._model,
                abort_signal=abort_signal,
                disable_streaming=self._model.disable_streaming,
            )
            # Then bind tools if needed - this will use the default bind_tools
            # which creates a RunnableBind, but since we're wrapping the base model,
            # the _stream method will still be called with abort support
            model = wrapped_model
            if self._tool_classes:
                model = wrapped_model.bind_tools(self._tool_classes)
            model_runnable = (
                self._prompt | self._file_msg_converter | drop_empty_messages | model
            )
        else:
            model_runnable = (
                self._prompt
                | self._tool_prompt
                | convert_messages
                | self._file_msg_converter
                | drop_empty_messages
                | InterruptableModel(
                    model=self._model,
                    abort_signal=abort_signal,
                    disable_streaming=self._model.disable_streaming,
                )
            )

        response = await model_runnable.ainvoke(state, config)
        if isinstance(response, AIMessage):
            response = extract_tool_calls(response)
        if self._check_more_steps_needed(state, response):
            response = AIMessage(
                id=response.id,
                content="Sorry, need more steps to process this request.",
            )
        responses = [response]
        if abort_signal and abort_signal.is_set():
            responses = complete_tool_calls(responses)
        return cast(AgentState, {"messages": responses})

    def _generate_structured_response(
        self, state: AgentState, config: RunnableConfig
    ) -> AgentState:
        messages = state["messages"][:-1]
        if isinstance(self._response_format, tuple):
            system_prompt, structured_response_schema = self._response_format
            messages = [SystemMessage(content=system_prompt), *list(messages)]

        model_with_structured_output = self._model.with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )

        response = model_with_structured_output.invoke(messages, config)
        return cast(AgentState, {"structured_response": response})

    def _before_agent(self, state: AgentState, config: RunnableConfig) -> AgentState:
        configurable = config.get("configurable", {})
        max_input_tokens: int | None = configurable.get("max_input_tokens")
        oversize_policy: Literal["window"] | None = configurable.get("oversize_policy")

        new_messages: list[BaseMessage] = []
        new_messages.extend(tool_call_order(state["messages"]))

        if max_input_tokens is None or oversize_policy is None:
            return cast(AgentState, {"messages": new_messages})

        if oversize_policy == "window":
            messages: list[BaseMessage] = trim_messages(
                state["messages"],
                max_tokens=max_input_tokens,
                token_counter=count_tokens_approximately,
            )
            remove_messages = [
                RemoveMessage(id=m.id)  # type: ignore
                for m in state["messages"]
                if m not in messages
            ]
            new_messages.extend(remove_messages)
            return cast(AgentState, {"messages": new_messages})

        return cast(AgentState, {"messages": []})

    def _after_agent(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return (
                END if self._response_format is None else "generate_structured_response"
            )
        return "tools"

    def _after_tools(self, state: AgentState) -> str:
        for m in reversed(state["messages"]):
            if not isinstance(m, ToolMessage):
                break
            if m.name in self._should_return_direct:
                return END
        return "before_agent"

    def _build_graph(self) -> None:
        graph = StateGraph(AgentState)

        graph.add_node("before_agent", self._before_agent)
        graph.set_entry_point("before_agent")

        # create agent node
        graph.add_node("agent", self._call_model)
        graph.add_edge("before_agent", "agent")

        tool_node = (
            self._tools
            if isinstance(self._tools, ToolNode)
            else HackedToolNode(self._tools)
        )
        self._tool_classes = list(tool_node.tools_by_name.values())
        graph.add_node("tools", tool_node)
        self._should_return_direct = {
            t.name for t in self._tool_classes if t.return_direct
        }

        if self._response_format:
            graph.add_node(
                "generate_structured_response", self._generate_structured_response
            )
            graph.add_edge("generate_structured_response", END)
            next_node = ["tools", "generate_structured_response"]
        else:
            next_node = ["tools", END]

        graph.add_conditional_edges(
            "agent",
            self._after_agent,
            next_node,
        )

        # one of the tools should return direct
        if self._should_return_direct:
            graph.add_conditional_edges("tools", self._after_tools)
        else:
            graph.add_edge("tools", "before_agent")

        self._graph = graph

    def create_agent(
        self,
        *,
        prompt: PromptType | ChatPromptTemplate,
        checkpointer: BaseCheckpointSaver[V] | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
    ) -> CompiledStateGraph:
        """Create a react agent."""
        self._prompt = get_prompt_runnable(prompt)
        if self._graph is None:
            raise ValueError("Graph is not built")
        return self._graph.compile(checkpointer=checkpointer, store=store, debug=debug)

    def create_initial_state(
        self,
        *,
        query: str | HumanMessage | list[BaseMessage],
    ) -> AgentState:
        """Create an initial state for the query."""
        return AgentState(
            messages=initial_messages(query),
            is_last_step=False,
            today_datetime=today_datetime(),
            remaining_steps=100,
        )  # type: ignore

    def state_type(
        self,
    ) -> type[AgentState]:
        """Get the state type."""
        return AgentState


def get_chat_agent_factory(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | ToolNode,
    tools_in_prompt: bool = False,
    store: StoreManagerProtocol | None = None,
) -> ChatAgentFactory:
    """Get an agent factory."""
    return ChatAgentFactory(
        model=model,
        tools=tools,
        tools_in_prompt=tools_in_prompt,
        store=store,
    )


@RunnableCallable
def drop_empty_messages(inpt: ChatPromptValue | list[BaseMessage]) -> list[BaseMessage]:
    """Drop empty messages."""
    messages = inpt.to_messages() if isinstance(inpt, ChatPromptValue) else inpt

    result = []
    for message in messages:
        # AIMessage have more constraints
        if isinstance(message, AIMessage):
            if (
                not message.content
                and not message.tool_calls
                and not message.invalid_tool_calls
            ):
                continue
        # ToolMessage, SystemMessage, HumanMessage needs to have content
        elif not message.content:
            continue
        result.append(message)
    return result
