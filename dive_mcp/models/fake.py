import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


class FakeMessageToolModel(BaseChatModel):
    """A fake tool model.

    Use this model to test the tool calling.

    Example:
        responses = [
            AIMessage(
                content="I am a fake model.",
                tool_calls=[ToolCall(name="fake_tool", args={"arg": "arg"}, id="id")],
            ),
            AIMessage(
                content="final AI message",
            ),
        ]
        model = FakeMessageToolModel(responses=responses)
    """

    responses: list[AIMessage]
    sleep: float | None = None
    i: int = 0

    def _generate(
        self,
        _messages: list[BaseMessage],
        _stop: list[str] | None = None,
        _run_manager: CallbackManagerForLLMRun | None = None,
        **_kwargs: Any,
    ) -> ChatResult:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    # TODO: Implement this
    # def _call(
    #     self,
    #     _messages: list[BaseMessage],
    #     _stop: list[str] | None = None,
    #     _run_manager: CallbackManagerForLLMRun | None = None,
    #     **_kwargs: Any,
    # ) -> str: ...

    def _stream(
        self,
        _messages: list[BaseMessage],
        _stop: list[str] | None = None,
        _run_manager: CallbackManagerForLLMRun | None = None,
        **_kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for c in response:
            if self.sleep is not None:
                time.sleep(self.sleep)
            yield ChatGenerationChunk(message=cast(AIMessageChunk, c))

    async def _astream(
        self,
        _messages: list[BaseMessage],
        _stop: list[str] | None = None,
        _run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for c in response:
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)
            yield ChatGenerationChunk(message=cast(AIMessageChunk, c))

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model."""
        formatted_tools = [convert_to_openai_tool(tool, strict=False) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-model"


def load_model(responses: list[AIMessage]) -> FakeMessageToolModel:
    """Load the fake model."""
    return FakeMessageToolModel(responses=responses)
