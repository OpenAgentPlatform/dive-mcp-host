import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, suppress
from dataclasses import asdict, dataclass, field
from hashlib import md5
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self
from urllib.parse import urlparse
from uuid import uuid4

from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from openai import APIError as OpenAIAPIError
from pydantic import BaseModel
from starlette.datastructures import State

from dive_mcp_host.host.agents.file_in_additional_kwargs import (
    DOCUMENTS_KEY,
    IMAGES_KEY,
    OAP_MIN_COUNT,
)
from dive_mcp_host.host.agents.message_order import FAKE_TOOL_RESPONSE
from dive_mcp_host.host.custom_events import (
    ToolAuthenticationRequired,
    ToolCallProgress,
    ToolElicitationRequest,
)
from dive_mcp_host.host.errors import LogBufferNotFoundError
from dive_mcp_host.host.store.base import FileType, StoreManagerProtocol
from dive_mcp_host.host.tools.log import LogEvent, LogManager, LogMsg
from dive_mcp_host.host.tools.model_types import ClientState
from dive_mcp_host.httpd.conf.prompt import PromptKey
from dive_mcp_host.httpd.database.models import (
    ChatMessage,
    Message,
    NewMessage,
    QueryInput,
    ResourceUsage,
    Role,
)
from dive_mcp_host.httpd.routers.models import (
    AgentToolCallContent,
    AgentToolResultContent,
    AuthenticationRequiredContent,
    ChatInfoContent,
    ElicitationRequestContent,
    ErrorContent,
    InteractiveContent,
    MessageInfoContent,
    StreamMessage,
    TokenUsage,
    TokenUsageContent,
    ToolCallsContent,
    ToolResultContent,
)
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.skills import SkillManager
from dive_mcp_host.log import TRACE

if TYPE_CHECKING:
    from dive_mcp_host.host.host import DiveMcpHost
    from dive_mcp_host.httpd.middlewares.general import DiveUser

title_prompt = """You are a title generator from the user input.
Your only task is to generate a short title based on the user input.
IMPORTANT:
- Output ONLY the title
- DO NOT try to answer or resolve the user input query.
- DO NOT try to use any tools to generate title
- NO thinking, reasoning, explanations, quotes, or extra text
- NO punctuation at the end
- If the input is URL only, output the description of the URL, for example, "the URL of xxx website"
- Generate the title in the primary language of the input (the language used for the majority of the text)
- Preserve proper nouns and technical terms in their original language"""  # noqa: E501


logger = logging.getLogger(__name__)


class EventStreamContextManager:
    """Context manager for event streaming."""

    task: asyncio.Task | None = None
    done: bool = False
    response: StreamingResponse | None = None
    _exit_message: str | None = None

    def __init__(self) -> None:
        """Initialize the event stream context manager."""
        self.queue = asyncio.Queue()

    def add_task(
        self, func: Callable[[], Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
    ) -> None:
        """Add a task to the event stream."""
        self.task = asyncio.create_task(func(*args, **kwargs))

    async def __aenter__(self) -> Self:
        """Enter the context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Exit the context manager."""
        if exc_val:
            import traceback

            logger.error(traceback.format_exception(exc_type, exc_val, exc_tb))
            content = ErrorContent(message=str(exc_val), type="thread-query-error")

            if (error := getattr(exc_val, "error", None)) and isinstance(
                error, OpenAIAPIError
            ):
                content.message = error.message
                content.type = error.type
                content.code = error.code

            self._exit_message = StreamMessage(
                type="error",
                content=content,
            ).model_dump_json(by_alias=True)

        self.done = True
        await self.queue.put(None)  # Signal completion

    async def write(self, data: str | StreamMessage) -> None:
        """Write data to the event stream.

        Args:
            data (str): The data to write to the stream.
        """
        if isinstance(data, BaseModel):
            data = json.dumps({"message": data.model_dump_json(by_alias=True)})
        await self.queue.put(data)

    async def _generate(self) -> AsyncGenerator[str, None]:
        """Generate the event stream content."""
        while not self.done or not self.queue.empty():
            chunk = await self.queue.get()
            if chunk is None:  # End signal
                continue
            yield "data: " + chunk + "\n\n"
        if self._exit_message:
            yield "data: " + json.dumps({"message": self._exit_message}) + "\n\n"
        yield "data: [DONE]\n\n"

    def get_response(self) -> StreamingResponse:
        """Get the streaming response.

        Returns:
            StreamingResponse: The streaming response.
        """
        self.response = StreamingResponse(
            content=self._generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        return self.response


class ChatError(Exception):
    """Chat error."""

    def __init__(self, message: str) -> None:
        """Initialize chat error."""
        self.message = message


@dataclass(slots=True)
class TextContent:
    """Structure for text content."""

    text: str
    type: Literal["text"] = "text"

    @classmethod
    def create(cls, text: str) -> dict[str, str]:
        """Create text content dict."""
        return asdict(cls(text=text))


def _extract_text_from_message(message: HumanMessage) -> str:
    """Extract plain text content from a HumanMessage.

    This extracts the actual text content, ignoring the dict wrapper
    that would otherwise inflate token counts.
    """
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        texts = []
        for item in message.content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "".join(texts)
    return ""


def count_user_message_tokens(message: HumanMessage | None) -> int:
    """Count tokens for a user message based on actual text content.

    This avoids inflated token counts caused by the dict wrapper format
    used internally for message content.
    """
    if not message:
        return 0
    text = _extract_text_from_message(message)
    # Use a simple HumanMessage with plain text for accurate counting
    plain_message = HumanMessage(content=text)
    return count_tokens_approximately([plain_message])


def count_prompt_tokens(prompt: str | None) -> int:
    """Count tokens for a prompt string using langchain's estimation.

    Args:
        prompt: The prompt string to count tokens for.

    Returns:
        The estimated token count.
    """
    if not prompt:
        return 0
    message = SystemMessage(content=prompt)
    return count_tokens_approximately([message])


@dataclass(slots=True)
class ImageAndDocuments:
    """Structure that contains image and documents."""

    images: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)


class ContentHandler:
    """Some models will return more then just pure text in content response.

    We need to have a customized handler for those special models.
    """

    def __init__(
        self,
        store: StoreManagerProtocol,
    ) -> None:
        """Initialize ContentHandler."""
        self._store = store
        self._str_output_parser = StrOutputParser()
        # Cache that contains the md5 hash and file path / urls for the file.
        # Prevents dupicate save / uploads.
        self._cache: dict[str, list[str]] = {}

    async def invoke(self, msg: AIMessage) -> str:
        """Extract various types of content."""
        result = self._text_content(msg)
        if image_content := await self._gemini_image(msg):
            result = f"{result} {image_content}"
        return result

    def _text_content(self, msg: AIMessage) -> str:
        return self._str_output_parser.invoke(msg)

    async def _save_with_cache(self, data: str) -> list[str]:
        """Prevents duplicate save and uploads.

        Returns:
            Saved locations, 'local file path' or 'url'
        """
        md5_hash = md5(data.encode(), usedforsecurity=False).hexdigest()
        locations = self._cache.get(md5_hash)
        if not locations:
            locations = await self._store.save_base64_image(data)
            self._cache[md5_hash] = locations
        return locations

    def _retrive_optimal_location(self, locations: list[str]) -> str:
        """Prioritize urls, prevents broken image in case we need to sync
        user chat history some day.
        """  # noqa: D205
        url = locations[0]
        for item in locations[1:]:
            if self._store.is_url(item):
                url = item
        if self._store.is_local_file(url):
            url = f"file://{url}"
        return url

    async def _gemini_image(self, msg: AIMessage) -> str:
        """Gemini will return base64 image content.

        {
            "content": [
                "Here is a cuddly cat wearing a hat! ",
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,XXXXXXXX"
                    }
                }
            ]
        }

        """
        result = ""
        for content in msg.content:
            if (
                isinstance(content, dict)
                and (image_url := content.get("image_url"))
                and (inline_base64 := image_url.get("url"))
            ):
                base64_data: str = inline_base64.split(",")[-1]
                assert isinstance(base64_data, str), "base64_data must be string"
                locations = await self._save_with_cache(base64_data)
                url = self._retrive_optimal_location(locations)
                image_tag = f"![image]({url})"
                result = f"{result} {image_tag}"

        return result


class ChatProcessor:
    """Chat processor."""

    def __init__(
        self,
        app: DiveHostAPI,
        request_state: State,
        stream: EventStreamContextManager,
        skill_manager: SkillManager,
    ) -> None:
        """Initialize chat processor."""
        self.app = app
        self.request_state = request_state
        self.stream = stream
        self.store: StoreManagerProtocol = app.store
        self.dive_host: DiveMcpHost = app.dive_host["default"]
        self._str_output_parser = StrOutputParser()
        self._content_handler = ContentHandler(self.store)
        self.disable_dive_system_prompt = (
            app.model_config_manager.full_config.disable_dive_system_prompt
            if app.model_config_manager.full_config
            else False
        )
        self.enable_local_tools = (
            app.model_config_manager.full_config.enable_local_tools
            if app.model_config_manager.full_config
            else False
        )
        self.skill_manager = skill_manager

    async def handle_chat(
        self,
        chat_id: str | None,
        query_input: QueryInput | None,
        regenerate_message_id: str | None,
    ) -> tuple[str, TokenUsage]:
        """Handle chat."""
        logger.debug(
            "Handle chat, chat_id: %s, query_input: %s, regenerate_message_id: %s",
            chat_id,
            query_input,
            regenerate_message_id,
        )

        chat_id = chat_id if chat_id else str(uuid4())
        dive_user: DiveUser = self.request_state.dive_user
        title = "New Chat"
        title_await = None
        result = ""

        if regenerate_message_id:
            if query_input:
                query_message = await self._query_input_to_message(
                    query_input, message_id=regenerate_message_id
                )
            else:
                query_message = await self._get_history_user_input(
                    chat_id, regenerate_message_id
                )
        elif query_input:
            query_message = await self._query_input_to_message(
                query_input, message_id=str(uuid4())
            )
        else:
            query_message = None

        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            original_msg_exist = False
            if not await db.check_chat_exists(chat_id, dive_user["user_id"]):
                await db.create_chat(
                    chat_id, title, dive_user["user_id"], dive_user["user_type"]
                )
                if query_input:
                    title_await = asyncio.create_task(
                        self._generate_title(query_input.text)
                    )
            elif regenerate_message_id:
                await db.delete_messages_after(chat_id, query_message.id)
                original_msg_exist = await db.lock_msg(
                    chat_id=chat_id,
                    message_id=query_message.id,
                )
                if query_input and original_msg_exist:
                    await db.update_message_content(
                        query_message.id,  # type: ignore
                        QueryInput(
                            text=query_input.text or "",
                            images=query_input.images or [],
                            documents=query_input.documents or [],
                            tool_calls=query_input.tool_calls,
                        ),
                    )

            if query_input and not original_msg_exist:
                await db.create_message(
                    NewMessage(
                        chatId=chat_id,
                        role=Role.USER,
                        messageId=query_message.id,
                        content=query_input.text or "",  # type: ignore
                        files=(
                            (query_input.images or []) + (query_input.documents or [])
                        ),
                    ),
                )
            await session.commit()

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            )
        )

        start = time.time()
        try:
            user_message, ai_message, current_messages, ttft = await self._process_chat(
                chat_id,
                query_message,
                is_resend=regenerate_message_id is not None,
                start_time=start,
            )
        except Exception as exc:
            # Generate assistant message ID upfront for error recovery
            placeholder_ai_message_id = str(uuid4())
            # Send message_info even on error so client can retry
            user_msg_id = (
                query_message.id
                if query_message and hasattr(query_message, "id")
                else None
            )
            if user_msg_id:
                # Create placeholder assistant message for retry support
                async with self.app.db_sessionmaker() as session:
                    db = self.app.msg_store(session)
                    await db.create_message(
                        NewMessage(
                            chatId=chat_id,
                            role=Role.ASSISTANT,
                            messageId=placeholder_ai_message_id,
                            content=f"<chat-error>{exc}</chat-error>",
                        ),
                    )
                    await session.commit()
                await self.stream.write(
                    StreamMessage(
                        type="message_info",
                        content=MessageInfoContent(
                            userMessageId=user_msg_id,
                            assistantMessageId=placeholder_ai_message_id,
                        ),
                    )
                )
            raise
        end = time.time()
        if ai_message is None:
            if title_await:
                title_await.cancel()
            return "", TokenUsage()
        assert user_message.id
        assert ai_message.id

        # Calculate user message tokens early, before the loop
        user_tokens = count_user_message_tokens(user_message)

        # Calculate prompt tokens using langchain estimation
        custom_prompt = self.app.prompt_config_manager.get_prompt(PromptKey.CUSTOM)
        system_prompt = self.app.prompt_config_manager.get_prompt(PromptKey.SYSTEM)
        custom_prompt_tokens = count_prompt_tokens(custom_prompt)
        system_prompt_tokens = count_prompt_tokens(system_prompt)

        if title_await:
            title = await title_await

        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)

            if title_await:
                await db.patch_chat(chat_id, user_id=dive_user["user_id"], title=title)

            for message in current_messages:
                assert message.id
                if isinstance(message, HumanMessage):
                    # User message no longer stores user_token
                    # user_token is now stored in AIMessage
                    pass
                elif isinstance(message, AIMessage):
                    # Get duration from metadata or calculate from timing
                    duration = None
                    if message.usage_metadata:
                        duration = message.usage_metadata.get("total_duration")

                    # If no valid duration from metadata, calculate it
                    if not duration or duration <= 0:
                        duration = end - start if message.id == ai_message.id else 0

                    # Calculate tokens per second
                    output_tokens = (
                        message.usage_metadata["output_tokens"]
                        if message.usage_metadata
                        else 0
                    )
                    tokens_per_second = (
                        output_tokens / duration if duration > 0 else 0.0
                    )

                    # Debug logging
                    logger.debug(
                        "AI message tokens_per_second calculation: "
                        "output_tokens=%s, duration=%s, tps=%s, "
                        "is_final=%s, usage_metadata=%s",
                        output_tokens,
                        duration,
                        tokens_per_second,
                        message.id == ai_message.id,
                        message.usage_metadata,
                    )

                    # Use TTFT only for the final AI message
                    message_ttft = ttft if message.id == ai_message.id else 0.0

                    resource_usage = ResourceUsage(
                        model=message.response_metadata.get("model")
                        or message.response_metadata.get("model_name")
                        or "",
                        total_input_tokens=message.usage_metadata["input_tokens"]
                        if message.usage_metadata
                        else 0,
                        total_output_tokens=output_tokens,
                        user_token=user_tokens,
                        custom_prompt_token=custom_prompt_tokens,
                        system_prompt_token=system_prompt_tokens,
                        time_to_first_token=message_ttft,
                        tokens_per_second=tokens_per_second,
                        total_run_time=duration,
                    )
                    result = (
                        await self._content_handler.invoke(message)
                        if message.content
                        else ""
                    )
                    await db.create_message(
                        NewMessage(
                            chatId=chat_id,
                            role=Role.ASSISTANT,
                            messageId=message.id,
                            content=result,
                            toolCalls=message.tool_calls,
                            resource_usage=resource_usage,
                        ),
                    )
                elif isinstance(message, ToolMessage):
                    if isinstance(message.content, list):
                        content = json.dumps(message.content)
                    elif isinstance(message.content, str):
                        content = message.content
                    else:
                        raise ValueError(
                            f"got unknown type: {type(message.content)}, "
                            f"data: {message.content}"
                        )
                    await db.create_message(
                        NewMessage(
                            chatId=chat_id,
                            role=Role.TOOL_RESULT,
                            messageId=message.id,
                            content=content,
                        ),
                    )

            await session.commit()

        logger.log(TRACE, "usermessage.id: %s", user_message.id)
        await self.stream.write(
            StreamMessage(
                type="message_info",
                content=MessageInfoContent(
                    userMessageId=user_message.id,
                    assistantMessageId=ai_message.id,
                ),
            )
        )

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            )
        )

        # Calculate tokens per second for the final response
        total_duration = end - start
        output_tokens_count = (
            ai_message.usage_metadata["output_tokens"]
            if ai_message.usage_metadata
            else 0
        )
        tps = output_tokens_count / total_duration if total_duration > 0 else 0.0

        # Debug logging
        logger.debug(
            "Final token usage calculation: output_tokens=%s, total_duration=%s, tps=%s, "  # noqa: E501
            "usage_metadata=%s",
            output_tokens_count,
            total_duration,
            tps,
            ai_message.usage_metadata,
        )

        token_usage = TokenUsage(
            totalInputTokens=ai_message.usage_metadata["input_tokens"]
            if ai_message.usage_metadata
            else 0,
            totalOutputTokens=output_tokens_count,
            userToken=user_tokens,
            customPromptToken=custom_prompt_tokens,
            systemPromptToken=system_prompt_tokens,
            totalTokens=ai_message.usage_metadata["total_tokens"]
            if ai_message.usage_metadata
            else 0,
        )

        # Send token usage message before stream completion
        await self.stream.write(
            StreamMessage(
                type="token_usage",
                content=TokenUsageContent(
                    inputTokens=ai_message.usage_metadata["input_tokens"]
                    if ai_message.usage_metadata
                    else 0,
                    outputTokens=output_tokens_count,
                    userToken=user_tokens,
                    customPromptToken=custom_prompt_tokens,
                    systemPromptToken=system_prompt_tokens,
                    timeToFirstToken=ttft,
                    tokensPerSecond=tps,
                    modelName=ai_message.response_metadata.get("model")
                    or ai_message.response_metadata.get("model_name")
                    or "",
                ),
            )
        )

        return result, token_usage

    async def handle_chat_with_history(
        self,
        chat_id: str,
        query_input: BaseMessage | None,
        history: list[BaseMessage],
        tools: list | None = None,
    ) -> tuple[str, TokenUsage]:
        """Handle chat with history.

        Args:
            chat_id (str): The chat ID.
            query_input (BaseMessage | None): The query input.
            history (list[BaseMessage]): The history.
            tools (list | None): The tools.

        Returns:
            tuple[str, TokenUsage]: The result and token usage.
        """
        user_message, ai_message, _, _ = await self._process_chat(
            chat_id, query_input, history, tools
        )

        # Calculate user message tokens
        user_tokens = count_user_message_tokens(user_message)

        # Calculate prompt tokens using langchain estimation
        custom_prompt = self.app.prompt_config_manager.get_prompt(PromptKey.CUSTOM)
        system_prompt = self.app.prompt_config_manager.get_prompt(PromptKey.SYSTEM)
        custom_prompt_tokens = count_prompt_tokens(custom_prompt)
        system_prompt_tokens = count_prompt_tokens(system_prompt)

        usage = TokenUsage()
        if ai_message.usage_metadata:
            usage.total_input_tokens = ai_message.usage_metadata["input_tokens"]
            usage.total_output_tokens = ai_message.usage_metadata["output_tokens"]
            usage.user_token = user_tokens
            usage.custom_prompt_token = custom_prompt_tokens
            usage.system_prompt_token = system_prompt_tokens
            usage.total_tokens = ai_message.usage_metadata["total_tokens"]

        return str(ai_message.content), usage

    async def _process_chat(
        self,
        chat_id: str | None,
        query_input: str | QueryInput | BaseMessage | None,
        history: list[BaseMessage] | None = None,
        tools: list | None = None,
        is_resend: bool = False,
        start_time: float | None = None,
    ) -> tuple[HumanMessage, AIMessage, list[BaseMessage], float]:
        messages = [*history] if history else []

        # if retry input is empty
        if query_input:
            if isinstance(query_input, str):
                messages.append(HumanMessage(content=query_input))
            elif isinstance(query_input, QueryInput):
                messages.append(await self._query_input_to_message(query_input))
            else:
                messages.append(query_input)

        dive_user: DiveUser = self.request_state.dive_user

        def _prompt_cb(_: Any) -> list[BaseMessage]:
            return messages

        prompt: str | Callable[..., list[BaseMessage]] | None = None
        if any(isinstance(m, SystemMessage) for m in messages):
            prompt = _prompt_cb
        elif self.disable_dive_system_prompt and (
            custom_prompt := self.app.prompt_config_manager.get_prompt(PromptKey.CUSTOM)
        ):
            prompt = custom_prompt
        elif system_prompt := self.app.prompt_config_manager.get_prompt(
            PromptKey.SYSTEM
        ):
            prompt = system_prompt

        chat = self.dive_host.chat(
            chat_id=chat_id,
            user_id=dive_user.get("user_id") or "default",
            tools=tools,
            extend_tools=self.skill_manager.get_tools(),
            system_prompt=prompt,
            disable_default_system_prompt=self.disable_dive_system_prompt,
            include_local_tools=self.enable_local_tools,
            skill_manager=self.skill_manager,
        )
        async with AsyncExitStack() as stack:
            if chat_id:
                await stack.enter_async_context(
                    self.app.abort_controller.abort_signal(chat_id, chat.abort)
                )
            await stack.enter_async_context(chat)
            response_generator = chat.query(
                messages,
                stream_mode=["messages", "values", "updates", "custom"],
                is_resend=is_resend,
            )
            # Use provided start_time or current time
            query_start_time = start_time if start_time is not None else time.time()
            return await self._handle_response(response_generator, query_start_time)

        raise RuntimeError("Unreachable")

    async def _stream_text_msg(self, message: AIMessage) -> None:
        content = await self._content_handler.invoke(message)
        if content:
            await self.stream.write(StreamMessage(type="text", content=content))
        if message.response_metadata.get("stop_reason") == "max_tokens":
            await self.stream.write(
                StreamMessage(
                    type="error",
                    content=ErrorContent(
                        message="stop_reason: max_tokens", type="max_tokens"
                    ),
                )
            )

    # Tools that belong to installer agent (sub-agent)
    # These are streamed as agent_tool_call/agent_tool_result, not tool_calls
    _INSTALLER_AGENT_TOOLS = frozenset(
        {
            "fetch",
            "bash",
            "read_file",
            "write_file",
            "add_mcp_server",
            "reload_mcp_server",
            "request_confirmation",
        }
    )

    async def _stream_tool_calls_msg(self, message: AIMessage) -> None:
        tool_calls = list(message.tool_calls)
        if not tool_calls:
            logger.debug("Skipping tool_calls - all are installer agent tools")
            return

        await self.stream.write(
            StreamMessage(
                type="tool_calls",
                content=[
                    ToolCallsContent(name=c["name"], arguments=c["args"])
                    for c in tool_calls
                ],
            )
        )

    async def _stream_tool_result_msg(self, message: ToolMessage) -> None:
        result = message.content
        with suppress(json.JSONDecodeError):
            if isinstance(result, list):
                result = [json.loads(r) if isinstance(r, str) else r for r in result]
            else:
                result = json.loads(result)
        await self.stream.write(
            StreamMessage(
                type="tool_result",
                content=ToolResultContent(name=message.name or "", result=result),
            )
        )

    async def _stream_interactive_msg(self, content: InteractiveContent) -> None:
        await self.stream.write(StreamMessage(type="interactive", content=content))

    async def _handle_response(
        self, response: AsyncIterator[dict[str, Any] | Any], start_time: float
    ) -> tuple[HumanMessage | Any, AIMessage | Any, list[BaseMessage], float]:
        """Handle response.

        Returns:
            tuple[HumanMessage | Any, AIMessage | Any, list[BaseMessage], float]:
            The human message, the AI message, all messages of the current
            query, and time to first token.
        """
        user_message = None
        ai_message = None
        values_messages: list[BaseMessage] = []
        current_messages: list[BaseMessage] = []
        time_to_first_token: float = 0.0
        async for res_type, res_content in response:
            if res_type == "messages":
                message, _ = res_content
                if isinstance(message, AIMessage):
                    logger.log(TRACE, "got AI message: %s", message.model_dump_json())
                    if message.content:
                        # Record time to first token if not already recorded
                        if time_to_first_token == 0.0:
                            time_to_first_token = time.time() - start_time
                        await self._stream_text_msg(message)
                elif isinstance(message, ToolMessage):
                    logger.log(TRACE, "got tool message: %s", message.model_dump_json())
                    if message.response_metadata.get(FAKE_TOOL_RESPONSE, False):
                        logger.log(
                            TRACE,
                            "ignore fake tool response: %s",
                            message.model_dump_json(),
                        )
                        continue
                    await self._stream_tool_result_msg(message)
                else:
                    # idk what is this
                    logger.warning("Unknown message type: %s", message)
            elif res_type == "values" and len(res_content["messages"]) >= 2:  # type: ignore  # noqa: PLR2004
                values_messages = res_content["messages"]  # type: ignore
            elif res_type == "updates":
                # Get tool call message
                if not isinstance(res_content, dict):
                    continue

                for value in res_content.values():
                    if not isinstance(value, dict):
                        continue

                    msgs = value.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            logger.log(
                                TRACE,
                                "got tool call message: %s",
                                msg.model_dump_json(),
                            )
                            await self._stream_tool_calls_msg(msg)
            elif res_type == "custom":
                logger.debug("res_content: %s", res_content)
                if res_content[0] == ToolCallProgress.NAME:
                    await self.stream.write(
                        StreamMessage(
                            type="tool_call_progress",
                            content=res_content[1],
                        )
                    )
                elif res_content[0] == ToolAuthenticationRequired.NAME:
                    content = res_content[1]
                    assert isinstance(content, ToolAuthenticationRequired)
                    await self._stream_interactive_msg(
                        InteractiveContent(
                            type="authentication_required",
                            content=AuthenticationRequiredContent(
                                server_name=content.server_name,
                                auth_url=content.auth_url,
                            ),
                        )
                    )
                elif res_content[0] == ToolElicitationRequest.NAME:
                    content = res_content[1]
                    assert isinstance(content, ToolElicitationRequest)
                    await self._stream_interactive_msg(
                        InteractiveContent(
                            type="elicitation_request",
                            content=ElicitationRequestContent(
                                request_id=content.request_id,
                                message=content.message,
                                requested_schema=content.requested_schema,
                            ),
                        )
                    )
                elif res_content[0] == "agent_tool_call":
                    # Tool call from agent tools (installer tools)
                    content = res_content[1]
                    await self.stream.write(
                        StreamMessage(
                            type="agent_tool_call",
                            content=AgentToolCallContent(
                                tool_call_id=content.tool_call_id,
                                name=content.name,
                                args=content.args,
                            ),
                        )
                    )
                elif res_content[0] == "agent_tool_result":
                    # Tool result from agent tools (installer tools)
                    content = res_content[1]
                    await self.stream.write(
                        StreamMessage(
                            type="agent_tool_result",
                            content=AgentToolResultContent(
                                tool_call_id=content.tool_call_id,
                                name=content.name,
                                result=content.result,
                            ),
                        )
                    )

        # Find the most recent user and AI messages from newest to oldest
        user_message = next(
            (msg for msg in reversed(values_messages) if isinstance(msg, HumanMessage)),
            None,
        )
        ai_message = next(
            (msg for msg in reversed(values_messages) if isinstance(msg, AIMessage)),
            None,
        )
        if user_message:
            current_messages = values_messages[values_messages.index(user_message) :]

        return user_message, ai_message, current_messages, time_to_first_token

    async def _generate_title(self, query: str) -> str:
        """Generate title."""
        chat = self.dive_host.chat(
            tools=[],  # do not use tools
            system_prompt=title_prompt,
            volatile=True,
        )
        try:
            async with chat:
                response = await chat.active_agent.ainvoke(
                    {"messages": [HumanMessage(content=query)]}
                )
                if isinstance(response["messages"][-1], AIMessage):
                    return strip_title(
                        self._str_output_parser.invoke(response["messages"][-1])
                    )
        except Exception as e:
            logger.exception("Error generating title: %s", e)
        return "New Chat"

    def _is_using_oap(self, files: list[str]) -> bool:
        return (
            len(files) >= OAP_MIN_COUNT
            and len(files) % OAP_MIN_COUNT == 0
            and self.store.is_local_file(files[0])
            and self.store.is_url(files[1])
        )

    def _seperate_img_and_doc_oap(self, files: list[str]) -> ImageAndDocuments:
        """OAP file order, [local_path, url, ... etc]."""
        result = ImageAndDocuments()
        for local_path, url in batched(files, 2):
            if FileType.from_file_path(local_path) == FileType.IMAGE:
                result.images.extend([local_path, url])
                continue
            result.documents.extend([local_path, url])
        return result

    def _seperate_img_and_doc(self, files: list[str]) -> ImageAndDocuments:
        """File order, [local_path, local_path, ... etc]."""
        result = ImageAndDocuments()
        for local_path in files:
            if FileType.from_file_path(local_path) == FileType.IMAGE:
                result.images.append(local_path)
                continue
            result.documents.append(local_path)
        return result

    def _extract_image_and_documents(self, files: list[str]) -> ImageAndDocuments:
        if self._is_using_oap(files):
            return self._seperate_img_and_doc_oap(files)
        return self._seperate_img_and_doc(files)

    async def _process_history_message(self, message: Message) -> HumanMessage:
        """Process history message."""
        assert message.role == Role.USER, "Must be user message"
        content = []
        if message_content := message.content.strip():
            content.append(TextContent.create(message_content))

        if not message.files:
            logger.debug("message has no files attatched")
            return HumanMessage(content=message_content, id=message.message_id)

        additional_kwargs: dict = {}
        files = self._extract_image_and_documents(message.files)
        if files.images:
            logger.debug("found images: %s", len(files.images))
            additional_kwargs[IMAGES_KEY] = files.images
        if files.documents:
            logger.debug("found documents: %s", len(files.documents))
            additional_kwargs[DOCUMENTS_KEY] = files.documents

        return HumanMessage(
            content=content,
            id=message.message_id,
            additional_kwargs=additional_kwargs,
        )

    async def _query_input_to_message(
        self, query_input: QueryInput, message_id: str | None = None
    ) -> HumanMessage:
        """Convert query input to message."""
        content = []

        # Check if text starts with a slash command
        if query_input.text and query_input.text.startswith("/"):
            # Parse skill name from slash command (e.g., "/commit some message")
            parts = query_input.text[1:].split(maxsplit=1)
            if parts:
                skill_name = parts[0]
                skill_content = self.skill_manager.get_skill_content(skill_name)
                if skill_content:
                    # Prepend skill content before the user's message
                    content.append(TextContent.create(skill_content))

        if query_input.text:
            content.append(TextContent.create(query_input.text))

        # We will convert image and documents into their respective msg format
        # inside the graph.
        additional_kwargs: dict = {}
        if query_input.images:
            additional_kwargs[IMAGES_KEY] = query_input.images
        if query_input.documents:
            additional_kwargs[DOCUMENTS_KEY] = query_input.documents

        return HumanMessage(
            content=content,
            id=message_id,
            additional_kwargs=additional_kwargs,
        )

    async def _get_history_user_input(
        self, chat_id: str, message_id: str
    ) -> BaseMessage:
        """Get the last user input message from history."""
        dive_user: DiveUser = self.request_state.dive_user
        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            chat = await db.get_chat_with_messages(chat_id, dive_user["user_id"])
            if chat is None:
                raise ChatError("chat not found")
            message = None
            for i in chat.messages:
                if i.role == Role.USER:
                    message = i
                if i.message_id == message_id:
                    break
            else:
                message = None
            if message is None:
                raise ChatError("message not found")

            return await self._process_history_message(message)


class LogStreamHandler:
    """Handles streaming of logs."""

    def __init__(
        self,
        stream: EventStreamContextManager,
        log_manager: LogManager,
        stream_until: ClientState | None = None,
        stop_on_notfound: bool = True,
        max_retries: int = 10,
        server_names: list[str] | None = None,
    ) -> None:
        """Initialize the log processor."""
        self._stream = stream
        self._log_manager = log_manager
        self._end_event = asyncio.Event()
        self._stop_on_notfound = stop_on_notfound
        self._max_retries = max_retries

        self._stream_until: set[ClientState] = {
            ClientState.CLOSED,
            ClientState.FAILED,
        }
        if stream_until:
            self._stream_until.add(stream_until)

        self._servers_reached_target: set[str] = set()
        self._server_names: set[str] = set(server_names) if server_names else set()

    async def _log_listener(self, msg: LogMsg) -> None:
        await self._stream.write(msg.model_dump_json())
        if msg.client_state in self._stream_until:
            self._servers_reached_target.add(msg.mcp_server_name)
            if self._server_names == self._servers_reached_target:
                self._end_event.set()

    async def stream_logs(self) -> None:
        """Stream logs from MCP servers.

        Keep the connection open until client disconnects or
        client state is reached.

        Streams logs from the given server names.

        If self._stop_on_notfound is False, it will keep retrying until
        the log buffer is found or max retries is reached.
        """
        while self._max_retries > 0:
            self._max_retries -= 1
            self._servers_reached_target = set()

            try:
                async with self._log_manager.listen_log(
                    names=list(self._server_names),
                    listener=self._log_listener,
                ):
                    with suppress(asyncio.CancelledError):
                        await self._end_event.wait()
                        break
            except LogBufferNotFoundError as e:
                logger.warning(
                    "Log buffer not found for server %s, retries left: %d",
                    e.mcp_name,
                    self._max_retries,
                )

                msg = LogMsg(
                    event=LogEvent.STREAMING_ERROR,
                    body=f"Error streaming logs: {e}",
                    mcp_server_name=e.mcp_name,
                )
                await self._stream.write(msg.model_dump_json())

                if self._stop_on_notfound or self._max_retries == 0:
                    break

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.exception(
                    "Error in log streaming for servers %s", self._server_names
                )
                msg = LogMsg(
                    event=LogEvent.STREAMING_ERROR,
                    body=f"Error streaming logs: {e}",
                    mcp_server_name="unknown",
                )
                await self._stream.write(msg.model_dump_json())
                break


def strip_title(title: str) -> str:
    """Strip the title, remove any tags."""
    title = re.sub(r"\s*<.+>.*?</.+>\s*", "", title, flags=re.DOTALL)
    return " ".join(title.split())


def get_original_filename(local_path: str) -> str:
    """Extract the original name from cache file path."""
    return Path(local_path).name.split("-", 1)[-1]


def is_url(file_path: str) -> bool:
    """Check if the file is a URL."""
    result = urlparse(file_path)
    return bool(result.scheme and result.netloc)


def get_filename_remove_url(chat: ChatMessage) -> ChatMessage:
    """Files sould remain their original name, urls created by OAP souldn't exist."""
    for msg in chat.messages:
        files: list[str] = []
        for file in msg.files:
            if is_url(file):
                continue

            file_type = FileType.from_file_path(file)

            if file_type == FileType.IMAGE:
                # Image files need the complete path to be displayed in the UI
                files.append(file)
            else:
                # Other files should remain their original name
                files.append(get_original_filename(file))

        msg.files = files
    return chat


def calculate_token_usage(messages: list[Message]) -> TokenUsage | None:
    """Calculate aggregated token usage from a list of messages.

    Args:
        messages: List of messages to calculate token usage from.

    Returns:
        TokenUsage object with aggregated statistics, or None if no usage data exists.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_user_tokens = 0
    total_custom_prompt_tokens = 0
    total_system_prompt_tokens = 0
    total_tokens = 0
    time_to_first_token = 0.0
    weighted_tps_sum = 0.0

    for message in messages:
        if message.resource_usage:
            total_input_tokens += message.resource_usage.total_input_tokens
            total_output_tokens += message.resource_usage.total_output_tokens
            total_user_tokens += message.resource_usage.user_token
            total_custom_prompt_tokens += message.resource_usage.custom_prompt_token
            total_system_prompt_tokens += message.resource_usage.system_prompt_token
            # Calculate total tokens if not directly available
            total_tokens += (
                message.resource_usage.total_input_tokens
                + message.resource_usage.total_output_tokens
            )

            # Get the first TTFT from assistant messages
            if (
                time_to_first_token == 0.0
                and message.resource_usage.time_to_first_token > 0
            ):
                time_to_first_token = message.resource_usage.time_to_first_token

            # Calculate weighted average of tokens_per_second
            if message.resource_usage.total_output_tokens > 0:
                weighted_tps_sum += (
                    message.resource_usage.tokens_per_second
                    * message.resource_usage.total_output_tokens
                )

    # Calculate average tokens per second
    tokens_per_second = (
        weighted_tps_sum / total_output_tokens if total_output_tokens > 0 else 0.0
    )

    if total_input_tokens > 0 or total_output_tokens > 0:
        return TokenUsage(
            totalInputTokens=total_input_tokens,
            totalOutputTokens=total_output_tokens,
            userToken=total_user_tokens,
            customPromptToken=total_custom_prompt_tokens,
            systemPromptToken=total_system_prompt_tokens,
            totalTokens=total_tokens,
            timeToFirstToken=time_to_first_token,
            tokensPerSecond=tokens_per_second,
        )

    return None
