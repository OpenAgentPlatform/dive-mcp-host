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
from langchain_core.output_parsers import StrOutputParser
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
    AuthenticationRequiredContent,
    ChatInfoContent,
    InteractiveContent,
    MessageInfoContent,
    StreamMessage,
    TokenUsage,
    ToolCallsContent,
    ToolResultContent,
)
from dive_mcp_host.httpd.server import DiveHostAPI
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
            self._exit_message = StreamMessage(
                type="error",
                content=f"<thread-query-error>{exc_val}</thread-query-error>",
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
        model_name = msg.response_metadata.get("model_name")

        if model_name in {"gemini-2.5-flash-image-preview"}:
            result = f"{result} {await self._gemini_25_image(msg)}"

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

    async def _gemini_25_image(self, msg: AIMessage) -> str:
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

    async def handle_chat(  # noqa: C901, PLR0912, PLR0915
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

        if isinstance(query_input, QueryInput) and query_input.text:
            async with self.app.db_sessionmaker() as session:
                db = self.app.msg_store(session)
                if not await db.check_chat_exists(chat_id, dive_user["user_id"]):
                    title_await = asyncio.create_task(
                        self._generate_title(query_input.text)
                    )

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            )
        )

        start = time.time()
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
        user_message, ai_message, current_messages = await self._process_chat(
            chat_id,
            query_message,
            is_resend=regenerate_message_id is not None,
        )
        end = time.time()
        if ai_message is None:
            if title_await:
                title_await.cancel()
            return "", TokenUsage()
        assert user_message.id
        assert ai_message.id

        if title_await:
            title = await title_await

        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            if not await db.check_chat_exists(chat_id, dive_user["user_id"]):
                await db.create_chat(
                    chat_id, title, dive_user["user_id"], dive_user["user_type"]
                )

            original_msg_exist: bool = False
            if regenerate_message_id and query_message:
                assert query_message.id, "Message ID doesn't exist"
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

            for message in current_messages:
                assert message.id
                if isinstance(message, HumanMessage):
                    if not query_input or (
                        regenerate_message_id and original_msg_exist
                    ):
                        continue
                    await db.create_message(
                        NewMessage(
                            chatId=chat_id,
                            role=Role.USER,
                            messageId=message.id,
                            content=query_input.text or "",  # type: ignore
                            files=(
                                (query_input.images or [])
                                + (query_input.documents or [])
                            ),
                        ),
                    )
                elif isinstance(message, AIMessage):
                    if (
                        message.usage_metadata is None
                        or (duration := message.usage_metadata.get("total_duration"))
                        is None
                    ):
                        duration = 0 if message.id == ai_message.id else end - start
                    resource_usage = ResourceUsage(
                        model=message.response_metadata.get("model")
                        or message.response_metadata.get("model_name")
                        or "",
                        total_input_tokens=message.usage_metadata["input_tokens"]
                        if message.usage_metadata
                        else 0,
                        total_output_tokens=message.usage_metadata["output_tokens"]
                        if message.usage_metadata
                        else 0,
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

        token_usage = TokenUsage(
            totalInputTokens=ai_message.usage_metadata["input_tokens"]
            if ai_message.usage_metadata
            else 0,
            totalOutputTokens=ai_message.usage_metadata["output_tokens"]
            if ai_message.usage_metadata
            else 0,
            totalTokens=ai_message.usage_metadata["total_tokens"]
            if ai_message.usage_metadata
            else 0,
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
        _, ai_message, _ = await self._process_chat(
            chat_id, query_input, history, tools
        )
        usage = TokenUsage()
        if ai_message.usage_metadata:
            usage.total_input_tokens = ai_message.usage_metadata["input_tokens"]
            usage.total_output_tokens = ai_message.usage_metadata["output_tokens"]
            usage.total_tokens = ai_message.usage_metadata["total_tokens"]

        return str(ai_message.content), usage

    async def _process_chat(
        self,
        chat_id: str | None,
        query_input: str | QueryInput | BaseMessage | None,
        history: list[BaseMessage] | None = None,
        tools: list | None = None,
        is_resend: bool = False,
    ) -> tuple[HumanMessage, AIMessage, list[BaseMessage]]:
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
            system_prompt=prompt,
            disable_default_system_prompt=self.disable_dive_system_prompt,
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
            return await self._handle_response(response_generator)

        raise RuntimeError("Unreachable")

    async def _stream_text_msg(self, message: AIMessage) -> None:
        content = await self._content_handler.invoke(message)
        if content:
            await self.stream.write(StreamMessage(type="text", content=content))
        if message.response_metadata.get("stop_reason") == "max_tokens":
            await self.stream.write(
                StreamMessage(
                    type="error",
                    content="stop_reason: max_tokens",
                )
            )

    async def _stream_tool_calls_msg(self, message: AIMessage) -> None:
        await self.stream.write(
            StreamMessage(
                type="tool_calls",
                content=[
                    ToolCallsContent(name=c["name"], arguments=c["args"])
                    for c in message.tool_calls
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

    async def _handle_response(  # noqa: C901, PLR0912
        self, response: AsyncIterator[dict[str, Any] | Any]
    ) -> tuple[HumanMessage | Any, AIMessage | Any, list[BaseMessage]]:
        """Handle response.

        Returns:
            tuple[HumanMessage | Any, AIMessage | Any, list[BaseMessage]]:
            The human message, the AI message, and all messages of the current query.
        """
        user_message = None
        ai_message = None
        values_messages: list[BaseMessage] = []
        current_messages: list[BaseMessage] = []
        async for res_type, res_content in response:
            if res_type == "messages":
                message, _ = res_content
                if isinstance(message, AIMessage):
                    logger.log(TRACE, "got AI message: %s", message.model_dump_json())
                    if message.content:
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

        return user_message, ai_message, current_messages

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

    def __init__(  # noqa: PLR0913
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
