import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, suppress
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from starlette.datastructures import State

from dive_mcp_host.httpd.database.models import (
    Message,
    NewMessage,
    QueryInput,
    ResourceUsage,
    Role,
)
from dive_mcp_host.httpd.routers.models import (
    ChatInfoContent,
    MessageInfoContent,
    StreamMessage,
    TokenUsage,
    ToolCallsContent,
    ToolResultContent,
)
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.httpd.store.store import SUPPORTED_IMAGE_EXTENSIONS, Store
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
- NO explanations, quotes, or extra text
- NO punctuation at the end
- If the input is URL only, output the description of the URL, for example, "the URL of xxx website"
- If the input contains Traditional Chinese characters, use Traditional Chinese for the title.
- For all other languages, generate the title in the same language as the input."""  # noqa: E501


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
                type="error", content=str(exc_val)
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
        self.store: Store = app.store
        self.dive_host: DiveMcpHost = app.dive_host["default"]
        self._str_output_parser = StrOutputParser()

    async def handle_chat(  # noqa: C901, PLR0912, PLR0915
        self,
        chat_id: str | None,
        query_input: QueryInput | None,
        regenerate_message_id: str | None,
    ) -> tuple[str, TokenUsage]:
        """Handle chat."""
        chat_id = chat_id if chat_id else str(uuid4())
        dive_user: DiveUser = self.request_state.dive_user
        title = "New Chat"
        title_await = None

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
        result = str(ai_message.content)

        if title_await:
            title = await title_await

        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            if not await db.check_chat_exists(chat_id, dive_user["user_id"]):
                await db.create_chat(
                    chat_id, title, dive_user["user_id"], dive_user["user_type"]
                )

            if regenerate_message_id:
                await db.delete_messages_after(chat_id, regenerate_message_id)
                if query_input and query_message:
                    await db.update_message_content(
                        query_message.id,  # type: ignore
                        QueryInput(
                            text=query_input.text or "",
                            images=query_input.images or [],
                            documents=query_input.documents or [],
                        ),
                    )

            for message in current_messages:
                assert message.id
                if isinstance(message, HumanMessage):
                    if not query_input or regenerate_message_id:
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
                    if message.tool_calls:
                        await db.create_message(
                            NewMessage(
                                chatId=chat_id,
                                role=Role.TOOL_CALL,
                                messageId=message.id,
                                content=json.dumps(message.tool_calls),
                                resource_usage=resource_usage,
                            ),
                        )
                    else:
                        await db.create_message(
                            NewMessage(
                                chatId=chat_id,
                                role=Role.ASSISTANT,
                                messageId=message.id,
                                content=result,
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
        elif user_prompt := self.app.prompt_config_manager.get_prompt("system"):
            prompt = user_prompt

        chat = self.dive_host.chat(
            chat_id=chat_id,
            user_id=dive_user.get("user_id") or "default",
            tools=tools,
            system_prompt=prompt,
        )
        async with AsyncExitStack() as stack:
            if chat_id:
                await stack.enter_async_context(
                    self.app.abort_controller.abort_signal(chat_id, chat.abort)
                )
            await stack.enter_async_context(chat)
            response_generator = chat.query(
                messages,
                stream_mode=["messages", "values", "updates"],
                is_resend=is_resend,
            )
            return await self._handle_response(response_generator)

        raise RuntimeError("Unreachable")

    async def _stream_text_msg(self, message: AIMessage) -> None:
        content = self._str_output_parser.invoke(message)
        if content:
            await self.stream.write(StreamMessage(type="text", content=content))

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
        async with chat:
            responses = [
                response async for response in chat.query(query, stream_mode="updates")
            ]
            return responses[0]["agent"]["messages"][0].content
        return "New Chat"

    async def _process_history_messages(
        self, history_messages: list[Message], history: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Process history messages."""
        for message in history_messages:
            files: list[str] = message.files
            if not files:
                message_content = message.content.strip()
                if message.role == Role.USER:
                    history.append(
                        HumanMessage(content=message_content, id=message.message_id)
                    )
                else:
                    history.append(
                        AIMessage(content=message_content, id=message.message_id)
                    )
            else:
                content = []
                if message.content:
                    content.append(
                        {
                            "type": "text",
                            "text": message.content,
                        }
                    )

                for file_path in files:
                    local_path = file_path
                    if any(
                        local_path.endswith(suffix)
                        for suffix in SUPPORTED_IMAGE_EXTENSIONS
                    ):
                        base64_image = await self.store.get_image(local_path)

                        content.append(
                            {
                                "type": "text",
                                "text": f"![Image]({base64_image})",
                            }
                        )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                },
                            }
                        )
                    else:
                        content.append(
                            {
                                "type": "text",
                                "text": f"![Document]({local_path})",
                            }
                        )

                if message.role == Role.ASSISTANT:
                    history.append(AIMessage(content=content))
                else:
                    history.append(HumanMessage(content=content))

        return history

    async def _query_input_to_message(
        self, query_input: QueryInput, message_id: str | None = None
    ) -> HumanMessage:
        """Convert query input to message."""
        content = []

        if query_input.text:
            content.append(
                {
                    "type": "text",
                    "text": query_input.text,
                }
            )

        for image in query_input.images or []:
            local_path = image
            base64_image = await self.store.get_image(local_path)
            content.append(
                {
                    "type": "text",
                    "text": f"![Image]({base64_image})",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                    },
                }
            )

        for document in query_input.documents or []:
            local_path = document
            content.append(
                {
                    "type": "text",
                    "text": f"![Document]({local_path})",
                }
            )
        return HumanMessage(content=content, id=message_id)

    async def _get_history_user_input(
        self, chat_id: str, message_id: str
    ) -> BaseMessage:
        """Get history user input."""
        dive_user: DiveUser = self.request_state.dive_user
        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            chat = await db.get_chat_with_messages(chat_id, dive_user["user_id"])
            if chat is None:
                raise ChatError("chat not found")
            message = next(
                (msg for msg in chat.messages if msg.message_id == message_id),
                None,
            )
            if message is None:
                raise ChatError("message not found")

            return (
                await self._process_history_messages(
                    [message],
                    [],
                )
            )[0]
