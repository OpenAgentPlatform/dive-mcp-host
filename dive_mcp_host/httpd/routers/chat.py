import datetime
import json
from typing import TYPE_CHECKING, Annotated, TypeVar

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from dive_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    Message,
    QueryInput,
    Role,
)
from dive_mcp_host.httpd.dependencies import get_app, get_dive_user
from dive_mcp_host.httpd.routers.models import (
    ResultResponse,
    UserInputError,
)
from dive_mcp_host.httpd.routers.utils import ChatProcessor, EventStreamContextManager
from dive_mcp_host.httpd.server import DiveHostAPI

if TYPE_CHECKING:
    from dive_mcp_host.httpd.middlewares.general import DiveUser

chat = APIRouter(tags=["chat"])

T = TypeVar("T")


class DataResult[T](ResultResponse):
    """Generic result that extends ResultResponse with a data field."""

    data: T | None


@chat.get("/list")
async def list_chat(
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> DataResult[list[Chat]]:
    """List all available chats.

    Args:
        app (DiveHostAPI): The DiveHostAPI instance.
        dive_user (DiveUser): The DiveUser instance.

    Returns:
        DataResult[list[Chat]]: List of available chats.
    """
    async with app.db_sessionmaker() as session:
        chats = await app.msg_store(session).get_all_chats(dive_user["user_id"])
    return DataResult(success=True, message=None, data=chats)


@chat.post("")
async def create_chat(  # noqa: PLR0913
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """Create a new chat.

    Args:
        request (Request): The request object.
        app (DiveHostAPI): The DiveHostAPI instance.
        chat_id (str | None): The ID of the chat to create.
        message (str | None): The message to send.
        files (list[UploadFile] | None): The files to upload.
        filepaths (list[str] | None): The file paths to upload.
    """
    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await app.store.upload_files(files, filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()
    query_input = QueryInput(text=message, images=images, documents=documents)

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, query_input, None)

    stream.add_task(process)
    return response


@chat.post("/edit")
async def edit_chat(  # noqa: PLR0913
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message_id: Annotated[str | None, Form(alias="messageId")] = None,
    content: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """Edit a chat.

    Args:
        request (Request): The request object.
        app (DiveHostAPI): The DiveHostAPI instance.
        chat_id (str | None): The ID of the chat to edit.
        message_id (str | None): The ID of the message to edit.
        content (str | None): The content to send.
        files (list[UploadFile] | None): The files to upload.
        filepaths (list[str] | None): The file paths to upload.
    """
    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await app.store.upload_files(files, filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()
    query_input = QueryInput(text=content, images=images, documents=documents)

    async def process() -> None:
        async with stream:
            dive_user = request.state.dive_user
            async with app.db_sessionmaker() as session:
                await app.msg_store(session).update_message_content(
                    message_id, query_input, dive_user["user_id"]
                )

                next_ai_message = await app.msg_store(session).get_next_ai_message(
                    chat_id, message_id
                )
                await session.commit()
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(
                chat_id, query_input, next_ai_message.message_id
            )

    stream.add_task(process)
    return response


@chat.post("/retry")
async def retry_chat(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message_id: Annotated[str | None, Form(alias="messageId")] = None,
) -> StreamingResponse:
    """Retry a chat.

    Args:
        request (Request): The request object.
        app (DiveHostAPI): The DiveHostAPI instance.
        chat_id (str | None): The ID of the chat to retry.
        message_id (str | None): The ID of the message to retry.
    """
    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    stream = EventStreamContextManager()
    response = stream.get_response()

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, None, message_id)

    stream.add_task(process)
    return response


@chat.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> DataResult[ChatMessage]:
    """Get a specific chat by ID with its messages.

    Args:
        chat_id (str): The ID of the chat to retrieve.
        app (DiveHostAPI): The DiveHostAPI instance.
        dive_user (DiveUser): The DiveUser instance.

    Returns:
        DataResult[ChatMessage]: The chat and its messages.
    """
    async with app.db_sessionmaker() as session:
        chat = await app.msg_store(session).get_chat_with_messages(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )

    if chat is None:
        raise UserInputError("Chat not found")

    checkpointer_messages = await app.dive_host["default"].get_messages(
        thread_id=chat_id,
        user_id=dive_user["user_id"] or "",
    )

    checkpointer_chat_message = []

    for index, msg in enumerate(checkpointer_messages):
        if msg.type == "human":
            msg_content = ""
            if isinstance(msg.content, str):
                msg_content = msg.content
            elif isinstance(msg.content, list):
                first_content = msg.content[0] if msg.content else {}
                msg_content = (
                    first_content.get("text", "")
                    if isinstance(first_content, dict)
                    else ""
                )
            else:
                msg_content = ""
            checkpointer_chat_message.append(
                Message(
                    id=index,
                    createdAt=datetime.datetime.now(datetime.UTC),
                    content=msg_content,
                    role=Role("user"),
                    chatId=chat_id,
                    messageId=msg.id or "",
                    files="[]",
                    resource_usage=None,
                )
            )
        if msg.type == "ai":
            checkpointer_chat_message.append(
                Message(
                      id=index,
                      createdAt=datetime.datetime.now(datetime.UTC),
                      content=str(msg.content),
                      role=Role("assistant"),
                      chatId=chat_id,
                      messageId=msg.id or "",
                      files="[]",
                      resource_usage=None,
                )
            )
            tool_calls = msg.additional_kwargs.get("tool_calls", [])
            if tool_calls:
                tool_call_array = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    tool_call_array.append(
                        {
                            "name": function.get("name", ""),
                            "args": json.loads(function.get("arguments", "{}")),
                        }
                    )
                content = json.dumps(tool_call_array)
                checkpointer_chat_message.append(
                    Message(
                        id=index,
                        createdAt=datetime.datetime.now(datetime.UTC),
                        content=content,
                        role=Role("tool_call"),
                        chatId=chat_id,
                        messageId=msg.id or "",
                        files="[]",
                        resource_usage=None,
                    )
                )
        if msg.type == "tool":
            checkpointer_chat_message.append(
                Message(
                    id=index,
                    createdAt=datetime.datetime.now(datetime.UTC),
                    content=str(msg.content),
                    role=Role("tool_result"),
                    chatId=chat_id,
                    messageId=msg.id or "",
                    files="[]",
                    resource_usage=None,
                )
            )
    # print(
    #     json.dumps(
    #         checkpointer_chat_message,
    #         default=lambda o: o.dict() if hasattr(o, "dict") else str(o),
    #     )
    # )

    # TODO:  merge with msg_store chat by message_id
    return DataResult(success=True, message=None, data=chat)


@chat.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> ResultResponse:
    """Delete a specific chat by ID.

    Args:
        chat_id (str): The ID of the chat to delete.
        app (DiveHostAPI): The DiveHostAPI instance.
        dive_user (DiveUser): The DiveUser instance.

    Returns:
        ResultResponse: Result of the delete operation.
    """
    async with app.db_sessionmaker() as session:
        await app.msg_store(session).delete_chat(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )
        await session.commit()
    return ResultResponse(success=True, message=None)


@chat.post("/{chat_id}/abort")
async def abort_chat(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Abort an ongoing chat operation.

    Args:
        chat_id (str): The ID of the chat to abort.
        app (DiveHostAPI): The DiveHostAPI instance.

    Returns:
        ResultResponse: Result of the abort operation.
    """
    abort_controller = app.abort_controller
    ok = await abort_controller.abort(chat_id)
    if not ok:
        raise UserInputError("Chat not found")

    return ResultResponse(success=True, message="Chat abort signal sent successfully")
