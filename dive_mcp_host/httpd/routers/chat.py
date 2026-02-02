from asyncio import TaskGroup
from logging import getLogger
from typing import TYPE_CHECKING, Annotated, TypeVar
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dive_mcp_host.httpd.database.models import Chat, ChatMessage, FTSResult, QueryInput
from dive_mcp_host.httpd.dependencies import get_app, get_dive_user
from dive_mcp_host.httpd.routers.models import (
    CHAT_EVENT_STREAM,
    DataResult,
    ResultResponse,
    SortBy,
    UserInputError,
)
from dive_mcp_host.httpd.routers.utils import (
    ChatProcessor,
    EventStreamContextManager,
    calculate_token_usage,
    get_filename_remove_url,
)
from dive_mcp_host.httpd.server import DiveHostAPI

if TYPE_CHECKING:
    from dive_mcp_host.httpd.middlewares.general import DiveUser

logger = getLogger(__name__)

chat = APIRouter(tags=["chat"])

T = TypeVar("T")


class ChatList(BaseModel):
    """Result data type for list API."""

    starred: list[Chat] = Field(default_factory=list)
    normal: list[Chat] = Field(default_factory=list)


@chat.post("/search")
async def search(
    query: Annotated[str, Body(description="Text to search for")],
    max_length: Annotated[
        int, Body(description="Max snippet length for title and content")
    ] = 150,
    app: DiveHostAPI = Depends(get_app),
) -> DataResult[list[FTSResult]]:
    """Full text search on chat title and message.

    Returns a list of search results containing a chat title and message snippet.

    Orderd by chat.updated_at DESC (Newest first), msg.created_at ASC (Oldest first)
    """
    async with app.db_sessionmaker() as session:
        matches = await app.msg_store(session).full_text_search(
            query=query, max_length=max_length, start_sel="", stop_sel=""
        )

    result: list[FTSResult] = []
    existing_chat: set[str] = set()
    for match in matches:
        if match.chat_id in existing_chat:
            continue
        existing_chat.add(match.chat_id)
        result.append(match)

    return DataResult(success=True, message=None, data=result)


@chat.post("/bulk-delete")
async def bulk_delete(
    chat_ids: Annotated[list[str], Body(description="List of chat IDs to delete")],
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> ResultResponse:
    """Delete multiple chats by their IDs."""
    async with app.db_sessionmaker() as session:
        await app.msg_store(session).bulk_delete(
            chat_ids=chat_ids,
            user_id=dive_user["user_id"],
        )
        await session.commit()

    async with TaskGroup() as group:
        for chat in chat_ids:
            group.create_task(app.dive_host["default"].delete_thread(chat))

    return ResultResponse(success=True, message=None)


@chat.delete("/purge")
async def purge(
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> ResultResponse:
    """Delete ALL chats."""
    chat_ids: list[str] = []

    async with app.db_sessionmaker() as session:
        chats = await app.msg_store(session).get_all_chats(user_id=dive_user["user_id"])
        chat_ids = [c.id for c in chats]
        await app.msg_store(session).bulk_delete(
            chat_ids=chat_ids,
            user_id=dive_user["user_id"],
        )
        await session.commit()

    async with TaskGroup() as group:
        for chat in chat_ids:
            group.create_task(app.dive_host["default"].delete_thread(chat))

    return ResultResponse(success=True, message=None)


@chat.get("/list")
async def list_chat(
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
    sort_by: Annotated[
        SortBy,
        Query(description="Sort by 'chat' or 'msg' creation time"),
    ] = SortBy.CHAT,
) -> DataResult[ChatList]:
    """List all available chats."""
    result = ChatList()
    async with app.db_sessionmaker() as session:
        chats = await app.msg_store(session).get_all_chats(
            dive_user["user_id"],
            sort_by=sort_by,
        )
        for chat in chats:
            if chat.starred_at:
                result.starred.append(chat)
                continue
            result.normal.append(chat)

    return DataResult(success=True, message=None, data=result)


<<<<<<< HEAD
@chat.post(
    "",
    responses={200: CHAT_EVENT_STREAM},
    response_class=StreamingResponse,
)
async def create_chat(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[
        str | None, Form(alias="chatId", description="ID for the new chat")
    ] = None,
    message: Annotated[str | None, Form(description="Initial message to send")] = None,
    files: Annotated[
        list[UploadFile] | None, File(description="Files to upload")
    ] = None,
    filepaths: Annotated[
        list[str] | None, Form(description="File paths to upload")
    ] = None,
) -> StreamingResponse:
    """Create a new chat."""
=======
class _ChatMetadata(BaseModel):
    skills_activate: list[str] = Field(description="List of skills to load")
    skills_deactivate: list[str] = Field(description="List of skills to unload")


@chat.post("")
async def create_chat(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
    metadata: Annotated[_ChatMetadata | None, Form()] = None,
) -> StreamingResponse:
    """Create a new chat.

    Args:
        request (Request): The request object.
        app (DiveHostAPI): The DiveHostAPI instance.
        chat_id (str | None): The ID of the chat to create.
        message (str | None): The message to send.
        files (list[UploadFile] | None): The files to upload.
        filepaths (list[str] | None): The file paths to upload.
        metadata (_ChatMetadata | None): Chat metadata.
    """
>>>>>>> 3e535e9 (wip: skill related api definition and adjustments)
    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await app.store.upload_files(files + filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()
    query_input = QueryInput(text=message, images=images, documents=documents)

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, query_input, None)

    stream.add_task(process)
    return response


@chat.patch("/{chat_id}")
async def patch_chat(
    chat_id: str,
    dive_user: "DiveUser" = Depends(get_dive_user),
    app: DiveHostAPI = Depends(get_app),
    title: Annotated[str | None, Body(description="New title for the chat")] = None,
    star: Annotated[
        bool | None, Body(description="New star status for the chat")
    ] = None,
) -> ResultResponse:
    """Update chat title or star status."""
    async with app.db_sessionmaker() as session:
        chat = await app.msg_store(session).patch_chat(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
            title=title,
            star=star,
        )
        if chat is None:
            raise UserInputError(f"Chat {chat_id} not found")
        await session.commit()

    return ResultResponse(success=True)


# Frontend sets the message id to "0" when calling edit API
# on an errored message.
ERROR_MSG_ID = "0"


@chat.post(
    "/edit",
    responses={200: CHAT_EVENT_STREAM},
    response_class=StreamingResponse,
)
async def edit_chat(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[
        str | None, Form(alias="chatId", description="ID of the chat to edit")
    ] = None,
    message_id: Annotated[
        str | None, Form(alias="messageId", description="ID of the message to edit")
    ] = None,
    content: Annotated[
        str | None, Form(description="New content for the message")
    ] = None,
    files: Annotated[
        list[UploadFile] | None, File(description="Files to upload")
    ] = None,
    filepaths: Annotated[
        list[str] | None, Form(description="File paths to upload")
    ] = None,
) -> StreamingResponse:
    """Edit a message in a chat and query again."""
    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    # message id needs to be unique
    if message_id == ERROR_MSG_ID:
        message_id = str(uuid4())

    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await app.store.upload_files(files + filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()
    query_input = QueryInput(text=content, images=images, documents=documents)

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, query_input, message_id)

    stream.add_task(process)
    return response


@chat.post(
    "/retry",
    responses={200: CHAT_EVENT_STREAM},
    response_class=StreamingResponse,
)
async def retry_chat(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    chat_id: Annotated[
        str | None, Body(alias="chatId", description="ID of the chat to retry")
    ] = None,
    message_id: Annotated[
        str | None,
        Body(alias="messageId", description="ID of the message to retry from"),
    ] = None,
) -> StreamingResponse:
    """Retry a chat from a specific message."""
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
) -> DataResult[ChatMessage | None]:
    """Get a specific chat by ID with its messages."""
    async with app.db_sessionmaker() as session:
        chat = await app.msg_store(session).get_chat_with_messages(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )
        if chat:
            chat = get_filename_remove_url(chat)
            # Calculate token_usage from all assistant messages
            chat.token_usage = calculate_token_usage(chat.messages)
    return DataResult(success=True, message=None, data=chat)


@chat.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
    dive_user: "DiveUser" = Depends(get_dive_user),
) -> ResultResponse:
    """Delete a specific chat by ID."""
    async with app.db_sessionmaker() as session:
        await app.msg_store(session).delete_chat(
            chat_id=chat_id,
            user_id=dive_user["user_id"],
        )
        await session.commit()
    await app.dive_host["default"].delete_thread(chat_id)
    return ResultResponse(success=True, message=None)


@chat.post("/{chat_id}/abort")
async def abort_chat(
    chat_id: str,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Abort an ongoing chat operation."""
    abort_controller = app.abort_controller
    ok = await abort_controller.abort(chat_id)
    if not ok:
        raise UserInputError("Chat not found")

    return ResultResponse(success=True, message="Chat abort signal sent successfully")
