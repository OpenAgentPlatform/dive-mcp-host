import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, desc, exists, func, insert, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dive_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    FTSResult,
    Message,
    NewMessage,
    QueryInput,
    ResourceUsage,
    Role,
)
from dive_mcp_host.httpd.database.orm_models import Chat as ORMChat
from dive_mcp_host.httpd.database.orm_models import Message as ORMMessage
from dive_mcp_host.httpd.database.orm_models import (
    ResourceUsage as ORMResourceUsage,
)
from dive_mcp_host.httpd.routers.models import SortBy

from .abstract import AbstractMessageStore

if TYPE_CHECKING:
    from collections.abc import Sequence

# Trigram tokenizer / tsquery requires at least 3 characters
MIN_TRIGRAM_LENGTH = 3


def _build_snippet(
    content: str,
    query: str,
    max_length: int = 150,
    start_sel: str = "<b>",
    stop_sel: str = "</b>",
) -> str:
    """Build a highlighted snippet from content, similar to FTS5 snippet()."""
    escaped = re.escape(query)

    # Find the position of the first match
    match = re.search(escaped, content, re.IGNORECASE)
    match_pos = match.start() if match else 0

    # Extract a window of characters around the first match
    half = max_length // 2
    start = max(0, match_pos - half)
    end = min(len(content), start + max_length)
    # Re-adjust start if we hit the end of content
    start = max(0, end - max_length)
    snippet_text = content[start:end]

    # Add ellipsis if truncated
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""

    # Highlight all occurrences of the query
    highlighted = re.sub(
        f"({escaped})",
        lambda m: f"{start_sel}{m.group(1)}{stop_sel}",
        snippet_text,
        flags=re.IGNORECASE,
    )

    return f"{prefix}{highlighted}{suffix}"


class BaseMessageStore(AbstractMessageStore):
    """Base Message store.

    Contains queries that can function in both SQLite and PostgreSQL
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the message store.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    async def get_all_chats(
        self,
        user_id: str | None = None,
        sort_by: SortBy = SortBy.CHAT,
    ) -> list[Chat]:
        """Retrieve all chats from the database.

        Args:
            user_id: User ID or fingerprint, depending on the prefix.
            sort_by: Sort by.
                - 'chat': Sort by chat creation time.
                - 'msg': Sort by message creation time.
                default: 'chat'

        Starred chat will always be at top.

        Returns:
            List of Chat objects.
        """
        if sort_by == SortBy.MESSAGE:
            query = (
                select(
                    ORMChat,
                    func.coalesce(
                        func.max(ORMMessage.created_at), ORMChat.created_at
                    ).label("last_message_at"),
                )
                .outerjoin(ORMMessage, ORMChat.id == ORMMessage.chat_id)
                .group_by(
                    ORMChat.id,
                    ORMChat.title,
                    ORMChat.created_at,
                    ORMChat.user_id,
                )
                .where(ORMChat.user_id == user_id)
                .order_by(desc(ORMChat.starred_at))
                .order_by(desc("last_message_at"))
            )
            result = await self._session.execute(query)
            chats: Sequence[ORMChat] = result.scalars().all()

        elif sort_by == SortBy.CHAT:
            query = (
                select(ORMChat)
                .where(ORMChat.user_id == user_id)
                .order_by(desc(ORMChat.starred_at))
                .order_by(desc(ORMChat.created_at))
            )
            result = await self._session.scalars(query)
            chats: Sequence[ORMChat] = result.all()

        else:
            raise ValueError(f"Invalid sort_by value: {sort_by}")

        return [
            Chat(
                id=chat.id,
                title=chat.title,
                createdAt=chat.created_at,
                user_id=chat.user_id,
                updatedAt=chat.updated_at,
                starredAt=chat.starred_at,
            )
            for chat in chats
        ]

    async def patch_chat(
        self,
        chat_id: str,
        user_id: str | None = None,
        title: str | None = None,
        star: bool | None = None,
    ) -> Chat | None:
        """Patch chat.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.
            title: New title for the chat.
            star: Star the chat.

        Returns:
            The updated chat, or None if chat is not found.
        """
        query = update(ORMChat).where(ORMChat.user_id == user_id, ORMChat.id == chat_id)

        current_ts = datetime.now(UTC)
        query = query.values(updated_at=current_ts)
        if star is True:
            query = query.values(starred_at=current_ts)
        elif star is False:
            query = query.values(starred_at=None)
        if title is not None:
            query = query.values(title=title)

        query = query.returning(ORMChat)

        result: ORMChat | None = await self._session.scalar(query)
        if not result:
            return None

        return Chat(
            id=result.id,
            title=result.title,
            createdAt=result.created_at,
            updatedAt=result.updated_at,
            starredAt=result.starred_at,
            user_id=result.user_id,
        )

    async def get_chat_with_messages(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> ChatMessage | None:
        """Retrieve a chat with all its messages.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            ChatMessage object or None if not found.
        """
        query = (
            select(ORMChat)
            .options(
                selectinload(ORMChat.messages).selectinload(ORMMessage.resource_usage),
            )
            .where(ORMChat.user_id == user_id)
            .where(ORMChat.id == chat_id)
            .order_by(ORMChat.created_at.desc())
        )
        data = await self._session.scalar(query)
        if data is None:
            return None

        chat = Chat(
            id=data.id,
            title=data.title,
            createdAt=data.created_at,
            updatedAt=data.updated_at,
            starredAt=data.starred_at,
            user_id=data.user_id,
        )
        messages: list[Message] = []
        for msg in data.messages:
            resource_usage = (
                ResourceUsage.model_validate(
                    msg.resource_usage,
                    from_attributes=True,
                )
                if msg.resource_usage is not None
                else None
            )
            messages.append(
                Message(
                    id=msg.id,
                    createdAt=msg.created_at,
                    content=msg.content,
                    role=Role(msg.role),
                    chatId=msg.chat_id,
                    messageId=msg.message_id,
                    files=json.loads(msg.files) if msg.files else [],
                    toolCalls=msg.tool_calls or [],
                    resource_usage=resource_usage,
                ),
            )
        return ChatMessage(chat=chat, messages=messages)

    async def create_chat(
        self,
        chat_id: str,
        title: str,
        user_id: str | None = None,
        user_type: str | None = None,
    ) -> Chat | None:
        """Create a new chat.

        Args:
            chat_id: Unique identifier for the chat.
            title: Title of the chat.
            user_id: User ID or fingerprint, depending on the prefix.
            user_type: Optional user type

        Returns:
            Created Chat object or None if creation failed.
        """
        raise NotImplementedError(
            "The implementation of the method varies on different database.",
        )

    async def create_message(self, message: NewMessage) -> Message:
        """Create a new message.

        Args:
            message: NewMessage object containing message data.

        Returns:
            Created Message object.
        """
        query = (
            insert(ORMMessage)
            .values(
                {
                    "created_at": datetime.now(UTC),
                    "content": message.content,
                    "role": message.role,
                    "chat_id": message.chat_id,
                    "message_id": message.message_id,
                    "files": json.dumps(message.files),
                    "tool_calls": message.tool_calls,
                },
            )
            .returning(ORMMessage)
        )
        new_msg = await self._session.scalar(query)
        if new_msg is None:
            raise Exception(f"Create message failed: {message}")

        # NOTE: Only LLM messages will have resource usage
        new_resource_usage = None
        if message.role == Role.ASSISTANT and message.resource_usage is not None:
            query = (
                insert(ORMResourceUsage)
                .values(
                    {
                        "message_id": message.message_id,
                        "model": message.resource_usage.model,
                        "total_input_tokens": message.resource_usage.total_input_tokens,
                        "total_output_tokens": message.resource_usage.total_output_tokens,  # noqa: E501
                        "user_token": message.resource_usage.user_token,
                        "custom_prompt_token": message.resource_usage.custom_prompt_token,  # noqa: E501
                        "system_prompt_token": message.resource_usage.system_prompt_token,  # noqa: E501
                        "time_to_first_token": message.resource_usage.time_to_first_token,  # noqa: E501
                        "tokens_per_second": message.resource_usage.tokens_per_second,
                        "total_run_time": message.resource_usage.total_run_time,
                    },
                )
                .returning(ORMResourceUsage)
            )
            new_resource_usage = await self._session.scalar(query)
            if new_resource_usage is None:
                raise Exception(f"Create resource usage failed: {message}")

        resource_usage = (
            ResourceUsage.model_validate(
                new_resource_usage,
                from_attributes=True,
            )
            if new_resource_usage is not None
            else None
        )
        return Message(
            id=new_msg.id,
            createdAt=new_msg.created_at,
            content=new_msg.content,
            role=Role(new_msg.role),
            chatId=new_msg.chat_id,
            messageId=new_msg.message_id,
            files=json.loads(new_msg.files),
            toolCalls=new_msg.tool_calls or [],
            resource_usage=resource_usage,
        )

    async def check_chat_exists(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Check if a chat exists in the database.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            True if chat exists, False otherwise.
        """
        query = (
            exists(ORMChat)
            .where(ORMChat.id == chat_id)
            .where(ORMChat.user_id == user_id)
            .select()
        )
        exist = await self._session.scalar(query)
        return bool(exist)

    async def delete_chat(self, chat_id: str, user_id: str | None = None) -> None:
        """Delete a chat from the database.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.
        """
        query = (
            delete(ORMChat)
            .where(ORMChat.id == chat_id)
            .where(ORMChat.user_id == user_id)
        )
        await self._session.execute(query)

    async def bulk_delete(
        self, chat_ids: list[str], user_id: str | None = None
    ) -> None:
        """Bulk delete chat from the database.

        Args:
            chat_ids: A list of chat id.
            user_id: User ID or fingerprint, depending on the prefix.
        """
        query = (
            delete(ORMChat)
            .where(ORMChat.id.in_(chat_ids))
            .where(ORMChat.user_id == user_id)
        )
        await self._session.execute(query)

    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete all messages after a specific message in a chat."""
        query = (
            delete(ORMMessage)
            .where(ORMMessage.chat_id == chat_id)
            .where(
                ORMMessage.created_at
                > (
                    select(ORMMessage.created_at)
                    .where(ORMMessage.chat_id == chat_id)
                    .where(ORMMessage.message_id == message_id)
                    .scalar_subquery()
                )
            )
        )
        await self._session.execute(query)

    async def lock_msg(
        self,
        chat_id: str,
        message_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Locks the message.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Unique identifier for the message.
            user_id: User ID or fingerprint, depending on the prefix.
                Should not be used in this current implementation.

        Returns:
            If the message exist and is locked, returns True, False otherwise.
        """
        if user_id is not None:
            raise ValueError("user_id should not be used.")

        query = (
            select(ORMMessage.message_id)
            .where(
                ORMMessage.message_id == message_id,
                ORMMessage.chat_id == chat_id,
            )
            .with_for_update()
        )
        message = await self._session.scalar(query)
        return message is not None

    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        user_id: str | None = None,
    ) -> Message:
        """Update the content of a message.

        Args:
            message_id: Unique identifier for the message.
            data: New content for the message.
            user_id: User ID or fingerprint, depending on the prefix.
                Should not be used in this current implementation.

        Returns:
            Updated Message object.
        """
        if user_id is not None:
            raise ValueError("user_id should not be used.")

        # Prepare files list
        files = []
        if data.images:
            files.extend(data.images)
        if data.documents:
            files.extend(data.documents)

        # Update the message content and files with a single query
        query = (
            update(ORMMessage)
            .where(
                ORMMessage.message_id == message_id,
            )
            .values(
                content=data.text or "",
                files=json.dumps(files) if files else "",
                tool_calls=data.tool_calls,
            )
            .returning(ORMMessage)
            .options(selectinload(ORMMessage.resource_usage))
        )
        updated_message = await self._session.scalar(query)
        if updated_message is None:
            raise ValueError(f"Message {message_id} not found")

        resource_usage = (
            ResourceUsage.model_validate(
                updated_message.resource_usage,
                from_attributes=True,
            )
            if updated_message.resource_usage is not None
            else None
        )
        return Message(
            id=updated_message.id,
            createdAt=updated_message.created_at,
            content=updated_message.content,
            role=Role(updated_message.role),
            chatId=updated_message.chat_id,
            messageId=updated_message.message_id,
            files=json.loads(updated_message.files) if updated_message.files else [],
            toolCalls=updated_message.tool_calls or [],
            resource_usage=resource_usage,
        )

    async def get_next_ai_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> Message:
        """Get the next AI message after a specific message.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID to find the next AI message after.

        Returns:
            Next AI Message object.
        """
        query = (
            select(ORMMessage)
            .where(ORMMessage.message_id == message_id)
            .where(ORMMessage.role == Role.USER)
        )
        user_message = await self._session.scalar(query)
        if user_message is None:
            raise ValueError("Can only get next AI message for user messages")

        query = (
            select(ORMMessage)
            .options(
                selectinload(ORMMessage.resource_usage),
            )
            .where(ORMMessage.chat_id == chat_id)
            .where(ORMMessage.id > user_message.id)
            .where(ORMMessage.role == Role.ASSISTANT)
            .limit(1)
        )
        message = await self._session.scalar(query)
        if not message:
            raise ValueError(
                f"No AI message found after user message ${message_id}."
                "This indicates a data integrity issue.",
            )

        resource_usage = (
            ResourceUsage.model_validate(
                message.resource_usage,
                from_attributes=True,
            )
            if message.resource_usage is not None
            else None
        )
        return Message(
            id=message.id,
            createdAt=message.created_at,
            content=message.content,
            role=Role(message.role),
            chatId=message.chat_id,
            messageId=message.message_id,
            files=json.loads(message.files) if message.files else [],
            toolCalls=message.tool_calls or [],
            resource_usage=resource_usage,
        )

    async def _like_query_search(
        self,
        query: str,
        user_id: str | None = None,
        max_length: int = 150,
        start_sel: str = "<b>",
        stop_sel: str = "</b>",
    ) -> list[FTSResult]:
        """ILIKE fallback for queries shorter than 3 characters."""
        stmt = (
            select(
                ORMMessage,
                ORMChat.title,
                ORMChat.updated_at.label("chat_updated_at"),
            )
            .join(ORMChat, ORMMessage.chat_id == ORMChat.id)
            .where(
                or_(
                    ORMChat.title.ilike(f"%{query}%"),
                    ORMMessage.content.ilike(f"%{query}%"),
                )
            )
            .order_by(ORMChat.updated_at.desc(), ORMMessage.created_at.asc())
        )
        if user_id is not None:
            stmt = stmt.where(ORMChat.user_id == user_id)

        result = await self._session.execute(stmt)
        return [
            FTSResult(
                chat_id=row.Message.chat_id,
                message_id=row.Message.message_id,
                title_snippet=_build_snippet(
                    row.title,
                    query,
                    max_length=max_length,
                    start_sel=start_sel,
                    stop_sel=stop_sel,
                ),
                content_snippet=_build_snippet(
                    row.Message.content,
                    query,
                    max_length=max_length,
                    start_sel=start_sel,
                    stop_sel=stop_sel,
                ),
                msg_created_at=row.Message.created_at,
                chat_updated_at=row.chat_updated_at,
            )
            for row in result
        ]

    async def update_message_resource_usage(
        self,
        message_id: str,
        resource_usage: ResourceUsage,
    ) -> None:
        """Update or create resource usage for a message.

        Args:
            message_id: Unique identifier for the message.
            resource_usage: ResourceUsage data to update or create.
        """
        # Check if resource usage already exists for this message
        query = select(ORMResourceUsage).where(
            ORMResourceUsage.message_id == message_id
        )
        existing = await self._session.scalar(query)

        if existing:
            # Update existing resource usage
            update_query = (
                update(ORMResourceUsage)
                .where(ORMResourceUsage.message_id == message_id)
                .values(
                    model=resource_usage.model,
                    total_input_tokens=resource_usage.total_input_tokens,
                    total_output_tokens=resource_usage.total_output_tokens,
                    user_token=resource_usage.user_token,
                    custom_prompt_token=resource_usage.custom_prompt_token,
                    system_prompt_token=resource_usage.system_prompt_token,
                    time_to_first_token=resource_usage.time_to_first_token,
                    tokens_per_second=resource_usage.tokens_per_second,
                    total_run_time=resource_usage.total_run_time,
                )
            )
            await self._session.execute(update_query)
        else:
            # Create new resource usage
            insert_query = insert(ORMResourceUsage).values(
                message_id=message_id,
                model=resource_usage.model,
                total_input_tokens=resource_usage.total_input_tokens,
                total_output_tokens=resource_usage.total_output_tokens,
                user_token=resource_usage.user_token,
                custom_prompt_token=resource_usage.custom_prompt_token,
                system_prompt_token=resource_usage.system_prompt_token,
                time_to_first_token=resource_usage.time_to_first_token,
                tokens_per_second=resource_usage.tokens_per_second,
                total_run_time=resource_usage.total_run_time,
            )
            await self._session.execute(insert_query)

    async def full_text_search(
        self,
        query: str,
        user_id: str | None = None,
        max_length: int = 150,
        start_sel: str = "<b>",
        stop_sel: str = "</b>",
    ) -> list[FTSResult]:
        """Run full text search on chat titles and message content.

        Args:
            query: Search query string.
            user_id: Optional user ID to filter results.
            max_length: Maximum number of characters in content snippet.
            start_sel: Opening tag for highlighted matches.
            stop_sel: Closing tag for highlighted matches.

        Returns:
            List of FTSResult objects sorted by relevance.
        """
        raise NotImplementedError(
            "The implementation of the method varies on different database."
        )
