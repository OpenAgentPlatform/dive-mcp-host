from datetime import UTC, datetime

from sqlalchemy import column, func, literal_column, select, table
from sqlalchemy.dialects.sqlite import insert

from dive_mcp_host.httpd.database.models import Chat, FTSResult
from dive_mcp_host.httpd.database.msg_store.base import (
    MIN_TRIGRAM_LENGTH,
    BaseMessageStore,
)
from dive_mcp_host.httpd.database.orm_models import Chat as ORMChat
from dive_mcp_host.httpd.database.orm_models import Message as ORMMessage
from dive_mcp_host.httpd.database.orm_models import Users as ORMUsers


class SQLiteMessageStore(BaseMessageStore):
    """Message store for SQLite."""

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
        if user_id is not None:
            query = (
                insert(ORMUsers)
                .values(
                    {
                        "id": user_id,
                        "user_type": user_type,
                    }
                )
                .on_conflict_do_nothing()
            )
            await self._session.execute(query)

        current_ts = datetime.now(UTC)
        query = (
            insert(ORMChat)
            .values(
                {
                    "id": chat_id,
                    "title": title,
                    "created_at": current_ts,
                    "updated_at": current_ts,
                    "user_id": user_id,
                },
            )
            .on_conflict_do_nothing()
            .returning(ORMChat)
        )
        chat = await self._session.scalar(query)
        if chat is None:
            return None
        return Chat(
            id=chat.id,
            title=chat.title,
            createdAt=chat.created_at,
            updatedAt=chat.updated_at,
            starredAt=chat.starred_at,
            user_id=chat.user_id,
        )

    async def full_text_search(
        self,
        query: str,
        user_id: str | None = None,
        max_length: int = 150,
        start_sel: str = "<b>",
        stop_sel: str = "</b>",
    ) -> list[FTSResult]:
        """Run full text search using SQLite FTS5.

        Args:
            query: Search query string.
            user_id: Optional user ID to filter results.
            max_length: Maximum number of characters in content snippet.
            start_sel: Opening tag for highlighted matches.
            stop_sel: Closing tag for highlighted matches.

        Returns:
            List of FTSResult objects sorted by relevance.
        """
        if len(query) < MIN_TRIGRAM_LENGTH:
            return await self._like_query_search(
                query, user_id, max_length, start_sel, stop_sel
            )

        fts = table(
            "message_fts",
            column("rowid"),
            column("rank"),
            column("chat_id"),
        )

        stmt = (
            select(
                ORMMessage.message_id,
                ORMMessage.chat_id,
                ORMMessage.created_at,
                func.snippet(
                    literal_column("message_fts"),
                    1,
                    start_sel,
                    stop_sel,
                    "...",
                    max_length,
                ).label("title_snippet"),
                func.snippet(
                    literal_column("message_fts"),
                    2,
                    start_sel,
                    stop_sel,
                    "...",
                    max_length,
                ).label("content_snippet"),
            )
            .select_from(fts)
            .join(
                ORMMessage,
                literal_column("messages.rowid") == fts.c.rowid,
            )
            .join(
                ORMChat,
                ORMChat.id == ORMMessage.chat_id,
            )
            .where(literal_column("message_fts").match(query))
            .order_by(ORMMessage.created_at.asc())
        )

        if user_id is not None:
            stmt = stmt.where(ORMChat.user_id == user_id)

        result = await self._session.execute(stmt)
        return [
            FTSResult(
                chat_id=row.chat_id,
                message_id=row.message_id,
                title_snippet=row.title_snippet,
                content_snippet=row.content_snippet,
                msg_created_at=row.created_at,
            )
            for row in result.mappings()
        ]
