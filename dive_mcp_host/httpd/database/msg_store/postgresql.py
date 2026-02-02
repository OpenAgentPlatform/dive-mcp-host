from datetime import UTC, datetime

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert

from dive_mcp_host.httpd.database.models import Chat, FTSResult
from dive_mcp_host.httpd.database.msg_store.base import (
    MIN_TRIGRAM_LENGTH,
    BaseMessageStore,
)
from dive_mcp_host.httpd.database.orm_models import Chat as ORMChat
from dive_mcp_host.httpd.database.orm_models import Message as ORMMessage
from dive_mcp_host.httpd.database.orm_models import Users as ORMUsers


class PostgreSQLMessageStore(BaseMessageStore):
    """Message store for PostgreSQL."""

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
        max_words: int = 60,
        start_sel: str = "<b>",
        stop_sel: str = "</b>",
    ) -> list[FTSResult]:
        """Run full text search using PostgreSQL trigram indexes.

        Args:
            query: Search query string.
            user_id: Optional user ID to filter results.
            max_words: Maximum number of words in content snippet.
            start_sel: Opening tag for highlighted matches.
            stop_sel: Closing tag for highlighted matches.

        Returns:
            List of FTSResult objects sorted by relevance.
        """
        if len(query) < MIN_TRIGRAM_LENGTH:
            return await self._short_query_search(
                query, user_id, max_words, start_sel, stop_sel
            )

        tsquery = func.websearch_to_tsquery("simple", query)
        content_options = (
            f"StartSel={start_sel}, StopSel={stop_sel}, "
            f"MaxWords={max_words // 3}, MinWords=10, MaxFragments=1"
        )
        title_options = f"StartSel={start_sel}, StopSel={stop_sel}"

        stmt = (
            select(
                ORMMessage.message_id,
                ORMMessage.chat_id,
                func.ts_headline("simple", ORMChat.title, tsquery, title_options).label(
                    "title_snippet"
                ),
                func.ts_headline(
                    "simple", ORMMessage.content, tsquery, content_options
                ).label("content_snippet"),
            )
            .join(ORMChat, ORMChat.id == ORMMessage.chat_id)
            .where(
                or_(
                    ORMChat.title.ilike(f"%{query}%"),
                    ORMMessage.content.ilike(f"%{query}%"),
                )
            )
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
            )
            for row in result.mappings()
        ]
