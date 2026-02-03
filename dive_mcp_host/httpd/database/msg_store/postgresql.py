from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert

from dive_mcp_host.httpd.database.models import Chat, FTSResult
from dive_mcp_host.httpd.database.msg_store.base import BaseMessageStore
from dive_mcp_host.httpd.database.orm_models import Chat as ORMChat
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
        max_length: int = 150,
        start_sel: str = "<b>",
        stop_sel: str = "</b>",
    ) -> list[FTSResult]:
        """Run full text search using ILIKE and Python-based snippets.

        Args:
            query: Search query string.
            user_id: Optional user ID to filter results.
            max_length: Maximum number of characters in content snippet.
            start_sel: Opening tag for highlighted matches.
            stop_sel: Closing tag for highlighted matches.

        Returns:
            List of FTSResult objects sorted by created_at ascending.
        """
        return await self._like_query_search(
            query, user_id, max_length, start_sel, stop_sel
        )
