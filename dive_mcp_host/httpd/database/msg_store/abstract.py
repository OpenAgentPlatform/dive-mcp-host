# new abstraction for database

from abc import ABC, abstractmethod

from dive_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    FTSResult,
    Message,
    NewMessage,
    QueryInput,
    ResourceUsage,
)


class AbstractMessageStore(ABC):
    """Abstract base class for database operations."""

    @abstractmethod
    async def get_all_chats(
        self,
        user_id: str | None = None,
    ) -> list[Chat]:
        """Retrieve all chats from the database.

        Args:
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            List of Chat objects.
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    async def create_message(
        self,
        message: NewMessage,
    ) -> Message:
        """Create a new message.

        Args:
            message: NewMessage object containing message data.
            user_id: User ID or fingerprint, depending on the prefix.

        Returns:
            Created Message object.
        """

    @abstractmethod
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

    @abstractmethod
    async def delete_chat(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> None:
        """Delete a chat from the database.

        Args:
            chat_id: Unique identifier for the chat.
            user_id: User ID or fingerprint, depending on the prefix.
        """

    @abstractmethod
    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete all messages after a specific message in a chat.

        Args:
            chat_id: Unique identifier for the chat.
            message_id: Message ID after which all messages will be deleted.
            user_id: User ID or fingerprint, depending on the prefix.
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    async def update_message_resource_usage(
        self,
        message_id: str,
        resource_usage: "ResourceUsage",
    ) -> None:
        """Update or create resource usage for a message.

        Args:
            message_id: Unique identifier for the message.
            resource_usage: ResourceUsage data to update or create.
        """

    @abstractmethod
    async def full_text_search(
        self,
        query: str,
        user_id: str | None = None,
        max_words: int = 60,
        start_sel: str = "<b>",
        stop_sel: str = "</b>",
    ) -> list[FTSResult]:
        """Run full text search on chat titles and message content.

        Args:
            query: Search query string.
            user_id: Optional user ID to filter results.
            max_words: Maximum number of words in content snippet.
            start_sel: Opening tag for highlighted matches.
            stop_sel: Closing tag for highlighted matches.

        Returns:
            List of FTSResult objects sorted by relevance.
        """
