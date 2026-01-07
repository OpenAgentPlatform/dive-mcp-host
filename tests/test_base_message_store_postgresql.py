import asyncio
import uuid
from datetime import UTC, datetime
from os import environ

import pytest
import pytest_asyncio
from alembic import command
from langchain_core.messages import ToolCall
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dive_mcp_host.httpd.database.migrate import db_migration
from dive_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    Message,
    NewMessage,
    ResourceUsage,
    Role,
)
from dive_mcp_host.httpd.database.msg_store.base import BaseMessageStore
from dive_mcp_host.httpd.database.msg_store.postgresql import PostgreSQLMessageStore
from dive_mcp_host.httpd.database.orm_models import (
    Chat as ORMChat,
)
from dive_mcp_host.httpd.database.orm_models import (
    Message as ORMMessage,
)
from dive_mcp_host.httpd.database.orm_models import (
    Users as ORMUsers,
)
from dive_mcp_host.httpd.routers.models import SortBy
from tests.helper import POSTGRES_URI, POSTGRES_URI_ASYNC

# Fixtures for database setup and teardown


@pytest_asyncio.fixture
async def engine():
    """Create an in-memory SQLite database for testing."""
    config = db_migration(POSTGRES_URI)
    engine = create_async_engine(POSTGRES_URI_ASYNC)
    yield engine
    await engine.dispose()
    command.downgrade(config, "base")


@pytest_asyncio.fixture
async def session(engine: AsyncEngine):
    """Create a session for database operations."""
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def message_store(session: AsyncSession):
    """Create a BaseMessageStore instance for testing."""
    return BaseMessageStore(session)


@pytest_asyncio.fixture
async def sample_user(session: AsyncSession):
    """Create a sample user for testing."""
    user_id = "user123"
    user = ORMUsers(id=user_id)
    session.add(user)
    await session.commit()
    return user


@pytest_asyncio.fixture
async def sample_chat(session: AsyncSession, sample_user: ORMUsers):
    """Create a sample chat for testing."""
    chat_id = str(uuid.uuid4())
    title = "Test Chat"

    chat = ORMChat(
        id=chat_id,
        title=title,
        user_id=sample_user.id,
        created_at=datetime.now(UTC),
    )
    session.add(chat)
    await session.commit()

    return chat


@pytest_asyncio.fixture
async def sample_messages(session: AsyncSession, sample_chat: ORMChat):
    """Create sample messages for testing."""
    # User message
    user_msg = ORMMessage(
        message_id=str(uuid.uuid4()),
        chat_id=sample_chat.id,
        role=Role.USER,
        content="Hello, this is a test message",
        created_at=datetime.now(UTC),
        files="",
    )
    session.add(user_msg)

    # Assistant message
    assistant_msg = ORMMessage(
        message_id=str(uuid.uuid4()),
        chat_id=sample_chat.id,
        role=Role.ASSISTANT,
        content="Hello, I am an AI assistant",
        created_at=datetime.now(UTC),
        files="",
        tool_calls=[ToolCall(name="test", args={"test": "test"}, id="test")],
    )
    session.add(assistant_msg)

    await session.commit()

    return {"user_message": user_msg, "assistant_message": assistant_msg}


@pytest_asyncio.fixture
async def sample_chat_no_user(session: AsyncSession):
    """Create a sample chat for testing."""
    chat_id = str(uuid.uuid4())
    title = "Test Chat"

    chat = ORMChat(
        id=chat_id,
        title=title,
        user_id=None,
        created_at=datetime.now(UTC),
    )
    session.add(chat)
    await session.commit()

    return chat


@pytest_asyncio.fixture
async def sample_messages_no_user(session: AsyncSession, sample_chat_no_user: ORMChat):
    """Create sample messages for testing."""
    # User message
    user_msg = ORMMessage(
        message_id=str(uuid.uuid4()),
        chat_id=sample_chat_no_user.id,
        role=Role.USER,
        content="Hello, this is a test message",
        created_at=datetime.now(UTC),
        files="",
    )
    session.add(user_msg)

    # Assistant message
    assistant_msg = ORMMessage(
        message_id=str(uuid.uuid4()),
        chat_id=sample_chat_no_user.id,
        role=Role.ASSISTANT,
        content="Hello, I am an AI assistant",
        created_at=datetime.now(UTC),
        files="",
        tool_calls=[ToolCall(name="test", args={"test": "test"}, id="test")],
    )
    session.add(assistant_msg)

    await session.commit()

    return {"user_message": user_msg, "assistant_message": assistant_msg}


@pytest_asyncio.fixture
async def insert_dummy_data(sample_messages: dict[str, ORMMessage]):
    """Create a sample resource usage for testing."""
    return sample_messages


# Tests for BaseMessageStore methods
@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_get_all_chats(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
    session: AsyncSession,
):
    """Test retrieving all chats for a user."""
    # Create another chat for the same user
    another_chat = ORMChat(
        id=str(uuid.uuid4()),
        title="Another Test Chat",
        user_id=sample_chat.user_id,
        created_at=datetime.now(UTC),
    )
    session.add(another_chat)
    await session.commit()

    # Get all chats
    chats = await message_store.get_all_chats(sample_chat.user_id)

    # Verify results
    assert len(chats) == 2
    assert all(isinstance(chat, Chat) for chat in chats)
    assert chats[0].id == another_chat.id
    assert chats[1].id == sample_chat.id

    # Update create message for sample_chat
    await asyncio.sleep(0.5)
    await message_store.create_message(
        NewMessage(
            messageId=str(uuid.uuid4()),
            chatId=sample_chat.id,
            role=Role.USER,
            content="Hello, this is a test message",
            files=[],
        )
    )
    await session.commit()

    # Get all chats again (sort by message)
    chats = await message_store.get_all_chats(sample_chat.user_id, SortBy.MESSAGE)

    # Verify results
    assert len(chats) == 2
    assert all(isinstance(chat, Chat) for chat in chats)
    assert chats[0].id == sample_chat.id
    assert chats[1].id == another_chat.id

    # Get all chats again (sort by chat)
    chats = await message_store.get_all_chats(sample_chat.user_id, SortBy.CHAT)

    # Verify results
    assert len(chats) == 2
    assert all(isinstance(chat, Chat) for chat in chats)
    assert chats[0].id == another_chat.id
    assert chats[1].id == sample_chat.id


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_get_all_chats_no_user(
    message_store: BaseMessageStore,
    sample_chat_no_user: ORMChat,
    session: AsyncSession,
):
    """Test retrieving all chats for a user."""
    # Create another chat for the same user
    another_chat = ORMChat(
        id=str(uuid.uuid4()),
        title="Another Test Chat",
        user_id=sample_chat_no_user.user_id,
        created_at=datetime.now(UTC),
    )
    session.add(another_chat)
    await session.commit()

    # Get all chats
    chats = await message_store.get_all_chats()

    # Verify results
    assert len(chats) == 2
    assert all(isinstance(chat, Chat) for chat in chats)
    assert any(chat.id == sample_chat_no_user.id for chat in chats)
    assert any(chat.id == another_chat.id for chat in chats)


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_get_chat_with_messages(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
    insert_dummy_data: dict[str, ORMMessage],
):
    """Test retrieving a chat with all its messages."""
    # Get chat with messages
    chat_with_messages = await message_store.get_chat_with_messages(
        sample_chat.id,
        sample_chat.user_id,
    )

    # Verify results
    assert chat_with_messages is not None
    assert isinstance(chat_with_messages, ChatMessage)
    assert chat_with_messages.chat.id == sample_chat.id
    assert len(chat_with_messages.messages) == 2

    # Check message roles
    roles = [msg.role for msg in chat_with_messages.messages]
    assert Role.USER in roles
    assert Role.ASSISTANT in roles


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_get_chat_with_messages_not_found(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
):
    """Test retrieving a non-existent chat."""
    # Get non-existent chat
    chat_with_messages = await message_store.get_chat_with_messages(
        "non_existent_chat_id",
        sample_chat.user_id,
    )

    # Verify results
    assert chat_with_messages is None


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_create_message(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
    session: AsyncSession,
):
    """Test creating a new message."""
    # Create message data
    message_id = str(uuid.uuid4())
    new_message = NewMessage(
        messageId=message_id,
        chatId=sample_chat.id,
        role=Role.USER,
        content="This is a new test message",
        files=[],
    )

    # Create message
    created_message = await message_store.create_message(new_message)

    # Verify results
    assert created_message is not None
    assert isinstance(created_message, Message)
    assert created_message.message_id == message_id
    assert created_message.content == "This is a new test message"
    assert created_message.role == Role.USER
    assert created_message.tool_calls == []

    # Verify message was saved to database
    query = select(ORMMessage).where(ORMMessage.message_id == message_id)
    result = await session.scalar(query)
    assert result is not None
    assert result.content == "This is a new test message"


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_create_message_with_resource_usage(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
):
    """Test creating a new assistant message with resource usage."""
    # Create message data with resource usage
    message_id = str(uuid.uuid4())
    resource_usage = ResourceUsage(
        model="test-model",
        total_input_tokens=10,
        total_output_tokens=20,
        total_run_time=1.5,
    )

    new_message = NewMessage(
        messageId=message_id,
        chatId=sample_chat.id,
        role=Role.ASSISTANT,
        content="This is an assistant message",
        files=[],
        resource_usage=resource_usage,
    )

    # Mock the resource usage insertion since it's not fully implemented in the test
    created_message = await message_store.create_message(new_message)
    assert created_message is not None
    assert created_message.resource_usage is not None
    assert created_message.resource_usage.model == resource_usage.model
    assert (
        created_message.resource_usage.total_input_tokens
        == resource_usage.total_input_tokens
    )
    assert (
        created_message.resource_usage.total_output_tokens
        == resource_usage.total_output_tokens
    )
    assert (
        created_message.resource_usage.total_run_time == resource_usage.total_run_time
    )


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_check_chat_exists(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
):
    """Test checking if a chat exists."""
    # Check existing chat
    exists = await message_store.check_chat_exists(
        sample_chat.id,
        sample_chat.user_id,
    )
    assert exists is True

    # Check non-existent chat
    exists = await message_store.check_chat_exists(
        "non_existent_chat_id",
        sample_chat.user_id,
    )
    assert exists is False


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_delete_chat(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
    session: AsyncSession,
):
    """Test deleting a chat."""
    # Delete chat
    await message_store.delete_chat(sample_chat.id, sample_chat.user_id)

    # Verify chat was deleted
    query = select(ORMChat).where(ORMChat.id == sample_chat.id)
    result = await session.scalar(query)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_delete_messages_after(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
    sample_messages: dict[str, ORMMessage],
    session: AsyncSession,
):
    """Test deleting messages after a specific message."""
    # Add another message
    new_msg = ORMMessage(
        message_id=str(uuid.uuid4()),
        chat_id=sample_chat.id,
        role=Role.USER,
        content="This message should be deleted",
        created_at=datetime.now(UTC),
        files="",
    )
    session.add(new_msg)
    await session.commit()

    # Delete messages after the first user message
    await message_store.delete_messages_after(
        sample_chat.id,
        sample_messages["user_message"].message_id,
    )

    # Verify messages were deleted
    query = select(ORMMessage).where(ORMMessage.chat_id == sample_chat.id)
    results = await session.scalars(query)
    messages = list(results)

    # Only the user message should remain
    assert len(messages) == 1
    assert messages[0].message_id == sample_messages["user_message"].message_id


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_get_next_ai_message(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
    sample_messages: dict[str, ORMMessage],
):
    """Test getting the next AI message after a user message."""
    # Get next AI message
    next_ai_message = await message_store.get_next_ai_message(
        sample_chat.id,
        sample_messages["user_message"].message_id,
    )

    # Verify results
    assert next_ai_message is not None
    assert isinstance(next_ai_message, Message)
    assert next_ai_message.role == Role.ASSISTANT
    assert next_ai_message.message_id == sample_messages["assistant_message"].message_id
    assert next_ai_message.tool_calls == [
        ToolCall(name="test", args={"test": "test"}, id="test")
    ]


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_get_next_ai_message_error(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
    sample_messages: dict[str, ORMMessage],
):
    """Test error when getting next AI message for an assistant message."""
    # Try to get next AI message for an assistant message
    with pytest.raises(
        ValueError,
        match="Can only get next AI message for user messages",
    ):
        await message_store.get_next_ai_message(
            sample_chat.id,
            sample_messages["assistant_message"].message_id,
        )


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_get_next_ai_message_not_found(
    message_store: BaseMessageStore,
    sample_chat: ORMChat,
    session: AsyncSession,
):
    """Test error when no next AI message is found."""
    # Create a user message without a following AI message
    user_msg = ORMMessage(
        message_id=str(uuid.uuid4()),
        chat_id=sample_chat.id,
        role=Role.USER,
        content="Message without response",
        created_at=datetime.now(UTC),
        files="",
    )
    session.add(user_msg)
    await session.commit()

    # Try to get next AI message
    with pytest.raises(ValueError, match="No AI message found after user message"):
        await message_store.get_next_ai_message(
            sample_chat.id,
            user_msg.message_id,
        )


@pytest_asyncio.fixture
async def postgresql_message_store(session: AsyncSession):
    """Create a PostgreSQLMessageStore instance for testing."""
    return PostgreSQLMessageStore(session)


# Tests for SQLiteMessageStore methods


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_create_chat(
    postgresql_message_store: PostgreSQLMessageStore,
    sample_user: ORMUsers,
    session: AsyncSession,
):
    """Test creating a new chat."""
    # Create chat data
    chat_id = str(uuid.uuid4())
    title = "Test SQLite Chat"

    # Create chat
    created_chat = await postgresql_message_store.create_chat(
        chat_id=chat_id,
        title=title,
        user_id=sample_user.id,
    )

    # Verify results
    assert created_chat is not None
    assert isinstance(created_chat, Chat)
    assert created_chat.id == chat_id
    assert created_chat.title == title
    assert created_chat.user_id == sample_user.id

    # Verify chat was saved to database
    query = select(ORMChat).where(ORMChat.id == chat_id)
    result = await session.scalar(query)
    assert result is not None
    assert result.title == title
    assert result.user_id == sample_user.id


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_create_chat_duplicate(
    postgresql_message_store: PostgreSQLMessageStore,
    sample_user: ORMUsers,
    session: AsyncSession,
):
    """Test creating a chat with a duplicate ID."""
    # Create chat data
    chat_id = str(uuid.uuid4())
    title = "Original Chat"

    # Create first chat
    first_chat = await postgresql_message_store.create_chat(
        chat_id=chat_id,
        title=title,
        user_id=sample_user.id,
    )
    assert first_chat is not None

    # Try to create duplicate chat
    duplicate_chat = await postgresql_message_store.create_chat(
        chat_id=chat_id,
        title="Duplicate Chat",
        user_id=sample_user.id,
    )

    # Verify results - should return None for duplicate
    assert duplicate_chat is None

    # Verify original chat still exists with original title
    query = select(ORMChat).where(ORMChat.id == chat_id)
    result = await session.scalar(query)
    assert result is not None
    assert result.title == title  # Title should not be updated


@pytest.mark.asyncio
@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
async def test_patch_chat(postgresql_message_store: PostgreSQLMessageStore):
    """Test if patch works."""
    chat = await postgresql_message_store.create_chat(
        chat_id="dummy_chat_id", title="dummy_title"
    )
    assert chat
    assert chat.id == "dummy_chat_id"
    assert chat.title == "dummy_title"

    new_chat = await postgresql_message_store.patch_chat(chat_id="does not exist")
    assert new_chat is None

    new_chat = await postgresql_message_store.patch_chat(chat_id=chat.id)
    assert new_chat
    assert new_chat.title == chat.title
    assert new_chat.starred_at == chat.starred_at

    new_chat = await postgresql_message_store.patch_chat(
        chat_id=chat.id, title="new_title"
    )
    assert new_chat
    assert new_chat.title == "new_title"
    assert new_chat.starred_at is None

    new_chat = await postgresql_message_store.patch_chat(chat_id=chat.id, star=True)
    assert new_chat
    assert new_chat.title == "new_title"
    assert new_chat.starred_at is not None

    new_chat = await postgresql_message_store.patch_chat(chat_id=chat.id, star=False)
    assert new_chat
    assert new_chat.title == "new_title"
    assert new_chat.starred_at is None
