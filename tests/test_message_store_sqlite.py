import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from alembic import command
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dive_mcp_host.httpd.database.migrate import db_migration
from dive_mcp_host.httpd.database.models import Role
from dive_mcp_host.httpd.database.msg_store.sqlite import SQLiteMessageStore
from dive_mcp_host.httpd.database.orm_models import Chat as ORMChat
from dive_mcp_host.httpd.database.orm_models import Message as ORMMessage
from tests.helper import SQLITE_URI, SQLITE_URI_ASYNC


@pytest_asyncio.fixture
async def engine():
    """Create an in-memory SQLite database for testing."""
    config = db_migration(SQLITE_URI)
    engine = create_async_engine(SQLITE_URI_ASYNC)
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
async def store(session: AsyncSession):
    """Create a SQLiteMessageStore instance for testing."""
    return SQLiteMessageStore(session)


@pytest_asyncio.fixture
async def sample_data(session: AsyncSession):
    """Create two chats with two messages each for FTS testing."""
    chat1_id = str(uuid.uuid4())
    chat1 = ORMChat(
        id=chat1_id,
        title="Test Chat",
        created_at=datetime.now(UTC),
    )
    session.add(chat1)
    await session.flush()

    msg1_id = str(uuid.uuid4())
    msg1 = ORMMessage(
        message_id=msg1_id,
        chat_id=chat1_id,
        role=Role.USER,
        content="Hello, this is a test message",
        created_at=datetime.now(UTC),
        files="",
    )
    msg2_id = str(uuid.uuid4())
    msg2 = ORMMessage(
        message_id=msg2_id,
        chat_id=chat1_id,
        role=Role.ASSISTANT,
        content="Sure, I can help you with that request",
        created_at=datetime.now(UTC),
        files="",
    )
    session.add_all([msg1, msg2])
    await session.flush()

    chat2_id = str(uuid.uuid4())
    chat2 = ORMChat(
        id=chat2_id,
        title="Science Discussion",
        created_at=datetime.now(UTC),
    )
    session.add(chat2)
    await session.flush()

    msg3_id = str(uuid.uuid4())
    msg3 = ORMMessage(
        message_id=msg3_id,
        chat_id=chat2_id,
        role=Role.USER,
        content="Explain quantum entanglement briefly",
        created_at=datetime.now(UTC),
        files="",
    )
    msg4_id = str(uuid.uuid4())
    msg4 = ORMMessage(
        message_id=msg4_id,
        chat_id=chat2_id,
        role=Role.ASSISTANT,
        content="Quantum entanglement links particles so measuring one affects the other",  # noqa: E501
        created_at=datetime.now(UTC),
        files="",
    )
    session.add_all([msg3, msg4])
    await session.flush()

    return {
        "chat1_id": chat1_id,
        "chat1_msg1_id": msg1_id,
        "chat1_msg2_id": msg2_id,
        "chat2_id": chat2_id,
        "chat2_msg1_id": msg3_id,
        "chat2_msg2_id": msg4_id,
    }


@pytest.mark.asyncio
async def test_full_text_search(
    store: SQLiteMessageStore,
    sample_data: dict[str, str],
):
    """Test FTS on message content including short query fallback."""
    results = await store.full_text_search("test message")
    assert len(results) == 1
    assert results[0].chat_id == sample_data["chat1_id"]
    assert results[0].message_id == sample_data["chat1_msg1_id"]
    assert results[0].title_snippet == "<b>Test</b> Chat"
    assert results[0].content_snippet == "Hello, this is a <b>test</b> <b>message</b>"

    # Short query (< 3 chars) should use ILIKE fallback
    # "He" matches: msg1 ("Hello"), msg2 ("help"), msg4 ("the other")
    results = await store.full_text_search("He")
    assert len(results) == 3
    assert results[0].chat_id == sample_data["chat1_id"]
    assert results[0].message_id == sample_data["chat1_msg1_id"]
    assert results[0].title_snippet == "Test Chat"
    assert results[0].content_snippet == "<b>He</b>llo, this is a test message"
    assert results[1].chat_id == sample_data["chat1_id"]
    assert results[1].message_id == sample_data["chat1_msg2_id"]
    assert results[1].title_snippet == "Test Chat"
    assert results[1].content_snippet == "Sure, I can <b>he</b>lp you with that request"
    assert results[2].chat_id == sample_data["chat2_id"]
    assert results[2].message_id == sample_data["chat2_msg2_id"]
    assert results[2].title_snippet == "Science Discussion"
    assert results[2].content_snippet == (
        "Quantum entanglement links particles so measuring one affects"
        " t<b>he</b> ot<b>he</b>r"
    )


@pytest.mark.asyncio
async def test_full_text_search_no_result(
    store: SQLiteMessageStore,
    sample_data: dict[str, str],
):
    """Test FTS with no matches."""
    results = await store.full_text_search("nonexistentxyzzy")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_full_text_search_by_title(
    store: SQLiteMessageStore,
    session: AsyncSession,
):
    """Test that searching by chat title returns results."""
    chat = ORMChat(
        id="fts-title-test",
        title="xyzzyplugh",
        created_at=datetime.now(UTC),
    )
    session.add(chat)
    await session.flush()

    msg = ORMMessage(
        message_id="fts-title-msg-1",
        chat_id="fts-title-test",
        role=Role.USER,
        content="some ordinary content here",
        created_at=datetime.now(UTC),
        files="",
    )
    session.add(msg)
    await session.flush()

    results = await store.full_text_search("xyzzyplugh")
    assert len(results) == 1
    assert results[0].chat_id == "fts-title-test"
    assert results[0].title_snippet == "<b>xyzzyplugh</b>"


@pytest.mark.asyncio
async def test_sqlite_fts_message_triggers(
    store: SQLiteMessageStore,
    session: AsyncSession,
):
    """Test INSERT/UPDATE/DELETE on messages keeps FTS in sync."""
    # INSERT
    chat = ORMChat(
        id="fts-trig",
        title="trigger test",
        created_at=datetime.now(UTC),
    )
    session.add(chat)
    await session.flush()

    msg = ORMMessage(
        message_id="fts-trig-msg-1",
        chat_id="fts-trig",
        role=Role.USER,
        content="unique trigger content" + " asdf" * 100,
        created_at=datetime.now(UTC),
        files="",
    )
    session.add(msg)
    await session.flush()

    results = await store.full_text_search("unique trigger content")
    assert len(results) == 1
    assert results[0].chat_id == "fts-trig"
    assert results[0].message_id == "fts-trig-msg-1"

    # UPDATE
    await session.execute(
        update(ORMMessage)
        .where(ORMMessage.message_id == "fts-trig-msg-1")
        .values(content="modified trigger content")
    )
    await session.flush()

    results = await store.full_text_search("unique trigger content")
    assert len(results) == 0

    results = await store.full_text_search("modified trigger content")
    assert len(results) == 1
    assert results[0].chat_id == "fts-trig"

    # DELETE
    await session.execute(
        delete(ORMMessage).where(ORMMessage.message_id == "fts-trig-msg-1")
    )
    await session.flush()

    results = await store.full_text_search("modified trigger content")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_sqlite_fts_chat_title_update_trigger(
    store: SQLiteMessageStore,
    session: AsyncSession,
):
    """Updating chat title re-syncs FTS entries."""
    chat = ORMChat(
        id="fts-title-upd",
        title="foobarbaz",
        created_at=datetime.now(UTC),
    )
    session.add(chat)
    await session.flush()

    msg = ORMMessage(
        message_id="fts-title-upd-msg-1",
        chat_id="fts-title-upd",
        role=Role.USER,
        content="some content for title trigger",
        created_at=datetime.now(UTC),
        files="",
    )
    session.add(msg)
    await session.flush()

    results = await store.full_text_search("foobarbaz")
    assert len(results) == 1

    await session.execute(
        update(ORMChat)
        .where(ORMChat.id == "fts-title-upd")
        .values(title="quxquuxcorge")
    )
    await session.flush()

    results = await store.full_text_search("foobarbaz")
    assert len(results) == 0

    results = await store.full_text_search("quxquuxcorge")
    assert len(results) == 1
    assert results[0].chat_id == "fts-title-upd"


@pytest.mark.asyncio
async def test_sqlite_fts_chat_delete_trigger(
    store: SQLiteMessageStore,
    session: AsyncSession,
):
    """Deleting chat cleans up FTS entries."""
    chat = ORMChat(
        id="fts-chat-del",
        title="deletablechat",
        created_at=datetime.now(UTC),
    )
    session.add(chat)
    await session.flush()

    msg = ORMMessage(
        message_id="fts-chat-del-msg-1",
        chat_id="fts-chat-del",
        role=Role.USER,
        content="ephemeral stuff here",
        created_at=datetime.now(UTC),
        files="",
    )
    session.add(msg)
    await session.flush()

    results = await store.full_text_search("ephemeral stuff here")
    assert len(results) == 1

    await session.execute(delete(ORMChat).where(ORMChat.id == "fts-chat-del"))
    await session.flush()

    results = await store.full_text_search("ephemeral stuff here")
    assert len(results) == 0
