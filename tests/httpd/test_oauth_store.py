"""Tests for BaseOAuthtokenStore oauth_metadata persistence."""

import pytest
import pytest_asyncio
from alembic import command
from mcp.shared.auth import OAuthMetadata
from pydantic import AnyHttpUrl
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dive_mcp_host.httpd.database.migrate import db_migration
from dive_mcp_host.httpd.database.oauth_store.base import BaseOAuthtokenStore
from dive_mcp_host.httpd.database.orm_models import Users
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
async def session_maker(engine: AsyncEngine):
    """Create a session maker for database operations."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture
async def sample_user(session_maker: async_sessionmaker[AsyncSession]):
    """Create a sample user for testing."""
    user_id = "test_user"
    async with session_maker() as session:
        query = (
            insert(Users).values(id=user_id).on_conflict_do_nothing().returning(Users)
        )
        await session.execute(query)
        await session.commit()

        query = select(Users).where(Users.id == user_id)
        return await session.scalar(query)


@pytest_asyncio.fixture
async def oauth_store(
    session_maker: async_sessionmaker[AsyncSession],
    sample_user: Users,
):
    """Create a BaseOAuthtokenStore instance for testing."""
    return BaseOAuthtokenStore(session_maker, user_id=sample_user.id)


@pytest.mark.asyncio
async def test_store_and_retrieve_oauth_metadata(oauth_store: BaseOAuthtokenStore):
    """Test that oauth_metadata is properly stored and retrieved."""
    server_name = "test_server"

    # Create test OAuth metadata
    oauth_metadata = OAuthMetadata(
        issuer=AnyHttpUrl("https://example.com"),
        authorization_endpoint=AnyHttpUrl("https://example.com/oauth/authorize"),
        token_endpoint=AnyHttpUrl("https://example.com/oauth/token"),
        registration_endpoint=AnyHttpUrl("https://example.com/oauth/register"),
        scopes_supported=["read", "write"],
        response_types_supported=["code"],
        grant_types_supported=["authorization_code", "refresh_token"],
    )

    # Get the token store and set oauth_metadata using the setter method
    token_store = await oauth_store.get(server_name)
    await token_store.set_oauth_metadata(oauth_metadata)

    # Retrieve the token store and verify oauth_metadata
    retrieved_store = await oauth_store.get(server_name)
    assert retrieved_store.oauth_metadata is not None
    assert retrieved_store.oauth_metadata.issuer == oauth_metadata.issuer
    assert (
        retrieved_store.oauth_metadata.authorization_endpoint
        == oauth_metadata.authorization_endpoint
    )
    assert (
        retrieved_store.oauth_metadata.token_endpoint == oauth_metadata.token_endpoint
    )
    assert (
        retrieved_store.oauth_metadata.registration_endpoint
        == oauth_metadata.registration_endpoint
    )
    assert (
        retrieved_store.oauth_metadata.scopes_supported
        == oauth_metadata.scopes_supported
    )


@pytest.mark.asyncio
async def test_oauth_metadata_none_when_not_set(oauth_store: BaseOAuthtokenStore):
    """Test that oauth_metadata is None when not set."""
    server_name = "no_metadata_server"

    # Get store without setting oauth_metadata
    token_store = await oauth_store.get(server_name)

    # oauth_metadata should be None
    assert token_store.oauth_metadata is None


@pytest.mark.asyncio
async def test_delete_clears_oauth_metadata(oauth_store: BaseOAuthtokenStore):
    """Test that delete removes oauth_metadata along with other data."""
    server_name = "delete_test_server"

    # Store oauth_metadata
    token_store = await oauth_store.get(server_name)
    await token_store.set_oauth_metadata(
        OAuthMetadata(
            issuer=AnyHttpUrl("https://example.com"),
            authorization_endpoint=AnyHttpUrl("https://example.com/authorize"),
            token_endpoint=AnyHttpUrl("https://example.com/token"),
        )
    )

    # Verify it was stored
    retrieved_store = await oauth_store.get(server_name)
    assert retrieved_store.oauth_metadata is not None

    # Delete the store
    await oauth_store.delete(server_name)

    # Verify it's gone - new get should return empty store
    new_store = await oauth_store.get(server_name)
    assert new_store.oauth_metadata is None
