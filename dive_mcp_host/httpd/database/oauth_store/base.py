from datetime import UTC, datetime, timedelta

from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dive_mcp_host.host.tools.oauth import BaseTokenStore, TokenStore
from dive_mcp_host.httpd.database.orm_models import OAuth


class BaseOAuthtokenStore(BaseTokenStore):
    """Base OAuth token store."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], user_id: str = "default"
    ) -> None:
        """Initialize the OAuth token store."""
        self._session_maker = session_maker
        self._user_id = user_id

    async def _update(self, name: str, store: TokenStore) -> None:
        """Update the OAuth token store for a given name."""
        async with self._session_maker() as session:
            query = select(OAuth).where(
                OAuth.name == name, OAuth.user_id == self._user_id
            )
            oauth = await session.scalar(query)

            op = insert(OAuth).values(
                user_id=self._user_id,
                name=name,
            )
            if oauth:
                op = update(OAuth).where(
                    OAuth.name == name, OAuth.user_id == self._user_id
                )

            query = op.values(
                access_token=store.tokens.access_token if store.tokens else None,
                refresh_token=store.tokens.refresh_token if store.tokens else None,
                expire=datetime.now(UTC) + timedelta(seconds=store.tokens.expires_in)
                if store.tokens and store.tokens.expires_in
                else None,
                scope=store.tokens.scope if store.tokens else None,
                client_id=store.client_info.client_id if store.client_info else None,
                client_secret=store.client_info.client_secret
                if store.client_info
                else None,
                client_info=store.client_info.model_dump(by_alias=True, mode="json")
                if store.client_info
                else None,
                oauth_metadata=store.oauth_metadata.model_dump(
                    by_alias=True, mode="json"
                )
                if store.oauth_metadata
                else None,
                token_expiry_time=store.token_expiry_time,
            )
            await session.execute(query)
            await session.commit()

    async def get(self, name: str) -> TokenStore:
        """Get the OAuth token store for a given name."""
        async with self._session_maker() as session:
            query = select(OAuth).where(
                OAuth.name == name, OAuth.user_id == self._user_id
            )
            oauth = await session.scalar(query)

            token_store = TokenStore()
            if oauth and oauth.access_token:
                token_store.tokens = OAuthToken(
                    access_token=oauth.access_token,
                    refresh_token=oauth.refresh_token,
                    expires_in=(
                        oauth.expire.replace(tzinfo=UTC) - datetime.now(UTC)
                    ).seconds
                    if oauth.expire
                    else None,
                    scope=oauth.scope,
                )
            if oauth and oauth.client_info:
                token_store.client_info = OAuthClientInformationFull.model_validate(
                    oauth.client_info
                )
            if oauth and oauth.oauth_metadata:
                token_store.oauth_metadata = OAuthMetadata.model_validate(
                    oauth.oauth_metadata
                )
            if oauth and oauth.token_expiry_time:
                token_store.token_expiry_time = oauth.token_expiry_time

            token_store.update_method = lambda s: self._update(name, s)
            return token_store

    async def delete(self, name: str) -> None:
        """Delete the OAuth token store for a given name."""
        async with self._session_maker() as session:
            query = delete(OAuth).where(
                OAuth.name == name, OAuth.user_id == self._user_id
            )
            await session.execute(query)
            await session.commit()

    async def list(self) -> list[str]:
        """List the OAuth token stores."""
        async with self._session_maker() as session:
            query = select(OAuth.name).where(OAuth.user_id == self._user_id)
            result = await session.scalars(query)
            return list(result.all())
