from http import HTTPStatus
from typing import Protocol, runtime_checkable

import httpx
from mcp.client.auth import OAuthClientProvider as OrigOAuthClientProvider
from mcp.shared.auth import OAuthMetadata


@runtime_checkable
class OauthMetadataStore(Protocol):
    """Implement these method in your token store to keep oauth metadata."""

    async def set_oauth_metadata(self, metadata: OAuthMetadata) -> None:
        """Set oauth metadata."""
        ...

    async def get_oauth_metadata(self) -> OAuthMetadata | None:
        """Get oauth metadata."""
        ...


class OAuthClientProvider(OrigOAuthClientProvider):
    """An hacked OauthClientProvider.

    Some Auth Provider returns 201 from token api.
    """

    async def _initialize(self) -> None:
        await super()._initialize()
        if isinstance(self.context.storage, OauthMetadataStore):
            self.context.oauth_metadata = (
                await self.context.storage.get_oauth_metadata()
            )

    async def _handle_refresh_response(self, response: httpx.Response) -> bool:
        if response.status_code == HTTPStatus.CREATED:
            response.status_code = HTTPStatus.OK
        return await super()._handle_refresh_response(response)

    async def _handle_token_response(self, response: httpx.Response) -> None:
        if response.status_code == HTTPStatus.CREATED:
            response.status_code = HTTPStatus.OK
        return await super()._handle_token_response(response)

    async def _perform_authorization(self) -> httpx.Request:
        if self.context.oauth_metadata and isinstance(
            self.context.storage, OauthMetadataStore
        ):
            await self.context.storage.set_oauth_metadata(self.context.oauth_metadata)
        return await super()._perform_authorization()
