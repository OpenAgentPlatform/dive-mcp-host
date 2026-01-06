import time
from http import HTTPStatus
from typing import Protocol, runtime_checkable

import httpx
from mcp.client.auth import OAuthClientProvider as OrigOAuthClientProvider
from mcp.client.auth.oauth2 import OAuthContext
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


def _patch_is_token_valid() -> None:
    """Patch token validation to refresh 30 seconds before expiry.

    MCP SDK doesn't trigger token refresh if access token expires during a request.
    This patch proactively refreshes tokens before they expire to avoid this issue.
    """
    org_method = OAuthContext.is_token_valid

    def p(self: OAuthContext) -> bool:
        if self.token_expiry_time and time.time() + 30 > self.token_expiry_time:
            self.token_expiry_time = 1
        return org_method(self)

    OAuthContext.is_token_valid = p


_patch_is_token_valid()


class OAuthClientProvider(OrigOAuthClientProvider):
    """An hacked OauthClientProvider.

    Some Auth Provider returns 201 from token api.
    """

    async def _initialize(self) -> None:
        """Initialize OAuth client with token expiry and metadata fixes.

        - Sets token_expiry_time if missing (workaround for pending upstream PR #1784)
        - Restores oauth_metadata from storage if available
        """
        await super()._initialize()
        # Set token expiry time if not already set
        if (
            self.context.current_tokens
            and self.context.token_expiry_time is None
            and self.context.current_tokens.expires_in is not None
        ):
            self.context.update_token_expiry(self.context.current_tokens)
        # Restore oauth metadata from storage
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
