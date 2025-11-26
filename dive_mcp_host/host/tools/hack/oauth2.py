from http import HTTPStatus

import httpx
from mcp.client.auth import OAuthClientProvider as OrigOAuthClientProvider


class OAuthClientProvider(OrigOAuthClientProvider):
    """An hacked OauthClientProvider.

    Some Auth Provider returns 201 from token api.
    """

    async def _handle_refresh_response(self, response: httpx.Response) -> bool:
        if response.status_code == HTTPStatus.CREATED:
            response.status_code = HTTPStatus.OK
        return await super()._handle_refresh_response(response)

    async def _handle_token_response(self, response: httpx.Response) -> None:
        if response.status_code == HTTPStatus.CREATED:
            response.status_code = HTTPStatus.OK
        return await super()._handle_token_response(response)
