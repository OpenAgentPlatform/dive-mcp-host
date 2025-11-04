import asyncio
import secrets
import time
from base64 import urlsafe_b64encode
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from hashlib import sha256
from logging import getLogger
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP
from pydantic import AnyHttpUrl, BaseModel, Field

EXIPRE = 3600

logger = getLogger(__name__)


host = "127.0.0.1"
port = 8000

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    host = args.host
    port = args.port


class SimpleTokenVerifier(TokenVerifier):
    """Simple token verifier for demonstration."""

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify token."""
        logger.debug("verify_token: %s", token)
        token_info = authorized_token.get(token)
        if token_info and token_info["expire"] > time.time():
            return AccessToken(
                token=token,
                client_id=token_info["client_id"],
                scopes=token_info["scopes"],
            )
        return None


# Create FastMCP instance as a Resource Server
mcp = FastMCP(
    "Weather Service",
    # Token verifier for authentication
    token_verifier=SimpleTokenVerifier(),
    # Auth settings for RFC 9728 Protected Resource Metadata
    auth=AuthSettings(
        # Authorization Server URL
        issuer_url=AnyHttpUrl(f"http://{host}:{port}"),
        resource_server_url=AnyHttpUrl(f"http://{host}:{port}"),  # This server's URL
        required_scopes=["user"],
    ),
    stateless_http=True,
)


@mcp.tool()
async def get_weather(ctx: Context, city: str = "London") -> dict[str, str]:
    """Get weather data for a city."""
    await ctx.report_progress(
        progress=0,
        total=1,
        message="query",
    )
    await asyncio.sleep(1)
    await ctx.report_progress(
        progress=1,
        total=1,
        message="done",
    )
    return {
        "city": city,
        "temperature": "22",
        "condition": "Partly cloudy",
        "humidity": "65%",
    }


@mcp.tool()
async def revoke_token(token_type: Literal["token", "refresh_token", "all"]) -> bool:
    """Revoke token."""

    async def revoke_task() -> None:
        await asyncio.sleep(5)
        if token_type in ["all", "refresh_token"]:
            authorized_refresh_token.clear()
        if token_type in ["all", "token"]:
            authorized_token.clear()

    task = asyncio.create_task(revoke_task())
    task.add_done_callback(
        lambda _: logger.debug(
            {
                "token": authorized_token,
                "refresh_token": authorized_refresh_token,
            }
        )
    )
    return True


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan."""
    with suppress(Exception):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(mcp.session_manager.run())
            yield


app = FastAPI(lifespan=lifespan)

app.mount("/weather", mcp.streamable_http_app())


def get_host(req: Request) -> str:
    """Get host from request."""
    return req.headers["host"]


Host = Annotated[str, Depends(get_host)]

# RFC 8615 Well-Known Uniform Resource Identifiers (URIs)
# https://www.iana.org/assignments/well-known-uris/well-known-uris.xhtml


# RFC 9728 OAuth 2.0 Protected Resource Metadata
@app.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource(host: Host) -> dict[str, Any]:
    """OAuth protected resource metadata."""
    return {
        "resource": f"http://{host}",
        "authorization_servers": [f"http://{host}/auth"],
    }


# RFC 8414 OAuth 2.0 Authorization Server Metadata
#  - RFC 7591
@app.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server(host: Host) -> dict[str, Any]:
    """OAuth authorization server metadata."""
    return {
        "issuer": f"http://{host}",
        "authorization_endpoint": f"http://{host}/auth",
        "token_endpoint": f"http://{host}/token",
        "response_types_supported": ["code"],
        "registration_endpoint": f"http://{host}/register",
        "grant_types_supported": [
            "authorization_code",
            "refresh_token",
        ],
        "code_challenge_methods_supported": ["S256"],
    }


def token_genator() -> str:
    """Token generator."""
    return secrets.token_urlsafe(32)


class APP(BaseModel):
    """OAuth application."""

    client_name: str
    redirect_uris: list[str]
    client_id: UUID = Field(default_factory=uuid4)
    client_secret: str = Field(default_factory=token_genator)


apps: dict[str, APP] = {}


# RFC 7591 OAuth 2.0 Dynamic Client Registration Protocol
@app.post("/register", status_code=201)
async def oauth_register(app: APP) -> APP:
    """OAuth dynamic client registration."""
    apps.update({str(app.client_id): app})
    logger.debug("apps: %s", apps)
    return app


authorized_code = {}


class AuthRequest(BaseModel):
    """OAuth authentication request."""

    response_type: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    resource: str
    scope: str | None = None
    state: str | None = None


# RFC 6749 The OAuth 2.0 Authorization Framework
@app.get("/auth")
async def oauth_auth_redirect(req: Request) -> RedirectResponse:
    """OAuth authentication redirect."""
    return RedirectResponse(f"/auth/web?{req.query_params}")


@app.get("/auth/web")
async def oauth_auth(auth_request: Annotated[AuthRequest, Query()]) -> HTMLResponse:
    """OAuth authentication redirect."""
    logger.debug("auth_request: %s", auth_request)
    app = apps.get(auth_request.client_id)
    if app is None:
        return HTMLResponse(
            f"""<html>
            <body>
            <p>not registerd app</p>
            <p>client_id: {auth_request.client_id}</p>
            </body>
            </html>""",
        )
    code = secrets.token_urlsafe(32)
    authorized_code.update(
        {
            code: auth_request.model_dump(
                mode="json",
                include={
                    "client_id",
                    "code_challenge",
                    "code_challenge_method",
                },
            )
        }
    )
    logger.debug("authorized_code: %s", authorized_code)

    link = auth_request.redirect_uri + f"?code={code}"
    if state := auth_request.state:
        link += f"&state={state}"
    return HTMLResponse(
        f"""<html>
  <head>
    <title>auth page</title>
  </head>
  <body>
    <p>click link to continue</p>
    <a href={link}>{link}</a>
  </body>
</html>""",
        headers={"X-Redirect-Link": link},
    )


authorized_token = {}
authorized_refresh_token = {}


class TokenRequest(BaseModel):
    """OAuth token request."""

    client_id: str
    client_secret: str
    grant_type: str

    code: str | None = None
    redirect_uri: str | None = None
    code_verifier: str | None = None

    refresh_token: str | None = None


# RFC 6749
@app.post("/token")
async def oauth_token(token_request: Annotated[TokenRequest, Form()]) -> JSONResponse:
    """OAuth token endpoint."""
    logger.debug("token_request: %s", token_request)
    app = apps.get(token_request.client_id)
    error = None
    if app is None or app.client_secret != token_request.client_secret:
        error = "invalid_client"
    elif token_request.grant_type == "authorization_code":
        if code := authorized_code.get(token_request.code):
            if code["client_id"] != str(app.client_id):
                error = "invalid_grant"

            challenge = code.get("code_challenge")
            # method = code.get("code_challenge_method")
            if challenge:
                if token_request.code_verifier:
                    hashed = sha256(token_request.code_verifier.encode())
                    b64_bashed = urlsafe_b64encode(hashed.digest()).decode().rstrip("=")
                    if challenge != b64_bashed:
                        error = "invalid_grant"
                else:
                    error = "invalid_grant"
        else:
            error = "invalid_grant"
    elif token_request.grant_type == "refresh_token":
        if authorized_refresh_token.get(token_request.refresh_token) is None:
            error = "invalid_grant"
    else:
        error = "invalid_request"

    if error:
        return JSONResponse({"error": error}, 401)

    assert app
    access_token = f"{app.client_id}.{token_genator()}"
    refresh_token = f"{app.client_id}.{token_genator()}"
    expire = round(time.time() + EXIPRE)

    authorized_token.update(
        {
            access_token: {
                "expire": expire,
                "client_id": str(app.client_id),
                "scopes": ["user"],
            }
        }
    )

    authorized_refresh_token.update({refresh_token: {"access_token": access_token}})

    # prevent replay attack
    if token_request.grant_type == "authorization_code":
        assert token_request.code
        authorized_code.pop(token_request.code, None)
    elif token_request.grant_type == "refresh_token":
        assert token_request.refresh_token
        old_refresh_token = authorized_refresh_token.pop(token_request.refresh_token)
        authorized_token.pop(old_refresh_token["access_token"], None)
    logger.debug("authorized_token: %s", authorized_token)
    logger.debug("authorized_refresh_token: %s", authorized_refresh_token)

    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": EXIPRE,
            "refresh_token": refresh_token,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
