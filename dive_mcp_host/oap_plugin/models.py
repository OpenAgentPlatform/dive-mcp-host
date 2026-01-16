from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field


class OAPConfig(BaseModel):
    """OAP Config."""

    auth_key: str | None = None
    store_url: str = "https://storage.oaphub.ai"
    oap_root_url: str = "https://oaphub.ai"
    verify_ssl: bool = False


class ExternalEndpointCMD(BaseModel):
    """External Endpoint of local installed MCPs."""

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    protocol: str = "stdio"


class ExternalEndpointHttp(BaseModel):
    """External Endpoint of Third Party MCPs."""

    url: str
    headers: Annotated[
        dict[str, str],
        BeforeValidator(lambda t: t if t else {}),
    ] = Field(default_factory=dict)
    protocol: Literal["sse", "streamable"]


type ExternalEndpoint = ExternalEndpointCMD | ExternalEndpointHttp | dict


# /api/v1/user/mcp/configs
class UserMcpConfig(BaseModel):
    """User MCP Config."""

    id: str
    name: str
    description: str
    transport: Literal["sse", "streamable"]
    auth_type: Literal["header", "oauth2"] = "header"
    url: str
    headers: dict[str, str]
    external_endpoint: ExternalEndpoint | None = None

    plan: str


class BaseResponse[T](BaseModel):
    """Base Response."""

    status: Literal["success", "error"]
    error: str | None = None
    data: T | None = None


class TokenNotSetError(Exception):
    """Token not set error."""

    message: str = "Token is not set"

    def __str__(self) -> str:
        """Return the error message."""
        return self.message
