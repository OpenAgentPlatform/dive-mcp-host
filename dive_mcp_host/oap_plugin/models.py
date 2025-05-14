from typing import Literal

from pydantic import BaseModel


class OAPConfig(BaseModel):
    """OAP Config."""

    auth_key: str
    store_url: str
    verify_ssl: bool = True


# /api/v1/user/mcp/configs
class UserMcpConfig(BaseModel):
    """User MCP Config."""

    id: str
    name: str
    description: str
    transport: Literal["sse"]
    url: str
    headers: dict[str, str]

    plan: str


class BaseResponse[T](BaseModel):
    """Base Response."""

    status: Literal["success", "error"]
    error: str | None = None
    data: T | None = None
