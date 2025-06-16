from typing import Literal

from pydantic import BaseModel


class OAPConfig(BaseModel):
    """OAP Config."""

    auth_key: str | None = None
    store_url: str = "https://storage.oaphub.ai"
    oap_root_url: str = "https://oaphub.ai"
    verify_ssl: bool = False


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
