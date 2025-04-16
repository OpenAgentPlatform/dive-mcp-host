from typing import Annotated, Literal

from pydantic import AnyUrl, BaseModel, Field, UrlConstraints

from dive_mcp_host.host.conf.llm import LLMConfigTypes
from dive_mcp_host.httpd.routers.models import EmbedConfig


class CheckpointerConfig(BaseModel):
    """Configuration for the checkpointer."""

    # more parameters in the future. like pool size, etc.
    uri: Annotated[
        AnyUrl,
        UrlConstraints(allowed_schemes=["sqlite", "postgres", "postgresql"]),
    ]


class ServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    exclude_tools: list[str] = Field(default_factory=list)
    url: str | None = None
    keep_alive: float | None = None
    transport: Literal["stdio", "sse", "websocket"]


class HostConfig(BaseModel):
    """Configuration for the MCP host."""

    llm: LLMConfigTypes
    embed: EmbedConfig | None = None
    checkpointer: CheckpointerConfig | None = None
    mcp_servers: dict[str, ServerConfig]


class AgentConfig(BaseModel):
    """Configuration for an MCP agent."""

    model: str
