import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    Field,
    SecretStr,
    UrlConstraints,
    field_serializer,
)

from dive_mcp_host.host.conf.llm import LLMConfigTypes


class CheckpointerConfig(BaseModel):
    """Configuration for the checkpointer."""

    # more parameters in the future. like pool size, etc.
    uri: Annotated[
        AnyUrl,
        UrlConstraints(allowed_schemes=["sqlite", "postgres", "postgresql"]),
    ]


class ProxyUrl(AnyUrl):
    """Proxy URL with protocol validation.

    Only support http and socks5.
    """

    _constraints = UrlConstraints(max_length=1024, allowed_schemes=["http", "socks5"])


def _rewrite_socks(v: Any) -> Any:
    if isinstance(v, str):
        return re.sub(r"^socks4?://", "socks5://", v)
    return v


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
    transport: Literal["stdio", "sse", "streamable", "websocket"]
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    proxy: Annotated[
        ProxyUrl | None,
        BeforeValidator(_rewrite_socks),
    ] = None
    initial_timeout: float = 10

    @field_serializer("headers", when_used="json")
    def dump_headers(self, v: dict[str, SecretStr] | None) -> dict[str, str] | None:
        """Serialize the headers field to plain text."""
        return {k: v.get_secret_value() for k, v in v.items()} if v else None


class LogConfig(BaseModel):
    """Config for mcp server logs.

    Attributes:
        log_dir: base directory for log files.
        rotation_files: max log rotation files per mcp server.
        buffer_length: the amount of log entries in log buffer.
    """

    log_dir: Path = Field(default_factory=lambda: Path.cwd() / "logs")
    rotation_files: int = 5
    buffer_length: int = 1000


class EmbedConfig(BaseModel):
    """Config for embedding model."""

    provider: str | None = None
    model: str | None = None
    embed_dims: int | None = None
    api_key: str | None = None


class HostConfig(BaseModel):
    """Configuration for the MCP host."""

    llm: LLMConfigTypes
    embed: EmbedConfig | None = None
    checkpointer: CheckpointerConfig | None = None
    mcp_servers: dict[str, ServerConfig]
    log_config: LogConfig = Field(default_factory=LogConfig)


class AgentConfig(BaseModel):
    """Configuration for an MCP agent."""

    model: str
