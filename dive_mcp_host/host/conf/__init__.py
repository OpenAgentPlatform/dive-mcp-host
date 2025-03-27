from typing import Annotated, Any, Literal

from pydantic import AnyUrl, BaseModel, Field, UrlConstraints

SpecialProvider = Literal["dive", "__load__"]
"""
special providers:
- dive: use the model in dive_mcp_host.models
- __load__: load the model from the configuration
"""


class LLMConfig(BaseModel):
    """Configuration for the LLM model."""

    model: str = "gpt-4o"
    modelProvider: str | SpecialProvider = Field(default="openai")  # noqa: N815
    embed: str | None = None
    embed_dims: int = 0
    apiKey: str | None = None  # noqa: N815
    temperature: float | None = 0
    vector_store: str | None = None
    topP: float | None = None  # noqa: N815
    maxTokens: int | None = None  # noqa: N815
    configuration: dict | None = None

    def model_post_init(self, _: Any) -> None:
        """Set the default embed dimensions for known models."""
        if self.embed and self.embed_dims == 0:
            if self.embed == "text-embedding-3-small":
                self.embed_dims = 1536
            elif self.embed == "text-embedding-3-large":
                self.embed_dims = 3072
            else:
                raise ValueError("invalid dims")

    def to_load_model_kwargs(self) -> dict:
        """Convert the LLM config to kwargs for load_model."""
        if self.configuration:
            kwargs = {
                k: v
                for k, v in self.configuration.items()
                if not hasattr(self, k) and not k.startswith("_")
            }
        else:
            kwargs = {}
        if self.apiKey:
            kwargs["api_key"] = self.apiKey
        if self.temperature:
            kwargs["temperature"] = self.temperature
        if self.topP:
            kwargs["top_p"] = self.topP
        if self.maxTokens:
            kwargs["max_tokens"] = self.maxTokens
        return kwargs


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

    llm: LLMConfig
    checkpointer: CheckpointerConfig | None = None
    mcp_servers: dict[str, ServerConfig]


class AgentConfig(BaseModel):
    """Configuration for an MCP agent."""

    model: str
