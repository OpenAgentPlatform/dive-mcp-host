from enum import StrEnum
from typing import Annotated, Any, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    model_validator,
)
from pydantic.alias_generators import to_camel

from dive_mcp_host.host.conf.llm import (
    LLMConfigTypes,
    LLMConfiguration,
    get_llm_config_type,
)

T = TypeVar("T")


class ResultResponse(BaseModel):
    """Generic response model with success status and message."""

    success: bool
    message: str | None = None


Transport = Literal["stdio", "sse", "websocket"]


class McpServerConfig(BaseModel):
    """MCP server configuration with transport and connection settings."""

    transport: Annotated[
        Transport, BeforeValidator(lambda v: "stdio" if v == "command" else v)
    ]
    enabled: bool | None
    command: str | None = None
    args: list[str] | None = Field(default_factory=list)
    env: dict[str, str] | None = Field(default_factory=dict)
    url: str | None = None

    def model_post_init(self, _: Any) -> None:
        """Post-initialization hook."""
        if self.transport in ["sse", "websocket"]:
            if self.url is None:
                raise ValueError("url is required for sse and websocket transport")
        elif self.transport == "stdio" and self.command is None:
            raise ValueError("command is required for stdio transport")


class McpServers(BaseModel):
    """Collection of MCP server configurations."""

    mcp_servers: dict[str, McpServerConfig] = Field(
        alias="mcpServers", default_factory=dict
    )


class McpServerError(BaseModel):
    """Represents an error from an MCP server."""

    server_name: str = Field(alias="serverName")
    error: Any  # any


class ModelType(StrEnum):
    """Model type."""

    OLLAMA = "ollama"
    MISTRAL = "mistralai"
    BEDROCK = "bedrock"
    DEEPSEEK = "deepseek"
    OTHER = "other"

    @classmethod
    def get_model_type(cls, llm_config: LLMConfigTypes) -> "ModelType":
        """Get model type from model name."""
        # Direct mapping for known providers
        try:
            return cls(llm_config.model_provider)
        except ValueError:
            pass
        # Special case for deepseek
        if "deepseek" in llm_config.model.lower():
            return cls.DEEPSEEK

        return cls.OTHER


class ModelSettingsProperty(BaseModel):
    """Defines a property for model settings with type information and metadata."""

    type: Literal["string", "number"]
    description: str
    required: bool
    default: Any | None = None
    placeholder: Any | None = None


class ModelSettingsDefinition(ModelSettingsProperty):
    """Model settings definition with nested properties."""

    type: Literal["string", "number", "object"]
    properties: dict[str, ModelSettingsProperty] | None = None


class ModelInterfaceDefinition(BaseModel):
    """Defines the interface for model settings."""

    model_settings: dict[str, ModelSettingsDefinition]


class SimpleToolInfo(BaseModel):
    """Represents an MCP tool with its properties and metadata."""

    name: str
    description: str


class McpTool(BaseModel):
    """Represents an MCP tool with its properties and metadata."""

    name: str
    tools: list[SimpleToolInfo]
    description: str
    enabled: bool
    icon: str
    error: str | None = None


class ToolsCache(RootModel[dict[str, McpTool]]):
    """Tools cache."""

    root: dict[str, McpTool]


class ToolCallsContent(BaseModel):
    """Tool call content."""

    name: str
    arguments: Any


class ToolResultContent(BaseModel):
    """Tool result content."""

    name: str
    result: Any


class ChatInfoContent(BaseModel):
    """Chat info."""

    id: str
    title: str


class MessageInfoContent(BaseModel):
    """Message info."""

    user_message_id: str = Field(alias="userMessageId")
    assistant_message_id: str = Field(alias="assistantMessageId")


class StreamMessage(BaseModel):
    """Stream message."""

    type: Literal[
        "text", "tool_calls", "tool_result", "error", "chat_info", "message_info"
    ]
    content: (
        str
        | list[ToolCallsContent]
        | ToolResultContent
        | ChatInfoContent
        | MessageInfoContent
    )


class TokenUsage(BaseModel):
    """Token usage."""

    total_input_tokens: int = Field(default=0, alias="totalInputTokens")
    total_output_tokens: int = Field(default=0, alias="totalOutputTokens")
    total_tokens: int = Field(default=0, alias="totalTokens")


class ModelSingleConfig(BaseModel):
    """Model single config."""

    model_provider: str
    model: str
    max_tokens: int | None = None
    api_key: str | None = None
    configuration: LLMConfiguration | None = None
    active: bool = Field(default=True)
    checked: bool = Field(default=False)

    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
        extra="allow",
    )

    @model_validator(mode="after")
    def post_validate(self) -> Self:
        """Validate the model config by converting to LLMConfigTypes."""
        get_llm_config_type(self.model_provider).model_validate(self.model_dump())
        return self


class ModelFullConfigs(BaseModel):
    """Configuration for the model."""

    active_provider: str
    enable_tools: bool
    configs: dict[str, ModelSingleConfig] = Field(default_factory=dict)

    disable_dive_system_prompt: bool = False
    # If True, custom rules will be used directly without extra system prompt from Dive.

    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )


class UserInputError(Exception):
    """User input error."""


class SortBy(StrEnum):
    """Sort by."""

    CHAT = "chat"
    MESSAGE = "msg"
