from enum import StrEnum
from typing import Any, Literal, Self, TypeVar

from mcp.types import Icon
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SecretStr,
    field_serializer,
    model_validator,
)
from pydantic.alias_generators import to_camel

from dive_mcp_host.host.conf import EmbedConfig
from dive_mcp_host.host.conf.llm import (
    LLMConfigTypes,
    LLMConfiguration,
    get_llm_config_type,
)
from dive_mcp_host.host.custom_events import ToolCallProgress

T = TypeVar("T")


class ResultResponse(BaseModel):
    """Generic response model with success status and message."""

    success: bool
    message: str | None = None


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

    type: Literal["string", "number", "object"]  # type: ignore
    properties: dict[str, ModelSettingsProperty] | None = None


class ModelInterfaceDefinition(BaseModel):
    """Defines the interface for model settings."""

    model_settings: dict[str, ModelSettingsDefinition]


class SimpleToolInfo(BaseModel):
    """Represents an MCP tool with its properties and metadata."""

    name: str
    description: str
    enabled: bool = True
    icons: list[Icon] | None = None


class McpTool(BaseModel):
    """Represents an MCP tool with its properties and metadata."""

    name: str
    tools: list[SimpleToolInfo]
    description: str
    enabled: bool
    icon: str
    status: str
    url: str | None = None
    status: str
    error: str | None = None
    icons: list[Icon] | None = None
    has_credential: bool = False


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


class AgentToolCallContent(BaseModel):
    """Agent (sub-agent) tool call content."""

    model_config = ConfigDict(populate_by_name=True)

    tool_call_id: str = Field(alias="toolCallId")
    name: str
    args: Any


class AgentToolResultContent(BaseModel):
    """Agent (sub-agent) tool result content."""

    model_config = ConfigDict(populate_by_name=True)

    tool_call_id: str = Field(alias="toolCallId")
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


class AuthenticationRequiredContent(BaseModel):
    """Authentication required content."""

    server_name: str
    auth_url: str


class ElicitationRequestContent(BaseModel):
    """Elicitation request content from MCP server."""

    request_id: str
    message: str
    requested_schema: dict


class InteractiveContent(BaseModel):
    """Interactive content."""

    type: Literal["authentication_required", "elicitation_request"]
    content: AuthenticationRequiredContent | ElicitationRequestContent


class ErrorContent(BaseModel):
    """Error content."""

    message: str
    type: str

    model_config = ConfigDict(
        extra="allow",
    )


class TokenUsageContent(BaseModel):
    """Token usage content for streaming."""

    input_tokens: int = Field(default=0, alias="inputTokens")
    output_tokens: int = Field(default=0, alias="outputTokens")
    user_token: int = Field(default=0, alias="userToken")
    time_to_first_token: float = Field(default=0.0, alias="timeToFirstToken")
    tokens_per_second: float = Field(default=0.0, alias="tokensPerSecond")
    model_name: str = Field(alias="modelName")


class StreamMessage(BaseModel):
    """Stream message."""

    type: Literal[
        "text",
        "tool_calls",
        "tool_call_progress",
        "tool_result",
        "error",
        "chat_info",
        "message_info",
        "interactive",
        "token_usage",
        "agent_tool_call",
        "agent_tool_result",
    ]
    content: (
        str
        | list[ToolCallsContent]
        | ToolResultContent
        | ErrorContent
        | ChatInfoContent
        | MessageInfoContent
        | InteractiveContent
        | ToolCallProgress
        | TokenUsageContent
        | AgentToolCallContent
        | AgentToolResultContent
    )


class TokenUsage(BaseModel):
    """Token usage."""

    total_input_tokens: int = Field(default=0, alias="totalInputTokens")
    total_output_tokens: int = Field(default=0, alias="totalOutputTokens")
    user_token: int = Field(default=0, alias="userToken")
    total_tokens: int = Field(default=0, alias="totalTokens")
    time_to_first_token: float = Field(default=0.0, alias="timeToFirstToken")
    tokens_per_second: float = Field(default=0.0, alias="tokensPerSecond")


class ModelSingleConfig(BaseModel):
    """Model single config."""

    model_provider: str
    model: str
    max_tokens: int | None = None
    api_key: SecretStr | None = None
    configuration: LLMConfiguration | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None
    active: bool = Field(default=True)
    checked: bool = Field(default=False)
    tools_in_prompt: bool = Field(default=False)

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
        # ollama doesn't work well with normal bind tools
        if self.model_provider == "ollama":
            self.tools_in_prompt = True

        self.to_host_llm_config()

        return self

    def to_host_llm_config(self) -> LLMConfigTypes:
        """Convert to LLMConfigTypes."""
        return get_llm_config_type(self.model_provider).model_validate(
            self.model_dump()
        )

    @field_serializer("api_key", when_used="json")
    def dump_api_key(self, v: SecretStr | None) -> str | None:
        """Serialize the api_key field to plain text."""
        return v.get_secret_value() if v else None


class ModelFullConfigs(BaseModel):
    """Configuration for the model."""

    active_provider: str
    enable_tools: bool
    configs: dict[str, ModelSingleConfig] = Field(default_factory=dict)
    embed_config: EmbedConfig | None = None

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
