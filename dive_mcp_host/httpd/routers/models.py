import json
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


class DataResult[T](ResultResponse):
    """Generic result that extends ResultResponse with a data field."""

    data: T


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
    custom_prompt_token: int = Field(default=0, alias="customPromptToken")
    system_prompt_token: int = Field(default=0, alias="systemPromptToken")
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
    custom_prompt_token: int = Field(default=0, alias="customPromptToken")
    system_prompt_token: int = Field(default=0, alias="systemPromptToken")
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

    enable_local_tools: bool = True
    # If True, local tools (fetch, bash, read_file, write_file) will be available
    # to the LLM directly without going through the installer agent.
    # Default is True - enabled by default when not specified in config.

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


# ==== For OpenAPI Doc ====


# OpenAPI Doc - Helper to format SSE data
def _sse(msg: StreamMessage) -> str:
    data = json.dumps({"message": msg.model_dump_json(by_alias=True)})
    return f"data: {data}\n\n"


CHAT_EVENT_STREAM = {
    "description": (
        "Real-time event stream using Server-Sent Events (SSE) format. "
        'Each event is sent as `data: {"message": "<json-encoded StreamMessage>"}\\n\\n`. '  # noqa: E501
        "The stream ends with `data: [DONE]\\n\\n`."
    ),
    "content": {
        "text/event-stream": {
            "schema": {"type": "string"},
            "examples": {
                "text": {
                    "summary": "AI response text chunks",
                    "value": _sse(
                        StreamMessage(
                            type="text", content="Hello! How can I help you today?"
                        )
                    ),
                },
                "chat_info": {
                    "summary": "Chat ID and title information",
                    "value": _sse(
                        StreamMessage(
                            type="chat_info",
                            content=ChatInfoContent(
                                id="chat-abc123", title="New Conversation"
                            ),
                        )
                    ),
                },
                "message_info": {
                    "summary": "User and assistant message IDs",
                    "value": _sse(
                        StreamMessage(
                            type="message_info",
                            content=MessageInfoContent(
                                userMessageId="msg-user-001",
                                assistantMessageId="msg-asst-001",
                            ),
                        )
                    ),
                },
                "tool_calls": {
                    "summary": "Tool invocation requests from the AI",
                    "value": _sse(
                        StreamMessage(
                            type="tool_calls",
                            content=[
                                ToolCallsContent(
                                    name="search", arguments={"query": "weather today"}
                                )
                            ],
                        )
                    ),
                },
                "tool_result": {
                    "summary": "Tool execution results",
                    "value": _sse(
                        StreamMessage(
                            type="tool_result",
                            content=ToolResultContent(
                                name="search",
                                result={"temperature": "72F", "condition": "sunny"},
                            ),
                        )
                    ),
                },
                "tool_call_progress": {
                    "summary": "Progress updates during tool execution",
                    "value": _sse(
                        StreamMessage(
                            type="tool_call_progress",
                            content=ToolCallProgress(
                                progress=50,
                                total=100,
                                message="Processing...",
                                tool_call_id="call-123",
                            ),
                        )
                    ),
                },
                "error": {
                    "summary": "Error messages",
                    "value": _sse(
                        StreamMessage(
                            type="error",
                            content=ErrorContent(
                                message="Connection timeout", type="NetworkError"
                            ),
                        )
                    ),
                },
                "interactive_auth": {
                    "summary": "Authentication required for MCP server",
                    "value": _sse(
                        StreamMessage(
                            type="interactive",
                            content=InteractiveContent(
                                type="authentication_required",
                                content=AuthenticationRequiredContent(
                                    server_name="github",
                                    auth_url="https://github.com/login/oauth/authorize?...",
                                ),
                            ),
                        )
                    ),
                },
                "interactive_elicitation": {
                    "summary": "User input request from MCP server",
                    "value": _sse(
                        StreamMessage(
                            type="interactive",
                            content=InteractiveContent(
                                type="elicitation_request",
                                content=ElicitationRequestContent(
                                    request_id="req-456",
                                    message="Please provide your API key",
                                    requested_schema={
                                        "type": "object",
                                        "properties": {"api_key": {"type": "string"}},
                                    },
                                ),
                            ),
                        )
                    ),
                },
                "token_usage": {
                    "summary": "Token usage statistics",
                    "value": _sse(
                        StreamMessage(
                            type="token_usage",
                            content=TokenUsageContent(
                                inputTokens=150,
                                outputTokens=75,
                                userToken=50,
                                customPromptToken=0,
                                systemPromptToken=100,
                                timeToFirstToken=0.5,
                                tokensPerSecond=25.0,
                                modelName="gpt-4",
                            ),
                        )
                    ),
                },
                "agent_tool_call": {
                    "summary": "Sub-agent tool invocation",
                    "value": _sse(
                        StreamMessage(
                            type="agent_tool_call",
                            content=AgentToolCallContent(
                                toolCallId="call-789",
                                name="code_interpreter",
                                args={"code": "print(2+2)"},
                            ),
                        )
                    ),
                },
                "agent_tool_result": {
                    "summary": "Sub-agent tool execution result",
                    "value": _sse(
                        StreamMessage(
                            type="agent_tool_result",
                            content=AgentToolResultContent(
                                toolCallId="call-789",
                                name="code_interpreter",
                                result="4",
                            ),
                        )
                    ),
                },
                "done": {
                    "summary": "Stream completion signal",
                    "value": "data: [DONE]\n\n",
                },
            },
        }
    },
}
