from enum import StrEnum
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from dive_mcp_host.host.conf import LLMConfig

T = TypeVar("T")


class ResultResponse(BaseModel):
    """Generic response model with success status and message."""

    success: bool
    message: str | None


Transport = Literal["command", "sse", "websocket"]


class McpServerConfig(BaseModel):
    """MCP server configuration with transport and connection settings."""

    transport: Transport
    enabled: bool | None
    command: str | None
    args: list[str] | None
    env: dict[str, str] | None
    url: str | None


class McpServers(BaseModel):
    """Collection of MCP server configurations."""

    mcp_servers: dict[str, McpServerConfig] = Field(alias="mcpServers")


class McpServerError(BaseModel):
    """Represents an error from an MCP server."""

    server_name: str = Field(alias="serverName")
    error: object  # any


class ModelType(StrEnum):
    """Model type."""

    OLLAMA = "ollama"
    MISTRAL = "mistralai"
    BEDROCK = "bedrock"
    DEEPSEEK = "deepseek"
    OTHER = "other"

    @classmethod
    def get_model_type(cls, llm_config: LLMConfig) -> "ModelType":
        """Get model type from model name."""
        if llm_config.provider == "ollama":
            return cls.OLLAMA

        if llm_config.provider == "mistralai":
            return cls.MISTRAL

        if llm_config.provider == "bedrock":
            return cls.BEDROCK

        if "deepseek" in llm_config.model.lower():
            return cls.DEEPSEEK

        return cls.OTHER


class ModelSettingsProperty(BaseModel):
    """Defines a property for model settings with type information and metadata."""

    type: Literal["string", "number"]
    description: str
    required: bool
    default: object | None
    placeholder: object | None


class ModelSettingsDefinition(ModelSettingsProperty):
    """Model settings definition with nested properties."""

    type: Literal["string", "number", "object"]
    properties: dict[str, ModelSettingsProperty] | None


class ModelInterfaceDefinition(BaseModel):
    """Defines the interface for model settings."""

    model_settings: dict[str, ModelSettingsDefinition]


class McpTool(BaseModel):
    """Represents an MCP tool with its properties and metadata."""

    name: str
    tools: list
    description: str
    enabled: bool
    icon: str


class ToolCallsContent(BaseModel):
    """Tool call content."""

    name: str
    arguments: object


class ToolResultContent(BaseModel):
    """Tool result content."""

    name: str
    result: object


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

    total_input_tokens: int = Field(alias="totalInputTokens")
    total_output_tokens: int = Field(alias="totalOutputTokens")
    total_tokens: int = Field(alias="totalTokens")


class UserInputError(Exception):
    """User input error.

    Args:
        Exception (Exception): The exception.
    """

    def __init__(self, message: str) -> None:
        """Initialize the error.

        Args:
            message (str): The error message.
        """
        self.message = message
