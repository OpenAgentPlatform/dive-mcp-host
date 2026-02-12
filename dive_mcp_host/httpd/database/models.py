from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import StrEnum

from langchain_core.messages import ToolCall  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field

from dive_mcp_host.httpd.routers.models import TokenUsage  # noqa: TC001


class ResourceUsage(BaseModel):
    """Represents information about a language model's usage statistics."""

    model: str
    total_input_tokens: int
    total_output_tokens: int
    user_token: int = 0
    custom_prompt_token: int = 0
    system_prompt_token: int = 0
    time_to_first_token: float = 0.0
    tokens_per_second: float = 0.0
    total_run_time: float


class QueryInput(BaseModel):
    """User input for a query with text, images and documents."""

    text: str | None
    images: list[str] | None
    documents: list[str] | None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Chat(BaseModel):
    """Represents a chat conversation with its basic properties."""

    id: str
    title: str
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime | None = Field(alias="updatedAt")
    starred_at: datetime | None = Field(alias="starredAt")
    user_id: str | None


class Role(StrEnum):
    """Role for Messages."""

    ASSISTANT = "assistant"
    USER = "user"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class NewMessage(BaseModel):
    """Represents a message within a chat conversation."""

    content: str
    role: Role
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    resource_usage: ResourceUsage | None = None
    files: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list, alias="toolCalls")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class Message(BaseModel):
    """Represents a message within a chat conversation."""

    id: int
    create_at: datetime = Field(alias="createdAt")
    content: str
    role: Role
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    resource_usage: ResourceUsage | None = None
    files: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list, alias="toolCalls")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class FTSResult(BaseModel):
    """Full text search result."""

    chat_id: str
    message_id: str
    title_snippet: str
    content_snippet: str
    msg_created_at: datetime
    chat_updated_at: datetime


class ChatMessage(BaseModel):
    """Combines a chat with its associated messages."""

    chat: Chat
    messages: list[Message]
    token_usage: TokenUsage | None = None
