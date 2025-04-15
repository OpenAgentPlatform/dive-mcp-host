from datetime import datetime
from enum import StrEnum

from langchain_core.messages import ToolCall
from pydantic import BaseModel, ConfigDict, Field


class ResourceUsage(BaseModel):
    """Represents information about a language model's usage statistics."""

    model: str
    total_input_tokens: int
    total_output_tokens: int
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


class ChatMessage(BaseModel):
    """Combines a chat with its associated messages."""

    chat: Chat
    messages: list[Message]
