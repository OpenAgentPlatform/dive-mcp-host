from enum import StrEnum

from pydantic import BaseModel


class ClientState(StrEnum):
    """The state of the client.

    States and transitions:
    """

    INIT = "init"
    RUNNING = "running"
    CLOSED = "closed"
    RESTARTING = "restarting"
    FAILED = "failed"


class ToolCallProgress(BaseModel):
    """The progress of a tool call."""

    progress: float
    total: float | None
    message: str | None
    tool_call_id: str | None
