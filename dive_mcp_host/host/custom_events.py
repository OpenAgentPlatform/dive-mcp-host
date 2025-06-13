from typing import ClassVar

from pydantic import BaseModel


class CustomEvent(BaseModel):
    """Custom event.

    Example:
        ```python
        class SomeEvent(CustomEvent):
            value: int

            ...


        get_stream_writer()(
            (
                SomeEvent.NAME,
                SomeEvent(
                    value=1,
                ),
            )
        )
        ```
    """

    NAME: ClassVar[str]
    """The name of the event."""


class ToolCallProgress(CustomEvent):
    """The progress of a tool call."""

    NAME: ClassVar[str] = "tool_call_progress"

    progress: float
    total: float | None
    message: str | None
    tool_call_id: str | None


__all__ = ["ToolCallProgress"]
