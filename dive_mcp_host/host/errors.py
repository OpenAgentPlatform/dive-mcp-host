"""Errors for the MCP host."""

from typing import Any


class MCPHostError(Exception):
    """Base exception for MCP host errors."""


class ThreadNotFoundError(MCPHostError):
    """Exception raised when a thread is not found."""

    def __init__(self, thread_id: str) -> None:
        """Initialize the error.

        Args:
            thread_id: The thread ID that was not found.
        """
        self.thread_id = thread_id
        super().__init__(f"Thread {thread_id} not found")


class ThreadQueryError(MCPHostError):
    """Exception raised when a query is not valid."""

    def __init__(
        self,
        query: Any,
        state_values: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            query: The query that was not valid.
            state_values: Thread state values.
            error: The error that occurred.
        """
        self.query = query
        self.state_values = state_values
        self.error = error
        super().__init__(
            f"Error in query, type: {type(self.error).__name__}, content: {error}"
        )


class GraphNotCompiledError(MCPHostError):
    """Exception raised when the graph is not compiled."""

    def __init__(self, thread_id: str | None = None) -> None:
        """Initialize the error.

        Args:
            thread_id: The thread ID that was not found.
        """
        self.thread_id = thread_id
        super().__init__(f"Graph not compiled for thread {thread_id}")


class MessageTypeError(MCPHostError, ValueError):
    """Exception raised when a message is not the correct type."""

    def __init__(self, msg: str | None = None) -> None:
        """Initialize the error."""
        if msg is None:
            msg = "Message is not the correct type"
        super().__init__(msg)


class InvalidMcpServerError(MCPHostError, ValueError):
    """Exception raised when a MCP server is not valid."""

    def __init__(self, mcp_server: str, reason: str | None = None) -> None:
        """Initialize the error."""
        if reason is None:
            reason = "Invalid MCP server"
        super().__init__(f"{mcp_server}: {reason}")


class McpSessionGroupError(MCPHostError, ValueError, BaseExceptionGroup):
    """Exception group of MCP session errors."""


class McpSessionNotInitializedError(MCPHostError):
    """Exception raised when a MCP session is not initialized."""

    def __init__(self, mcp_server: str) -> None:
        """Initialize the error."""
        super().__init__(f"MCP session not initialized for {mcp_server}")


class McpSessionClosedOrFailedError(MCPHostError):
    """Exception raised when a MCP session is closed or failed."""

    def __init__(self, mcp_server: str, state: str) -> None:
        """Initialize the error."""
        super().__init__(f"MCP session {state} for {mcp_server}")


class McpSessionNotRunningError(MCPHostError):
    """Trying to get a MCP session from session store, but the session was closed.

    This exception is raised when a client tries to get a MCP session from session
    store, and waiting for the session to be initialized. But the session was closed
    or failed.
    """

    def __init__(self, mcp_server: str, chat_id: str) -> None:
        """Initialize the error."""
        super().__init__(f"MCP session not running for {mcp_server} chat_id: {chat_id}")


class LogBufferNotFoundError(MCPHostError):
    """Exception raised when a log buffer is not found."""

    def __init__(self, name: str) -> None:
        """Initialize the error."""
        super().__init__(f"Log buffer {name} not found")
