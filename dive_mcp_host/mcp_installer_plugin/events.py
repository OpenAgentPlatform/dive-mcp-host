"""Custom events for the MCP Server Installer Agent."""

from typing import Any, ClassVar, Literal

from dive_mcp_host.host.custom_events import CustomEvent


class InstallerProgress(CustomEvent):
    """Progress update from the installer agent."""

    NAME: ClassVar[str] = "installer_progress"

    phase: Literal[
        "analyzing",
        "fetching_info",
        "preparing",
        "installing",
        "configuring",
        "verifying",
        "completed",
        "failed",
    ]
    """Current phase of the installation."""

    message: str
    """Human-readable progress message."""

    progress: float | None = None
    """Progress percentage (0-100), if available."""


class InstallerResult(CustomEvent):
    """Final result of the installation."""

    NAME: ClassVar[str] = "installer_result"

    success: bool
    """Whether the installation succeeded."""

    server_name: str | None = None
    """Name of the installed MCP server (if successful)."""

    config: dict[str, Any] | None = None
    """Configuration for the installed server (if successful)."""

    error_message: str | None = None
    """Error message (if failed)."""


class InstallerToolLog(CustomEvent):
    """Log entry from installer tool execution."""

    NAME: ClassVar[str] = "installer_tool_log"

    tool: Literal[
        "bash",
        "fetch",
        "write_file",
        "read_file",
        "add_mcp_server",
        "reload_mcp_server",
    ]
    """The tool that generated this log."""

    action: str
    """Description of what the tool is doing."""

    details: dict[str, Any] | None = None
    """Additional details about the action."""


class InstallerElicitationRequest(CustomEvent):
    """Request for user approval of an installer operation."""

    NAME: ClassVar[str] = "installer_elicitation_request"

    request_id: str
    """Unique identifier for this request."""

    operation_type: Literal["bash", "fetch", "write_file", "read_file"]
    """Type of operation requiring approval."""

    message: str
    """Human-readable description of the operation."""

    details: dict[str, Any]
    """Additional details about the operation."""

    risk_level: Literal["low", "medium", "high"]
    """Assessed risk level of the operation."""


class InstallerElicitationResponse(CustomEvent):
    """User response to an elicitation request."""

    NAME: ClassVar[str] = "installer_elicitation_response"

    action: Literal["allow", "allow_always", "deny"]
    """User's decision on the operation."""


class AgentToolCall(CustomEvent):
    """Event emitted when an agent tool starts execution."""

    NAME: ClassVar[str] = "agent_tool_call"

    tool_call_id: str
    """Unique identifier for this tool call."""

    name: str
    """Name of the tool being called."""

    args: dict[str, Any]
    """Arguments passed to the tool."""


class AgentToolResult(CustomEvent):
    """Event emitted when an agent tool completes execution."""

    NAME: ClassVar[str] = "agent_tool_result"

    tool_call_id: str
    """Unique identifier for this tool call."""

    name: str
    """Name of the tool that was called."""

    result: str
    """Result returned by the tool."""
