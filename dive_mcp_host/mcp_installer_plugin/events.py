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

    tool: Literal["bash", "fetch", "write_file", "read_file", "add_mcp_server"]
    """The tool that generated this log."""

    action: str
    """Description of what the tool is doing."""

    details: dict[str, Any] | None = None
    """Additional details about the action."""
