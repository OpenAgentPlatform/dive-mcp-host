"""Custom events for the MCP Server Installer tools."""

from typing import Any, ClassVar, Literal

from dive_mcp_host.host.custom_events import CustomEvent


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
        "install_mcp_instructions",
        "get_mcp_config",
    ]
    """The tool that generated this log."""

    action: str
    """Description of what the tool is doing."""

    details: dict[str, Any] | None = None
    """Additional details about the action."""
