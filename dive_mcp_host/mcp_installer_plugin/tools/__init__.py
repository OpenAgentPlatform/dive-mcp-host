"""Tools for the MCP Server Installer Agent.

These tools provide fetch, bash, and filesystem operations with built-in
elicitation support for user approval of potentially dangerous operations.
"""

from langchain_core.tools import BaseTool

# Import tool functions and classes from submodules
from dive_mcp_host.mcp_installer_plugin.tools.bash import (
    BashInput,
    InstallerBashTool,
    _detect_high_risk_command,
    _detect_write_command,
    bash,
    execute_bash,
    kill_process_tree,
    wait_with_abort,
)
from dive_mcp_host.mcp_installer_plugin.tools.common import (
    AbortedError,
    _check_aborted,
    _ensure_config,
    _get_abort_signal,
    _get_dry_run,
    _get_httpd_base_url,
    _get_mcp_reload_callback,
    _get_stream_writer,
    _get_tool_call_id,
)
from dive_mcp_host.mcp_installer_plugin.tools.confirmation import (
    InstallerRequestConfirmationTool,
    RequestConfirmationInput,
    request_confirmation,
)
from dive_mcp_host.mcp_installer_plugin.tools.fetch import (
    FetchInput,
    InstallerFetchTool,
    fetch,
)
from dive_mcp_host.mcp_installer_plugin.tools.file_ops import (
    InstallerReadFileTool,
    InstallerWriteFileTool,
    ReadFileInput,
    WriteFileInput,
    execute_write,
    read_file,
    write_file,
)
from dive_mcp_host.mcp_installer_plugin.tools.mcp_server import (
    AddMcpServerInput,
    InstallerAddMcpServerTool,
    InstallerGetMcpConfigTool,
    InstallerReloadMcpServerTool,
    InstallMCPInstructions,
    ReloadMcpServerInput,
    add_mcp_server,
    get_mcp_config,
    install_mcp_instructions,
    reload_mcp_server,
    trigger_mcp_reload,
)

__all__ = [  # noqa: RUF022
    # Common utilities
    "AbortedError",
    "_ensure_config",
    "_get_stream_writer",
    "_get_tool_call_id",
    "_get_dry_run",
    "_get_mcp_reload_callback",
    "_get_abort_signal",
    "_check_aborted",
    "_get_httpd_base_url",
    # Fetch tool
    "FetchInput",
    "fetch",
    "InstallerFetchTool",
    # Bash tool
    "BashInput",
    "_detect_write_command",
    "_detect_high_risk_command",
    "bash",
    "execute_bash",
    "wait_with_abort",
    "kill_process_tree",
    "InstallerBashTool",
    # File operations
    "ReadFileInput",
    "read_file",
    "InstallerReadFileTool",
    "WriteFileInput",
    "write_file",
    "execute_write",
    "InstallerWriteFileTool",
    # MCP server tools
    "install_mcp_instructions",
    "InstallMCPInstructions",
    "get_mcp_config",
    "InstallerGetMcpConfigTool",
    "AddMcpServerInput",
    "add_mcp_server",
    "trigger_mcp_reload",
    "InstallerAddMcpServerTool",
    "ReloadMcpServerInput",
    "reload_mcp_server",
    "InstallerReloadMcpServerTool",
    # Confirmation tool
    "RequestConfirmationInput",
    "request_confirmation",
    "InstallerRequestConfirmationTool",
    # Convenience functions
    "get_installer_tools",
    "get_local_tools",
]


def get_installer_tools() -> list[BaseTool]:
    """Get all installer agent tools."""
    return []


def get_local_tools() -> list[BaseTool]:
    """Get local tools that can be exposed to external LLMs.

    These tools (fetch, bash, read_file, write_file) can be used by external LLMs
    directly without going through the installer agent. They include built-in
    safety mechanisms like user confirmation for potentially dangerous operations.

    Returns:
        List of local tools: fetch, bash, read_file, write_file.
    """
    return [
        fetch,
        bash,
        read_file,
        write_file,
        get_mcp_config,
        add_mcp_server,
        reload_mcp_server,
        request_confirmation,
        install_mcp_instructions,
    ]
