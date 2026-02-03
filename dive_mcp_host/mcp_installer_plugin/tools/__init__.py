"""Tools for the MCP Server Installer Agent.

These tools provide fetch, bash, and filesystem operations with built-in
elicitation support for user approval of potentially dangerous operations.
"""

from langchain_core.tools import BaseTool

from dive_mcp_host.mcp_installer_plugin.tools.bash import bash
from dive_mcp_host.mcp_installer_plugin.tools.confirmation import request_confirmation
from dive_mcp_host.mcp_installer_plugin.tools.fetch import fetch
from dive_mcp_host.mcp_installer_plugin.tools.file_ops import read_file, write_file
from dive_mcp_host.mcp_installer_plugin.tools.mcp_server import (
    add_mcp_server,
    get_mcp_config,
    install_mcp_instructions,
    reload_mcp_server,
)
from dive_mcp_host.mcp_installer_plugin.tools.skills import (
    create_dive_skill_tool,
    dive_install_skill_from_path,
)


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
        create_dive_skill_tool(),
        dive_install_skill_from_path,
    ]
