from langchain_core.tools import BaseTool

from dive_mcp_host.internal_tools.tools.bash import bash
from dive_mcp_host.internal_tools.tools.confirmation import request_confirmation
from dive_mcp_host.internal_tools.tools.fetch import fetch
from dive_mcp_host.internal_tools.tools.file_ops import read_file, write_file
from dive_mcp_host.internal_tools.tools.mcp_server import (
    add_mcp_server,
    get_mcp_config,
    install_mcp_instructions,
    reload_mcp_server,
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
    ]
