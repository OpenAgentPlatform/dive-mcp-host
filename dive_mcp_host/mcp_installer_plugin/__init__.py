"""MCP Server Installer Tools.

This module provides tools for installing MCP servers with elicitation support
for user approval of potentially dangerous operations.

## Architecture

The installer consists of:

1. **Installer Tools**: fetch, bash, read_file, write_file with elicitation support
2. **MCP Server Tools**: add_mcp_server, reload_mcp_server, get_mcp_config

## Usage

```python
from dive_mcp_host.mcp_installer_plugin import get_local_tools

# Get all local tools for external LLMs
tools = get_local_tools()
```

## Elicitation Flow

When a tool needs to execute a potentially dangerous operation:

1. Tool sends an elicitation request via the shared ElicitationManager
2. The frontend displays the request to the user
3. User responds with accept/decline/cancel
4. The operation proceeds or is cancelled based on the response

## Events

- **InstallerToolLog**: Log entry when a tool executes (bash, fetch, read/write)
"""
