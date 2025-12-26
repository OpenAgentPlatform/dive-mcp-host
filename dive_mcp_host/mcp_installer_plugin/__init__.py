"""MCP Server Installer Agent.

This module provides an LLM-powered agent for automatically installing MCP servers.
The agent uses tools (fetch, bash, filesystem) with elicitation support to safely
execute commands and file operations with user approval.

## Architecture

The installer consists of:

1. **InstallerAgent**: LangGraph-based agent that orchestrates installation
2. **Installer Tools**: fetch, bash, read_file, write_file with elicitation support
3. **InstallMcpServerTool**: LangChain tool that wraps the agent

## Usage

```python
from langchain_openai import ChatOpenAI
from dive_mcp_host.mcp_installer_plugin import install_mcp_server_tool

# Create the tool with a model
model = ChatOpenAI(model="gpt-4")
tool = install_mcp_server_tool(model)

# Use in a LangChain agent or call directly
result = await tool.ainvoke(
    {
        "server_description": "mcp-server-fetch",
    }
)
```

## Elicitation Flow

When the installer needs to execute a potentially dangerous operation:

1. Tool sends a ToolElicitationRequest event via the shared ElicitationManager
2. The frontend displays the request to the user
3. User responds with accept/decline/cancel
4. The operation proceeds or is cancelled based on the response

## Events

- **InstallerProgress**: Progress updates during installation
- **InstallerResult**: Final result of the installation
- **InstallerToolLog**: Log entry when a tool executes (bash, fetch, read/write)
"""

from dive_mcp_host.mcp_installer_plugin.agent import InstallerAgent
from dive_mcp_host.mcp_installer_plugin.config import (
    InstallerConfigManager,
    InstallerSettings,
    PluginSettings,
)
from dive_mcp_host.mcp_installer_plugin.events import (
    InstallerProgress,
    InstallerResult,
    InstallerToolLog,
)
from dive_mcp_host.mcp_installer_plugin.prompt import (
    get_installer_system_prompt,
    get_system_tools_info,
    is_tool_available,
)
from dive_mcp_host.mcp_installer_plugin.runtime import (
    get_httpd_base_url,
    set_httpd_base_url,
)
from dive_mcp_host.mcp_installer_plugin.tool import (
    InstallMcpServerTool,
    InstallMcpServerToolManager,
    install_mcp_server_tool,
)
from dive_mcp_host.mcp_installer_plugin.tools import (
    InstallerBashTool,
    InstallerFetchTool,
    InstallerReadFileTool,
    InstallerWriteFileTool,
    get_installer_tools,
    get_local_tools,
)

__all__ = [
    "InstallMcpServerTool",
    "InstallMcpServerToolManager",
    "InstallerAgent",
    "InstallerBashTool",
    "InstallerConfigManager",
    "InstallerFetchTool",
    "InstallerProgress",
    "InstallerReadFileTool",
    "InstallerResult",
    "InstallerSettings",
    "InstallerToolLog",
    "InstallerWriteFileTool",
    "PluginSettings",
    "get_httpd_base_url",
    "get_installer_system_prompt",
    "get_installer_tools",
    "get_local_tools",
    "get_system_tools_info",
    "install_mcp_server_tool",
    "is_tool_available",
    "set_httpd_base_url",
]
