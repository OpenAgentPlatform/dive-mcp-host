from datetime import timedelta
from typing import Any

from mcp import ClientSession as McpClientSession
from mcp import types
from mcp.shared.message import MessageMetadata
from mcp.shared.session import ProgressFnT


class ClientSession(McpClientSession):
    """Hacked MCP Client Session.

    This is a hack to allow for metadata to be passed to the call_tool method.
    """

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        metadata: MessageMetadata | None = None,
    ) -> types.CallToolResult:
        """Call a tool with optional metadata for resumption."""
        return await self.send_request(
            types.ClientRequest(
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(
                        name=name,
                        arguments=arguments,
                    ),
                )
            ),
            types.CallToolResult,
            request_read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
            metadata=metadata,
        )
