"""Hacks to make the MCP SDKs work with our needs.

This module contains hacks to make the MCP SDKs work with our needs.
Tools in this module may be removed in the future if SDKs support our needs.
"""

from .client_session import ClientSession, create_mcp_http_client_factory
from .stdio_server import stdio_client

__all__ = ["ClientSession", "create_mcp_http_client_factory", "stdio_client"]
