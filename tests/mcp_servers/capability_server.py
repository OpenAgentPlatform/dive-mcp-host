"""A configurable MCP server used to test capability negotiation.

Run with `--features tools,prompts` to advertise only the chosen features.
Supported feature names: ``tools``, ``prompts``.
Use ``--features none`` to advertise nothing (the server will reject every
``*/list`` call with Method not found).

Optionally pass ``--break tools`` (or ``prompts``) to ADVERTISE the capability
but raise -32601 from the corresponding ``*/list`` call. This simulates a
misbehaving server and lets us exercise the defensive Method-not-found
fallback.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import anyio
from mcp import McpError
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import ErrorData
from mcp.types import (
    METHOD_NOT_FOUND,
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    TextContent,
    Tool,
)


def build_server(
    features: set[str], broken: set[str]
) -> tuple[Server, NotificationOptions]:
    """Build an MCP server advertising only the requested capabilities."""
    server: Server[Any, Any] = Server("capability-server")
    notif = NotificationOptions()

    if "tools" in features:

        @server.list_tools()
        async def _list_tools() -> list[Tool]:
            if "tools" in broken:
                raise McpError(
                    ErrorData(code=METHOD_NOT_FOUND, message="tools/list missing")
                )
            return [
                Tool(
                    name="ping",
                    description="echo pong",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "msg": {"type": "string"},
                        },
                    },
                )
            ]

        @server.call_tool()
        async def _call_tool(
            name: str,  # noqa: ARG001
            arguments: dict,
        ) -> list[TextContent]:
            return [TextContent(type="text", text=arguments.get("msg", "pong"))]

    if "prompts" in features:
        notif.prompts_changed = True

        @server.list_prompts()
        async def _list_prompts() -> list[Prompt]:
            if "prompts" in broken:
                raise McpError(
                    ErrorData(code=METHOD_NOT_FOUND, message="prompts/list missing")
                )
            return [
                Prompt(
                    name="greet",
                    description="Greet the given name",
                    arguments=[
                        PromptArgument(
                            name="name",
                            description="Person to greet",
                            required=True,
                        )
                    ],
                )
            ]

        @server.get_prompt()
        async def _get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
            if name != "greet":
                raise McpError(
                    ErrorData(code=METHOD_NOT_FOUND, message=f"unknown prompt {name}")
                )
            target = (arguments or {}).get("name", "world")
            return GetPromptResult(
                description="A greeting",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"Hello, {target}!"),
                    )
                ],
            )

    return server, notif


async def _amain(features: set[str], broken: set[str]) -> None:
    server, notif = build_server(features, broken)
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            server.create_initialization_options(notification_options=notif),
        )


def main() -> None:
    """CLI entry point for the capability test server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="tools,prompts",
        help="comma-separated list: tools,prompts or 'none'",
    )
    parser.add_argument(
        "--break",
        dest="broken",
        default="",
        help="comma-separated list of */list methods to fail with -32601",
    )
    args = parser.parse_args()
    raw = {f.strip() for f in args.features.split(",") if f.strip()}
    features = set() if raw == {"none"} else raw
    broken = {f.strip() for f in args.broken.split(",") if f.strip()}
    try:
        anyio.run(_amain, features, broken)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
