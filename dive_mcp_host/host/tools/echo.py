"""A simple echo mcp server for testing."""

import asyncio
from argparse import ArgumentParser
from base64 import standard_b64encode
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP, Icon
from pydantic import Field

Instructions = """Echo a message."""

icon_path = Path(__file__).parent / "mcp.png"
icon_data = standard_b64encode(icon_path.read_bytes()).decode()
icon = Icon(
    src=f"data:image/png;base64,{icon_data}", mimeType="image/png", sizes=["64x64"]
)

mcp = FastMCP(name="echo", instructions=Instructions, icons=[icon])

ECHO_DESCRIPTION = """A simple echo tool to verify if the MCP server is working properly.
It returns a characteristic response containing the input message."""  # noqa: E501

IGNORE_DESCRIPTION = """Do nothing."""

SLEEP_INTERVAL_MS = 100


@mcp.tool(name="echo", description=ECHO_DESCRIPTION, icons=[icon])
async def echo(
    message: Annotated[str, Field(description="Message to be echoed back")],
    ctx: Context,
    delay_ms: Annotated[
        int | None,
        Field(description="Optional delay in milliseconds before responding"),
    ] = None,
) -> str:
    """Echo a message.i lalala."""
    if delay_ms and delay_ms > 0:
        left_ms = delay_ms
        while left_ms > 0:
            interval_ms = SLEEP_INTERVAL_MS if left_ms > SLEEP_INTERVAL_MS else left_ms
            await asyncio.sleep(interval_ms / 1000)
            left_ms -= interval_ms
            if left_ms > 0:
                await ctx.report_progress(
                    progress=delay_ms - left_ms,
                    total=delay_ms,
                    message=f"Halfway there: {left_ms}ms left",
                )
            else:
                await ctx.report_progress(
                    progress=delay_ms,
                    total=delay_ms,
                    message="Done",
                )
    return message


@mcp.tool(name="ignore", description=IGNORE_DESCRIPTION)
async def ignore(
    message: Annotated[str, Field(description="The message I should ignore.")],  # noqa: ARG001
) -> None:
    """Do nothing."""
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--transport", type=str, default="stdio")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)

    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    elif args.transport == "streamable":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Invalid transport: {args.transport}")
