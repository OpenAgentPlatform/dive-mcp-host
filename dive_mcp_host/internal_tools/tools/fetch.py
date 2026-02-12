"""Fetch tool for the MCP Server Installer Agent.

This module provides the fetch tool for retrieving content from URLs.
"""

# ruff: noqa: E501, PLR2004
# E501: Line too long - tool descriptions require specific formatting
# PLR2004: Magic values are intentional truncation limits

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Annotated

import httpx
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from dive_mcp_host.host.agents.agent_factory import (
    ensure_config,
    get_abort_signal,
    get_stream_writer,
)
from dive_mcp_host.internal_tools.events import InstallerToolLog
from dive_mcp_host.internal_tools.tools.common import check_aborted

logger = logging.getLogger(__name__)


@tool(
    description="""Fetch content from a URL.
Use this to retrieve documentation, package information, or other web content.
The tool will request user approval for unfamiliar URLs."""
)
async def fetch(
    url: Annotated[str, Field(description="The URL to fetch content from.")],
    method: Annotated[
        str, Field(default="GET", description="HTTP method (GET, POST, etc.).")
    ] = "GET",
    headers: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Optional HTTP headers."),
    ] = None,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Fetch content from a URL.

    Note: User confirmation is handled by the confirm_install node in the graph,
    not by individual tools.
    """
    config = ensure_config(config)

    stream_writer = get_stream_writer(config)
    abort_signal = get_abort_signal(config)

    # Check if already aborted
    if check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Perform the fetch
    stream_writer(
        (
            InstallerToolLog.NAME,
            InstallerToolLog(
                tool="fetch",
                action=f"Fetching {url}",
                details={"url": url, "method": method},
            ),
        )
    )
    try:
        # Prepare headers with default User-Agent from env if set
        request_headers = dict(headers) if headers else {}
        user_agent = (
            os.environ.get("DIVE_USER_AGENT")
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) dive-mcp-host (+https://github.com/OpenAgentPlatform/dive-mcp-host)"
        )
        if user_agent and "User-Agent" not in request_headers:
            request_headers["User-Agent"] = user_agent

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            # Create a task for the request
            request_task = asyncio.create_task(
                client.request(method, url, headers=request_headers or None)
            )

            # Wait for either the request to complete or abort signal
            if abort_signal is not None:
                abort_task = asyncio.create_task(abort_signal.wait())
                done, pending = await asyncio.wait(
                    [request_task, abort_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                # Check if aborted
                if abort_task in done:
                    return "Error: Operation aborted."

                response = request_task.result()
            else:
                response = await request_task

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                result = response.text
            elif "text/" in content_type or "application/xml" in content_type:
                # Truncate very long responses
                text = response.text
                if len(text) > 50000:
                    result = text[:50000] + "\n... (truncated)"
                else:
                    result = text
            else:
                # For binary content, just return a summary
                result = f"Binary content ({content_type}), size: {len(response.content)} bytes"

    except httpx.HTTPError as e:
        result = f"Error fetching {url}: {e}"

    return result
