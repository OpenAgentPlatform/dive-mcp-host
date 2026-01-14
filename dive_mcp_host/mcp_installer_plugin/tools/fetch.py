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
from typing import Annotated, Any

import httpx
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import ArgsSchema, BaseTool, InjectedToolArg, tool
from pydantic import BaseModel, Field, SkipValidation

from dive_mcp_host.mcp_installer_plugin.events import InstallerToolLog
from dive_mcp_host.mcp_installer_plugin.tools.common import (
    _check_aborted,
    _ensure_config,
    _get_abort_signal,
    _get_stream_writer,
)

logger = logging.getLogger(__name__)


class FetchInput(BaseModel):
    """Input schema for the fetch tool."""

    url: Annotated[str, Field(description="The URL to fetch content from.")]
    method: Annotated[
        str, Field(default="GET", description="HTTP method (GET, POST, etc.).")
    ] = "GET"
    headers: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Optional HTTP headers."),
    ] = None


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
    config = _ensure_config(config)

    stream_writer = _get_stream_writer(config)
    abort_signal = _get_abort_signal(config)

    # Check if already aborted
    if _check_aborted(abort_signal):
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


class InstallerFetchTool(BaseTool):
    """Tool for fetching content from URLs.

    This tool fetches content from URLs with built-in approval for
    unfamiliar or potentially unsafe URLs.
    """

    name: str = "fetch"
    description: str = """Fetch content from a URL.
Use this to retrieve documentation, package information, or other web content.
The tool will request user approval for unfamiliar URLs."""
    args_schema: Annotated[ArgsSchema | None, SkipValidation] = FetchInput

    async def _arun(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Fetch content from a URL.

        Note: User confirmation is handled by the confirm_install node in the graph,
        not by individual tools.
        """
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
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

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")
