"""File operations tools for the MCP Server Installer Agent.

This module provides read_file and write_file tools for filesystem operations.
"""

# ruff: noqa: PLR0911, PLR2004
# E501: Line too long - tool descriptions require specific formatting
# PLR0911: Many return statements needed for complex control flow
# PLR2004: Magic values are intentional truncation limits

from __future__ import annotations

import logging
from collections.abc import Callable  # noqa: TC003
from pathlib import Path
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from mcp import types
from pydantic import Field

from dive_mcp_host.host.agents.agent_factory import (
    ensure_config,
    get_abort_signal,
    get_dry_run,
    get_stream_writer,
)
from dive_mcp_host.host.tools.elicitation_manager import (
    ElicitationManager,
    ElicitationTimeoutError,
)
from dive_mcp_host.internal_tools.events import InstallerToolLog
from dive_mcp_host.internal_tools.tools.common import check_aborted

logger = logging.getLogger(__name__)


@tool(
    description="""
Read content from a file.
Use this to read configuration files, check existing setups, etc.
Supports text files only.
"""
)
async def read_file(
    path: Annotated[str, Field(description="Path to the file to read.")],
    encoding: Annotated[
        str,
        Field(default="utf-8", description="File encoding."),
    ] = "utf-8",
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Read a file.

    Note: User confirmation is handled by the confirm_install node in the graph,
    not by individual tools.
    """
    config = ensure_config(config)

    stream_writer = get_stream_writer(config)
    abort_signal = get_abort_signal(config)

    # Check if already aborted
    if check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Expand user home directory
    expanded_path = str(Path(path).expanduser())

    # Read the file
    stream_writer(
        (
            InstallerToolLog.NAME,
            InstallerToolLog(
                tool="read_file",
                action=f"Reading: {path}",
                details={"path": expanded_path},
            ),
        )
    )
    try:
        file_path = Path(expanded_path)
        if not file_path.exists():
            result = f"Error: File not found: {path}"
        elif not file_path.is_file():
            result = f"Error: Not a file: {path}"
        else:
            content = file_path.read_text(encoding=encoding)

            # Truncate very long files
            if len(content) > 100000:
                result = content[:100000] + "\n... (truncated)"
            else:
                result = content

    except OSError as e:
        result = f"Error reading file {path}: {e}"

    return result


@tool(
    description="""
Write content to a file.
Use this to create or modify configuration files, scripts, etc.
Will create parent directories if needed.
Always requests user approval before writing.
"""
)
async def write_file(
    path: Annotated[str, Field(description="Path to the file to write.")],
    content: Annotated[str, Field(description="Content to write to the file.")],
    encoding: Annotated[
        str,
        Field(default="utf-8", description="File encoding."),
    ] = "utf-8",
    create_dirs: Annotated[
        bool,
        Field(
            default=True, description="Create parent directories if they don't exist."
        ),
    ] = True,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Write to a file.

    Requests user confirmation before writing.
    """
    config = ensure_config(config)

    stream_writer = get_stream_writer(config)
    dry_run = get_dry_run(config)

    return await execute_write(
        path=path,
        content=content,
        encoding=encoding,
        create_dirs=create_dirs,
        stream_writer=stream_writer,
        dry_run=dry_run,
        config=config,
    )


async def execute_write(
    path: str,
    content: str,
    encoding: str,
    create_dirs: bool,
    stream_writer: Callable[[tuple[str, Any]], None],
    dry_run: bool,
    config: RunnableConfig,
) -> str:
    """Execute the write file operation (internal implementation)."""
    abort_signal = get_abort_signal(config)
    elicitation_manager: ElicitationManager | None = config.get("configurable", {}).get(
        "elicitation_manager"
    )

    # Check if already aborted
    if check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Expand user home directory
    expanded_path = str(Path(path).expanduser())
    file_path = Path(expanded_path)
    file_exists = file_path.exists()

    # Prepare content preview (truncate if too long)
    content_preview = content
    if len(content) > 500:
        content_preview = content[:500] + f"\n... ({len(content) - 500} more bytes)"

    # Log the write operation
    log_details: dict[str, Any] = {
        "path": expanded_path,
        "size": len(content),
        "file_exists": file_exists,
        "dry_run": dry_run,
    }

    action_prefix = "[DRY RUN] " if dry_run else ""

    stream_writer(
        (
            InstallerToolLog.NAME,
            InstallerToolLog(
                tool="write_file",
                action=f"{action_prefix}Writing: {path}",
                details=log_details,
            ),
        )
    )

    # If dry_run is enabled, simulate success without writing
    if dry_run:
        return (
            f"[DRY RUN] Would write {len(content)} bytes to {path}\nSimulated success."
        )

    # Request user confirmation before writing
    if elicitation_manager is not None:
        operation = "overwrite" if file_exists else "create"
        confirm_message = (
            f"The write_file tool wants to {operation} the following file:\n\n"
            f"**Path:** `{path}`\n"
            f"**Size:** {len(content)} bytes\n\n"
            f"**Content:**\n```\n{content_preview}\n```"
        )

        confirm_schema = {
            "type": "object",
            "properties": {},
        }

        params = types.ElicitRequestFormParams(
            message=confirm_message,
            requestedSchema=confirm_schema,
        )

        logger.info("Requesting user confirmation for write_file: %s", path)

        try:
            result = await elicitation_manager.request(
                params=params,
                writer=stream_writer,
                abort_signal=abort_signal,
            )

            if result.action == "decline":
                return f"Write cancelled: User declined to {operation} the file."
            if result.action != "accept":
                return "Write cancelled: User cancelled the confirmation."

        except ElicitationTimeoutError:
            return "Error: Confirmation timed out. File not written."
        except Exception as e:
            logger.exception("Error getting confirmation via elicitation")
            return f"Error getting confirmation: {e}"

    # Check abort before writing
    if check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Write the file
    try:
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding=encoding)

        return f"Successfully wrote {len(content)} bytes to {path}"

    except OSError as e:
        return f"Error writing to file {path}: {e}"
