"""Confirmation tool for the MCP Server Installer Agent.

This module provides the request_confirmation tool for requesting
user approval before performing installation actions.
"""

# ruff: noqa: E501, PLR0911
# E501: Line too long - tool descriptions require specific formatting
# PLR0911: Many return statements needed for complex control flow

from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from dive_mcp_host.host.agents.agent_factory import (
    ensure_config,
    get_abort_signal,
    get_stream_writer,
)
from dive_mcp_host.internal_tools.tools.common import check_aborted

logger = logging.getLogger(__name__)


@tool(
    description="""Request user confirmation before performing installation actions.

Use this tool BEFORE executing any installation commands, file writes, or server configurations.
You MUST provide:
1. A clear message explaining what you want to do (in the user's language)
2. A list of specific actions that will be performed

The user will see your message and can approve or reject the actions.
If rejected, do NOT proceed with the actions.

Example:
  message: "I need to perform the following actions to install the MCP server. Do you approve?"
  actions: ["Run command: npm install -g @anthropic/mcp-server", "Write config to mcp_config.json"]
"""
)
async def request_confirmation(
    message: Annotated[
        str,
        Field(
            description="The message to display to the user asking for confirmation. "
            "Should be in the user's preferred language."
        ),
    ],
    actions: Annotated[
        list[str],
        Field(
            description="List of actions that will be performed if confirmed. "
            "Each action should be a clear, concise description."
        ),
    ],
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Request user confirmation.

    Returns:
        'approved' if user approves, 'rejected' if user rejects.
    """
    from mcp import types

    from dive_mcp_host.host.tools.elicitation_manager import (
        ElicitationManager,
        ElicitationTimeoutError,
    )

    config = ensure_config(config)

    stream_writer = get_stream_writer(config)
    abort_signal = get_abort_signal(config)
    elicitation_manager: ElicitationManager | None = config.get("configurable", {}).get(
        "elicitation_manager"
    )

    # Check if already aborted
    if check_aborted(abort_signal):
        return "aborted"

    if elicitation_manager is None:
        logger.warning("No elicitation manager, auto-approving")
        return "approved (no elicitation manager available)"

    # Build the full message including actions list
    # This ensures actions are always visible even if LLM provides minimal message
    actions_text = "\n".join(f"• {action}" for action in actions)
    full_message = f"{message}\n\n{actions_text}" if message else actions_text

    # Empty schema - no form fields needed
    # The Accept/Decline buttons in the UI are sufficient for confirmation
    requested_schema = {
        "type": "object",
        "properties": {},
    }

    params = types.ElicitRequestFormParams(
        message=full_message,
        requestedSchema=requested_schema,
    )

    logger.info(
        "InstallerRequestConfirmationTool._arun() - requesting confirmation for %d actions",
        len(actions),
    )

    try:
        result = await elicitation_manager.request(
            params=params,
            writer=stream_writer,
            abort_signal=abort_signal,
        )

        logger.info(
            "InstallerRequestConfirmationTool._arun() - result: action=%s, content=%s",
            result.action,
            result.content,
        )

        if result.action == "accept":
            return "approved"
        if result.action == "decline":
            return "rejected"
        # cancel
        return "cancelled"

    except ElicitationTimeoutError:
        logger.warning("Elicitation timeout")
        return "timeout"
    except Exception as e:
        logger.exception("Elicitation error")
        return f"error: {e}"
