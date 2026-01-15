"""Common utilities for MCP Server Installer Agent tools.

This module provides shared helper functions and classes used across
all installer tools.
"""

# E501: Line too long - tool descriptions require specific formatting
# PLR0911: Many return statements needed for complex control flow
# PLR2004: Magic values are intentional truncation limits
# S105: password_prompt is not a hardcoded password, it's a prompt message

from __future__ import annotations

import asyncio  # noqa: TC003
import logging
from typing import TYPE_CHECKING, Any

from langgraph.config import get_config

if TYPE_CHECKING:
    from collections.abc import Callable

from langchain_core.runnables import (
    RunnableConfig,  # noqa: TC002 Langchain needs this to get type hits in runtime.
)

logger = logging.getLogger(__name__)


def _ensure_config(config: RunnableConfig | None) -> RunnableConfig:
    """Ensure config is available, falling back to LangGraph context if needed.

    InjectedToolArg may not work correctly with LangGraph's ToolNode,
    so we use get_config() as a fallback to get the config from context.
    """
    if config is not None and config.get("configurable"):
        return config

    try:
        ctx_config = get_config()
        if ctx_config and ctx_config.get("configurable"):
            logger.debug("Using config from LangGraph context")
            return ctx_config
    except (RuntimeError, LookupError):
        pass

    return config or {}


def _get_stream_writer(
    config: RunnableConfig,
) -> Callable[[tuple[str, Any]], None]:
    """Extract stream writer from config or LangGraph context.

    Priority:
    1. Explicitly set stream_writer in config (used by InstallerAgent)
    2. LangGraph's get_stream_writer() (used when running in ToolNode)
    3. No-op lambda as fallback
    """
    # First check if stream_writer is explicitly set in config (InstallerAgent case)
    writer = config.get("configurable", {}).get("stream_writer")
    if writer is not None:
        logger.debug("Using stream_writer from config")
        return writer

    # Try to get stream writer from LangGraph context (ToolNode case)
    try:
        from langgraph.config import get_stream_writer as lg_get_stream_writer

        writer = lg_get_stream_writer()
        if writer is not None:
            logger.debug("Using stream_writer from LangGraph context")
            return writer
        logger.debug("LangGraph get_stream_writer() returned None")
    except (ImportError, RuntimeError, LookupError) as e:
        logger.debug("Could not get stream writer from LangGraph context: %s", e)

    # Fallback to no-op
    logger.debug("Falling back to no-op stream_writer")
    return lambda _: None


def _get_tool_call_id(config: RunnableConfig) -> str | None:
    """Extract tool_call_id from config metadata."""
    return config.get("metadata", {}).get("tool_call_id")


def _get_dry_run(config: RunnableConfig) -> bool:
    """Extract dry_run setting from config."""
    return config.get("configurable", {}).get("dry_run", False)


def _get_mcp_reload_callback(config: RunnableConfig) -> Callable[[], Any] | None:
    """Extract MCP reload callback from config (deprecated)."""
    return config.get("configurable", {}).get("mcp_reload_callback")


def _get_abort_signal(config: RunnableConfig) -> asyncio.Event | None:
    """Extract abort signal from config."""
    return config.get("configurable", {}).get("abort_signal")


def _check_aborted(abort_signal: asyncio.Event | None) -> bool:
    """Check if the abort signal has been set."""
    return abort_signal is not None and abort_signal.is_set()


class AbortedError(Exception):
    """Raised when an operation is aborted."""


def _get_httpd_base_url() -> str | None:
    """Get httpd base URL from runtime config."""
    from dive_mcp_host.mcp_installer_plugin.runtime import get_httpd_base_url

    return get_httpd_base_url()
