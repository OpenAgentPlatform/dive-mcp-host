"""Common utilities for MCP Server Installer Agent tools.

This module provides shared helper functions and classes used across
all installer tools.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio


logger = logging.getLogger(__name__)


def check_aborted(abort_signal: asyncio.Event | None) -> bool:
    """Check if the abort signal has been set."""
    return abort_signal is not None and abort_signal.is_set()


class AbortedError(Exception):
    """Raised when an operation is aborted."""


def get_httpd_base_url() -> str | None:
    """Get httpd base URL from runtime config."""
    from dive_mcp_host.internal_tools.runtime import get_httpd_base_url

    return get_httpd_base_url()
