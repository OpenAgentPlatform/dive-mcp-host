"""Runtime configuration for the MCP installer plugin.

This module provides runtime settings that are set by the httpd server
and read by the installer tools. This keeps httpd-specific knowledge
out of the host module.
"""

import logging

logger = logging.getLogger(__name__)

# Module-level runtime state container (avoids global statement)
_runtime_state: dict[str, str | None] = {"httpd_base_url": None}


def set_httpd_base_url(url: str | None) -> None:
    """Set the HTTPD base URL for MCP reload API.

    This is called by httpd when it starts.

    Args:
        url: The base URL for the httpd server (e.g., "http://127.0.0.1:61990").
    """
    _runtime_state["httpd_base_url"] = url
    logger.debug("HTTPD base URL set: %s", url)


def get_httpd_base_url() -> str | None:
    """Get the HTTPD base URL.

    Returns:
        The httpd base URL, or None if not set.
    """
    return _runtime_state["httpd_base_url"]
