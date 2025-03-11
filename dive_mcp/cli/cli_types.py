"""Dive MCP Host CLI types."""

from dataclasses import dataclass


@dataclass
class CLIArgs:
    """CLI arguments.

    Args:
        thread_id: The thread id to continue from.
        query: The input query.
        config_path: The path to the configuration file.
    """

    thread_id: str | None
    query: list
    config_path: str
