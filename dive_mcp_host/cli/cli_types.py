"""Dive MCP Host CLI types."""

from dataclasses import dataclass


@dataclass
class CLIArgs:
    """CLI arguments.

    Args:
        chat_id: The thread id to continue from.
        query: The input query.
        config_path: The path to the configuration file.
        config_dir: The directory containing configuration files.
        mcp_config_path: The path to the MCP servers configuration file.
        model_config_path: The path to the model configuration file.
        prompt_file: The path to the system prompt file.
    """

    chat_id: str | None
    query: list
    config_path: str | None
    config_dir: str | None
    mcp_config_path: str | None
    model_config_path: str | None
    prompt_file: str | None
