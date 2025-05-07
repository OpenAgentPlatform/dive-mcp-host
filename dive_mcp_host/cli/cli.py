"""Dive MCP Host CLI."""

import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage

from dive_mcp_host.cli.cli_types import CLIArgs
from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.host import DiveMcpHost


def parse_query(args: type[CLIArgs]) -> HumanMessage:
    """Parse the query from the command line arguments."""
    query = " ".join(args.query)
    return HumanMessage(content=query)


def setup_argument_parser() -> type[CLIArgs]:
    """Setup the argument parser."""
    parser = argparse.ArgumentParser(description="Dive MCP Host CLI")
    parser.add_argument(
        "query",
        nargs="*",
        default=[],
        help="The input query.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="The path to the configuration file.",
        dest="config_path",
    )
    parser.add_argument(
        "-c",
        type=str,
        default=None,
        help="Continue from given CHAT_ID.",
        dest="chat_id",
    )
    parser.add_argument(
        "-p",
        type=str,
        default=None,
        help="With given system prompt in the file.",
        dest="prompt_file",
    )
    return parser.parse_args(namespace=CLIArgs)


def load_config(config_path: str) -> HostConfig:
    """Load the configuration."""
    with Path(config_path).open("r") as f:
        return HostConfig.model_validate_json(f.read())


async def run() -> None:
    """dive_mcp_host CLI entrypoint."""
    args = setup_argument_parser()
    query = parse_query(args)
    config = load_config(args.config_path)

    current_chat_id: str | None = args.chat_id
    system_prompt = None
    if args.prompt_file:
        with Path(args.prompt_file).open("r") as f:
            system_prompt = f.read()

    async with DiveMcpHost(config) as mcp_host:
        print("Waiting for tools to initialize...")
        await mcp_host.tools_initialized_event.wait()
        print("Tools initialized")
        chat = mcp_host.chat(chat_id=current_chat_id, system_prompt=system_prompt)
        current_chat_id = chat.chat_id
        async with chat:
            async for response in chat.query(query, stream_mode="messages"):
                print(response[0].content, end="")  # type: ignore

    print()
    print(f"Chat ID: {current_chat_id}")
