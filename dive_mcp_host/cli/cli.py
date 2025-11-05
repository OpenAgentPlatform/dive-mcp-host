"""Dive MCP Host CLI."""

import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

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
        default=None,
        help="The path to the configuration file.",
        dest="config_path",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="The directory containing mcp_config.json and model_config.json.",
        dest="config_dir",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="The path to the MCP servers configuration file.",
        dest="mcp_config_path",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="The path to the model configuration file.",
        dest="model_config_path",
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


def load_merged_config(mcp_config_path: str, model_config_path: str) -> HostConfig:
    """Load and merge MCP and model configurations."""
    # Load MCP config
    with Path(mcp_config_path).open("r") as f:
        mcp_data = json.load(f)

    # Load model config
    with Path(model_config_path).open("r") as f:
        model_data = json.load(f)

    # Get active provider config
    active_provider = model_data.get("activeProvider")
    if not active_provider:
        raise ValueError("model_config must have 'activeProvider' field")

    configs = model_data.get("configs", {})
    if active_provider not in configs:
        raise ValueError(f"activeProvider '{active_provider}' not found in configs")

    active_config = configs[active_provider]

    # Process MCP servers and add name field
    mcp_servers = {}
    for server_name, server_config in mcp_data.get("mcpServers", {}).items():
        server_config_with_name = {**server_config, "name": server_name}
        mcp_servers[server_name] = server_config_with_name

    # Merge configs
    merged_config = {
        "llm": active_config,
        "mcp_servers": mcp_servers
    }

    return HostConfig.model_validate(merged_config)


async def run() -> None:
    """dive_mcp_host CLI entrypoint."""
    args = setup_argument_parser()
    query = parse_query(args)

    # Load config based on provided arguments
    if args.config_path:
        config = load_config(args.config_path)
    elif args.config_dir or args.mcp_config_path or args.model_config_path:
        # User explicitly provided config options
        if args.config_dir:
            config_dir = Path(args.config_dir)
            mcp_config_path = str(config_dir / "mcp_config.json")
            model_config_path = str(config_dir / "model_config.json")
        else:
            mcp_config_path = args.mcp_config_path or "mcp_config.json"
            model_config_path = args.model_config_path or "model_config.json"

        config = load_merged_config(mcp_config_path, model_config_path)
    else:
        # No config options provided, try default files in order
        default_config = Path("config.json")
        if default_config.exists():
            config = load_config(str(default_config))
        else:
            # Fall back to separate config files
            mcp_config_path = "mcp_config.json"
            model_config_path = "model_config.json"
            config = load_merged_config(mcp_config_path, model_config_path)

    current_chat_id: str | None = args.chat_id
    system_prompt = None
    if args.prompt_file:
        with Path(args.prompt_file).open("r") as f:
            system_prompt = f.read()

    output_parser = StrOutputParser()
    async with DiveMcpHost(config) as mcp_host:
        print("Waiting for tools to initialize...")
        await mcp_host.tools_initialized_event.wait()
        print("Tools initialized")
        chat = mcp_host.chat(chat_id=current_chat_id, system_prompt=system_prompt)
        current_chat_id = chat.chat_id
        async with chat:
            async for response in chat.query(query, stream_mode="messages"):
                assert isinstance(response, tuple)
                msg = response[0]
                if isinstance(msg, AIMessage):
                    content = output_parser.invoke(msg)
                    print(content, end="")
                    continue
                print(f"\n\n==== Start Of {type(msg)} ===")
                print(msg)
                print(f"==== End Of {type(msg)} ===\n")

    print()
    print(f"Chat ID: {current_chat_id}")
