"""Dive MCP Host CLI."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from dive_mcp_host.cli.cli_types import CLIArgs
from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.host import DiveMcpHost

# Default paths for CLI
CLI_DATA_DIR = Path.home() / ".dive_mcp_host"
CHECKPOINTER_PATH = CLI_DATA_DIR / "checkpoints.db"

# Loading animation characters
LOADING_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ASCII art logo
LOGO = """
    ╭────────────────────────────────────╮
    │                                    │
    │         ████████████████           │
    │      ████              ████        │
    │    ████    ████  ████    ████      │
    │   ███     ██  ████  ██     ███     │
    │  ███      ██        ██      ███    │
    │  ███      ██  OwwO  ██      ███    │
    │  ███      ██        ██      ███    │
    │   ███     ██  ████  ██     ███     │
    │    ████    ████  ████    ████      │
    │      ████              ████        │
    │         ████████████████           │
    │   /                                │
    │  /                                 │
    │ /   Dive MCP Host CLI              │
    ╰────────────────────────────────────╯
"""


def print_logo() -> None:
    """Print the CLI logo."""
    print(LOGO)


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
        config_data = json.load(f)

    # Add default checkpointer if not present
    if "checkpointer" not in config_data:
        CHECKPOINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        config_data["checkpointer"] = {"uri": f"sqlite:///{CHECKPOINTER_PATH}"}

    return HostConfig.model_validate(config_data)


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

    # Setup default checkpointer for CLI (use sqlite in home directory)
    CHECKPOINTER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Merge configs
    merged_config = {
        "llm": active_config,
        "mcp_servers": mcp_servers,
        "checkpointer": {"uri": f"sqlite:///{CHECKPOINTER_PATH}"},
    }

    return HostConfig.model_validate(merged_config)


def get_config_from_args(args: type[CLIArgs]) -> HostConfig:
    """Load configuration based on CLI arguments."""
    if args.config_path:
        return load_config(args.config_path)

    if args.config_dir or args.mcp_config_path or args.model_config_path:
        # User explicitly provided config options
        if args.config_dir:
            config_dir = Path(args.config_dir)
            mcp_config_path = str(config_dir / "mcp_config.json")
            model_config_path = str(config_dir / "model_config.json")
        else:
            mcp_config_path = args.mcp_config_path or "mcp_config.json"
            model_config_path = args.model_config_path or "model_config.json"

        return load_merged_config(mcp_config_path, model_config_path)

    # No config options provided, try default files in order
    default_config = Path("config.json")
    if default_config.exists():
        return load_config(str(default_config))

    # Fall back to separate config files
    return load_merged_config("mcp_config.json", "model_config.json")


async def interactive_chat_loop(
    chat, output_parser: StrOutputParser, config: HostConfig, chat_id: str
) -> None:
    """Run the interactive chat loop."""
    print("\nChat started. Type 'exit' or press Ctrl-C to quit.")
    print(f"Chat ID: {chat_id}")
    print(f"Model: {config.llm.model_provider}/{config.llm.model}")
    print("=" * 60)

    while True:
        try:
            # Read user input
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            # Process the query
            query = HumanMessage(content=user_input)
            print()
            await process_query(chat, query, output_parser)

        except EOFError:
            # Handle Ctrl-D
            print("\n\nGoodbye!")
            break


async def run() -> None:
    """dive_mcp_host CLI entrypoint."""
    # Print logo
    print_logo()

    args = setup_argument_parser()

    # Get initial query if provided
    initial_query = parse_query(args) if args.query else None

    # Load configuration
    config = get_config_from_args(args)

    # Load system prompt if provided
    system_prompt = None
    if args.prompt_file:
        with Path(args.prompt_file).open("r") as f:
            system_prompt = f.read()

    output_parser = StrOutputParser()

    try:
        async with DiveMcpHost(config) as mcp_host:
            print("Waiting for tools to initialize...")
            await mcp_host.tools_initialized_event.wait()
            print("Tools initialized")
            print("=" * 60)

            chat = mcp_host.chat(chat_id=args.chat_id, system_prompt=system_prompt)
            current_chat_id = chat.chat_id

            async with chat:
                # Process initial query if provided
                if initial_query:
                    await process_query(chat, initial_query, output_parser)

                # Start interactive chat loop
                await interactive_chat_loop(
                    chat, output_parser, config, current_chat_id
                )

    except KeyboardInterrupt:
        # Handle Ctrl-C
        print("\n\nGoodbye!")
        sys.exit(0)


async def show_loading(stop_event: asyncio.Event) -> None:
    """Show a loading animation until stop_event is set."""
    idx = 0
    while not stop_event.is_set():
        char = LOADING_CHARS[idx % len(LOADING_CHARS)]
        print(f"\r{char} Thinking...", end="", flush=True)
        idx += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.1)
        except TimeoutError:
            continue
    # Clear the loading line
    print("\r" + " " * 20 + "\r", end="", flush=True)


async def process_query(
    chat, query: HumanMessage, output_parser: StrOutputParser
) -> None:
    """Process a single query and print the response."""
    # Start loading animation
    stop_loading = asyncio.Event()
    loading_task = asyncio.create_task(show_loading(stop_loading))

    first_response = True
    try:
        async for response in chat.query(query, stream_mode="messages"):
            # Stop loading on first response
            if first_response:
                stop_loading.set()
                await loading_task
                first_response = False

            assert isinstance(response, tuple)
            msg = response[0]
            if isinstance(msg, AIMessage):
                content = output_parser.invoke(msg)
                print(content, end="", flush=True)
                continue
            print(f"\n\n==== Start Of {type(msg)} ===")
            print(msg)
            print(f"==== End Of {type(msg)} ===\n")
    finally:
        # Ensure loading is stopped
        if not stop_loading.is_set():
            stop_loading.set()
            await loading_task

    print()  # Add newline after response
