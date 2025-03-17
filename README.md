# Dive MCP Host

Dive MCP Host is a language model host service based on the Model Context Protocol (MCP), providing a unified interface to manage and interact with various language models.

## Purpose of the Program

Dive MCP Host offers the following features:

- A unified language model interaction interface, supporting multiple models (such as OpenAI, Anthropic, Google, etc.)
- Conversation management and persistent storage
- HTTP API and WebSocket support
- Command-line tools for quick testing and interaction
- Support for multi-threaded conversations and user management

This project uses LangChain and LangGraph to build and manage language model workflows, providing a standardized way to interact with different language models.

## How to Run

### Environment Setup

1. Ensure you have Python 3.12 or higher installed
2. Clone this repository
3. Install dependencies:

```bash
# Using pip
pip install -e .

# Or using uv pip
uv pip install -e .

# Or using uv sync (recommended, will respect uv.lock file) 
uv sync
```

### Starting the HTTP Service

Use the following command to start the HTTP service:

```bash
dive_httpd
```

This will start a FastAPI server, providing REST API and WebSocket endpoints.

### Using the Command Line Tool

You can use the command line tool for quick testing:

```bash
# General conversation
dive_cli "Hello"

# Resume a conversation with a specific thread
dive_cli -c THREADID "How are you?"
```

### Using in Code

```python
from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host import DiveMcpHost

# Initialize configuration
config = HostConfig(...)

# Use async context manager
async with DiveMcpHost(config) as host:
    # Start or resume a conversation
    async with host.conversation(thread_id="123") as conversation:
        # Send a query and get a response
        async for response in conversation.query("Hello, how can you help me today?"):
            print(response)
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```
or
```bash
uv sync --extra dev
```

(Optional) Start local PostgreSQL
```
./scripts/run_pg.sh
```

Run tests:

```bash
pytest
```
or with uv, (no need to activate enviroment) 
```bash
uv run --extra dev pytest
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
