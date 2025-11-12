# Dive MCP Host CLI

A command-line interface tool for testing and interacting with the Dive MCP Host.

## Usage

### Basic Chat

```bash
# Simple conversation
dive_cli "Hello"

# Continue from a previous chat
dive_cli -c CHAT_ID "How are you?"
```

## Configuration Options

The CLI supports multiple ways to specify configuration:

### Option 1: Single Configuration File

```bash
dive_cli --config /path/to/config.json "query"
```

Use a single configuration file that contains both LLM and MCP server settings.

### Option 2: Configuration Directory

```bash
dive_cli --config-dir /path/to/configs "query"
```

Specify a directory containing `mcp_config.json` and `model_config.json`. The CLI will automatically load both files from this directory.

### Option 3: Separate Configuration Files

```bash
dive_cli --mcp-config /path/to/mcp_config.json --model-config /path/to/model_config.json "query"
```

Explicitly specify paths for MCP server and model configuration files.

### Option 4: Default Configuration

```bash
dive_cli "query"
```

If no configuration options are provided, the CLI will use `mcp_config.json` and `model_config.json` from the current directory.

## Command Line Flags

- `--config PATH`: Path to a single configuration file
- `--config-dir PATH`: Directory containing mcp_config.json and model_config.json
- `--mcp-config PATH`: Path to MCP servers configuration file (default: mcp_config.json)
- `--model-config PATH`: Path to model configuration file (default: model_config.json)
- `-c CHAT_ID`: Continue from a previous chat session
- `-p PATH`: Use a system prompt from the specified file

## Configuration Priority

1. `--config` (if provided, uses single configuration file)
2. `--config-dir` (if provided, loads fixed filenames from the directory)
3. `--mcp-config` and `--model-config` (if provided, uses specified files)
4. Default (uses mcp_config.json and model_config.json from current directory)

## Configuration File Format

### Model Config (model_config.json)

```json
{
  "activeProvider": "ollama",
  "configs": {
    "openai": {
      "modelProvider": "openai",
      "model": "gpt-4o-mini",
      "apiKey": "your_api_key"
    },
    "ollama": {
      "modelProvider": "ollama",
      "model": "qwen2.5:14b",
      "configuration": {
        "baseURL": "https://ollama.example.com"
      }
    }
  }
}
```

The CLI will use the configuration specified by `activeProvider`.

### MCP Config (mcp_config.json)

```json
{
  "mcpServers": {
    "server-name": {
      "transport": "command",
      "command": "uvx",
      "args": ["package@latest"]
    }
  }
}
```

The CLI automatically adds a `name` field to each server configuration using its key.

## Migration

`migrate.py`: Used for database migration when upgrading from older versions.
