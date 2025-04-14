# Dive HTTPD

## Description

Dive HTTPD is a FastAPI-based HTTP server component that provides an API interface for the Dive MCP system. It serves as the main entry point for handling HTTP requests and managing various aspects of the Dive MCP host, including:

- Chat and conversation management
- OpenAI API compatibility
- MCP-Server management and execution
- Configuration management
- Model verification
- Message storage and caching

The server is designed to be modular and extensible, with support for:
- Middleware-based request processing
- Database integration for message storage
- Local file caching
- Configuration management for services, models, and prompts
- Asynchronous request handling

## Startup Parameters

Configuration loading priority:
1. Command line arguments
2. Configuration files (dive_httpd.json)

### Environment Variables
- `DIVE_CONFIG_DIR`: Directory containing configuration files
  - Default: Current working directory (CWD)
  - Description: Base directory for all configuration files
- `RESOURCE_DIR`: Directory for resource files
  - Default: Current working directory (CWD)
  - Description: Contains upload files and cache files

### Command Line Arguments

#### Configuration Files
- `--config <path>`: Main service configuration file
  - Default: `${DIVE_CONFIG_DIR}/dive_httpd.json`
  - Description: Specifies the location of the main service configuration file
- `--model_config <path>`: Model configuration file
  - Default: `${DIVE_CONFIG_DIR}/model_config.json`
  - Description: Configuration for model settings and parameters
- `--mcp_config <path>`: MCP server configuration file
  - Default: `${DIVE_CONFIG_DIR}/mcp_config.json`
  - Description: Configuration for MCP server settings
- `--custom_rules <path>`: Custom rules file
  - Default: `${DIVE_CONFIG_DIR}/custom_rules`
  - Description: Custom rules for LLM
- `--command_alias_config <path>`: Command alias configuration file
  - Default: `${DIVE_CONFIG_DIR}/command_alias.json`
  - Description: Configuration for command aliases

#### Server Settings
- `--listen <address>`: Server binding address
  - Default: `127.0.0.1`
  - Description: Network interface to bind the server to
- `--port <number>`: Server port number
  - Default: `61990`
  - Description: TCP port number to listen on. Use 0 for automatic port selection
- `--auto_reload`: Enable configuration auto-reload
  - Default: `false`
  - Description: Automatically reload configurations when changes are detected
- `--cors_origin <origin>`: CORS origin
  - Default: Disabled
  - Description: CORS origin to allow, use full url, e.g. `http://127.0.0.1:1234`

#### Directory and Status Settings
- `--working_dir <directory>`: Working directory
  - Default: Current working directory
  - Description: Base directory for server operations
- `--report_status-file <file>`: Status report file
  - Default: Disabled
  - Description: File path to write server status information
- `--report_status-fd <fd>`: Status report file descriptor
  - Default: Disabled
  - Description: File descriptor to write server status information

#### Logging Settings
- `--log_level <level>`: Log level
  - Default: `INFO`
  - Description: Log level
- `--log_dir <directory>`: Log directory
  - Default: Disabled
  - Description: log file directory

### Configuration File Structure

#### Dive Httpd
TODO

#### Model Config
TODO

#### MCP Config
TODO

#### Command Alias
TODO

### Report Status

The server can report its status in JSON Lines format to either a file or file descriptor. Each status report is a single JSON object containing the following information:

```json
{
    "timestamp": "2024-03-27T10:00:00Z",
    "server": {
        "listen": {
            "ip": "127.0.0.1",
            "port": 61990
        }
    },
    "status": {
        "state": "UP|FAILED",
        "last_error": "Error message if any",
        "error_code": "ERROR_CODE"
    }
}
```

#### Status Fields

- `timestamp`: ISO 8601 formatted timestamp of the status report
- `server`: Server information
  - `listen`: Network binding details
- `status`: Server status information
  - `state`: Overall server state (UP/FAILED)
  - `last_error`: Last error message if any
  - `error_code`: Error code if applicable
