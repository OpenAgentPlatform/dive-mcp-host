"""MCP Server configuration management in OAP Plugin."""

from dive_mcp_host.httpd.conf.mcp_servers import Config


class MCPServerManagerPlugin:
    """Manage MCP Server configurations in OAP Plugin."""

    def __init__(self, device_token: str = "") -> None:
        """Initialize the MCPServerConfigs from OAP."""
        self.device_token: str = device_token
        self._configs: Config
        self._refresh_ts: float = 0

    def update_device_token(self, device_token: str) -> None:
        """Update the device token and refresh the configs."""
        self.device_token = device_token
        self.refresh()

    def refresh(self) -> None:
        """Refresh the MCP server configs."""

    def update_all_config_callback(self, new_config: Config) -> Config:
        """Callback function for updating all configs."""
        return new_config

    def current_config_callback(self, config: Config) -> Config:
        """Callback function for getting current config."""
        return config
