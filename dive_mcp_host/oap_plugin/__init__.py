import json
from collections.abc import Callable
from typing import Any

from dive_mcp_host.httpd.conf.mcp_servers import (
    Config,
    CurrentConfigHookName,
    MCPServerConfig,
)
from dive_mcp_host.oap_plugin.config_mcp_servers import MCPServerManagerPlugin
from dive_mcp_host.plugins.registry import PluginCallbackDef


def get_static_callbacks() -> dict[str, tuple[Callable[..., Any], PluginCallbackDef]]:
    """Get the static callbacks."""

    with open("oap_config.json", "r") as f:
        oap_config = json.load(f)

    mcp_plugin = MCPServerManagerPlugin(oap_config["auth_key"])

    return {
        "get_mcp_configs": (
            mcp_plugin.current_config_callback,
            PluginCallbackDef(
                hook_point=CurrentConfigHookName, callback="get_mcp_configs"
            ),
        )
    }
