import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dive_mcp_host.env import DIVE_CONFIG_DIR
from dive_mcp_host.httpd.conf.mcp_servers import (
    CurrentConfigHookName,
    UpdateAllConfigsHookName,
)
from dive_mcp_host.oap_plugin.config_mcp_servers import MCPServerManagerPlugin
from dive_mcp_host.oap_plugin.http_handlers import OAPHttpHandlers
from dive_mcp_host.plugins.registry import PluginCallbackDef


def get_static_callbacks() -> dict[str, tuple[Callable[..., Any], PluginCallbackDef]]:
    """Get the static callbacks."""
    with Path(DIVE_CONFIG_DIR, "oap_config.json").open("r") as f:
        oap_config = json.load(f)

    mcp_plugin = MCPServerManagerPlugin(oap_config["auth_key"])
    handlers = OAPHttpHandlers(mcp_plugin)

    return {
        "get_mcp_configs": (
            mcp_plugin.current_config_callback,
            PluginCallbackDef(
                hook_point=CurrentConfigHookName, callback="get_mcp_configs"
            ),
        ),
        "update_all_configs": (
            mcp_plugin.update_all_config_callback,
            PluginCallbackDef(
                hook_point=UpdateAllConfigsHookName, callback="update_all_configs"
            ),
        ),
        "http_routes": (
            handlers.get_router,
            PluginCallbackDef(hook_point="httpd.routers", callback="http_routes"),
        ),
    }
