from collections.abc import AsyncGenerator
from pathlib import Path

import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic import SecretStr

from dive_mcp_host.httpd.app import create_app
from dive_mcp_host.httpd.conf.httpd_service import ConfigLocation, ServiceManager
from dive_mcp_host.httpd.conf.mcp_servers import Config, MCPServerConfig
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.oap_plugin.config_mcp_servers import (
    TOKEN_PLACEHOLDER,
)
from tests.helper import dict_subset
from tests.httpd.routers.conftest import ConfigFileNames, config_files  # noqa: F401


@pytest_asyncio.fixture
async def test_client(
    config_files: ConfigFileNames,  # noqa: F811
) -> AsyncGenerator[tuple[TestClient, DiveHostAPI], None]:
    """Create a server for testing."""
    service_manager = ServiceManager(config_files.service_config_file)
    service_manager.initialize()

    service_manager.overwrite_paths(
        ConfigLocation(
            plugin_config_path="plugin_config.json",
            mcp_server_config_path=config_files.mcp_server_config_file,
            model_config_path=config_files.model_config_file,
            prompt_config_path=config_files.prompt_config_file,
        )
    )

    Path("oap_config.json").unlink(missing_ok=True)

    app = create_app(service_manager)
    app.set_status_report_info(listen="127.0.0.1")
    app.set_listen_port(61990)
    with TestClient(app, raise_server_exceptions=False) as client:
        # create a simple chat
        client.get("/api/tools/initialized")

        yield client, app


def test_oap_plugin(test_client: tuple[TestClient, DiveHostAPI]):
    """Test the OAP plugin."""
    oap_token = "fake-token"  # noqa: S105
    client, _ = test_client
    echo_config = MCPServerConfig.model_validate(
        {
            "transport": "stdio",
            "enabled": True,
            "command": "python3",
            "args": ["-m", "dive_mcp_host.host.tools.echo", "--transport=stdio"],
            "env": {"NODE_ENV": "production"},
            "url": None,
            "extraData": None,
            "proxy": None,
            "headers": None,
            "exclude_tools": [],
            "initialTimeout": 10.0,
            "toolCallTimeout": 600.0,
        }
    )
    oap_extra_data = {
        "oap": {
            "id": "181672830075666436",
            "planTag": "pro",
            "description": "SearXNG is a ...",
        }
    }

    oap_token_set = MCPServerConfig.model_validate(
        {
            "transport": "streamable",
            "enabled": True,
            "url": "https://proxy.oaphub.ai/v1/mcp/181672830075666436",
            "extraData": oap_extra_data,
            "headers": {"Authorization": f"Bearer {oap_token}"},
            "exclude_tools": [],
        }
    )
    oap_header_need_token = MCPServerConfig(
        transport="streamable",
        enabled=True,
        url="https://proxy.oaphub.ai/v1/mcp/181672830075666436",
        extraData=oap_extra_data,
        headers={"Authorization": SecretStr(f"Bearer {TOKEN_PLACEHOLDER}")},
    )
    oap_env_need_token = MCPServerConfig(
        transport="stdio",
        command="python3",
        args=["-m", "dive_mcp_host.host.tools.echo", "--transport=stdio"],
        env={"OAP_MCP_TOKEN": TOKEN_PLACEHOLDER},
        extraData=oap_extra_data,
    )

    # Get current config
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200
    servers = response.json()["config"]["mcpServers"]
    assert len(servers) == 1
    assert dict_subset(servers, {"echo": echo_config.model_dump()})

    # Login
    response = client.post("/api/plugins/oap-platform/auth", json={"token": oap_token})
    assert response.status_code == 200

    # Add OAP MCP
    config = Config(
        mcpServers={
            "echo": echo_config,
            "oap_token_set": oap_token_set,
            "oap_header_need_token": oap_header_need_token,
            "oap_env_need_token": oap_env_need_token,
        }
    )
    response = client.post("/api/config/mcpserver", json=config.model_dump())
    assert response.status_code == 200

    # Get mcp server, device token should be correct
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200
    new_config = Config.model_validate(response.json()["config"])
    assert new_config.mcp_servers["echo"] == config.mcp_servers["echo"]
    # No need to change
    assert (
        new_config.mcp_servers["oap_token_set"] == config.mcp_servers["oap_token_set"]
    )
    # Headers should be set, becomes the same as the previous one
    assert (
        new_config.mcp_servers["oap_header_need_token"]
        == config.mcp_servers["oap_token_set"]
    )
    # Env should be set
    assert new_config.mcp_servers["oap_env_need_token"] == MCPServerConfig(
        transport="stdio",
        command="python3",
        args=["-m", "dive_mcp_host.host.tools.echo", "--transport=stdio"],
        env={"OAP_MCP_TOKEN": oap_token},
        extraData=oap_extra_data,
        headers=None,
    )

    # Logout
    response = client.delete("/api/plugins/oap-platform/auth")
    assert response.status_code == 200

    # Token should be removed
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200
    new_config_2 = Config.model_validate(response.json()["config"])

    assert new_config_2.mcp_servers["echo"] == config.mcp_servers["echo"]
    # Token should be replaced with placeholder
    assert new_config_2.mcp_servers["oap_token_set"] == MCPServerConfig.model_validate(
        {
            "transport": "streamable",
            "enabled": True,
            "url": "https://proxy.oaphub.ai/v1/mcp/181672830075666436",
            "extraData": oap_extra_data,
            "headers": {"Authorization": f"Bearer {TOKEN_PLACEHOLDER}"},
            "exclude_tools": [],
        }
    )
    assert (
        new_config_2.mcp_servers["oap_header_need_token"]
        == new_config_2.mcp_servers["oap_token_set"]
    )
    # Env token should be replaced with placeholder
    assert new_config_2.mcp_servers["oap_env_need_token"] == MCPServerConfig(
        transport="stdio",
        command="python3",
        args=["-m", "dive_mcp_host.host.tools.echo", "--transport=stdio"],
        env={"OAP_MCP_TOKEN": TOKEN_PLACEHOLDER},
        extraData=oap_extra_data,
        headers=None,
    )

    # login again
    new_oap_token = "new-fake-oap-token"  # noqa: S105
    response = client.post(
        "/api/plugins/oap-platform/auth", json={"token": new_oap_token}
    )
    assert response.status_code == 200

    # Get mcp server, token should be updated
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200
    new_config_3 = Config.model_validate(response.json()["config"])
    assert new_config_3.mcp_servers["echo"] == config.mcp_servers["echo"]
    # Token should be updated to new token
    assert new_config_3.mcp_servers["oap_token_set"] == MCPServerConfig.model_validate(
        {
            "transport": "streamable",
            "enabled": True,
            "url": "https://proxy.oaphub.ai/v1/mcp/181672830075666436",
            "extraData": oap_extra_data,
            "headers": {"Authorization": f"Bearer {new_oap_token}"},
            "exclude_tools": [],
        }
    )
    # Headers should be set with new token
    assert (
        new_config_3.mcp_servers["oap_header_need_token"]
        == new_config_3.mcp_servers["oap_token_set"]
    )
    # Env should be set with new token
    assert new_config_3.mcp_servers["oap_env_need_token"] == MCPServerConfig(
        transport="stdio",
        command="python3",
        args=["-m", "dive_mcp_host.host.tools.echo", "--transport=stdio"],
        env={"OAP_MCP_TOKEN": new_oap_token},
        extraData=oap_extra_data,
        headers=None,
    )
