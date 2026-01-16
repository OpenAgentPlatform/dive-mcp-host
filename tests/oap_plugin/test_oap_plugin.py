from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from dive_mcp_host.httpd.app import create_app
from dive_mcp_host.httpd.conf.httpd_service import ConfigLocation, ServiceManager
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.oap_plugin.config_mcp_servers import MCPServerManagerPlugin
from dive_mcp_host.oap_plugin.models import UserMcpConfig
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


def test_oap_plugin(
    test_client: tuple[TestClient, DiveHostAPI], monkeypatch: pytest.MonkeyPatch
):
    """Test the OAP plugin."""
    oap_token = "fake-token"  # noqa: S105

    config = UserMcpConfig(
        id="19181672830075666433",
        name="Fake Mcp Server",
        description="a fake mcp server, for testing",
        transport="sse",
        url="http://127.0.0.1:3260",
        headers={},
        plan="free",
    )

    async def mock_get_user_mcp(
        self: MCPServerManagerPlugin, *args: Any, **kwargs: Any
    ) -> list[UserMcpConfig] | None:
        """Mock the get_user_mcp method."""
        if not self.device_token:
            return None

        return [config]

    monkeypatch.setattr(
        "dive_mcp_host.oap_plugin.config_mcp_servers.MCPServerManagerPlugin._get_user_mcp_configs",
        mock_get_user_mcp,
    )

    client, _ = test_client
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) == 1

    # load mcp token
    response = client.post("/api/plugins/oap-platform/auth", json={"token": oap_token})
    assert response.status_code == 200

    # get mcp server
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) > 1

    # disable oap mcp and check info is present
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            value["enabled"] = False

    response = client.post(
        "/api/config/mcpserver",
        json={"mcpServers": servers},
    )
    assert response.status_code == 200

    # get mcp server
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) > 1

    # check oap mcp is disabled
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            assert value["headers"] == {"Authorization": f"Bearer {oap_token}"}
            assert value["enabled"] is False

    # enable oap mcp
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            value["enabled"] = True

    response = client.post(
        "/api/config/mcpserver",
        json={"mcpServers": servers},
    )
    assert response.status_code == 200

    # check oap mcp is enabled
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            assert value["headers"] == {"Authorization": f"Bearer {oap_token}"}
            assert value["enabled"] is True

    # drop mcp token (logout)
    response = client.delete(
        "/api/plugins/oap-platform/auth",
    )
    assert response.status_code == 200

    # check oap mcp is disabled
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) == 1

    # login again
    response = client.post("/api/plugins/oap-platform/auth", json={"token": oap_token})
    assert response.status_code == 200

    # get mcp server
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) > 1

    config.name = "Fake Mcp Server (updated)"
    # refresh mcp server
    response = client.post(
        "/api/plugins/oap-platform/config/refresh",
    )
    assert response.status_code == 200

    # get mcp server
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) > 1

    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            assert key == config.name
            assert value["headers"] == {"Authorization": f"Bearer {oap_token}"}

    config.auth_type = "oauth2"
    # refresh mcp server
    response = client.post(
        "/api/plugins/oap-platform/config/refresh",
    )
    assert response.status_code == 200

    # get mcp server
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) > 1

    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            assert key == config.name
            assert value["headers"] is None


def test_oap_plugin_external_endpoint_http(
    test_client: tuple[TestClient, DiveHostAPI], monkeypatch: pytest.MonkeyPatch
):
    """Test the OAP plugin with external endpoint (HTTP/SSE)."""
    oap_token = "fake-token"  # noqa: S105

    config = UserMcpConfig(
        id="19181672830075666433",
        name="HubSpot MCP",
        description="HubSpot MCP Server with external endpoint",
        transport="sse",
        url="http://127.0.0.1:3260",
        headers={},
        plan="free",
        external_endpoint={
            "url": "https://mcp.hubspot.com/",
            "headers": {"Authorization": "Bearer ${HUBSPOT_ACCESS_TOKEN}"},
            "protocol": "streamable",
        },
    )

    async def mock_get_user_mcp(
        self: MCPServerManagerPlugin, *args: Any, **kwargs: Any
    ) -> list[UserMcpConfig] | None:
        """Mock the get_user_mcp method."""
        if not self.device_token:
            return None

        return [config]

    monkeypatch.setattr(
        "dive_mcp_host.oap_plugin.config_mcp_servers.MCPServerManagerPlugin._get_user_mcp_configs",
        mock_get_user_mcp,
    )

    client, _ = test_client

    # Login with oap token
    response = client.post("/api/plugins/oap-platform/auth", json={"token": oap_token})
    assert response.status_code == 200

    # Get mcp server config
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) > 1

    # Find the HubSpot MCP server
    hubspot_server = None
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            hubspot_server = value
            assert key == config.name
            assert value["url"] == "https://mcp.hubspot.com/"
            assert value["transport"] == "streamable"
            assert value["headers"]["Authorization"] == "Bearer ${HUBSPOT_ACCESS_TOKEN}"
            assert value["enabled"] is True
            break

    assert hubspot_server is not None

    # Update the config with new external endpoint URL
    from dive_mcp_host.oap_plugin.models import ExternalEndpointHttp

    config.external_endpoint = ExternalEndpointHttp(
        url="https://mcp.hubspot.com/v2/",
        headers={"Authorization": "Bearer ${HUBSPOT_ACCESS_TOKEN}"},
        protocol="streamable",
    )

    # Refresh mcp server
    response = client.post("/api/plugins/oap-platform/config/refresh")
    assert response.status_code == 200

    # Get mcp server again
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            assert key == config.name
            assert value["url"] == "https://mcp.hubspot.com/v2/"
            assert value["transport"] == "streamable"
            assert value["enabled"] is True


def test_oap_plugin_external_endpoint_stdio(
    test_client: tuple[TestClient, DiveHostAPI], monkeypatch: pytest.MonkeyPatch
):
    """Test the OAP plugin with external endpoint (stdio/command)."""
    oap_token = "fake-token"  # noqa: S105

    config = UserMcpConfig(
        id="19181672830075666434",
        name="File Uploader MCP",
        description="File uploader MCP Server with external endpoint",
        transport="sse",
        url="http://127.0.0.1:3261",
        headers={},
        plan="free",
        external_endpoint={
            "command": "npx",
            "args": ["@oaphub/file-uploader-mcp"],
            "env": {"OAP_CLIENT_KEY": "{{AccessToken}}"},
            "protocol": "stdio",
        },
    )

    async def mock_get_user_mcp(
        self: MCPServerManagerPlugin, *args: Any, **kwargs: Any
    ) -> list[UserMcpConfig] | None:
        """Mock the get_user_mcp method."""
        if not self.device_token:
            return None

        return [config]

    monkeypatch.setattr(
        "dive_mcp_host.oap_plugin.config_mcp_servers.MCPServerManagerPlugin._get_user_mcp_configs",
        mock_get_user_mcp,
    )

    client, _ = test_client

    # Login with oap token
    response = client.post("/api/plugins/oap-platform/auth", json={"token": oap_token})
    assert response.status_code == 200

    # Get mcp server config
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) > 1

    # Find the File Uploader MCP server
    file_uploader_server = None
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            file_uploader_server = value
            assert key == config.name
            assert value["command"] == "npx"
            assert value["args"] == ["@oaphub/file-uploader-mcp"]
            assert value["env"]["OAP_CLIENT_KEY"] == oap_token  # Should be replaced
            assert value["transport"] == "stdio"
            assert value["enabled"] is True
            break

    assert file_uploader_server is not None

    # Update the config with new external endpoint args
    from dive_mcp_host.oap_plugin.models import ExternalEndpointCMD

    config.external_endpoint = ExternalEndpointCMD(
        command="npx",
        args=["@oaphub/file-uploader-mcp", "--verbose"],
        env={"OAP_CLIENT_KEY": "{{AccessToken}}"},
        protocol="stdio",
    )

    # Refresh mcp server
    response = client.post("/api/plugins/oap-platform/config/refresh")
    assert response.status_code == 200

    # Get mcp server again
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            assert key == config.name
            assert value["command"] == "npx"
            assert value["args"] == ["@oaphub/file-uploader-mcp", "--verbose"]
            assert value["env"]["OAP_CLIENT_KEY"] == oap_token
            assert value["transport"] == "stdio"
            assert value["enabled"] is True


def test_oap_plugin_external_endpoint_mixed(
    test_client: tuple[TestClient, DiveHostAPI], monkeypatch: pytest.MonkeyPatch
):
    """Test the OAP plugin with multiple servers including external endpoints."""
    oap_token = "fake-token"  # noqa: S105

    configs = [
        UserMcpConfig(
            id="19181672830075666433",
            name="HubSpot MCP",
            description="HubSpot MCP Server with external endpoint",
            transport="sse",
            url="http://127.0.0.1:3260",
            headers={},
            plan="free",
            external_endpoint={
                "url": "https://mcp.hubspot.com/",
                "headers": {"Authorization": "Bearer ${HUBSPOT_ACCESS_TOKEN}"},
                "protocol": "streamable",
            },
        ),
        UserMcpConfig(
            id="19181672830075666434",
            name="File Uploader MCP",
            description="File uploader MCP Server with external endpoint",
            transport="sse",
            url="http://127.0.0.1:3261",
            headers={},
            plan="free",
            external_endpoint={
                "command": "npx",
                "args": ["@oaphub/file-uploader-mcp"],
                "env": {"OAP_CLIENT_KEY": "{{AccessToken}}"},
                "protocol": "stdio",
            },
        ),
        UserMcpConfig(
            id="19181672830075666435",
            name="Regular MCP",
            description="Regular MCP Server without external endpoint",
            transport="sse",
            url="http://127.0.0.1:3262",
            headers={},
            plan="free",
        ),
    ]

    async def mock_get_user_mcp(
        self: MCPServerManagerPlugin, *args: Any, **kwargs: Any
    ) -> list[UserMcpConfig] | None:
        """Mock the get_user_mcp method."""
        if not self.device_token:
            return None

        return configs

    monkeypatch.setattr(
        "dive_mcp_host.oap_plugin.config_mcp_servers.MCPServerManagerPlugin._get_user_mcp_configs",
        mock_get_user_mcp,
    )

    client, _ = test_client

    # Login with oap token
    response = client.post("/api/plugins/oap-platform/auth", json={"token": oap_token})
    assert response.status_code == 200

    # Get mcp server config
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    assert len(servers) >= 3

    # Verify all servers are configured correctly
    found_hubspot = False
    found_uploader = False
    found_regular = False

    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            if key == "HubSpot MCP":
                found_hubspot = True
                assert value["url"] == "https://mcp.hubspot.com/"
                assert value["transport"] == "streamable"
            elif key == "File Uploader MCP":
                found_uploader = True
                assert value["command"] == "npx"
                assert value["args"] == ["@oaphub/file-uploader-mcp"]
                assert value["env"]["OAP_CLIENT_KEY"] == oap_token
                assert value["transport"] == "stdio"
            elif key == "Regular MCP":
                found_regular = True
                assert value["url"] == "http://127.0.0.1:3262"
                assert value["transport"] == "sse"
                assert value["headers"]["Authorization"] == f"Bearer {oap_token}"

    assert found_hubspot
    assert found_uploader
    assert found_regular

    # Disable the HubSpot MCP and enable others
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            if key == "HubSpot MCP":
                value["enabled"] = False
            else:
                value["enabled"] = True

    response = client.post("/api/config/mcpserver", json={"mcpServers": servers})
    assert response.status_code == 200

    # Verify the settings are saved
    response = client.get("/api/config/mcpserver")
    assert response.status_code == 200

    servers = response.json()["config"]["mcpServers"]
    for key in servers:
        value = servers[key]
        if value["extraData"] and value["extraData"].get("oap"):
            if key == "HubSpot MCP":
                assert value["enabled"] is False
            else:
                assert value["enabled"] is True
