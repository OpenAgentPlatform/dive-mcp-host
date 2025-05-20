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
        headers={"Authorization": f"Bearer {oap_token}"},
        plan="free",
    )

    def mock_get_user_mcp(
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
    response = client.post(
        f"/api/plugins/oap-platform/auth?token={oap_token}",
    )
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
    response = client.post(
        f"/api/plugins/oap-platform/auth?token={oap_token}",
    )
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
