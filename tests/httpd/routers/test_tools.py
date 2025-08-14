import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from dive_mcp_host.host.tools.echo import ECHO_DESCRIPTION, IGNORE_DESCRIPTION
from dive_mcp_host.host.tools.log import LogEvent, LogMsg
from dive_mcp_host.host.tools.model_types import ClientState
from dive_mcp_host.httpd.conf.mcp_servers import MCPServerConfig
from dive_mcp_host.httpd.routers.models import SimpleToolInfo
from dive_mcp_host.httpd.routers.tools import McpTool, ToolsResult, list_tools
from dive_mcp_host.httpd.server import DiveHostAPI
from tests import helper


class MockTool:
    """Mock Tool class."""

    def __init__(self, name: str, description: str | None = None, enable: bool = True):
        self.name = name
        self.description = description
        self.enable = enable


class MockServerInfo:
    """Mock server info class."""

    def __init__(self, tools=None, error=None):
        self.tools = tools or []
        self.error_str = error


def test_initialized(test_client):
    """Test the GET endpoint."""
    client, _ = test_client
    response = client.get("/api/tools/initialized")
    assert response.status_code == status.HTTP_200_OK


def test_list_tools_no_mock(test_client):
    """Test the GET endpoint."""
    client, _ = test_client
    response = client.get("/api/tools")
    assert response.status_code == status.HTTP_200_OK

    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "tools": [
                {
                    "name": "echo",
                    "description": "",
                    "enabled": True,
                    "icon": "",
                    "error": None,
                    "tools": [
                        {
                            "name": "echo",
                            "description": ECHO_DESCRIPTION,
                        },
                        {
                            "name": "ignore",
                            "description": IGNORE_DESCRIPTION,
                        },
                    ],
                }
            ],
        },
    )


@patch(
    "dive_mcp_host.httpd.server.DiveHostAPI.local_file_cache", new_callable=PropertyMock
)
@patch(
    "dive_mcp_host.host.tools.ToolManager.mcp_server_info", new_callable=PropertyMock
)
def test_list_tools_mock_cache(
    mock_mcp_server_info, mock_local_file_cache, test_client
):
    """Test the GET endpoint without cache."""
    (client, _) = test_client
    mock_mcp_server_info.return_value = {
        "test_tool": MockServerInfo(
            tools=[
                MockTool(
                    name="test_tool", description="Test tool description", enable=True
                )
            ]
        )
    }
    mocked_cache = MagicMock()
    mocked_cache.get = MagicMock(return_value=None)
    mock_local_file_cache.return_value = mocked_cache
    response = client.get("/api/tools")
    assert response.status_code == status.HTTP_200_OK

    response_data = cast(dict[str, Any], response.json())
    response_data["tools"] = sorted(response_data["tools"], key=lambda x: x["name"])
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "tools": [
                {  # echo is still in server list, but it was not in mcp_server_info
                    "name": "echo",
                    "description": "",
                    "enabled": False,
                    "icon": "",
                    "tools": [],
                    "error": None,
                },
                {
                    "name": "test_tool",
                    "tools": [
                        {"name": "test_tool", "description": "Test tool description"}
                    ],
                    "description": "",
                    "enabled": True,
                    "icon": "",
                    "error": None,
                },
            ],
        },
    )


def test_tools_result_serialization():
    """Test that ToolsResult can be properly serialized."""
    # Create a sample response
    response = ToolsResult(
        success=True,
        message=None,
        tools=[
            McpTool(
                name="test_tool",
                tools=[SimpleToolInfo(name="test", description="Test function")],
                description="Test tool description",
                enabled=True,
                icon="test",
                error=None,
            ),
        ],
    )

    response_dict = response.model_dump(by_alias=True)

    helper.dict_subset(
        response_dict,
        {
            "success": True,
            "tools": [
                {
                    "name": "test_tool",
                    "description": "Test tool description",
                    "enabled": True,
                    "icon": "test",
                    "error": None,
                    "tools": [
                        {
                            "name": "test",
                            "description": "Test function",
                        },
                    ],
                },
            ],
        },
    )


@pytest.mark.asyncio
@patch("dive_mcp_host.httpd.routers.tools.list_tools")
async def test_list_tools_with_error(mock_list_tools, test_client):
    """Test list_tools function with server error."""
    _, app = test_client

    # Mock return value
    mock_list_tools.return_value = ToolsResult(
        success=True,
        message=None,
        tools=[
            McpTool(
                name="error_server",
                tools=[],
                description="",
                enabled=True,
                icon="",
                error="Test error",
            ),
        ],
    )

    response = await mock_list_tools(app)
    response_dict = response.model_dump(by_alias=True)

    helper.dict_subset(
        response_dict,
        {
            "success": True,
            "tools": [
                {
                    "name": "error_server",
                    "tools": [],
                    "description": "",
                    "enabled": True,
                    "icon": "",
                    "error": "Test error",
                },
            ],
        },
    )
    mock_list_tools.assert_called_once_with(app)


@pytest.mark.asyncio
@patch("dive_mcp_host.httpd.routers.tools.list_tools")
async def test_list_tools_with_no_config(mock_list_tools, test_client):
    """Test list_tools function with no configuration."""
    _, app = test_client

    # Mock return value
    mock_list_tools.return_value = ToolsResult(success=True, message=None, tools=[])

    # Call API endpoint test
    response = await mock_list_tools(app)

    # Verify results
    assert response.success is True
    assert isinstance(response.tools, list)
    assert len(response.tools) == 0

    # Verify mock was called
    mock_list_tools.assert_called_once_with(app)


@pytest.mark.asyncio
@patch(
    "dive_mcp_host.httpd.conf.mcp_servers.MCPServerManager.get_current_config",
)
async def test_list_tools_with_missing_server_not_in_cache(
    mock_current_config,
    test_client,
):
    """Test list_tools function with missing server not in cache."""
    _, app = test_client

    # Create Mock configuration
    config_mock = AsyncMock()
    config_mock.mcp_servers = {
        "missing_server": MCPServerConfig(command="123"),
    }
    mock_current_config.return_value = config_mock

    response = await list_tools(app)
    response_dict = response.model_dump(by_alias=True)
    response_dict["tools"] = sorted(response_dict["tools"], key=lambda x: x["name"])
    helper.dict_subset(
        response_dict,
        {
            "success": True,
            "tools": [
                {
                    "name": "echo",
                    "description": "",
                    "enabled": True,
                    "icon": "",
                    "error": None,
                    "tools": [
                        {
                            "name": "echo",
                            "description": ECHO_DESCRIPTION,
                        },
                        {
                            "name": "ignore",
                            "description": IGNORE_DESCRIPTION,
                        },
                    ],
                },
                {
                    "name": "missing_server",
                    "tools": [],
                    "description": "",
                    "enabled": False,
                    "icon": "",
                    "error": None,
                },
            ],
        },
    )


def test_empty_tools_result():
    """Test empty ToolsResult serialization."""
    response = ToolsResult(success=True, message=None, tools=[])
    response_dict = response.model_dump(by_alias=True)

    assert "success" in response_dict
    assert response_dict["success"] is True
    assert "tools" in response_dict
    assert isinstance(response_dict["tools"], list)
    assert len(response_dict["tools"]) == 0

    assert "message" in response_dict
    assert response_dict["message"] is None


def test_tools_cache_after_update(test_client):
    """Test that tools cache is updated after various config updates."""
    client, _ = test_client
    conf = {
        "mcpServers": {
            "echo": {
                "transport": "stdio",
                "enabled": True,
                "command": "python",
                "args": ["-m", "dive_mcp_host.host.tools.echo", "--transport=stdio"],
            },
            "missing_server": {
                "transport": "stdio",
                "enabled": True,
                "command": "no-such-command",
            },
        }
    }
    assert (
        client.post("/api/config/mcpserver", json=conf).status_code
        == status.HTTP_200_OK
    )
    response = client.get("/api/tools")
    assert response.status_code == status.HTTP_200_OK
    first_time = cast(dict[str, Any], response.json())
    # we can have 2 tools even missing_server is failed to load
    first_time["tools"] = sorted(first_time["tools"], key=lambda x: x["name"])
    assert len(first_time["tools"]) == 2

    conf = {
        "mcpServers": {
            "echo": {
                "transport": "stdio",
                "enabled": False,
                "command": "python",
                "args": ["-m", "dive_mcp_host.host.tools.echo", "--transport=stdio"],
            },
            "missing_server": {
                "transport": "stdio",
                "enabled": False,
                "command": "no-such-command",
            },
        }
    }
    assert (
        client.post("/api/config/mcpserver", json=conf).status_code
        == status.HTTP_200_OK
    )
    response = client.get("/api/tools")
    assert response.status_code == status.HTTP_200_OK
    # Even when all servers are disabled, we can still see them from the cache
    for tool in first_time["tools"]:
        tool["enabled"] = False  # all servers are disabled
    confirm = response.json()
    confirm["tools"] = sorted(confirm["tools"], key=lambda x: x["name"])
    assert first_time == confirm


def test_stream_logs_notfound(test_client: tuple[TestClient, DiveHostAPI]):
    """Test stream_logs function with not found server."""
    client, _ = test_client
    response = client.get(
        "/api/tools/logs/stream", params={"server_name": "missing_server"}
    )
    for line in response.iter_lines():
        content = line.removeprefix("data: ")
        if content in ("[DONE]", ""):
            continue

        data = LogMsg.model_validate_json(content)
        assert data.event == LogEvent.STREAMING_ERROR
        assert data.body == "Error streaming logs: Log buffer missing_server not found"
        assert data.mcp_server_name == "missing_server"


def test_stream_logs_notfound_wait(test_client: tuple[TestClient, DiveHostAPI]):
    """Test stream_logs before log buffer is registered."""
    client, _ = test_client

    def update_tools():
        time.sleep(2)
        _ = client.post(
            "/api/config/mcpserver",
            json={
                "mcpServers": {
                    "missing_server": {
                        "transport": "stdio",
                        "enabled": True,
                        "command": "python",
                        "args": [
                            "-m",
                            "dive_mcp_host.host.tools.echo",
                            "--transport=stdio",
                        ],
                    }
                }
            },
        )
        assert response.status_code == status.HTTP_200_OK

    with ThreadPoolExecutor(1) as executor:
        executor.submit(update_tools)
        response = client.get(
            "/api/tools/logs/stream",
            params={
                "server_name": "missing_server",
                "stop_on_notfound": False,
                "max_retries": 5,
                "stream_until": "running",
            },
        )
        responses: list[LogMsg] = []
        for line in response.iter_lines():
            content = line.removeprefix("data: ")
            if content in ("[DONE]", ""):
                continue

            data = LogMsg.model_validate_json(content)
            responses.append(data)

        assert len(responses) >= 3
        assert responses[-3].event == LogEvent.STREAMING_ERROR

        assert responses[-2].event == LogEvent.STDERR
        assert responses[-2].client_state == ClientState.INIT

        assert responses[-1].event == LogEvent.STATUS_CHANGE
        assert responses[-1].client_state == ClientState.RUNNING


def test_stream_logs_name_with_slash(test_client: tuple[TestClient, DiveHostAPI]):
    """Test stream_logs before log buffer is registered."""
    client, _ = test_client

    def update_tools():
        _ = client.post(
            "/api/config/mcpserver",
            json={
                "mcpServers": {
                    "name/with/slash": {
                        "transport": "stdio",
                        "enabled": True,
                        "command": "python",
                        "args": [
                            "-m",
                            "dive_mcp_host.host.tools.echo",
                            "--transport=stdio",
                        ],
                    }
                }
            },
        )
        assert response.status_code == status.HTTP_200_OK

    with ThreadPoolExecutor(1) as executor:
        executor.submit(update_tools)
        response = client.get(
            "/api/tools/logs/stream",
            params={
                "server_name": "name/with/slash",
                "stop_on_notfound": False,
                "max_retries": 5,
                "stream_until": "running",
            },
        )
        responses: list[LogMsg] = []
        for line in response.iter_lines():
            content = line.removeprefix("data: ")
            if content in ("[DONE]", ""):
                continue

            data = LogMsg.model_validate_json(content)
            responses.append(data)

        assert len(responses) >= 3
        assert responses[-3].event == LogEvent.STREAMING_ERROR

        assert responses[-2].event == LogEvent.STDERR
        assert responses[-2].client_state == ClientState.INIT

        assert responses[-1].event == LogEvent.STATUS_CHANGE
        assert responses[-1].client_state == ClientState.RUNNING
