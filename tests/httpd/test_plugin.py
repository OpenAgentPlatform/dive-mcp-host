import json
import tempfile
from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest_asyncio
from fastapi import APIRouter, Request, Response
from fastapi.testclient import TestClient

from dive_mcp_host.host.helpers.context import ContextProtocol
from dive_mcp_host.httpd.conf.httpd_service import ConfigLocation, ServiceManager
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.plugins.registry import Callbacks
from tests.httpd.routers.conftest import ConfigFileNames, config_files  # noqa: F401


@pytest_asyncio.fixture
async def server(config_files: ConfigFileNames) -> AsyncGenerator[DiveHostAPI, None]:  # noqa: F811
    """Create a server for testing."""
    with tempfile.NamedTemporaryFile(
        prefix="testPluginConfig_", suffix=".json"
    ) as plugin_config_file:
        plugin_config_file.write(
            json.dumps(
                [
                    {
                        "name": "test_plugin",
                        "config": {"header": "header_plugin"},
                        "module": "tests.httpd.test_plugin",
                        "ctx_manager": "tests.httpd.test_plugin.HttpdPlugin",
                        "static_callbacks": "tests.httpd.test_plugin.static_callbacks",
                    }
                ]
            ).encode("utf-8")
        )
        plugin_config_file.flush()
        service_config_manager = ServiceManager(config_files.service_config_file)
        service_config_manager.initialize()
        service_config_manager.overwrite_paths(
            ConfigLocation(
                plugin_config_path=plugin_config_file.name,
            )
        )
        server = DiveHostAPI(service_config_manager)
        async with server.prepare():
            yield server


class HttpdPlugin(ContextProtocol):
    """Middleware plugin."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize method."""
        self.config = config

    def callbacks(self) -> Callbacks:
        """Get the callbacks."""
        return {
            "hook1": (
                self._extra_input_plugin,
                "httpd.middlewares",
            ),
            "hook2": (
                self._extra_output_plugin,
                "httpd.middlewares",
            ),
        }

    async def _extra_input_plugin(self, request: Request, call_next: Callable):
        """Extra input plugin."""
        request.state.extra_input = self.config["header"]
        return await call_next(request)

    async def _extra_output_plugin(self, request: Request, call_next: Callable):
        """Extra output plugin."""
        response: Response = await call_next(request)
        response.headers["X-Custom-Header"] = request.state.extra_input
        return response


plugin_router = APIRouter()


@plugin_router.get("/")
async def plugin_router_get(request: Request):
    """Plugin router get."""
    return {"message": "Hello, plugin!"}


def static_callbacks():
    """Static callbacks."""
    return {
        "routers": (
            lambda: plugin_router,
            "httpd.routers",
        )
    }


def test_plugin_middleware(server: DiveHostAPI):
    """Test plugin middleware."""
    with TestClient(server) as client:
        response = client.get("/")
        assert response.status_code == 404
        assert response.headers["X-Custom-Header"] == "header_plugin"


def test_plugin_router(server: DiveHostAPI):
    """Test plugin router."""
    with TestClient(server) as client:
        response = client.get("/api/plugins/test_plugin/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello, plugin!"}
