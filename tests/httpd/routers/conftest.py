import tempfile
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic import AnyUrl

from dive_mcp_host.host.conf import CheckpointerConfig
from dive_mcp_host.host.conf.llm import LLMConfiguration
from dive_mcp_host.httpd.app import DiveHostAPI, create_app
from dive_mcp_host.httpd.conf.httpd_service import (
    ConfigLocation,
    DBConfig,
    ServiceConfig,
    ServiceManager,
)
from dive_mcp_host.httpd.conf.mcp_servers import Config, MCPServerConfig
from dive_mcp_host.httpd.routers.models import ModelFullConfigs

TEST_CHAT_ID = "41a81e8c-ed6d-4d93-8988-c8763f7b3e30"


@dataclass(slots=True)
class ConfigFileNames:
    """Config file names."""

    service_config_file: str
    mcp_server_config_file: str
    model_config_file: str
    prompt_config_file: str


@pytest.fixture
def config_files() -> Generator[ConfigFileNames, None, None]:
    """Create config files."""
    with (
        tempfile.NamedTemporaryFile(
            prefix="testServiceConfig_", suffix=".json"
        ) as service_config_file,
        tempfile.NamedTemporaryFile(
            prefix="testMcpServerConfig_", suffix=".json"
        ) as mcp_server_config_file,
        tempfile.NamedTemporaryFile(
            prefix="testModelConfig_", suffix=".json"
        ) as model_config_file,
        tempfile.NamedTemporaryFile(suffix=".testCustomrules") as prompt_config_file,
        tempfile.NamedTemporaryFile(suffix=".sqlite") as db_file,
        tempfile.TemporaryDirectory() as resource_dir,
    ):
        service_config_file.write(
            ServiceConfig(
                db=DBConfig(
                    uri=f"sqlite:///{db_file.name}",
                ),
                checkpointer=CheckpointerConfig(
                    uri=AnyUrl(f"sqlite:///{db_file.name}"),
                ),
                config_location=ConfigLocation(
                    mcp_server_config_path=mcp_server_config_file.name,
                    model_config_path=model_config_file.name,
                    prompt_config_path=prompt_config_file.name,
                ),
                resource_dir=Path(resource_dir),
            )
            .model_dump_json(by_alias=True)
            .encode("utf-8")
        )
        service_config_file.flush()

        mcp_server_config_file.write(
            Config(
                mcpServers={
                    "echo": MCPServerConfig(
                        transport="stdio",
                        command="python3",
                        args=[
                            "-m",
                            "dive_mcp_host.host.tools.echo",
                            "--transport=stdio",
                        ],
                        env={"NODE_ENV": "production"},
                    ),
                },
            )
            .model_dump_json(by_alias=True)
            .encode("utf-8")
        )
        mcp_server_config_file.flush()

        model_config_file.write(
            ModelFullConfigs.model_validate(
                {
                    "activeProvider": "dive",
                    "enableTools": True,
                    "configs": {
                        "dive": {
                            "modelProvider": "dive",
                            "model": "fake",
                            "configuration": LLMConfiguration(),
                        },
                    },
                }
            )
            .model_dump_json(by_alias=True)
            .encode("utf-8")
        )
        model_config_file.flush()

        prompt_config_file.write(b"testCustomrules")
        prompt_config_file.flush()

        yield ConfigFileNames(
            service_config_file=service_config_file.name,
            mcp_server_config_file=mcp_server_config_file.name,
            model_config_file=model_config_file.name,
            prompt_config_file=prompt_config_file.name,
        )


@pytest_asyncio.fixture
async def test_client(
    config_files: ConfigFileNames,
) -> AsyncGenerator[tuple[TestClient, DiveHostAPI], None]:
    """Create a test client with fake model.

    This fixture creates a test client with a DiveHostAPI instance. The DiveHostAPI
    instance can be used to mock methods and test router endpoints.
    The fixture yields both the test client and app instance to allow access to both
    during testing.

    Returns:
        A tuple of the test client and the app.
    """
    service_manager = ServiceManager(config_files.service_config_file)
    service_manager.initialize()
    app = create_app(service_manager)
    app.set_status_report_info(listen="127.0.0.1")
    app.set_listen_port(61990)
    with TestClient(app, raise_server_exceptions=False) as client:
        # create a simple chat
        client.get("/api/tools/initialized")
        client.post(
            "/api/chat", data={"message": "Hello, world!", "chatId": TEST_CHAT_ID}
        )

        yield client, app
