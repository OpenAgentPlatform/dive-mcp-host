import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio

from dive_mcp_host.httpd.conf.httpd_service import ServiceManager
from dive_mcp_host.httpd.conf.prompt import PromptKey
from dive_mcp_host.httpd.routers.utils import ChatProcessor, HumanMessage
from dive_mcp_host.httpd.server import DiveHostAPI
from tests.httpd.routers.conftest import config_files  # noqa: F401


@pytest_asyncio.fixture
async def server(config_files) -> AsyncGenerator[DiveHostAPI, None]:  # noqa: F811
    """Create a server for testing."""
    service_config_manager = ServiceManager(config_files.service_config_file)
    service_config_manager.initialize()
    server = DiveHostAPI(service_config_manager)
    async with server.prepare():
        yield server


@pytest_asyncio.fixture
async def processor(server: DiveHostAPI) -> ChatProcessor:
    """Create a processor for testing."""

    class State:
        dive_user: dict[str, str]

    state = State()
    state.dive_user = {"user_id": "default"}
    return ChatProcessor(server, state, EmptyStream())  # type: ignore


class EmptyStream:
    """Empty stream."""

    async def write(self, *args: Any, **kwargs: Any) -> None:
        """Write data to the stream."""


@pytest.mark.asyncio
async def test_prompt(processor: ChatProcessor, monkeypatch):
    """Test the chat processor."""
    server = processor.app

    custom_rules = "You are a helpful assistant."
    server.prompt_config_manager.write_custom_rules(custom_rules)
    server.prompt_config_manager.update_prompts()
    prompt = server.prompt_config_manager.get_prompt(PromptKey.SYSTEM)

    mock_called = False

    def mock_chat(*args: Any, **kwargs: Any):
        nonlocal mock_called
        mock_called = True
        if system_prompt := kwargs.get("system_prompt"):
            assert system_prompt == prompt

    monkeypatch.setattr(server.dive_host["default"], "chat", mock_chat)

    chat_id = str(uuid.uuid4())
    user_message = HumanMessage(content="Hello, how are you?")
    with pytest.raises(AttributeError):
        await processor.handle_chat_with_history(
            chat_id,
            user_message,
            [],
        )

    assert mock_called
