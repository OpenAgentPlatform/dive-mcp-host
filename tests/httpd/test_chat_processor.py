import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage

from dive_mcp_host.httpd.conf.httpd_service import ServiceManager
from dive_mcp_host.httpd.conf.mcp_servers import Config
from dive_mcp_host.httpd.conf.prompt import PromptKey
from dive_mcp_host.httpd.routers.utils import ChatProcessor
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.models.fake import FakeMessageToolModel  # noqa: TC001
from tests.httpd.routers.conftest import config_files  # noqa: F401


@pytest_asyncio.fixture
async def server(config_files) -> AsyncGenerator[DiveHostAPI, None]:  # noqa: F811
    """Create a server for testing."""
    service_config_manager = ServiceManager(config_files.service_config_file)
    service_config_manager.initialize()
    server = DiveHostAPI(service_config_manager)
    await server.mcp_server_config_manager.update_all_configs(Config(mcpServers={}))
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
async def test_prompt(processor: ChatProcessor, monkeypatch: pytest.MonkeyPatch):
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


def test_strip_title():
    """Test the strip_title function."""
    from dive_mcp_host.httpd.routers.utils import strip_title

    # Test basic whitespace normalization
    assert strip_title("  hello   world  ") == "hello world"
    assert strip_title("hello\nworld") == "hello world"
    assert strip_title("hello\tworld") == "hello world"

    # Test HTML tag removal
    assert (
        strip_title("  <think>I'm thinking about\nhelloworld</think>hello world\t")
        == "hello world"
    )
    # Test empty or whitespace-only input
    assert strip_title("hello world") == "hello world"
    assert strip_title("  hello world  ") == "hello world"


@pytest.mark.asyncio
async def test_generate_title(processor: ChatProcessor):
    """Test the title function."""
    model = cast("FakeMessageToolModel", processor.dive_host.model)
    model.responses = [
        AIMessage(
            content="Simple Greeting",
        ),
        AIMessage(content=[{"type": "text", "text": "Simple Greeting 2", "index": 0}]),
    ]
    r = await processor._generate_title("Hello, how are you?")
    assert r == "Simple Greeting"
    r = await processor._generate_title("Hello, how are you?")
    assert r == "Simple Greeting 2"
