import tempfile
import uuid
from collections.abc import AsyncGenerator
from hashlib import md5
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage

from dive_mcp_host.httpd.conf.httpd_service import ServiceManager
from dive_mcp_host.httpd.conf.mcp_servers import Config
from dive_mcp_host.httpd.conf.prompt import PromptKey
from dive_mcp_host.httpd.routers.utils import ChatProcessor, ContentHandler
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.httpd.store.manager import StoreManager
from tests.httpd.routers.conftest import config_files  # noqa: F401

if TYPE_CHECKING:
    from dive_mcp_host.models.fake import FakeMessageToolModel


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


@pytest.mark.asyncio
async def test_content_handler_gemini_image_with_url():
    """Check if content handler can extract what is needed."""
    store = StoreManager()
    store.save_base64_image = AsyncMock(
        return_value=["/some/path", "http://someurl.com"]
    )
    content_handler = ContentHandler(store)
    message = AIMessage(
        content=[
            "Here is a cuddly cat wearing a hat! ",
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,XXXXXXXX"},
            },
        ],
        response_metadata={"model_name": "gemini-2.5-flash-image-preview"},
    )
    content = await content_handler.invoke(message)
    assert (
        content == "Here is a cuddly cat wearing a hat!   ![image](http://someurl.com)"
    )

    # Cache should exist
    md5_hash = md5(b"XXXXXXXX", usedforsecurity=False).hexdigest()
    assert md5_hash in content_handler._cache
    assert content_handler._cache[md5_hash] == ["/some/path", "http://someurl.com"]


@pytest.mark.asyncio
async def test_content_handler_gemini_image_with_local_file():
    """Make sure local file also works."""
    with tempfile.NamedTemporaryFile(prefix="dummyfile-") as f:
        store = StoreManager()
        store.save_base64_image = AsyncMock(return_value=[f.name])
        content_handler = ContentHandler(store)
        message = AIMessage(
            content=[
                "Here is a cuddly cat wearing a hat! ",
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,XXXXXXXX"},
                },
            ],
            response_metadata={"model_name": "gemini-2.5-flash-image-preview"},
        )
        content = await content_handler.invoke(message)
        assert (
            content
            == f"Here is a cuddly cat wearing a hat!   ![image](file://{f.name})"
        )

        # Cache should exist
        md5_hash = md5(b"XXXXXXXX", usedforsecurity=False).hexdigest()
        assert md5_hash in content_handler._cache
        assert content_handler._cache[md5_hash] == [f.name]
