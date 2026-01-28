import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast
from unittest import mock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import AnyUrl, SecretStr

from dive_mcp_host.host.chat import Chat
from dive_mcp_host.host.conf import CheckpointerConfig, HostConfig
from dive_mcp_host.host.conf.llm import LLMConfig
from dive_mcp_host.host.custom_events import (
    ToolAuthenticationRequired,
    ToolCallProgress,
)
from dive_mcp_host.host.errors import ThreadNotFoundError, ThreadQueryError
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.host.tools import ServerConfig
from dive_mcp_host.models.fake import FakeMessageToolModel, default_responses


@pytest.fixture
def echo_tool_stdio_config() -> dict[str, ServerConfig]:  # noqa: D103
    return {
        "echo": ServerConfig(
            name="echo",
            command="python3",
            args=[
                "-m",
                "dive_mcp_host.host.tools.echo",
                "--transport=stdio",
            ],
            transport="stdio",
        ),
    }


@pytest.mark.asyncio
async def test_host_context() -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers={},
    )
    espect_responses = default_responses()
    # prompt = ChatPromptTemplate.from_messages(
    #     [("system", "You are a helpful assistant."), ("placeholder", "{messages}")],
    # )
    async with DiveMcpHost(config) as mcp_host:
        chat = mcp_host.chat()
        async with chat:
            responses = [
                response["agent"]["messages"][0]
                async for response in chat.query(
                    "Hello, world!",
                    stream_mode=None,
                )
                if response.get("agent")
            ]
            for res, expect in zip(responses, espect_responses, strict=True):
                assert res.content == expect.content  # type: ignore[attr-defined]
        chat = mcp_host.chat()
        async with chat:
            responses = [
                response["agent"]["messages"][0]
                async for response in chat.query(
                    HumanMessage(content="Hello, world!"),
                    stream_mode=None,
                )
                if response.get("agent")
            ]
            for res, expect in zip(responses, espect_responses, strict=True):
                assert res.content == expect.content  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_query_two_messages() -> None:
    """Test that the query method can handle two or more messages."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers={},
    )
    async with DiveMcpHost(config) as mcp_host, mcp_host.chat() as chat:
        responses = [
            response
            async for response in chat.query(
                [
                    HumanMessage(content="Attachment"),
                    HumanMessage(content="Hello, world!"),
                ],
                stream_mode=["values"],
            )
        ]
        for i in responses:
            human_messages = [
                i
                for i in i[1]["messages"]  # type: ignore[index]
                if isinstance(i, HumanMessage)
            ]
            assert len(human_messages) == 2
            assert human_messages[0].content == "Attachment"
            assert human_messages[1].content == "Hello, world!"


@pytest.mark.asyncio
async def test_get_messages(
    sqlite_uri, echo_tool_stdio_config: dict[str, ServerConfig]
) -> None:
    """Test the get_messages."""
    user_id = "default"
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
        checkpointer=CheckpointerConfig(uri=AnyUrl(sqlite_uri)),
    )

    async with DiveMcpHost(config) as mcp_host:
        fake_responses = [
            AIMessage(
                content="Call echo tool",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world! 許個願望吧"},
                        id="123",
                        type="tool_call",
                    ),
                ],
            ),
        ]
        cast("FakeMessageToolModel", mcp_host.model).responses = fake_responses
        await mcp_host.tools_initialized_event.wait()
        chat = mcp_host.chat()
        async with chat:
            async for _ in chat.query(
                HumanMessage(content="Hello, world! 許個願望吧"),
                stream_mode=["messages"],
            ):
                pass

            chat_id = chat.chat_id
            messages = await mcp_host.get_messages(chat_id, user_id)
            assert len(messages) > 0

            human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
            assert any(
                msg.content == "Hello, world! 許個願望吧" for msg in human_messages
            )

            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            assert any(msg.content == "Call echo tool" for msg in ai_messages)

            tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
            assert tool_messages[0].name == "echo"
            assert (
                json.loads(str(tool_messages[0].content))[0]["text"]
                == "Hello, world! 許個願望吧"
            )

            with pytest.raises(ThreadNotFoundError):
                _ = await mcp_host.get_messages("non-existent-thread-id", user_id)

            messages = await mcp_host.get_messages(chat_id, user_id)
            assert len(messages) > 0
            for i, msg in enumerate(
                [msg for msg in messages if isinstance(msg, HumanMessage)]
            ):
                match msg.content:
                    case "Hello, world! 許個願望吧":
                        assert i == 0, "First message should be the first message"
                    case "Second message":
                        assert i == 1, "Second message should be the second message"
                    case _:
                        raise ValueError(f"Unexpected message: {msg.content}")


@pytest.mark.asyncio
async def test_callable_system_prompt() -> None:
    """Test that the system prompt can be a callable."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers={},
    )
    msgs = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Line 1!"),
    ]

    mock_system_prompt = MagicMock(return_value=msgs)

    async with (
        DiveMcpHost(config) as mcp_host,
        mcp_host.chat(system_prompt=mock_system_prompt, volatile=True) as chat,
    ):
        assert mcp_host.model is not None
        model = cast("FakeMessageToolModel", mcp_host.model)
        async for _ in chat.query(msgs):
            ...
        assert len(model.query_history) == 2
        assert model.query_history[0].content == "You are a helpful assistant."
        assert model.query_history[1].content == "Line 1!"

        assert mock_system_prompt.call_count == 1
        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Line 2!"),
        ]
        model.query_history = []
        mock_system_prompt.reset_mock()
        mock_system_prompt.return_value = msgs

        async for _ in chat.query(msgs):
            ...
        assert len(model.query_history) == 2
        assert model.query_history[0].content == "You are a helpful assistant."
        assert model.query_history[1].content == "Line 2!"
        assert mock_system_prompt.call_count == 1


@pytest.mark.asyncio
async def test_abort_chat() -> None:
    """Test that the chat can be aborted during a long-running query."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers={},
    )

    # Create a fake model with a long sleep time to simulate a long-running query
    fake_responses = [
        AIMessage(content="This is a long running response that should be aborted"),
    ]

    async with DiveMcpHost(config) as mcp_host:
        model = cast("FakeMessageToolModel", mcp_host.model)
        model.disable_streaming = False  # enable streaming
        model.responses = fake_responses
        model.sleep = 2.0  # 2 seconds sleep to simulate long running query

        chat = mcp_host.chat()
        async with chat:
            # Start the query in a separate task
            async def _query() -> list[dict[str, Any]]:
                return [
                    i
                    async for i in chat.query(
                        "This is a long running query", stream_mode=["messages"]
                    )
                ]

            query_task = asyncio.create_task(_query())

            # Wait a bit and then abort
            await asyncio.sleep(0.5)
            chat.abort()

            # Wait for the query task to complete
            async with asyncio.timeout(5):
                await query_task
            assert query_task.exception() is None
            responses = query_task.result()

            # Verify that we got fewer responses than expected and no AIMessages
            assert len(responses) == 3
            assert responses[0][1][0].content == "This"
            assert responses[1][1][0].content == "<user_aborted>"
            assert responses[2][1][0].chunk_position == "last"

            # Verify the abort signal was cleared
            model.sleep = 0
            model.i = 0
            responses = [
                i
                async for i in chat.query(
                    "This is a long running query", stream_mode=["messages"]
                )
            ]
            assert (
                len(responses) == len(fake_responses[0].content.split(" ")) + 1
            )  # last AIMessage with chunk_position = "last"
            assert (
                sum(
                    [chunk[1][0] for chunk in responses],
                    start=AIMessageChunk(content=""),
                ).content
                == fake_responses[0].content
            )

            # abort a non-running chat
            chat.abort()
            responses = [
                i
                async for i in chat.query(
                    "This is a long running query", stream_mode=["messages"]
                )
            ]
            assert len(responses) == len(fake_responses[0].content.split(" ")) + 1


@pytest.mark.asyncio
async def test_abort_chat_with_tools(
    echo_tool_stdio_config: dict[str, ServerConfig],
) -> None:
    """Test that the chat can be aborted during a long-running query with tools."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
    )

    # Create a fake model with a long sleep time to simulate a long-running query
    fake_responses = [
        [
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_KHVa8wD16tglUbtgEkM8XdqC",
                            "function": {"arguments": "", "name": "echo"},
                            "type": "function",
                        }
                    ]
                },
                response_metadata={},
                id="run--5a261755-5ae5-4a9d-91e7-9ec4909a6652",
                tool_calls=[
                    {
                        "name": "echo",
                        "args": {},
                        "id": "call_KHVa8wD16tglUbtgEkM8XdqC",
                        "type": "tool_call",
                    }
                ],
                tool_call_chunks=[
                    {
                        "name": "echo",
                        "args": "",
                        "id": "call_KHVa8wD16tglUbtgEkM8XdqC",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"arguments": '{"', "name": None},
                            "type": None,
                        }
                    ]
                },
                response_metadata={},
                id="run--5a261755-5ae5-4a9d-91e7-9ec4909a6652",
                tool_calls=[{"name": "", "args": {}, "id": None, "type": "tool_call"}],
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": '{"',
                        "id": None,
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"arguments": "message", "name": None},
                            "type": None,
                        }
                    ]
                },
                response_metadata={},
                id="run--5a261755-5ae5-4a9d-91e7-9ec4909a6652",
                invalid_tool_calls=[
                    {
                        "name": None,
                        "args": "message",
                        "id": None,
                        "error": None,
                        "type": "invalid_tool_call",
                    }
                ],
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": "message",
                        "id": None,
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"arguments": '":"', "name": None},
                            "type": None,
                        }
                    ]
                },
                response_metadata={},
                id="run--5a261755-5ae5-4a9d-91e7-9ec4909a6652",
                invalid_tool_calls=[
                    {
                        "name": None,
                        "args": '":"',
                        "id": None,
                        "error": None,
                        "type": "invalid_tool_call",
                    }
                ],
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": '":"',
                        "id": None,
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"arguments": "hello", "name": None},
                            "type": None,
                        }
                    ]
                },
                response_metadata={},
                id="run--5a261755-5ae5-4a9d-91e7-9ec4909a6652",
                invalid_tool_calls=[
                    {
                        "name": None,
                        "args": "hello",
                        "id": None,
                        "error": None,
                        "type": "invalid_tool_call",
                    }
                ],
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": "hello",
                        "id": None,
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"arguments": " world", "name": None},
                            "type": None,
                        }
                    ]
                },
                response_metadata={},
                id="run--5a261755-5ae5-4a9d-91e7-9ec4909a6652",
                invalid_tool_calls=[
                    {
                        "name": None,
                        "args": " world",
                        "id": None,
                        "error": None,
                        "type": "invalid_tool_call",
                    }
                ],
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": " world",
                        "id": None,
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"arguments": '"}', "name": None},
                            "type": None,
                        }
                    ]
                },
                response_metadata={},
                id="run--5a261755-5ae5-4a9d-91e7-9ec4909a6652",
                invalid_tool_calls=[
                    {
                        "name": None,
                        "args": '"}',
                        "id": None,
                        "error": None,
                        "type": "invalid_tool_call",
                    }
                ],
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": '"}',
                        "id": None,
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={},
                response_metadata={
                    "finish_reason": "tool_calls",
                    "model_name": "gpt-4o-mini-2024-07-18",
                    "system_fingerprint": "fp_560af6e559",
                    "service_tier": "default",
                },
                id="run--5a261755-5ae5-4a9d-91e7-9ec4909a6652",
            ),
        ],
        [
            AIMessageChunk(content="tool"),
            AIMessageChunk(content=" call"),
            AIMessageChunk(content=" completed"),
        ],
    ]

    async with DiveMcpHost(config) as mcp_host:
        await mcp_host.tools_initialized_event.wait()
        model = cast("FakeMessageToolModel", mcp_host.model)
        model.disable_streaming = False  # enable streaming
        model.responses = fake_responses
        model.sleep = 0.2

        chat = mcp_host.chat()
        async with chat:
            # Start the query in a separate task
            async def _query() -> list[dict[str, Any]]:
                return [
                    i
                    async for i in chat.query(
                        "call echo tool",
                        stream_mode=["messages"],
                    )
                ]

            query_task = asyncio.create_task(_query())

            # Wait a bit and then abort
            await asyncio.sleep(0.5)
            chat.abort()

            # Wait for the query task to complete
            async with asyncio.timeout(5):
                await query_task
            assert query_task.exception() is None
            responses = query_task.result()

            # Verify that we got fewer responses than expected and no AIMessages
            assert len(responses) == 6
            assert responses[-2][1][0].chunk_position == "last"
            assert responses[-1][1][0].content == "canceled"


@pytest.mark.asyncio
async def test_abort_tool_call(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the get_messages."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
    )

    async with DiveMcpHost(config) as mcp_host:
        fake_responses = [
            AIMessage(
                content="Call echo tool",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world!", "delay_ms": 5000},
                        id="123",
                        type="tool_call",
                    ),
                ],
            ),
            AIMessage(content="Bye"),
        ]
        cast("FakeMessageToolModel", mcp_host.model).responses = fake_responses
        await mcp_host.tools_initialized_event.wait()
        chat = mcp_host.chat()
        async with chat:

            async def _query() -> list[dict[str, Any]]:
                return [
                    i
                    async for i in chat.query("Hello, world!", stream_mode=["messages"])
                ]

            query_task = asyncio.create_task(_query())
            await asyncio.sleep(0.5)
            chat.abort()
            await query_task
            assert query_task.exception() is None
            responses = query_task.result()
            assert len(responses) == 2

            tool_message = responses[-1][1][0]
            assert isinstance(tool_message, ToolMessage)
            assert "<user_aborted>" in tool_message.content


@pytest.mark.asyncio
async def test_resend_message(sqlite_uri: str) -> None:
    """Test the resend_message method."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers={},
        checkpointer=CheckpointerConfig(uri=AnyUrl(sqlite_uri)),
    )

    async with DiveMcpHost(config) as mcp_host:
        chat = mcp_host.chat()
        model = cast("FakeMessageToolModel", mcp_host.model)
        async with chat:
            resps = cast(
                list[tuple[str, dict[str, list[BaseMessage]]]],
                [
                    i
                    async for i in chat.query(
                        HumanMessage(content="Hello, world!"),
                        stream_mode=["values"],
                    )
                ],
            )
            _, msgs = resps[-1]
            assert isinstance(msgs["messages"][0], HumanMessage)
            human_message_id = msgs["messages"][0].id
            assert msgs["messages"][0].content == "Hello, world!"
            assert isinstance(msgs["messages"][1], AIMessage)
            ai_message_id = msgs["messages"][1].id
            assert msgs["messages"][1].content == model.responses[0].content

            model.responses = [AIMessage(content="2")]
            resend = [HumanMessage(content="Resend message!", id=human_message_id)]
            resps = cast(
                list[tuple[str, dict[str, list[BaseMessage]]]],
                [
                    i
                    async for i in chat.query(
                        resend,  # type: ignore
                        stream_mode=["values"],
                        is_resend=True,
                    )
                ],
            )
            _, msgs = resps[-1]
            assert len(msgs["messages"]) == 2
            assert msgs["messages"][0].content == "Resend message!"
            assert msgs["messages"][0].id == human_message_id
            assert msgs["messages"][1].content == "2"
            assert msgs["messages"][1].id != ai_message_id


@pytest.mark.asyncio
async def test_host_reload(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host reload functionality."""
    # Initial configuration
    initial_config = HostConfig(
        llm=LLMConfig(
            model="gpt-4o",
            model_provider="openai",
            api_key=SecretStr("fake"),
        ),
        mcp_servers=echo_tool_stdio_config,
    )

    # New configuration with different model settings
    new_config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers={
            "echo": ServerConfig(
                name="echo",
                command="python3",
                args=["-m", "dive_mcp_host.host.tools.echo", "--transport=stdio"],
                transport="stdio",
            ),
            # Added new server
            "fetch": ServerConfig(
                name="fetch",
                command="uvx",
                args=["mcp-server-fetch"],
                transport="stdio",
            ),
        },
    )

    # Mock reloader function
    reloader_called = False

    async def mock_reloader() -> None:
        nonlocal reloader_called
        reloader_called = True

    # Test reload functionality
    async with DiveMcpHost(initial_config) as host:
        await host.tools_initialized_event.wait()

        # Verify initial state
        # echo, elicit, ignore, raise_elicit, url_elicit tools
        assert len(host.tools) == 5
        assert isinstance(host.config.llm, LLMConfig)
        assert host.config.llm.configuration is None

        # Perform reload
        await host.reload(new_config, mock_reloader)

        # Verify reloader was called
        assert reloader_called

        # Verify config was updated
        assert host.config.llm.model == "fake"

        # Verify tools were updated
        # echo, elicit, ignore, raise_elicit, url_elicit + fetch
        assert len(host.tools) == 6
        tool_names = [tool.name for tool in host.tools]
        assert "echo" in tool_names
        assert "fetch" in tool_names

        # Test chat still works after reload
        async with host.chat() as chat:
            responses = []
            async for response in chat.query("Hello"):
                responses.append(response)

            assert len(responses) > 0

        # Test reload with same config
        reloader_called = False
        await host.reload(new_config, mock_reloader)
        assert reloader_called
        # echo, elicit, ignore, raise_elicit, url_elicit + fetch
        assert len(host.tools) == 6


@pytest.mark.asyncio
async def test_thread_query_error_with_state(sqlite_uri: str) -> None:
    """Test that ThreadQueryError includes state when DEBUG is enabled."""
    """Test the resend_message method."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers={},
        checkpointer=CheckpointerConfig(uri=AnyUrl(sqlite_uri)),
    )
    delete_debug = False
    if "DEBUG" not in os.environ:
        os.environ["DEBUG"] = "1"
        delete_debug = True

    async with DiveMcpHost(config) as mcp_host:
        chat = mcp_host.chat()
        chat_id = chat.chat_id
        model = cast("FakeMessageToolModel", mcp_host.model)

        # Make the model raise an exception by using an invalid tool call
        model.responses = [
            AIMessage(
                content="Hello",
            )
        ]

        async with chat:
            async for _ in chat.query("Hello, world!"):
                pass

        async with mcp_host.chat(chat_id=chat_id) as chat:
            with mock.patch.object(model, "_generate") as mock_responses:
                mock_responses.side_effect = Exception("Test exception")

                with pytest.raises(ThreadQueryError) as exc_info:
                    async for _ in chat.query("Hello, world!"):
                        pass

            # Verify error contains state
            assert exc_info.value.state_values is not None
            assert "messages" in exc_info.value.state_values
            assert len(exc_info.value.state_values["messages"]) > 0

            # Verify the query is preserved
            assert exc_info.value.query == "Hello, world!"

    # Clean up DEBUG environment
    if delete_debug:
        del os.environ["DEBUG"]


@pytest.mark.asyncio
async def test_custom_event_streamable(
    echo_tool_streamable_server: tuple[int, dict[str, ServerConfig]],
) -> None:
    """Test the custom event."""
    _, configs = echo_tool_streamable_server
    async with (
        DiveMcpHost(
            HostConfig(
                llm=LLMConfig(
                    model="fake",
                    model_provider="dive",
                ),
                mcp_servers=configs,
            )
        ) as mcp_host,
    ):
        await mcp_host.tools_initialized_event.wait()
        chat = mcp_host.chat()
        model = cast("FakeMessageToolModel", mcp_host.model)
        model.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world!", "delay_ms": 1000},
                        id="123",
                        type="tool_call",
                    ),
                ],
            ),
            AIMessage(
                content="Bye",
            ),
        ]
        done = False
        async with chat:
            async for i in chat.query(
                HumanMessage(content="Hello, world!"),
                stream_mode=["custom", "messages"],
            ):
                i = cast(tuple[str, Any], i)
                if i[0] == "custom":
                    assert isinstance(i[1], tuple)
                    assert i[1][0] == "tool_call_progress"
                    assert isinstance(i[1][1], ToolCallProgress)
                    done = True
        assert done


@pytest.mark.asyncio
async def test_custom_event(
    echo_tool_sse_server: tuple[int, dict[str, ServerConfig]],
) -> None:
    """Test the custom event."""
    _, configs = echo_tool_sse_server
    async with (
        DiveMcpHost(
            HostConfig(
                llm=LLMConfig(
                    model="fake",
                    model_provider="dive",
                ),
                mcp_servers=configs,
            )
        ) as mcp_host,
    ):
        await mcp_host.tools_initialized_event.wait()
        chat = mcp_host.chat()
        model = cast("FakeMessageToolModel", mcp_host.model)
        model.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world!", "delay_ms": 1000},
                        id="123",
                        type="tool_call",
                    ),
                ],
            ),
            AIMessage(
                content="Bye",
            ),
        ]
        done = False
        async with chat:
            async for i in chat.query(
                HumanMessage(content="Hello, world!"),
                stream_mode=["custom", "messages"],
            ):
                i = cast(tuple[str, Any], i)
                if i[0] == "custom":
                    assert isinstance(i[1], tuple)
                    assert i[1][0] == "tool_call_progress"
                    assert isinstance(i[1][1], ToolCallProgress)
                    done = True
        assert done


@pytest.mark.asyncio
async def test_resend_after_abort(
    echo_tool_stdio_config: dict[str, ServerConfig],
) -> None:
    """Test that resend after abort works."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
    )

    async with DiveMcpHost(config) as mcp_host:
        mcp_host._checkpointer = InMemorySaver()
        fake_responses = [
            AIMessage(
                content="Call echo tool",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world!", "delay_ms": 2000},
                        id="phase1-tool-1",
                        type="tool_call",
                    ),
                ],
                id="phase1-1",
            ),
            AIMessage(
                content="Bye",
                id="phase1-2",
            ),
        ]

        ts = time.time()
        got_msg = False

        async def _abort_task(chat: Chat) -> None:
            while True:
                if got_msg and time.time() - ts > 0.5:
                    chat.abort()
                    break
                await asyncio.sleep(0.1)

        fake_model = cast("FakeMessageToolModel", mcp_host.model)
        fake_model.responses = fake_responses
        await mcp_host.tools_initialized_event.wait()
        chat = mcp_host.chat(chat_id="chat_id")
        async with chat:
            task = asyncio.create_task(_abort_task(chat))
            async for r in chat.query(
                HumanMessage(content="Hello, world!", id="H1"),
                stream_mode=["messages", "values", "updates"],
            ):
                if r[0] == "messages":  # type: ignore
                    got_msg = True
                    ts = time.time()
        await task

        resend = [HumanMessage(content="Resend message!", id="H1")]
        fake_responses = [
            AIMessage(
                content="Call echo tool",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world!"},
                        id="phase2-tool-1",
                        type="tool_call",
                    ),
                ],
                id="phase2-1",
            ),
            AIMessage(
                content="Bye",
                id="phase2-2",
            ),
        ]
        fake_model.i = 0
        fake_model.responses = fake_responses
        chat = mcp_host.chat(chat_id="chat_id")
        async with chat:
            async for r in chat.query(
                resend,  # type: ignore
                is_resend=True,
            ):
                r = cast(tuple[str, list[BaseMessage]], r)
                if r[0] == "messages":
                    for m in r[1]:
                        assert m.id
                        if isinstance(m, HumanMessage):
                            assert m.id == "H1"
                        elif isinstance(m, AIMessage):
                            assert m.id.startswith("phase2-")


@pytest.mark.asyncio
async def test_oauth_required_event(
    weather_tool_streamable_server: tuple[
        int, dict[str, ServerConfig], Callable[[str], Awaitable[tuple[str, str]]]
    ],
) -> None:
    """Test the oauth required event during tool call."""
    _, configs, get_auth_code = weather_tool_streamable_server
    async with (
        DiveMcpHost(
            HostConfig(
                llm=LLMConfig(
                    model="fake",
                    model_provider="dive",
                ),
                mcp_servers=configs,
            )
        ) as mcp_host,
    ):
        await mcp_host.tools_initialized_event.wait()
        mcp_server = mcp_host.get_mcp_server("weather")
        progress = await mcp_server.create_oauth_authorization()

        assert progress.auth_url

        code, state = await get_auth_code(progress.auth_url)

        await mcp_host.oauth_manager.set_oauth_code(
            code=code,
            state=state,
        )

        await mcp_host.oauth_manager.wait_authorization(
            state=progress.state,  # type: ignore
            timeout=10,
        )
        assert await mcp_host.oauth_manager.store.list() == ["weather"]

        await mcp_host.restart_mcp_server("weather")
        await mcp_host.oauth_manager.store.delete("weather")
        assert await mcp_host.oauth_manager.store.list() == []

        chat = mcp_host.chat()
        model = cast("FakeMessageToolModel", mcp_host.model)
        model.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="get_weather",
                        args={"city": "Tokyo"},
                        id="123",
                        type="tool_call",
                    ),
                ],
            ),
            AIMessage(
                content="Bye",
            ),
        ]
        async with chat:
            async for i in chat.query(
                "Hello, world!", stream_mode=["messages", "custom"]
            ):
                i = cast(tuple[str, Any], i)
                if i[0] == "custom" and i[1][0] == "tool_authentication_required":
                    assert isinstance(i[1], tuple)
                    assert i[1][0] == "tool_authentication_required"
                    assert isinstance(i[1][1], ToolAuthenticationRequired)
                    assert i[1][1].server_name == "weather"
                    has_auth_required = True
                    code, state = await get_auth_code(i[1][1].auth_url)
                    await mcp_host.oauth_manager.set_oauth_code(
                        code=code,
                        state=state,
                    )
                if i[0] == "messages" and isinstance(i[1][0], ToolMessage):
                    completed_tool_call = True
        assert has_auth_required
        assert completed_tool_call
        assert await mcp_host.oauth_manager.store.list() == ["weather"]
