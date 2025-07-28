import asyncio
import json
import os
import time
from contextlib import AbstractAsyncContextManager
from typing import Any, cast
from unittest import mock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
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
from dive_mcp_host.host.custom_events import ToolCallProgress
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
            assert len(responses) == 0, (
                "Query should have been aborted before completion"
            )
            # Check that there are no AIMessages in the responses
            for response in responses:
                messages = response.get("agent", {}).get("messages", [])
                assert not any(isinstance(msg, AIMessage) for msg in messages), (
                    "Should not have any AIMessages after abort"
                )

            # Verify the abort signal was cleared
            model.sleep = 0
            model.i = 0
            responses = [
                i
                async for i in chat.query(
                    "This is a long running query", stream_mode=["messages"]
                )
            ]
            assert len(responses) == 1
            # Verify that we have AIMessages in the responses
            _, (message) = cast(tuple[str, tuple[AIMessage]], responses[0])
            assert (
                message[0].content
                == "This is a long running response that should be aborted"
            )

            # abort a non-running chat
            chat.abort()
            responses = [
                i
                async for i in chat.query(
                    "This is a long running query", stream_mode=["messages"]
                )
            ]
            assert len(responses) == 1


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
        assert len(host.tools) == 2  # echo and ignore tools
        assert isinstance(host.config.llm, LLMConfig)
        assert host.config.llm.configuration is None

        # Perform reload
        await host.reload(new_config, mock_reloader)

        # Verify reloader was called
        assert reloader_called

        # Verify config was updated
        assert host.config.llm.model == "fake"

        # Verify tools were updated
        assert len(host.tools) == 3  # echo, fetch, and their ignore counterparts
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
        assert len(host.tools) == 3


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
async def test_custom_event(
    echo_tool_sse_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
) -> None:
    """Test the custom event."""
    async with (
        echo_tool_sse_server as (_, configs),
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
async def test_resend_after_abort(  # noqa: C901
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
