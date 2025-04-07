import asyncio
import json
from typing import Any, cast
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
from pydantic import AnyUrl

from dive_mcp_host.host.conf import CheckpointerConfig, HostConfig
from dive_mcp_host.host.conf.llm import LLMConfig
from dive_mcp_host.host.errors import ThreadNotFoundError
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
            modelProvider="dive",
        ),
        mcp_servers={},
    )
    espect_responses = default_responses()
    # prompt = ChatPromptTemplate.from_messages(
    #     [("system", "You are a helpful assistant."), ("placeholder", "{messages}")],
    # )
    async with DiveMcpHost(config) as mcp_host:
        conversation = mcp_host.conversation()
        async with conversation:
            responses = [
                response["agent"]["messages"][0]
                async for response in conversation.query(
                    "Hello, world!",
                    stream_mode=None,
                )
            ]
            for res, expect in zip(responses, espect_responses, strict=True):
                assert res.content == expect.content  # type: ignore[attr-defined]
        conversation = mcp_host.conversation()
        async with conversation:
            responses = [
                response["agent"]["messages"][0]
                async for response in conversation.query(
                    HumanMessage(content="Hello, world!"),
                    stream_mode=None,
                )
            ]
            for res, expect in zip(responses, espect_responses, strict=True):
                assert res.content == expect.content  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_query_two_messages() -> None:
    """Test that the query method can handle two or more messages."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
        ),
        mcp_servers={},
    )
    async with DiveMcpHost(config) as mcp_host, mcp_host.conversation() as conversation:
        responses = [
            response
            async for response in conversation.query(
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
            modelProvider="dive",
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
        conversation = mcp_host.conversation()
        async with conversation:
            async for _ in conversation.query(
                HumanMessage(content="Hello, world! 許個願望吧"),
                stream_mode=["messages"],
            ):
                pass

            thread_id = conversation.thread_id
            messages = await mcp_host.get_messages(thread_id, user_id)
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

            messages = await mcp_host.get_messages(thread_id, user_id)
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
            modelProvider="dive",
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
        mcp_host.conversation(
            system_prompt=mock_system_prompt, volatile=True
        ) as conversation,
    ):
        assert mcp_host.model is not None
        model = cast("FakeMessageToolModel", mcp_host.model)
        async for _ in conversation.query(
            msgs,
        ):
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

        async for _ in conversation.query(
            msgs,
        ):
            ...
        assert len(model.query_history) == 2
        assert model.query_history[0].content == "You are a helpful assistant."
        assert model.query_history[1].content == "Line 2!"
        assert mock_system_prompt.call_count == 1


@pytest.mark.asyncio
async def test_abort_conversation() -> None:
    """Test that the conversation can be aborted during a long-running query."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
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

        conversation = mcp_host.conversation()
        async with conversation:
            # Start the query in a separate task
            async def _query() -> list[dict[str, Any]]:
                return [
                    i
                    async for i in conversation.query(
                        "This is a long running query", stream_mode=["messages"]
                    )
                ]

            query_task = asyncio.create_task(_query())

            # Wait a bit and then abort
            await asyncio.sleep(0.5)
            conversation.abort()

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
                async for i in conversation.query(
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

            # abort a non-running conversation
            conversation.abort()
            responses = [
                i
                async for i in conversation.query(
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
            modelProvider="dive",
        ),
        mcp_servers={},
        checkpointer=CheckpointerConfig(uri=AnyUrl(sqlite_uri)),
    )

    async with DiveMcpHost(config) as mcp_host:
        conversation = mcp_host.conversation()
        model = cast("FakeMessageToolModel", mcp_host.model)
        async with conversation:
            resps = cast(
                list[tuple[str, dict[str, list[BaseMessage]]]],
                [
                    i
                    async for i in conversation.query(
                        HumanMessage(content="Hello, world!"),
                        stream_mode=["values"],
                    )
                ],
            )
            assert len(resps) == 2
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
                    async for i in conversation.query(
                        resend,  # type: ignore
                        stream_mode=["values"],
                        is_resend=True,
                    )
                ],
            )
            assert len(resps) == 2
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
            modelProvider="openai",
            apiKey="fake",
        ),
        mcp_servers=echo_tool_stdio_config,
    )

    # New configuration with different model settings
    new_config = HostConfig(
        llm=LLMConfig(
            model="fake",
            modelProvider="dive",
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
        # Verify initial state
        assert len(host.tools) == 2  # echo and ignore tools
        assert host.config.llm.temperature == 0

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

        # Test conversation still works after reload
        async with host.conversation() as conversation:
            responses = []
            async for response in conversation.query("Hello"):
                responses.append(response)

            assert len(responses) > 0

        # Test reload with same config
        reloader_called = False
        await host.reload(new_config, mock_reloader)
        assert reloader_called
        assert len(host.tools) == 3
