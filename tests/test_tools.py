import asyncio
import json
import logging
import random
import secrets
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch
from uuid import UUID

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from mcp.types import Tool

from dive_mcp_host.host.conf import HostConfig, LogConfig, ProxyUrl
from dive_mcp_host.host.conf.llm import LLMConfig
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.host.tools import McpServer, McpServerInfo, ServerConfig, ToolManager
from dive_mcp_host.host.tools.mcp_server import McpTool
from dive_mcp_host.host.tools.model_types import ClientState

if TYPE_CHECKING:
    from dive_mcp_host.models.fake import FakeMessageToolModel


@pytest.fixture
def no_such_file_mcp_server() -> dict[str, ServerConfig]:
    """MCP server that does not exist."""
    return {
        "no_such_file": ServerConfig(
            name="no_such_file",
            command="no_such_file",
            transport="stdio",
        ),
        "sse": ServerConfig(
            name="sse_server",
            url="http://localhost:2/sse",
            transport="sse",
        ),
    }


@pytest.mark.asyncio
async def test_tool_manager_sse(
    echo_tool_sse_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
    log_config: LogConfig,
) -> None:
    """Test the tool manager."""
    async with (
        echo_tool_sse_server as (port, configs),
        ToolManager(configs, log_config) as tool_manager,
    ):
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"message": "Hello, world!"},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []


@pytest.mark.asyncio
async def test_tool_manager_stdio(
    echo_tool_stdio_config: dict[str, ServerConfig],
    log_config: LogConfig,
) -> None:
    """Test the tool manager."""
    async with ToolManager(echo_tool_stdio_config, log_config) as tool_manager:
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"message": "Hello, world!"},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []


@pytest.mark.asyncio
async def test_tool_manager_streamable(
    echo_tool_streamable_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
    log_config: LogConfig,
) -> None:
    """Test the tool manager."""
    async with (
        echo_tool_streamable_server as (port, configs),
        ToolManager(configs, log_config) as tool_manager,
    ):
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"message": "Hello, world!"},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []


@pytest.mark.asyncio
async def test_tool_manager_reload(
    echo_tool_stdio_config: dict[str, ServerConfig],
    log_config: LogConfig,
) -> None:
    """Test the tool manager's reload."""
    async with ToolManager(echo_tool_stdio_config, log_config) as tool_manager:
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]

        # test reload with same config
        await tool_manager.reload(echo_tool_stdio_config)
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]

        # test reload with modified config
        new_config = echo_tool_stdio_config.copy()
        new_config["fetch"] = ServerConfig(
            name="fetch",
            command="uvx",
            args=["mcp-server-fetch"],
            transport="stdio",
        )
        await tool_manager.reload(new_config)
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "fetch", "ignore"]

        # test remove tool
        await tool_manager.reload(echo_tool_stdio_config)
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]

        # verify tools still work after reload
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"message": "Hello, world!"},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"

        # remove all tools
        await tool_manager.reload({})
        tools = tool_manager.langchain_tools()
        assert len(tools) == 0


@pytest.mark.asyncio
async def test_stdio_parallel(
    echo_tool_stdio_config: dict[str, ServerConfig], log_config: LogConfig
) -> None:
    """Test that stdio tools can execute in parallel.

    This test is to ensure that the tool manager can handle multiple requests
    simultaneously and respond correctly.
    """
    async with ToolManager(echo_tool_stdio_config, log_config) as tool_manager:
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        echo_tool = None
        ignore_tool = None
        for tool in tools:
            if tool.name == "echo":
                echo_tool = tool
            elif tool.name == "ignore":
                ignore_tool = tool
        assert echo_tool is not None
        assert ignore_tool is not None

        random_message = secrets.token_hex(2048)

        async def test_echo():
            return await echo_tool.ainvoke(
                ToolCall(
                    name=echo_tool.name,
                    id=str(random.randint(1, 1000000)),  # noqa: S311
                    args={"message": random_message},
                    type="tool_call",
                ),
            )

        async def test_ignore():
            return await ignore_tool.ainvoke(
                ToolCall(
                    name=ignore_tool.name,
                    id=str(random.randint(1, 1000000)),  # noqa: S311
                    args={"message": random_message},
                    type="tool_call",
                ),
            )

        n_tasks = 30
        async with asyncio.TaskGroup() as tg:
            echo_tasks = [tg.create_task(test_echo()) for _ in range(n_tasks)]
            ignore_tasks = [tg.create_task(test_ignore()) for _ in range(n_tasks)]
        echo_results = await asyncio.gather(*echo_tasks)
        ignore_results = await asyncio.gather(*ignore_tasks)
        assert len(echo_results) == n_tasks
        assert len(ignore_results) == n_tasks
        for result in echo_results:
            assert json.loads(str(result.content))[0]["text"] == random_message
        for result in ignore_results:
            assert json.loads(str(result.content)) == []


@pytest.mark.asyncio
async def test_tool_manager_massive_tools(
    echo_tool_stdio_config: dict[str, ServerConfig],
    log_config: LogConfig,
) -> None:
    """Test starting the tool manager with a large number of tools."""
    echo_config = echo_tool_stdio_config["echo"]
    more_tools = 10
    for i in range(more_tools):
        echo_tool_stdio_config[f"echo_{i}"] = echo_config.model_copy(
            update={"name": f"echo_{i}"},
        )
    async with ToolManager(echo_tool_stdio_config, log_config) as tool_manager:
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert len(tools) == 2 * (more_tools + 1)


@pytest.mark.asyncio
async def test_remote_http_mcp_tool_exception_handling(
    echo_tool_sse_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
    log_config: LogConfig,
) -> None:
    """Test the exception handling of the MCP tool.

    This test verifies that:
    1. When a tool call fails, the exception is properly propagated to the caller
    2. Subsequent tool calls succeed after the connection is restored
    """
    async with (
        echo_tool_sse_server as (_, configs),
        McpServer(
            name="echo",
            config=configs["echo"],
            log_buffer_length=log_config.buffer_length,
        ) as server,
    ):
        server.RESTART_INTERVAL = 0.1
        tools = server.mcp_tools
        await server.wait([ClientState.RUNNING])

        # First successful tool call creates a session
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )
        session = server._session_store._map["default"].session

        # session should be reused
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )
        assert server._session_store._map["default"].session == session

        # Error removes the session
        with patch("dive_mcp_host.host.tools.hack.ClientSession.call_tool") as mocked:
            mocked.side_effect = RuntimeError("test")
            with pytest.raises(RuntimeError, match="test"):
                await tools[0].ainvoke(
                    ToolCall(
                        name=tools[0].name,
                        id="123",
                        args={"xxxx": "Hello, world!"},
                        type="tool_call",
                    ),
                )
            assert mocked.call_count == 1
        assert server._client_status in [
            ClientState.RUNNING,
            ClientState.RESTARTING,
        ]
        assert not server._session_store._map.get("default")

        # New session is created
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )
        assert server._session_store._map["default"].session
        session = server._session_store._map["default"].session

        # The session should be reused
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )
        assert server._session_store._map["default"].session == session


@pytest.mark.asyncio
async def test_local_http_mcp_tool_exception_handling(
    echo_tool_local_sse_config: dict[str, ServerConfig],
    log_config: LogConfig,
):
    """Test the exception handling of the MCP tool.

    This test verifies that:
    1. When a tool call fails, the exception is properly propagated to the caller
    2. Subsequent tool calls succeed after the connection is restored
    """
    async with McpServer(
        name="echo",
        config=echo_tool_local_sse_config["echo"],
        log_buffer_length=log_config.buffer_length,
    ) as server:
        server.RESTART_INTERVAL = 0.1
        tools = server.mcp_tools
        await server.wait([ClientState.RUNNING])

        # First successful tool call creates a session
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )
        session = server._session_store["default"]

        # session should be reused
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )
        assert server._session_store["default"] == session

        # Error removes the session
        with patch("dive_mcp_host.host.tools.hack.ClientSession.call_tool") as mocked:
            mocked.side_effect = RuntimeError("test")
            with pytest.raises(RuntimeError, match="test"):
                await tools[0].ainvoke(
                    ToolCall(
                        name=tools[0].name,
                        id="123",
                        args={"xxxx": "Hello, world!"},
                        type="tool_call",
                    ),
                )
            assert mocked.call_count == 1
        assert server._client_status in [
            ClientState.RUNNING,
            ClientState.RESTARTING,
        ]
        assert not server._session_store._map.get("default")

        # New session is created
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )
        assert server._session_store["default"]
        session = server._session_store["default"]

        # The session should be reused
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )
        assert server._session_store["default"] == session


@pytest.mark.asyncio
async def test_stdio_mcp_tool_exception_handling(
    echo_tool_stdio_config: dict[str, ServerConfig],
    log_config: LogConfig,
):
    """Test the exception handling of the MCP tool.

    This test verifies that:
    1. When a tool call fails, the exception is properly propagated to the caller
    2. Subsequent tool calls succeed after the connection is restored
    """
    async with McpServer(
        name="echo",
        config=echo_tool_stdio_config["echo"],
        log_buffer_length=log_config.buffer_length,
    ) as server:
        server.RESTART_INTERVAL = 0.1
        tools = server.mcp_tools
        session = server._stdio_client_session
        with patch("dive_mcp_host.host.tools.hack.ClientSession.call_tool") as mocked:
            mocked.side_effect = RuntimeError("test")
            with pytest.raises(RuntimeError, match="test"):
                await tools[0].ainvoke(
                    ToolCall(
                        name=tools[0].name,
                        id="123",
                        args={"xxxx": "Hello, world!"},
                        type="tool_call",
                    ),
                )
            assert mocked.call_count == 1
        assert server._client_status in [
            ClientState.RUNNING,
            ClientState.RESTARTING,
        ]
        await server.wait([ClientState.RUNNING])
        # The session should be recreated
        assert server._stdio_client_session != session
        session = server._stdio_client_session
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )

        await server.wait([ClientState.RUNNING])
        # The session should be reused
        assert server._stdio_client_session == session
        await tools[0].ainvoke(
            ToolCall(
                name=tools[0].name,
                id="123",
                args={"message": "Hello, world!"},
                type="tool_call",
            ),
        )


@pytest.mark.asyncio
async def test_tool_manager_local_sse(
    echo_tool_local_sse_config: dict[str, ServerConfig],
    log_config: LogConfig,
) -> None:
    """Test the tool manager."""
    async with ToolManager(echo_tool_local_sse_config, log_config) as tool_manager:
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"message": "Hello, world!"},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []
            logging.info("Tool %s tested", tool.name)


@pytest.mark.asyncio
async def test_host_with_tools(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
    )
    async with DiveMcpHost(config) as mcp_host:
        await mcp_host._tool_manager.initialized_event.wait()
        fake_responses = [
            AIMessage(
                content="Call echo tool",
                tool_calls=[
                    ToolCall(
                        name="echo",
                        args={"message": "Hello, world!"},
                        id="123",
                        type="tool_call",
                    ),
                ],
            ),
            AIMessage(
                content="General message",
            ),
        ]
        cast("FakeMessageToolModel", mcp_host._model).responses = fake_responses
        async with mcp_host.chat() as chat:
            responses = [
                response
                async for response in chat.query(
                    HumanMessage(content="Hello, world!"),
                    stream_mode=["messages"],
                )
            ]
            assert len(responses) == len(fake_responses) + 1  # plus one tool message
            # need more understanding of the response structure
            tool_message = responses[-2][1][0]  # type: ignore
            assert isinstance(tool_message, ToolMessage)
            assert tool_message.name == "echo"
            assert json.loads(str(tool_message.content))[0]["text"] == "Hello, world!"


@pytest.mark.asyncio
async def test_mcp_server_info(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers=echo_tool_stdio_config,
    )
    import dive_mcp_host.host.tools.echo as echo_tool

    async with DiveMcpHost(config) as mcp_host:
        await mcp_host._tool_manager.initialized_event.wait()
        assert list(mcp_host.mcp_server_info.keys()) == ["echo"]
        assert isinstance(mcp_host.mcp_server_info["echo"], McpServerInfo)
        assert mcp_host.mcp_server_info["echo"].initialize_result is not None
        assert mcp_host.mcp_server_info["echo"].initialize_result.capabilities
        assert (
            mcp_host.mcp_server_info["echo"].initialize_result.instructions
            == echo_tool.Instructions
        )


@pytest.mark.asyncio
async def test_mcp_server_info_no_such_file(
    no_such_file_mcp_server: dict[str, ServerConfig],
) -> None:
    """Test the host context initialization."""
    config = HostConfig(
        llm=LLMConfig(
            model="fake",
            model_provider="dive",
        ),
        mcp_servers=no_such_file_mcp_server,
    )
    async with DiveMcpHost(config) as mcp_host:
        await mcp_host._tool_manager.initialized_event.wait()
        assert list(mcp_host.mcp_server_info.keys()) == [
            "no_such_file",
            "sse",
        ]
        assert mcp_host.mcp_server_info["no_such_file"] is not None
        assert mcp_host.mcp_server_info["no_such_file"].initialize_result is None
        assert mcp_host.mcp_server_info["no_such_file"].error is not None
        assert (
            mcp_host.mcp_server_info["no_such_file"].client_status == ClientState.FAILED
        )
        assert mcp_host.mcp_server_info["sse"] is not None
        assert mcp_host.mcp_server_info["sse"].initialize_result is None
        assert mcp_host.mcp_server_info["sse"].error is not None
        assert mcp_host.mcp_server_info["sse"].client_status == ClientState.FAILED


@pytest.mark.asyncio
async def test_mcp_server_info_sse_connection_refused(
    echo_tool_sse_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
    log_config: LogConfig,
) -> None:
    """Test the tool manager's SSE connection refused."""
    async with echo_tool_sse_server as (port, configs):
        configs["echo"].url = f"http://localhost:{port + 1}/sse"
        async with (
            ToolManager(configs, log_config) as tool_manager,
        ):
            await tool_manager.initialized_event.wait()
            tools = tool_manager.langchain_tools()
            assert len(tools) == 0
            assert tool_manager.mcp_server_info["echo"].error is not None
            assert (
                tool_manager.mcp_server_info["echo"].client_status == ClientState.FAILED
            )


@pytest.mark.asyncio
async def test_tool_kwargs(
    echo_tool_stdio_config: dict[str, ServerConfig],
    log_config: LogConfig,
) -> None:
    """Some LLM set the tool call argument in kwargs."""
    async with ToolManager(echo_tool_stdio_config, log_config) as tool_manager:
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"kwargs": {"message": "Hello, world!"}},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []

        for tool in tools:
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"kwargs": """{"message": "Hello, world!"}"""},
                    type="tool_call",
                ),
            )
            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []


@pytest.mark.asyncio
async def test_tool_manager_uvx_failed(log_config: LogConfig) -> None:
    """Test the tool manager."""
    config = {
        "uvx": ServerConfig(
            name="uvx",
            command="uvx",
            args=["no-such-command"],
            transport="stdio",
        ),
    }
    async with asyncio.timeout(15), ToolManager(config, log_config) as tool_manager:
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert len(tools) == 0


def test_tool_missing_properties(log_config: LogConfig) -> None:
    """Test handling of MCP server tool schemas that lack properties.

    Some MCP servers may return tool schemas without a properties field, but certain
    model providers require properties to be present in the tool call schema.
    """
    tool = Tool(
        name="dummy",
        description="A dummy tool that returns a fixed string.",
        inputSchema={"type": "Object"},
    )
    mcp_server = McpServer(
        name="dummy",
        config=ServerConfig(
            name="dummy",
            command="dummy",
            transport="stdio",
        ),
    )
    mcp_tool = McpTool.from_tool(tool, mcp_server)

    assert mcp_tool.args_schema is not None
    if isinstance(mcp_tool.args_schema, dict):
        assert "properties" in mcp_tool.args_schema
    else:
        assert "properties" in mcp_tool.args_schema.model_json_schema()


@pytest.mark.asyncio
async def test_tool_progress(
    echo_tool_stdio_config: dict[str, ServerConfig],
    log_config: LogConfig,
) -> None:
    """Test the tool progress report."""
    import logging

    class CustomCallbackManager(AsyncCallbackHandler):
        async def on_custom_event(
            self,
            name: str,
            data: dict[str, Any],
            *,
            run_id: UUID,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            logging.error(
                "Custom event: %s, run_id: %s, tags: %s, metadata: %s,"
                " kwargs: %s, data: %s",
                name,
                run_id,
                tags,
                metadata,
                kwargs,
                data,
            )

    async with ToolManager(echo_tool_stdio_config, log_config) as tool_manager:
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert sorted([i.name for i in tools]) == ["echo", "ignore"]
        for tool in tools:
            if tool.name != "echo":
                continue
            result = await tool.ainvoke(
                ToolCall(
                    name=tool.name,
                    id="123",
                    args={"message": "Hello, world!", "delay_ms": 1000},
                    type="tool_call",
                ),
                config={
                    "callbacks": [CustomCallbackManager()],
                },
            )

            assert isinstance(result, ToolMessage)
            if tool.name == "echo":
                assert json.loads(str(result.content))[0]["text"] == "Hello, world!"
            else:
                assert json.loads(str(result.content)) == []


@pytest.mark.asyncio
async def test_tool_proxy(
    subtests,
    pproxy_server: str,
    echo_tool_sse_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
    echo_tool_streamable_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
    log_config: LogConfig,
) -> None:
    """Test proxy settings."""
    with subtests.test("scheme rewrite"):
        for prot in ["http", "socks5", "socks", "socks4"]:
            cfg = json.dumps(
                {
                    "name": "echo",
                    "url": "http://localhost:8888/mcp",
                    "transport": "streamable",
                    "proxy": f"{prot}://{pproxy_server}",
                }
            )
            m = ServerConfig.model_validate_json(cfg)
            assert m.proxy
            match prot:
                case "http":
                    assert m.proxy.scheme == "http"
                case _ if prot.startswith("socks"):
                    assert m.proxy.scheme == "socks5"

    for test_cfg in [echo_tool_sse_server, echo_tool_streamable_server]:
        async with test_cfg as (_, config):
            cfg = config.copy()
            for prot in ["http", "socks5"]:
                cfg["echo"].proxy = ProxyUrl(f"{prot}://{pproxy_server}")
                with subtests.test(prot=prot, url=cfg["echo"].url):
                    async with ToolManager(cfg, log_config) as tool_manager:
                        await tool_manager.initialized_event.wait()
                        tools = tool_manager.langchain_tools()
                        assert sorted([i.name for i in tools]) == ["echo", "ignore"]


@pytest.mark.asyncio
async def test_tool_manager_exclude_tools(
    echo_tool_sse_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
):
    """Make sure excluded tools are not passed to the llm."""
    async with (
        echo_tool_sse_server as (_, configs),
        ToolManager(configs) as tool_manager,
    ):
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert len(tools) == 2
        assert tools[0].name == "echo"
        assert tools[1].name == "ignore"

        # Disable 'igonre' tool
        new_config = deepcopy(configs)
        new_config["echo"].exclude_tools = ["ignore"]
        await tool_manager.reload(new_config)
        await tool_manager.initialized_event.wait()
        tools = tool_manager.langchain_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"


@pytest.mark.asyncio
async def test_custum_initalize_timeout(
    echo_tool_local_sse_config: dict[str, ServerConfig],
    echo_tool_stdio_config: dict[str, ServerConfig],
    echo_tool_streamable_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
    echo_tool_sse_server: AbstractAsyncContextManager[
        tuple[int, dict[str, ServerConfig]]
    ],
    log_config: LogConfig,
):
    """Test if our customized timeout actually apply."""
    echo_tool_local_sse_config["echo"].initial_timeout = 0
    async with McpServer(
        name="echo",
        config=echo_tool_local_sse_config["echo"],
        log_buffer_length=log_config.buffer_length,
    ) as server:
        assert server.server_info.client_status == ClientState.FAILED

    echo_tool_stdio_config["echo"].initial_timeout = 0
    async with McpServer(
        name="echo",
        config=echo_tool_stdio_config["echo"],
        log_buffer_length=log_config.buffer_length,
    ) as server:
        assert server.server_info.client_status == ClientState.FAILED

    async with echo_tool_streamable_server as (_, config):
        config["echo"].initial_timeout = 0
        async with McpServer(
            name="echo",
            config=config["echo"],
            log_buffer_length=log_config.buffer_length,
        ) as server:
            assert server.server_info.client_status == ClientState.FAILED

    async with echo_tool_sse_server as (_, config):
        config["echo"].initial_timeout = 0
        async with McpServer(
            name="echo",
            config=config["echo"],
            log_buffer_length=log_config.buffer_length,
        ) as server:
            assert server.server_info.client_status == ClientState.FAILED
