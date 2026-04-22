"""Tests for capability negotiation and MCP prompts support.

Covers Dive issues #348 (don't blindly call tools/list) and #349 (prompts
support). The `capability_server` test fixture lets us spin up servers that
advertise any combination of tools/prompts capabilities.
"""

import asyncio

import pytest

from dive_mcp_host.host.conf import LogConfig, ServerConfig
from dive_mcp_host.host.tools import ToolManager
from dive_mcp_host.host.tools.elicitation_manager import ElicitationManager
from dive_mcp_host.host.tools.mcp_server import McpServer, _safe_list
from dive_mcp_host.host.tools.model_types import ClientState
from dive_mcp_host.host.tools.oauth import OAuthManager


def _capability_config(name: str, features: str, broken: str = "") -> ServerConfig:
    args = [
        "-m",
        "tests.mcp_servers.capability_server",
        f"--features={features}",
    ]
    if broken:
        args.append(f"--break={broken}")
    return ServerConfig(
        name=name,
        command="python3",
        args=args,
        transport="stdio",
    )


@pytest.mark.asyncio
async def test_tools_only_server(log_config: LogConfig) -> None:
    """A tools-only server should initialize and expose its tool, no prompts."""
    cfg = {"srv": _capability_config("srv", "tools")}
    async with ToolManager(cfg, log_config) as tm:
        await tm.initialized_event.wait()
        info = tm.mcp_server_info["srv"]
        assert info.client_status == ClientState.RUNNING
        assert [t.name for t in info.tools] == ["ping"]
        assert info.prompts == []
        assert info.initialize_result is not None
        assert info.initialize_result.capabilities.tools is not None
        assert info.initialize_result.capabilities.prompts is None


@pytest.mark.asyncio
async def test_prompts_only_server(log_config: LogConfig) -> None:
    """A prompts-only server must initialize without crashing on tools/list."""
    cfg = {"srv": _capability_config("srv", "prompts")}
    async with ToolManager(cfg, log_config) as tm:
        await tm.initialized_event.wait()
        info = tm.mcp_server_info["srv"]
        assert info.client_status == ClientState.RUNNING, info.error_str
        assert info.tools == []
        assert [p.name for p in info.prompts] == ["greet"]


@pytest.mark.asyncio
async def test_tools_and_prompts_server(log_config: LogConfig) -> None:
    """Both capabilities should be discovered and exposed."""
    cfg = {"srv": _capability_config("srv", "tools,prompts")}
    async with ToolManager(cfg, log_config) as tm:
        await tm.initialized_event.wait()
        info = tm.mcp_server_info["srv"]
        assert info.client_status == ClientState.RUNNING
        assert [t.name for t in info.tools] == ["ping"]
        assert [p.name for p in info.prompts] == ["greet"]


@pytest.mark.asyncio
async def test_no_capabilities_server(log_config: LogConfig) -> None:
    """A server that advertises nothing must still initialize cleanly."""
    cfg = {"srv": _capability_config("srv", "none")}
    async with ToolManager(cfg, log_config) as tm:
        await tm.initialized_event.wait()
        info = tm.mcp_server_info["srv"]
        assert info.client_status == ClientState.RUNNING, info.error_str
        assert info.tools == []
        assert info.prompts == []
        assert info.initialize_result is not None
        assert info.initialize_result.capabilities.tools is None
        assert info.initialize_result.capabilities.prompts is None


@pytest.mark.asyncio
async def test_method_not_found_is_swallowed(log_config: LogConfig) -> None:
    """A server that advertises tools but raises -32601 should still init."""
    cfg = {
        "srv": _capability_config(
            "srv", "tools,prompts", broken="tools,prompts"
        )
    }
    async with ToolManager(cfg, log_config) as tm:
        await tm.initialized_event.wait()
        info = tm.mcp_server_info["srv"]
        assert info.client_status == ClientState.RUNNING, info.error_str
        assert info.tools == []
        assert info.prompts == []


@pytest.mark.asyncio
async def test_get_prompt_round_trip(log_config: LogConfig) -> None:
    """Prompt list and prompt fetch should both work."""
    cfg = _capability_config("srv", "prompts")
    async with McpServer(
        name="srv",
        config=cfg,
        log_buffer_length=log_config.buffer_length,
        auth_manager=OAuthManager(),
        elicitation_manager=ElicitationManager(),
    ) as server:
        await server.wait([ClientState.RUNNING])
        prompts = await server.list_prompts()
        assert [p.name for p in prompts] == ["greet"]
        result = await server.get_prompt("greet", arguments={"name": "Dive"})
        assert len(result.messages) == 1
        msg = result.messages[0]
        assert msg.role == "user"
        assert msg.content.text == "Hello, Dive!"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_safe_list_swallows_method_not_found() -> None:
    """The defensive helper should map -32601 to None and re-raise others."""
    from mcp import McpError
    from mcp.shared.exceptions import ErrorData
    from mcp.types import METHOD_NOT_FOUND

    async def raises_method_not_found() -> None:
        raise McpError(ErrorData(code=METHOD_NOT_FOUND, message="nope"))

    async def raises_other() -> None:
        raise McpError(ErrorData(code=-1, message="boom"))

    assert await _safe_list(raises_method_not_found, "srv", "x/list") is None
    with pytest.raises(McpError):
        await _safe_list(raises_other, "srv", "x/list")


@pytest.mark.asyncio
async def test_init_does_not_call_optional_lists(log_config: LogConfig) -> None:
    """tools/list and prompts/list must NOT be invoked if not advertised."""
    from unittest.mock import patch

    cfg = {"srv": _capability_config("srv", "none")}
    with (
        patch(
            "mcp.client.session.ClientSession.list_tools",
            side_effect=AssertionError("list_tools should not be called"),
        ),
        patch(
            "mcp.client.session.ClientSession.list_prompts",
            side_effect=AssertionError("list_prompts should not be called"),
        ),
    ):
        async with ToolManager(cfg, log_config) as tm:
            await tm.initialized_event.wait()
            info = tm.mcp_server_info["srv"]
            assert info.client_status == ClientState.RUNNING


@pytest.mark.asyncio
async def test_prompts_list_changed_refreshes_cache(log_config: LogConfig) -> None:
    """The prompt cache should refresh after a notifications/prompts/list_changed."""
    from mcp.types import Prompt, PromptListChangedNotification, ServerNotification

    cfg = _capability_config("srv", "prompts")
    async with McpServer(
        name="srv",
        config=cfg,
        log_buffer_length=log_config.buffer_length,
        auth_manager=OAuthManager(),
        elicitation_manager=ElicitationManager(),
    ) as server:
        await server.wait([ClientState.RUNNING])
        assert [p.name for p in server.prompts] == ["greet"]

        # Force-prime the cache with stale data and dispatch the notification
        # directly to the server's message handler. _refresh_prompts will then
        # re-query the live server.
        async with server._cond:
            server._prompts = [
                Prompt(name="stale", description="should be replaced")
            ]

        await server._message_handler(
            ServerNotification(root=PromptListChangedNotification())
        )
        # Background refresh; give it a moment.
        for _ in range(50):
            await asyncio.sleep(0.05)
            if [p.name for p in server.prompts] == ["greet"]:
                break
        assert [p.name for p in server.prompts] == ["greet"]


@pytest.mark.asyncio
async def test_list_prompts_cache_does_not_refetch_when_empty(
    log_config: LogConfig,
) -> None:
    """Servers with zero prompts must still hit the cache, not re-query."""
    cfg = _capability_config("srv", "tools")  # advertises tools, no prompts
    async with McpServer(
        name="srv",
        config=cfg,
        log_buffer_length=log_config.buffer_length,
        auth_manager=OAuthManager(),
        elicitation_manager=ElicitationManager(),
    ) as server:
        await server.wait([ClientState.RUNNING])
        from unittest.mock import patch

        # If the cache check is buggy (truthiness on empty list), this would
        # fall through to opening a fresh session.
        with patch.object(
            server, "session", side_effect=AssertionError("session should not be opened")
        ):
            assert await server.list_prompts() == []
            assert await server.list_prompts(use_cache=True) == []


@pytest.mark.asyncio
async def test_get_prompt_without_capability_raises_value_error(
    log_config: LogConfig,
) -> None:
    """get_prompt should ValueError when the server doesn't advertise prompts."""
    cfg = _capability_config("srv", "tools")
    async with McpServer(
        name="srv",
        config=cfg,
        log_buffer_length=log_config.buffer_length,
        auth_manager=OAuthManager(),
        elicitation_manager=ElicitationManager(),
    ) as server:
        await server.wait([ClientState.RUNNING])
        with pytest.raises(ValueError, match="does not support prompts"):
            await server.get_prompt("anything")


@pytest.mark.asyncio
async def test_refresh_prompts_dedupes_concurrent_notifications(
    log_config: LogConfig,
) -> None:
    """Repeated notifications must not stack up multiple refresh tasks."""
    from mcp.types import PromptListChangedNotification, ServerNotification

    cfg = _capability_config("srv", "prompts")
    async with McpServer(
        name="srv",
        config=cfg,
        log_buffer_length=log_config.buffer_length,
        auth_manager=OAuthManager(),
        elicitation_manager=ElicitationManager(),
    ) as server:
        await server.wait([ClientState.RUNNING])

        notification = ServerNotification(root=PromptListChangedNotification())
        # Fire several notifications in a row.
        for _ in range(5):
            await server._message_handler(notification)

        first_task = server._refresh_prompts_task
        assert first_task is not None
        # All notifications should have collapsed into a single in-flight task.
        await first_task
        # After completion a fresh notification can schedule a new task.
        await server._message_handler(notification)
        second_task = server._refresh_prompts_task
        assert second_task is not None
        assert second_task is not first_task
        await second_task
