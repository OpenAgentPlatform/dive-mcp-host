from logging import getLogger

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from dive_mcp_host.host.tools.model_types import ClientState
from dive_mcp_host.httpd.conf.mcp_servers import Config
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.models import (
    McpTool,
    ResultResponse,
    SimpleToolInfo,
    ToolsCache,
)
from dive_mcp_host.httpd.routers.utils import (
    EventStreamContextManager,
    LogStreamHandler,
)
from dive_mcp_host.httpd.server import DiveHostAPI
from dive_mcp_host.httpd.store.cache import CacheKeys

logger = getLogger(__name__)
tools = APIRouter(tags=["tools"])


class ToolsResult(ResultResponse):
    """Response model for listing available MCP tools."""

    tools: list[McpTool]


@tools.get("/initialized")
async def initialized(
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Check if initial setup is complete.

    Only useful on initial startup, not when reloading.
    """
    await app.dive_host["default"].tools_initialized_event.wait()
    return ResultResponse(success=True, message=None)


@tools.get("/")
async def list_tools(  # noqa: PLR0912, C901
    app: DiveHostAPI = Depends(get_app),
) -> ToolsResult:
    """Lists all available MCP tools.

    Returns:
        ToolsResult: A list of available tools.
    """
    result: dict[str, McpTool] = {}

    # get full list of servers from config
    if (config := await app.mcp_server_config_manager.get_current_config()) is not None:
        all_server_configs = config
    else:
        all_server_configs = Config()

    # get tools from dive host
    for server_name, server_info in app.dive_host["default"].mcp_server_info.items():
        result[server_name] = McpTool(
            name=server_name,
            tools=[
                SimpleToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    enabled=tool.enable,
                )
                for tool in server_info.tools
            ],
            url=server_info.url,
            description="",
            enabled=True,
            icon="",
            error=server_info.error_str,
        )
    logger.debug("active mcp servers: %s", result.keys())
    # find missing servers
    missing_servers = set(all_server_configs.mcp_servers.keys()) - set(result.keys())
    logger.debug("disabled mcp servers: %s", missing_servers)

    # get missing from local cache
    if missing_servers:
        raw_cached_tools = app.local_file_cache.get(CacheKeys.LIST_TOOLS)
        cached_tools = ToolsCache(root={})
        if raw_cached_tools is not None:
            try:
                cached_tools = ToolsCache.model_validate_json(raw_cached_tools)
            except ValidationError as e:
                logger.warning(
                    "Failed to validate cached tools: %s %s", e, raw_cached_tools
                )
        for server_name in missing_servers:
            if server_info := cached_tools.root.get(server_name, None):
                server_info.enabled = False

                # Sync tool 'enabled' with 'exclude_tools'
                if mcp_config := all_server_configs.mcp_servers.get(server_name):
                    for tool in server_info.tools:
                        if tool.name in mcp_config.exclude_tools:
                            tool.enabled = False
                        else:
                            tool.enabled = True

                result[server_name] = server_info
            else:
                logger.warning("Server %s not found in cached tools", server_name)
                result[server_name] = McpTool(
                    name=server_name,
                    tools=[],
                    description="",
                    enabled=False,
                    icon="",
                    error=None,
                )

    # update local cache
    app.local_file_cache.set(
        CacheKeys.LIST_TOOLS,
        ToolsCache(root=result).model_dump_json(),
    )
    return ToolsResult(success=True, message=None, tools=list(result.values()))


@tools.get("/logs/stream")
async def stream_server_logs(
    server_name: str,
    stream_until: ClientState | None = None,
    stop_on_notfound: bool = True,
    max_retries: int = 10,
    app: DiveHostAPI = Depends(get_app),
) -> StreamingResponse:
    """Stream logs from a specific MCP server.

    Args:
        server_name (str): The name of the MCP server to stream logs from.
        stream_until (ClientState | None): stream until client state is reached.
        stop_on_notfound (bool): If True, stop streaming if the server is not found.
        max_retries (int): The maximum number of retries to stream logs.
        app (DiveHostAPI): The DiveHostAPI instance.

    Returns:
        StreamingResponse: A streaming response of the server logs.
        Keep streaming until client disconnects.
    """
    log_manager = app.dive_host["default"].log_manager
    stream = EventStreamContextManager()
    response = stream.get_response()

    async def process() -> None:
        async with stream:
            processor = LogStreamHandler(
                stream=stream,
                log_manager=log_manager,
                stream_until=stream_until,
                stop_on_notfound=stop_on_notfound,
                max_retries=max_retries,
            )
            await processor.stream_logs(server_name)

    stream.add_task(process)
    return response
