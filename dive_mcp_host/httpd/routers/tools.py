from asyncio import create_task, wait_for
from logging import getLogger
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from dive_mcp_host.host.tools.mcp_server import McpServer
from dive_mcp_host.host.tools.model_types import ClientState
from dive_mcp_host.host.tools.oauth import OAuthManager
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


class OAuthList(ResultResponse):
    """Response model for listing OAuth."""

    servers: list[str]


class OAuthRequest(BaseModel):
    """Request model for OAuth."""

    server_name: str


class OAuthResult(ResultResponse):
    """Response model for OAuth."""

    stage: (
        Literal[
            "auth_success",
            "auth_failed",
            "no_auth_required",
            "wait_code",
            "code_set",
            "canceled",
        ]
        | None
    ) = None
    server_name: str | None = None
    auth_url: str | None = None
    error: str | None = None


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
                    icons=tool.icons,
                )
                for tool in server_info.tools
            ],
            url=server_info.url,
            description="",
            enabled=True,
            error=server_info.error_str,
            status=server_info.client_status.value,
            icons=server_info.initialize_result.serverInfo.icons,
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
                    error=None,
                    status="",
                )

    # update local cache
    app.local_file_cache.set(
        CacheKeys.LIST_TOOLS,
        ToolsCache(root=result).model_dump_json(),
    )
    return ToolsResult(success=True, message=None, tools=list(result.values()))


class LogsStreamBody(BaseModel):
    """Body for logs stream API."""

    names: list[str]
    stream_until: ClientState | None = None
    stop_on_notfound: bool = True
    max_retries: int = 10


@tools.post("/logs/stream")
async def stream_server_logs(
    body: LogsStreamBody,
    app: DiveHostAPI = Depends(get_app),
) -> StreamingResponse:
    """Stream logs from MCP servers.

    Args:
        body (LogsStreamBody):
            - names: MCP servers to listen for logs
            - stream_until: Stream until mcp server state matches the provided state
            - stop_on_notfound: Stop streaming if mcp server is not found
            - max_retries: Retry N times to listen for logs
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
                stream_until=body.stream_until,
                stop_on_notfound=body.stop_on_notfound,
                max_retries=body.max_retries,
                server_names=body.names,
            )
            await processor.stream_logs()

    stream.add_task(process)
    return response


class OAuthProcessor:
    """OAuth processor."""

    def __init__(
        self,
        stream: EventStreamContextManager,
        oauth_manager: OAuthManager,
        server: McpServer,
        server_name: str,
    ) -> None:
        """Initialize OAuth processor."""
        self.stream = stream
        self.server_name = server_name
        self.server = server
        self.oauth_manager = oauth_manager

    async def execute(self) -> None:
        """Execute OAuth."""
        progress = await self.server.create_oauth_authorization()
        await self.stream.write(
            OAuthResult(
                success=progress.error is None,
                server_name=self.server_name,
                stage=progress.type,
                auth_url=progress.auth_url,
                error=progress.error,
            ).model_dump_json(by_alias=True, exclude_none=True)
        )
        if state := progress.state:
            result = await self.oauth_manager.wait_authorization(state)
            await self.stream.write(
                OAuthResult(
                    success=True,
                    server_name=self.server_name,
                    stage=result.type,
                    error=result.error,
                ).model_dump_json(by_alias=True, exclude_none=True)
            )


@tools.post("/login/oauth")
async def login_oauth(
    oauth_request: OAuthRequest,
    app: DiveHostAPI = Depends(get_app),
) -> StreamingResponse:
    """Login to an OAuth provider."""
    stream = EventStreamContextManager()
    response = stream.get_response()

    oauth_manager = app.dive_host["default"].oauth_manager
    server = app.dive_host["default"].get_mcp_server(oauth_request.server_name)

    async def process() -> None:
        async with stream:
            processor = OAuthProcessor(
                stream=stream,
                oauth_manager=oauth_manager,
                server=server,
                server_name=oauth_request.server_name,
            )
            await processor.execute()

    stream.add_task(process)
    return response


@tools.get("/login/oauth/callback")
async def oauth_callback(
    code: str,
    state: str,
    app: DiveHostAPI = Depends(get_app),
) -> OAuthResult:
    """OAuth callback."""
    oauth_manager = app.dive_host["default"].oauth_manager
    try:
        server_name = await oauth_manager.set_oauth_code(code, state)
        if server_name is None:
            return OAuthResult(success=False, message="invalid state")
        task = create_task(oauth_manager.wait_authorization(state))
        result = await wait_for(task, timeout=10)
        if result.type == "auth_success":
            await app.dive_host["default"].restart_mcp_server(server_name)
        return OAuthResult(
            success=True,
            server_name=server_name,
            stage=result.type,
        )
    except Exception:
        logger.exception("cannot set oauth code")
        return OAuthResult(success=False, message="cannot set oauth code")


@tools.post("/login/oauth/delete")
async def delete_oauth(
    oauth_request: OAuthRequest,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Delete OAuth."""
    oauth_manager = app.dive_host["default"].oauth_manager
    await oauth_manager.store.delete(oauth_request.server_name)
    await app.dive_host["default"].restart_mcp_server(oauth_request.server_name)
    return ResultResponse(success=True)


@tools.get("/login/oauth/list")
async def list_oauth(
    app: DiveHostAPI = Depends(get_app),
) -> OAuthList:
    """List OAuth."""
    oauth_manager = app.dive_host["default"].oauth_manager
    return OAuthList(success=True, servers=await oauth_manager.store.list())
