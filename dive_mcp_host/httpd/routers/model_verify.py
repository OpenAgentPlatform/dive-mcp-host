import asyncio
import json
from collections.abc import Callable, Generator
from contextlib import AsyncExitStack, contextmanager
from logging import getLogger
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.conf.llm import LLMConfig, LLMConfigTypes
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.host.tools.misc import TestTool
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.utils import EventStreamContextManager
from dive_mcp_host.httpd.server import DiveHostAPI

logger = getLogger(__name__)

model_verify = APIRouter(tags=["model_verify"])


class ModelVerifyResult(BaseModel):
    """Model verify result."""

    success: bool = False
    connecting_success: bool = Field(default=False, alias="connectingSuccess")
    # connecting_result: str = Field(default="", alias="connectingResult")
    support_tools: bool = Field(default=False, alias="supportTools")
    # support_tools_result: str = Field(default="", alias="supportToolsResult")


class ModelVerifyService:
    """Model verify service."""

    _abort_signal: asyncio.Event = asyncio.Event()
    _stream_progress: Callable | None = None

    def __init__(self, stream_progress: Callable | None = None) -> None:
        """Initialize the model verify service.

        Args:
            stream_progress (Callable): The stream progress callback.
        """
        self._abort_signal = asyncio.Event()
        self._stream_progress = stream_progress

    async def test_models(
        self,
        llm_configs: list[LLMConfigTypes],
        subjects: list[Literal["connection", "tools"]],
    ) -> dict[str, ModelVerifyResult]:
        """Test the models.

        Args:
            llm_configs (list[LLMConfig]): The LLM configurations.
            subjects (list[Literal["connection", "tools"]]): The subjects to verify.

        Returns:
            dict[str, ModelVerifyResult]: The results.
        """
        results = {}
        for llm_config in llm_configs:
            if self._abort_signal.is_set():
                break
            results[llm_config.model] = await self.test_model(
                llm_config, subjects, llm_configs.index(llm_config) * len(subjects)
            )
        return results

    def abort(self) -> None:
        """Abort the test."""
        self._abort_signal.set()

    @contextmanager
    def _handle_abort(self, abort_func: Callable) -> Generator[None, None, None]:
        async def wait_for_abort() -> None:
            await self._abort_signal.wait()
            abort_func()

        task = asyncio.create_task(wait_for_abort())
        try:
            yield
        finally:
            task.cancel()

    async def _report_progress(
        self,
        step: int,
        model_name: str,
        test_type: str,
        status: bool,
        error: str | None,
    ) -> None:
        if not self._stream_progress:
            return
        await self._stream_progress(
            {
                "type": "progress",
                "step": step,
                "modelName": model_name,
                "testType": test_type,
                "status": "success" if status else "error",
                "error": error,
            }
        )

    async def test_model(
        self,
        llm_config: LLMConfigTypes,
        subjects: list[Literal["connection", "tools"]],
        steps: int,
    ) -> ModelVerifyResult:
        """Run the model.

        Args:
            llm_config (LLMConfig): The LLM configuration.
            subjects (list[Literal["connection", "tools"]]): The subjects to verify.
            steps (int): The steps to verify.

        Returns:
            ModelVerifyResult
        """
        host = DiveMcpHost(HostConfig(llm=llm_config, mcp_servers={}))
        async with host:
            con_ok = False
            tools_ok = False
            n_step = steps
            if "connection" in subjects:
                n_step += 1
                con_ok, con_error = await self._check_connection(host)
                await self._report_progress(
                    n_step, llm_config.model, "connection", con_ok, con_error
                )
            if "tools" in subjects:
                n_step += 1
                tools_ok, tools_error = await self._check_tools(host)
                await self._report_progress(
                    n_step, llm_config.model, "tools", tools_ok, tools_error
                )
            return ModelVerifyResult(
                success=True,
                connectingSuccess=con_ok,
                supportTools=tools_ok,
            )

    async def _check_connection(self, host: DiveMcpHost) -> tuple[bool, str | None]:
        """Check if the model is connected."""
        try:
            chat = host.chat(volatile=True)
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(chat)
                stack.enter_context(self._handle_abort(chat.abort))
                _responses = [
                    response
                    async for response in chat.query(
                        "Only return 'Hi' strictly", stream_mode=["updates"]
                    )
                ]
            return True, None
        except Exception as e:
            logger.exception("Failed to check connection")
            return False, str(e)

    async def _check_tools(self, host: DiveMcpHost) -> tuple[bool, str | None]:
        """Check if the model supports tools."""
        try:
            test_tool = TestTool()
            chat = host.chat(volatile=True, tools=[test_tool])
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(chat)
            stack.enter_context(self._handle_abort(chat.abort))
            _responses = [
                response
                async for response in chat.query(
                    "run test_tool", stream_mode=["updates"]
                )
            ]
            return test_tool.called, None
        except Exception as e:
            logger.exception("Failed to check tools")
            return False, str(e)


class ModelVerifyRequest(BaseModel):
    """Model verify request."""

    model_settings: LLMConfig | None = Field(alias="modelSettings", default=None)


@model_verify.post("")
async def do_verify_model(
    app: DiveHostAPI = Depends(get_app),
    settings: ModelVerifyRequest | None = None,
) -> ModelVerifyResult:
    """Verify if a model supports streaming capabilities.

    Returns:
        ModelVerifyResult
    """
    dive_host = app.dive_host["default"]

    llm_config = settings.model_settings if settings else None

    if not llm_config:
        llm_config = dive_host._config.llm  # noqa: SLF001

    test_service = ModelVerifyService()
    return await test_service.test_model(llm_config, ["connection", "tools"], 0)


@model_verify.post("/streaming")
async def verify_model(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
    settings: dict[str, list[LLMConfigTypes]] | None = None,
) -> StreamingResponse:
    """Verify if a model supports streaming capabilities.

    Returns:
        CompletionEventStreamContextManager
    """
    dive_host = app.dive_host["default"]

    llm_configs = settings.get("modelSettings") if settings else None
    if not llm_configs:
        llm_configs = [dive_host.config.llm]
    stream = EventStreamContextManager()
    test_service = ModelVerifyService(lambda x: stream.write(json.dumps(x)))
    response = stream.get_response()

    @contextmanager
    def handle_connection() -> Generator[None, None, None]:
        async def abort_func() -> None:
            while not await request.is_disconnected():
                await asyncio.sleep(1)
            test_service.abort()

        task = asyncio.create_task(abort_func())
        try:
            yield
        finally:
            task.cancel()

    async def process() -> None:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(stream)
            stack.enter_context(handle_connection())
            results = await test_service.test_models(
                llm_configs, ["connection", "tools"]
            )
            await stream.write(
                json.dumps(
                    {
                        "type": "final",
                        "results": [
                            {
                                "modelName": n,
                                "connection": {
                                    "status": "success"
                                    if r.connecting_success
                                    else "error",
                                    "result": "",  # r.connecting_result,
                                },
                                "tools": {
                                    "status": "success" if r.support_tools else "error",
                                    "result": "",  # r.support_tools_result,
                                },
                            }
                            for n, r in results.items()
                        ],
                    }
                )
            )

    stream.add_task(process)
    return response
