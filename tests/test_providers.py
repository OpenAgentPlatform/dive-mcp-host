import json
from os import environ
from typing import TYPE_CHECKING, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import SecretStr

if TYPE_CHECKING:
    from langgraph.pregel.io import AddableUpdatesDict

from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.conf.llm import (
    Credentials,
    LLMAnthropicConfig,
    LLMAzureConfig,
    LLMBedrockConfig,
    LLMConfig,
    LLMConfiguration,
    LLMOapConfig,
    LLMOapConfiguration,
)
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.host.tools import ServerConfig
from tests import helper


async def _run_the_test(
    config: HostConfig,
    model_tool_call: dict | None = None,
) -> None:
    """Run the test."""
    model_tool_call = model_tool_call or {
        "name": "echo",
        "args": {"delay_ms": 10, "message": "helloXXX"},
    }
    async with (
        DiveMcpHost(config) as mcp_host,
    ):
        await mcp_host.tools_initialized_event.wait()
        async with mcp_host.chat() as chat:
            got_tool_msg = False
            ai_messages: list[AIMessage] = []
            async for response in chat.query(
                HumanMessage(content="echo helloXXX with 10ms delay"),
                stream_mode=["updates", "messages"],
            ):
                event_type, event_data = response
                # if event_type == "messages":
                #     msg = event_data[0]
                #     continue
                if event_type == "updates":
                    event_data = cast("AddableUpdatesDict", event_data)
                    for msg_dict in event_data.values():
                        rep = {}
                        for msg in msg_dict.get("messages", []):
                            if isinstance(msg, ToolMessage) and isinstance(
                                msg.content, str
                            ):
                                rep = json.loads(msg.content)[0]
                                assert helper.dict_subset(
                                    rep,
                                    {
                                        "type": "text",
                                        "text": "helloXXX",
                                        "annotations": None,
                                    },
                                )
                                got_tool_msg = True
                            if isinstance(msg, AIMessage):
                                ai_messages.append(msg)
        assert got_tool_msg, "no tool message found"
        assert len(ai_messages) >= 2
        for i in ai_messages:
            if i.tool_calls:
                helper.dict_subset(
                    dict(i.tool_calls[0]),
                    model_tool_call,
                )
                break
        else:
            raise Exception("no tool calls found")


@pytest.mark.asyncio
async def test_ollama(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    echo_tool_stdio_config["fetch"] = ServerConfig(
        name="fetch",
        command="uvx",
        args=["mcp-server-fetch"],
        transport="stdio",
    )
    if (base_url := environ.get("OLLAMA_URL")) and (
        olama_model := environ.get("OLLAMA_MODEL")
    ):
        config = HostConfig(
            llm=LLMConfig(
                model=olama_model,
                model_provider="ollama",
                configuration=LLMConfiguration(
                    baseURL=base_url,
                ),
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip(
            "need environment variable OLLAMA_URL and OLLAMA_MODEL to run this test"
        )

    await _run_the_test(config)


@pytest.mark.asyncio
async def test_anthropic(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if api_key := environ.get("ANTHROPIC_API_KEY"):
        pass
    else:
        pytest.skip("need environment variable ANTHROPIC_API_KEY to run this test")

    async def _one_model(model: str):
        config = HostConfig(
            llm=LLMAnthropicConfig(
                model=model,
                model_provider="anthropic",
                api_key=SecretStr(api_key),
            ),
            mcp_servers=echo_tool_stdio_config,
        )

        await _run_the_test(config)

    for m in ["claude-3-7-sonnet-20250219"]:
        await _one_model(m)


@pytest.mark.asyncio
async def test_openai(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    echo_tool_stdio_config["fetch"] = ServerConfig(
        name="fetch",
        command="uvx",
        args=["mcp-server-fetch"],
        transport="stdio",
    )
    if api_key := environ.get("OPENAI_API_KEY"):
        config = HostConfig(
            llm=LLMConfig(
                model="gpt-4o-mini",
                model_provider="openai",
                api_key=SecretStr(api_key),
                configuration=LLMConfiguration(
                    temperature=0.0,
                    top_p=0,
                ),
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip("need environment variable OPENAI_API_KEY to run this test")

    await _run_the_test(config)


@pytest.mark.asyncio
async def test_host_google(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if api_key := environ.get("GOOGLE_API_KEY"):
        config = HostConfig(
            llm=LLMConfig(
                model="gemini-2.0-flash",
                model_provider="google-genai",
                api_key=SecretStr(api_key),
                configuration=LLMConfiguration(
                    temperature=0.0,
                    top_p=0,
                ),
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip("need environment variable GOOGLE_API_KEY to run this test")

    await _run_the_test(config)


@pytest.mark.asyncio
async def test_host_google_image_gen(
    echo_tool_stdio_config: dict[str, ServerConfig],
) -> None:
    """Test the host context initialization."""
    if api_key := environ.get("GOOGLE_API_KEY"):
        config = HostConfig(
            llm=LLMConfig(
                model="gemini-2.5-flash-image-preview",
                model_provider="google-genai",
                api_key=SecretStr(api_key),
                configuration=LLMConfiguration(
                    temperature=0.0,
                    top_p=0,
                ),
                tools_in_prompt=True,
                disable_streaming=True,
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip("need environment variable GOOGLE_API_KEY to run this test")

    await _run_the_test(config)


@pytest.mark.asyncio
async def test_bedrock(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    # claude-3-7-sonnet
    if (key_id := environ.get("BEDROCK_ACCESS_KEY_ID")) and (
        access_key := environ.get("BEDROCK_SECRET_ACCESS_KEY")
    ):
        token = environ.get("BEDROCK_SESSION_TOKEN")
        config = HostConfig(
            llm=LLMBedrockConfig(
                model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                model_provider="bedrock",
                credentials=Credentials(
                    access_key_id=SecretStr(key_id),
                    secret_access_key=SecretStr(access_key),
                    session_token=SecretStr(token or ""),
                ),
                region="us-east-1",
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip(
            "need environment variable BEDROCK_ACCESS_KEY_ID,"
            " BEDROCK_SECRET_ACCESS_KEY, BEDROCK_SESSION_TOKEN to run this test"
        )

    await _run_the_test(config)

    # claude-sonnet-4
    if (key_id := environ.get("BEDROCK_ACCESS_KEY_ID")) and (
        access_key := environ.get("BEDROCK_SECRET_ACCESS_KEY")
    ):
        token = environ.get("BEDROCK_SESSION_TOKEN")
        config = HostConfig(
            llm=LLMBedrockConfig(
                model="us.anthropic.claude-sonnet-4-20250514-v1:0",
                model_provider="bedrock",
                credentials=Credentials(
                    access_key_id=SecretStr(key_id),
                    secret_access_key=SecretStr(access_key),
                    session_token=SecretStr(token or ""),
                ),
                region="us-east-1",
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip(
            "need environment variable BEDROCK_ACCESS_KEY_ID,"
            " BEDROCK_SECRET_ACCESS_KEY, BEDROCK_SESSION_TOKEN to run this test"
        )

    await _run_the_test(config)


@pytest.mark.asyncio
async def test_mistralai(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if api_key := environ.get("MISTRAL_API_KEY"):
        config = HostConfig(
            llm=LLMConfig(
                model="mistral-large-latest",
                model_provider="mistralai",
                api_key=SecretStr(api_key),
                configuration=LLMConfiguration(
                    temperature=0.5,
                    top_p=0.5,
                    baseURL="https://api.mistral.ai/v1",
                ),
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip("need environment variable MISTRAL_API_KEY to run this test")

    await _run_the_test(config)


@pytest.mark.asyncio
async def test_siliconflow(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if api_key := environ.get("SILICONFLOW_API_KEY"):
        config = HostConfig(
            llm=LLMConfig(
                model="Qwen/Qwen2.5-7B-Instruct",
                model_provider="openai",
                api_key=SecretStr(api_key),
                configuration=LLMConfiguration(
                    temperature=0.5,
                    top_p=0.5,
                    baseURL="https://api.siliconflow.com/v1",
                ),
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip("need environment variable SILICONFLOW_API_KEY to run this test")
    await _run_the_test(config)


@pytest.mark.asyncio
async def test_openrouter(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if api_key := environ.get("OPENROUTER_API_KEY"):
        config = HostConfig(
            llm=LLMConfig(
                model="qwen/qwen3-30b-a3b",
                model_provider="openai",
                api_key=SecretStr(api_key),
                tools_in_prompt=True,
                configuration=LLMConfiguration(
                    temperature=0.5, top_p=0.5, baseURL="https://openrouter.ai/api/v1"
                ),
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip("need environment variable OPENROUTER_API_KEY to run this test")
    await _run_the_test(config)


@pytest.mark.asyncio
async def test_azure(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if (
        (api_key := environ.get("AZURE_OPENAI_API_KEY"))
        and (endpoint := environ.get("AZURE_OPENAI_ENDPOINT"))
        and (deployment_name := environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"))
        and (api_version := environ.get("AZURE_OPENAI_API_VERSION"))
    ):
        config = HostConfig(
            llm=LLMAzureConfig(
                model="gpt4",
                model_provider="azure_openai",
                api_key=SecretStr(api_key),
                azure_endpoint=endpoint,
                azure_deployment=deployment_name,
                api_version=api_version,
                max_tokens=800,
                configuration=LLMConfiguration(
                    temperature=0.7,
                    top_p=0,
                ),
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip(
            "need environment variable AZURE_API_KEY, AZURE_ENDPOINT,"
            " AZURE_DEPLOYMENT_NAME, AZURE_API_VERSION to run this test"
        )

    await _run_the_test(config)


@pytest.mark.asyncio
async def test_oap(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test oap model.

    It support large content window of anthropic claude 3.x.
    """
    configuration = LLMOapConfiguration()
    if base_url := environ.get("OAP_BASE_URL"):
        configuration.base_url = base_url
    if api_key := environ.get("OAP_API_KEY"):
        config = HostConfig(
            llm=LLMOapConfig(
                model="anthropic/claude_sonnet_4.5",
                model_provider="oap",
                api_key=SecretStr(api_key),
                configuration=configuration,
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip("need environment variable OAP_API_KEY to run this test")
    await _run_the_test(config)
