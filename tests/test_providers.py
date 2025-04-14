import json
from os import environ
from typing import TYPE_CHECKING, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from langgraph.pregel.io import AddableUpdatesDict

from dive_mcp_host.host.chat import MessageChunkHolder
from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.conf.llm import (
    Credentials,
    LLMAzureConfig,
    LLMBedrockConfig,
    LLMConfig,
    LLMConfiguration,
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
        mcp_host.chat() as chat,
    ):
        got = False
        holders = MessageChunkHolder()
        ai_messages: list[AIMessage] = []
        async for response in chat.query(
            HumanMessage(content="echo helloXXX with 10ms delay"),
            stream_mode=["updates", "messages"],
        ):
            event_type, event_data = response
            if event_type == "messages":
                msg = event_data[0]
                if isinstance(msg, AIMessage) and (msg := holders.feed(msg)):
                    ai_messages.append(msg)
                continue
            event_data = cast("AddableUpdatesDict", event_data)
            if msg_dict := event_data.get("tools"):
                rep = {}
                for msg in msg_dict.get("messages", []):
                    if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                        rep = json.loads(msg.content)[0]
                assert helper.dict_subset(
                    rep, {"type": "text", "text": "helloXXX", "annotations": None}
                )
                got = True
        assert got, "no tool message found"
        assert len(ai_messages) >= 2
        assert len(ai_messages[0].tool_calls) == 1
        helper.dict_subset(
            dict(ai_messages[0].tool_calls[0]),
            model_tool_call,
        )


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
        config = HostConfig(
            llm=LLMConfig(
                model="claude-3-7-sonnet-20250219",
                model_provider="anthropic",
                api_key=api_key,
            ),
            mcp_servers=echo_tool_stdio_config,
        )
    else:
        pytest.skip("need environment variable ANTHROPIC_API_KEY to run this test")

    await _run_the_test(config)


@pytest.mark.asyncio
async def test_host_openai(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
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
                api_key=api_key,
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
                api_key=api_key,
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
async def test_bedrock(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if (key_id := environ.get("BEDROCK_ACCESS_KEY_ID")) and (
        access_key := environ.get("BEDROCK_SECRET_ACCESS_KEY")
    ):
        token = environ.get("BEDROCK_SESSION_TOKEN")
        config = HostConfig(
            llm=LLMBedrockConfig(
                model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                model_provider="bedrock",
                credentials=Credentials(
                    access_key_id=key_id,
                    secret_access_key=access_key,
                    session_token=token or "",
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
async def test__mistralai(echo_tool_stdio_config: dict[str, ServerConfig]) -> None:
    """Test the host context initialization."""
    if api_key := environ.get("MISTRAL_API_KEY"):
        config = HostConfig(
            llm=LLMConfig(
                model="mistral-large-latest",
                model_provider="mistralai",
                api_key=api_key,
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
                api_key=api_key,
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
                api_key=api_key,
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
