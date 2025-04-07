import json
from os import environ
from typing import cast

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from dive_mcp_host.host.conf import HostConfig
from dive_mcp_host.host.conf.llm import (
    Credentials,
    LLMBedrockConfig,
    LLMConfig,
    LLMConfiguration,
)
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.host.tools import ServerConfig


async def _run_the_test(
    config: HostConfig,
) -> None:
    """Run the test."""
    async with (
        DiveMcpHost(config) as mcp_host,
        mcp_host.conversation(
            # system_prompt=system_prompt(""),
            # tools=[TestTool()],
        ) as conversation,
    ):
        # r = await conversation.invoke("test mcp tool echo with 'hello'")
        async for response in conversation.query(
            HumanMessage(content="echo helloXXX with 10ms delay"),
            stream_mode=["updates"],
        ):
            response = cast("tuple[str, dict[str, dict[str, BaseMessage]]]", response)
            if msg_dict := response[1].get("tools"):
                contents = list[str]()
                for msg in msg_dict.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        # XXX the content type is complex.
                        if isinstance(msg.content, str):
                            try:
                                rep = json.loads(msg.content)
                            except json.JSONDecodeError:
                                continue
                        else:
                            rep = msg.content
                        for r in rep:
                            assert r["type"] == "text"  # type: ignore[index]
                            contents.append(r["text"])  # type: ignore[index]
                assert any("helloXXX" in c for c in contents)


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
                model="o3-mini",
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
        pytest.skip("need environment variable GOOGLE_API_KEY to run this test")

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
