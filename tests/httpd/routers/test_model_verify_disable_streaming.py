import os
from contextlib import suppress
from typing import Any

import httpx
import pytest
from fastapi import status
from openai import AuthenticationError

from tests import helper

MOCK_MODEL_SETTING = {
    "model": "gpt-4o-mini",
    "modelProvider": "openai",
    "apiKey": "openai-api-key",
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokens": 4000,
    "active": True,
    "configuration": {"base_url": "https://api.openai.com/v1"},
    "disableStreaming": True,
}


def test_verify_openai(test_client):
    """Test the /api/model_verify/streaming POST endpoint."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    # Get client and app from test_client fixture
    client, _ = test_client

    # Prepare test data
    test_model_settings = {
        "modelSettings": [
            {
                **MOCK_MODEL_SETTING,
                "apiKey": os.environ.get("OPENAI_API_KEY"),
            },
        ],
    }

    # Send request
    response = client.post(
        "/model_verify/streaming",
        json=test_model_settings,
    )

    # Verify response status code for SSE stream
    assert response.status_code == status.HTTP_200_OK

    _check_verify_streaming_response(
        response,
        MOCK_MODEL_SETTING["model"],
    )


def test_do_verify_model_with_mock_key_should_fail(test_client):
    """Test the /api/model_verify POST endpoint with mock API key."""
    client, _ = test_client

    # Prepare test data
    test_settings = {
        "modelSettings": MOCK_MODEL_SETTING,
    }

    with suppress(AuthenticationError):
        client.post(
            "/model_verify",
            json=test_settings,
        )


def test_verify_model_streaming_with_mock_key_should_fail(test_client):
    """Test the /api/model_verify/streaming POST endpoint with mock API key."""
    # Get client and app from test_client fixture
    client, _ = test_client

    # Prepare test data
    test_model_settings = {
        "modelSettings": [
            MOCK_MODEL_SETTING,
        ],
    }

    with suppress(AuthenticationError):
        client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )


def _check_verify_streaming_response(
    response: httpx.Response,
    model_name: str,
    tool_result_override: dict[str, Any] | None = None,
    tool_in_prompt_result_override: dict[str, Any] | None = None,
) -> None:
    assert response.status_code == status.HTTP_200_OK, response.content

    # Verify response headers for SSE
    assert "text/event-stream" in response.headers.get("Content-Type")
    assert "Cache-Control" in response.headers
    assert "Connection" in response.headers

    # Verify content contains expected SSE format
    content = response.content.decode()
    # assert the basic format
    assert "data: " in content
    assert "data: [DONE]\n\n" in content

    check_connection = False
    check_tools = False
    check_tools_in_prompt = False
    check_final = False
    tool_result = {
        "step": 2,
        "modelName": model_name,
        "testType": "tools",
        "ok": True,
        "finalState": "TOOL_RESPONDED",
        "error": None,
    }
    if tool_result_override:
        tool_result.update(tool_result_override)
    tool_in_prompt_result = {
        "step": 3,
        "modelName": model_name,
        "testType": "tools_in_prompt",
        "ok": False,
        "finalState": "SKIPPED",
        "error": None,
    }
    if tool_in_prompt_result_override:
        tool_in_prompt_result.update(tool_in_prompt_result_override)

    # extract and parse the JSON data
    for json_obj in helper.extract_stream(content):
        assert "type" in json_obj
        if json_obj["type"] == "progress":
            step = json_obj.get("step")
            test_type = json_obj.get("testType")

            if step == 1 and test_type == "connection":
                helper.dict_subset(
                    json_obj,
                    {
                        "step": 1,
                        "modelName": model_name,
                        "testType": "connection",
                        "ok": True,
                        "finalState": "CONNECTED",
                        "error": None,
                    },
                )
                check_connection = True
            elif step == 2 and test_type == "tools":
                helper.dict_subset(
                    json_obj,
                    tool_result,
                )
                check_tools = True
            elif step == 3 and test_type == "tools_in_prompt":
                helper.dict_subset(
                    json_obj,
                    tool_in_prompt_result,
                )
                check_tools_in_prompt = True
        elif json_obj["type"] == "final":
            helper.dict_subset(
                json_obj,
                {
                    "type": "final",
                    "results": [
                        {
                            "modelName": model_name,
                            "connection": {
                                "ok": True,
                                "finalState": "CONNECTED",
                                "error": None,
                            },
                            "tools": {
                                "ok": tool_result["ok"],
                                "finalState": tool_result["finalState"],
                                "error": tool_result["error"],
                            },
                            "toolsInPrompt": {
                                "ok": tool_in_prompt_result["ok"],
                                "finalState": tool_in_prompt_result["finalState"],
                                "error": tool_in_prompt_result["error"],
                            },
                        },
                    ],
                },
            )
            check_final = True

    assert check_connection
    assert check_tools
    assert check_tools_in_prompt
    assert check_final


def test_verify_ollama(test_client):
    """Test the /api/model_verify POST endpoint with ollama."""
    client, _ = test_client

    if (base_url := os.environ.get("OLLAMA_URL")) and (
        olama_model := os.environ.get("OLLAMA_MODEL")
    ):
        test_model_settings = {
            "modelSettings": [
                {
                    "model": olama_model,
                    "provider": "ollama",
                    "modelProvider": "ollama",
                    "configuration": {"baseURL": base_url},
                    "disableStreaming": True,
                },
            ],
        }
        response = client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
        _check_verify_streaming_response(
            response,
            olama_model,
            tool_result_override={
                "finalState": "SKIPPED",
                "ok": False,
            },
            tool_in_prompt_result_override={
                "finalState": "TOOL_RESPONDED",
                "ok": True,
            },
        )
    else:
        pytest.skip("OLLAMA_URL is not set")


def test_verify_google(test_client):
    """Test the /api/model_verify POST endpoint with google."""
    client, _ = test_client

    if api_key := os.environ.get("GOOGLE_API_KEY"):
        model_name = "gemini-2.0-flash"
        test_model_settings = {
            "modelSettings": [
                {
                    "model": model_name,
                    "modelProvider": "google-genai",
                    "apiKey": api_key,
                    "configuration": {
                        "temperature": 0.0,
                        "topP": 0,
                    },
                    "disableStreaming": True,
                },
            ],
        }
        response = client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
        _check_verify_streaming_response(response, model_name)
    else:
        pytest.skip("GOOGLE_API_KEY is not set")


def test_verify_anthropic(test_client):
    """Test the /api/model_verify POST endpoint with anthropic."""
    client, _ = test_client

    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        model_name = "claude-3-7-sonnet-20250219"
        test_model_settings = {
            "modelSettings": [
                {
                    "model": model_name,
                    "modelProvider": "anthropic",
                    "apiKey": api_key,
                    "disableStreaming": True,
                },
            ],
        }
        response = client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
        _check_verify_streaming_response(response, model_name)
    else:
        pytest.skip("ANTHROPIC_API_KEY is not set")


def test_verify_bedrock(test_client):
    """Test the /api/model_verify POST endpoint with bedrock."""
    client, _ = test_client

    if (key_id := os.environ.get("BEDROCK_ACCESS_KEY_ID")) and (
        access_key := os.environ.get("BEDROCK_SECRET_ACCESS_KEY")
    ):
        model_name = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        token = os.environ.get("BEDROCK_SESSION_TOKEN")
        test_model_settings = {
            "modelSettings": [
                {
                    "model": model_name,
                    "modelProvider": "bedrock",
                    "credentials": {
                        "accessKeyId": key_id,
                        "secretAccessKey": access_key,
                        "sessionToken": token or "",
                    },
                    "region": "us-east-1",
                    "disableStreaming": True,
                },
            ],
        }
        response = client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
        _check_verify_streaming_response(response, model_name)
    else:
        pytest.skip("BEDROCK_ACCESS_KEY_ID and BEDROCK_SECRET_ACCESS_KEY are not set")


def test_verify_mistralai(test_client):
    """Test the /api/model_verify POST endpoint with mistralai."""
    client, _ = test_client

    if api_key := os.environ.get("MISTRAL_API_KEY"):
        model_name = "mistral-large-latest"
        test_model_settings = {
            "modelSettings": [
                {
                    "model": model_name,
                    "modelProvider": "mistralai",
                    "apiKey": api_key,
                    "configuration": {
                        "temperature": 0.5,
                        "topP": 0.5,
                        "baseURL": "https://api.mistral.ai/v1",
                    },
                    "disableStreaming": True,
                },
            ],
        }
        response = client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
        _check_verify_streaming_response(response, model_name)
    else:
        pytest.skip("MISTRAL_API_KEY is not set")


def test_verify_siliconflow(test_client):
    """Test the /api/model_verify POST endpoint with siliconflow."""
    client, _ = test_client

    if api_key := os.environ.get("SILICONFLOW_API_KEY"):
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        test_model_settings = {
            "modelSettings": [
                {
                    "model": model_name,
                    "modelProvider": "openai",
                    "apiKey": api_key,
                    "configuration": {
                        "baseURL": "https://api.siliconflow.com/v1",
                    },
                    "disableStreaming": True,
                },
            ],
        }
        response = client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
        _check_verify_streaming_response(response, model_name)
    else:
        pytest.skip("SILICONFLOW_API_KEY is not set")
