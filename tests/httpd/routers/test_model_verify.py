import json
import os
import re
from contextlib import suppress
from typing import cast

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
}


def test_do_verify_model_with_env_api_key(test_client):
    """Test the /api/model_verify POST endpoint."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    client, _ = test_client

    # Prepare test data
    test_settings = {
        "modelSettings": {
            **MOCK_MODEL_SETTING,
            "apiKey": os.environ.get("OPENAI_API_KEY"),
        },
    }

    # Send request
    response = client.post(
        "/model_verify",
        json=test_settings,
    )

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = cast("dict", response.json())

    # Validate result structure
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "connecting": {
                "success": True,
                "final_state": "CONNECTED",
                "error_msg": None,
            },
            "supportTools": {
                "success": True,
                "final_state": "TOOL_RESPONDED",
                "error_msg": None,
            },
        },
    )


def test_verify_model_streaming_with_env_api_key(test_client):
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

    # Verify response headers for SSE
    assert "text/event-stream" in response.headers.get("Content-Type")
    assert "Cache-Control" in response.headers
    assert "Connection" in response.headers

    # Verify content contains expected SSE format
    content = response.content.decode()
    # assert the basic format
    assert "data: " in content
    assert "data: [DONE]\n\n" in content

    # extract and parse the JSON data
    data_messages = re.findall(r"data: (.*?)\n\n", content)
    for data in data_messages:
        if data != "[DONE]":
            json_obj = json.loads(data)
            assert "type" in json_obj
            if json_obj["type"] == "progress":
                step = json_obj.get("step")
                test_type = json_obj.get("testType")

                if step == 1 and test_type == "connection":
                    helper.dict_subset(
                        json_obj,
                        {
                            "step": 1,
                            "modelName": "gpt-4o-mini",
                            "testType": "connection",
                            "ok": True,
                            "finalState": "CONNECTED",
                            "error": None,
                        },
                    )
                elif step == 2 and test_type == "tools":
                    helper.dict_subset(
                        json_obj,
                        {
                            "step": 2,
                            "modelName": "gpt-4o-mini",
                            "testType": "tools",
                            "ok": True,
                            "finalState": "TOOL_RESPONDED",
                            "error": None,
                        },
                    )

            elif json_obj["type"] == "final":
                helper.dict_subset(
                    json_obj,
                    {
                        "type": "final",
                        "results": [
                            {
                                "modelName": "gpt-4o-mini",
                                "connection": {
                                    "ok": True,
                                    "finalState": "CONNECTED",
                                    "error": None,
                                },
                                "tools": {
                                    "ok": True,
                                    "finalState": "TOOL_RESPONDED",
                                    "error": None,
                                },
                            },
                        ],
                    },
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


def _check_verify_streaming_response(response: httpx.Response, model_name: str) -> None:
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
    check_final = False

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
                    {
                        "step": 2,
                        "modelName": model_name,
                        "testType": "tools",
                        "ok": True,
                        "finalState": "TOOL_RESPONDED",
                        "error": None,
                    },
                )
                check_tools = True
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
                                "ok": True,
                                "finalState": "TOOL_RESPONDED",
                                "error": None,
                            },
                        },
                    ],
                },
            )
            check_final = True

    assert check_connection
    assert check_tools
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
                },
            ],
        }
        response = client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
        _check_verify_streaming_response(response, olama_model)
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


def test_verify_azure_openapi(test_client):
    """Test the /api/model_verify POST endpoint with azure openai."""
    client, _ = test_client

    if (
        (api_key := os.environ.get("AZURE_OPENAI_API_KEY"))
        and (endpoint := os.environ.get("AZURE_OPENAI_ENDPOINT"))
        and (deployment_name := os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"))
        and (api_version := os.environ.get("AZURE_OPENAI_API_VERSION"))
    ):
        model_name = "gpt4"
        test_model_settings = {
            "modelSettings": [
                {
                    "model": model_name,
                    "modelProvider": "azure_openai",
                    "apiKey": api_key,
                    "apiVersion": api_version,
                    "azureEndpoint": endpoint,
                    "azureDeployment": deployment_name,
                    "maxTokens": 800,
                    "configuration": {
                        "temperature": 0.7,
                        "topP": 0,
                    },
                },
            ],
        }
        response = client.post(
            "/model_verify/streaming",
            json=test_model_settings,
        )
        _check_verify_streaming_response(response, model_name)
    else:
        pytest.skip(
            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME,"
            " AZURE_OPENAI_API_VERSION are not set"
        )
