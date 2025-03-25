from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from dive_mcp_host.httpd.app import DiveHostAPI
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.routers.model_verify import (
    model_verify,
)

ModelVerificationStatus = Literal["success", "warning", "error"]


class ModelVerificationResult(BaseModel):
    """Result of model verification containing connection and tools test results."""

    model_name: str = Field(alias="modelName")
    connecting_status: ModelVerificationStatus = Field(alias="connectingStatus")
    connecting_result: object | None = Field(alias="connectingResult")
    support_tools_status: ModelVerificationStatus = Field(alias="supportToolsStatus")
    support_tools_result: object | None = Field(alias="supportToolsResult")


class ModelVerificationFinalResponse(BaseModel):
    """Final response model for model verification results."""

    type: Literal["final"]
    results: list[ModelVerificationResult]
    aborted: bool


MOCK_MODEL_SETTING = {
    "model": "gpt-4o-mini",
    "modelProvider": "openai",
    "apiKey": "openai-api-key",
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokens": 4000,
    "configuration": {"baseURL": "https://api.openai.com/v1"},
}


class MockDiveHostAPI:
    """Mock DiveHostAPI for testing."""

    def __init__(self):
        self.dive_host = Mock()
        self.conversation = AsyncMock()
        self.conversation.__aenter__ = AsyncMock()
        self.conversation.__aexit__ = AsyncMock()

        # simulate streaming response
        self.conversation.query = AsyncMock()
        self.conversation.query.return_value = [{"content": "test response"}]

        self.dive_host.conversation.return_value = self.conversation


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    app = DiveHostAPI()
    app.include_router(model_verify, prefix="/api/model_verify")

    mock_app = MockDiveHostAPI()

    def get_mock_app():
        return Mock(dive_host={"default": mock_app.dive_host})

    app.dependency_overrides[get_app] = get_mock_app

    with TestClient(app) as client:
        yield client


def test_do_verify_model(client):
    """Test the /api/model_verify POST endpoint."""
    # Prepare test data
    test_settings = {
        "modelSettings": MOCK_MODEL_SETTING,
    }

    # Send request
    response = client.post(
        "/api/model_verify",
        json=test_settings,
    )

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert response_data["success"] is True

    # Validate result structure
    assert isinstance(response_data["connectingSuccess"], bool)
    assert isinstance(response_data["supportTools"], bool)


def test_verify_model_streaming(client):
    """Test the /api/model_verify/streaming POST endpoint."""
    # Prepare test data
    test_model_settings = {
        "modelSettings": [
            MOCK_MODEL_SETTING,
        ],
    }

    # Send request
    response = client.post(
        "/api/model_verify/streaming",
        json=test_model_settings,
    )

    # Verify response status code for SSE stream
    assert response.status_code == status.HTTP_200_OK

    # Verify response headers for SSE
    assert "text/event-stream" in response.headers["Content-Type"]
    assert "Cache-Control" in response.headers
    assert "Connection" in response.headers

    # Verify content contains expected SSE format
    content = response.content.decode()
    assert "data: " in content

    # Verify stream transmission ends correctly
    assert "data: [DONE]" in content

    # Check if content contains expected keywords
    found_fields = []
    important_fields = ["type", "modelName", "step", "testType", "status"]

    for field in important_fields:
        if field in content:
            found_fields.append(field)

    assert found_fields, "no expected fields found in the response"


def test_model_verification_response_structure():
    """Test that ModelVerificationFinalResponse can be properly serialized."""
    # Create a sample response
    response = ModelVerificationFinalResponse(
        type="final",
        results=[
            ModelVerificationResult(
                modelName="gpt-4",
                connectingStatus="success",
                connectingResult="Test connection result",
                supportToolsStatus="success",
                supportToolsResult="Test tools result",
            ),
        ],
        aborted=False,
    )

    # Convert to dict
    response_dict = response.model_dump(by_alias=True)

    # Validate structure
    assert isinstance(response_dict["type"], str)
    assert response_dict["type"] == "final"
    assert isinstance(response_dict["results"], list)
    assert len(response_dict["results"]) == 1
    assert isinstance(response_dict["aborted"], bool)

    # Validate first result structure
    result = response_dict["results"][0]
    assert isinstance(result["modelName"], str)
    assert result["modelName"] == "gpt-4"
    assert isinstance(result["connectingStatus"], str)
    assert result["connectingStatus"] == "success"
    assert result["connectingResult"] == "Test connection result"
    assert isinstance(result["supportToolsStatus"], str)
    assert result["supportToolsStatus"] == "success"
    assert result["supportToolsResult"] == "Test tools result"
