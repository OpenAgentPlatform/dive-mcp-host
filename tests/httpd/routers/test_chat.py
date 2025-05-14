import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fastapi import status
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from dive_mcp_host.httpd.routers.models import SortBy
from dive_mcp_host.httpd.server import DiveHostAPI

if TYPE_CHECKING:
    from dive_mcp_host.host.host import DiveMcpHost

from dive_mcp_host.httpd.database.models import Chat, Message
from dive_mcp_host.models.fake import FakeMessageToolModel
from tests import helper

from .conftest import TEST_CHAT_ID

# Constants
SUCCESS_CODE = status.HTTP_200_OK
BAD_REQUEST_CODE = status.HTTP_400_BAD_REQUEST


@dataclass
class ChatWithMessages:
    """Mock chat with messages response."""

    chat: Chat
    messages: list[Message]


def test_list_chat_with_sort_by(test_client: tuple[TestClient, DiveHostAPI]):
    """Test the /api/chat/list endpoint with sort by."""
    client, app = test_client

    # create another chat
    test_chat_id_2 = str(uuid.uuid4())
    response = client.post(
        "/api/chat",
        data={"message": "Hello World 22222", "chatId": test_chat_id_2},
    )
    assert response.status_code == SUCCESS_CODE

    # create another message
    response = client.post(
        "/api/chat",
        data={"message": "Hello World in old chat", "chatId": TEST_CHAT_ID},
    )
    assert response.status_code == SUCCESS_CODE

    # sort by chat
    response = client.get("/api/chat/list", params={"sort_by": SortBy.CHAT})
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "data": [
                {
                    "id": test_chat_id_2,
                    "title": "I am a fake model.",
                    "user_id": None,
                },
                {
                    "id": TEST_CHAT_ID,
                    "title": "I am a fake model.",
                    "user_id": None,
                },
            ],
        },
    )

    # sort by message
    response = client.get("/api/chat/list", params={"sort_by": SortBy.MESSAGE})
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "data": [
                {
                    "id": TEST_CHAT_ID,
                    "title": "I am a fake model.",
                    "user_id": None,
                },
                {
                    "id": test_chat_id_2,
                    "title": "I am a fake model.",
                    "user_id": None,
                },
            ],
        },
    )


def test_list_chat(test_client):
    """Test the /api/chat/list endpoint."""
    client, app = test_client

    # Call the API
    response = client.get("/api/chat/list")
    # Verify response status code
    assert response.status_code == SUCCESS_CODE
    # Parse JSON response
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "data": [
                {
                    "id": TEST_CHAT_ID,
                    "title": "I am a fake model.",
                    "user_id": None,
                }
            ],
        },
    )


def test_get_chat(test_client):
    """Test the /api/chat/{chat_id} endpoint."""
    client, app = test_client

    # Send request
    response = client.get(f"/api/chat/{TEST_CHAT_ID}")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    helper.dict_subset(
        response_data,
        {
            "success": True,
            "data": {
                "chat": {
                    "id": TEST_CHAT_ID,
                    "title": "I am a fake model.",
                    "user_id": None,
                },
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!",
                        "chatId": TEST_CHAT_ID,
                        "id": 1,
                        "files": [],
                    },
                    {
                        "role": "assistant",
                        "content": "I am a fake model.",
                        "chatId": TEST_CHAT_ID,
                        "id": 2,
                    },
                ],
            },
        },
    )


def test_delete_chat(test_client):
    """Test the /api/chat/{chat_id} DELETE endpoint."""
    client, app = test_client

    chat_id = uuid.uuid4()
    # create a chat
    client.post("/api/chat", data={"message": "Hello, world!", "chatId": chat_id})

    # Send request
    response = client.delete(f"/api/chat/{chat_id}")

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse JSON response
    response_data = response.json()

    # Validate response structure
    assert isinstance(response_data, dict)
    assert response_data["success"] is True


def test_abort_chat(test_client):
    """Test the /api/chat/{chat_id}/abort endpoint."""
    client, app = test_client
    # fake model sleep few seconds
    app.dive_host["default"]._model.sleep = 3  # type: ignore

    # abort a non-existent chat
    response = client.post("/api/chat/00000000-0000-0000-0000-000000000000/abort")
    assert response.status_code == BAD_REQUEST_CODE
    body = response.json()
    assert "Chat not found" in body["message"]  # type: ignore

    fake_id = uuid.uuid4()

    def create_chat():
        response = client.post(
            "/api/chat",
            data={"message": "long long time", "chatId": fake_id},
        )
        line = next(response.iter_lines())
        message = json.loads(line[5:])["message"]  # type: ignore
        assert message["content"]["id"] == fake_id

    with ThreadPoolExecutor(1) as executor:
        executor.submit(create_chat)
        time.sleep(2)

        abort_response = client.post(f"/api/chat/{fake_id}/abort")
        assert abort_response.status_code == SUCCESS_CODE
        abort_message = abort_response.json()
        assert abort_message["success"]  # type: ignore


def test_create_chat(test_client):
    """Test the /api/chat POST endpoint."""
    client, app = test_client

    chat_id = str(uuid.uuid4())

    test_file = io.BytesIO(b"test file content")
    response = client.post(
        "/api/chat",
        data={
            "chatId": chat_id,
            "message": "test message",
            "filepaths": ["test_path.txt"],
        },
        files={"files": ("test.txt", test_file, "text/plain")},
    )

    assert response.status_code == SUCCESS_CODE
    assert "text/event-stream" in response.headers.get("Content-Type")

    has_chat_info = False
    has_message_info = False

    # extract and parse the JSON data
    for json_obj in helper.extract_stream(response.text):
        assert "message" in json_obj
        if json_obj["message"]:
            inner_json = json.loads(json_obj["message"])
            assert "type" in inner_json
            assert "content" in inner_json

            # assert the specific type of message
            if inner_json["type"] == "chat_info":
                has_chat_info = True
                helper.dict_subset(
                    inner_json["content"],
                    {
                        "id": chat_id,
                    },
                )
                assert "title" in inner_json["content"]
            if inner_json["type"] == "message_info":
                has_message_info = True
                assert "userMessageId" in inner_json["content"]
                assert "assistantMessageId" in inner_json["content"]

    assert has_chat_info
    assert has_message_info


def test_edit_chat(test_client):
    """Test the /api/chat/edit endpoint."""
    client, app = test_client
    test_file = io.BytesIO(b"test file content")
    test_chat_id = "test_edit_chat"
    response = client.post(
        "/api/chat",
        data={
            "chatId": test_chat_id,
            "message": "test message",
        },
        files={"files": ("test.txt", test_file, "text/plain")},
    )
    assert response.status_code == SUCCESS_CODE
    assert response.headers.get("Content-Type").startswith("text/event-stream")

    user_message_id = ""
    ai_message_id = ""
    fist_ai_reply = ""
    for json_obj in helper.extract_stream(response.text):
        message = json.loads(json_obj["message"])  # type: ignore
        content = message["content"]
        match message["type"]:
            case "chat_info":
                assert content["id"] == test_chat_id  # type: ignore
            case "message_info":
                user_message_id = content["userMessageId"]  # type: ignore
                ai_message_id = content["assistantMessageId"]  # type: ignore
                assert user_message_id
                assert ai_message_id
            case "text":
                fist_ai_reply = content

    ai_messages = [
        AIMessage(content="message 1"),
        AIMessage(content="message 2"),
    ]
    host = cast("dict[str, DiveMcpHost]", app.dive_host)["default"]
    host.model.responses = ai_messages  # type: ignore
    response = client.post(
        "/api/chat/edit",
        data={
            "chatId": test_chat_id,
            "messageId": user_message_id,
            "content": "edited message",
        },
        files={"files": ("test_edit.txt", test_file, "text/plain")},
    )

    assert response.status_code == SUCCESS_CODE
    assert response.headers.get("Content-Type").startswith("text/event-stream")

    new_user_message_id = ""
    new_ai_message_id = ""
    for json_obj in helper.extract_stream(response.text):
        message = json.loads(json_obj["message"])  # type: ignore
        content = message["content"]
        match message["type"]:
            case "chat_info":
                assert content["id"] == test_chat_id  # type: ignore
            case "message_info":
                new_user_message_id = content["userMessageId"]  # type: ignore
                new_ai_message_id = content["assistantMessageId"]  # type: ignore
                assert new_user_message_id
                assert new_ai_message_id
                assert new_ai_message_id != ai_message_id
                assert new_user_message_id == user_message_id
            case "text":
                assert fist_ai_reply != content
    assert new_ai_message_id
    assert new_user_message_id
    response = client.get(f"/api/chat/{test_chat_id}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "data": {
                "messages": [
                    {"messageId": new_user_message_id},
                    {"messageId": new_ai_message_id},
                ]
            },
        },
    )


def test_edit_chat_missing_params(test_client):
    """Test the /api/chat/edit endpoint with missing required parameters."""
    client, app = test_client
    response = client.post(
        "/api/chat/edit",
        data={
            "content": "edited message",
        },
    )
    assert response.status_code == BAD_REQUEST_CODE
    body = response.json()
    assert "Chat ID and Message ID are required" in body["message"]  # type: ignore


def test_retry_chat(test_client):  # noqa: C901, PLR0915
    """Test the /api/chat/retry endpoint."""
    client, app = test_client

    # get message id
    response = client.get(f"/api/chat/{TEST_CHAT_ID}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    message_id = response_data["data"]["messages"][0]["messageId"]  # type: ignore

    host = cast("dict[str, DiveMcpHost]", app.dive_host)["default"]
    model = cast(FakeMessageToolModel, host.model)
    model.responses = [
        AIMessage(content="retry response"),
    ]
    response = client.post(
        "/api/chat/retry",
        json={
            "chatId": TEST_CHAT_ID,
            "messageId": message_id,
        },
    )

    assert response.status_code == SUCCESS_CODE
    assert "text/event-stream" in response.headers.get("Content-Type")

    has_chat_info = False
    has_message_info = False

    # extract and parse the JSON data
    for json_obj in helper.extract_stream(response.text):
        assert "message" in json_obj
        if json_obj["message"]:
            inner_json = json.loads(json_obj["message"])
            assert "type" in inner_json
            assert "content" in inner_json

            # assert the specific type of message
            if inner_json["type"] == "chat_info":
                has_chat_info = True
                helper.dict_subset(
                    inner_json["content"],
                    {
                        "id": TEST_CHAT_ID,
                    },
                )
                assert "title" in inner_json["content"]
            if inner_json["type"] == "message_info":
                has_message_info = True
                assert "userMessageId" in inner_json["content"]
                assert "assistantMessageId" in inner_json["content"]
            if inner_json["type"] == "text":
                assert inner_json["content"] == "retry response"

    assert has_chat_info
    assert has_message_info

    # call retry api with the ai message
    response = client.get(f"/api/chat/{TEST_CHAT_ID}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert len(response_data["data"]["messages"]) == 2  # type: ignore
    message_id = response_data["data"]["messages"][1]["messageId"]  # type: ignore

    model.responses = [
        AIMessage(content="retry response 2"),
    ]

    response = client.post(
        "/api/chat/retry",
        json={
            "chatId": TEST_CHAT_ID,
            "messageId": message_id,
        },
    )

    assert response.status_code == SUCCESS_CODE
    assert "text/event-stream" in response.headers.get("Content-Type")

    has_chat_info = False
    has_message_info = False

    # extract and parse the JSON data
    for json_obj in helper.extract_stream(response.text):
        assert "message" in json_obj
        if json_obj["message"]:
            inner_json = json.loads(json_obj["message"])
            assert "type" in inner_json
            assert "content" in inner_json

            # assert the specific type of message
            if inner_json["type"] == "chat_info":
                has_chat_info = True
                helper.dict_subset(
                    inner_json["content"],
                    {
                        "id": TEST_CHAT_ID,
                    },
                )
                assert "title" in inner_json["content"]
            if inner_json["type"] == "message_info":
                has_message_info = True
                assert "userMessageId" in inner_json["content"]
                assert "assistantMessageId" in inner_json["content"]
            if inner_json["type"] == "text":
                assert inner_json["content"] == "retry response 2"

    assert has_chat_info
    assert has_message_info

    response = client.get(f"/api/chat/{TEST_CHAT_ID}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert len(response_data["data"]["messages"]) == 2  # type: ignore


def test_retry_chat_with_files(test_client):
    """Test the /api/chat/retry endpoint with files."""
    client, app = test_client
    test_file = io.BytesIO(b"test file content")
    test_chat_id = "test_retry_with_files"
    response = client.post(
        "/api/chat",
        data={
            "chatId": test_chat_id,
            "message": "message with files",
        },
        files={"files": ("test.txt", test_file, "text/plain")},
    )
    assert response.status_code == SUCCESS_CODE
    assert response.headers.get("Content-Type").startswith("text/event-stream")

    user_message_id = ""
    ai_message_id = ""
    for json_obj in helper.extract_stream(response.text):
        message = json.loads(json_obj["message"])  # type: ignore
        content = message["content"]
        match message["type"]:
            case "chat_info":
                assert content["id"] == test_chat_id  # type: ignore
            case "message_info":
                user_message_id = content["userMessageId"]  # type: ignore
                ai_message_id = content["assistantMessageId"]  # type: ignore
                assert user_message_id
                assert ai_message_id

    host = cast("dict[str, DiveMcpHost]", app.dive_host)["default"]
    model = cast(FakeMessageToolModel, host.model)
    model.responses = [
        AIMessage(content="retry response"),
    ]
    response = client.post(
        "/api/chat/retry",
        json={
            "chatId": test_chat_id,
            "messageId": ai_message_id,
        },
    )

    assert response.status_code == SUCCESS_CODE
    assert "text/event-stream" in response.headers.get("Content-Type")

    has_chat_info = False
    has_message_info = False

    # extract and parse the JSON data
    for json_obj in helper.extract_stream(response.text):
        assert "message" in json_obj
        if json_obj["message"]:
            inner_json = json.loads(json_obj["message"])
            assert "type" in inner_json
            assert "content" in inner_json

            # assert the specific type of message
            if inner_json["type"] == "chat_info":
                has_chat_info = True
                helper.dict_subset(
                    inner_json["content"],
                    {
                        "id": test_chat_id,
                    },
                )
                assert "title" in inner_json["content"]
            if inner_json["type"] == "message_info":
                has_message_info = True
                assert "userMessageId" in inner_json["content"]
                assert "assistantMessageId" in inner_json["content"]
            if inner_json["type"] == "text":
                assert inner_json["content"] == "retry response"

    assert has_chat_info
    assert has_message_info

    response = client.get(f"/api/chat/{test_chat_id}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert len(response_data["data"]["messages"]) == 2  # type: ignore
    assert response_data["data"]["messages"][0]["messageId"] == user_message_id  # type: ignore


def test_retry_chat_missing_params(test_client):
    """Test the /api/chat/retry endpoint with missing required parameters."""
    client, app = test_client
    response = client.post("/api/chat/retry", data={})
    assert response.status_code == BAD_REQUEST_CODE

    body = response.json()
    # Verify the exception message
    assert "Chat ID and Message ID are required" in body.get("message")


def test_chat_with_tool_calls(test_client, monkeypatch):  # noqa: C901, PLR0912, PLR0915
    """Test the chat endpoint with tool calls."""
    client, app = test_client

    # Import necessary message types
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.messages.tool import tool_call, tool_call_chunk

    # mock the query method
    def mock_query(*args, **kwargs):
        async def title_generator():
            yield {"agent": {"messages": [AIMessage(content="Calculate 2+2")]}}

        if kwargs.get("stream_mode") == "updates":
            return title_generator()

        # mock the response generator
        async def response_generator():
            # mock the tool call
            yield (
                "messages",
                (
                    AIMessageChunk(
                        id="tool-call-msg-id",
                        content="",
                        response_metadata={},
                        tool_call_chunks=[
                            tool_call_chunk(
                                name="calculator",
                                args="",
                                index=0,
                                id="tool-call-id",
                            )
                        ],
                        tool_calls=[
                            tool_call(
                                name="calculator",
                                args={},
                                id="tool-call-id",
                            )
                        ],
                    ),
                    None,
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(
                        id="tool-call-msg-id",
                        content="",
                        tool_call_chunks=[
                            tool_call_chunk(
                                args=json.dumps({"expression": "2+2"}),
                                index=0,
                            )
                        ],
                        tool_calls=[],
                    ),
                    None,
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(
                        id="tool-call-msg-id",
                        content="",
                        response_metadata={
                            "finish_reason": "tool_calls",
                        },
                        tool_call_chunks=[],
                        tool_calls=[],
                    ),
                    None,
                ),
            )

            # mock the tool result
            yield (
                "messages",
                (
                    ToolMessage(
                        content=json.dumps(4),
                        name="calculator",
                        tool_call_id="tool-call-id",
                    ),
                    None,
                ),
            )

            # mock the final ai response
            yield (
                "messages",
                (
                    AIMessage(
                        content="The result of 2+2 is 4.",
                    ),
                    None,
                ),
            )

            # mock the values response
            user_message = HumanMessage(content="Calculate 2+2", id="user-msg-id")
            ai_message = AIMessage(
                content="The result of 2+2 is 4.",
                id="assistant-msg-id",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 15,
                    "total_tokens": 25,
                },
            )

            current_messages = [
                user_message,
                AIMessage(
                    content="",
                    id="tool-call-msg-id",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "2+2"},
                            "id": "tool-call-id",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=json.dumps(4),
                    name="calculator",
                    id="tool-result-msg-id",
                    tool_call_id="tool-call-id",
                ),
                ai_message,
            ]

            yield "values", {"messages": current_messages}

            yield (
                "updates",
                {
                    "chat": {
                        "messages": [
                            AIMessage(
                                content=[
                                    {
                                        "text": "I'll help you calculate that by using the calculator function.",  # noqa: E501
                                        "type": "text",
                                    },
                                    {
                                        "id": "toolu_01AiUPAqBGGDR8RL1uz6SUuE",
                                        "input": {"expression": "2+2"},
                                        "name": "calculator",
                                        "type": "tool_use",
                                    },
                                ],
                                additional_kwargs={},
                                response_metadata={
                                    "id": "msg_01VPZAfL674xURgFGEbUXLhD",
                                    "model": "claude-3-5-haiku-20241022",
                                    "stop_reason": "tool_use",
                                    "stop_sequence": None,
                                    "usage": {
                                        "cache_creation_input_tokens": 0,
                                        "cache_read_input_tokens": 0,
                                        "input_tokens": 345,
                                        "output_tokens": 85,
                                    },
                                    "model_name": "claude-3-5-haiku-20241022",
                                },
                                id="run-c6574da4-41f2-4d64-9a88-77e398113e2f-0",
                                tool_calls=[
                                    {
                                        "name": "calculator",
                                        "args": {"expression": "2+2"},
                                        "id": "toolu_01AiUPAqBGGDR8RL1uz6SUuE",
                                        "type": "tool_call",
                                    }
                                ],
                                usage_metadata={
                                    "input_tokens": 345,
                                    "output_tokens": 85,
                                    "total_tokens": 430,
                                    "input_token_details": {
                                        "cache_read": 0,
                                        "cache_creation": 0,
                                    },
                                },
                            )
                        ]
                    }
                },
            )

            yield (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content="4",
                                name="calculator",
                                id="6a15648d-6498-4f81-b1f6-5c9f71abbd15",
                                tool_call_id="toolu_01AiUPAqBGGDR8RL1uz6SUuE",
                            )
                        ]
                    }
                },
            )

            yield (
                "updates",
                {
                    "chat": {
                        "messages": [
                            AIMessage(
                                content="The result is 4. 2 + 2 = 4.",
                                additional_kwargs={},
                                response_metadata={
                                    "id": "ms g_01K6igyvdVwf8sAoKiHH32fZ",
                                    "model": "claude-3-5-haiku-20241022",
                                    "stop_reason": "end_turn",
                                    "stop_sequence": None,
                                    "usage": {
                                        "cache_creation _input_tokens": 0,
                                        "cache_read_input_tokens": 0,
                                        "input_tokens": 442,
                                        "output_tokens": 21,
                                    },
                                    "model_name": "claude-3-5-haiku-20241022",
                                },
                                id="ru n-415bff33-6772-4e4a-9d25-bdba8e5470fc-0",
                                usage_metadata={
                                    "input_tokens": 442,
                                    "output_tokens": 21,
                                    "total_tokens": 463,
                                    "input_token_details ": {
                                        "cache_read": 0,
                                        "cache_creation": 0,
                                    },
                                },
                            )
                        ]
                    }
                },
            )

        return response_generator()

    # mock the query method
    monkeypatch.setattr("dive_mcp_host.host.chat.Chat.query", mock_query)

    chat_id = str(uuid.uuid4())

    # Make the request
    response = client.post(
        "/api/chat", data={"chatId": chat_id, "message": "Calculate 2+2"}
    )

    # Extract and parse the JSON data

    has_tool_calls = False
    has_tool_result = False
    has_text_response = False
    has_chat_info = False
    has_message_info = False

    for json_obj in helper.extract_stream(response.text):
        assert "message" in json_obj

        if json_obj["message"]:
            inner_json = json.loads(json_obj["message"])
            assert "type" in inner_json
            assert "content" in inner_json

            if inner_json["type"] == "tool_calls":
                has_tool_calls = True
                assert inner_json["content"] == [
                    {"name": "calculator", "arguments": {"expression": "2+2"}}
                ]

            if inner_json["type"] == "tool_result":
                has_tool_result = True
                tool_result = inner_json["content"]
                assert tool_result == {"name": "calculator", "result": 4}

            if inner_json["type"] == "text":
                has_text_response = True
                assert "The result of 2+2 is 4." in inner_json["content"]

            if inner_json["type"] == "chat_info":
                has_chat_info = True
                assert inner_json["content"]["id"] == chat_id
                assert "title" in inner_json["content"]

            if inner_json["type"] == "message_info":
                has_message_info = True
                assert "userMessageId" in inner_json["content"]
                assert "assistantMessageId" in inner_json["content"]

    # Verify all message types were received
    assert has_tool_calls, "Tool calls message not found"
    assert has_tool_result, "Tool result message not found"
    assert has_text_response, "Text response message not found"
    assert has_chat_info, "Chat info message not found"
    assert has_message_info, "Message info message not found"

    # Check that the messages were stored in the database
    with ThreadPoolExecutor() as executor:
        future = executor.submit(lambda: client.get(f"/api/chat/{chat_id}"))
        response = future.result()

    # Verify response status code
    assert response.status_code == SUCCESS_CODE

    # Parse the response
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["success"] is True
    assert response_data["data"] is not None

    # Check for tool call and tool result messages in history
    has_tool_call_msg = False
    has_tool_result_msg = False
    has_assistant_msg = False

    assert isinstance(response_data["data"], dict)
    assert response_data["data"]["messages"] is not None
    for msg in response_data["data"]["messages"]:
        assert isinstance(msg, dict)
        # NOTE: tool_call is included in the assistant message for this test
        if msg["role"] == "tool_call":
            has_tool_call_msg = True
            tool_call_content = json.loads(msg["content"])
            assert tool_call_content == [
                {
                    "name": "calculator",
                    "args": {"expression": "2+2"},
                    "id": "tool-call-id",
                    "type": "tool_call",
                }
            ]
        if msg["role"] == "tool_result":
            has_tool_result_msg = True
            tool_result_content = json.loads(msg["content"])
            assert tool_result_content == 4

        if msg["role"] == "assistant":
            if "The result of 2+2 is 4." in msg["content"]:
                has_assistant_msg = True
            if tool_call_content := msg.get("toolCalls"):
                # NOTE: tool_call might be included in the assistant message
                has_tool_call_msg = True
                assert tool_call_content == [
                    {
                        "name": "calculator",
                        "args": {"expression": "2+2"},
                        "id": "tool-call-id",
                        "type": "tool_call",
                    }
                ]

    assert has_tool_call_msg, "Tool call message not found in database"
    assert has_tool_result_msg, "Tool result message not found in database"
    assert has_assistant_msg, "Assistant message not found in database"


def test_chat_error(test_client, monkeypatch):
    """Test the chat endpoint with an error."""
    client, app = test_client

    def mock_process_chat(*args, **kwargs):
        raise RuntimeError("an test error")

    monkeypatch.setattr(
        "dive_mcp_host.httpd.routers.utils.ChatProcessor._process_chat",
        mock_process_chat,
    )
    response = client.post(
        "/api/chat", data={"chatId": "test_chat_id", "message": "Calculate 2+2"}
    )
    assert response.status_code == SUCCESS_CODE

    has_chat_info = False
    has_error = False

    for json_obj in helper.extract_stream(response.text):
        assert "message" in json_obj
        if json_obj["message"]:
            inner_json = json.loads(json_obj["message"])
            if inner_json["type"] == "chat_info":
                has_chat_info = True
            if inner_json["type"] == "error":
                has_error = True
                assert (
                    inner_json["content"]
                    == "<ThreadQueryError> an test error </ThreadQueryError>"
                )

    assert has_chat_info
    assert has_error
