import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, ToolCall

from dive_mcp_host.httpd.database.orm_models import Chat as ORMChat
from dive_mcp_host.httpd.database.orm_models import Message as ORMMessage
from dive_mcp_host.httpd.routers.chat import ERROR_MSG_ID, DataResult
from dive_mcp_host.httpd.routers.models import SortBy
from dive_mcp_host.httpd.server import DiveHostAPI

if TYPE_CHECKING:
    from dive_mcp_host.host.host import DiveMcpHost

from dive_mcp_host.httpd.database.models import Chat, ChatMessage, Message
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
            "data": {
                "normal": [
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
                ]
            },
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
            "data": {
                "normal": [
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
                ]
            },
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
            "data": {
                "normal": [
                    {
                        "id": TEST_CHAT_ID,
                        "title": "I am a fake model.",
                        "user_id": None,
                    }
                ]
            },
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


def test_bulk_delete_chat(test_client):
    """Test the /api/chat/bulk-delete endpoint."""
    client, app = test_client

    # Create three chats
    chat_ids = [str(uuid.uuid4()) for _ in range(3)]
    for chat_id in chat_ids:
        response = client.post(
            "/api/chat", data={"message": "Hello, world!", "chatId": chat_id}
        )
        assert response.status_code == SUCCESS_CODE

    # Verify all three chats exist in the list
    response = client.get("/api/chat/list")
    assert response.status_code == SUCCESS_CODE
    listed_ids = {c["id"] for c in response.json()["data"]["normal"]}
    for chat_id in chat_ids:
        assert chat_id in listed_ids

    # Bulk delete the first two chats
    ids_to_delete = chat_ids[:2]
    response = client.post("/api/chat/bulk-delete", json=ids_to_delete)
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert response_data["success"] is True

    # Verify only the third chat (and the conftest chat) remain
    response = client.get("/api/chat/list")
    assert response.status_code == SUCCESS_CODE
    remaining_ids = {c["id"] for c in response.json()["data"]["normal"]}
    for deleted_id in ids_to_delete:
        assert deleted_id not in remaining_ids
    assert chat_ids[2] in remaining_ids


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


def test_edit_chat_none_existing_msg(test_client: tuple[TestClient, DiveHostAPI]):
    """Test if editing none existing message is handeled correctly.

    It sould still work, just turns update into insert
    """
    client, _ = test_client
    test_chat_id = "test_edit_chat"
    test_msg_id = ERROR_MSG_ID
    resp = client.post(
        "/api/chat/edit", data={"chatId": test_chat_id, "messageId": test_msg_id}
    )
    assert resp.status_code == SUCCESS_CODE

    resp = client.get(f"/api/chat/{test_chat_id}")
    assert resp.status_code == SUCCESS_CODE
    resp_data = resp.json()
    helper.dict_subset(
        resp_data,
        {
            "success": True,
            "message": None,
            "data": {
                "chat": {
                    "id": "test_edit_chat",
                    "title": "New Chat",
                    "user_id": None,
                },
                "messages": [
                    {
                        "id": 3,
                        "content": "",
                        "role": "user",
                        "chatId": "test_edit_chat",
                        # "messageId": "none-existing-msg-123123",
                        # user messages no longer store resource_usage
                        # user_token is now stored in AIMessage
                        "resource_usage": None,
                        "files": [],
                        "toolCalls": [],
                    },
                    {
                        "id": 4,
                        "content": "I am a fake model.",
                        "role": "assistant",
                        "chatId": "test_edit_chat",
                        "resource_usage": {
                            "model": "",
                            "total_input_tokens": 0,
                            "total_output_tokens": 0,
                            # total_run_time varies
                        },
                        "files": [],
                        "toolCalls": [],
                    },
                ],
            },
        },
    )

    # Make sure the message id doesn't remain as ERROR_MSG_ID
    # The lenght sould be the same as uuid4
    parsed_resp_data = DataResult[ChatMessage].model_validate(resp_data)
    assert parsed_resp_data.data
    assert parsed_resp_data.data.messages[0]
    assert len(parsed_resp_data.data.messages[0].message_id) == len(str(uuid.uuid4()))


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


def test_retry_chat(test_client):
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


def test_patch_chat(test_client):
    """Test the /api/chat/{chat_id} PATCH endpoint."""
    client, app = test_client

    # Create a test chat first
    test_chat_id = str(uuid.uuid4())
    response = client.post(
        "/api/chat",
        data={"message": "Test message for patch", "chatId": test_chat_id},
    )
    assert response.status_code == SUCCESS_CODE

    # Test updating title
    new_title = "Updated Chat Title"
    response = client.patch(
        f"/api/chat/{test_chat_id}",
        json={"title": new_title},
    )
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert response_data["success"] is True

    # Verify the title was updated
    response = client.get(f"/api/chat/{test_chat_id}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert response_data["data"]["chat"]["title"] == new_title

    # Test updating star status
    response = client.patch(
        f"/api/chat/{test_chat_id}",
        json={"star": True},
    )
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert response_data["success"] is True

    # Verify the star status was updated
    response = client.get("/api/chat/list")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "data": {
                "starred": [
                    {
                        "title": "Updated Chat Title",
                    }
                ],
                "normal": [
                    {
                        "title": "I am a fake model.",
                    },
                ],
            },
        },
    )

    # Test updating both title and star status
    final_title = "Final Updated Title"
    response = client.patch(
        f"/api/chat/{test_chat_id}",
        json={"title": final_title, "star": False},
    )
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert response_data["success"] is True

    # Check that both is applied
    response = client.get("/api/chat/list")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    helper.dict_subset(
        response_data,
        {
            "success": True,
            "message": None,
            "data": {
                "starred": [],
                "normal": [
                    {
                        "title": final_title,
                    },
                    {
                        "title": "I am a fake model.",
                    },
                ],
            },
        },
    )


def test_patch_chat_nonexistent(test_client):
    """Test the /api/chat/{chat_id} PATCH endpoint with non-existent chat."""
    client, app = test_client

    # Try to patch a non-existent chat
    nonexistent_chat_id = str(uuid.uuid4())
    response = client.patch(
        f"/api/chat/{nonexistent_chat_id}",
        json={"title": "New Title"},
    )

    assert response.status_code == BAD_REQUEST_CODE


def test_chat_with_tool_calls(test_client, monkeypatch):
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
                error_content = inner_json["content"]
                assert error_content["message"] == "an test error"
                assert error_content["type"] == "thread-query-error"

    assert has_chat_info
    assert has_error


def test_chat_with_tool_progress(
    test_client: tuple[TestClient, DiveHostAPI], monkeypatch
):
    """I can get the progress message."""
    client, app = test_client

    fake_responses = [
        AIMessage(
            content="msg 1",
            tool_calls=[
                ToolCall(
                    name="echo",
                    args={"message": "Hello, world!", "delay_ms": 100},
                    id="123",
                    type="tool_call",
                ),
            ],
        ),
        AIMessage(
            content="msg 2",
        ),
    ]
    model = cast(FakeMessageToolModel, app.dive_host["default"].model)
    model.responses = fake_responses

    response = client.post(
        "/api/chat", data={"chatId": TEST_CHAT_ID, "message": "Calculate 2+2"}
    )

    has_progress = False
    for json_obj in helper.extract_stream(response.text):
        if json_obj["message"]:
            inner_json = json.loads(json_obj["message"])
            if inner_json["type"] == "tool_call_progress":
                has_progress = True

    assert has_progress


def test_chat_with_openai_error(test_client, monkeypatch):
    """Test the chat endpoint with an OpenAI error."""
    client, app = test_client
    from openai import APIError as OpenAIAPIError

    def mock_process_chat(*args, **kwargs):
        raise OpenAIAPIError(
            message="an test error",
            request=None,
            body={
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            },
        )

    monkeypatch.setattr(
        "dive_mcp_host.models.fake.FakeMessageToolModel._generate",
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
                error_content = inner_json["content"]
                assert error_content["type"] == "insufficient_quota"

    assert has_chat_info
    assert has_error


def test_chat_with_token_usage(test_client):
    """Test that token usage is properly stored and retrieved.

    This test verifies that:
    1. Token usage is stored during chat
    2. Token usage appears in chat response
    3. Token usage can be retrieved via get_chat API
    4. Token usage aggregation works correctly for multiple messages
    """
    from langchain_core.messages import AIMessage

    from dive_mcp_host.models.fake import FakeMessageToolModel

    client, app = test_client
    test_chat_id = str(uuid.uuid4())

    # Create FakeMessageToolModel with responses that include usage_metadata
    # Note: The conftest fixture creates an initial chat which consumes one response,
    # so we need to provide responses for that plus our two test interactions
    fake_responses = [
        AIMessage(content="I am a fake model."),  # For conftest's initial chat
        AIMessage(
            content="The answer is 4.",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 8,
                "total_tokens": 18,
            },
        ),
        AIMessage(
            content="The answer is 6.",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 9,
                "total_tokens": 21,
            },
        ),
    ]
    fake_model = FakeMessageToolModel(responses=fake_responses)

    # Replace the model in the host
    app.dive_host["default"]._model = fake_model

    # First interaction
    response = client.post(
        "/api/chat",
        data={"chatId": test_chat_id, "message": "What is 2+2?"},
    )
    assert response.status_code == SUCCESS_CODE

    # Verify token usage in streaming response
    has_token_usage_in_stream = False
    for json_obj in helper.extract_stream(response.text):
        if json_obj.get("message"):
            inner_json = json.loads(json_obj["message"])
            if inner_json.get("type") == "token_usage":
                token_usage = inner_json.get("content", {})
                if token_usage:
                    has_token_usage_in_stream = True
                    assert token_usage.get("inputTokens") == 10
                    assert token_usage.get("outputTokens") == 8

    assert has_token_usage_in_stream, "Token usage not found in stream response"

    # Second interaction
    response = client.post(
        "/api/chat",
        data={"chatId": test_chat_id, "message": "What is 3+3?"},
    )
    assert response.status_code == SUCCESS_CODE

    # Verify token usage in streaming response (shows latest message usage)
    has_second_usage = False
    for json_obj in helper.extract_stream(response.text):
        if json_obj.get("message"):
            inner_json = json.loads(json_obj["message"])
            if inner_json.get("type") == "token_usage":
                token_usage = inner_json.get("content", {})
                if token_usage:
                    has_second_usage = True
                    # Stream shows the latest message's token usage
                    assert token_usage.get("inputTokens") == 12
                    assert token_usage.get("outputTokens") == 9

    assert has_second_usage, "Second token usage not found"

    # Test get_chat API to verify token usage is persisted and aggregated
    response = client.get(f"/api/chat/{test_chat_id}")
    assert response.status_code == SUCCESS_CODE

    response_data = response.json()
    assert "data" in response_data, "data not in response"

    chat_data = response_data["data"]
    assert "token_usage" in chat_data, "token_usage not in get_chat response"

    token_usage = chat_data["token_usage"]
    assert token_usage is not None, "token_usage is None"
    # Verify aggregated token usage
    assert token_usage["totalInputTokens"] == 22  # 10 + 12
    assert token_usage["totalOutputTokens"] == 17  # 8 + 9

    # Verify messages also have resource_usage
    messages = chat_data.get("messages", [])
    assert len(messages) > 0, "No messages found"

    # Check AI messages have resource_usage
    ai_messages = [msg for msg in messages if msg["role"] == "assistant"]
    assert len(ai_messages) == 2, f"Expected 2 AI messages, got {len(ai_messages)}"

    # Verify first AI message
    first_ai_msg = ai_messages[0]
    assert "resource_usage" in first_ai_msg
    assert first_ai_msg["resource_usage"]["total_input_tokens"] == 10
    assert first_ai_msg["resource_usage"]["total_output_tokens"] == 8

    # Verify second AI message
    second_ai_msg = ai_messages[1]
    assert "resource_usage" in second_ai_msg
    assert second_ai_msg["resource_usage"]["total_input_tokens"] == 12
    assert second_ai_msg["resource_usage"]["total_output_tokens"] == 9


def test_chat_with_abort_local_tool_calls(test_client):
    """Test the chat endpoint with abort during local tool calls.

    This test verifies that:
    1. Abort works correctly during local tool execution
    2. Elicitation requests are properly handled (auto-accepted)
    3. After abortion, we can still continue chatting
    """
    client, app = test_client
    host = cast("dict[str, DiveMcpHost]", app.dive_host)["default"]
    model = cast(FakeMessageToolModel, host.model)

    # Use bash tool which triggers elicitation for user confirmation
    model.responses = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="bash",
                    args={
                        "command": "sleep 100",
                    },
                    id="tool-call-123",
                    type="tool_call",
                ),
            ],
        ),
        AIMessage(content="hihi"),
    ]

    def handle_elicitation_accept_and_abort(chat_id: str):
        """Accept elicitation then abort after tool starts executing."""
        elicitation_manager = host.elicitation_manager

        # Wait for elicitation request to appear
        max_wait = 3.0
        start = time.time()
        request_id = None

        while time.time() - start < max_wait:
            pending_ids = list(elicitation_manager._pending_requests.keys())
            if pending_ids:
                request_id = pending_ids[0]
                break
            time.sleep(0.05)

        if request_id:
            # Accept the elicitation request
            resp = client.post(
                "/api/tools/elicitation/respond",
                json={
                    "request_id": request_id,
                    "action": "accept",
                    "content": {},
                },
            )
            assert resp.status_code == SUCCESS_CODE

            # Wait a bit for the tool to start executing
            time.sleep(0.3)

        # Now abort the chat
        abort_response = client.post(f"/api/chat/{chat_id}/abort")
        assert abort_response.status_code == SUCCESS_CODE, abort_response.text
        abort_message = abort_response.json()
        assert abort_message["success"]  # type: ignore

        return request_id is not None

    def req():
        response = client.post(
            "/api/chat",
            data={"message": "run sleep command", "chatId": TEST_CHAT_ID},
        )
        return response.text

    for iteration in range(2):
        with ThreadPoolExecutor(2) as executor:
            req_future = executor.submit(req)
            elicit_future = executor.submit(
                handle_elicitation_accept_and_abort, TEST_CHAT_ID
            )

            # Wait for both to complete
            elicitation_handled = elicit_future.result()
            response_text = req_future.result()

            # Verify elicitation was handled
            assert elicitation_handled, f"Iteration {iteration}: Elicitation not found"

            # Verify we got tool_result (with abort message) or the stream completed
            assert (
                "tool_result" in response_text
                or "agent_tool_result" in response_text
                or "interactive" in response_text  # elicitation request was sent
            ), f"Iteration {iteration}: Unexpected response: {response_text[:500]}"

        # After abortion, we can still have chats
        response = client.post(
            "/api/chat", data={"message": "hi", "chatId": TEST_CHAT_ID}
        )
        assert "hihi" in response.text, (
            f"Iteration {iteration}: Expected 'hihi' in response"
        )
        time.sleep(3)
        model.i = 0


def test_chat_with_abort_tool_calls(test_client):
    """Test the chat endpoint with tool calls."""
    client, app = test_client
    host = cast("dict[str, DiveMcpHost]", app.dive_host)["default"]
    model = cast(FakeMessageToolModel, host.model)

    model.responses = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="echo",
                    args={"message": "Hello, world!", "delay_ms": 1000},
                    id="123",
                    type="tool_call",
                ),
            ],
        ),
        AIMessage(content="hihi"),
    ]

    def req():
        response = client.post(
            "/api/chat",
            data={"message": "call", "chatId": TEST_CHAT_ID},
        )
        assert "tool_result" in response.text

    for _ in range(2):
        with ThreadPoolExecutor(1) as executor:
            f = executor.submit(req)
            time.sleep(0.5)

            abort_response = client.post(f"/api/chat/{TEST_CHAT_ID}/abort")
            assert abort_response.status_code == SUCCESS_CODE, abort_response.text
            abort_message = abort_response.json()
            assert abort_message["success"]  # type: ignore

            f.result()
        # after abortion, we can still have chats
        response = client.post(
            "/api/chat", data={"message": "hi", "chatId": TEST_CHAT_ID}
        )
        assert "hihi" in response.text
        time.sleep(5)
        model.i = 0


def test_chat_error_sends_message_info_for_retry(test_client, monkeypatch):
    """Test that chat error still sends message_info for client retry support.

    This test verifies that when _process_chat raises an exception:
    1. message_info is still sent in the stream with user and assistant message IDs
    2. A placeholder assistant message is stored in the database
    3. Retrying an earlier error message deletes subsequent messages
    4. The retry still fails (error persists) but DB state is correct
    """
    client, app = test_client
    test_chat_id = str(uuid.uuid4())

    # Mock _process_chat to raise an error
    async def mock_process_chat(self, *args, **kwargs):
        raise RuntimeError("simulated error for retry test")

    monkeypatch.setattr(
        "dive_mcp_host.httpd.routers.utils.ChatProcessor._process_chat",
        mock_process_chat,
    )

    # First message - will fail
    response = client.post(
        "/api/chat",
        data={"chatId": test_chat_id, "message": "first message"},
    )
    assert response.status_code == SUCCESS_CODE

    # Capture message IDs from first failed chat
    user1_msg_id = None
    ai1_msg_id = None
    for json_obj in helper.extract_stream(response.text):
        if json_obj.get("message"):
            inner_json = json.loads(json_obj["message"])
            if inner_json["type"] == "message_info":
                user1_msg_id = inner_json["content"]["userMessageId"]
                ai1_msg_id = inner_json["content"]["assistantMessageId"]
            if inner_json["type"] == "error":
                err_msg = inner_json["content"]["message"]
                assert "simulated error for retry test" in err_msg

    assert user1_msg_id is not None
    assert ai1_msg_id is not None

    # Second message - will also fail
    response = client.post(
        "/api/chat",
        data={"chatId": test_chat_id, "message": "second message"},
    )
    assert response.status_code == SUCCESS_CODE

    # Capture message IDs from second failed chat
    user2_msg_id = None
    ai2_msg_id = None
    for json_obj in helper.extract_stream(response.text):
        if json_obj.get("message"):
            inner_json = json.loads(json_obj["message"])
            if inner_json["type"] == "message_info":
                user2_msg_id = inner_json["content"]["userMessageId"]
                ai2_msg_id = inner_json["content"]["assistantMessageId"]

    assert user2_msg_id is not None
    assert ai2_msg_id is not None

    # Verify DB has 4 messages: user1, ai1(error), user2, ai2(error)
    response = client.get(f"/api/chat/{test_chat_id}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    messages = response_data["data"]["messages"]
    assert len(messages) == 4, f"Expected 4 messages, got {len(messages)}"

    # Verify both AI messages have error content
    ai_messages = [m for m in messages if m["role"] == "assistant"]
    assert len(ai_messages) == 2
    for ai_msg in ai_messages:
        assert "<chat-error>" in ai_msg["content"]

    # Now retry ai1 - should still fail but user2 and ai2 should be deleted
    response = client.post(
        "/api/chat/retry",
        json={
            "chatId": test_chat_id,
            "messageId": ai1_msg_id,
        },
    )
    assert response.status_code == SUCCESS_CODE

    # Verify retry still returns error
    has_error = False
    has_message_info = False
    for json_obj in helper.extract_stream(response.text):
        if json_obj.get("message"):
            inner_json = json.loads(json_obj["message"])
            if inner_json["type"] == "error":
                has_error = True
                err_msg = inner_json["content"]["message"]
                assert "simulated error for retry test" in err_msg
            if inner_json["type"] == "message_info":
                has_message_info = True

    assert has_error, "Retry should still fail"
    assert has_message_info, "message_info should be sent on retry error"

    # Verify DB now only has 2 messages: user1, ai1(new error placeholder)
    # user2 and ai2 should be deleted
    response = client.get(f"/api/chat/{test_chat_id}")
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    messages = response_data["data"]["messages"]

    assert len(messages) == 2, f"Expected 2 messages after retry, got {len(messages)}"

    # Verify user2 and ai2 are gone
    message_ids = [m.get("messageId") for m in messages]
    assert user2_msg_id not in message_ids, "user2 should be deleted"
    assert ai2_msg_id not in message_ids, "ai2 should be deleted"

    # Verify user1 still exists and new ai message has error
    assert messages[0]["messageId"] == user1_msg_id
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "<chat-error>" in messages[1]["content"]


@pytest.mark.asyncio
async def test_search(test_client: tuple[TestClient, DiveHostAPI]):
    """Test the /api/chat/search endpoint filters duplicate chat results."""
    from datetime import timedelta

    client, app = test_client

    # Setup: create 3 chats with 2 messages each directly in the database.
    # Chat 1 and Chat 2 contain "quantum" in their messages.
    # Chat 3 does not contain "quantum".
    # Use explicit timestamps to verify ordering:
    # ORDER BY chats.updated_at DESC, messages.created_at ASC
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

    async with app.db_sessionmaker() as session:
        # Chat 1: older updated_at (should appear second in results)
        chat1_id = str(uuid.uuid4())
        chat1_updated = base_time
        chat1_msg1_id = str(uuid.uuid4())  # oldest message
        chat1_msg2_id = str(uuid.uuid4())
        session.add(
            ORMChat(
                id=chat1_id,
                title="Physics Discussion",
                created_at=base_time,
                updated_at=chat1_updated,
            )
        )
        await session.flush()
        session.add_all(
            [
                ORMMessage(
                    message_id=chat1_msg1_id,
                    chat_id=chat1_id,
                    role="user",
                    content="Tell me about quantum computing",
                    created_at=base_time,
                    files="",
                ),
                ORMMessage(
                    message_id=chat1_msg2_id,
                    chat_id=chat1_id,
                    role="assistant",
                    content="Quantum computing uses qubits for processing",
                    created_at=base_time + timedelta(seconds=1),
                    files="",
                ),
            ]
        )
        await session.flush()

        # Chat 2: newer updated_at (should appear first in results)
        chat2_id = str(uuid.uuid4())
        chat2_updated = base_time + timedelta(hours=1)
        chat2_msg1_id = str(uuid.uuid4())  # oldest message
        chat2_msg2_id = str(uuid.uuid4())
        session.add(
            ORMChat(
                id=chat2_id,
                title="Science Review",
                created_at=base_time + timedelta(minutes=30),
                updated_at=chat2_updated,
            )
        )
        await session.flush()
        session.add_all(
            [
                ORMMessage(
                    message_id=chat2_msg1_id,
                    chat_id=chat2_id,
                    role="user",
                    content="Explain quantum entanglement",
                    created_at=base_time + timedelta(minutes=30),
                    files="",
                ),
                ORMMessage(
                    message_id=chat2_msg2_id,
                    chat_id=chat2_id,
                    role="assistant",
                    content="Quantum entanglement links particles together",
                    created_at=base_time + timedelta(minutes=30, seconds=1),
                    files="",
                ),
            ]
        )
        await session.flush()

        # Chat 3: no quantum content
        chat3_id = str(uuid.uuid4())
        chat3_time = base_time + timedelta(hours=2)
        session.add(
            ORMChat(
                id=chat3_id,
                title="Finance Report",
                created_at=chat3_time,
                updated_at=chat3_time,
            )
        )
        await session.flush()
        session.add_all(
            [
                ORMMessage(
                    message_id=str(uuid.uuid4()),
                    chat_id=chat3_id,
                    role="user",
                    content="What are the stock market trends",
                    created_at=chat3_time,
                    files="",
                ),
                ORMMessage(
                    message_id=str(uuid.uuid4()),
                    chat_id=chat3_id,
                    role="assistant",
                    content="The market shows positive growth patterns",
                    created_at=chat3_time + timedelta(seconds=1),
                    files="",
                ),
            ]
        )
        await session.flush()
        await session.commit()

    # Search for "quantum" — matches all 4 messages across chat1 and chat2,
    # but the API deduplicates by chat_id, so only 2 results should be returned.
    response = client.post("/api/chat/search", json={"query": "quantum"})
    assert response.status_code == SUCCESS_CODE
    response_data = response.json()
    assert response_data["success"] is True
    assert len(response_data["data"]) == 2

    result_chat_ids = {r["chat_id"] for r in response_data["data"]}
    assert chat1_id in result_chat_ids
    assert chat2_id in result_chat_ids
    assert chat3_id not in result_chat_ids

    # Verify results are ordered by chat.updated_at DESC
    results = response_data["data"]
    assert results[0]["chat_id"] == chat2_id, (
        "Most recently updated chat should be first"
    )
    assert results[1]["chat_id"] == chat1_id, "Older updated chat should be second"

    # Verify messages within results are ordered by created_at ASC
    assert results[0]["message_id"] == chat2_msg1_id, (
        "Chat2 should return the oldest matching message"
    )
    assert results[1]["message_id"] == chat1_msg1_id, (
        "Chat1 should return the oldest matching message"
    )
