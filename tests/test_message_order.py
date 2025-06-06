from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from dive_mcp_host.host.agents.message_order import FAKE_TOOL_RESPONSE, tool_call_order


def test_msg_order():
    """Test if message order correction is successful."""
    tool_call_id = "toolu_012N5cw28KM9QfRLeYdik5V6"
    messages = [
        HumanMessage(content="Hi, please generate a image of xxx for me.", id="1"),
        AIMessage(
            content="Sure, I will us xxx to generate and image of xxx for you.",
            tool_calls=[
                {
                    "name": "xxx",
                    "args": {
                        "prompt": "A xxx",
                    },
                    "id": "toolu_012N5cw28KM9QfRLeYdik5V6",
                    "type": "tool_call",
                }
            ],
            id="2",
        ),
        HumanMessage(content="Hi, again", id="3"),
    ]
    result = tool_call_order(messages)
    assert len(result) == 3

    # Remove messages after the tool call
    assert isinstance(result[0], RemoveMessage)

    # Insert ToolMessage
    assert isinstance(result[1], ToolMessage)
    assert result[1].tool_call_id == tool_call_id
    assert result[1].id
    assert result[1].response_metadata[FAKE_TOOL_RESPONSE]

    # Other messages behind ToolMessage
    assert isinstance(result[2], HumanMessage)
    assert result[2].content == messages[2].content
    assert result[2].id
