from langchain_core.messages import AIMessage

from dive_mcp_host.host.agents.tools_in_prompt import extract_tool_calls


def test_extract_tool_calls_with_json_format():
    """Test case with JSON format tool call."""
    message = AIMessage(
        content="""<tool_call>
  <name>echo</name>
  <arguments>{"message": "helloXXX", "delay_ms": 10}</arguments>
</tool_call>""",
        additional_kwargs={},
        response_metadata={"finish_reason": "stop", "model_name": "qwen/qwen3-30b-a3b"},
        id="run--7d28da9d-493f-4803-b9c9-5d1b9ef752fd",
        usage_metadata={
            "input_tokens": 2143,
            "output_tokens": 217,
            "total_tokens": 2360,
            "input_token_details": {},
            "output_token_details": {},
        },
    )

    result = extract_tool_calls(message)

    # Verify the tool call was extracted correctly
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "echo"
    assert tool_call["args"] == {"message": "helloXXX", "delay_ms": 10}
    assert tool_call["id"] is not None  # Should have a UUID assigned

    # Verify the tool call was removed from content
    assert "<tool_call>" not in result.content


def test_extract_tool_calls_with_xml_format():
    """Test case with XML format tool call."""
    message = AIMessage(
        content="""<tool_call>
  <name>search</name>
  <arguments>{"query": "When was the Eiffel Tower built"}</arguments>
</tool_call>"""
    )

    result = extract_tool_calls(message)

    # Verify the tool call was extracted correctly
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "search"
    assert tool_call["args"] == {"query": "When was the Eiffel Tower built"}
    assert tool_call["id"] is not None

    # Verify the tool call was removed from content
    assert "<tool_call>" not in result.content


def test_extract_tool_calls_with_specific_xml_format():
    """Test case with specific XML format tool call."""
    message = AIMessage(
        content="""<tool_call>
  <name>echo</name>
  <arguments>{"message": "helloXXX", "delay_ms": 10}</arguments>
</tool_call>"""
    )

    result = extract_tool_calls(message)

    # Verify the tool call was extracted correctly
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "echo"
    assert tool_call["args"] == {"message": "helloXXX", "delay_ms": 10}
    assert tool_call["id"] is not None

    # Verify the tool call was removed from content
    assert "<tool_call>" not in result.content


def test_extract_tool_calls_with_mixed_formats():
    """Test case with mixed JSON and XML format tool calls."""
    message = AIMessage(
        content="""Some text before
        <tool_call>
        {"name": "first_tool", "arguments": {"arg1": "value1"}}
        </tool_call>
        Some text in between
        <tool_call>
          <name>second_tool</name>
          <arguments>{"arg2": "value2"}</arguments>
        </tool_call>
        Some text after"""
    )

    result = extract_tool_calls(message)

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["name"] == "first_tool"
    assert result.tool_calls[0]["args"] == {"arg1": "value1"}
    assert result.tool_calls[1]["name"] == "second_tool"
    assert result.tool_calls[1]["args"] == {"arg2": "value2"}

    # Verify tool calls were removed from content
    assert "<tool_call>" not in result.content
    assert "Some text before" in result.content
    assert "Some text in between" in result.content
    assert "Some text after" in result.content


def test_extract_tool_calls_with_invalid_json():
    """Test case with invalid JSON in tool call."""
    message = AIMessage(
        content="""<tool_call>
{"name": "echo" "arguments": {"message": "helloXXX", "delay_ms": 10}}
</tool_call>"""
    )

    result = extract_tool_calls(message)

    # Should not extract any tool calls due to invalid JSON
    assert len(result.tool_calls) == 0
    assert "<tool_call>" in result.content


def test_extract_tool_calls_with_invalid_xml():
    """Test case with invalid XML format tool call."""
    message = AIMessage(
        content="""<tool_call>
          <name>search</name>
          <arguments>{"query": "When was the Eiffel Tower built"</arguments>
        </tool_call>"""
    )

    result = extract_tool_calls(message)

    # Should not extract any tool calls due to invalid XML
    assert len(result.tool_calls) == 0
    assert "<tool_call>" in result.content


def test_extract_tool_calls_with_non_string_content():
    """Test case with non-string content."""
    message = AIMessage(content=["This is a list content"])

    result = extract_tool_calls(message)

    # Should not modify the message
    assert len(result.tool_calls) == 0
    assert result.content == ["This is a list content"]


def test_extract_tool_calls_with_multiline_json():
    """Test case with multi-line JSON in tool call."""
    message = AIMessage(
        content="""<tool_call>
  <name>spec_search</name>
  <arguments>
    {
      "index": "spec_grouping_refrigerator",
      "elasticsearch_query": {
        "query": {
          "bool": {
            "must": [
              { "match": { "region": "tw" } }
            ],
            "must_not": [
              { "match": { "status": "deleted" } }
            ]
          }
        },
        "size": 10,
        "sort": [
          { "specs.technical_specs.power_consumption": "asc" }
        ]
      }
    }
  </arguments>
</tool_call>
"""
    )

    result = extract_tool_calls(message)
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "spec_search"
    assert tool_call["args"] == {
        "index": "spec_grouping_refrigerator",
        "elasticsearch_query": {
            "query": {
                "bool": {
                    "must": [{"match": {"region": "tw"}}],
                    "must_not": [{"match": {"status": "deleted"}}],
                }
            },
            "size": 10,
            "sort": [{"specs.technical_specs.power_consumption": "asc"}],
        },
    }
