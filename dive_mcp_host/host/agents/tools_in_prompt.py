"""Handle tools in prompt.

For models that "doesn't support tool calls" in the sense that it cannot bind tools,
we need to put "tool calls" and "tool result" in the message content before
passing them to the model.

But for our graph to function, we also need to extract tool calls from message content
and trasform them into ToolCall objects inside AIMessage.tool_calls.
"""

import json
import re
import uuid
from logging import getLogger

from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langgraph.utils.runnable import RunnableCallable

logger = getLogger(__name__)


def extract_tool_calls(response: AIMessage) -> AIMessage:
    """Extract the tool calls from the response content."""
    if isinstance(response.content, str):
        # Extract the tool call content - match both JSON and XML formats
        tool_call_content = r"<tool_call>.*?</tool_call>"
        regex = re.compile(tool_call_content, re.DOTALL)
        matches = regex.findall(response.content)

        # Transform the tool call content into a ToolCall
        for match in matches:
            # Try XML format
            name_match = re.search(r"<name>(.*?)</name>", match, re.DOTALL)
            args_match = re.search(r"<arguments>(.*?)</arguments>", match, re.DOTALL)
            if name_match and args_match:
                try:
                    name = name_match.group(1).strip()
                    args = json.loads(args_match.group(1).strip())
                    tool_call_id = str(uuid.uuid4())
                    tool_call = ToolCall(
                        name=name,
                        args=args,
                        id=tool_call_id,
                    )
                    logger.debug("found tool call: %s", tool_call)
                    response.content = response.content.replace(match, "")
                    response.tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse tool call arguments: %s", args_match.group(1)
                    )
                continue
            # Try JSON format
            json_match = re.search(r"\{[^<]*\}", match, re.DOTALL)
            if json_match:
                try:
                    tool_call = json.loads(json_match.group().strip())
                    tool_call_id = str(uuid.uuid4())
                    tool_call = ToolCall(
                        name=tool_call.get("name", ""),
                        args=tool_call.get("arguments", ""),
                        id=tool_call_id,
                    )
                    logger.debug("found tool call: %s", tool_call)
                    response.content = response.content.replace(match, "")
                    response.tool_calls.append(tool_call)
                    continue
                except json.JSONDecodeError:
                    pass

    else:
        logger.debug(
            "Response content is not a string, cannot extract tool calls: %s",
            response.content,
        )

    return response


@RunnableCallable
def convert_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Convert a list of messages for sending to the model."""
    ret = []
    for message in messages:
        if isinstance(message, AIMessage):
            ret.append(convert_ai_message(message))
        elif isinstance(message, ToolMessage):
            ret.append(convert_tool_message(message))
        else:
            ret.append(message)
    return ret


def convert_ai_message(ai_message: AIMessage) -> AIMessage:
    """Convert an AI message for sending to the model."""
    if tool_calls := ai_message.tool_calls:
        content = (
            ai_message.content
            if isinstance(ai_message.content, list)
            else [ai_message.content]
            + [
                f"<tool_call>\n<name>{tool_call['name']}</name>\n<arguments>{json.dumps(tool_call['args'])}</arguments>\n</tool_call>"
                for tool_call in tool_calls
            ]
        )
        return AIMessage(
            content=str(content),  # type: ignore
            id=ai_message.id,
            usage_metadata=ai_message.usage_metadata,
            response_metadata=ai_message.response_metadata,
            additional_kwargs=ai_message.additional_kwargs,
        )
    return ai_message


def convert_tool_message(tool_message: ToolMessage) -> ToolMessage:
    """Convert a tool message for sending to the model."""
    if tool_message.status == "success":
        content = f"""
<tool_call_result>
  <name>{tool_message.name}</name>
  <result>{tool_message.content}</result>
</tool_call_result>
"""
    else:
        content = f"""
<tool_call_failed>
  <name>{tool_message.name}</name>
  <error>{tool_message.content}</error>
</tool_call_failed>
"""

    return ToolMessage(
        content=content,
        name=tool_message.name,
        tool_call_id=tool_message.tool_call_id,
        id=tool_message.id,
        response_metadata=tool_message.response_metadata,
        additional_kwargs=tool_message.additional_kwargs,
    )
