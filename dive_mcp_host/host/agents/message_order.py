from logging import getLogger
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    RemoveMessage,
    ToolMessage,
)

from dive_mcp_host.log import TRACE

logger = getLogger(__name__)


FAKE_TOOL_RESPONSE = "FAKE_TOOL_RESPONSE"


def _has_tool_call(msg: AnyMessage | None) -> bool:
    return msg is not None and isinstance(msg, AIMessage) and len(msg.tool_calls) > 0


def _not_tool_result(msg: AnyMessage) -> bool:
    return not isinstance(msg, ToolMessage)


def tool_call_order(messages: list[AnyMessage]) -> list[BaseMessage]:
    """Guarantee tool call tool result pair.

    Providers like Anthropic requires each tool call to have their
    corresponding tool result.
    """
    logger.log(TRACE, "Examine tool call order. msgs: %s", messages)

    new_msgs: list[BaseMessage] = []
    remove_msgs: list[RemoveMessage] = []
    found_error: bool = False
    prev_msg: BaseMessage | None = None

    for index, msg in enumerate(messages):
        if _has_tool_call(prev_msg) and _not_tool_result(msg):
            assert isinstance(prev_msg, AIMessage), "Could only be AIMessage"
            logger.warning(
                "Found tool call that doesn't have tool result as next message: %s",
                prev_msg.model_dump_json(),
            )

            # Add tool results for each tool call
            for tool_call in prev_msg.tool_calls:
                new_msgs.append(
                    ToolMessage(
                        content="Previous tool call was not processed",
                        tool_call_id=tool_call["id"],
                        response_metadata={FAKE_TOOL_RESPONSE: True},
                        id=uuid4().hex,
                    ),
                )

            # Because we will rearrange all messages after this tool_call,
            # we will remove the original messages.
            if not found_error:
                found_error = True
                remove_msgs = [
                    RemoveMessage(id=msg.id) for msg in messages[index:] if msg.id
                ]

        # Add original messages back (with new id)
        if found_error:
            new = type(msg)(**msg.model_dump())
            new.id = uuid4().hex
            new_msgs.append(new)

        prev_msg = msg

    result = remove_msgs + new_msgs
    logger.log(TRACE, "Tool call order result: %s", result)
    logger.debug(
        "tool call order result, fake tool result needed: %s"
        ", new_msgs: %s, remove_msgs: %s",
        found_error,
        len(new_msgs),
        len(remove_msgs),
    )
    return result
