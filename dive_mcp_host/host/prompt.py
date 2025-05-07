"""Prompt for the host."""

import json
from collections.abc import Callable, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool

from dive_mcp_host.host.helpers import today_datetime

PromptType = SystemMessage | str | Callable[..., list[BaseMessage]]

SYSTEM_PROMPT = """You are an AI assistant helping a software engineer.
Your user is a professional software engineer who works on various programming projects.
Today's date is {today_datetime}. I aim to provide clear, accurate, and helpful
responses with a focus on software development best practices.

I should be direct, technical, and practical in my communication style.
When doing git diff operation, do check the README.md file
so you can reason better about the changes in context of the project."""


def default_system_prompt() -> str:
    """The default system prompt."""
    return SYSTEM_PROMPT.format(today_datetime=today_datetime())


def tools_definition(tools: Sequence[BaseTool]) -> str:
    """The description of the tools."""
    return "\n".join(
        f"""
<tool>
  <name>{tool.name}</name>
  <description>{tool.description}</description>
  <arguments>{json.dumps(tool.args)}</arguments>
</tool>"""
        for tool in tools
    )


def tools_prompt(tools: Sequence[BaseTool]) -> str:
    """Example and description for tools in prompt."""
    return f"""
## Available Tools
---
<tools>
{tools_definition(tools)}
</tools>
---

## Function Tool Use Example
---
User: When was the Eiffel Tower built?

Assistant: I can use the search tool to find information about the Eiffel Tower.
<tool_call>
  <name>search</name>
  <arguments>{{"query": "When was the Eiffel Tower built"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>search</name>
  <result>The Eiffel Tower was built between 1887 and 1889 and was officially opened on March 31, 1889.</result>
</tool_call_result>

Assistant: The Eiffel Tower was built between 1887 and 1889 and was officially opened on March 31, 1889.

---
User: "What is 15% of $85.50 plus a $5 fee?"

Assistant: I can use the calculator tool to find 15% of $85.50.
<tool_call>
  <name>calculator</name>
  <arguments>{{"expression": "85.50 * 0.15"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>calculator</name>
  <result>12.83</result>
</tool_call_result>

Assistant: Now I'll add the $5 fee.
<tool_call>
  <name>calculator</name>
  <arguments>{{"expression": "12.83 + 5"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>calculator</name>
  <result>17.83</result>
</tool_call_result>

Assistant: 15% of $85.50 is $12.83, and when you add the $5 fee, the total is $17.83.

---
User: "What is the population of Tokyo and how does it compare to New York City?"

Assistant: I can use the search tool to find the population of Tokyo.
<tool_call>
  <name>search</name>
  <arguments>{{"query": 123131}}</arguments>
</tool_call>

ToolMessage: <tool_call_failed>
  <name>search</name>
  <error>Input sould be a string</error>
</tool_call_failed>

Assistant: Oh.. I see that my argumnet needs to be a string, let me try again.
<tool_call>
  <name>search</name>
  <arguments>{{"query": "Population of Tokyo"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>search</name>
  <result>Tokyo has a population of approximately 13.96 million people (2023 estimate).</result>
</tool_call_result>

Assistant: I can also use the search tool to find the population of New York City.
<tool_call>
  <name>search</name>
  <arguments>{{"query": "Population of New York City"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>search</name>
  <result>New York City has a population of approximately 8.8 million people (2022 estimate).</result>
</tool_call_result>

Assistant: Tokyo has a population of approximately 13.96 million people, while New York City has approximately 8.8 million people. Tokyo is significantly larger, with about 5.16 million more residents than New York City.
---

## Important Notes
- Use the tools listed in the 'Available Tools' section whenever you can.
- Do not print out code and pretend you have used the tool.
- Actually use the tool by returning with a <tool_call> tag.
"""  # noqa: E501
