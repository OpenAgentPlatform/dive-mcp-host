import pytest
from langchain_core.messages import AIMessage, HumanMessage

from dive_mcp_host.host.agents.chat_agent import ChatAgentFactory
from dive_mcp_host.models.fake import FakeMessageToolModel


@pytest.fixture
def agent() -> ChatAgentFactory:
    """Return a chat agent."""
    model = FakeMessageToolModel()
    return ChatAgentFactory(
        model=model,
        tools=[],
    )


@pytest.mark.asyncio
async def test_chat_agent(agent: ChatAgentFactory):
    """Test the chat agent."""
    graph = agent.create_agent(
        prompt="you are a helpful assistant",
    )

    initial_state = agent.create_initial_state(query="Hello, world!")
    config = agent.create_config(user_id="default", thread_id="default")

    end_state = await graph.ainvoke(initial_state, config)
    assert len(end_state["messages"]) == 2

    messages = [
        HumanMessage(content="Hello, world!"),
        AIMessage(content="I am a fake model."),
    ] * 100

    messages.append(HumanMessage(content="last human message"))

    initial_state = agent.create_initial_state(query=messages)
    config = agent.create_config(
        user_id="default",
        thread_id="default",
        max_input_tokens=100,
        oversize_policy="window",
    )

    end_state = await graph.ainvoke(initial_state, config)
    assert len(end_state["messages"]) < 100

    assert end_state["messages"][-2].content == "last human message"
    assert end_state["messages"][-1].content == "I am a fake model."
