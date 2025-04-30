from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool


class TestTool:
    """Test tool."""

    def __init__(self) -> None:
        """Initialize test state."""
        self._called: bool = False

    @property
    def called(self) -> bool:
        """Whether the tool has been called."""
        return self._called

    @property
    def weather_tool(self) -> BaseTool:
        """Weather tool."""

        @tool
        def weather_tool(city: str) -> str:
            """Get current weather information."""
            self._called = True
            return f"The weather in {city} is sunny."

        return weather_tool
