import logging
import os
from enum import StrEnum
from pathlib import Path

from dive_mcp_host.env import DIVE_CONFIG_DIR
from dive_mcp_host.httpd.conf.misc import write_then_replace
from dive_mcp_host.httpd.conf.system_prompt import system_prompt

# Logger setup
logger = logging.getLogger(__name__)


class PromptKey(StrEnum):
    """Prompt key enum."""

    SYSTEM = "system"
    CUSTOM = "custom"


class PromptManager:
    """Prompt Manager for handling system prompts and custom rules."""

    def __init__(self, custom_rules_path: str | None = None) -> None:
        """Initialize the PromptManager.

        The system prompt is set according to the following priority:
        1. Environment variable DIVE_CUSTOM_RULES_CONTENT if present
        2. File specified by custom_rules_path parameter if set
        3. ".customrules" file in current working directory if exists
        4. Default to empty string if no other source is available

        Args:
            custom_rules_path: Optional path to the custom rules file.
        """
        self.prompts: dict[str, str] = {}
        self.custom_rules_path = custom_rules_path or str(
            DIVE_CONFIG_DIR / "custom_rules"
        )

    def initialize(self) -> None:
        """Initialize the PromptManager."""
        logger.info("Initializing PromptManager from %s", self.custom_rules_path)
        if custom_rules := os.environ.get("DIVE_CUSTOM_RULES_CONTENT"):
            self.prompts[PromptKey.SYSTEM] = system_prompt(custom_rules)
            self.prompts[PromptKey.CUSTOM] = custom_rules
        elif (path := Path(self.custom_rules_path)) and path.exists():
            self.prompts[PromptKey.SYSTEM] = system_prompt(path.read_text("utf-8"))
            self.prompts[PromptKey.CUSTOM] = path.read_text("utf-8")
        else:
            self.prompts[PromptKey.SYSTEM] = system_prompt("")
            self.prompts[PromptKey.CUSTOM] = ""

    def set_prompt(self, key: str, prompt: str) -> None:
        """Set a prompt by key.

        Args:
            key: The key to store the prompt under.
            prompt: The prompt text.
        """
        self.prompts[key] = prompt

    def get_prompt(self, key: str) -> str | None:
        """Get a prompt by key.

        Args:
            key: The key of the prompt to retrieve.

        Returns:
            The prompt text or None if not found.
        """
        return self.prompts.get(key)

    def write_custom_rules(self, prompt: str) -> None:
        """Write custom rules to file.

        Args:
            prompt: The prompt text.
        """
        write_then_replace(Path(self.custom_rules_path), prompt)

    def load_custom_rules(self) -> str:
        """Load custom rules from file or environment variable.

        Returns:
            The custom rules text.
        """
        try:
            return os.environ.get("DIVE_CUSTOM_RULES_CONTENT") or Path(
                self.custom_rules_path
            ).read_text(encoding="utf-8")
        except OSError as error:
            logger.warning("Cannot read %s: %s", self.custom_rules_path, error)
            return ""

    def update_prompts(self) -> None:
        """Update the system prompt with current custom rules."""
        custom_rules = self.load_custom_rules()
        self.prompts[PromptKey.SYSTEM] = system_prompt(custom_rules)
        self.prompts[PromptKey.CUSTOM] = custom_rules
