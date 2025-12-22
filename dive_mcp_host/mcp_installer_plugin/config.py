"""Configuration for the MCP installer plugin.

This module provides configuration settings for the installer agent,
loaded from host.json in DIVE_CONFIG_DIR.

Configuration file: host.json

Example:
{
    "agent": {
        "installer": {
            "dry_run": true
        }
    }
}
"""

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from dive_mcp_host.env import DIVE_CONFIG_DIR

logger = logging.getLogger(__name__)

SETTINGS_FILENAME = "host.json"


class InstallerSettings(BaseModel):
    """Settings for the installer agent."""

    dry_run: bool = False
    """If True, the agent will simulate operations without executing them."""


class AgentSettings(BaseModel):
    """Settings for agents."""

    installer: InstallerSettings = Field(default_factory=InstallerSettings)
    """Settings for the installer agent."""


class PluginSettings(BaseModel):
    """Plugin settings from host.json."""

    agent: AgentSettings = Field(default_factory=AgentSettings)
    """Agent-specific settings."""


def get_default_settings_path() -> Path:
    """Get the default path for host.json."""
    return DIVE_CONFIG_DIR / SETTINGS_FILENAME


def load_settings(config_path: Path | str | None = None) -> PluginSettings:
    """Load settings from file.

    Args:
        config_path: Path to the config file. If None, uses default location.

    Returns:
        PluginSettings with loaded settings, or default settings if file not found.
    """
    if config_path is None:
        config_path = get_default_settings_path()
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.debug("Settings file not found at %s, using defaults", config_path)
        return PluginSettings()

    try:
        content = config_path.read_text(encoding="utf-8")
        settings = PluginSettings.model_validate_json(content)
        logger.debug("Loaded settings from %s", config_path)
        return settings
    except (OSError, ValueError) as e:
        logger.warning("Failed to load settings from %s: %s", config_path, e)
        return PluginSettings()


class InstallerConfigManager:
    """Manager for installer plugin settings.

    Provides a singleton-like interface for accessing settings.
    """

    _instance: "InstallerConfigManager | None" = None
    _settings: PluginSettings | None = None

    def __init__(self, config_path: Path | str | None = None) -> None:
        """Initialize the manager.

        Args:
            config_path: Path to the config file. If None, uses default location.
        """
        self._config_path = config_path
        self._settings = None

    def load(self) -> PluginSettings:
        """Load or reload the settings.

        Returns:
            The loaded PluginSettings.
        """
        self._settings = load_settings(self._config_path)
        return self._settings

    @property
    def settings(self) -> PluginSettings:
        """Get the current settings, loading if necessary.

        Returns:
            The current PluginSettings.
        """
        if self._settings is None:
            self.load()
        return self._settings  # type: ignore[return-value]

    def get_installer_settings(self) -> InstallerSettings:
        """Get settings for the installer agent.

        Returns:
            InstallerSettings for the installer agent.
        """
        return self.settings.agent.installer

    @classmethod
    def get_instance(
        cls, config_path: Path | str | None = None
    ) -> "InstallerConfigManager":
        """Get or create the singleton instance.

        Args:
            config_path: Path to the config file (only used on first call).

        Returns:
            The InstallerConfigManager instance.
        """
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
