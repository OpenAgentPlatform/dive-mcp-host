import json
import logging
import os
from pathlib import Path

from pydantic import ValidationError

from dive_mcp_host.host.conf.llm import LLMConfigTypes, get_llm_config_type
from dive_mcp_host.httpd.conf.misc import DIVE_CONFIG_DIR, write_then_replace
from dive_mcp_host.httpd.routers.models import (
    EmbedConfig,
    ModelFullConfigs,
    ModelSingleConfig,
)

# Logger setup
logger = logging.getLogger(__name__)


class ModelManager:
    """Model Manager."""

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize the ModelManager.

        Args:
            config_path: Optional path to the model configuration file.
                If not provided, it will be set to "modelConfig.json" in current
                working directory.
        """
        self._config_path: str = config_path or str(
            DIVE_CONFIG_DIR / "model_config.json"
        )
        self._current_setting: LLMConfigTypes | None = None
        self._full_config: ModelFullConfigs | None = None

    def initialize(self) -> bool:
        """Initialize the ModelManager."""
        logger.info("Initializing ModelManager from %s", self._config_path)
        if env_config := os.environ.get("DIVE_MODEL_CONFIG_CONTENT"):
            config_content = env_config
        elif Path(self._config_path).exists():
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()
        else:
            logger.warning("Model configuration not found")
            return False

        config_dict = json.loads(config_content)
        if not config_dict:
            logger.error("Model configuration not found")
            return False
        try:
            self._full_config = ModelFullConfigs.model_validate(config_dict)
            if model_config := (
                self._full_config.configs.get(self._full_config.active_provider)
            ):
                self._current_setting = get_llm_config_type(
                    model_config.model_provider
                ).model_validate(model_config.model_dump())
            else:
                self._current_setting = None
        except ValidationError as e:
            logger.error("Error parsing model settings: %s", e)
            return False

        return True

    @property
    def current_setting(self) -> LLMConfigTypes | None:
        """Get the active model settings.

        Returns:
            Model settings or None if configuration or active provider is not found.
        """
        return self._current_setting

    @property
    def full_config(self) -> ModelFullConfigs | None:
        """Get the full model configuration.

        Returns:
            Model configuration or None if configuration is not found.
        """
        return self._full_config

    @property
    def config_path(self) -> str:
        """Get the configuration path."""
        return self._config_path

    def get_settings_by_provider(self, provider: str) -> ModelSingleConfig | None:
        """Get the model settings by provider.

        Args:
            provider: Model provider name.
        """
        if not self._full_config:
            return None
        return self._full_config.configs.get(provider, None)

    def save_single_settings(
        self,
        provider: str,
        upload_model_settings: ModelSingleConfig,
        enable_tools: bool = True,
    ) -> None:
        """Save single model configuration.

        Args:
            provider: Model provider name.
            upload_model_settings: Model settings to upload.
            enable_tools: Whether to enable tools.
        """
        if not self._full_config:
            self._full_config = ModelFullConfigs(
                active_provider=provider,
                enable_tools=enable_tools,
                configs={provider: upload_model_settings},
            )
        else:
            self._full_config.active_provider = provider
            self._full_config.configs[provider] = upload_model_settings
            self._full_config.enable_tools = enable_tools

        write_then_replace(
            Path(self._config_path),
            self._full_config.model_dump_json(by_alias=True, exclude_none=True),
        )

    def save_embed_settings(
        self,
        embed_settings: EmbedConfig,
    ) -> None:
        """Save embedding model configuration.

        Args:
            embed_settings: Embedding model settings to upload.
        """
        if not self._full_config:
            raise ValueError("Model configuration not initialized")
        self._full_config.embed_config = embed_settings

        write_then_replace(
            Path(self._config_path),
            self._full_config.model_dump_json(by_alias=True, exclude_none=True),
        )

    def replace_all_settings(
        self,
        upload_model_settings: ModelFullConfigs,
    ) -> None:
        """Replace all model configurations.

        Args:
            upload_model_settings: Model settings to upload.

        Returns:
            True if successful.
        """
        self._full_config = upload_model_settings
        write_then_replace(
            Path(self._config_path),
            upload_model_settings.model_dump_json(by_alias=True, exclude_none=True),
        )
