from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic.alias_generators import to_camel, to_snake

SpecialProvider = Literal["dive", "__load__"]
"""
special providers:
- dive: use the model in dive_mcp_host.models
- __load__: load the model from the configuration
"""


def to_snake_dict(d: dict[str, str]) -> dict[str, str]:
    """Convert a dictionary to snake case."""
    return {to_snake(k): v for k, v in d.items()}


pydantic_model_config = ConfigDict(
    alias_generator=to_camel,
    validate_by_name=True,
    validate_assignment=True,
    validate_by_alias=True,
)


class Credentials(BaseModel):
    """Credentials for the LLM model."""

    access_key_id: SecretStr = Field(default_factory=lambda: SecretStr(""))
    secret_access_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    session_token: SecretStr = Field(default_factory=lambda: SecretStr(""))
    credentials_profile_name: str = ""

    model_config = pydantic_model_config


class BaseLLMConfig(BaseModel):
    """Base configuration for the LLM model."""

    model: str = "gpt-4o"
    model_provider: str | SpecialProvider = Field(default="openai")
    streaming: bool | None = True
    max_tokens: int | None = Field(default=None)

    model_config = pydantic_model_config


class LLMConfiguration(BaseModel):
    """Configuration for the LLM model."""

    base_url: str | None = Field(default=None, alias="baseURL")
    temperature: float | None = Field(default=0)
    top_p: float | None = Field(default=None)

    model_config = pydantic_model_config

    def to_load_model_kwargs(self) -> dict:
        """Convert the LLM config to kwargs for load_model."""
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.temperature:
            kwargs["temperature"] = self.temperature
        if self.top_p:
            kwargs["top_p"] = self.top_p
        return kwargs


class LLMConfig(BaseLLMConfig):
    """Configuration for general LLM models."""

    api_key: SecretStr | None = Field(default=None)
    configuration: LLMConfiguration | None = Field(default=None)

    model_config = pydantic_model_config

    def to_load_model_kwargs(self: LLMConfig) -> dict:
        """Convert the LLM config to kwargs for load_model."""
        exclude = {
            "configuration",
            "model_provider",
            "model",
            "streaming",
        }
        if self.model_provider == "anthropic" and self.max_tokens is None:
            exclude.add("max_tokens")
        kwargs = self.model_dump(
            exclude=exclude,
            exclude_none=True,
        )
        if self.configuration:
            kwargs.update(self.configuration.to_load_model_kwargs())
        remove_keys = []
        if self.model_provider == "openai" and self.model == "o3-mini":
            remove_keys.extend(["temperature", "top_p"])
        for key in remove_keys:
            kwargs.pop(key, None)
        return to_snake_dict(kwargs)


class LLMBedrockConfig(BaseLLMConfig):
    """Configuration for Bedrock LLM models."""

    model_provider: Literal["bedrock"] = "bedrock"
    region: str = "us-east-1"
    credentials: Credentials

    model_config = pydantic_model_config

    def to_load_model_kwargs(self) -> dict:
        """Convert the LLM config to kwargs for load_model."""
        model_kwargs = {}
        model_kwargs["aws_access_key_id"] = self.credentials.access_key_id
        model_kwargs["aws_secret_access_key"] = self.credentials.secret_access_key
        model_kwargs["credentials_profile_name"] = (
            self.credentials.credentials_profile_name
        )
        model_kwargs["aws_session_token"] = self.credentials.session_token
        model_kwargs["region_name"] = self.region
        model_kwargs["streaming"] = True if self.streaming is None else self.streaming
        return model_kwargs


type LLMConfigTypes = Annotated[
    LLMBedrockConfig | LLMConfig, Field(union_mode="left_to_right")
]


model_provider_map: dict[str, type[LLMConfigTypes]] = {
    "bedrock": LLMBedrockConfig,
}


def get_llm_config_type(model_provider: str) -> type[LLMConfigTypes]:
    """Get the model config for the given model provider."""
    return model_provider_map.get(model_provider, LLMConfig)
