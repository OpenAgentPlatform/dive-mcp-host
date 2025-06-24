from functools import cache
from typing import Any

from langchain_core.language_models import BaseChatModel


@cache
def _model_keys(model: type[BaseChatModel]) -> set[str]:
    """Get the keys of the model."""
    mk = set()
    for k, v in model.model_fields.items():
        if v.alias:
            mk.add(v.alias)
        mk.add(k)
    return mk


@cache
def _resolve_model_provider_keys(model_provider: str) -> set[str] | None:
    """Resolve the chat model class for the given model provider."""
    match model_provider:
        case "openai":
            from langchain_openai import ChatOpenAI as M
        case "anthropic":
            from langchain_anthropic import ChatAnthropic as M
        case "azure_openai":
            from langchain_openai import AzureChatOpenAI as M
        case "google_genai":
            from langchain_google_genai import ChatGoogleGenerativeAI as M
        case "ollama":
            from langchain_ollama import ChatOllama as M
        case "mistralai":
            from langchain_mistralai import ChatMistralAI as M
        case "bedrock":
            from langchain_aws import ChatBedrock as M
        case "bedrock_converse":
            from langchain_aws import ChatBedrockConverse as M
        case "deepseek":
            from langchain_deepseek import ChatDeepSeek as M
        case _:
            return None
    return _model_keys(M)


def clean_model_kwargs(
    provider: str | type[BaseChatModel], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Remove the kwargs that are not supported by the given provider."""
    if isinstance(provider, BaseChatModel):
        return {k: v for k, v in kwargs.items() if k in _model_keys(provider)}
    if keys := _resolve_model_provider_keys(provider):
        return {k: v for k, v in kwargs.items() if k in keys}
    return kwargs
