"""Additional models for the MCP."""

import logging
from importlib import import_module
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("dive_mcp_host.models")


def load_model(
    provider: str,
    model_name: str,
    *args: Any,
    **kwargs: Any,
) -> BaseChatModel:
    """Load a model from the models directory.

    Args:
        provider: provider name. Two special providers are supported:
            - "dive": use the model in dive_mcp_host.models
            - "__load__": load the model from the configuration
        model_name: The name of the model to load.
        args: Additional arguments to pass to the model.
        kwargs: Additional keyword arguments to pass to the model.

    Returns:
        The loaded model.

    If the provider is "dive", it should be like this:
        import dive_mcp_host.models.model_name_in_lower_case as model_module
        model = model_module.load_model(*args, **kwargs)
    If the provider is "__load__", the model_name is the class name of the model.
    For example, with model_name="package.module:ModelClass", it will be like this:
        import package.module as model_module
        model = model_module.ModelClass(*args, **kwargs)
    If the provider is neither "dive" nor "__load__", it will load model from langchain.
    """
    # XXX Pass configurations/parameters to the model

    logger.debug(
        "Loading model %s with provider %s, kwargs: %s",
        model_name,
        provider,
        kwargs,
    )
    if provider == "dive":
        model_name_lower = model_name.replace("-", "_").replace(".", "_").lower()
        model_module = import_module(
            f"dive_mcp_host.models.{model_name_lower}",
        )
        model = model_module.load_model(*args, **kwargs)
    elif provider == "oap":
        model = init_chat_model(model=model_name, model_provider="openai", **kwargs)
    elif provider == "__load__":
        module_path, class_name = model_name.rsplit(":", 1)
        model_module = import_module(module_path)
        class_ = getattr(model_module, class_name)
        model = class_(*args, **kwargs)
    else:
        if len(args) > 0:
            raise ValueError(
                f"Additional arguments are not supported for {provider} provider.",
            )
        model = init_chat_model(model=model_name, model_provider=provider, **kwargs)
    return model
