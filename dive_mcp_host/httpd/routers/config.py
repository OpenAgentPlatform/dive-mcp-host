from logging import getLogger

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from dive_mcp_host.httpd.conf.mcp_servers import Config
from dive_mcp_host.httpd.dependencies import get_app
from dive_mcp_host.httpd.server import DiveHostAPI

from .models import (
    EmbedConfig,
    McpServerError,
    McpServers,
    ModelFullConfigs,
    ModelInterfaceDefinition,
    ModelSettingsDefinition,
    ModelSingleConfig,
    ResultResponse,
)

logger = getLogger(__name__)

config = APIRouter(tags=["config"])


class ConfigResult[T](ResultResponse):
    """Generic configuration result that extends ResultResponse with a config field."""

    config: T | None


class SaveConfigResult(ResultResponse):
    """Result of saving configuration, including any errors that occurred."""

    errors: list[McpServerError]


class InterfaceResult(ResultResponse):
    """Result containing model interface definition."""

    interface: ModelInterfaceDefinition


class RulesResult(ResultResponse):
    """Result containing custom rules as a string."""

    rules: str


class SaveModelSettingsRequest(BaseModel):
    """Request model for saving model settings."""

    provider: str
    model_settings: ModelSingleConfig = Field(alias="modelSettings")
    enable_tools: bool = Field(alias="enableTools")


@config.get("/mcpserver")
async def get_mcp_server(
    app: DiveHostAPI = Depends(get_app),
) -> ConfigResult[McpServers]:
    """Get MCP server configurations.

    Returns:
        ConfigResult[McpServers]: Configuration for MCP servers.
    """
    config = await app.mcp_server_config_manager.get_current_config()
    if config is None:
        logger.warning("MCP server configuration not found")
        return ConfigResult(
            success=True,
            config=McpServers(),
        )

    config = McpServers.model_validate(config.model_dump(by_alias=True))
    return ConfigResult(
        success=True,
        config=config,
    )


@config.post("/mcpserver")
async def post_mcp_server(
    servers: McpServers,
    app: DiveHostAPI = Depends(get_app),
    force: bool = False,
) -> SaveConfigResult:
    """Save MCP server configurations.

    Args:
        servers (McpServers): The server configurations to save.
        app (DiveHostAPI): The DiveHostAPI instance.
        force (bool): If True, reload all mcp servers even if they are not changed.

    Returns:
        SaveConfigResult: Result of the save operation with any errors.
    """
    # Update conifg
    new_config = Config.model_validate(servers.model_dump(by_alias=True))
    if not await app.mcp_server_config_manager.update_all_configs(new_config):
        raise ValueError("Failed to update MCP server configurations")

    # Reload host
    host_config = await app.load_host_config()
    await app.dive_host["default"].reload(new_config=host_config, force_mcp=force)

    # Get failed MCP servers
    failed_servers: list[McpServerError] = []
    for server_name, server_info in app.dive_host["default"].mcp_server_info.items():
        if server_info.error is not None:
            failed_servers.append(
                McpServerError(
                    serverName=server_name,
                    error=str(server_info.error),
                )
            )

    return SaveConfigResult(
        success=True,
        errors=failed_servers,
    )


@config.get("/model")
async def get_model(
    app: DiveHostAPI = Depends(get_app),
) -> ConfigResult["ModelFullConfigs"]:
    """Get current model configuration.

    Returns:
        ConfigResult[ModelConfig]: Current model configuration.
    """
    if app.model_config_manager.full_config is None:
        logger.warning("Model configuration not found")
        return ConfigResult(
            success=True,
            config=None,
        )

    return ConfigResult(success=True, config=app.model_config_manager.full_config)


@config.post("/model")
async def post_model(
    model_settings: SaveModelSettingsRequest,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Save model settings for a specific provider.

    Args:
        model_settings (SaveModelSettingsRequest): The model settings to save.
        app (DiveHostAPI): The DiveHostAPI instance.

    Returns:
        ResultResponse: Result of the save operation.
    """
    app.model_config_manager.save_single_settings(
        provider=model_settings.provider,
        upload_model_settings=model_settings.model_settings,
        enable_tools=model_settings.enable_tools,
    )

    # Reload model config
    if not app.model_config_manager.initialize():
        raise ValueError("Failed to reload model configuration")

    # Reload host
    host_config = await app.load_host_config()
    await app.dive_host["default"].reload(new_config=host_config)

    return ResultResponse(success=True)


@config.post("/model-embedding")
async def post_model_embedding(
    embed_config: EmbedConfig,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Save embedding model settings.

    Args:
        embed_config (EmbedConfig): The embedding model settings to save.
        app (DiveHostAPI): The DiveHostAPI instance.

    Returns:
        ResultResponse: Result of the save operation.
    """
    app.model_config_manager.save_embed_settings(embed_config)

    # Reload model config
    if not app.model_config_manager.initialize():
        raise ValueError("Failed to reload model configuration")

    # Reload host
    host_config = await app.load_host_config()
    await app.dive_host["default"].reload(new_config=host_config)

    return ResultResponse(success=True)


@config.post("/model/replaceAll")
async def post_model_replace_all(
    model_config: "ModelFullConfigs",
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Replace all model configurations.

    Args:
        model_config (ModelConfig): The complete model configuration to use.
        app (DiveHostAPI): The DiveHostAPI instance.

    Returns:
        ResultResponse: Result of the replace operation.
    """
    app.model_config_manager.replace_all_settings(model_config)
    if not app.model_config_manager.initialize():
        raise ValueError("Failed to reload model configuration")

    # Reload host
    host_config = await app.load_host_config()
    await app.dive_host["default"].reload(new_config=host_config)

    return ResultResponse(success=True)


@config.get("/model/interface")
async def get_model_interface() -> InterfaceResult:
    """Get model interface definition.

    Returns:
        InterfaceResult: Model interface definition.
    """
    return InterfaceResult(
        success=True,
        interface=ModelInterfaceDefinition(
            model_settings={
                "modelProvider": ModelSettingsDefinition(
                    type="string",
                    description="The provider sdk of the model",
                    required=True,
                    default="",
                    placeholder="openai",
                ),
                "model": ModelSettingsDefinition(
                    type="string",
                    description="The model's name to use",
                    required=True,
                    default="gpt-4o-mini",
                ),
                "apiKey": ModelSettingsDefinition(
                    type="string",
                    description="The Model Provider API key",
                    required=False,
                    default="",
                    placeholder="YOUR_API_KEY",
                ),
                "baseURL": ModelSettingsDefinition(
                    type="string",
                    description="The model's base URL",
                    required=False,
                    default="",
                    placeholder="",
                ),
            },
        ),
    )


@config.get("/customrules")
async def get_custom_rules(app: DiveHostAPI = Depends(get_app)) -> RulesResult:
    """Get custom rules configuration.

    Returns:
        RulesResult: Custom rules as a string.
    """
    custom_rules = app.prompt_config_manager.load_custom_rules()
    return RulesResult(success=True, rules=custom_rules)


@config.post("/customrules")
async def post_custom_rules(
    request: Request,
    app: DiveHostAPI = Depends(get_app),
) -> ResultResponse:
    """Save custom rules configuration.

    Returns:
        ResultResponse: Result of the save operation.
    """
    raw_rules = await request.body()
    rules = raw_rules.decode("utf-8")
    app.prompt_config_manager.write_custom_rules(rules)
    app.prompt_config_manager.update_prompts()
    return ResultResponse(success=True)
