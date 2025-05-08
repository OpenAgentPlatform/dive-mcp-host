from argparse import ArgumentParser
from os import environ
from pathlib import Path
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, Field, model_validator

from dive_mcp_host.httpd.conf.misc import DIVE_CONFIG_DIR

type StrPath = str | Path


def config_file(env_name: str, file_name: str) -> Path:
    """Get the path to the configuration file."""
    return Path(environ.get(env_name, Path.cwd())).joinpath(file_name)


def _convert_path(x: str | None) -> StrPath | None:
    return Path(x) if x else x


class Arguments(BaseModel):
    """Command line arguments of dive_httpd."""

    httpd_config: Annotated[StrPath, AfterValidator(_convert_path)] = Field(
        alias="config",
        default="",
        description="Main service configuration file.",
    )
    llm_config: Annotated[StrPath, AfterValidator(_convert_path)] = Field(
        alias="model_config",
        default="",
        description="Model configuration file.",
    )

    mcp_config: Annotated[StrPath, AfterValidator(_convert_path)] = Field(
        description="MCP configuration file.",
        default="",
    )

    plugin_config: Annotated[StrPath | None, AfterValidator(_convert_path)] = Field(
        description="Plugin configuration file.",
        default="",
    )

    custom_rules: Annotated[StrPath | None, AfterValidator(_convert_path)] = Field(
        description="Custom rules for LLM.",
        default="",
    )

    command_alias_config: Annotated[StrPath | None, AfterValidator(_convert_path)] = (
        Field(
            description="Configuration for command aliases.",
            default="",
        )
    )

    listen: str = Field(
        default="127.0.0.1",
        description="Network interface to bind the server to.",
    )

    port: int | None = Field(
        default=None,
        description="TCP port number to listen on. Use 0 for automatic port selection.",
    )

    auto_reload: bool = Field(
        default=False,
        description="Automatically reload configurations when changes are detected.",
    )

    working_dir: Annotated[StrPath | None, AfterValidator(_convert_path)] = Field(
        default=None,
        description="Base directory for server operations.",
    )

    report_status_file: Annotated[StrPath | None, AfterValidator(_convert_path)] = (
        Field(
            default=None,
            description="File path to write server status information.",
        )
    )

    report_status_fd: int | None = Field(
        default=None,
        description="File descriptor to write server status information.",
    )

    auto_generate_configs: bool = Field(
        default=True,
        description="Automatically generate configuration files if they don't exist.",
    )

    cors_origin: str | None = Field(
        default=None,
        description="CORS origin to allow.",
    )

    log_dir: Annotated[StrPath | None, AfterValidator(_convert_path)] = Field(
        default=None,
        description="Directory to write log files.",
    )

    log_level: str = Field(
        default="INFO",
        description="Log level to use.",
    )

    @model_validator(mode="after")
    def rewrite_default_path(self) -> Self:
        """Rewrite default config file path according to working directory."""
        cwd = Path(self.working_dir) if self.working_dir else DIVE_CONFIG_DIR
        if not self.httpd_config:
            self.httpd_config = cwd.joinpath("dive_httpd.json")
        if not self.llm_config:
            self.llm_config = cwd.joinpath("model_config.json")
        if not self.mcp_config:
            self.mcp_config = cwd.joinpath("mcp_config.json")
        if not self.command_alias_config:
            self.command_alias_config = cwd.joinpath("command_alias.json")
        if not self.plugin_config:
            self.plugin_config = cwd.joinpath("plugin_config.json")
        return self

    @classmethod
    def parse_args(cls, args: list[str] | None = None) -> Self:
        """Create an argumentparser from the arguments model."""
        parser = ArgumentParser(prog="dive_httpd", exit_on_error=False)
        for name, field in cls.model_fields.items():
            kw = {}
            if field.is_required():
                kw["required"] = True
            arg_name = field.alias or name
            kw["help"] = field.description
            kw["default"] = field.get_default(call_default_factory=True)
            # Handle different field types for argument parsing
            # Convert field names with underscores to dashes for CLI arguments
            if field.annotation is int or field.annotation == int | None:
                kw["type"] = int
            elif field.annotation is bool:
                kw["action"] = (
                    "store_false" if kw.get("default", True) else "store_true"
                )
                if "default" in kw:
                    kw.pop("default")
            elif (
                field.annotation is str
                or field.annotation == StrPath
                or field.annotation == StrPath | None
                or field.annotation is Path
            ):
                kw["type"] = str
            parser.add_argument(f"--{arg_name}", dest=arg_name, **kw)

        return cls.model_validate(vars(parser.parse_args(args)))
