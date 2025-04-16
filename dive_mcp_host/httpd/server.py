import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from dive_mcp_host.host.conf import HostConfig, ServerConfig
from dive_mcp_host.host.conf.llm import LLMConfig
from dive_mcp_host.host.host import DiveMcpHost
from dive_mcp_host.httpd.abort_controller import AbortController
from dive_mcp_host.httpd.conf.command_alias import CommandAliasManager
from dive_mcp_host.httpd.conf.httpd_service import ServiceManager
from dive_mcp_host.httpd.conf.mcp_servers import MCPServerManager
from dive_mcp_host.httpd.conf.models import ModelManager
from dive_mcp_host.httpd.conf.prompt import PromptManager
from dive_mcp_host.httpd.database.migrate import db_migration
from dive_mcp_host.httpd.database.msg_store.base import BaseMessageStore
from dive_mcp_host.httpd.database.msg_store.sqlite import SQLiteMessageStore
from dive_mcp_host.httpd.store.cache import LocalFileCache
from dive_mcp_host.httpd.store.local import LocalStore

logger = getLogger(__name__)


class Listen(BaseModel):
    """Listen of the DiveHostAPI."""

    ip: str | None = None
    port: int | None = None


class Server(BaseModel):
    """Server of the DiveHostAPI."""

    listen: Listen = Field(default_factory=Listen)


class Status(BaseModel):
    """Status of the DiveHostAPI."""

    state: Literal["UP", "FAILED"]
    last_error: str | None = None
    error_code: str | None = None


class ReportStatus(BaseModel):
    """Report the status of the DiveHostAPI."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    server: Server = Field(default_factory=Server)
    status: Status


class DiveHostAPI(FastAPI):
    """DiveHostAPI is a FastAPI application that is used to host the DiveHost API."""

    dive_host: dict[str, DiveMcpHost]  # shoud init "default" when preapre stage

    def __init__(
        self,
        service_config_manager: ServiceManager,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the DiveHostAPI."""
        super().__init__(*args, **kwargs)
        self._service_config_manager = service_config_manager

        self._listen_ip: str | None = None
        self._listen_port: int | None = None
        self._report_status_file: str | None = None
        self._report_status_fd: int | None = None

        self._engine: AsyncEngine | None = None

        if self._service_config_manager.current_setting is None:
            raise ValueError("Service manager is not initialized")

        self._mcp_server_config_manager = MCPServerManager(
            self._service_config_manager.current_setting.config_location.mcp_server_config_path
        )
        self._model_config_manager = ModelManager(
            self._service_config_manager.current_setting.config_location.model_config_path
        )
        self._prompt_config_manager = PromptManager(
            self._service_config_manager.current_setting.config_location.prompt_config_path
        )
        self._command_alias_config_manager = CommandAliasManager(
            self._service_config_manager.current_setting.config_location.command_alias_config_path
        )

        self._abort_controller = AbortController()

    def _load_configs(self) -> None:
        """Load all configs."""
        logger.info("Loading configs")
        self._mcp_server_config_manager.initialize()
        self._model_config_manager.initialize()
        self._prompt_config_manager.initialize()
        self._command_alias_config_manager.initialize()
        logger.info("Configs loaded")

    @asynccontextmanager
    async def prepare(self) -> AsyncGenerator[None, None]:
        """Setup the DiveHostAPI."""
        logger.info("Server Prepare")
        self._load_configs()

        # ================================================
        # Database
        # ================================================
        if self._service_config_manager.current_setting is None:
            raise ValueError("Service manager is not initialized")

        if self._service_config_manager.current_setting.db.migrate:
            db_migration(uri=self._service_config_manager.current_setting.db.uri)

        self._engine = create_async_engine(
            self._service_config_manager.current_setting.db.async_uri,
            echo=self._service_config_manager.current_setting.db.echo,
            # check connection before using
            pool_pre_ping=self._service_config_manager.current_setting.db.pool_pre_ping,
            # max connections
            pool_size=self._service_config_manager.current_setting.db.pool_size,
            # close connection after 60 seconds
            pool_recycle=self._service_config_manager.current_setting.db.pool_recycle,
            # burst connections
            max_overflow=self._service_config_manager.current_setting.db.max_overflow,
        )
        self._db_sessionmaker = async_sessionmaker(self._engine, class_=AsyncSession)
        self._msg_store = SQLiteMessageStore

        # ================================================
        # Store
        # ================================================
        self._store = LocalStore(
            root_dir=self._service_config_manager.current_setting.resource_dir
        )
        self._local_file_cache = LocalFileCache(
            root_dir=self._service_config_manager.current_setting.resource_dir
        )

        # ================================================
        # Dive Host
        # ================================================
        config = self.load_host_config()
        async with AsyncExitStack() as stack:
            default_host = DiveMcpHost(config)
            await stack.enter_async_context(default_host)
            self.dive_host = {"default": default_host}

            logger.info("Server Prepare Complete")
            yield

    def load_host_config(self) -> HostConfig:
        """Generate all host configs."""
        model_setting = self._model_config_manager.current_setting
        if model_setting is None:
            model_setting = LLMConfig(
                model_provider="dive",
                model="fake",
            )

        if self._service_config_manager.current_setting is None:
            raise ValueError("MCPServer config manager is not initialized")

        if self._command_alias_config_manager.current_config is None:
            raise ValueError("Command alias config manager is not initialized")

        if self._model_config_manager.full_config is None:
            raise ValueError("Model config manager is not initialized")

        mcp_servers: dict[str, ServerConfig] = {}
        for (
            server_name,
            server_config,
        ) in self._mcp_server_config_manager.get_enabled_servers().items():
            if not server_config.enabled:
                continue

            # Apply command alias
            if server_config.command:
                command = self._command_alias_config_manager.current_config.get(
                    server_config.command, server_config.command
                )
            else:
                command = ""

            mcp_servers[server_name] = ServerConfig(
                name=server_name,
                command=command,
                args=server_config.args or [],
                env=server_config.env or {},
                enabled=server_config.enabled,
                url=server_config.url or None,
                transport=server_config.transport or "stdio",
            )

        logger.debug("got %s mcp servers in config", len(mcp_servers))

        return HostConfig(
            llm=model_setting,
            embed=self._model_config_manager.full_config.embed_config,
            checkpointer=self._service_config_manager.current_setting.checkpointer,
            mcp_servers=mcp_servers,
        )

    async def ready(self) -> bool:
        """Ready the DiveHostAPI."""
        try:
            # check db connection
            async with self._db_sessionmaker() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.exception("Server not ready")
            return False

    async def cleanup(self) -> None:
        """Cleanup the DiveHostAPI."""
        logger.info("Server Cleanup")
        if self._engine:
            await self._engine.dispose()
        logger.info("Server Cleanup Complete")

    def set_status_report_info(
        self,
        listen: str,
        report_status_file: str | None = None,
        report_status_fd: int | None = None,
    ) -> None:
        """Set the status report info."""
        self._listen_ip = listen
        self._report_status_file = report_status_file
        self._report_status_fd = report_status_fd

    def set_listen_port(self, port: int) -> None:
        """Set the listen port."""
        self._listen_port = port

    def report_status(self, error: str | None = None) -> None:
        """Report the status of the DiveHostAPI."""
        if error:
            msg = ReportStatus(
                status=Status(state="FAILED", last_error=error),
            )

        elif self._listen_ip and self._listen_port:
            msg = ReportStatus(
                server=Server(
                    listen=Listen(ip=self._listen_ip, port=self._listen_port)
                ),
                status=Status(state="UP"),
            )
        else:
            raise ValueError("Status info not complete")

        if self._report_status_file:
            logger.info("Report status to file: %s", self._report_status_file)
            with Path(self._report_status_file).open("w") as f:
                f.write(msg.model_dump_json())

        elif self._report_status_fd:
            logger.info("Report status to fd: %s", self._report_status_fd)
            os.write(
                self._report_status_fd,
                f"{msg.model_dump_json()}\r\n".encode(),
            )
        else:
            logger.info("No fd or file to report status")

    @property
    def db_sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        """Get the database sessionmaker."""
        return self._db_sessionmaker

    @property
    def msg_store(self) -> type[BaseMessageStore]:
        """Get message store type."""
        return self._msg_store

    @property
    def store(self) -> LocalStore:
        """Get the store."""
        return self._store

    @property
    def local_file_cache(self) -> LocalFileCache:
        """Get the local file cache."""
        return self._local_file_cache

    @property
    def service_manager(self) -> ServiceManager:
        """Get the service manager."""
        return self._service_config_manager

    @property
    def mcp_server_config_manager(self) -> MCPServerManager:
        """Get the MCP server config manager."""
        return self._mcp_server_config_manager

    @property
    def model_config_manager(self) -> ModelManager:
        """Get the model config manager."""
        return self._model_config_manager

    @property
    def prompt_config_manager(self) -> PromptManager:
        """Get the prompt config manager."""
        return self._prompt_config_manager

    @property
    def abort_controller(self) -> AbortController:
        """Get the abort controller."""
        return self._abort_controller
