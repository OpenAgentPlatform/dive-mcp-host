"""Dive MCP Host HTTPD.

Support Restful API and websocket.
"""

import socket
from pathlib import Path

import uvicorn

from dive_mcp_host.env import RESOURCE_DIR
from dive_mcp_host.httpd.app import create_app
from dive_mcp_host.httpd.conf.arguments import Arguments
from dive_mcp_host.httpd.conf.httpd_service import ConfigLocation, ServiceManager


def main() -> None:
    """Dive MCP Host HTTPD entrypoint."""
    args = Arguments.parse_args()

    service_config_manager = ServiceManager(str(args.httpd_config))
    service_config_manager.initialize()
    if service_config_manager.current_setting is None:
        raise ValueError("Service config manager is not initialized")

    # Overwrite defaults from command line arguments
    resource_dir = Path(args.working_dir) if args.working_dir else RESOURCE_DIR
    service_config_manager.overwrite_paths(
        ConfigLocation(
            mcp_server_config_path=str(args.mcp_config),
            model_config_path=str(args.llm_config),
            prompt_config_path=str(args.custom_rules),
            command_alias_config_path=str(args.command_alias_config),
        ),
        resource_dir=resource_dir,
    )

    if args.cors_origin:
        service_config_manager.current_setting.cors_origin = args.cors_origin

    service_config_manager.current_setting.logging_config["root"]["level"] = (
        args.log_level
    )

    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        service_config_manager.current_setting.logging_config["handlers"]["rotate"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_dir.joinpath("dive_httpd.log"),
            "maxBytes": 1048576,
            "backupCount": 5,
        }
        service_config_manager.current_setting.logging_config["root"][
            "handlers"
        ].append("rotate")

    app = create_app(service_config_manager)
    app.set_status_report_info(
        listen=args.listen,
        report_status_file=str(args.report_status_file)
        if args.report_status_file
        else None,
        report_status_fd=args.report_status_fd,
    )

    serversocket = socket.socket(
        socket.AF_INET6 if ":" in args.listen else socket.AF_INET,
        socket.SOCK_STREAM,
    )
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    if args.port:
        app.set_listen_port(args.port)

        try:
            serversocket.bind((args.listen, args.port))
            serversocket.close()
        except OSError:
            error_msg = f"Failed to bind to {args.listen}:{args.port}"
            app.report_status(error=error_msg)
            raise

        uvicorn.run(
            app,
            host=args.listen,
            port=args.port,
            log_config=service_config_manager.current_setting.logging_config,
        )

    else:
        start = 61990
        port = 0
        for i in range(1000):
            port = start + i
            app.set_listen_port(port)
            try:
                serversocket.bind((args.listen, port))
                serversocket.close()
                break
            except OSError:
                pass
        else:
            error_msg = f"No available port found in range {start}-{port}"
            app.report_status(error=error_msg)
            raise RuntimeError(error_msg)

        uvicorn.run(
            app,
            host=args.listen,
            port=port,
            log_config=service_config_manager.current_setting.logging_config,
        )
