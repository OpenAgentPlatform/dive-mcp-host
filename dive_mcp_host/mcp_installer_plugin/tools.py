"""Tools for the MCP Server Installer Agent.

These tools provide fetch, bash, and filesystem operations with built-in
elicitation support for user approval of potentially dangerous operations.
"""

# ruff: noqa: E501, PLR0911, PLR2004, S105
# E501: Line too long - tool descriptions require specific formatting
# PLR0911: Many return statements needed for complex control flow
# PLR2004: Magic values are intentional truncation limits
# S105: password_prompt is not a hardcoded password, it's a prompt message

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import httpx
from langchain_core.tools import BaseTool, InjectedToolArg
from langgraph.config import get_config
from pydantic import BaseModel, Field

from dive_mcp_host.mcp_installer_plugin.events import (
    AgentToolCall,
    AgentToolResult,
    InstallerToolLog,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def _ensure_config(config: RunnableConfig | None) -> RunnableConfig:
    """Ensure config is available, falling back to LangGraph context if needed.

    InjectedToolArg may not work correctly with LangGraph's ToolNode,
    so we use get_config() as a fallback to get the config from context.
    """
    if config is not None and config.get("configurable"):
        return config

    try:
        ctx_config = get_config()
        if ctx_config and ctx_config.get("configurable"):
            logger.debug("Using config from LangGraph context")
            return ctx_config
    except (RuntimeError, LookupError):
        pass

    return config or {}


def _get_stream_writer(
    config: RunnableConfig,
) -> Callable[[tuple[str, Any]], None]:
    """Extract stream writer from config or LangGraph context.

    Priority:
    1. Explicitly set stream_writer in config (used by InstallerAgent)
    2. LangGraph's get_stream_writer() (used when running in ToolNode)
    3. No-op lambda as fallback
    """
    # First check if stream_writer is explicitly set in config (InstallerAgent case)
    writer = config.get("configurable", {}).get("stream_writer")
    if writer is not None:
        logger.debug("Using stream_writer from config")
        return writer

    # Try to get stream writer from LangGraph context (ToolNode case)
    try:
        from langgraph.config import get_stream_writer as lg_get_stream_writer

        writer = lg_get_stream_writer()
        if writer is not None:
            logger.debug("Using stream_writer from LangGraph context")
            return writer
        logger.debug("LangGraph get_stream_writer() returned None")
    except (ImportError, RuntimeError, LookupError) as e:
        logger.debug("Could not get stream writer from LangGraph context: %s", e)

    # Fallback to no-op
    logger.debug("Falling back to no-op stream_writer")
    return lambda _: None


def _get_tool_call_id(config: RunnableConfig) -> str | None:
    """Extract tool_call_id from config metadata."""
    return config.get("metadata", {}).get("tool_call_id")


def _get_dry_run(config: RunnableConfig) -> bool:
    """Extract dry_run setting from config."""
    return config.get("configurable", {}).get("dry_run", False)


def _get_mcp_reload_callback(config: RunnableConfig) -> Callable[[], Any] | None:
    """Extract MCP reload callback from config (deprecated)."""
    return config.get("configurable", {}).get("mcp_reload_callback")


def _get_abort_signal(config: RunnableConfig) -> asyncio.Event | None:
    """Extract abort signal from config."""
    return config.get("configurable", {}).get("abort_signal")


def _check_aborted(abort_signal: asyncio.Event | None) -> bool:
    """Check if the abort signal has been set."""
    return abort_signal is not None and abort_signal.is_set()


class AbortedError(Exception):
    """Raised when an operation is aborted."""


def _get_httpd_base_url() -> str | None:
    """Get httpd base URL from runtime config."""
    from dive_mcp_host.mcp_installer_plugin.runtime import get_httpd_base_url

    return get_httpd_base_url()


def _emit_tool_call(
    writer: Callable[[tuple[str, Any]], None],
    tool_call_id: str | None,
    name: str,
    args: dict[str, Any],
) -> None:
    """Emit agent_tool_call event."""
    if tool_call_id:
        writer(
            (
                AgentToolCall.NAME,
                AgentToolCall(tool_call_id=tool_call_id, name=name, args=args),
            )
        )


def _emit_tool_result(
    writer: Callable[[tuple[str, Any]], None],
    tool_call_id: str | None,
    name: str,
    result: str,
) -> None:
    """Emit agent_tool_result event."""
    if tool_call_id:
        writer(
            (
                AgentToolResult.NAME,
                AgentToolResult(tool_call_id=tool_call_id, name=name, result=result),
            )
        )


class FetchInput(BaseModel):
    """Input schema for the fetch tool."""

    url: Annotated[str, Field(description="The URL to fetch content from.")]
    method: Annotated[
        str, Field(default="GET", description="HTTP method (GET, POST, etc.).")
    ] = "GET"
    headers: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Optional HTTP headers."),
    ] = None


class InstallerFetchTool(BaseTool):
    """Tool for fetching content from URLs.

    This tool fetches content from URLs with built-in approval for
    unfamiliar or potentially unsafe URLs.
    """

    name: str = "fetch"
    description: str = """Fetch content from a URL.
Use this to retrieve documentation, package information, or other web content.
The tool will request user approval for unfamiliar URLs."""
    args_schema: type[BaseModel] | None = FetchInput

    async def _arun(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Fetch content from a URL.

        Note: User confirmation is handled by the confirm_install node in the graph,
        not by individual tools.
        """
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        tool_call_id = _get_tool_call_id(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        tool_args = {"url": url, "method": method, "headers": headers}
        _emit_tool_call(stream_writer, tool_call_id, self.name, tool_args)

        # Perform the fetch
        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="fetch",
                    action=f"Fetching {url}",
                    details={"url": url, "method": method},
                ),
            )
        )
        try:
            # Prepare headers with default User-Agent from env if set
            request_headers = dict(headers) if headers else {}
            user_agent = (
                os.environ.get("DIVE_USER_AGENT")
                or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) dive-mcp-host (+https://github.com/OpenAgentPlatform/dive-mcp-host)"
            )
            if user_agent and "User-Agent" not in request_headers:
                request_headers["User-Agent"] = user_agent

            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                # Create a task for the request
                request_task = asyncio.create_task(
                    client.request(method, url, headers=request_headers or None)
                )

                # Wait for either the request to complete or abort signal
                if abort_signal is not None:
                    abort_task = asyncio.create_task(abort_signal.wait())
                    done, pending = await asyncio.wait(
                        [request_task, abort_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

                    # Check if aborted
                    if abort_task in done:
                        return "Error: Operation aborted."

                    response = request_task.result()
                else:
                    response = await request_task

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    result = response.text
                elif "text/" in content_type or "application/xml" in content_type:
                    # Truncate very long responses
                    text = response.text
                    if len(text) > 50000:
                        result = text[:50000] + "\n... (truncated)"
                    else:
                        result = text
                else:
                    # For binary content, just return a summary
                    result = f"Binary content ({content_type}), size: {len(response.content)} bytes"

        except httpx.HTTPError as e:
            result = f"Error fetching {url}: {e}"

        _emit_tool_result(stream_writer, tool_call_id, self.name, result)

        return result

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class BashInput(BaseModel):
    """Input schema for the bash tool."""

    command: Annotated[str, Field(description="The bash command to execute.")]
    working_dir: Annotated[
        str | None,
        Field(default=None, description="Working directory for the command."),
    ] = None
    timeout: Annotated[
        int,
        Field(
            default=120,
            description="Timeout in seconds (max 600). Use longer timeout for "
            "commands that take time (e.g., npm install, cargo build).",
        ),
    ] = 120
    requires_password: Annotated[
        bool,
        Field(
            default=False,
            description="Set to true if the command requires password input "
            "(e.g., sudo commands). This will prompt user for password securely.",
        ),
    ] = False
    password_prompt: Annotated[
        str | None,
        Field(
            default=None,
            description="Custom prompt message for password input. "
            "Required if requires_password is true.",
        ),
    ] = None
    is_high_risk: Annotated[
        bool,
        Field(
            default=False,
            description="Set to true for high-risk commands (e.g., sudo, rm -rf, "
            "system modifications). This adds extra warning in confirmation.",
        ),
    ] = False


def _detect_high_risk_command(command: str) -> tuple[bool, list[str]]:
    """Detect if a command is high-risk.

    Returns:
        Tuple of (is_high_risk, list of reasons).
    """
    reasons = []
    command_lower = command.lower()

    # Check for sudo
    if "sudo " in command_lower or command_lower.startswith("sudo"):
        reasons.append("Uses sudo (elevated privileges)")

    # Check for dangerous rm commands
    if "rm " in command_lower and ("-rf" in command_lower or "-fr" in command_lower):
        reasons.append("Recursive force delete (rm -rf)")

    # Check for system directories
    system_paths = ["/etc/", "/usr/", "/bin/", "/sbin/", "/var/", "/boot/", "/root/"]
    for path in system_paths:
        if path in command_lower:
            reasons.append(f"Modifies system directory ({path})")
            break

    # Check for package manager with sudo
    if "sudo " in command_lower and any(
        pm in command_lower for pm in ["apt", "yum", "dnf", "pacman", "brew"]
    ):
        reasons.append("System package installation")

    # Check for chmod/chown
    if "chmod " in command_lower or "chown " in command_lower:
        reasons.append("Changes file permissions/ownership")

    return len(reasons) > 0, reasons


class InstallerBashTool(BaseTool):
    """Tool for executing bash commands.

    This tool executes bash commands with built-in approval for
    potentially dangerous commands.
    """

    name: str = "bash"
    description: str = """Execute a bash command.
Use this for installation commands, checking versions, and system operations.

Parameters:
- command: The bash command to execute
- working_dir: Optional working directory
- timeout: Timeout in seconds (default 120, max 600). Set higher for slow commands.
- requires_password: Set true if command needs password (e.g., sudo). Will prompt user securely.
- password_prompt: Message shown when prompting for password
- is_high_risk: Set true for dangerous commands. Auto-detected for sudo, rm -rf, etc.

Examples:
- Simple check: bash(command="node --version")
- Install with npm: bash(command="npm install -g package", timeout=300)
- With sudo: bash(command="sudo apt install package", requires_password=true,
               password_prompt="Enter password for apt install", is_high_risk=true)

Safety notes:
- Commands with sudo are automatically marked as high-risk
- Avoid commands that could damage the system
- Prefer package managers (uvx, npx) over manual installations"""
    args_schema: type[BaseModel] | None = BashInput

    async def _arun(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int = 120,
        requires_password: bool = False,
        password_prompt: str | None = None,
        is_high_risk: bool = False,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Execute a bash command.

        Note: User confirmation is handled by the request_confirmation tool.
        Password input uses elicitation with password format.
        """
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        tool_call_id = _get_tool_call_id(config)
        dry_run = _get_dry_run(config)

        args = {
            "command": command,
            "working_dir": working_dir,
            "timeout": timeout,
            "requires_password": requires_password,
            "is_high_risk": is_high_risk,
        }

        _emit_tool_call(stream_writer, tool_call_id, self.name, args)

        result = await self._execute_bash(
            command=command,
            working_dir=working_dir,
            timeout=timeout,
            requires_password=requires_password,
            password_prompt=password_prompt,
            is_high_risk=is_high_risk,
            stream_writer=stream_writer,
            dry_run=dry_run,
            config=config,
        )

        _emit_tool_result(stream_writer, tool_call_id, self.name, result)

        return result

    async def _execute_bash(
        self,
        command: str,
        working_dir: str | None,
        timeout: int,
        requires_password: bool,
        password_prompt: str | None,
        is_high_risk: bool,
        stream_writer: Callable[[tuple[str, Any]], None],
        dry_run: bool,
        config: RunnableConfig,
    ) -> str:
        """Execute the bash command (internal implementation)."""
        from mcp import types

        from dive_mcp_host.host.tools.elicitation_manager import (
            ElicitationManager,
            ElicitationTimeoutError,
        )

        abort_signal = _get_abort_signal(config)
        elicitation_manager: ElicitationManager | None = config.get(
            "configurable", {}
        ).get("elicitation_manager")

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        # Cap timeout at 10 minutes
        timeout = min(timeout, 600)

        # Auto-detect high-risk commands
        auto_high_risk, risk_reasons = _detect_high_risk_command(command)
        is_high_risk = is_high_risk or auto_high_risk

        # Auto-detect if password is needed (sudo without -n flag)
        if "sudo " in command.lower() and "-n " not in command.lower():
            requires_password = True
            if not password_prompt:
                password_prompt = "Enter sudo password to execute the command"

        # Log the command with high-risk warning if applicable
        log_details: dict[str, Any] = {
            "command": command,
            "working_dir": working_dir,
            "dry_run": dry_run,
            "timeout": timeout,
        }
        if is_high_risk:
            log_details["high_risk"] = True
            log_details["risk_reasons"] = risk_reasons

        action_prefix = ""
        if dry_run:
            action_prefix = "[DRY RUN] "
        if is_high_risk:
            action_prefix += "[HIGH RISK] "

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="bash",
                    action=f"{action_prefix}Executing: {command}",
                    details=log_details,
                ),
            )
        )

        # If dry_run is enabled, simulate success without executing
        if dry_run:
            return f"[DRY RUN] Command would be executed: {command}\nSimulated success."

        # Request user confirmation before executing the command
        if elicitation_manager is not None:
            confirm_message = (
                f"The installer agent wants to execute the following command:\n\n"
                f"```\n{command}\n```"
            )
            if is_high_risk:
                confirm_message += f"\n\n⚠️ **High Risk**: {', '.join(risk_reasons)}"

            confirm_schema = {
                "type": "object",
                "properties": {},
            }

            params = types.ElicitRequestFormParams(
                message=confirm_message,
                requestedSchema=confirm_schema,
            )

            logger.info(
                "Requesting user confirmation for bash command: %s", command[:100]
            )

            try:
                result = await elicitation_manager.request(
                    params=params,
                    writer=stream_writer,
                )

                if result.action == "decline":
                    return "Command cancelled: User declined to execute the command."
                if result.action != "accept":
                    return "Command cancelled: User cancelled the confirmation."

            except ElicitationTimeoutError:
                return "Error: Confirmation timed out. Command not executed."
            except Exception as e:
                logger.exception("Error getting confirmation via elicitation")
                return f"Error getting confirmation: {e}"

        # Handle password input via elicitation
        password: str | None = None
        if requires_password:
            if elicitation_manager is None:
                return (
                    "Error: Command requires password but no elicitation manager available. "
                    "Cannot execute privileged commands."
                )

            # Request password using elicitation with password format
            password_schema = {
                "type": "object",
                "properties": {
                    "password": {
                        "type": "string",
                        "format": "password",
                        "description": "Password for command execution",
                    },
                },
                "required": ["password"],
            }

            params = types.ElicitRequestFormParams(
                message=password_prompt or "Enter password to execute the command",
                requestedSchema=password_schema,
            )

            logger.info("Requesting password for command execution")

            try:
                result = await elicitation_manager.request(
                    params=params,
                    writer=stream_writer,
                )

                if result.action == "accept" and result.content:
                    password_value = result.content.get("password")
                    if not password_value or not isinstance(password_value, str):
                        return "Error: No password provided. Command not executed."
                    password = password_value
                elif result.action == "decline":
                    return "Command cancelled: User declined to provide password."
                else:
                    return "Command cancelled: User cancelled password input."

            except ElicitationTimeoutError:
                return "Error: Password input timed out. Command not executed."
            except Exception as e:
                logger.exception("Error getting password via elicitation")
                return f"Error getting password: {e}"

        # Execute the command
        try:
            # Check abort before execution
            if _check_aborted(abort_signal):
                return "Error: Operation aborted."

            # For commands requiring password (sudo), use stdin to pipe the password
            if password and "sudo " in command.lower():
                # Use sudo -S to read password from stdin
                if "-S" not in command:
                    # Insert -S after sudo
                    command = command.replace("sudo ", "sudo -S ", 1)

                process = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                    env={**os.environ},
                    start_new_session=True,  # Create new process group for proper cleanup
                )

                try:
                    # Send password to stdin with abort signal monitoring
                    communicate_task = asyncio.create_task(
                        process.communicate(input=f"{password}\n".encode())
                    )
                    stdout, stderr = await self._wait_with_abort(
                        communicate_task, abort_signal, process, timeout
                    )
                except AbortedError:
                    return "Error: Operation aborted."
                except TimeoutError:
                    self._kill_process_tree(process)
                    return f"Error: Command timed out after {timeout}s"
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                    env={**os.environ},
                    start_new_session=True,  # Create new process group for proper cleanup
                )

                try:
                    # Monitor abort signal during command execution
                    communicate_task = asyncio.create_task(process.communicate())
                    stdout, stderr = await self._wait_with_abort(
                        communicate_task, abort_signal, process, timeout
                    )
                except AbortedError:
                    return "Error: Operation aborted."
                except TimeoutError:
                    self._kill_process_tree(process)
                    return f"Error: Command timed out after {timeout}s"

            result_parts = []
            if stdout:
                stdout_text = stdout.decode("utf-8", errors="replace")
                result_parts.append(f"stdout:\n{stdout_text}")
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                # Filter out sudo password prompt from stderr
                if password:
                    stderr_text = "\n".join(
                        line
                        for line in stderr_text.split("\n")
                        if "[sudo]" not in line and "Password:" not in line
                    )
                if stderr_text.strip():
                    result_parts.append(f"stderr:\n{stderr_text}")

            exit_code = process.returncode
            result_parts.append(f"\nexit_code: {exit_code}")

            result = "\n".join(result_parts)

            # Truncate very long output
            if len(result) > 20000:
                result = result[:20000] + "\n... (truncated)"

            return result

        except (OSError, TimeoutError) as e:
            return f"Error executing command: {e}"

    async def _wait_with_abort(
        self,
        task: asyncio.Task[tuple[bytes, bytes]],
        abort_signal: asyncio.Event | None,
        process: asyncio.subprocess.Process,
        timeout: int,
    ) -> tuple[bytes, bytes]:
        """Wait for a task with abort signal and timeout support.

        Args:
            task: The asyncio task to wait for.
            abort_signal: Optional abort signal event.
            process: The subprocess to kill if aborted.
            timeout: Timeout in seconds.

        Returns:
            The result of the task (stdout, stderr).

        Raises:
            AbortedError: If the operation was aborted.
            TimeoutError: If the operation timed out.
        """
        if abort_signal is None:
            return await asyncio.wait_for(task, timeout=timeout)

        abort_task = asyncio.create_task(abort_signal.wait())
        timeout_task = asyncio.create_task(asyncio.sleep(timeout))

        done, pending = await asyncio.wait(
            [task, abort_task, timeout_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for pending_task in pending:
            pending_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending_task

        # Check what completed first
        if abort_task in done:
            # Abort was signaled - kill process group
            self._kill_process_tree(process)
            raise AbortedError("Operation aborted")

        if timeout_task in done:
            # Timeout occurred - kill process group
            self._kill_process_tree(process)
            raise TimeoutError(f"Command timed out after {timeout}s")

        # Task completed successfully
        return task.result()

    def _kill_process_tree(self, process: asyncio.subprocess.Process) -> None:
        """Kill the process and all its children.

        Uses process group kill on Unix/Linux to ensure all child processes
        are terminated. On Windows, falls back to regular process kill.
        """
        if process.pid is None:
            return

        try:
            if sys.platform != "win32":
                # On Unix/Linux, kill the entire process group
                # The process was started with start_new_session=True,
                # so its PID is also the PGID
                os.killpg(process.pid, signal.SIGKILL)
            else:
                # On Windows, just kill the process
                # Windows doesn't have process groups in the same way
                process.kill()
        except (ProcessLookupError, OSError):
            # Process already terminated
            pass

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class ReadFileInput(BaseModel):
    """Input schema for the read_file tool."""

    path: Annotated[str, Field(description="Path to the file to read.")]
    encoding: Annotated[
        str,
        Field(default="utf-8", description="File encoding."),
    ] = "utf-8"


class InstallerReadFileTool(BaseTool):
    """Tool for reading files from the filesystem.

    This tool reads files with built-in approval for sensitive paths.
    """

    name: str = "read_file"
    description: str = """Read content from a file.
Use this to read configuration files, check existing setups, etc.
Supports text files only."""
    args_schema: type[BaseModel] | None = ReadFileInput

    async def _arun(
        self,
        path: str,
        encoding: str = "utf-8",
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Read a file.

        Note: User confirmation is handled by the confirm_install node in the graph,
        not by individual tools.
        """
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        tool_call_id = _get_tool_call_id(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        args = {"path": path, "encoding": encoding}
        _emit_tool_call(stream_writer, tool_call_id, self.name, args)

        # Expand user home directory
        expanded_path = str(Path(path).expanduser())

        # Read the file
        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="read_file",
                    action=f"Reading: {path}",
                    details={"path": expanded_path},
                ),
            )
        )
        try:
            file_path = Path(expanded_path)
            if not file_path.exists():
                result = f"Error: File not found: {path}"
            elif not file_path.is_file():
                result = f"Error: Not a file: {path}"
            else:
                content = file_path.read_text(encoding=encoding)

                # Truncate very long files
                if len(content) > 100000:
                    result = content[:100000] + "\n... (truncated)"
                else:
                    result = content

        except OSError as e:
            result = f"Error reading file {path}: {e}"

        _emit_tool_result(stream_writer, tool_call_id, self.name, result)

        return result

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class WriteFileInput(BaseModel):
    """Input schema for the write_file tool."""

    path: Annotated[str, Field(description="Path to the file to write.")]
    content: Annotated[str, Field(description="Content to write to the file.")]
    encoding: Annotated[
        str,
        Field(default="utf-8", description="File encoding."),
    ] = "utf-8"
    create_dirs: Annotated[
        bool,
        Field(
            default=True, description="Create parent directories if they don't exist."
        ),
    ] = True


class InstallerWriteFileTool(BaseTool):
    """Tool for writing files to the filesystem.

    This tool writes files with built-in approval to prevent
    accidental overwrites or writing to sensitive locations.
    """

    name: str = "write_file"
    description: str = """Write content to a file.
Use this to create or modify configuration files, scripts, etc.
Will create parent directories if needed.
Always requests user approval before writing."""
    args_schema: type[BaseModel] | None = WriteFileInput

    async def _arun(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Write to a file.

        Requests user confirmation before writing.
        """
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        tool_call_id = _get_tool_call_id(config)
        dry_run = _get_dry_run(config)

        args = {
            "path": path,
            "content": content[:200] + "..." if len(content) > 200 else content,
            "encoding": encoding,
            "create_dirs": create_dirs,
        }

        _emit_tool_call(stream_writer, tool_call_id, self.name, args)

        result = await self._execute_write(
            path=path,
            content=content,
            encoding=encoding,
            create_dirs=create_dirs,
            stream_writer=stream_writer,
            dry_run=dry_run,
            config=config,
        )

        _emit_tool_result(stream_writer, tool_call_id, self.name, result)

        return result

    async def _execute_write(
        self,
        path: str,
        content: str,
        encoding: str,
        create_dirs: bool,
        stream_writer: Callable[[tuple[str, Any]], None],
        dry_run: bool,
        config: RunnableConfig,
    ) -> str:
        """Execute the write file operation (internal implementation)."""
        from mcp import types

        from dive_mcp_host.host.tools.elicitation_manager import (
            ElicitationManager,
            ElicitationTimeoutError,
        )

        abort_signal = _get_abort_signal(config)
        elicitation_manager: ElicitationManager | None = config.get(
            "configurable", {}
        ).get("elicitation_manager")

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        # Expand user home directory
        expanded_path = str(Path(path).expanduser())
        file_path = Path(expanded_path)
        file_exists = file_path.exists()

        # Prepare content preview (truncate if too long)
        content_preview = content
        if len(content) > 500:
            content_preview = content[:500] + f"\n... ({len(content) - 500} more bytes)"

        # Log the write operation
        log_details: dict[str, Any] = {
            "path": expanded_path,
            "size": len(content),
            "file_exists": file_exists,
            "dry_run": dry_run,
        }

        action_prefix = "[DRY RUN] " if dry_run else ""

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="write_file",
                    action=f"{action_prefix}Writing: {path}",
                    details=log_details,
                ),
            )
        )

        # If dry_run is enabled, simulate success without writing
        if dry_run:
            return f"[DRY RUN] Would write {len(content)} bytes to {path}\nSimulated success."

        # Request user confirmation before writing
        if elicitation_manager is not None:
            operation = "overwrite" if file_exists else "create"
            confirm_message = (
                f"The agent wants to {operation} the following file:\n\n"
                f"**Path:** `{path}`\n"
                f"**Size:** {len(content)} bytes\n\n"
                f"**Content:**\n```\n{content_preview}\n```"
            )

            confirm_schema = {
                "type": "object",
                "properties": {},
            }

            params = types.ElicitRequestFormParams(
                message=confirm_message,
                requestedSchema=confirm_schema,
            )

            logger.info("Requesting user confirmation for write_file: %s", path)

            try:
                result = await elicitation_manager.request(
                    params=params,
                    writer=stream_writer,
                )

                if result.action == "decline":
                    return f"Write cancelled: User declined to {operation} the file."
                if result.action != "accept":
                    return "Write cancelled: User cancelled the confirmation."

            except ElicitationTimeoutError:
                return "Error: Confirmation timed out. File not written."
            except Exception as e:
                logger.exception("Error getting confirmation via elicitation")
                return f"Error getting confirmation: {e}"

        # Check abort before writing
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        # Write the file
        try:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            file_path.write_text(content, encoding=encoding)

            return f"Successfully wrote {len(content)} bytes to {path}"

        except OSError as e:
            return f"Error writing to file {path}: {e}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class InstallerGetMcpConfigTool(BaseTool):
    """Tool for getting the current MCP server configuration.

    This tool reads the current MCP configuration file and returns
    the list of configured servers.
    """

    name: str = "get_mcp_config"
    description: str = """Get the current MCP server configuration.

Use this tool to check what MCP servers are already configured before adding new ones.
This helps avoid duplicate installations and understand the current setup.

Returns a JSON object with all configured MCP servers, including:
- Server names
- Transport type (stdio, sse, websocket, streamable)
- Command and arguments (for stdio transport)
- URL (for sse/websocket transport)
- Enabled status
"""
    args_schema: type[BaseModel] | None = None

    async def _arun(
        self,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Get the current MCP configuration."""
        import json

        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        tool_call_id = _get_tool_call_id(config)
        abort_signal = _get_abort_signal(config)

        _emit_tool_call(stream_writer, tool_call_id, self.name, {})

        # Check if already aborted
        if _check_aborted(abort_signal):
            result = "Error: Operation aborted."
            _emit_tool_result(stream_writer, tool_call_id, self.name, result)
            return result

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="get_mcp_config",
                    action="Reading MCP configuration",
                    details={},
                ),
            )
        )

        try:
            from dive_mcp_host.httpd.conf.mcp_servers import MCPServerManager

            manager = MCPServerManager()
            manager.initialize()

            current_config = manager._current_config  # noqa: SLF001
            if current_config is None:
                result = json.dumps({"mcpServers": {}}, indent=2)
            else:
                result = current_config.model_dump_json(
                    by_alias=True, exclude_unset=True, indent=2
                )

        except Exception as e:
            logger.exception("Error reading MCP config")
            result = f"Error reading MCP configuration: {e}"

        _emit_tool_result(stream_writer, tool_call_id, self.name, result)

        return result

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class AddMcpServerInput(BaseModel):
    """Input schema for the add_mcp_server tool."""

    server_name: Annotated[
        str,
        Field(description="Unique name for the MCP server (e.g., 'yt-dlp', 'fetch')."),
    ]
    command: Annotated[
        str | None,
        Field(
            default=None,
            description="Command to run for stdio transport (e.g., 'npx', 'uvx', 'python').",
        ),
    ] = None
    args: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Arguments for the command (e.g., ['-y', 'yt-dlp-mcp']).",
        ),
    ] = None
    env: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description="Environment variables for the server.",
        ),
    ] = None
    url: Annotated[
        str | None,
        Field(
            default=None,
            description="URL for sse/websocket transport.",
        ),
    ] = None
    transport: Annotated[
        str,
        Field(
            default="stdio",
            description="Transport type: 'stdio', 'sse', 'websocket', or 'streamable'.",
        ),
    ] = "stdio"
    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the server should be enabled.",
        ),
    ] = True


class InstallerAddMcpServerTool(BaseTool):
    """Tool for adding an MCP server configuration.

    This tool adds a new MCP server to the configuration and reloads
    the host to apply the changes.
    """

    name: str = "add_mcp_server"
    description: str = """Add a new MCP server configuration.

Use this tool to register a newly installed MCP server with the system.
This will add the server to mcp_config.json and reload the host.

For stdio transport (most common), provide:
- server_name: A unique identifier for the server
- command: The command to run (e.g., 'npx', 'uvx', 'python')
- args: List of arguments for the command

For sse/websocket/streamable transport, provide:
- server_name: A unique identifier
- url: The server URL
- transport: 'sse', 'websocket', or 'streamable'

Example for npx-based server:
  server_name="yt-dlp"
  command="npx"
  args=["-y", "yt-dlp-mcp"]

Example for uvx-based server:
  server_name="mcp-server-fetch"
  command="uvx"
  args=["mcp-server-fetch"]
"""
    args_schema: type[BaseModel] | None = AddMcpServerInput

    async def _arun(
        self,
        server_name: str,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        transport: str = "stdio",
        enabled: bool = True,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Add an MCP server configuration.

        Note: User confirmation is handled by the confirm_install node in the graph,
        not by individual tools.
        """
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        # Validate input
        if transport == "stdio" and command is None:
            return "Error: 'command' is required for stdio transport."
        if transport in ["sse", "websocket", "streamable"] and url is None:
            return "Error: 'url' is required for sse/websocket/streamable transport."

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="add_mcp_server",
                    action=f"Adding MCP server: {server_name}",
                    details={
                        "server_name": server_name,
                        "command": command,
                        "args": args,
                        "transport": transport,
                    },
                ),
            )
        )

        try:
            # Import here to avoid circular imports
            from dive_mcp_host.httpd.conf.mcp_servers import (
                Config,
                MCPServerConfig,
                MCPServerManager,
            )

            # Load current config
            manager = MCPServerManager()
            manager.initialize()

            current_config = manager._current_config  # noqa: SLF001
            if current_config is None:
                current_config = Config()

            # Create new server config
            new_server = MCPServerConfig(
                transport=transport,  # type: ignore
                enabled=enabled,
                command=command,
                args=args or [],
                env=env or {},
                url=url,
            )

            # Add to config
            current_config.mcp_servers[server_name] = new_server

            # Trigger reload via HTTP API
            reload_status = await self._trigger_mcp_reload(
                config, current_config, server_name
            )

            return (
                f"Successfully added MCP server '{server_name}' to configuration.{reload_status} "
                f"Config: command={command}, args={args}, transport={transport}"
            )

        except Exception as e:
            logger.exception("Error adding MCP server")
            return f"Error adding MCP server '{server_name}': {e}"

    async def _trigger_mcp_reload(
        self,
        config: RunnableConfig,
        mcp_config: Any,
        server_name: str,
    ) -> str:
        """Trigger MCP server reload via HTTP API.

        Args:
            config: The runnable config (for deprecated mcp_reload_callback).
            mcp_config: The MCP server configuration to send.
            server_name: The name of the server being added (to check for errors).

        Returns:
            Status message for the reload operation, including any errors.
        """
        # First try HTTP API
        httpd_base_url = _get_httpd_base_url()
        if httpd_base_url:
            try:
                async with httpx.AsyncClient() as client:
                    # Serialize config properly
                    payload = mcp_config.model_dump(by_alias=True, exclude_unset=True)
                    response = await client.post(
                        f"{httpd_base_url}/api/config/mcpserver",
                        json=payload,
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        errors = result.get("errors", [])

                        # Check if the specific server we added has an error
                        server_error = None
                        for error in errors:
                            if error.get("serverName") == server_name:
                                server_error = error.get("error", "Unknown error")
                                break

                        if server_error:
                            logger.warning(
                                "MCP server '%s' failed to load: %s",
                                server_name,
                                server_error,
                            )
                            return (
                                f" ERROR: Server '{server_name}' failed to load: {server_error}. "
                                "You may need to install missing dependencies and then use "
                                "reload_mcp_server to retry loading."
                            )

                        logger.info("MCP reload via HTTP API succeeded")
                        return " The server has been loaded and is now available."

                    logger.warning(
                        "MCP reload HTTP API returned %s: %s",
                        response.status_code,
                        response.text,
                    )
                    return " Note: Auto-reload failed, you may need to reload manually."
            except (httpx.HTTPError, OSError, ValueError) as e:
                logger.warning("MCP reload via HTTP API failed: %s", e)
                return (
                    f" Note: Auto-reload failed ({e}), you may need to reload manually."
                )

        # Fallback to callback (deprecated)
        reload_callback = _get_mcp_reload_callback(config)
        if reload_callback:
            try:
                callback_result = reload_callback()
                if asyncio.iscoroutine(callback_result):
                    await callback_result
                logger.info("MCP reload callback executed successfully")
                return " The server has been loaded and is now available."
            except (OSError, RuntimeError) as e:
                logger.warning("MCP reload callback failed: %s", e)
                return (
                    f" Note: Auto-reload failed ({e}), you may need to reload manually."
                )

        return " The server will be available after reloading."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class ReloadMcpServerInput(BaseModel):
    """Input schema for the reload_mcp_server tool."""

    server_name: Annotated[
        str,
        Field(description="Name of the MCP server to reload."),
    ]


class InstallerReloadMcpServerTool(BaseTool):
    """Tool for reloading a specific MCP server.

    Use this tool after installing dependencies to retry loading a server
    that previously failed.
    """

    name: str = "reload_mcp_server"
    description: str = """Reload a specific MCP server.

Use this tool to retry loading an MCP server after:
- Installing missing dependencies
- Fixing configuration issues
- Resolving environment problems

This will reload the server configuration and attempt to start the server again.
Check the result for any errors.

Example:
  reload_mcp_server(server_name="my-server")
"""
    args_schema: type[BaseModel] | None = ReloadMcpServerInput

    async def _arun(
        self,
        server_name: str,
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Reload a specific MCP server."""
        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        abort_signal = _get_abort_signal(config)

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "Error: Operation aborted."

        stream_writer(
            (
                InstallerToolLog.NAME,
                InstallerToolLog(
                    tool="reload_mcp_server",
                    action=f"Reloading MCP server: {server_name}",
                    details={"server_name": server_name},
                ),
            )
        )

        httpd_base_url = _get_httpd_base_url()
        if not httpd_base_url:
            return "Error: Cannot reload - no HTTP API available."

        try:
            from dive_mcp_host.httpd.conf.mcp_servers import MCPServerManager

            # Load current config
            manager = MCPServerManager()
            manager.initialize()

            current_config = manager._current_config  # noqa: SLF001
            if current_config is None:
                return "Error: No MCP configuration found."

            if server_name not in current_config.mcp_servers:
                return f"Error: Server '{server_name}' not found in configuration."

            async with httpx.AsyncClient() as client:
                payload = current_config.model_dump(by_alias=True, exclude_unset=True)
                response = await client.post(
                    f"{httpd_base_url}/api/config/mcpserver?force=true",
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    errors = result.get("errors", [])

                    # Check if the specific server has an error
                    server_error = None
                    for error in errors:
                        if error.get("serverName") == server_name:
                            server_error = error.get("error", "Unknown error")
                            break

                    if server_error:
                        return (
                            f"ERROR: Server '{server_name}' still failed to load: {server_error}. "
                            "Check if all dependencies are correctly installed."
                        )

                    return (
                        f"Successfully reloaded MCP server '{server_name}'. "
                        "The server is now available."
                    )

                return (
                    f"Error: Reload request failed with status "
                    f"{response.status_code}: {response.text}"
                )

        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.exception("Error reloading MCP server")
            return f"Error reloading MCP server '{server_name}': {e}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


class RequestConfirmationInput(BaseModel):
    """Input schema for the request_confirmation tool."""

    message: Annotated[
        str,
        Field(
            description="The message to display to the user asking for confirmation. "
            "Should be in the user's preferred language."
        ),
    ]
    actions: Annotated[
        list[str],
        Field(
            description="List of actions that will be performed if confirmed. "
            "Each action should be a clear, concise description."
        ),
    ]


class InstallerRequestConfirmationTool(BaseTool):
    """Tool for requesting user confirmation before performing actions.

    This tool allows the installer agent to request explicit user approval
    with a custom message in the user's preferred language.
    """

    name: str = "request_confirmation"
    description: str = """Request user confirmation before performing installation actions.

Use this tool BEFORE executing any installation commands, file writes, or server configurations.
You MUST provide:
1. A clear message explaining what you want to do (in the user's language)
2. A list of specific actions that will be performed

The user will see your message and can approve or reject the actions.
If rejected, do NOT proceed with the actions.

Example:
  message: "I need to perform the following actions to install the MCP server. Do you approve?"
  actions: ["Run command: npm install -g @anthropic/mcp-server", "Write config to mcp_config.json"]
"""
    args_schema: type[BaseModel] | None = RequestConfirmationInput

    async def _arun(
        self,
        message: str,
        actions: list[str],
        config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
    ) -> str:
        """Request user confirmation.

        Returns:
            'approved' if user approves, 'rejected' if user rejects.
        """
        from mcp import types

        from dive_mcp_host.host.tools.elicitation_manager import (
            ElicitationManager,
            ElicitationTimeoutError,
        )

        config = _ensure_config(config)

        stream_writer = _get_stream_writer(config)
        abort_signal = _get_abort_signal(config)
        elicitation_manager: ElicitationManager | None = config.get(
            "configurable", {}
        ).get("elicitation_manager")

        # Check if already aborted
        if _check_aborted(abort_signal):
            return "aborted"

        if elicitation_manager is None:
            logger.warning("No elicitation manager, auto-approving")
            return "approved (no elicitation manager available)"

        # Build the full message including actions list
        # This ensures actions are always visible even if LLM provides minimal message
        actions_text = "\n".join(f"• {action}" for action in actions)
        full_message = f"{message}\n\n{actions_text}" if message else actions_text

        # Empty schema - no form fields needed
        # The Accept/Decline buttons in the UI are sufficient for confirmation
        requested_schema = {
            "type": "object",
            "properties": {},
        }

        params = types.ElicitRequestFormParams(
            message=full_message,
            requestedSchema=requested_schema,
        )

        logger.info(
            "InstallerRequestConfirmationTool._arun() - requesting confirmation for %d actions",
            len(actions),
        )

        try:
            result = await elicitation_manager.request(
                params=params,
                writer=stream_writer,
            )

            logger.info(
                "InstallerRequestConfirmationTool._arun() - result: action=%s, content=%s",
                result.action,
                result.content,
            )

            if result.action == "accept":
                return "approved"
            if result.action == "decline":
                return "rejected"
            # cancel
            return "cancelled"

        except ElicitationTimeoutError:
            logger.warning("Elicitation timeout")
            return "timeout"
        except Exception as e:
            logger.exception("Elicitation error")
            return f"error: {e}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Sync version - not implemented."""
        raise NotImplementedError("Use async version")


def get_installer_tools() -> list[BaseTool]:
    """Get all installer agent tools."""
    return [
        InstallerFetchTool(),
        InstallerBashTool(),
        InstallerReadFileTool(),
        InstallerWriteFileTool(),
        InstallerGetMcpConfigTool(),
        InstallerAddMcpServerTool(),
        InstallerReloadMcpServerTool(),
        InstallerRequestConfirmationTool(),
    ]


def get_local_tools() -> list[BaseTool]:
    """Get local tools that can be exposed to external LLMs.

    These tools (fetch, bash, read_file, write_file) can be used by external LLMs
    directly without going through the installer agent. They include built-in
    safety mechanisms like user confirmation for potentially dangerous operations.

    Returns:
        List of local tools: fetch, bash, read_file, write_file.
    """
    return [
        InstallerFetchTool(),
        InstallerBashTool(),
        InstallerReadFileTool(),
        InstallerWriteFileTool(),
    ]
