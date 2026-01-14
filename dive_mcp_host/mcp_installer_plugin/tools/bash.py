"""Bash tool for the MCP Server Installer Agent.

This module provides the bash tool for executing shell commands with
built-in safety features including write command detection and elicitation.
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
from collections.abc import Callable  # noqa: TC003
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from dive_mcp_host.mcp_installer_plugin.events import InstallerToolLog
from dive_mcp_host.mcp_installer_plugin.tools.common import (
    AbortedError,
    _check_aborted,
    _ensure_config,
    _get_abort_signal,
    _get_dry_run,
    _get_stream_writer,
)
from dive_mcp_host.mcp_installer_plugin.tools.patterns import (
    _detect_high_risk_command,
    _detect_write_command,
)

logger = logging.getLogger(__name__)


@tool(
    description="""Execute a bash command.
Use this for installation commands, checking versions, and system operations.

Parameters:
- command: The bash command to execute
- working_dir: Optional working directory
- timeout: Timeout in seconds (default 120, max 600). Set higher for slow commands.
- requires_password: Set true if command needs password (e.g., sudo). Will prompt user securely.
- password_prompt: Message shown when prompting for password
- is_high_risk: Set true for dangerous commands. Auto-detected for sudo, rm -rf, etc.
- requires_confirmation: Set to false for read-only commands (e.g., ls, cat, pwd, grep, echo).
  Write/update commands (rm, mv, sed -i, etc.) will still require confirmation even if false.

Examples:
- Simple check: bash(command="node --version", requires_confirmation=false)
- Read file: bash(command="cat /etc/hosts", requires_confirmation=false)
- List files: bash(command="ls -la", requires_confirmation=false)
- Install with npm: bash(command="npm install -g package", timeout=300)
- Delete file: bash(command="rm file.txt")  # requires confirmation (write operation)
- With sudo: bash(command="sudo apt install package", requires_password=true,
               password_prompt="Enter password for apt install", is_high_risk=true)

Safety notes:
- Commands with sudo are automatically marked as high-risk
- Write/update operations always require user confirmation regardless of requires_confirmation
- Avoid commands that could damage the system
- Prefer package managers (uvx, npx) over manual installations"""
)
async def bash(
    command: Annotated[str, Field(description="The bash command to execute.")],
    working_dir: Annotated[
        str | None,
        Field(default=None, description="Working directory for the command."),
    ] = None,
    timeout: Annotated[
        int,
        Field(
            default=120,
            description="Timeout in seconds (max 600). Use longer timeout for "
            "commands that take time (e.g., npm install, cargo build).",
        ),
    ] = 120,
    requires_password: Annotated[
        bool,
        Field(
            default=False,
            description="Set to true if the command requires password input "
            "(e.g., sudo commands). This will prompt user for password securely.",
        ),
    ] = False,
    password_prompt: Annotated[
        str | None,
        Field(
            default=None,
            description="Custom prompt message for password input. "
            "Required if requires_password is true.",
        ),
    ] = None,
    is_high_risk: Annotated[
        bool,
        Field(
            default=False,
            description="Set to true for high-risk commands (e.g., sudo, rm -rf, "
            "system modifications). This adds extra warning in confirmation.",
        ),
    ] = False,
    requires_confirmation: Annotated[
        bool,
        Field(
            default=True,
            description="Set to false for read-only commands that don't need user "
            "confirmation (e.g., ls, cat, pwd, echo, grep). Write/update operations "
            "like rm, mv, sed -i will still require confirmation even if set to false.",
        ),
    ] = True,
    config: Annotated[RunnableConfig | None, InjectedToolArg] = None,
) -> str:
    """Execute a bash command.

    Note: User confirmation is handled by the request_confirmation tool.
    Password input uses elicitation with password format.
    """
    config = _ensure_config(config)

    stream_writer = _get_stream_writer(config)
    dry_run = _get_dry_run(config)

    return await execute_bash(
        command=command,
        working_dir=working_dir,
        timeout=timeout,
        requires_password=requires_password,
        password_prompt=password_prompt,
        is_high_risk=is_high_risk,
        requires_confirmation=requires_confirmation,
        stream_writer=stream_writer,
        dry_run=dry_run,
        config=config,
    )


async def execute_bash(
    command: str,
    working_dir: str | None,
    timeout: int,
    requires_password: bool,
    password_prompt: str | None,
    is_high_risk: bool,
    requires_confirmation: bool,
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
    elicitation_manager: ElicitationManager | None = config.get("configurable", {}).get(
        "elicitation_manager"
    )

    # Check if already aborted
    if _check_aborted(abort_signal):
        return "Error: Operation aborted."

    # Cap timeout at 10 minutes
    timeout = min(timeout, 600)

    # Auto-detect high-risk commands
    auto_high_risk, risk_reasons = _detect_high_risk_command(command)
    is_high_risk = is_high_risk or auto_high_risk

    # Auto-detect write commands - these always require confirmation
    is_write_command, write_reasons = _detect_write_command(command)

    # Determine if confirmation is actually needed:
    # - If requires_confirmation is True (default), always confirm
    # - If requires_confirmation is False but it's a write command, still confirm
    # - If requires_confirmation is False and not a write command, skip confirmation
    needs_confirmation = requires_confirmation or is_write_command

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
    if is_write_command:
        log_details["is_write_command"] = True
        log_details["write_reasons"] = write_reasons

    action_prefix = ""
    if dry_run:
        action_prefix = "[DRY RUN] "
    if is_high_risk:
        action_prefix += "[HIGH RISK] "
    if is_write_command and not is_high_risk:
        action_prefix += "[WRITE] "

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

    # Request user confirmation before executing the command (if needed)
    if needs_confirmation and elicitation_manager is not None:
        confirm_message = (
            f"The bash tool wants to execute the following command:\n\n"
            f"```bash\n{command}\n```"
        )
        if is_high_risk:
            confirm_message += f"\n\n⚠️ **High Risk**: {', '.join(risk_reasons)}"
        elif is_write_command:
            confirm_message += f"\n\n📝 **Write Operation**: {', '.join(write_reasons)}"

        confirm_schema = {
            "type": "object",
            "properties": {},
        }

        params = types.ElicitRequestFormParams(
            message=confirm_message,
            requestedSchema=confirm_schema,
        )

        logger.info("Requesting user confirmation for bash command: %s", command[:100])

        try:
            result = await elicitation_manager.request(
                params=params,
                writer=stream_writer,
                abort_signal=abort_signal,
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
                abort_signal=abort_signal,
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
                stdout, stderr = await wait_with_abort(
                    communicate_task, abort_signal, process, timeout
                )
            except AbortedError:
                return "Error: Operation aborted."
            except TimeoutError:
                kill_process_tree(process)
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
                stdout, stderr = await wait_with_abort(
                    communicate_task, abort_signal, process, timeout
                )
            except AbortedError:
                return "Error: Operation aborted."
            except TimeoutError:
                kill_process_tree(process)
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


async def wait_with_abort(
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
        kill_process_tree(process)
        raise AbortedError("Operation aborted")

    if timeout_task in done:
        # Timeout occurred - kill process group
        kill_process_tree(process)
        raise TimeoutError(f"Command timed out after {timeout}s")

    # Task completed successfully
    return task.result()


def kill_process_tree(process: asyncio.subprocess.Process) -> None:
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
