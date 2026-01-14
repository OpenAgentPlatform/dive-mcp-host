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
import re
import signal
import sys
from collections.abc import Callable  # noqa: TC003
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import ArgsSchema, BaseTool, InjectedToolArg, tool
from pydantic import BaseModel, Field, SkipValidation

from dive_mcp_host.mcp_installer_plugin.events import InstallerToolLog
from dive_mcp_host.mcp_installer_plugin.tools.common import (
    AbortedError,
    _check_aborted,
    _ensure_config,
    _get_abort_signal,
    _get_dry_run,
    _get_stream_writer,
)

logger = logging.getLogger(__name__)


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
    requires_confirmation: Annotated[
        bool,
        Field(
            default=True,
            description="Set to false for read-only commands that don't need user "
            "confirmation (e.g., ls, cat, pwd, echo, grep). Write/update operations "
            "like rm, mv, sed -i will still require confirmation even if set to false.",
        ),
    ] = True


def _compile_patterns(
    patterns: list[tuple[str, str]], flags: int = 0
) -> list[tuple[re.Pattern[str], str]]:
    """Compile a list of (pattern, reason) tuples into (compiled_pattern, reason)."""
    return [(re.compile(p, flags), r) for p, r in patterns]


# Cross-platform commands (Git, Docker, Kubernetes, package managers, etc.)
_COMMON_WRITE_PATTERNS = _compile_patterns(
    [
        (r"\bgit\s+push", "Git push"),
        (r"\bgit\s+commit", "Git commit"),
        (r"\bgit\s+reset", "Git reset"),
        (r"\bgit\s+checkout\s", "Git checkout"),
        (r"\bgit\s+merge", "Git merge"),
        (r"\bgit\s+rebase", "Git rebase"),
        (r"\bgit\s+cherry-pick", "Git cherry-pick"),
        (r"\bgit\s+revert", "Git revert"),
        (r"\bgit\s+stash", "Git stash"),
        (r"\bgit\s+clean", "Git clean"),
        (r"\bgit\s+rm\s", "Git remove"),
        (r"\bgit\s+mv\s", "Git move"),
        (r"\bnpm\s+install", "Package install (npm)"),
        (r"\bnpm\s+uninstall", "Package uninstall (npm)"),
        (r"\bnpm\s+update", "Package update (npm)"),
        (r"\byarn\s+add", "Package install (yarn)"),
        (r"\byarn\s+remove", "Package remove (yarn)"),
        (r"\bpnpm\s+add", "Package install (pnpm)"),
        (r"\bpnpm\s+remove", "Package remove (pnpm)"),
        (r"\bpip\s+install", "Package install (pip)"),
        (r"\bpip\s+uninstall", "Package uninstall (pip)"),
        (r"\buv\s+pip\s+install", "Package install (uv pip)"),
        (r"\buv\s+add", "Package install (uv)"),
        (r"\buv\s+remove", "Package remove (uv)"),
        (r"\bcargo\s+install", "Package install (cargo)"),
        (r"\bgem\s+install", "Package install (gem)"),
        (r"\bgo\s+install", "Package install (go)"),
        (r"\bdocker\s+rm\s", "Docker remove"),
        (r"\bdocker\s+rmi\s", "Docker image remove"),
        (r"\bdocker\s+stop\s", "Docker stop"),
        (r"\bdocker\s+kill\s", "Docker kill"),
        (r"\bdocker\s+run\s", "Docker run"),
        (r"\bdocker\s+build\s", "Docker build"),
        (r"\bdocker\s+push\s", "Docker push"),
        (r"\bdocker\s+pull\s", "Docker pull"),
        (r"\bdocker\s+compose\s+up", "Docker compose up"),
        (r"\bdocker\s+compose\s+down", "Docker compose down"),
        (r"\bkubectl\s+apply", "Kubernetes apply"),
        (r"\bkubectl\s+delete", "Kubernetes delete"),
        (r"\bkubectl\s+create", "Kubernetes create"),
        (r"\bkubectl\s+patch", "Kubernetes patch"),
        (r"\bkubectl\s+replace", "Kubernetes replace"),
        (r"\bkubectl\s+set", "Kubernetes set"),
        (r"\bkubectl\s+scale", "Kubernetes scale"),
        (r"\bkubectl\s+rollout", "Kubernetes rollout"),
        (
            r"\bmysql\s+.*-e\s+['\"]?(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)",
            "MySQL write operation",
        ),
        (
            r"\bpsql\s+.*-c\s+['\"]?(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)",
            "PostgreSQL write operation",
        ),
        (r"\bmongosh?\s+.*--(eval|file)", "MongoDB operation"),
        (r"\bredis-cli\s+.*(SET|DEL|FLUSHALL|FLUSHDB)", "Redis write operation"),
        (r"\btar\s+.*-[xc]", "Archive operation (tar)"),
        (r"\btar\s+.*--extract", "Archive extraction (tar)"),
        (r"\btar\s+.*--create", "Archive creation (tar)"),
        (r"\bunzip\s", "Archive extraction (unzip)"),
        (r"\bgunzip\s", "Decompression (gunzip)"),
        (r"\bbunzip2\s", "Decompression (bunzip2)"),
        (r"\bxz\s+-d", "Decompression (xz)"),
        (r"\b7z\s+[xea]", "Archive operation (7z)"),
        (r"\bpython3?\s+-c\s", "Python code execution (python -c)"),
        (r"\bpython3?\s+-m\s", "Python module execution (python -m)"),
        (r"\bperl\s+-e\s", "Perl code execution (perl -e)"),
        (r"\bruby\s+-e\s", "Ruby code execution (ruby -e)"),
        (r"\bnode\s+-e\s", "Node.js code execution (node -e)"),
        (r"\$\(", "Command substitution $()"),
        (r"`[^`]+`", "Command substitution ``"),
    ],
    re.IGNORECASE,
)

# Unix/Linux specific commands (case-sensitive)
_UNIX_WRITE_PATTERNS = _compile_patterns(
    [
        (r"\brm\s", "File deletion (rm)"),
        (r"\brmdir\s", "Directory deletion (rmdir)"),
        (r"\bunlink\s", "File deletion (unlink)"),
        (r"\bmv\s", "File move/rename (mv)"),
        (r"\bcp\s", "File copy (cp)"),
        (r"\binstall\s", "File install (install)"),
        (r"\brsync\s", "File sync (rsync)"),
        (r"\bscp\s", "Secure copy (scp)"),
        (r"\bsed\s+-i", "In-place edit (sed -i)"),
        (r"\bsed\s+--in-place", "In-place edit (sed --in-place)"),
        (r"\bawk\s+-i", "In-place edit (awk -i)"),
        (r"\bpatch\s", "Apply patch (patch)"),
        (r"\bchmod\s", "Permission change (chmod)"),
        (r"\bchown\s", "Ownership change (chown)"),
        (r"\bchgrp\s", "Group change (chgrp)"),
        (r"\btouch\s", "File creation/modification (touch)"),
        (r"\bmkdir\s", "Directory creation (mkdir)"),
        (r"\bln\s", "Link creation (ln)"),
        (r"\bmkfifo\s", "FIFO creation (mkfifo)"),
        (r"\bmknod\s", "Node creation (mknod)"),
        (r"\btee\s", "Write to file (tee)"),
        (r"\btruncate\s", "File truncation (truncate)"),
        (r"\bdd\s", "Disk/file operations (dd)"),
        (r"\bmkfs", "Filesystem creation (mkfs)"),
        (r"\bfdisk\s", "Disk partitioning (fdisk)"),
        (r"\bparted\s", "Disk partitioning (parted)"),
        (r"\bbrew\s+install", "Package install (brew)"),
        (r"\bbrew\s+uninstall", "Package uninstall (brew)"),
        (r"\bapt\s+install", "Package install (apt)"),
        (r"\bapt\s+remove", "Package remove (apt)"),
        (r"\bapt-get\s+install", "Package install (apt-get)"),
        (r"\bapt-get\s+remove", "Package remove (apt-get)"),
        (r"\byum\s+install", "Package install (yum)"),
        (r"\byum\s+remove", "Package remove (yum)"),
        (r"\bdnf\s+install", "Package install (dnf)"),
        (r"\bdnf\s+remove", "Package remove (dnf)"),
        (r"\bpacman\s+-S", "Package install (pacman)"),
        (r"\bpacman\s+-R", "Package remove (pacman)"),
        (
            r"\bsystemctl\s+(start|stop|restart|enable|disable)",
            "Systemd service control",
        ),
        (r"\bservice\s+\w+\s+(start|stop|restart)", "Service control"),
        (r"\bkill\s", "Process termination (kill)"),
        (r"\bkillall\s", "Process termination (killall)"),
        (r"\bpkill\s", "Process termination (pkill)"),
        (r"\bwget\s+.*-O\s", "Download to file (wget -O)"),
        (r"\bwget\s+.*--output-document", "Download to file (wget)"),
        (r"\bcurl\s+.*-o\s", "Download to file (curl -o)"),
        (r"\bcurl\s+.*--output\s", "Download to file (curl)"),
        (r"\bcurl\s+.*-O\s", "Download to file (curl -O)"),
        (r"\bxargs\s", "Command execution via xargs"),
        (r"\bbash\s+-c\s", "Shell command execution (bash -c)"),
        (r"\bsh\s+-c\s", "Shell command execution (sh -c)"),
        (r"\bzsh\s+-c\s", "Shell command execution (zsh -c)"),
        (r"\beval\s", "Command evaluation (eval)"),
        (r"\bexec\s", "Command execution (exec)"),
        (r"\bsource\s", "Script sourcing (source)"),
        (r"^\.", "Script sourcing (.)"),
        (r"\|\s*sh\b", "Piped shell execution (| sh)"),
        (r"\|\s*bash\b", "Piped shell execution (| bash)"),
    ]
)

# Windows specific commands (case-insensitive)
_WINDOWS_WRITE_PATTERNS = _compile_patterns(
    [
        (r"\bdel\s", "File deletion (del)"),
        (r"\berase\s", "File deletion (erase)"),
        (r"\brd\s", "Directory deletion (rd)"),
        (r"\brmdir\s", "Directory deletion (rmdir)"),
        (r"\bmove\s", "File move (move)"),
        (r"\bcopy\s", "File copy (copy)"),
        (r"\bxcopy\s", "File copy (xcopy)"),
        (r"\brobocopy\s", "File copy (robocopy)"),
        (r"\bren\s", "File rename (ren)"),
        (r"\brename\s", "File rename (rename)"),
        (r"\bmd\s", "Directory creation (md)"),
        (r"\bmkdir\s", "Directory creation (mkdir)"),
        (r"\battrib\s", "Attribute change (attrib)"),
        (r"\bicacls\s", "Permission change (icacls)"),
        (r"\bcacls\s", "Permission change (cacls)"),
        (r"\btakeown\s", "Ownership change (takeown)"),
        (r"\breg\s+add", "Registry add (reg add)"),
        (r"\breg\s+delete", "Registry delete (reg delete)"),
        (r"\breg\s+import", "Registry import (reg import)"),
        (r"\bregedit\s+/s", "Registry edit (regedit)"),
        (r"\bformat\s", "Disk format (format)"),
        (r"\bdiskpart", "Disk partitioning (diskpart)"),
        (r"\bchkdsk\s+.*(/f|/r|/x)", "Disk repair (chkdsk)"),
        (r"Remove-Item\s", "PowerShell remove (Remove-Item)"),
        (r"New-Item\s", "PowerShell create (New-Item)"),
        (r"Set-Content\s", "PowerShell write (Set-Content)"),
        (r"Add-Content\s", "PowerShell append (Add-Content)"),
        (r"Out-File\s", "PowerShell write (Out-File)"),
        (r"Copy-Item\s", "PowerShell copy (Copy-Item)"),
        (r"Move-Item\s", "PowerShell move (Move-Item)"),
        (r"Rename-Item\s", "PowerShell rename (Rename-Item)"),
        (r"Clear-Content\s", "PowerShell clear (Clear-Content)"),
        (r"Set-ItemProperty\s", "PowerShell property set (Set-ItemProperty)"),
        (r"Remove-ItemProperty\s", "PowerShell property remove (Remove-ItemProperty)"),
        (r"New-ItemProperty\s", "PowerShell property create (New-ItemProperty)"),
        (r"Set-Acl\s", "PowerShell ACL set (Set-Acl)"),
        (r"Stop-Process\s", "PowerShell process stop (Stop-Process)"),
        (r"Start-Process\s", "PowerShell process start (Start-Process)"),
        (r"Stop-Service\s", "PowerShell service stop (Stop-Service)"),
        (r"Start-Service\s", "PowerShell service start (Start-Service)"),
        (r"Restart-Service\s", "PowerShell service restart (Restart-Service)"),
        (r"Install-Module\s", "PowerShell module install (Install-Module)"),
        (r"Uninstall-Module\s", "PowerShell module uninstall (Uninstall-Module)"),
        (r"Install-Package\s", "PowerShell package install (Install-Package)"),
        (r"Uninstall-Package\s", "PowerShell package uninstall (Uninstall-Package)"),
        (r"\bwinget\s+install", "Package install (winget)"),
        (r"\bwinget\s+uninstall", "Package uninstall (winget)"),
        (r"\bwinget\s+upgrade", "Package upgrade (winget)"),
        (r"\bchoco\s+install", "Package install (chocolatey)"),
        (r"\bchoco\s+uninstall", "Package uninstall (chocolatey)"),
        (r"\bscoop\s+install", "Package install (scoop)"),
        (r"\bscoop\s+uninstall", "Package uninstall (scoop)"),
        (r"\bschtasks\s+/create", "Scheduled task create (schtasks)"),
        (r"\bschtasks\s+/delete", "Scheduled task delete (schtasks)"),
        (r"\bsc\s+create", "Service create (sc)"),
        (r"\bsc\s+delete", "Service delete (sc)"),
        (r"\bsc\s+stop", "Service stop (sc)"),
        (r"\bsc\s+start", "Service start (sc)"),
        (r"\bnet\s+stop", "Service stop (net)"),
        (r"\bnet\s+start", "Service start (net)"),
        (r"\bnet\s+user\s+\w+\s+", "User management (net user)"),
        (r"\bnet\s+localgroup\s+.*/(add|delete)", "Group management (net localgroup)"),
        (r"\bwmic\s+.*delete", "WMI delete (wmic)"),
        (r"\bwmic\s+.*call", "WMI call (wmic)"),
        (r"\btaskkill\s", "Process termination (taskkill)"),
        (r"\bcmd\s+/c\s", "CMD command execution (cmd /c)"),
        (r"\bcmd\.exe\s+/c\s", "CMD command execution (cmd.exe /c)"),
        (r"\bpowershell\s+-c", "PowerShell command execution (powershell -c)"),
        (r"\bpowershell\s+-Command", "PowerShell command execution"),
        (r"\bpowershell\.exe\s+-c", "PowerShell command execution"),
        (r"\bpwsh\s+-c", "PowerShell Core command execution (pwsh -c)"),
        (r"Invoke-Expression\s", "PowerShell Invoke-Expression"),
        (r"\biex\s", "PowerShell iex (Invoke-Expression)"),
        (r"Invoke-Command\s", "PowerShell Invoke-Command"),
        (r"\bfor\s+/f\s", "CMD for /f command execution"),
        (r"\bforfiles\s", "CMD forfiles command execution"),
        (r"\bwscript\s", "Windows Script Host (wscript)"),
        (r"\bcscript\s", "Windows Script Host (cscript)"),
        (r"\bmshta\s", "HTML Application Host (mshta)"),
    ],
    re.IGNORECASE,
)

_REDIRECT_PATTERN = re.compile(r"[^-=]>")


def _detect_write_command(command: str) -> tuple[bool, list[str]]:
    """Detect if a command performs write/update operations.

    Returns:
        Tuple of (is_write_command, list of reasons).
    """
    reasons = []
    command_stripped = command.strip()

    if (
        ">" in command_stripped
        and not command_stripped.startswith("#")
        and (
            _REDIRECT_PATTERN.search(command_stripped)
            or command_stripped.startswith(">")
        )
    ):
        reasons.append("Output redirection to file (>)")

    for pattern, reason in _COMMON_WRITE_PATTERNS:
        if pattern.search(command):
            if reason not in reasons:
                reasons.append(reason)
            break

    if sys.platform != "win32":
        for pattern, reason in _UNIX_WRITE_PATTERNS:
            if pattern.search(command):
                if reason not in reasons:
                    reasons.append(reason)
                break
    else:
        for pattern, reason in _WINDOWS_WRITE_PATTERNS:
            if pattern.search(command):
                if reason not in reasons:
                    reasons.append(reason)
                break

    return len(reasons) > 0, reasons


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
    args_schema: Annotated[ArgsSchema | None, SkipValidation] = BashInput

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
        dry_run = _get_dry_run(config)

        return await self._execute_bash(
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
                f"The bash tool wants to execute the following command:\n\n"
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
