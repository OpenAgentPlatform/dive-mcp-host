"""Command patterns for detecting write and high-risk operations.

This module provides pattern matching for detecting commands that perform
write operations or are considered high-risk, used by the bash tool.
"""

from __future__ import annotations

import re
import sys


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
