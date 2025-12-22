"""Approval manager for installer agent operations.

This module manages user approval requests for potentially dangerous operations
performed by the installer agent, such as executing bash commands, writing files,
or accessing unfamiliar URLs.
"""

# ruff: noqa: PLR0911
# PLR0911: _assess_risk_level needs many return statements for different risk cases

from __future__ import annotations

import asyncio
import fnmatch
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from dive_mcp_host.mcp_installer_plugin.events import (
    InstallerElicitationRequest,
    InstallerElicitationResponse,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ApprovalRule(BaseModel):
    """A rule for auto-approving operations."""

    operation_type: Literal["bash", "fetch", "write_file", "read_file"]
    """Type of operation this rule applies to."""

    pattern: str
    """Pattern to match against.

    For bash: matches the command
    For fetch: matches the URL
    For write_file/read_file: matches the file path
    """


class ApprovalError(Exception):
    """Error raised when approval fails."""


class ApprovalTimeoutError(ApprovalError):
    """Error raised when approval times out."""


class ApprovalDeniedError(ApprovalError):
    """Error raised when user denies the operation."""


@dataclass
class ApprovalRequestInfo:
    """Information about a pending approval request."""

    operation_type: Literal["bash", "fetch", "write_file", "read_file"]
    message: str
    details: dict[str, Any]
    risk_level: Literal["low", "medium", "high"]


@dataclass
class InstallerApprovalManager:
    """Manages approval requests for installer agent operations.

    This manager tracks which operations have been approved by the user
    and can auto-approve similar operations based on "allow_always" responses.
    """

    DEFAULT_TIMEOUT: float = 300.0  # 5 minutes

    # Auto-approval rules (patterns that are always allowed)
    _always_allowed: list[ApprovalRule] = field(default_factory=list)

    # Pending approval requests
    _pending_requests: dict[str, asyncio.Future[InstallerElicitationResponse]] = field(
        default_factory=dict
    )
    _request_info: dict[str, ApprovalRequestInfo] = field(default_factory=dict)
    _request_counter: int = 0

    def __post_init__(self) -> None:
        """Initialize default safe patterns."""
        # Default safe read patterns
        self._default_safe_reads = [
            "/etc/os-release",
            "~/.config/*",
            "~/.local/*",
            "**/package.json",
            "**/pyproject.toml",
            "**/README*",
            "**/.gitignore",
        ]

        # Default safe fetch domains
        self._default_safe_domains = [
            "github.com",
            "raw.githubusercontent.com",
            "pypi.org",
            "npmjs.com",
            "registry.npmjs.org",
        ]

        # Default safe bash commands (empty - all bash commands require user approval)
        self._default_safe_commands: list[str] = []

    def add_always_allowed(self, rule: ApprovalRule) -> None:
        """Add a rule to always allow certain operations."""
        self._always_allowed.append(rule)
        logger.debug("Added always-allowed rule: %s", rule)

    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if a value matches a pattern (supports glob-style wildcards)."""
        # Expand ~ to home directory conceptually for matching
        if pattern.startswith("~"):
            # This is for matching purposes, actual expansion happens elsewhere
            pattern = pattern.replace("~", "*")
        return fnmatch.fnmatch(value.lower(), pattern.lower())

    def _is_safe_read(self, path: str) -> bool:
        """Check if a file read operation is considered safe."""
        for pattern in self._default_safe_reads:
            if self._matches_pattern(path, pattern):
                return True
        return False

    def _is_safe_fetch(self, url: str) -> bool:
        """Check if a URL fetch is considered safe."""
        return any(domain in url for domain in self._default_safe_domains)

    def _is_safe_command(self, command: str) -> bool:
        """Check if a bash command is considered safe.

        All bash commands require user approval by default.
        """
        for pattern in self._default_safe_commands:
            if self._matches_pattern(command, pattern):
                return True
        return False

    def _check_always_allowed(
        self,
        operation_type: Literal["bash", "fetch", "write_file", "read_file"],
        value: str,
    ) -> bool:
        """Check if an operation is always allowed based on rules."""
        # Check default safe operations first
        if operation_type == "read_file" and self._is_safe_read(value):
            return True
        if operation_type == "fetch" and self._is_safe_fetch(value):
            return True
        if operation_type == "bash" and self._is_safe_command(value):
            return True

        # Check user-defined always-allowed rules
        for rule in self._always_allowed:
            if rule.operation_type != operation_type:
                continue
            if self._matches_pattern(value, rule.pattern):
                return True

        return False

    def _assess_risk_level(
        self,
        operation_type: Literal["bash", "fetch", "write_file", "read_file"],
        details: dict[str, Any],
    ) -> Literal["low", "medium", "high"]:
        """Assess the risk level of an operation."""
        if operation_type == "read_file":
            return "low"

        if operation_type == "fetch":
            url = details.get("url", "")
            if self._is_safe_fetch(url):
                return "low"
            return "medium"

        if operation_type == "write_file":
            path = details.get("path", "")
            # Writing to config files is medium risk
            if "config" in path.lower() or path.endswith(".json"):
                return "medium"
            return "high"

        if operation_type == "bash":
            command = details.get("command", "")
            # Check for dangerous patterns (exact match)
            dangerous_patterns = [
                "rm -rf",
                "rm -r",
                "sudo rm",
                "mkfs",
                "dd if=",
                "> /dev/",
                "chmod 777",
            ]
            for pattern in dangerous_patterns:
                if pattern in command:
                    return "high"

            # Any command with pipe is high risk
            if "|" in command:
                return "high"

            # Install commands are medium risk
            install_patterns = ["npm install", "pip install", "uv pip install", "apt"]
            for pattern in install_patterns:
                if pattern in command:
                    return "medium"

            if self._is_safe_command(command):
                return "low"

            return "medium"

        return "medium"

    async def request_approval(
        self,
        operation_type: Literal["bash", "fetch", "write_file", "read_file"],
        message: str,
        details: dict[str, Any],
        stream_writer: Callable[[tuple[str, Any]], None],
        timeout: float | None = None,
    ) -> bool:
        """Request user approval for an operation.

        Args:
            operation_type: Type of operation.
            message: Human-readable description of the operation.
            details: Additional details about the operation.
            stream_writer: Callback to send events to the frontend.
            timeout: Timeout in seconds.

        Returns:
            True if approved, False otherwise.

        Raises:
            ApprovalTimeoutError: If the request times out.
            ApprovalDeniedError: If the user denies the operation.
        """
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT

        # Determine what value to check for auto-approval
        check_value = ""
        if operation_type == "bash":
            check_value = details.get("command", "")
        elif operation_type == "fetch":
            check_value = details.get("url", "")
        elif operation_type in ("write_file", "read_file"):
            check_value = details.get("path", "")

        # Check if this operation is always allowed
        if self._check_always_allowed(operation_type, check_value):
            logger.debug(
                "Auto-approved %s operation: %s", operation_type, check_value[:100]
            )
            return True

        # Need user approval
        risk_level = self._assess_risk_level(operation_type, details)
        request_id = f"installer_approval_{self._request_counter}"
        self._request_counter += 1

        future: asyncio.Future[InstallerElicitationResponse] = asyncio.Future()
        self._pending_requests[request_id] = future
        self._request_info[request_id] = ApprovalRequestInfo(
            operation_type=operation_type,
            message=message,
            details=details,
            risk_level=risk_level,
        )

        # Send elicitation request to frontend
        event = (
            InstallerElicitationRequest.NAME,
            InstallerElicitationRequest(
                request_id=request_id,
                operation_type=operation_type,
                message=message,
                details=details,
                risk_level=risk_level,
            ),
        )
        stream_writer(event)

        logger.debug("Waiting for approval: %s (risk: %s)", request_id, risk_level)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as e:
            self._cleanup_request(request_id)
            raise ApprovalTimeoutError(
                f"Approval request {request_id} timed out after {timeout}s"
            ) from e
        finally:
            self._cleanup_request(request_id)

        if response.action == "deny":
            raise ApprovalDeniedError(f"User denied operation: {message}")

        if response.action == "allow_always":
            # Add rule to always allow similar operations
            self.add_always_allowed(
                ApprovalRule(
                    operation_type=operation_type,
                    pattern=check_value,
                )
            )

        return True

    def respond_to_request(
        self,
        request_id: str,
        action: Literal["allow", "allow_always", "deny"],
    ) -> bool:
        """Respond to an approval request.

        Args:
            request_id: ID of the request to respond to.
            action: User's decision.

        Returns:
            True if request was found and resolved, False otherwise.
        """
        if request_id not in self._pending_requests:
            logger.warning("Approval request %s not found", request_id)
            return False

        future = self._pending_requests[request_id]
        if future.done():
            logger.warning("Approval request %s already resolved", request_id)
            return False

        response = InstallerElicitationResponse(action=action)
        future.set_result(response)
        logger.debug("Resolved approval request %s with action %s", request_id, action)
        return True

    def _cleanup_request(self, request_id: str) -> None:
        """Clean up a request."""
        self._pending_requests.pop(request_id, None)
        self._request_info.pop(request_id, None)

    def cancel_all_pending(self) -> None:
        """Cancel all pending approval requests."""
        for request_id, future in list(self._pending_requests.items()):
            if not future.done():
                response = InstallerElicitationResponse(action="deny")
                future.set_result(response)
                logger.debug("Cancelled approval request %s", request_id)
        self._pending_requests.clear()
        self._request_info.clear()
