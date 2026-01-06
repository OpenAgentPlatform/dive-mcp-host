"""Elicitation manager for MCP tools.

Manages elicitation requests and responses, bridging MCP servers and frontend.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from mcp import types
from pydantic import BaseModel

from dive_mcp_host.host.custom_events import ToolElicitationRequest
import contextlib

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ElicitationError(Exception):
    """Error raised when elicitation fails."""


class ElicitationTimeoutError(ElicitationError):
    """Error raised when elicitation times out."""


class ElicitationRequestInfo(BaseModel):
    """Information about an elicitation request."""

    message: str
    requested_schema: dict[str, Any]


class ElicitationManager:
    """Manages tool elicitation requests and responses.

    This manager bridges MCP server elicitation requests with frontend responses.
    It stores pending requests as asyncio.Future objects that block until
    the frontend responds.
    """

    DEFAULT_TIMEOUT: float = 600.0  # 10 minutes

    def __init__(self) -> None:
        """Initialize elicitation manager."""
        self._pending_requests: dict[str, asyncio.Future[types.ElicitResult]] = {}
        self._request_info: dict[str, ElicitationRequestInfo] = {}
        self._request_counter = 0

    async def request(
        self,
        params: types.ElicitRequestParams,
        writer: Callable[[tuple[str, Any]], None],
        timeout: float | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> types.ElicitResult:
        """Request elicitation from the user.

        Creates a request, sends the event to the frontend via writer,
        and waits for the response.

        Args:
            params: The elicitation request parameters from MCP.
            writer: Callback to send the elicitation event to frontend.
            timeout: Timeout in seconds. Defaults to DEFAULT_TIMEOUT.
            abort_signal: Optional event that when set, cancels the elicitation.

        Returns:
            ElicitResult from the frontend.

        Raises:
            ElicitationTimeoutError: If the request times out.
        """
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT

        message = params.message
        requested_schema: dict[str, Any] = params.requestedSchema  # type: ignore[assignment]

        request_id, future = self.create_request(message, requested_schema)

        event = (
            ToolElicitationRequest.NAME,
            ToolElicitationRequest(
                request_id=request_id,
                message=message,
                requested_schema=requested_schema,
            ),
        )

        logger.debug("Sending elicitation event: %s", request_id)
        writer(event)

        try:
            if abort_signal is not None:
                # Race between future completion and abort signal
                abort_task = asyncio.create_task(abort_signal.wait())
                future_task = asyncio.ensure_future(future)

                done, pending = await asyncio.wait(
                    [future_task, abort_task],
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                if not done:
                    # Timeout occurred
                    raise ElicitationTimeoutError(
                        f"Elicitation request {request_id} timed out after {timeout}s"
                    )

                # Check if abort was triggered
                if abort_task in done:
                    logger.debug("Elicitation request %s aborted", request_id)
                    return types.ElicitResult(action="cancel", content=None)  # type: ignore[arg-type]

                # Future completed
                return future_task.result()
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as e:
            raise ElicitationTimeoutError(
                f"Elicitation request {request_id} timed out after {timeout}s"
            ) from e
        except Exception:
            raise
        finally:
            self._cleanup_request(request_id)

    def _cleanup_request(self, request_id: str) -> None:
        """Clean up a request after timeout or error."""
        self._pending_requests.pop(request_id, None)
        self._request_info.pop(request_id, None)

    def create_request(
        self,
        message: str,
        requested_schema: dict[str, Any],
    ) -> tuple[str, asyncio.Future[types.ElicitResult]]:
        """Create a new elicitation request.

        Args:
            message: The message to display to the user.
            requested_schema: JSON schema for the requested data.

        Returns:
            Tuple of (request_id, future) where future will be resolved
            when the frontend responds.
        """
        request_id = f"elicit_{self._request_counter}"
        self._request_counter += 1
        future: asyncio.Future[types.ElicitResult] = asyncio.Future()
        self._pending_requests[request_id] = future
        self._request_info[request_id] = ElicitationRequestInfo(
            message=message,
            requested_schema=requested_schema,
        )
        logger.debug("Created elicitation request %s", request_id)
        return request_id, future

    def get_request_info(self, request_id: str) -> ElicitationRequestInfo | None:
        """Get information about an elicitation request.

        Args:
            request_id: ID of the request.

        Returns:
            ElicitationRequestInfo if found, None otherwise.
        """
        return self._request_info.get(request_id)

    def respond_to_request(
        self,
        request_id: str,
        action: str,
        content: dict[str, Any] | None = None,
    ) -> bool:
        """Respond to an elicitation request.

        Args:
            request_id: ID of the request to respond to.
            action: User's action ("accept", "decline", or "cancel").
            content: User-provided data (only for "accept" action).

        Returns:
            True if request was found and resolved, False otherwise.
        """
        if request_id not in self._pending_requests:
            logger.warning("Elicitation request %s not found", request_id)
            return False

        future = self._pending_requests.pop(request_id)
        self._request_info.pop(request_id, None)

        if future.done():
            logger.warning("Elicitation request %s already resolved", request_id)
            return False

        # Create ElicitResult based on action
        result = types.ElicitResult(
            action=action,  # type: ignore[arg-type]
            content=content,
        )

        # Use call_soon_threadsafe to ensure the result is set in the correct
        # event loop where the future was created. This is necessary because
        # this method may be called from a different async context than where
        # the future is being awaited.
        try:
            future_loop = future.get_loop()
            future_loop.call_soon_threadsafe(future.set_result, result)
        except RuntimeError:
            # Fallback if loop is closed or not running
            if not future.done():
                future.set_result(result)

        logger.debug(
            "Resolved elicitation request %s with action %s", request_id, action
        )
        return True

    def cancel_all_pending(self) -> None:
        """Cancel all pending elicitation requests.

        This should be called when a chat session ends or is aborted.
        """
        for request_id, future in list(self._pending_requests.items()):
            if not future.done():
                result = types.ElicitResult(action="cancel", content=None)  # type: ignore[arg-type]
                future.set_result(result)
                logger.debug("Cancelled elicitation request %s", request_id)
        self._pending_requests.clear()
        self._request_info.clear()
