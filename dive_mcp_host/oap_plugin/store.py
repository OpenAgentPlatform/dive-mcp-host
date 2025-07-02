import mimetypes
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from logging import getLogger
from pathlib import Path

import httpx
from fastapi import UploadFile
from pydantic import BaseModel

from dive_mcp_host.host.store.base import StoreProtocol
from dive_mcp_host.oap_plugin.models import TokenNotSetError

logger = getLogger(__name__)


class UploadFileResponse(BaseModel):
    """Upload file response."""

    result: bool
    url: str | None = None
    file_id: str | None = None
    error: str | None = None


class OAPStore(StoreProtocol):
    """OAP Store."""

    def __init__(self) -> None:
        """Initialize the OAP Store."""
        self._token: str | None = None
        self._store_url: str | None = None
        self._verify_ssl: bool = True
        self._http_client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def _get_http_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Get the HTTP client."""
        if self._token is None:
            raise TokenNotSetError

        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._token}"},
            verify=self._verify_ssl,
        ) as client:
            try:
                yield client
            except Exception:
                logger.exception("OAPStore error")
                raise

    def _generate_file_name(self, original_name: str | None) -> str:
        """Generate file name."""
        if original_name is None:
            original_name = "noname"
        return f"{uuid.uuid4()}-{original_name}"

    async def save_file(self, file: UploadFile | str) -> str | None:
        """Save file to the store."""
        logger.debug("Saving file to the OAP store: %s", file)

        if isinstance(file, str):
            path = Path(file)
            if not path.exists():
                logger.warning("File does not exist: %s", path)
                return None

            guess_content_type = mimetypes.guess_type(path)[0]
            new_name = self._generate_file_name(path.name)
            logger.debug("Saving as file %s", new_name)
            files = {"file": (new_name, path.open("rb"), guess_content_type)}

        else:
            new_name = self._generate_file_name(file.filename)
            logger.debug("Saving as file %s", new_name)
            files = {"file": (new_name, file.file, file.content_type)}

        with suppress(TokenNotSetError):
            async with self._get_http_client() as client:
                response = await client.post(
                    f"{self._store_url}/upload_file",
                    files=files,
                )
                if not response.is_success:
                    logger.error(
                        "Failed to save file to the OAP store: %s", response.text
                    )
                    return None

                json_resp = response.json()
                result = UploadFileResponse.model_validate(json_resp)
                if result.result:
                    logger.debug("File saved to the OAP store: %s", result.url)
                    return result.url
        return None

    async def get_file(self, file_path: str) -> bytes:
        """Get file from the store."""
        logger.debug("Getting file from the OAP store: %s", file_path)
        async with self._get_http_client() as client:
            response = await client.get(file_path)
            return response.content

    def update_token(self, token: str | None) -> None:
        """Update the token."""
        self._token = token

    def update_store_url(self, store_url: str, verify_ssl: bool = True) -> None:
        """Update the store URL."""
        self._store_url = store_url
        self._verify_ssl = verify_ssl


oap_store = OAPStore()
