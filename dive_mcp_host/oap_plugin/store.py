from fastapi import UploadFile

from dive_mcp_host.httpd.store.base import StoreProtocol


class OAPStore(StoreProtocol):
    """OAP Store."""

    async def save_file(self, file: UploadFile | str) -> str | None:
        """Save file to the store."""
        ...  # noqa: PIE790

    async def get_file(self, file_id: str) -> bytes:
        """Get file from the store."""
        ...  # noqa: PIE790

    async def update_token(self, token: str) -> None:
        """Update the token."""


oap_store = OAPStore()
