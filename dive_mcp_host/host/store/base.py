from enum import StrEnum
from pathlib import Path
from typing import Protocol

from fastapi import UploadFile

from dive_mcp_host.host.helpers.context import ContextProtocol

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".rtf",
    ".odt",
    ".html",
    ".csv",
    ".epub",
}


class FileType(StrEnum):
    """File type."""

    IMAGE = "image"
    DOCUMENT = "document"
    OTHER = "other"

    @classmethod
    def from_file_path(cls, file_path: Path | str) -> "FileType":
        """Get the file type from the file path."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix in SUPPORTED_IMAGE_EXTENSIONS:
            return cls.IMAGE
        if file_path.suffix in SUPPORTED_DOCUMENT_EXTENSIONS:
            return cls.DOCUMENT
        return cls.OTHER


class StoreProtocol(ContextProtocol, Protocol):
    """Abstract base class for store operations."""

    async def save_file(
        self,
        file: UploadFile | str,
    ) -> Path | str | None:
        """Upload file to the store."""
        ...

    async def get_file(self, file_path: str) -> bytes:
        """Get the file from the store."""
        ...


class StoreManagerProtocol(ContextProtocol, Protocol):
    """Protocol for store manager operations."""

    async def upload_files(
        self, files: list[UploadFile | str]
    ) -> tuple[list[str], list[str]]:
        """Upload files to the store.

        Returns:
            tuple[list[str], list[str]]: The paths of the saved files.
        """
        ...

    async def save_files(
        self, files: list[UploadFile | str]
    ) -> list[tuple[FileType, list[str]]]:
        """Save files to the store.

        Returns:
            list[tuple[FileType, list[str]]]: The paths of the saved files.
        """
        ...

    def is_local_file(self, file_path: str) -> bool:
        """Check if the file is a local file."""
        ...

    def is_url(self, file_path: str) -> bool:
        """Check if the file is a URL."""
        ...

    def is_pdf(self, file_path: str) -> bool:
        """Check if the file is a PDF."""
        ...

    def is_text(self, file_path: str) -> bool:
        """Check if the file is a TEXT file."""
        ...

    async def get_image(self, file_path: str) -> str | None:
        """Get the base64 encoded image from the store.

        Returns None if file is not found
        """
        ...

    async def get_document(self, file_path: str) -> tuple[str | None, str | None]:
        """Get the base64 encoded document from the store.

        Args:
            file_path: The path to the document.

        Returns:
            tuple[str, str | None]: The base64 encoded document and the mime type.
        """
        ...

    async def get_document_text(self, file_path: str) -> str | None:
        """Get document text content from the store."""
        ...
