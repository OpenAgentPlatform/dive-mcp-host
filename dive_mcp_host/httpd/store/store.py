from abc import ABC, abstractmethod

from fastapi import UploadFile

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


class Store(ABC):
    """Abstract base class for store operations."""

    @abstractmethod
    async def upload_files(
        self,
        files: list[UploadFile],
        file_paths: list[str],
    ) -> tuple[list[str], list[str]]:
        """Upload files to the store."""

    @abstractmethod
    async def get_image(self, file_path: str) -> str:
        """Get the base64 encoded image from the store."""

    @abstractmethod
    async def get_document(self, file_path: str) -> tuple[str, str | None]:
        """Get the base64 encoded document from the store.

        Args:
            file_path: The path to the document.

        Returns:
            tuple[str, str | None]: The base64 encoded document and the mime type.
        """
