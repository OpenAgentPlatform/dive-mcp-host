import base64
import time
from hashlib import md5
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from random import randint

from fastapi import UploadFile
from PIL import Image

from dive_mcp_host.env import RESOURCE_DIR
from dive_mcp_host.host.store.base import StoreProtocol


class LocalStore(StoreProtocol):
    """Local store implementation."""

    def __init__(self, root_dir: Path = RESOURCE_DIR) -> None:
        """Initialize the local store."""
        upload_dir = root_dir / "upload"
        upload_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = upload_dir

    async def save_file(
        self,
        file: UploadFile | str,
    ) -> str | None:
        """Save a file to the local store."""
        if isinstance(file, str):
            return file

        if file.filename is None:
            return None

        ext = Path(file.filename).suffix

        tmp_name = f"{int(time.time() * 1000)}-{randint(0, int(1e9))}{ext}"  # noqa: S311
        tmp_file = self.upload_dir.joinpath(tmp_name)

        hash_md5 = md5(usedforsecurity=False)

        with tmp_file.open("wb") as f:
            while buf := await file.read():
                hash_md5.update(buf)
                f.write(buf)
        hash_str = hash_md5.hexdigest()[:12]
        dst_filename = self.upload_dir.joinpath(hash_str + "-" + file.filename)

        if existing_files := list(self.upload_dir.glob(hash_str + "*")):
            tmp_file.unlink()
            return str(existing_files[0])
        tmp_file.rename(dst_filename)
        return str(dst_filename)

    async def get_file(self, file_path: str | Path) -> bytes:
        """Get the file from the local store."""
        path = Path(file_path)
        return path.read_bytes()

    async def get_text_file(self, file_path: str | Path) -> str:
        """Read text content from file."""
        path = Path(file_path)
        return path.read_text("utf-8")

    async def get_image(self, file_path: str) -> str:
        """Get the base64 encoded image from the local store."""
        path = Path(file_path)

        image = Image.open(path)

        image.resize((800, 800))
        if image.mode in ["P", "RGBA"]:
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/jpeg;base64,{base64_image}"

    async def get_document(self, file_path: str) -> tuple[str, str | None]:
        """Get the base64 encoded document from the local store.

        Args:
            file_path: The path to the document.

        Returns:
            tuple[str, str | None]: The base64 encoded document and the mime type.
        """
        path = Path(file_path)

        with path.open("rb") as f:
            content = f.read()

        mime_type = guess_type(file_path)[0]
        return base64.b64encode(content).decode("utf-8"), mime_type
