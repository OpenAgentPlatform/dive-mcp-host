import base64
import time
from hashlib import md5
from io import BytesIO
from pathlib import Path
from random import randint

from fastapi import UploadFile
from PIL import Image

from dive_mcp_host.httpd.conf.misc import RESOURCE_DIR

from .store import SUPPORTED_DOCUMENT_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS, Store


class LocalStore(Store):
    """Local store implementation."""

    def __init__(self, root_dir: Path = RESOURCE_DIR) -> None:
        """Initialize the local store."""
        upload_dir = root_dir / "upload"
        upload_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = upload_dir

    async def upload_files(
        self,
        files: list[UploadFile],
        file_paths: list[str],
    ) -> tuple[list[str], list[str]]:
        """Upload files to the local store."""
        images = []
        documents = []

        for file in files:
            if file.filename is None:
                continue

            ext = Path(file.filename).suffix
            tmp_name = (
                str(int(time.time() * 1000)) + "-" + str(randint(0, int(1e9))) + ext  # noqa: S311
            )
            upload_path = self.upload_dir.joinpath(tmp_name)
            hash_md5 = md5()  # noqa: S324
            with upload_path.open("wb") as f:
                while buf := await file.read():
                    hash_md5.update(buf)
                    f.write(buf)

            hash_str = hash_md5.hexdigest()[:12]
            dst_filename = self.upload_dir.joinpath(hash_str + "-" + file.filename)

            existing_files = list(self.upload_dir.glob(hash_str + "*"))
            if existing_files:
                upload_path.unlink()
            else:
                upload_path.rename(dst_filename)

            ext = ext.lower()

            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                images.append(str(dst_filename))
            elif ext in SUPPORTED_DOCUMENT_EXTENSIONS:
                documents.append(str(dst_filename))

        for file_path in file_paths:
            if Path(file_path).suffix in SUPPORTED_IMAGE_EXTENSIONS:
                images.append(file_path)
            elif Path(file_path).suffix in SUPPORTED_DOCUMENT_EXTENSIONS:
                documents.append(file_path)

        return images, documents

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
