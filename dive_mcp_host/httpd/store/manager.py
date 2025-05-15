import asyncio
import base64
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import AsyncExitStack
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

from fastapi import UploadFile
from PIL import Image

from dive_mcp_host.env import RESOURCE_DIR
from dive_mcp_host.httpd.store.base import FileType, StoreManagerProtocol, StoreProtocol
from dive_mcp_host.httpd.store.local import LocalStore
from dive_mcp_host.plugins.registry import HookInfo, PluginCallbackDef, PluginManager

type GetStoreCallback = Callable[[], Coroutine[Any, Any, StoreProtocol]]

StoreHookName = "dive_mcp_host.httpd.store"

IMAGE_MAX_SIZE = 800


class StoreManager(StoreManagerProtocol):
    """The storage manager."""

    def __init__(self, root_dir: Path = RESOURCE_DIR) -> None:
        """Initialize Storage manager.

        It always enables LocalStore.
        """
        super().__init__()
        self._local_store = LocalStore(root_dir)
        self._storage_callbacks: list[
            tuple[GetStoreCallback, PluginCallbackDef, str]
        ] = []
        self._storages: list[StoreProtocol] = []

    async def upload_files(
        self, files: list[UploadFile | str]
    ) -> tuple[list[str], list[str]]:
        """Upload files to the store.

        Returns:
            image and document storage locations.
            ex:
            (
                ["image_local_file_path", "image_remote_url" ...],
                ["document_local_file_path", "document_remote_url" ...],
            )
        """
        save_result = await self.save_files(files)
        images = [
            path
            for (typ, paths) in save_result
            if typ == FileType.IMAGE
            for path in paths
        ]
        documents = [
            path
            for (typ, paths) in save_result
            if typ != FileType.IMAGE
            for path in paths
        ]
        return images, documents

    async def _in_context(self) -> AsyncGenerator[Self, None]:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(self._local_store)
            for callback, _, _ in self._storage_callbacks:
                store = await callback()
                await stack.enter_async_context(store)
                self._storages.append(store)
            yield self

    async def save_files(
        self, files: list[UploadFile | str]
    ) -> list[tuple[FileType, list[str]]]:
        """Save files to the stores.

        Returns each file's file type and their list of storage locations.

        A file can be saved to multiple locations if plugins are registered.
        Locations might be local file paths, remote URLs, etc.

        The first location should be the local file path.
        """
        all_paths: list[tuple[FileType, list[str]]] = []
        for file in files:
            path = await self._local_store.save_file(file)
            if not path:
                continue
            paths = [path]
            if self._storage_callbacks:
                tasks: list[asyncio.Task] = []
                async with asyncio.TaskGroup() as tg:
                    for store in self._storages:
                        tasks.append(tg.create_task(store.save_file(file)))
                additional_paths = [i.result() for i in tasks if i.result()]
                paths.extend(additional_paths)
            all_paths.append((FileType.from_file_path(path), paths))
        return all_paths

    async def get_file(self, file_path: str | Path) -> bytes:
        """Get the file from the store."""
        return await self._local_store.get_file(file_path)

    def is_url(self, file_path: str) -> bool:
        """Check if the file is a URL."""
        result = urlparse(file_path)
        return bool(result.scheme and result.netloc)

    def is_local_file(self, file_path: str | Path) -> bool:
        """Check if the file is a local file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        return file_path.exists()

    async def get_image(self, file_path: str | Path) -> str:
        """Get the base64 encoded image from the store."""
        with (
            Image.open(BytesIO(await self.get_file(file_path))) as image,
            BytesIO() as buffer,
        ):
            if image.width > IMAGE_MAX_SIZE or image.height > IMAGE_MAX_SIZE:
                resized = image.resize((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE))
            else:
                resized = image
            if image.mode in ["P", "RGBA"]:
                resized = resized.convert("RGB")
            resized.save(buffer, format="JPEG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return f"data:image/jpeg;base64,{base64_image}"

    async def get_document(self, file_path: str) -> tuple[str, str | None]:
        """Get the base64 encoded document from the store.

        Args:
            file_path: The path to the document.

        Returns:
            tuple[str, str | None]: The base64 encoded document and the mime type.
        """
        mime_type = guess_type(file_path)[0]
        return base64.b64encode(await self.get_file(file_path)).decode(
            "utf-8"
        ), mime_type

    async def register_plugin(
        self,
        callback: GetStoreCallback,
        callback_def: PluginCallbackDef,
        plugin_name: str,
    ) -> bool:
        """Register the static plugin."""
        self._storage_callbacks.append((callback, callback_def, plugin_name))
        return True

    def register_hook(self, manager: PluginManager) -> None:
        """Register the hook."""
        manager.register_hookable(
            HookInfo(
                hook_name=StoreHookName,
                register=self.register_plugin,
            )
        )
