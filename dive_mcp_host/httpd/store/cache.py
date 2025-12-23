from enum import StrEnum
from logging import getLogger
from pathlib import Path

from dive_mcp_host.env import RESOURCE_DIR

logger = getLogger(__name__)


class CacheKeys(StrEnum):
    """Cache keys for the MCP host."""

    LIST_TOOLS = "list_tools"


class LocalFileCache:
    """Local file cache for the MCP host."""

    def __init__(
        self,
        root_dir: Path = RESOURCE_DIR,
        cache_file_prefix: str = "dive_mcp_host",
    ) -> None:
        """Initialize the local file cache.

        Args:
            root_dir: The root directory for the config
            cache_file_prefix: The prefix of the cache file.

        """
        self._cache_file_prefix = cache_file_prefix
        self._cache_dir = root_dir / "cache"

        logger.info("LocalCache directory: %s", self._cache_dir)
        logger.info("LocalCache file prefix: %s", self._cache_file_prefix)
        self._cache_dir.mkdir(mode=0o744, parents=True, exist_ok=True)
        logger.info("LocalCache directory prepared")

    def get_cache_file_path(self, key: CacheKeys, extension: str = "json") -> Path:
        """Get the cache file name.

        Args:
            key: The key of the cache.
            extension: The extension of the cache file.

        Returns:
            The cache file name.
        """
        return self._cache_dir / f"{self._cache_file_prefix}_{key.value}.{extension}"

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory."""
        return self._cache_dir

    def get(self, key: CacheKeys, extension: str = "json") -> str | None:
        """Get the value of the key.

        Args:
            key: The key of the cache.
            extension: The extension of the cache file.

        Returns:
            The value of the key.
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        if not cache_file_path.exists():
            return None
        try:
            with cache_file_path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:  # noqa: BLE001
            logger.error("load cache failed: %s", e)

        return None

    def set(self, key: CacheKeys, value: str, extension: str = "json") -> None:
        """Set the value of the key.

        Args:
            key: The key of the cache.
            value: The value of the key.
            extension: The extension of the cache file.

        Returns:
            None
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        with cache_file_path.open("w", encoding="utf-8") as f:
            f.write(value)

    def delete(self, key: CacheKeys, extension: str = "json") -> None:
        """Delete cache file.

        Args:
            key: The key of the cache.
            extension: The extension of the cache file.

        Returns:
            None
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        if cache_file_path.exists():
            cache_file_path.unlink()
