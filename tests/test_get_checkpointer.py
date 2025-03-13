import pytest

from dive_mcp_host.host.helpers.checkpointer import get_checkpointer
from tests.helper import POSTGRES_URI, SQLITE_URI


@pytest.mark.asyncio
async def test_get_checkpointer() -> None:
    """Test the get checkpointer."""
    # SQLite
    async with get_checkpointer(SQLITE_URI) as _:
        pass

    # Postgres
    async with get_checkpointer(POSTGRES_URI) as _:
        pass
