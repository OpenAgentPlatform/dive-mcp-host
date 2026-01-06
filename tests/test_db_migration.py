from os import environ

import pytest
from alembic import command

from dive_mcp_host.httpd.database.migrate import db_migration

from .helper import POSTGRES_URI, SQLITE_URI


@pytest.mark.skipif(environ.get("POSTGRES_URI") is None, reason="No postgres setting")
def test_postgres_db_migration():
    """Test postgres database migration."""
    config = db_migration(
        POSTGRES_URI,
    )
    command.downgrade(config, "base")


def test_sqlite_db_migration():
    """Test sqlite database migration."""
    config = db_migration(
        SQLITE_URI,
    )
    command.downgrade(config, "base")
