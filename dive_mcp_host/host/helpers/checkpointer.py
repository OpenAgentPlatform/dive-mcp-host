from contextlib import AbstractAsyncContextManager
from urllib.parse import unquote

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def get_checkpointer(
    uri: str,
) -> AbstractAsyncContextManager[AsyncSqliteSaver | AsyncPostgresSaver]:
    """Get an appropriate async checkpointer based on the database connection string.

    Args:
        uri (str): Database connection string, starting with either 'sqlite' or
        'postgres'

    Raises:
        ValueError: If the database type in the connection string is not supported

    Returns:
        AsyncIterator[BaseCheckpointSaver[V]]: An async checkpointer instance for the
        specified database
    """
    if uri.startswith("sqlite"):
        path = uri.removeprefix("sqlite:///")
        return AsyncSqliteSaver.from_conn_string(unquote(path))
    if uri.startswith("postgres"):
        return AsyncPostgresSaver.from_conn_string(uri)
    raise ValueError(f"Unsupported database: {uri}")
