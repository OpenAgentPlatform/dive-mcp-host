from os import getenv

POSTGRES_URI = getenv("POSTGRES_URI", "postgres://mcp:mcp@localhost:5432/mcp")
SQLITE_URI = getenv("SQLITE_URI", "sqlite:///dummy_checkpointer.db")
