"""Full text search.

Revision ID: 74c3ff8fbf62
Revises: faa40081e747
Create Date: 2026-02-02 11:32:23.170448

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "74c3ff8fbf62"
down_revision: str | None = "faa40081e747"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    engine_name = op.get_bind().engine.name

    if engine_name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        op.create_index(
            "idx_chats_title_trigram",
            "chats",
            ["title"],
            postgresql_using="gin",
            postgresql_ops={"title": "gin_trgm_ops"},
            if_not_exists=True,
        )
        op.create_index(
            "idx_messages_content_trigram",
            "messages",
            ["content"],
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
            if_not_exists=True,
        )

    elif engine_name == "sqlite":
        op.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                chat_id,
                title,
                content,
                tokenize='trigram'
            );
        """)

        op.execute("""
            INSERT INTO message_fts(rowid, chat_id, title, content)
            SELECT m.rowid, m.chat_id, c.title, m.content
            FROM messages m JOIN chats c ON m.chat_id = c.id;
        """)

        op.execute("""
            CREATE TRIGGER IF NOT EXISTS message_ai AFTER INSERT ON messages BEGIN
              INSERT INTO message_fts(rowid, chat_id, title, content)
              SELECT new.rowid, new.chat_id, c.title, new.content
              FROM chats c WHERE c.id = new.chat_id;
            END;
        """)

        op.execute("""
            CREATE TRIGGER IF NOT EXISTS message_ad AFTER DELETE ON messages BEGIN
              DELETE FROM message_fts WHERE rowid = old.rowid;
            END;
        """)

        op.execute("""
            CREATE TRIGGER IF NOT EXISTS message_au AFTER UPDATE ON messages BEGIN
              DELETE FROM message_fts WHERE rowid = old.rowid;
              INSERT INTO message_fts(rowid, chat_id, title, content)
              SELECT new.rowid, new.chat_id, c.title, new.content
              FROM chats c WHERE c.id = new.chat_id;
            END;
        """)

        op.execute("""
            CREATE TRIGGER IF NOT EXISTS chat_au AFTER UPDATE ON chats BEGIN
              DELETE FROM message_fts WHERE chat_id = old.id;
              INSERT INTO message_fts(rowid, chat_id, title, content)
              SELECT m.rowid, m.chat_id, new.title, m.content
              FROM messages m WHERE m.chat_id = new.id;
            END;
        """)

        op.execute("""
            CREATE TRIGGER IF NOT EXISTS chat_bd BEFORE DELETE ON chats BEGIN
              DELETE FROM message_fts WHERE chat_id = old.id;
            END;
        """)


def downgrade() -> None:
    """Downgrade schema."""
    engine_name = op.get_bind().engine.name

    if engine_name == "postgresql":
        op.drop_index(
            "idx_messages_content_trigram",
            table_name="messages",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
            if_exists=True,
        )
        op.drop_index(
            "idx_chats_title_trigram",
            table_name="chats",
            postgresql_using="gin",
            postgresql_ops={"title": "gin_trgm_ops"},
            if_exists=True,
        )

    elif engine_name == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS chat_bd")
        op.execute("DROP TRIGGER IF EXISTS chat_au")
        op.execute("DROP TRIGGER IF EXISTS message_ai")
        op.execute("DROP TRIGGER IF EXISTS message_ad")
        op.execute("DROP TRIGGER IF EXISTS message_au")
        op.drop_table("message_fts", if_exists=True)
