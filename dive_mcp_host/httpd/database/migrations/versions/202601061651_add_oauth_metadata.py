"""add_oauth_metadata.

Revision ID: 29a47d5b8fbc
Revises: a1b2c3d4e5f6
Create Date: 2026-01-06 16:51:54.999504

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB as PGJSONB
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON  # noqa: N811

# revision identifiers, used by Alembic.
revision: str = "29a47d5b8fbc"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("oauth_credentials") as batch_op:
        batch_op.add_column(
            sa.Column(
                "oauth_metadata",
                PGJSONB().with_variant(SQLiteJSON(), "sqlite"),
                nullable=True,
            )
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("oauth_credentials") as batch_op:
        batch_op.drop_column("oauth_metadata")
