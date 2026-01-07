"""add_token_expiry_time.

Revision ID: faa40081e747
Revises: 29a47d5b8fbc
Create Date: 2026-01-07 11:56:26.522984

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "faa40081e747"
down_revision: str | None = "29a47d5b8fbc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("oauth_credentials") as batch_op:
        batch_op.add_column(
            sa.Column(
                "token_expiry_time",
                sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
                nullable=True,
            )
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("oauth_credentials") as batch_op:
        batch_op.drop_column("token_expiry_time")
