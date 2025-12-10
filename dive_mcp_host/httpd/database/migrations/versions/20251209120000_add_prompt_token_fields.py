"""add_prompt_token_fields.

Revision ID: a1b2c3d4e5f6
Revises: 9f8e7d6c5b4a
Create Date: 2025-12-09 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "9f8e7d6c5b4a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("resource_usage") as batch_op:
        batch_op.add_column(
            sa.Column(
                "custom_prompt_token",
                sa.BigInteger(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(
            sa.Column(
                "system_prompt_token",
                sa.BigInteger(),
                nullable=False,
                server_default="0",
            )
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("resource_usage") as batch_op:
        batch_op.drop_column("system_prompt_token")
        batch_op.drop_column("custom_prompt_token")
