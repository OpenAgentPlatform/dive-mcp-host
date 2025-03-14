"""auto_increment_msg_id.

Revision ID: 9513a23adc62
Revises: d74bc7569f6c
Create Date: 2025-03-13 17:34:23.204292

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9513a23adc62"
down_revision: str | None = "d74bc7569f6c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("messages") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.BIGINT(),
            type_=sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            existing_nullable=False,
            autoincrement=True,
        )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    # ### end Alembic commands ###
