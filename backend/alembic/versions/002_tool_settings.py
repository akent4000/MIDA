"""Add tool_settings table

Revision ID: 002
Revises: 001
Create Date: 2026-04-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tool_settings",
        sa.Column("tool_id", sqlmodel.AutoString(), nullable=False),
        sa.Column("key", sqlmodel.AutoString(), nullable=False),
        sa.Column("value_json", sqlmodel.AutoString(), nullable=False, server_default="null"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("tool_id", "key"),
    )


def downgrade() -> None:
    op.drop_table("tool_settings")
