"""Initial schema: studies + inference_results

Revision ID: 001
Revises:
Create Date: 2026-04-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "studies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("file_format", sqlmodel.AutoString(), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("anonymized", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("file_key", sqlmodel.AutoString(), nullable=False),
        sa.Column("metadata_json", sqlmodel.AutoString(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "inference_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tool_id", sqlmodel.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.AutoString(), nullable=False, server_default="pending"),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sqlmodel.AutoString(), nullable=False),
        sa.Column("result_json", sqlmodel.AutoString(), nullable=True),
        sa.Column("gradcam_key", sqlmodel.AutoString(), nullable=True),
        sa.Column("error_message", sqlmodel.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("inference_results")
    op.drop_table("studies")
