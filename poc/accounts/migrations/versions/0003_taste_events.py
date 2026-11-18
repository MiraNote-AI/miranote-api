"""Tier 2: taste events and the switch that allows them (design 4.2)

- taste_events: choices only, never content. A plain table with an index
  on occurred_at for the 18-month retention delete; no partitions and no
  experiments column in 1.0. user_id cascades: a purged account takes its
  events with it.
- users.taste_events_enabled: "Use my choices to improve and personalize
  MiraNote", on by default.

Downgrading drops every user's switch with the events, so upgrading again
sets it back to on for everyone, also for users who had turned it off.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("taste_events_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.create_table(
        "taste_events",
        sa.Column("event_id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("app_version", sa.Text(), nullable=False),
        sa.Column("build", sa.Text(), nullable=False),
        sa.Column("tz_offset_minutes", sa.Integer(), nullable=False),
        sa.Column("event_version", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("surface", sa.Text(), nullable=False),
        sa.Column("item_type", sa.Text(), nullable=False),
        sa.Column("item_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("request_id", UUID(as_uuid=True), nullable=True),
        sa.CheckConstraint(
            "action IN ('shown', 'selected', 'kept', 'undone', 'deleted', 'regenerated')",
            name="ck_taste_events_action",
        ),
    )
    op.create_index("ix_taste_events_user_id", "taste_events", ["user_id"])
    op.create_index("ix_taste_events_occurred_at", "taste_events", ["occurred_at"])


def downgrade() -> None:
    op.drop_index("ix_taste_events_occurred_at", table_name="taste_events")
    op.drop_index("ix_taste_events_user_id", table_name="taste_events")
    op.drop_table("taste_events")
    op.drop_column("users", "taste_events_enabled")
