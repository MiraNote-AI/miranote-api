"""Indexes for the cost kill switch and for revoking a sign-in

- ai_usage (feature, created_at): the kill switch (design 6.3) sums
  cost_usd per feature over the rolling 24 h of every user. Without it
  that query reads the whole table, which keeps 13 months of rows.
- sessions (family_id): reuse detection and sign-out revoke every token of
  one sign-in (design 3.2). sessions gains a row on every refresh and
  nothing prunes it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-13
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_ai_usage_feature_created_at", "ai_usage", ["feature", "created_at"])
    op.create_index("ix_sessions_family_id", "sessions", ["family_id"])


def downgrade() -> None:
    op.drop_index("ix_sessions_family_id", table_name="sessions")
    op.drop_index("ix_ai_usage_feature_created_at", table_name="ai_usage")
