"""Tier 1 tables (design 2026-11-06, section 4.1)

users, sessions, subscriptions, app_store_notifications, ai_usage,
app_attest_keys.

Revision ID: 0001
Revises:
Create Date: 2026-11-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
TS = sa.DateTime(timezone=True)
NOW = sa.text("now()")
GEN_UUID = sa.text("gen_random_uuid()")  # built in since Postgres 13


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID, primary_key=True, server_default=GEN_UUID),
        sa.Column("apple_sub", sa.Text, nullable=False),
        sa.Column("apple_team_id", sa.Text, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("deletion_requested_at", TS, nullable=True),
        sa.UniqueConstraint("apple_team_id", "apple_sub", name="uq_users_apple_team_sub"),
    )

    op.create_table(
        "sessions",
        sa.Column("id", UUID, primary_key=True, server_default=GEN_UUID),
        sa.Column(
            "user_id", UUID, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("family_id", UUID, nullable=False),
        sa.Column("token_hash", sa.Text, nullable=False, unique=True),
        sa.Column("replaced_at", TS, nullable=True),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("expires_at", TS, nullable=False),
        sa.Column("revoked_at", TS, nullable=True),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "subscriptions",
        sa.Column("original_transaction_id", sa.Text, primary_key=True),
        # SET NULL: the purge pseudonymizes the chain so Restore can reclaim it.
        sa.Column(
            "user_id", UUID, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("linked_at", TS, nullable=True),
        sa.Column("product_id", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("expires_at", TS, nullable=False),
        sa.Column("grace_period_expires_at", TS, nullable=True),
        sa.Column("auto_renew", sa.Boolean, nullable=False),
        sa.Column("auto_renew_product_id", sa.Text, nullable=True),
        sa.Column("revoked_at", TS, nullable=True),
        sa.Column("revocation_reason", sa.Integer, nullable=True),
        sa.Column("offer_type", sa.Text, nullable=True),
        sa.Column("offer_id", sa.Text, nullable=True),
        sa.Column("environment", sa.Text, nullable=False),
        sa.Column("last_signed_date", TS, nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'grace', 'billing_retry', 'expired', 'revoked')",
            name="ck_subscriptions_status",
        ),
        sa.CheckConstraint(
            "environment IN ('Sandbox', 'Production')", name="ck_subscriptions_environment"
        ),
    )
    op.create_index("ix_subscriptions_user_id", "subscriptions", ["user_id"])

    op.create_table(
        "app_store_notifications",
        sa.Column("notification_uuid", sa.Text, primary_key=True),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("subtype", sa.Text, nullable=True),
        sa.Column("original_transaction_id", sa.Text, nullable=True),
        sa.Column("signed_date", TS, nullable=False),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("processed_at", TS, nullable=False, server_default=NOW),
    )

    op.create_table(
        "ai_usage",
        sa.Column("request_id", UUID, primary_key=True),
        sa.Column(
            "user_id", UUID, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True
        ),
        sa.Column("device_key_id", sa.Text, nullable=True),
        sa.Column("feature", sa.Text, nullable=False),
        sa.Column("model", sa.Text, nullable=False),
        sa.Column("units", sa.Integer, nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False, server_default=sa.text("0")),
        sa.Column("environment", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.CheckConstraint(
            "feature IN ('chat', 'expand', 'image_generate', 'image_edit', "
            "'photo_push', 'mood', 'free_image_grant')",
            name="ck_ai_usage_feature",
        ),
        # 'pending' is the row the guard inserts before the model call (6.1).
        sa.CheckConstraint("status IN ('pending', 'ok', 'failed')", name="ck_ai_usage_status"),
        sa.CheckConstraint(
            "environment IN ('Sandbox', 'Production')", name="ck_ai_usage_environment"
        ),
        # The device key is set only on signed-out rows. No check that a null
        # user_id means mood: the purge (7.2) nulls user_id on every row.
        sa.CheckConstraint(
            "user_id IS NULL OR device_key_id IS NULL",
            name="ck_ai_usage_device_key_signed_out_only",
        ),
    )
    # The quota windows (6.2): per user, and per device key when signed out.
    op.create_index("ix_ai_usage_user_id_created_at", "ai_usage", ["user_id", "created_at"])
    op.create_index(
        "ix_ai_usage_device_key_id_created_at", "ai_usage", ["device_key_id", "created_at"]
    )

    op.create_table(
        "app_attest_keys",
        sa.Column("key_id", sa.Text, primary_key=True),
        sa.Column("public_key", sa.LargeBinary, nullable=False),
        sa.Column("receipt", sa.LargeBinary, nullable=False),
        sa.Column("sign_count", sa.BigInteger, nullable=False),
        sa.Column("environment", sa.Text, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("last_used_at", TS, nullable=False, server_default=NOW),
        sa.CheckConstraint(
            "environment IN ('development', 'production')",
            name="ck_app_attest_keys_environment",
        ),
    )


def downgrade() -> None:
    op.drop_table("app_attest_keys")
    op.drop_table("ai_usage")
    op.drop_table("app_store_notifications")
    op.drop_table("subscriptions")
    op.drop_table("sessions")
    op.drop_table("users")
