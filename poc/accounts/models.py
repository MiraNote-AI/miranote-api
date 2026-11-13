"""Tier 1 tables (design docs/specs/2026-11-06-accounts-and-subscriptions-
design.md, section 4.1). All ids are UUIDs, all times timestamptz.

The migrations in migrations/versions/ are the source of truth for the
schema; these models mirror them for the code, and
tests/test_migrations.py fails when the two drift apart.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any, Dict, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _ts(nullable: bool = False, now: bool = False) -> Any:
    """A timestamptz column; `now` gives it a server default of now()."""
    return mapped_column(
        DateTime(timezone=True),
        nullable=nullable,
        server_default=func.now() if now else None,
    )


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("apple_team_id", "apple_sub", name="uq_users_apple_team_sub"),)

    # Also sent to StoreKit as appAccountToken.
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    apple_sub: Mapped[str] = mapped_column(Text, nullable=False)
    apple_team_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = _ts(now=True)
    deletion_requested_at: Mapped[Optional[dt.datetime]] = _ts(nullable=True)


class Session(Base):
    """Our refresh tokens; one sign-in is one family (reuse detection)."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    family_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)  # SHA-256
    replaced_at: Mapped[Optional[dt.datetime]] = _ts(nullable=True)
    created_at: Mapped[dt.datetime] = _ts(now=True)
    expires_at: Mapped[dt.datetime] = _ts()
    revoked_at: Mapped[Optional[dt.datetime]] = _ts(nullable=True)


class Subscription(Base):
    """One row per Apple renewal chain. user_id null = unowned (offer code
    before sign-in, or a deleted account)."""

    __tablename__ = "subscriptions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'grace', 'billing_retry', 'expired', 'revoked')",
            name="ck_subscriptions_status",
        ),
        CheckConstraint("environment IN ('Sandbox', 'Production')", name="ck_subscriptions_environment"),
    )

    original_transaction_id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    linked_at: Mapped[Optional[dt.datetime]] = _ts(nullable=True)
    product_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[dt.datetime] = _ts()
    grace_period_expires_at: Mapped[Optional[dt.datetime]] = _ts(nullable=True)
    auto_renew: Mapped[bool] = mapped_column(Boolean, nullable=False)
    auto_renew_product_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    revoked_at: Mapped[Optional[dt.datetime]] = _ts(nullable=True)
    revocation_reason: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    offer_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    offer_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    environment: Mapped[str] = mapped_column(Text, nullable=False)
    last_signed_date: Mapped[dt.datetime] = _ts()


class AppStoreNotification(Base):
    """Audit and idempotency: Apple retries, a seen uuid is skipped."""

    __tablename__ = "app_store_notifications"

    notification_uuid: Mapped[str] = mapped_column(Text, primary_key=True)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    subtype: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_transaction_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    signed_date: Mapped[dt.datetime] = _ts()
    payload: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    processed_at: Mapped[dt.datetime] = _ts(now=True)


class AiUsage(Base):
    """One row per AI request; the only source of quotas. Never holds a
    prompt, text, image or result."""

    __tablename__ = "ai_usage"
    __table_args__ = (
        CheckConstraint(
            "feature IN ('chat', 'expand', 'image_generate', 'image_edit', "
            "'photo_push', 'mood', 'free_image_grant')",
            name="ck_ai_usage_feature",
        ),
        CheckConstraint("status IN ('pending', 'ok', 'failed')", name="ck_ai_usage_status"),
        CheckConstraint("environment IN ('Sandbox', 'Production')", name="ck_ai_usage_environment"),
        # The device key is set only on signed-out rows (4.1). No check that a
        # null user_id means mood: the purge (7.2) nulls user_id on every row.
        CheckConstraint("user_id IS NULL OR device_key_id IS NULL", name="ck_ai_usage_device_key_signed_out_only"),
        Index("ix_ai_usage_user_id_created_at", "user_id", "created_at"),
        Index("ix_ai_usage_device_key_id_created_at", "device_key_id", "created_at"),
    )

    # Sent by the client: a retry with the same id is not charged twice.
    request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    device_key_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    feature: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    units: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, server_default=text("0"))
    environment: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = _ts(now=True)


class AppAttestKey(Base):
    """App Attest device keys; deliberately not linked to a user."""

    __tablename__ = "app_attest_keys"
    __table_args__ = (
        CheckConstraint("environment IN ('development', 'production')", name="ck_app_attest_keys_environment"),
    )

    key_id: Mapped[str] = mapped_column(Text, primary_key=True)
    public_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    receipt: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    sign_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    environment: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = _ts(now=True)
    last_used_at: Mapped[dt.datetime] = _ts(now=True)
