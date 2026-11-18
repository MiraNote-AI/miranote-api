"""Keeping subscriptions current (design 5).

    POST /subscriptions/sync       app     {signed_transaction, signed_renewal_info}
    POST /appstore/notifications   Apple   {signedPayload}  (Notifications V2)

Every input is Apple's signed snapshot of one renewal chain; the row is
recomputed from it (5.1), never moved through a per-type state machine.
Entitlement is read from the rows by quotas.entitlement.

Both routes answer 503 until Apple's root certificates are configured
(config.py), so the service runs before the App Store keys exist.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

import appstore
import quotas
from auth import current_user
from models import AppStoreNotification, Subscription, User

log = logging.getLogger("accounts.subscriptions")

router = APIRouter()

subscriptions = Subscription.__table__
notifications = AppStoreNotification.__table__
users = User.__table__

# Reclaiming a chain whose appAccountToken names no live user (5.3).
RELINK_INTERVAL = dt.timedelta(days=30)
LINKS_PER_USER_PER_DAY = 3


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class SyncRequest(BaseModel):
    signed_transaction: str
    signed_renewal_info: str


class NotificationRequest(BaseModel):
    signedPayload: str


async def _verified(call):
    try:
        return await call
    except appstore.NotConfigured:
        raise HTTPException(503, detail="subscriptions are not configured")
    except appstore.RetryLater:
        raise HTTPException(503, detail="try again")
    except appstore.InvalidSignedData as exc:
        log.info("signed data refused", extra={"reason": str(exc)})
        raise HTTPException(400, detail="invalid signed data")


async def apply_snapshot(conn: AsyncConnection, snap: appstore.Snapshot, now: dt.datetime) -> None:
    """Upsert the chain's row from the snapshot. One older than the row's
    last_signed_date is not applied. Never changes the owner."""
    values = snap.row(now)
    stmt = insert(subscriptions).values(original_transaction_id=snap.original_transaction_id, **values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[subscriptions.c.original_transaction_id],
        set_={name: stmt.excluded[name] for name in values},
        where=subscriptions.c.last_signed_date <= stmt.excluded.last_signed_date,
    )
    await conn.execute(stmt)


async def _is_live(conn: AsyncConnection, user_id: uuid.UUID) -> bool:
    found = (
        await conn.execute(
            select(users.c.id).where(users.c.id == user_id, users.c.deletion_requested_at.is_(None))
        )
    ).scalar_one_or_none()
    return found is not None


async def adopt_by_token(conn: AsyncConnection, snap: appstore.Snapshot, now: dt.datetime, source: str) -> None:
    """Outside the app (notifications, backfill) a chain gets an owner only
    when it has none and its appAccountToken names a live user (5.4). A
    deleted account's id would fail the foreign key; an owned chain is
    never moved."""
    token = snap.app_account_token
    if token is None or not await _is_live(conn, token):
        return
    linked = (
        await conn.execute(
            update(subscriptions)
            .where(
                subscriptions.c.original_transaction_id == snap.original_transaction_id,
                subscriptions.c.user_id.is_(None),
            )
            .values(user_id=token, linked_at=now)
            .returning(subscriptions.c.original_transaction_id)
        )
    ).scalar_one_or_none()
    if linked is not None:
        _log_link(token, linked, source)


def _log_link(user_id: uuid.UUID, original_transaction_id: str, source: str) -> None:
    log.info(
        "subscription linked",
        extra={"user_id": str(user_id), "original_transaction_id": original_transaction_id, "source": source},
    )


async def lock_user(conn: AsyncConnection, user_id: uuid.UUID) -> None:
    """Serializes this user's syncs, so the per-day link cap holds. FOR NO
    KEY UPDATE, not FOR UPDATE: the foreign key check of a notification
    linking a chain to this user takes FOR KEY SHARE on the same row, and
    against FOR UPDATE the two transactions deadlock."""
    await conn.execute(select(users.c.id).where(users.c.id == user_id).with_for_update(key_share=True))


async def _link(conn: AsyncConnection, otid: str, user_id: uuid.UUID, now: dt.datetime) -> None:
    await conn.execute(
        update(subscriptions)
        .where(subscriptions.c.original_transaction_id == otid)
        .values(user_id=user_id, linked_at=now)
    )


@router.post("/subscriptions/sync")
async def sync(body: SyncRequest, request: Request, user_id: uuid.UUID = Depends(current_user)):
    """One chain from Transaction.currentEntitlements; ownership rules of
    design 5.3. A rejected chain changes nothing."""
    store: appstore.AppStore = request.app.state.appstore
    snap = await _verified(store.snapshot(body.signed_transaction, body.signed_renewal_info))
    otid = snap.original_transaction_id
    token = snap.app_account_token
    now = _now()
    reclaimed = False
    async with request.app.state.engine.begin() as conn:
        await lock_user(conn, user_id)
        await apply_snapshot(conn, snap, now)
        row = (
            await conn.execute(
                select(subscriptions.c.user_id, subscriptions.c.linked_at)
                .where(subscriptions.c.original_transaction_id == otid)
                .with_for_update()
            )
        ).one()
        if token == user_id:
            # Apple says it is the caller's: bought in the app while signed in.
            if row.user_id != user_id:
                await _link(conn, otid, user_id, now)
                _log_link(user_id, otid, "token")
        elif token is not None and await _is_live(conn, token):
            # A signed transaction is not a secret: someone else's purchase.
            log.info("sync refused: chain of another account", extra={"user_id": str(user_id)})
            raise HTTPException(403, detail="this purchase belongs to another account")
        elif row.user_id is None:
            # No token (offer code, bought signed out) or a deleted account's.
            if row.linked_at is not None and now - row.linked_at < RELINK_INTERVAL:
                log.info("reclaim refused: chain relinked recently", extra={"user_id": str(user_id)})
                raise HTTPException(429, detail="this purchase was moved recently")
            links_today = (
                await conn.execute(
                    select(func.count()).where(
                        subscriptions.c.user_id == user_id,
                        subscriptions.c.linked_at > now - dt.timedelta(hours=24),
                    )
                )
            ).scalar_one()
            if links_today >= LINKS_PER_USER_PER_DAY:
                log.info("reclaim refused: daily link cap", extra={"user_id": str(user_id)})
                raise HTTPException(429, detail="too many purchases linked today")
            await _link(conn, otid, user_id, now)
            _log_link(user_id, otid, "reclaim")
            reclaimed = True
        elif row.user_id != user_id:
            log.info("sync refused: chain owned by another account", extra={"user_id": str(user_id)})
            raise HTTPException(409, detail="this purchase belongs to another account")
        current = (
            await conn.execute(
                select(subscriptions.c.status, subscriptions.c.expires_at).where(
                    subscriptions.c.original_transaction_id == otid
                )
            )
        ).one()
        plan, environment = await quotas.entitlement(conn, user_id, now)
    if reclaimed:
        # After the commit: Apple's later notifications then name the caller.
        await store.set_app_account_token(snap.environment, otid, user_id)
    return {
        "plan": plan,
        "environment": environment,
        "status": current.status,
        "expires_at": current.expires_at.isoformat(),
    }


@router.post("/appstore/notifications")
async def app_store_notification(body: NotificationRequest, request: Request):
    """Stored once per notification_uuid and applied in the same transaction
    (5.4): if applying fails nothing is stored and Apple's retry applies it.
    Anything other than 200 makes Apple retry."""
    store: appstore.AppStore = request.app.state.appstore
    n = await _verified(store.notification(body.signedPayload))
    now = _now()
    async with request.app.state.engine.begin() as conn:
        stored = (
            await conn.execute(
                insert(notifications)
                .values(
                    notification_uuid=n.uuid,
                    type=n.type,
                    subtype=n.subtype,
                    original_transaction_id=n.original_transaction_id,
                    signed_date=n.signed_date,
                    payload=n.payload,
                    processed_at=now,
                )
                .on_conflict_do_nothing(index_elements=[notifications.c.notification_uuid])
                .returning(notifications.c.notification_uuid)
            )
        ).scalar_one_or_none()
        if stored is None:
            log.info("notification already processed", extra={"type": n.type})
        elif n.snapshot is not None and n.type != appstore.CONSUMPTION_REQUEST:
            await apply_snapshot(conn, n.snapshot, now)
            await adopt_by_token(conn, n.snapshot, now, "notification")
    log.info("notification", extra={"type": n.type, "subtype": n.subtype})
    return Response(status_code=200)
