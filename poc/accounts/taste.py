"""Tier 2 taste events (design 4.2): which catalog items the user was
shown and picked, kept, undid, deleted or regenerated. Choices only, never
content: every field is a uuid, a number, a time, a version, or a catalog
id, and a body with any other field is refused.

    POST /events          app  {event_version, events: [...]}  -> {stored, duplicates}
    GET  /me/preferences  app  -> {taste_events_enabled}
    PUT  /me/preferences  app  {taste_events_enabled}  -> the same; off
                               deletes the user's collected events

POST /events answers 404 unless TASTE_EVENTS_ENABLED=1, and 403 while the
user's switch ("Use my choices to improve and personalize MiraNote") is
off. The preferences routes answer either way, so the switch can always be
turned off and the events deleted. The README lists every registered event.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert

from auth import current_user
from models import TasteEvent, User

log = logging.getLogger("accounts.taste")

router = APIRouter()

taste_events = TasteEvent.__table__
users = User.__table__

# One global version for the event format; it changes with an app release
# that changes the format, never per event.
EVENT_VERSION = 1

# Registered event names, each with the one item_type it carries. A name
# not here is refused. Adding one: a line here and a row in the README.
EVENTS = {
    "layout_choice": "layout",
    "font_choice": "font",
    "color_choice": "color",
    "sticker_style_choice": "sticker_style",
    # The tone option the user picked; never inferred from their text.
    "tone_choice": "tone",
    "image_style_choice": "image_style",
}

MAX_BATCH = 100

# An occurred_at further ahead than this is a broken clock; it would also
# outlive the 18-month retention.
MAX_CLOCK_AHEAD = dt.timedelta(days=1)

# Catalog ids: lowercase, no spaces. A format check, not the catalog: it
# refuses typed text (spaces, capitals, punctuation, non-ASCII), but a slug
# such as "my-dog-died" still fits, so the app sends only ids from its own
# catalog, never one built from what the user wrote.
_CATALOG_ID = r"^[a-z0-9][a-z0-9_.-]{0,63}$"


class TasteEventIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: uuid.UUID
    session_id: uuid.UUID
    occurred_at: AwareDatetime
    app_version: str = Field(pattern=r"^[0-9]{1,4}(\.[0-9]{1,4}){0,3}$")
    build: str = Field(pattern=r"^[0-9]{1,10}(\.[0-9]{1,10}){0,3}$")
    tz_offset_minutes: int = Field(ge=-720, le=840)
    name: str
    surface: str = Field(pattern=_CATALOG_ID)
    item_type: str
    item_id: str = Field(pattern=_CATALOG_ID)
    position: Optional[int] = Field(None, ge=0, le=999)
    action: Literal["shown", "selected", "kept", "undone", "deleted", "regenerated"]
    request_id: Optional[uuid.UUID] = None

    @field_validator("occurred_at")
    @classmethod
    def _not_in_the_future(cls, value: dt.datetime) -> dt.datetime:
        if value > dt.datetime.now(dt.timezone.utc) + MAX_CLOCK_AHEAD:
            raise ValueError("occurred_at is in the future")
        return value

    @model_validator(mode="after")
    def _registered(self) -> "TasteEventIn":
        if self.name not in EVENTS:
            raise ValueError("unregistered event name")
        if self.item_type != EVENTS[self.name]:
            raise ValueError("item_type does not match the event name")
        return self


class EventBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_version: Literal[EVENT_VERSION]
    events: List[TasteEventIn] = Field(min_length=1, max_length=MAX_BATCH)


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taste_events_enabled: bool


def collection_on(request: Request) -> None:
    """The whole endpoint is off unless TASTE_EVENTS_ENABLED=1."""
    if not request.app.state.taste_events_enabled:
        raise HTTPException(404)


@router.post("/events", dependencies=[Depends(collection_on)])
async def post_events(body: EventBatch, request: Request, user_id: uuid.UUID = Depends(current_user)):
    """Stores a batch; an event_id already stored is skipped, so the app
    can resend a batch whose answer it did not get. The share lock on the
    user's row orders this against PUT /me/preferences and DELETE /me: a
    batch either lands before the switch goes off or the account is marked
    (and is deleted with the rest), or it sees that and stores nothing."""
    # Inserted in event_id order: two sends of the same events in flight at
    # once then wait for each other's ids in the same order, never in a
    # circle (a deadlock, which Postgres ends by failing one of them).
    rows = sorted(
        (dict(e.model_dump(), user_id=user_id, event_version=body.event_version) for e in body.events),
        key=lambda row: row["event_id"],
    )
    async with request.app.state.engine.begin() as conn:
        user = (
            await conn.execute(
                select(users.c.taste_events_enabled, users.c.deletion_requested_at)
                .where(users.c.id == user_id)
                .with_for_update(read=True)
            )
        ).one()
        if user.deletion_requested_at is not None:
            raise HTTPException(401, detail="invalid access token")
        if not user.taste_events_enabled:
            raise HTTPException(403, detail="taste events are turned off for this account")
        stored = (
            await conn.execute(
                insert(taste_events)
                .values(rows)
                .on_conflict_do_nothing(index_elements=[taste_events.c.event_id])
                .returning(taste_events.c.event_id)
            )
        ).scalars().all()
    return {"stored": len(stored), "duplicates": len(rows) - len(stored)}


@router.get("/me/preferences")
async def get_preferences(request: Request, user_id: uuid.UUID = Depends(current_user)):
    async with request.app.state.engine.connect() as conn:
        enabled = (
            await conn.execute(select(users.c.taste_events_enabled).where(users.c.id == user_id))
        ).scalar_one()
    return {"taste_events_enabled": enabled}


@router.put("/me/preferences")
async def put_preferences(body: Preferences, request: Request, user_id: uuid.UUID = Depends(current_user)):
    async with request.app.state.engine.begin() as conn:
        await conn.execute(
            update(users).where(users.c.id == user_id).values(taste_events_enabled=body.taste_events_enabled)
        )
        if not body.taste_events_enabled:
            deleted = (await conn.execute(delete(taste_events).where(taste_events.c.user_id == user_id))).rowcount
            log.info("taste events turned off", extra={"user_id": str(user_id), "deleted": deleted})
    return {"taste_events_enabled": body.taste_events_enabled}
