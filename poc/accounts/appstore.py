"""Apple's signed subscription data and the App Store Server API (design 5).

Verification uses Apple's app-store-server-library: the JWS chain must end
at Apple's root certificate (committed in certs/, pinned by its SHA-256 in
config.py), the leaf and intermediate must carry Apple's OIDs, and
bundleId (and, for Production notifications, the app's Apple ID) must be
ours. Production and Sandbox are both accepted, one verifier each (5.2):
App Review and TestFlight buy with Sandbox accounts in the production
build. Xcode and LocalTesting data is not signed by Apple and is
refused.

Nothing here touches the database; subscriptions.py does.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
from appstoreserverlibrary.api_client import AsyncAppStoreServerAPIClient
from appstoreserverlibrary.models.Environment import Environment
from appstoreserverlibrary.models.JWSRenewalInfoDecodedPayload import JWSRenewalInfoDecodedPayload
from appstoreserverlibrary.models.JWSTransactionDecodedPayload import JWSTransactionDecodedPayload
from appstoreserverlibrary.models.UpdateAppAccountTokenRequest import UpdateAppAccountTokenRequest
from appstoreserverlibrary.signed_data_verifier import (
    SignedDataVerifier,
    VerificationException,
    VerificationStatus,
)
from cryptography import x509
from cryptography.hazmat.primitives import serialization

import config

log = logging.getLogger("accounts.appstore")

ENVIRONMENTS = {"Production": Environment.PRODUCTION, "Sandbox": Environment.SANDBOX}
AUTO_RENEWABLE = "Auto-Renewable Subscription"
# Carries transaction info but asks for consumption data, which needs the
# user's consent; not answered and not applied in 1.0 (5.4).
CONSUMPTION_REQUEST = "CONSUMPTION_REQUEST"


class NotConfigured(Exception):
    """No Apple root certificates (or, for the Server API, no .p8 key)."""


class InvalidSignedData(Exception):
    """Not Apple's signed data for this app. The message is a short reason
    for the log; it never contains the data."""


class RetryLater(Exception):
    """Apple's OCSP responder could not be reached: not the data's fault."""


def _time(ms: Optional[int]) -> Optional[dt.datetime]:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc) if ms is not None else None


def _without_token(claims: Dict[str, Any]) -> Dict[str, Any]:
    # The owner lives only in subscriptions.user_id (4.1).
    return {k: v for k, v in claims.items() if k != "appAccountToken"}


@dataclass(frozen=True)
class Snapshot:
    """One chain's verified transaction and renewal info (5.1)."""

    transaction: JWSTransactionDecodedPayload
    renewal: JWSRenewalInfoDecodedPayload

    @property
    def original_transaction_id(self) -> str:
        return self.transaction.originalTransactionId

    @property
    def environment(self) -> str:
        return self.transaction.rawEnvironment

    @property
    def app_account_token(self) -> Optional[uuid.UUID]:
        token = self.transaction.appAccountToken
        return uuid.UUID(token) if token else None

    @property
    def signed_date(self) -> dt.datetime:
        # The older part's: with the newer one, a transaction kept from
        # before a refund, sent with fresh renewal info, would undo it.
        return _time(min(self.transaction.signedDate, self.renewal.signedDate))

    def row(self, now: dt.datetime) -> Dict[str, Any]:
        """The subscriptions columns, all recomputed from this snapshot."""
        tx, renewal = self.transaction, self.renewal
        expires_at = _time(tx.expiresDate)
        grace = _time(renewal.gracePeriodExpiresDate)
        revoked_at = _time(tx.revocationDate)
        if revoked_at is not None:
            status = "revoked"
        elif expires_at > now:
            status = "active"
        elif grace is not None and grace > now:
            status = "grace"
        elif renewal.isInBillingRetryPeriod:
            status = "billing_retry"
        else:
            status = "expired"
        return {
            "product_id": tx.productId,
            "status": status,
            "expires_at": expires_at,
            "grace_period_expires_at": grace,
            "auto_renew": renewal.rawAutoRenewStatus == 1,
            "auto_renew_product_id": renewal.autoRenewProductId,
            "revoked_at": revoked_at,
            "revocation_reason": tx.rawRevocationReason,
            # Apple's code: 1 introductory, 2 promotional, 3 offer code, 4 win-back.
            "offer_type": str(tx.rawOfferType) if tx.rawOfferType is not None else None,
            "offer_id": tx.offerIdentifier,
            "environment": self.environment,
            "last_signed_date": self.signed_date,
        }


@dataclass(frozen=True)
class Notification:
    uuid: str
    type: str
    subtype: Optional[str]
    signed_date: dt.datetime
    original_transaction_id: Optional[str]
    # None when the notification carries no subscription snapshot (TEST).
    snapshot: Optional[Snapshot]
    # The decoded payload with the signed parts decoded too, without
    # appAccountToken; what app_store_notifications keeps.
    payload: Dict[str, Any]


def _claimed_environment(signed: str, notification: bool = False) -> str:
    """The environment a JWS claims, read before verification only to pick
    the verifier; that verifier then checks it again."""
    try:
        claims = jwt.decode(signed, options={"verify_signature": False})
    except jwt.PyJWTError:
        raise InvalidSignedData("malformed")
    if notification:
        # Where the library reads it too; appData is RESCIND_CONSENT's.
        claims = claims.get("data") or claims.get("summary") or claims.get("appData") or {}
    environment = claims.get("environment")
    if environment not in ENVIRONMENTS:
        raise InvalidSignedData("environment")
    return environment


def _decoded(signed: str) -> Dict[str, Any]:
    # Only for data a verifier has already accepted.
    return jwt.decode(signed, options={"verify_signature": False})


class AppStore:
    """The verifiers (one per environment) and Server API clients. Either
    dict may be empty: the service runs without Apple's keys."""

    def __init__(
        self,
        verifiers: Dict[str, SignedDataVerifier],
        clients: Dict[str, AsyncAppStoreServerAPIClient],
    ):
        self.verifiers = verifiers
        self.clients = clients

    async def aclose(self) -> None:
        for client in self.clients.values():
            await client.async_close()

    def _verifier(self, environment: str) -> SignedDataVerifier:
        if not self.verifiers:
            raise NotConfigured()
        return self.verifiers[environment]

    async def _verify(self, method, signed: str):
        """Run one library check off the event loop: with online checks on it
        makes blocking OCSP requests."""
        try:
            return await asyncio.to_thread(method, signed)
        except VerificationException as exc:
            if exc.status == VerificationStatus.RETRYABLE_VERIFICATION_FAILURE:
                raise RetryLater()
            raise InvalidSignedData(exc.status.name.lower())

    async def snapshot(self, signed_transaction: str, signed_renewal_info: str) -> Snapshot:
        verifier = self._verifier(_claimed_environment(signed_transaction))
        tx = await self._verify(verifier.verify_and_decode_signed_transaction, signed_transaction)
        # The renewal verifier checks the environment matches the transaction's.
        renewal = await self._verify(verifier.verify_and_decode_renewal_info, signed_renewal_info)
        if tx.rawType != AUTO_RENEWABLE:
            raise InvalidSignedData("not a subscription")
        if renewal.originalTransactionId != tx.originalTransactionId:
            raise InvalidSignedData("renewal info of another chain")
        return Snapshot(tx, renewal)

    async def notification(self, signed_payload: str) -> Notification:
        verifier = self._verifier(_claimed_environment(signed_payload, notification=True))
        decoded = await self._verify(verifier.verify_and_decode_notification, signed_payload)
        payload = _decoded(signed_payload)
        data = decoded.data
        snapshot = None
        otid = None
        if data is not None and data.signedTransactionInfo:
            data_claims = payload["data"]
            data_claims["transactionInfo"] = _without_token(_decoded(data_claims.pop("signedTransactionInfo")))
            otid = data_claims["transactionInfo"].get("originalTransactionId")
            if data.signedRenewalInfo:
                data_claims["renewalInfo"] = _without_token(_decoded(data_claims.pop("signedRenewalInfo")))
                snapshot = await self.snapshot(data.signedTransactionInfo, data.signedRenewalInfo)
            else:
                # Every notification about a subscription carries both;
                # verify the transaction anyway before storing it.
                await self._verify(verifier.verify_and_decode_signed_transaction, data.signedTransactionInfo)
        return Notification(
            uuid=decoded.notificationUUID,
            type=decoded.rawNotificationType,
            subtype=decoded.rawSubtype,
            signed_date=_time(decoded.signedDate),
            original_transaction_id=otid,
            snapshot=snapshot,
            payload=payload,
        )

    async def set_app_account_token(self, environment: str, original_transaction_id: str, user_id: uuid.UUID) -> None:
        """Point the chain's later transactions and notifications at
        `user_id` (5.3). Best effort: the link is already stored, and a
        notification never moves an owned chain."""
        client = self.clients.get(environment)
        if client is None:
            log.warning("set app account token skipped: no App Store Server API key")
            return
        try:
            await client.set_app_account_token(
                original_transaction_id, UpdateAppAccountTokenRequest(appAccountToken=str(user_id))
            )
        except Exception as exc:  # noqa: BLE001 -- logged; support can repeat it
            log.warning(
                "set app account token failed",
                extra={"original_transaction_id": original_transaction_id, "error": type(exc).__name__},
            )


def apple_root_der() -> bytes:
    """Apple's committed root, as the DER Apple publishes. It is committed
    as PEM: the repo's text checks would read a DER file as binary noise."""
    pem = x509.load_pem_x509_certificate(config.APPLE_ROOT_CERT_PATH.read_bytes())
    return pem.public_bytes(serialization.Encoding.DER)


def _root_certs() -> Optional[List[bytes]]:
    """The trust anchors, or None (App Store not configured) when one is
    unusable. Apple's committed root that does not match its pin is not
    Apple's. A replacement root from APPSTORE_ROOT_CERTS is the tests' own:
    a file that is not DER would load, then refuse every genuine payload;
    one that is DER is used, and logged as an error so it cannot go
    unnoticed anywhere else."""
    paths = config.appstore_root_cert_paths()
    if not paths:
        data = apple_root_der()
        digest = hashlib.sha256(data).hexdigest()
        if digest != config.APPLE_ROOT_CERT_SHA256:
            log.error(
                "app store not configured: Apple root certificate does not match its pin",
                extra={"path": str(config.APPLE_ROOT_CERT_PATH), "sha256": digest},
            )
            return None
        return [data]
    roots = []
    for path in paths:
        data = Path(path).read_bytes()
        try:
            x509.load_der_x509_certificate(data)
        except ValueError:
            log.error("app store not configured: root certificate is not DER", extra={"path": path})
            return None
        digest = hashlib.sha256(data).hexdigest()
        if digest != config.APPLE_ROOT_CERT_SHA256:
            log.error(
                "app store trusts a root certificate that is not Apple's (APPSTORE_ROOT_CERTS)",
                extra={"path": path, "sha256": digest},
            )
        roots.append(data)
    return roots


def from_config() -> AppStore:
    verifiers: Dict[str, SignedDataVerifier] = {}
    app_apple_id = config.appstore_app_apple_id()
    roots = _root_certs()
    if roots is not None and app_apple_id is not None:
        for name, env in ENVIRONMENTS.items():
            verifiers[name] = SignedDataVerifier(
                roots, config.appstore_online_checks(), env, config.APPLE_BUNDLE_ID, app_apple_id
            )
    elif app_apple_id is None:
        log.warning("app store not configured: APPSTORE_APP_APPLE_ID unset")
    clients: Dict[str, AsyncAppStoreServerAPIClient] = {}
    if config.APPSTORE_KEY_PATH and config.APPSTORE_KEY_ID and config.APPSTORE_ISSUER_ID:
        key = Path(config.APPSTORE_KEY_PATH).read_bytes()
        for name, env in ENVIRONMENTS.items():
            clients[name] = AsyncAppStoreServerAPIClient(
                key, config.APPSTORE_KEY_ID, config.APPSTORE_ISSUER_ID, config.APPLE_BUNDLE_ID, env
            )
    return AppStore(verifiers, clients)
