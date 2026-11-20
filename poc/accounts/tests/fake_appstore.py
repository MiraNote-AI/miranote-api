"""A stand-in for Apple's App Store signing: a locally made root CA, an
intermediate and a leaf with Apple's OIDs, and ES256 JWS with the x5c chain
in the header -- the shape app-store-server-library verifies. Nothing here
is Apple's; the verifier trusts this root only while a test has called
trust(), which patches the committed root and its pin in config.py."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

BUNDLE_ID = "ai.miranote.app"
APP_APPLE_ID = 6700000001
PRODUCT_MONTHLY = "ai.miranote.app.plus.monthly"

LEAF_OID = x509.ObjectIdentifier("1.2.840.113635.100.6.11.1")
INTERMEDIATE_OID = x509.ObjectIdentifier("1.2.840.113635.100.6.2.1")


def _cert(name, key, issuer_name, issuer_key, ca: bool, oid=None) -> x509.Certificate:
    now = dt.datetime.now(dt.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_name)]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=3650))
        .not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=not ca, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=ca, crl_sign=ca,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()), critical=False)
    )
    if oid is not None:
        builder = builder.add_extension(x509.UnrecognizedExtension(oid, b"\x05\x00"), critical=False)
    return builder.sign(issuer_key, hashes.SHA256())


def _der_b64(cert: x509.Certificate) -> str:
    return base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode()


def _ms(value: dt.datetime) -> int:
    return int(value.timestamp() * 1000)


class FakeAppStore:
    def __init__(self):
        root_key = ec.generate_private_key(ec.SECP256R1())
        mid_key = ec.generate_private_key(ec.SECP256R1())
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.root = _cert("Fake Apple Root", root_key, "Fake Apple Root", root_key, ca=True)
        mid = _cert("Fake WWDR", mid_key, "Fake Apple Root", root_key, ca=True, oid=INTERMEDIATE_OID)
        leaf = _cert("Fake App Store", self.key, "Fake WWDR", mid_key, ca=False, oid=LEAF_OID)
        self.x5c = [_der_b64(leaf), _der_b64(mid), _der_b64(self.root)]

    def root_der(self) -> bytes:
        return self.root.public_bytes(serialization.Encoding.DER)

    def trust(self, monkeypatch, tmp_path) -> None:
        """For one test, put this root where Apple's committed one is read
        (as PEM, like certs/) and pin it. No setting replaces Apple's root,
        so this is the only way to trust a test root, and it stays inside
        the test's process."""
        import config

        path = tmp_path / "fake-root.pem"
        path.write_bytes(self.root.public_bytes(serialization.Encoding.PEM))
        monkeypatch.setattr(config, "APPLE_ROOT_CERT_PATH", path)
        monkeypatch.setattr(config, "APPLE_ROOT_CERT_SHA256", hashlib.sha256(self.root_der()).hexdigest())

    def sign(self, claims: Dict[str, Any], key=None, x5c: Optional[List[str]] = None) -> str:
        return jwt.encode(claims, key or self.key, algorithm="ES256", headers={"x5c": x5c or self.x5c})

    def transaction(
        self,
        otid: str,
        token: Optional[uuid.UUID] = None,
        expires_in: dt.timedelta = dt.timedelta(days=30),
        revoked: bool = False,
        environment: str = "Sandbox",
        signed_at: Optional[dt.datetime] = None,
        **overrides: Any,
    ) -> str:
        now = dt.datetime.now(dt.timezone.utc)
        signed_at = signed_at or now
        claims: Dict[str, Any] = {
            "originalTransactionId": otid,
            "transactionId": str(int(time.time() * 1000)),
            "bundleId": BUNDLE_ID,
            "productId": PRODUCT_MONTHLY,
            "purchaseDate": _ms(now - dt.timedelta(days=1)),
            "expiresDate": _ms(now + expires_in),
            "type": "Auto-Renewable Subscription",
            "inAppOwnershipType": "PURCHASED",
            "signedDate": _ms(signed_at),
            "environment": environment,
        }
        if token is not None:
            claims["appAccountToken"] = str(token)
        if revoked:
            claims["revocationDate"] = _ms(now)
            claims["revocationReason"] = 0
        claims.update(overrides)
        return self.sign(claims)

    def renewal(
        self,
        otid: str,
        auto_renew: bool = True,
        grace_in: Optional[dt.timedelta] = None,
        billing_retry: bool = False,
        environment: str = "Sandbox",
        signed_at: Optional[dt.datetime] = None,
        **overrides: Any,
    ) -> str:
        now = dt.datetime.now(dt.timezone.utc)
        claims: Dict[str, Any] = {
            "originalTransactionId": otid,
            "productId": PRODUCT_MONTHLY,
            "autoRenewProductId": PRODUCT_MONTHLY,
            "autoRenewStatus": 1 if auto_renew else 0,
            "isInBillingRetryPeriod": billing_retry,
            "signedDate": _ms(signed_at or now),
            "environment": environment,
        }
        if grace_in is not None:
            claims["gracePeriodExpiresDate"] = _ms(now + grace_in)
        claims.update(overrides)
        return self.sign(claims)

    def notification(
        self,
        type_: str,
        subtype: Optional[str] = None,
        transaction: Optional[str] = None,
        renewal: Optional[str] = None,
        environment: str = "Sandbox",
        notification_uuid: Optional[str] = None,
        **data_overrides: Any,
    ) -> str:
        data: Dict[str, Any] = {
            "appAppleId": APP_APPLE_ID,
            "bundleId": BUNDLE_ID,
            "bundleVersion": "1",
            "environment": environment,
        }
        if transaction is not None:
            data["signedTransactionInfo"] = transaction
        if renewal is not None:
            data["signedRenewalInfo"] = renewal
        data.update(data_overrides)
        claims: Dict[str, Any] = {
            "notificationType": type_,
            "notificationUUID": notification_uuid or str(uuid.uuid4()),
            "data": data,
            "version": "2.0",
            "signedDate": _ms(dt.datetime.now(dt.timezone.utc)),
        }
        if subtype is not None:
            claims["subtype"] = subtype
        return self.sign(claims)
