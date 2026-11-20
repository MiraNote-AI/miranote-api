"""Apple's root certificate (api #123): committed, pinned by its SHA-256 and
checked at startup. Nothing replaces it: the tests that need their own
root patch config.py (FakeAppStore.trust)."""

from __future__ import annotations

import hashlib
import logging

import pytest

import appstore
import config
from tests.fake_appstore import FakeAppStore


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def errors():
    # The service's "accounts" logger does not propagate, so caplog would
    # not see it once main.py is imported.
    handler = _Records()
    logger = logging.getLogger("accounts.appstore")
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def app_apple_id(monkeypatch):
    monkeypatch.setenv("APPSTORE_APP_APPLE_ID", "1234567890")
    monkeypatch.setenv("APPSTORE_ONLINE_CHECKS", "0")


def test_the_committed_root_is_apples_and_matches_the_pin():
    assert hashlib.sha256(appstore.apple_root_der()).hexdigest() == config.APPLE_ROOT_CERT_SHA256


def test_apples_pinned_root_is_used(errors):
    store = appstore.from_config()
    assert set(store.verifiers) == {"Production", "Sandbox"}
    assert errors == []


def test_a_committed_root_that_does_not_match_the_pin_is_refused_loudly(errors, monkeypatch):
    monkeypatch.setattr(config, "APPLE_ROOT_CERT_SHA256", "0" * 64)
    assert appstore.from_config().verifiers == {}
    assert [r.getMessage() for r in errors] == [
        "app store not configured: Apple root certificate does not match its pin"
    ]


@pytest.mark.parametrize("content", [None, b"not a certificate\n"])
def test_an_unreadable_committed_root_leaves_only_the_app_store_unconfigured(errors, tmp_path, monkeypatch, content):
    # Missing or not a certificate: refused like a pin mismatch, not raised
    # out of startup, which would take sign-in and /usage/* down with it.
    path = tmp_path / "AppleRootCA-G3.pem"
    if content is not None:
        path.write_bytes(content)
    monkeypatch.setattr(config, "APPLE_ROOT_CERT_PATH", path)
    assert appstore.from_config().verifiers == {}
    assert [r.getMessage() for r in errors] == ["app store not configured: Apple root certificate is unreadable"]


def test_appstore_root_certs_left_in_an_env_is_ignored(errors, tmp_path, monkeypatch):
    # It used to replace the pinned root anywhere, with only an error log:
    # the one setting that made the service accept data a non-Apple root
    # signed, such as a forged purchase that grants Plus.
    der = tmp_path / "root.cer"
    der.write_bytes(FakeAppStore().root_der())
    monkeypatch.setenv("APPSTORE_ROOT_CERTS", str(der))
    assert appstore._root_certs() == [appstore.apple_root_der()]
    assert errors == []
