"""Submit-and-poll routes (handoff 5.1 / 5.2 plus the two gap fills)."""

from __future__ import annotations

import builtins
import json
import re
import threading
import time
from pathlib import Path

import pytest

import jobs
from tests.test_compose import IMAGES, _a_json, _b_json, _meta


def _files(ids=None, data=None):
    ids = ids or ["P1", "P2", "P3", "P4", "P5"]
    return [("photos", (pid + ".jpg", (data or {}).get(pid, IMAGES.get(pid, b"\xff\xd8x")),
                        "image/jpeg")) for pid in ids]


def _submit(client, meta=None, files=None):
    return client.post("/collections/compose",
                       data={"meta": meta if isinstance(meta, str) else json.dumps(meta or _meta())},
                       files=files if files is not None else _files())


def _poll(client, job_id, until=("ok", "rejected", "error"), seconds=15.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        r = client.get("/collections/jobs/" + job_id)
        if r.status_code != 200 or r.json()["status"] in until:
            return r
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_compose_returns_202_job_id_fast(client, fake_model):
    gate = threading.Event()
    fake_model.reply("a", _a_json(), gate=gate).reply("b", _b_json())
    started = time.monotonic()
    r = _submit(client)
    assert r.status_code == 202
    assert time.monotonic() - started < 2
    job_id = r.json()["job_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", job_id)
    assert client.get("/collections/jobs/" + job_id).json() == {"status": "running"}
    gate.set()
    assert _poll(client, job_id).json()["status"] == "ok"


def test_poll_running_then_ok(client, fake_model):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    job_id = _submit(client).json()["job_id"]
    r = _poll(client, job_id)
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "ok"
    assert body["memory"]["push"]["title"] == "10.18 · 达拉斯植物园 · 傍晚"


def test_rejected_job_reads_rejected(client, fake_model):
    fake_model.reply("a", "x").reply("a", "y")
    job_id = _submit(client).json()["job_id"]
    assert _poll(client, job_id).json() == {
        "status": "rejected", "reason": "check_failed", "message": "AI 的结果没通过检查"}


def test_unknown_job_404_expired(client):
    r = client.get("/collections/jobs/" + "0" * 32)
    assert r.status_code == 404
    assert r.json() == {"status": "expired"}


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_job_expires_after_ttl():
    clock = _Clock()
    store = jobs.JobStore(ttl=900, cap=300, clock=clock)
    job_id = store.submit(lambda deadline: {"status": "ok", "memory": {}})
    deadline = time.monotonic() + 2
    while store.get(job_id) == {"status": "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
    clock.now += 899
    assert store.get(job_id)["status"] == "ok"
    clock.now += 2
    assert store.get(job_id) is None


def test_job_over_cap_reads_timeout():
    clock = _Clock()
    store = jobs.JobStore(ttl=900, cap=300, clock=clock)
    release = threading.Event()
    job_id = store.submit(lambda deadline: release.wait(5) and {"status": "ok", "memory": {}})
    assert store.get(job_id) == {"status": "running"}
    clock.now += 301
    assert store.get(job_id) == {"status": "error", "reason": "timeout", "message": "AI 超时"}
    release.set()
    time.sleep(0.05)
    assert store.get(job_id)["reason"] == "timeout"  # a late result does not overwrite it


def test_crashed_job_reads_model_error():
    store = jobs.JobStore()

    def boom(deadline):
        raise RuntimeError("bug")

    job_id = store.submit(boom)
    end = time.monotonic() + 2
    while store.get(job_id) == {"status": "running"} and time.monotonic() < end:
        time.sleep(0.01)
    assert store.get(job_id)["reason"] == "model_error"


def test_expired_job_on_the_route(client, fake_model):
    clock = _Clock()
    client.main.STORE = jobs.JobStore(clock=clock)
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    job_id = _submit(client).json()["job_id"]
    _poll(client, job_id)
    clock.now += 901
    r = client.get("/collections/jobs/" + job_id)
    assert (r.status_code, r.json()) == (404, {"status": "expired"})


def _assert_400(r, fake_model):
    assert r.status_code == 400
    assert r.json() == {"status": "error", "reason": "bad_request", "message": "提交的照片或信息不完整"}
    assert fake_model.calls == []


def test_eleven_photos_400_no_model_call(client, fake_model):
    meta = _meta()
    ids = ["P{}".format(i) for i in range(1, 12)]
    meta["photos"] = [dict(meta["photos"][0], id=pid) for pid in ids]
    _assert_400(_submit(client, meta, _files(ids)), fake_model)


def test_zero_photos_400(client, fake_model):
    _assert_400(_submit(client, _meta(photos=[]), []), fake_model)


def test_filename_mismatch_400(client, fake_model):
    _assert_400(_submit(client, files=_files(["P1", "P2", "P3", "P4", "X5"])), fake_model)
    _assert_400(_submit(client, files=_files(["P1", "P2", "P3", "P4"])), fake_model)


def test_duplicate_ids_400(client, fake_model):
    meta = _meta()
    meta["photos"][1]["id"] = "P1"
    _assert_400(_submit(client, meta, _files(["P1", "P1", "P3", "P4", "P5"])), fake_model)


def test_bad_meta_json_400(client, fake_model):
    _assert_400(_submit(client, "{not json"), fake_model)
    _assert_400(_submit(client, "[]"), fake_model)


@pytest.mark.parametrize("field", ["group_key", "date", "weekday", "start_time", "end_time",
                                   "raw_count", "covers_all", "recent_question_ids", "photos"])
def test_missing_meta_field_400(client, fake_model, field):
    meta = _meta()
    del meta[field]
    _assert_400(_submit(client, meta), fake_model)


def test_bad_taken_at_400(client, fake_model):
    meta = _meta()
    meta["photos"][0]["taken_at"] = "yesterday"
    _assert_400(_submit(client, meta), fake_model)


def test_oversized_photo_400(client, fake_model):
    big = {"P1": b"\xff\xd8" + b"0" * (5 * 1024 * 1024)}
    _assert_400(_submit(client, files=_files(data=big)), fake_model)


def test_not_multipart_400(client, fake_model):
    _assert_400(client.post("/collections/compose", json={"meta": "{}"}), fake_model)


def test_photos_not_written_to_disk(client, fake_model, tmp_log, monkeypatch):
    writes = []
    real_open = builtins.open

    def spy_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "x", "+")):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    def no_write_bytes(self, data):
        raise AssertionError("Path.write_bytes called on " + str(self))

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "write_bytes", no_write_bytes)
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    job_id = _submit(client).json()["job_id"]
    assert _poll(client, job_id).json()["status"] == "ok"
    assert writes == [str(tmp_log)]
    logged = tmp_log.read_bytes()
    for data in IMAGES.values():
        assert data not in logged


def test_two_jobs_do_not_mix(client, fake_model):
    def a_for(messages):
        ids = []
        for part in messages[1]["content"]:
            if part["type"] == "text":
                m = re.match(r"^(\w+)｜", part["text"])
                if m:
                    ids.append(m.group(1))
        raw = json.loads(_a_json())
        for photo, pid in zip(raw["photos"], ids):
            photo["id"] = pid
        raw["photos"] = raw["photos"][: len(ids)]
        raw["selected_ids"] = ids[:4]
        raw["peak_id"] = ids[3]
        return json.dumps(raw, ensure_ascii=False)

    def b_for(messages):
        peak = re.search(r"峰值照 (\w+)", messages[1]["content"]).group(1)
        return _b_json(candidates=[{"text": "台中央，落下一束光", "photo_ids": [peak],
                                    "uses": ["一束光"]}])

    fake_model.respond("a", a_for, times=2).respond("b", b_for, times=2)
    m1 = _meta(group_key="g1")
    m2 = _meta(group_key="g2", place="休斯敦")
    for p in m2["photos"]:
        p["id"] = p["id"].replace("P", "Q")
    q_files = [("photos", (pid.replace("P", "Q") + ".jpg", data, "image/jpeg"))
               for pid, data in IMAGES.items()]
    j1 = _submit(client, m1).json()["job_id"]
    j2 = _submit(client, m2, q_files).json()["job_id"]
    r1 = _poll(client, j1).json()["memory"]
    r2 = _poll(client, j2).json()["memory"]
    assert (r1["id"], r1["push"]["coverPhotoId"]) == ("g1", "P4")
    assert (r2["id"], r2["push"]["coverPhotoId"]) == ("g2", "Q4")
    assert r2["ending"]["factLine"] == "10.18 · 休斯敦"
    assert all(p["id"].startswith("P") for p in r1["story"]["photos"])
    assert all(p["id"].startswith("Q") for p in r2["story"]["photos"])


@pytest.mark.parametrize("field,value", [
    ("weekday", 0), ("weekday", 8), ("weekday", "7"), ("weekday", True),
    ("raw_count", 0), ("raw_count", "23"),
    ("start_time", 1403), ("end_time", None),
    ("recent_question_ids", ["B4", 2]), ("recent_question_ids", "B4"),
])
def test_bad_meta_types_400(client, fake_model, field, value):
    _assert_400(_submit(client, _meta(**{field: value})), fake_model)


@pytest.mark.parametrize("field,value", [
    ("labels", ["food_drink", 3]), ("labels", "food_drink"),
    ("faces", "2"), ("faces", True),
    ("favorite", "yes"), ("edited", 1),
    ("represents", 0), ("represents", "1"),
])
def test_bad_photo_types_400(client, fake_model, field, value):
    meta = _meta()
    meta["photos"][0][field] = value
    _assert_400(_submit(client, meta), fake_model)


def test_optional_photo_fields_may_be_null(client, fake_model):
    meta = _meta()
    meta["photos"][0].update(faces=None, favorite=None, edited=None)
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    r = _submit(client, meta)
    assert r.status_code == 202
    assert _poll(client, r.json()["job_id"]).json()["status"] == "ok"


def test_non_image_upload_400(client, fake_model):
    _assert_400(_submit(client, files=_files(data={"P1": b"hello, not a photo"})), fake_model)


def test_png_and_heic_uploads_accepted(client, fake_model):
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    heic = b"\x00\x00\x00\x18ftypheic" + b"0" * 32
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    r = _submit(client, files=_files(data={"P1": png, "P2": heic}))
    assert r.status_code == 202
    _poll(client, r.json()["job_id"])
    content = fake_model.calls[0]["messages"][1]["content"]
    urls = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
    assert urls[0].startswith("data:image/png;base64,")
    assert urls[1].startswith("data:image/heic;base64,")
    assert urls[2].startswith("data:image/jpeg;base64,")


def test_job_finishing_after_cap_reads_timeout():
    clock = _Clock()
    store = jobs.JobStore(ttl=900, cap=300, clock=clock)

    def late(deadline):
        clock.now += 301
        return {"status": "ok", "memory": {}}

    job_id = store.submit(late)
    end = time.monotonic() + 2
    while store.get(job_id) == {"status": "running"} and time.monotonic() < end:
        time.sleep(0.01)
    assert store.get(job_id) == {"status": "error", "reason": "timeout", "message": "AI 超时"}
