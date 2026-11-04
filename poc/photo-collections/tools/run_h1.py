"""Acceptance H1-H3: run the prototype's 80 test photos through the running
service and write a summary table OUTSIDE the repo.

Real network and real model calls (costs money): not part of tests/.
Personal photos are read from the prototype's cache and sent from memory;
nothing is written into the repo.

Usage (from poc/photo-collections, with the service running on :8006):
    .venv/bin/python tools/run_h1.py \
        --prototype "<.../photo-themes-prototype>" \
        --validator "<.../v1/visuals/_group_validate.py>" \
        --photos "<.../zhaoyan-bot/new feature/photo trigger/input>" \
        --out <dir outside the repo> --base http://localhost:8006 \
        --token-env BETA_TOKEN --budget 5

Steps: run the V1 grouping validator; take its pushable groups; mark
`represents` from its dedupe; if a group has more than 10 unique photos,
coarse-filter it with a simple port of the coarse-filter rules; resize each
photo to 768 px (long side) and re-encode as JPEG without EXIF; submit; poll
every 5 s; read the cost from the service log; stop when the cumulative
cost passes --budget. The prototype's lecture-slides group (removed by V1
grouping as utility photos) is also sent once, as an expected rejection.
Then H2 re-runs the program checks on the outputs and H3 computes the
fallback and A-question shares.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from PIL import Image

POC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC))

import body  # noqa: E402
import check_a  # noqa: E402
import compose  # noqa: E402
import questions  # noqa: E402

# Copied from the prototype's photothemes/rules.py BROAD_CATEGORIES, minus
# "sign" in document (handoff 7.1: menus and signs must survive).
BROAD_CATEGORIES = {
    "people": {"people", "adult", "child", "baby", "crowd"},
    "animal": {"animal", "canine", "feline", "bird", "mammal", "reptile"},
    "food_drink": {"food", "dessert", "drink", "fruit", "vegetable", "baked_goods"},
    "tableware": {"tableware", "utensil"},
    "plant": {"plant", "flower", "foliage", "tree", "grass", "shrub"},
    "water": {"water", "water_body", "shore"},
    "sky": {"sky", "sunset_sunrise", "night_sky", "cloudy", "blue_sky"},
    "built": {"building", "structure", "cityscape", "skyscraper"},
    "street": {"street", "road", "crosswalk", "parking_lot"},
    "vehicle": {"vehicle", "automobile", "conveyance"},
    "interior": {"interior_room", "furniture", "kitchen", "interior_shop", "restaurant"},
    "event": {"concert", "performance", "celebration", "music", "musical_instrument"},
    "recreation": {"recreation", "sport", "games", "toy"},
    "art": {"art", "decoration"},
    "document": {"document", "screenshot", "printed_page", "chart"},
}
LABEL_TO_BROAD = {lab: cat for cat, labs in BROAD_CATEGORIES.items() for lab in labs}

# Coarse filter step 3 priority (lower first).
CATEGORY_PRIORITY = {
    "people": 1, "animal": 2, "event": 3, "food_drink": 3, "recreation": 3,
    "sky": 4, "water": 4, "plant": 4,
    "built": 5, "street": 5, "interior": 5, "vehicle": 5, "art": 5,
}
MOMENT_GAP_MIN = 20
SEND_MAX = 10


# ---------- bucket ----------
# The phone picks the bucket itself; this stand-in maps the prototype's broad
# categories to the seven ids by simple majority (ties: list order).
_BUCKET_OF = {
    "food_drink": "food", "sky": "scenery", "plant": "scenery", "water": "scenery",
    "animal": "animal", "built": "city", "street": "city", "vehicle": "city",
    "interior": "indoor", "tableware": "indoor", "event": "event", "recreation": "event",
}
_BUCKET_ORDER = ["food", "scenery", "animal", "city", "indoor", "event", "other"]


def guess_bucket(photos) -> str:
    counts = {}
    for p in photos:
        for b in {_BUCKET_OF.get(c, "other") for c in p["labels"]}:
            counts[b] = counts.get(b, 0) + 1
    if not counts:
        return "other"
    return max(_BUCKET_ORDER, key=lambda b: (counts.get(b, 0), -_BUCKET_ORDER.index(b)))


# ---------- data ----------

class Data:
    def __init__(self, prototype: Path):
        self.root = prototype
        results = json.loads((prototype / "output" / "results.json").read_text(encoding="utf-8"))
        analysis = json.loads((prototype / "cache" / "analysis.json").read_text(encoding="utf-8"))
        self.inv = {p["file"]: p for p in results["inventory"]}
        self.proto_groups = results["groups"]
        self.ana = {p["file"]: p for p in analysis["photos"]}
        self.fp = analysis["featureprint_distances"]
        geo_path = prototype / "cache" / "geocode.json"
        self.geo = json.loads(geo_path.read_text(encoding="utf-8")) if geo_path.exists() else {}

    def fpd(self, a: str, b: str) -> float:
        i, j = sorted((self.ana[a]["index"], self.ana[b]["index"]))
        return float(self.fp.get("{}-{}".format(i, j), 9))

    def taken(self, f: str) -> datetime:
        return datetime.fromisoformat(self.inv[f]["taken_local"])

    def broad(self, f: str) -> List[str]:
        cats = []
        for lab in self.inv[f].get("labels_high_precision") or []:
            cat = LABEL_TO_BROAD.get(lab)
            if cat and cat not in cats:
                cats.append(cat)
        return cats

    def aesthetics(self, f: str) -> float:
        value = (self.inv[f].get("aesthetics") or {}).get("overall")
        return 0.5 if value is None else float(value)

    def score(self, f: str) -> float:
        # 0.6 quality + 0.2 people + 0.2 favorite/edited (always false here)
        return 0.6 * self.aesthetics(f) + 0.2 * (1.0 if "people" in self.broad(f) else 0.0)

    def place(self, files: List[str]) -> Optional[str]:
        pts = [self.inv[f]["gps"] for f in files if self.inv[f].get("gps")]
        if not pts:
            return None
        lat = sum(p["lat"] for p in pts) / len(pts)
        lon = sum(p["lon"] for p in pts) / len(pts)
        best, best_d = None, None
        for key, entry in self.geo.items():
            try:
                klat, klon = (float(x) for x in key.split(","))
            except ValueError:
                continue
            d = math.hypot(klat - lat, klon - lon)
            if best_d is None or d < best_d:
                best, best_d = entry, d
        if best is None or best_d > 0.02:
            return None
        aoi = best.get("areas_of_interest") or []
        return aoi[0] if aoi else (best.get("locality") or best.get("mk_city"))

    def image(self, f: str) -> bytes:
        rel = self.ana[f]["jpeg"]["display"]
        with Image.open(self.root / "cache" / rel) as img:
            img = img.convert("RGB")
            img.thumbnail((768, 768))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)  # no exif= -> no EXIF written
        return buf.getvalue()


# ---------- coarse filter (simple port of the coarse-filter rules) ----------

def coarse_filter(data: Data, files: List[str]) -> List[str]:
    files = sorted(files, key=data.taken)
    chosen: List[str] = []

    def add(f):
        if f not in chosen and len(chosen) < SEND_MAX:
            chosen.append(f)

    # 1: favorites / edited, at most 3 -- the prototype has none.
    # 2: at least one per moment (a gap > 20 min starts a new moment).
    moments: List[List[str]] = []
    for f in files:
        if moments and (data.taken(f) - data.taken(moments[-1][-1])).total_seconds() <= MOMENT_GAP_MIN * 60:
            moments[-1].append(f)
        else:
            moments.append([f])
    if len(moments) > SEND_MAX:
        keep = [moments[0], moments[-1]]
        rest = sorted(moments[1:-1], key=lambda m: -len(m))
        keep += rest[: SEND_MAX - 2]
        moments = sorted(keep, key=lambda m: data.taken(m[0]))
    for m in moments:
        add(max(m, key=data.score))
    # 3: new categories first, up to 4, by category priority.
    added = 0
    covered = {c for f in chosen for c in data.broad(f)}
    missing = sorted({c for f in files for c in data.broad(f)} - covered,
                     key=lambda c: CATEGORY_PRIORITY.get(c, 9))
    for cat in missing:
        if added >= 4 or len(chosen) >= SEND_MAX:
            break
        pool = [f for f in files if cat in data.broad(f) and f not in chosen]
        if pool:
            add(max(pool, key=data.score))
            added += 1
    # 4: people floor; a second, least similar people photo if it is different.
    people = [f for f in files if "people" in data.broad(f)]
    if people and not any(f in people for f in chosen):
        add(max(people, key=data.score))
    chosen_people = [f for f in chosen if f in people]
    others = [f for f in people if f not in chosen]
    if chosen_people and others:
        far = max(others, key=lambda f: min(data.fpd(f, c) for c in chosen_people))
        if min(data.fpd(far, c) for c in chosen_people) > 0.35:
            add(far)
    # 5: fill by MMR, half score, half dissimilarity.
    while len(chosen) < SEND_MAX:
        pool = [f for f in files if f not in chosen]
        if not pool:
            break
        best = max(pool, key=lambda f: 0.5 * data.score(f)
                   + 0.5 * min(1.0, min(data.fpd(f, c) for c in chosen) if chosen else 1.0))
        add(best)
    return sorted(chosen, key=data.taken)


# ---------- one group ----------

def build(data: Data, name: str, files: List[str], dup_of: Dict[str, str],
          recent: List[str]) -> tuple:
    unique = [f for f in files if f not in dup_of]
    represents = {f: 1 for f in unique}
    for f, anchor in dup_of.items():
        if f in files and anchor in represents:
            represents[anchor] += 1
    send = unique if len(unique) <= SEND_MAX else coarse_filter(data, unique)
    send = sorted(send, key=data.taken)
    covers_all = all(f in send or dup_of.get(f) in send for f in files)
    first, last = data.taken(files[0]), data.taken(files[-1])
    meta_photos, images = [], {}
    for i, f in enumerate(send, start=1):
        pid = "P{}".format(i)
        cats = data.broad(f)
        meta_photos.append({
            "id": pid,
            "taken_at": data.inv[f]["taken_local"],
            "labels": cats,
            # The prototype has no face counts; 1 when Vision saw people.
            "faces": 1 if "people" in cats else 0,
            "favorite": False,
            "edited": False,
            "represents": represents[f],
        })
        images[pid] = data.image(f)
    meta = {
        "group_key": "h1_{}_{}".format(first.date().isoformat(), Path(files[0]).stem),
        "date": first.date().isoformat(),
        "weekday": first.isoweekday(),
        "start_time": first.strftime("%H:%M"),
        "end_time": last.strftime("%H:%M"),
        "place": data.place(files),
        "raw_count": len(files),
        "covers_all": covers_all,
        "recent_question_ids": list(recent[-5:]),
        "bucket": guess_bucket(meta_photos),
        "photos": meta_photos,
    }
    mapping = {p["id"]: f for p, f in zip(meta_photos, send)}
    return meta, images, mapping


def submit_and_wait(base: str, token: str, meta: dict, images: Dict[str, bytes],
                    timeout_s: float = 320) -> tuple:
    headers = {"Authorization": "Bearer " + token}
    files = [("photos", (pid + ".jpg", data, "image/jpeg")) for pid, data in images.items()]
    t0 = time.monotonic()
    with httpx.Client(timeout=30) as client:
        r = client.post(base + "/collections/compose", headers=headers,
                        data={"meta": json.dumps(meta, ensure_ascii=False)}, files=files)
        if r.status_code != 202:
            return {"status": "http_" + str(r.status_code), "body": r.text[:300]}, 0
        job = r.json()["job_id"]
        while time.monotonic() - t0 < timeout_s:
            time.sleep(5)
            r = client.get(base + "/collections/jobs/" + job, headers=headers)
            if r.status_code == 429:
                time.sleep(10)
                continue
            body_ = r.json()
            if r.status_code != 200 or body_.get("status") != "running":
                return body_, int((time.monotonic() - t0) * 1000)
    return {"status": "client_timeout"}, int((time.monotonic() - t0) * 1000)


def log_entry(log_path: Path, group_key: str) -> Optional[dict]:
    if not log_path.exists():
        return None
    found = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("group_key") == group_key:
            found = entry
    return found


# ---------- H2 re-check ----------

def recheck(meta: dict, entry: dict, memory: dict) -> List[str]:
    """Program checks re-run on a finished memory. Empty list == clean."""
    problems: List[str] = []
    raw_a = [c for c in entry.get("calls", []) if c["call"] == "a" and not c.get("error")]
    if not raw_a:
        return ["no successful call A in the log"]
    ids = [p["id"] for p in sorted(meta["photos"], key=lambda p: compose._sort_key(
        compose.parse_time(p["taken_at"])))]
    a = check_a.validate(check_a.parse_json(raw_a[-1]["raw"]), ids)
    detail = memory["ending"]["detail"] or memory["push"]["body"]
    src = memory["generation"]["bodySource"]
    cand = body.Candidate(text=detail, source=src, photo_ids=[], uses=[detail])
    if src != "fallback_F0":
        reason = body._reject_reason(cand, detail, a, body.program_counts(a))
        if reason:
            problems.append("detail: " + reason)
    q = memory["ending"]["question"]
    if body.char_len(q) > questions.MAX_CHARS:
        problems.append("question too long")
    word = questions._forbidden(q)
    if word:
        problems.append("question forbidden word")
    if memory["generation"]["questionSource"] == "llm":
        # Questions were offered for the cover before any switch.
        peak_time = compose.parse_time(
            next(p["taken_at"] for p in meta["photos"] if p["id"] == a.peak))
        daypart = compose.daypart_of(peak_time.hour) if peak_time else memory["daypart"]
        offered = {c.id: c for c in questions.candidates(a, meta, daypart)}
        qid = memory["ending"]["questionId"]
        if qid not in offered:
            problems.append("question id not offered")
        elif offered[qid].kind == "A":
            fill = (entry.get("question") or {}).get("fill", "")
            if fill not in offered[qid].menu:
                problems.append("question fill not in menu")
    return problems


# ---------- main ----------

def _token(env_name: str) -> str:
    token = os.environ.get(env_name, "").strip()
    if token:
        return token
    root_env = POC.parent.parent / ".env"
    if root_env.exists():
        for line in root_env.read_text(encoding="utf-8").splitlines():
            if line.startswith("BETA_TOKENS="):
                return line.split("=", 1)[1].split(",")[0].strip()
    raise SystemExit("no token: set {} or BETA_TOKENS in the repo-root .env".format(env_name))


def _cell(value) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prototype", required=True)
    ap.add_argument("--validator", required=True)
    ap.add_argument("--photos", required=True, help="input dir (checked to exist; pixels come from the prototype cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://localhost:8006")
    ap.add_argument("--token-env", default="BETA_TOKEN")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--spent", type=float, default=0.0, help="cost already spent in earlier runs")
    ap.add_argument("--label", default="run")
    ap.add_argument("--only", type=int, default=0, help="run only the first N groups (smoke test)")
    ap.add_argument("--no-slides", action="store_true")
    args = ap.parse_args(argv)

    prototype = Path(args.prototype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(args.photos).is_dir():
        raise SystemExit("photos dir not found")
    if str(out.resolve()).startswith(str(POC.parent.parent.resolve())):
        raise SystemExit("--out must be outside the repo")
    token = _token(args.token_env)
    data = Data(prototype)

    groups_path = out / "groups.json"
    subprocess.run([sys.executable, "-I", args.validator, str(prototype), str(groups_path)],
                   check=True, stdout=subprocess.DEVNULL)
    vg = json.loads(groups_path.read_text(encoding="utf-8"))
    dup_of = {p["file"]: p["dup"] for p in vg["photos"] if p.get("dup")}
    work = [("V1 " + g["day"] + " " + g["start"][11:16], g["files"], False)
            for g in vg["groups"] if g["ok"]]
    if args.only:
        work = work[: args.only]
    if not args.no_slides:
        slides = [g for g in data.proto_groups if g["id"].startswith("G12")]
        if slides:
            work.append(("slides " + slides[0]["id"], slides[0]["photos"], True))

    log_path = POC.parent.parent / "logs" / "photo-collections" / "compose.jsonl"
    spent = args.spent
    recent: List[str] = []
    rows = []
    for name, files, expect_reject in work:
        if spent > args.budget:
            print("budget reached: {:.4f} CNY -- stopping".format(spent))
            break
        files = sorted(files, key=data.taken)
        meta, images, mapping = build(data, name, files, dup_of, recent)
        result, wall_ms = submit_and_wait(args.base, token, meta, images)
        entry = log_entry(log_path, meta["group_key"]) or {}
        cost = float(entry.get("costCny") or 0.0)
        spent += cost
        row = {
            "group": name, "expect_reject": expect_reject, "sent": len(images),
            "raw_count": meta["raw_count"], "covers_all": meta["covers_all"],
            "place": meta["place"], "status": result.get("status"),
            "reason": result.get("reason", ""), "latency_ms": wall_ms, "cost_cny": round(cost, 5),
            "files": mapping,
        }
        memory = result.get("memory")
        if memory:
            recent.append(memory["ending"]["questionId"])
            row.update(
                selected=len(memory["story"]["photos"]),
                peak="{} ({})".format(memory["push"]["coverPhotoId"],
                                      mapping.get(memory["push"]["coverPhotoId"])),
                title=memory["push"]["title"],
                detail=memory["ending"]["detail"],
                push_body=memory["push"]["body"],
                body_source=memory["generation"]["bodySource"],
                question=memory["ending"]["question"],
                question_id=memory["ending"]["questionId"],
                question_source=memory["generation"]["questionSource"],
                service_latency_ms=memory["generation"]["latencyMs"],
                peak_rewrite=memory["generation"].get("peakRewrite"),
                peak_switched=memory["generation"].get("peakSwitched"),
                peak_switch=entry.get("peak_switch"),
                h2_problems=recheck(meta, entry, memory),
                body_rejected=(entry.get("body") or {}).get("rejected"),
                question_rejected=(entry.get("question") or {}).get("rejected"),
                facts=memory["facts"],
            )
        rows.append(row)
        print("{:<28} {:<9} {:<18} cost={:.5f} total={:.4f}".format(
            name, str(row["status"]), row.get("body_source") or row["reason"], cost, spent))

    pushable = [r for r in rows if not r["expect_reject"]]
    ok = [r for r in pushable if r["status"] == "ok"]
    n = len(pushable)
    f0 = sum(1 for r in ok if r["body_source"] == "fallback_F0")
    non_llm = sum(1 for r in ok if r["body_source"] != "llm")
    a_class = sum(1 for r in ok if str(r["question_id"]).startswith("A"))
    h2 = [(r["group"], p) for r in ok for p in r["h2_problems"]]
    summary = {
        "label": args.label, "groups_sent": len(rows), "pushable_groups": n,
        "ok": len(ok), "f0_fallback": f0, "non_llm_body": non_llm,
        "f0_share": round(f0 / n, 3) if n else None,
        "f0_share_of_ok": round(f0 / len(ok), 3) if ok else None,
        "non_llm_share_of_ok": round(non_llm / len(ok), 3) if ok else None,
        "rejected": [(r["group"], r["reason"]) for r in pushable if r["status"] != "ok"],
        "a_class": a_class, "a_share": round(a_class / len(ok), 3) if ok else None,
        "h2_problems": h2, "total_cost_cny": round(spent, 5),
        "peak_rewrite_calls": sum(1 for r in ok if r.get("peak_rewrite")),
        "peak_switched": sum(1 for r in ok if r.get("peak_switched")),
        "run_cost_cny": round(spent - args.spent, 5),
    }

    lines = [
        "# H1 summary ({})".format(args.label), "",
        "| group | sent | selected | peak | detail (push body) | bodySource | question | id | questionSource | status | latency s | cost CNY |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_cell(x) for x in [
            r["group"] + (" (expect reject)" if r["expect_reject"] else ""),
            r["sent"], r.get("selected", ""), r.get("peak", ""),
            r.get("push_body", ""), r.get("body_source", ""), r.get("question", ""),
            r.get("question_id", ""), r.get("question_source", ""),
            r["status"] + (" " + r["reason"] if r["reason"] else ""),
            round(r["latency_ms"] / 1000, 1), r["cost_cny"],
        ]) + " |")
    lines += [
        "", "## H2 / H3", "",
        "- pushable groups: {} (ok {})".format(n, len(ok)),
        "- F0 fallback detail: {} / {} = {}".format(f0, n, summary["f0_share"]),
        "- F0 among pushed (ok) groups: {} / {} = {}".format(f0, len(ok), summary["f0_share_of_ok"]),
        "- detail not written by the model (template or F0) among ok: {} / {} = {}".format(
            non_llm, len(ok), summary["non_llm_share_of_ok"]),
        "- pushable groups rejected by the rules: {}".format(summary["rejected"] or "none"),
        "- A-class questions: {} / {} = {}".format(a_class, len(ok), summary["a_share"]),
        "- H2 program re-check problems: {}".format(h2 or "none"),
        "- cover rewrite call made: {} / {}; cover switched to the body's photo: {} / {}".format(
            summary["peak_rewrite_calls"], len(ok), summary["peak_switched"], len(ok)),
        "- cost this run: {} CNY; cumulative: {} CNY".format(summary["run_cost_cny"], summary["total_cost_cny"]),
    ]
    stem = "h1_" + args.label
    (out / (stem + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / (stem + ".json")).write_text(json.dumps({"summary": summary, "rows": rows},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print("wrote", out / (stem + ".md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
