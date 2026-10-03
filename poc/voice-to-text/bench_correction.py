"""A/B/C: Qwen Flash and Qwen3.5 Flash against the current correction model.

The correction call in main.py:231 is the only paid, latency-visible API call
in the voice path, and it is hardwired to gemini-2.5-flash. This measures
whether a China-region Qwen Flash model can do the same job cheaper and
faster without over-editing.

Nothing in the service is changed. This script imports correction.py -- the
same module /transcribe calls -- so all three models get byte-identical
prompts and request shapes, and only the client, model id and (for Qwen) the
thinking switch differ.

Three things this deliberately does not do:

  1. It does not touch audio. The correction layer takes text, so the cases
     are transcripts. The three demo_data clips are macOS `say` TTS with no
     committed source text, so they would add real Whisper output but no
     ground truth to score against.
  2. It does not score with a judge model. A judge has its own opinion about
     what "minimal correction" means, which is the exact thing under test.
     Instead it emits mechanical flags (length ratio, dropped proper nouns,
     script drift) and a side-by-side sheet a human reads.
  3. It does not raise on a failed call. A model that refuses, truncates or
     times out on a case is the finding; that lands as a result row.

Usage:

    python3 bench_correction.py --dry-run                 # no spend, checks config
    python3 bench_correction.py --only zh_01 --repeat 1   # cheap smoke, 3 calls
    python3 bench_correction.py --repeat 3                # full matrix

Cases live in test_input/correction_cases.json (git-ignored, so the Chinese
transcripts may be non-ASCII even though this script may not be -- org
Rule 3). Everything lands in test_output/correction_bench/<timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import os
import statistics
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

import correction

load_dotenv()

HERE = Path(__file__).resolve().parent
CASES = HERE / "test_input" / "correction_cases.json"
OUT_ROOT = HERE / "test_output" / "correction_bench"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Prices are RMB per 1M tokens, China region, Bailian list prices.
#
# The Gemini rows were removed when the default moved to qwen3.5-flash: AI
# Studio 404s gemini-2.5-flash for newly issued keys, and the Vertex shim that
# was the only other route to it has been deleted. To bench a new candidate,
# add a row here -- any OpenAI-compatible endpoint works.
MODELS: Dict[str, Dict[str, Any]] = {
    "qwen-flash": {
        "label": "Qwen Flash",
        "model_env": "QWEN_FLASH_MODEL",
        "model": "qwen-flash",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "base_url": DASHSCOPE_BASE_URL,
        "api_key_env": "DASHSCOPE_API_KEY",
        "price_in": 0.15,
        "price_out": 1.50,
        "price_note": "Bailian list, China region",
        # Qwen3-series models expose a thinking mode. Correction is not a
        # reasoning task, and thinking would change both the latency and the
        # job being measured, so it is off for every Qwen row.
        "extra_body": {"enable_thinking": False},
    },
    "qwen3.5-flash": {
        "label": "Qwen3.5 Flash",
        "model_env": "QWEN35_FLASH_MODEL",
        "model": "qwen3.5-flash",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "base_url": DASHSCOPE_BASE_URL,
        "api_key_env": "DASHSCOPE_API_KEY",
        "price_in": 0.20,
        "price_out": 2.00,
        "price_note": "Bailian list, China region",
        "extra_body": {"enable_thinking": False},
    },
}

CATEGORIES = [
    "zh",
    "en",
    "zh_en_mixed",
    "proper_nouns",
    "homophone",
    "already_clean",
    "disfluency",
]

# ASCII stand-in, written only when the cases file is missing. The real corpus
# is Chinese and mixed Chinese/English, so it lives only in the git-ignored
# cases file (org Rule 3).
TEMPLATE = [
    {
        "id": "en_01",
        "category": "en",
        "text": "so i think we should ship the beta next week and then "
                "collect feedback before we lock the pricing",
        "must_keep": ["beta", "pricing"],
        "notes": "pure English, no punctuation -- should gain commas and a period only",
    },
    {
        "id": "already_clean_01",
        "category": "already_clean",
        "text": "The meeting is on Tuesday at 3 PM. Please bring the report.",
        "must_keep": ["Tuesday", "3 PM"],
        "notes": "already correct -- the right answer is to return it unchanged",
    },
    {
        "id": "proper_nouns_01",
        "category": "proper_nouns",
        "text": "we synced with the Altlook team about the Q3 roadmap and the "
                "Kubernetes migration",
        "must_keep": ["Q3", "Kubernetes"],
        "notes": "Altlook -> Outlook is a wanted fix; Q3 and Kubernetes must survive",
    },
]


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(TEMPLATE, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"No cases file -- wrote a template to {path}.")
        print("Edit it, then run again. The file is git-ignored, so its")
        print("transcripts may be non-ASCII even though this script may not be.")
        sys.exit(0)
    cases = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for case in cases:
        if not case.get("id") or not case.get("text"):
            sys.exit(f"case {case!r} needs both 'id' and 'text'")
        if case["id"] in seen:
            sys.exit(f"duplicate case id {case['id']!r}")
        seen.add(case["id"])
        if case.get("category") not in CATEGORIES:
            sys.exit(f"case {case['id']!r} has unknown category "
                     f"{case.get('category')!r}; expected one of {CATEGORIES}")
        # A planted error that is not actually in the text scores as "fixed"
        # on every model forever, silently inflating the headline number.
        for item in case.get("planted", []):
            if "wrong" not in item or "right" not in item:
                sys.exit(f"case {case['id']!r}: each planted entry needs "
                         f"'wrong' and 'right', got {item!r}")
            if item["wrong"] not in case["text"]:
                sys.exit(f"case {case['id']!r}: planted error "
                         f"{item['wrong']!r} does not appear in its own text")
            if item["right"] in case["text"]:
                sys.exit(f"case {case['id']!r}: the correct form "
                         f"{item['right']!r} is already in the text")
    return cases


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #
def resolve(key: str) -> Dict[str, Any]:
    """Merge the registry entry with env overrides. Every field is
    overridable so a new Qwen snapshot can be benched without editing code."""
    spec = dict(MODELS[key])
    spec["key"] = key
    spec["model"] = os.getenv(spec["model_env"]) or spec["model"]
    spec["base_url"] = os.getenv(spec["base_url_env"]) or spec["base_url"]
    spec["api_key"] = os.getenv(spec["api_key_env"]) or ""
    return spec


def make_client(spec: Dict[str, Any], timeout: float) -> OpenAI:
    return OpenAI(api_key=spec["api_key"], base_url=spec["base_url"], timeout=timeout)


def cost_rmb(spec: Dict[str, Any], prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1e6) * spec["price_in"] + \
           (completion_tokens / 1e6) * spec["price_out"]


# --------------------------------------------------------------------------- #
# Mechanical flags
# --------------------------------------------------------------------------- #
def has_cjk(text: str) -> bool:
    """CJK Unified Ideographs, written as escapes so this file stays ASCII
    (org Rule 3) even though what it detects is not."""
    return any(chr(0x4E00) <= ch <= chr(0x9FFF) for ch in text)


def strip_punct(text: str) -> str:
    """Drop punctuation, keep everything else including spaces.

    Adding punctuation is the model's job, so a planted span that gets a comma
    inserted into it -- a planted span coming back with a comma in the
    middle of it -- is a fix, not a
    miss. Matching on the raw string scored those as failures and undercounted
    every model, unevenly, since each puts commas in different places.

    Spaces are deliberately kept: the English cases plant compounds the ASR
    split ("end point" -> "endpoint"), and stripping spaces would collapse the
    wrong and right forms into the same string.
    """
    return "".join(ch for ch in text
                   if not unicodedata.category(ch).startswith("P"))


def score_planted(case: Dict[str, Any], out: str) -> tuple:
    """Which declared errors the model actually fixed.

    A case may plant several errors at once, which is the point -- fixing one
    in isolation does not predict fixing three in the same sentence. An error
    counts as fixed only when the correct form is present AND the wrong form
    is gone: a model that emits the right word but leaves the wrong one
    elsewhere has not corrected the sentence.
    """
    out_n = strip_punct(out)
    fixed, missed = [], []
    for item in case.get("planted", []):
        wrong, right = item["wrong"], item["right"]
        hit = (strip_punct(right) in out_n and strip_punct(wrong) not in out_n)
        (fixed if hit else missed).append(wrong)
    return fixed, missed


def flags_for(case: Dict[str, Any], output: Optional[str]) -> Dict[str, Any]:
    """Facts about an output, not judgements. These sort the side-by-side
    sheet so the human reader looks at the suspicious rows first."""
    raw = case["text"]
    if output is None:
        return {
            "len_ratio": None,
            "must_keep_missed": [],
            "planted_fixed": [],
            "planted_missed": [],
            "identical_to_input": False,
            "script_drift": False,
        }
    out = output.strip()
    # ASCII tokens compare case-insensitively: turning "api" into "API" is a
    # wanted fix, not a dropped term. CJK tokens have no case, so an exact
    # match there stays exact.
    missed = [t for t in case.get("must_keep", [])
              if (t.lower() not in out.lower() if t.isascii() else t not in out)]
    planted_fixed, planted_missed = score_planted(case, out)
    return {
        "planted_fixed": planted_fixed,
        "planted_missed": planted_missed,
        # Well below 1 means the model summarised; well above means it padded.
        "len_ratio": round(len(out) / max(len(raw), 1), 3),
        # Proper nouns, and the filler words the prompt says to preserve.
        "must_keep_missed": missed,
        # The correct answer for the already_clean category, a smell elsewhere.
        "identical_to_input": out == raw.strip(),
        # Chinese in, no Chinese out (or the reverse) means it translated.
        "script_drift": has_cjk(raw) != has_cjk(out),
    }


def variants_of(records, case_id: str, model_key: str) -> List[str]:
    """The distinct outputs one model gave for one case across repeats.

    This is why --repeat exists. qwen-flash translated an English case into
    Chinese on one run of three and kept English on another; a sheet showing
    only the first attempt would have called it clean. Order is preserved so
    the first variant shown is the first observed.
    """
    seen = []
    for r in records:
        if r["case_id"] == case_id and r["model_key"] == model_key \
                and r["status"] == "ok":
            text = (r["output"] or "").strip()
            if text not in seen:
                seen.append(text)
    return seen


def annotate_instability(records) -> None:
    """Mark every record whose (case, model) did not answer the same way
    twice. Non-determinism on a correction task is a defect in itself: the
    same recording would come back different on a retry."""
    pairs = {(r["case_id"], r["model_key"]) for r in records}
    unstable = {p for p in pairs if len(variants_of(records, *p)) > 1}
    for r in records:
        r["unstable"] = (r["case_id"], r["model_key"]) in unstable


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_matrix(cases, specs, clients, repeat, warmup) -> List[Dict[str, Any]]:
    """Attempt-major ordering: every model is retried in the same pass, so a
    slow minute on the network lands on all three rather than on whichever
    model happened to be scheduled then."""
    records: List[Dict[str, Any]] = []

    if warmup and cases:
        for spec in specs:
            print(f"  warmup {spec['key']} ...", file=sys.stderr)
            correction.correct_once(
                cases[0]["text"], clients[spec["key"]], spec["model"],
                extra_body=spec["extra_body"],
            )

    total = repeat * len(cases) * len(specs)
    done = 0
    for attempt in range(repeat):
        for case in cases:
            for spec in specs:
                result = correction.correct_once(
                    case["text"], clients[spec["key"]], spec["model"],
                    extra_body=spec["extra_body"],
                )
                done += 1
                records.append({
                    "case_id": case["id"],
                    "category": case.get("category", ""),
                    "model_key": spec["key"],
                    "model": spec["model"],
                    "attempt": attempt,
                    "input": case["text"],
                    "output": result.text,
                    "status": result.status,
                    "error": result.error,
                    "latency_ms": round(result.latency_ms, 1),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost_rmb": cost_rmb(spec, result.prompt_tokens,
                                         result.completion_tokens),
                    **flags_for(case, result.text),
                })
                mark = "ok" if result.status == "ok" else "FAIL"
                print(f"  [{done}/{total}] {case['id']:<20} {spec['key']:<14} "
                      f"{mark} {result.latency_ms / 1000:.2f}s", file=sys.stderr)
    return records


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def summarise(records, specs) -> List[Dict[str, Any]]:
    """Per-model rollup. Latency is reported in seconds because that is the
    unit the product decision is made in; the raw ms stay in results.jsonl."""
    rows = []
    for spec in specs:
        mine = [r for r in records if r["model_key"] == spec["key"]]
        ok = [r for r in mine if r["status"] == "ok"]
        lat = sorted(r["latency_ms"] for r in ok)
        n_cases = len({r["case_id"] for r in mine})
        mean_cost = statistics.mean([r["cost_rmb"] for r in ok]) if ok else 0.0
        rows.append({
            "model_key": spec["key"],
            "label": spec["label"],
            "model": spec["model"],
            "calls": len(mine),
            "errors": len(mine) - len(ok),
            "latency_median_s": round(statistics.median(lat) / 1000, 3) if lat else None,
            "latency_p95_s": round(lat[max(0, int(len(lat) * 0.95) - 1)] / 1000, 3)
                             if lat else None,
            "mean_prompt_tokens": round(statistics.mean(
                [r["prompt_tokens"] for r in ok]), 1) if ok else 0,
            "mean_completion_tokens": round(statistics.mean(
                [r["completion_tokens"] for r in ok]), 1) if ok else 0,
            "mean_cost_rmb_per_correction": round(mean_cost, 6),
            "cost_rmb_per_1k_corrections": round(mean_cost * 1000, 3),
            "cost_rmb_full_pass": round(mean_cost * n_cases, 5),
            "planted_fixed": sum(len(r["planted_fixed"]) for r in ok),
            "planted_total": sum(len(r["planted_fixed"]) + len(r["planted_missed"])
                                 for r in ok),
            "planted_fix_rate": (
                round(sum(len(r["planted_fixed"]) for r in ok)
                      / max(1, sum(len(r["planted_fixed"]) + len(r["planted_missed"])
                                   for r in ok)), 3)
                if ok else None),
            "must_keep_misses": sum(len(r["must_keep_missed"]) for r in ok),
            "script_drifts": sum(1 for r in ok if r["script_drift"]),
            # Counted per case, not per call, so one flaky case scores 1
            # however many repeats disagreed.
            "unstable_cases": len({r["case_id"] for r in mine
                                   if r.get("unstable")}),
            "price_in_rmb_per_m": spec["price_in"],
            "price_out_rmb_per_m": spec["price_out"],
            "price_note": spec["price_note"],
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k))
                             for k in fields})


def diff_html(before: str, after: str) -> str:
    """Character-level diff of output against input. Over-editing is the main
    failure mode under test and it is invisible in two walls of text; marking
    every insertion and deletion makes it a glance."""
    out = []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.append(html.escape(after[j1:j2]))
        elif tag == "insert":
            out.append(f'<ins>{html.escape(after[j1:j2])}</ins>')
        elif tag == "delete":
            out.append(f'<del>{html.escape(before[i1:i2])}</del>')
        else:
            out.append(f'<del>{html.escape(before[i1:i2])}</del>'
                       f'<ins>{html.escape(after[j1:j2])}</ins>')
    return "".join(out)


CSS = """
body{font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
 margin:0;padding:24px;background:#fafaf9;color:#1c1917}
h1{font-size:20px;margin:0 0 4px}
.meta{color:#78716c;font-size:12px;margin-bottom:20px}
table.sum{border-collapse:collapse;margin-bottom:28px;background:#fff;
 box-shadow:0 1px 2px rgba(0,0,0,.06)}
table.sum th,table.sum td{border:1px solid #e7e5e4;padding:6px 10px;text-align:right}
table.sum th{background:#f5f5f4;font-weight:600;text-align:right}
table.sum td:first-child,table.sum th:first-child{text-align:left}
.case{background:#fff;border:1px solid #e7e5e4;border-radius:6px;
 margin-bottom:18px;overflow:hidden}
.case>header{padding:8px 12px;background:#f5f5f4;border-bottom:1px solid #e7e5e4}
.cid{font-weight:600}
.cat{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;
 background:#e7e5e4;color:#57534e;margin-left:8px}
.notes{color:#78716c;font-size:12px;margin-top:2px}
.input{padding:10px 12px;border-bottom:1px solid #e7e5e4;background:#fffbeb}
.label{font-size:11px;text-transform:uppercase;letter-spacing:.04em;
 color:#a8a29e;margin-bottom:3px}
.cols{display:flex;flex-wrap:wrap}
.col{flex:1 1 300px;min-width:280px;padding:10px 12px;
 border-right:1px solid #e7e5e4}
.col:last-child{border-right:none}
.col h3{font-size:12px;margin:0 0 6px;color:#57534e}
.stats{font-size:11px;color:#78716c;margin-top:8px}
ins{background:#dcfce7;text-decoration:none}
del{background:#fee2e2;text-decoration:line-through;opacity:.7}
.flag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:3px;
 background:#fee2e2;color:#991b1b;margin:2px 4px 0 0}
.flag.ok{background:#dcfce7;color:#166534}
.fail{color:#991b1b;font-weight:600}
.unstable{font-size:11px;font-weight:600;color:#9a3412;background:#ffedd5;
 padding:2px 7px;border-radius:3px;display:inline-block;margin-bottom:6px}
.variant{border-left:3px solid #fdba74;padding-left:8px;margin:6px 0}
.vn{color:#a8a29e;font-size:11px;margin-right:6px}
"""


def write_html(path, records, cases, specs, summary, meta) -> None:
    by = {}
    for r in records:
        by.setdefault((r["case_id"], r["model_key"]), []).append(r)

    parts = [f"<style>{CSS}</style>",
             "<h1>Correction model comparison</h1>",
             f"<div class='meta'>{html.escape(meta)}</div>"]

    # Summary
    cols = ["planted_fix_rate", "latency_median_s", "latency_p95_s", "mean_prompt_tokens",
            "mean_completion_tokens", "cost_rmb_per_1k_corrections",
            "planted_fixed", "planted_total", "planted_fix_rate",
        "must_keep_misses", "script_drifts", "unstable_cases", "errors"]
    heads = ["planted fix rate", "median latency (s)", "p95 latency (s)",
             "tokens in", "tokens out",
             "RMB / 1k corrections", "proper-noun misses", "script drifts",
             "unstable cases", "errors"]
    parts.append("<table class='sum'><tr><th>model</th>" +
                 "".join(f"<th>{h}</th>" for h in heads) + "</tr>")
    for row in summary:
        note = f" <span style='color:#a8a29e'>({row['price_note']})</span>" \
               if "ESTIMATE" in row["price_note"] else ""
        cells = "".join(f"<td>{'' if row[c] is None else row[c]}</td>" for c in cols)
        parts.append(f"<tr><td>{html.escape(row['label'])} "
                     f"<code>{html.escape(row['model'])}</code>{note}</td>{cells}</tr>")
    parts.append("</table>")

    # Per case
    for case in cases:
        parts.append("<div class='case'><header>"
                     f"<span class='cid'>{html.escape(case['id'])}</span>"
                     f"<span class='cat'>{html.escape(case.get('category',''))}</span>"
                     f"<div class='notes'>{html.escape(case.get('notes',''))}</div>"
                     "</header>")
        parts.append(f"<div class='input'><div class='label'>input</div>"
                     f"{html.escape(case['text'])}</div><div class='cols'>")
        for spec in specs:
            runs = by.get((case["id"], spec["key"]), [])
            parts.append(f"<div class='col'><h3>{html.escape(spec['label'])}</h3>")
            ok = [r for r in runs if r["status"] == "ok"]
            if not ok:
                err = runs[0]["error"] if runs else "not run"
                parts.append(f"<div class='fail'>FAILED</div>"
                             f"<div class='stats'>{html.escape(str(err))}</div></div>")
                continue
            first = ok[0]
            seen = variants_of(records, case["id"], spec["key"])
            if len(seen) > 1:
                # Non-determinism is the finding, so show every answer the
                # model gave rather than an arbitrary one of them.
                parts.append(f"<div class='unstable'>{len(seen)} different "
                             f"answers across {len(ok)} runs</div>")
                for i, variant in enumerate(seen, 1):
                    parts.append(f"<div class='variant'><span class='vn'>#{i}"
                                 f"</span>{diff_html(case['text'], variant)}</div>")
            else:
                parts.append(diff_html(case["text"], (first["output"] or "").strip()))
            lat = statistics.median([r["latency_ms"] for r in ok]) / 1000
            parts.append(
                f"<div class='stats'>{lat:.2f}s &middot; "
                f"{first['prompt_tokens']}+{first['completion_tokens']} tok &middot; "
                f"{first['cost_rmb'] * 1000:.4f} RMB/1k</div>")
            for label, bad in (
                # Planted errors first: this is the accuracy number, and on a
                # multi-error case it is the whole point of the row.
                (f"fixed {len(first['planted_fixed'])}/"
                 f"{len(first['planted_fixed']) + len(first['planted_missed'])}"
                 + (f" -- missed: {', '.join(first['planted_missed'])}"
                    if first["planted_missed"] else ""),
                 bool(first["planted_missed"])),
                (f"len ratio {first['len_ratio']}",
                 first["len_ratio"] is not None and
                 not 0.8 <= first["len_ratio"] <= 1.6),
                (f"dropped: {', '.join(first['must_keep_missed'])}"
                 if first["must_keep_missed"] else "kept all",
                 bool(first["must_keep_missed"])),
                ("unchanged" if first["identical_to_input"] else "edited",
                 first["identical_to_input"] !=
                 (case.get("category") == "already_clean")),
                ("script drift", first["script_drift"]),
            ):
                if label == "script drift" and not first["script_drift"]:
                    continue
                if label.startswith("fixed 0/0"):
                    continue    # a case with nothing planted, e.g. already_clean
                parts.append(f"<span class='flag{'' if bad else ' ok'}'>"
                             f"{html.escape(label)}</span>")
            parts.append("</div>")
        parts.append("</div></div>")

    path.write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_price_overrides(values, specs) -> None:
    for raw in values or []:
        if "=" not in raw or "/" not in raw:
            sys.exit(f"--price wants KEY=IN/OUT, got {raw!r}")
        key, pair = raw.split("=", 1)
        pin, pout = pair.split("/", 1)
        for spec in specs:
            if spec["key"] == key:
                spec["price_in"], spec["price_out"] = float(pin), float(pout)
                spec["price_note"] = "override via --price"
                break
        else:
            sys.exit(f"--price: unknown model key {key!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated subset of: " + ", ".join(MODELS))
    ap.add_argument("--repeat", type=int, default=3,
                    help="calls per (case, model); latency is the median")
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--only", default=None,
                    help="run one case id, or every case in one category")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--price", action="append",
                    help="override a price row, e.g. baseline=2.13/17.75")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--no-warmup", action="store_true",
                    help="keep the first call of each model in the numbers")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve config and cases, make no API calls")
    args = ap.parse_args()

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for key in keys:
        if key not in MODELS:
            sys.exit(f"unknown model key {key!r}; known: {', '.join(MODELS)}")
    specs = [resolve(k) for k in keys]
    parse_price_overrides(args.price, specs)

    if not correction.CORRECTION_PROMPT:
        sys.exit("prompts/correction.txt is missing -- nothing to send.")

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases
                 if c["id"] == args.only or c.get("category") == args.only]
        if not cases:
            sys.exit(f"--only {args.only!r} matched no case id or category")

    print("Resolved configuration:")
    for spec in specs:
        auth = f"{spec['api_key_env']}={'set' if spec['api_key'] else 'MISSING'}"
        print(f"  {spec['key']:<16} model={spec['model']:<22} "
              f"base_url={spec['base_url']}")
        print(f"  {'':<16} {auth}  "
              f"price={spec['price_in']}/{spec['price_out']} RMB per M "
              f"({spec['price_note']})")
        if spec["extra_body"]:
            print(f"  {'':<14} extra_body={spec['extra_body']}")
    calls = args.repeat * len(cases) * len(specs)
    warmups = 0 if args.no_warmup else len(specs)
    print(f"\n{len(cases)} cases x {len(specs)} models x {args.repeat} repeats "
          f"= {calls} calls (+{warmups} warmup)")
    print(f"prompt: {len(correction.CORRECTION_PROMPT)} chars from "
          f"prompts/correction.txt")

    if args.dry_run:
        print("\n--dry-run: no API calls made.")
        return

    missing = [s["api_key_env"] for s in specs if not s["api_key"]]
    if missing:
        sys.exit(f"\nmissing API key(s) in .env: {', '.join(sorted(set(missing)))}")

    out_dir = args.out_dir or (OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    clients = {s["key"]: make_client(s, args.timeout) for s in specs}
    print()
    started = time.time()
    records = run_matrix(cases, specs, clients, args.repeat, not args.no_warmup)
    elapsed = time.time() - started

    annotate_instability(records)
    summary = summarise(records, specs)

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_csv(out_dir / "results.csv", records, [
        "case_id", "category", "model_key", "model", "attempt", "status",
        "latency_ms", "prompt_tokens", "completion_tokens", "cost_rmb",
        "len_ratio", "planted_fixed", "planted_missed",
        "identical_to_input", "script_drift", "unstable", "error",
    ])
    write_csv(out_dir / "summary.csv", summary, [
        "model_key", "label", "model", "calls", "errors",
        "latency_median_s", "latency_p95_s", "mean_prompt_tokens",
        "mean_completion_tokens", "mean_cost_rmb_per_correction",
        "cost_rmb_per_1k_corrections", "cost_rmb_full_pass",
        "must_keep_misses", "script_drifts", "unstable_cases",
        "price_in_rmb_per_m", "price_out_rmb_per_m", "price_note",
    ])

    meta = (f"{datetime.now():%Y-%m-%d %H:%M}  |  {len(cases)} cases  |  "
            f"repeat={args.repeat}  |  wall {elapsed:.0f}s  |  "
            f"prompt: prompts/correction.txt (unmodified)")
    write_html(out_dir / "compare.html", records, cases, specs, summary, meta)

    print(f"\n{'model':<18}{'fix rate':>10}{'median s':>10}{'p95 s':>8}"
          f"{'RMB/1k':>10}{'misses':>8}{'unstable':>10}{'errors':>8}")
    for row in summary:
        fr = row['planted_fix_rate']
        print(f"{row['model_key']:<18}"
              f"{('%d/%d' % (row['planted_fixed'], row['planted_total'])) if row['planted_total'] else '-':>10}"
              f"{row['latency_median_s'] or 0:>10.2f}"
              f"{row['latency_p95_s'] or 0:>8.2f}"
              f"{row['cost_rmb_per_1k_corrections']:>10.2f}"
              f"{row['must_keep_misses']:>8}{row['unstable_cases']:>10}"
              f"{row['errors']:>8}")
    total = sum(r["cost_rmb"] for r in records)
    print(f"\nthis run cost about {total:.4f} RMB")
    print(f"wrote {out_dir}")
    print(f"open {out_dir / 'compare.html'}")


if __name__ == "__main__":
    main()
