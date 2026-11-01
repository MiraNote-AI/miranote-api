"""In-memory job store for compose: submit returns at once, a worker thread
runs the pipeline, polls read the result.

Jobs live only in this process (a restart forgets them; the phone resubmits
on 404). A job is kept TTL seconds after submission. A job still running
past CAP seconds reads as a timeout error even if its thread is stuck.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

import config
import texts


class JobStore:
    def __init__(self, ttl: float = config.JOB_TTL_SECONDS, cap: float = config.JOB_CAP_SECONDS,
                 workers: int = config.WORKERS, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl
        self._cap = cap
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: Dict[str, dict] = {}
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="compose")

    def _purge(self, now: float) -> None:
        for job_id in [j for j, v in self._jobs.items() if now - v["created"] > self._ttl]:
            del self._jobs[job_id]

    def submit(self, fn: Callable[[float], dict]) -> str:
        """Run fn(deadline) in a worker; deadline is on time.monotonic()."""
        job_id = uuid.uuid4().hex
        now = self._clock()
        deadline = time.monotonic() + self._cap
        with self._lock:
            self._purge(now)
            self._jobs[job_id] = {"created": now, "result": None}
        self._pool.submit(self._run, job_id, fn, deadline)
        return job_id

    def _run(self, job_id: str, fn: Callable[[float], dict], deadline: float) -> None:
        try:
            result = fn(deadline)
        except Exception:  # noqa: BLE001 -- a crashed job must still answer
            traceback.print_exc()
            result = {"status": "error", "reason": "model_error",
                      "message": texts.message("model_error")}
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job["result"] is None:
                job["result"] = result

    def get(self, job_id: str) -> Optional[dict]:
        """The job's current body, or None when unknown or expired."""
        now = self._clock()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if now - job["created"] > self._ttl:
                del self._jobs[job_id]
                return None
            if job["result"] is not None:
                return job["result"]
            if now - job["created"] > self._cap:
                job["result"] = {"status": "error", "reason": "timeout",
                                 "message": texts.message("timeout")}
                return job["result"]
            return {"status": "running"}
