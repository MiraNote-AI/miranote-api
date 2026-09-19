"""Apple Vision background removal, via a resident Swift helper.

Replaces rembg's whole-foreground matte. rembg[cpu] takes 12.55 s per image on
this machine; `GenerateForegroundInstanceMaskRequest` takes 0.03 s and, on the
17-image bench, produced a cleaner matte on the cluttered cases (see
test_output/vision/RESULTS.md).

Vision has no Python binding, so the work happens in vision_bench/vision_bench
(`--serve`), kept resident because the model load costs ~1.4 s and a process per
request would pay it every time. This module owns that process. The helper uses
the Swift-native Vision API, so it needs macOS 15+ to build and to run.

Nothing here falls back to rembg, and since the service went Vision-only nothing
above it does either: a failure is a failure, raised from here and answered as a
503 by main.py. The service keeps rembg reachable by name (?mode=auto,
matte="rembg") and one config edit away from being the default again -- see
ROLLING BACK TO REMBG in config.py -- but it is never substituted in silently.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading

import config


class VisionMatteUnavailable(RuntimeError):
    """The Vision helper could not produce a matte.

    Its own class so main.py can answer 503 with the reason -- "not built",
    "helper exited", "no foreground instances" are all actionable -- rather than
    a bare 500. It is never caught as a cue to fall back to rembg.
    """


_process: subprocess.Popen | None = None

# The helper is a single conversation over one pipe: a reply belongs to whoever
# wrote last. Concurrent /cutout requests would interleave their lines and read
# each other's answers, so a request holds the pipe for its whole round trip --
# the same reason sam_segmenter guards its predictor with _predict_lock.
_lock = threading.Lock()


def binary_path() -> str:
    path = config.VISION_MATTE_BIN
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(__file__), "..", path)


def available() -> bool:
    """Whether this machine can run the helper at all."""
    return sys.platform == "darwin" and os.path.exists(binary_path())


def _parse(line: str, what: str) -> dict:
    """One JSON line from the helper, or VisionMatteUnavailable.

    A raw JSONDecodeError would escape main.py's `except VisionMatteUnavailable`
    and surface as a bare 500, which reads like a bug in the cutout itself
    rather than "the helper said something unexpected".
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        raise VisionMatteUnavailable(
            f"vision matte helper sent an unreadable {what}: {line!r}") from e


def _start() -> subprocess.Popen:
    if not available():
        raise VisionMatteUnavailable(
            f"vision matte unavailable: needs macOS and a built binary at "
            f"{os.path.normpath(binary_path())} "
            f"(swiftc -O -parse-as-library vision_bench/vision_bench.swift "
            f"-o vision_bench/vision_bench)"
        )
    proc = subprocess.Popen(
        [binary_path(), "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    # The helper warms the model before announcing itself, so this line means the
    # next request pays only for inference.
    handshake = proc.stdout.readline()
    if not handshake:
        proc.kill()
        raise VisionMatteUnavailable("vision matte helper did not start: no handshake")
    try:
        ready = _parse(handshake, "handshake").get("ready")
    except VisionMatteUnavailable:
        proc.kill()
        raise
    if not ready:
        proc.kill()
        raise VisionMatteUnavailable(f"vision matte helper did not start: {handshake!r}")
    return proc


def _helper() -> subprocess.Popen:
    global _process
    if _process is None or _process.poll() is not None:
        _process = _start()
    return _process


def preload() -> None:
    """Start the helper up front so the first request does not pay for it."""
    with _lock:
        _helper()


def shutdown() -> None:
    """Stop the helper. Safe to call when it was never started.

    Nothing reaps it otherwise: it is a child of the server process, so
    stop_backends.sh (which kills whatever holds the port) leaves it resident
    with its model still in memory, and every --reload cycle in start-all.sh
    strands one more.

    Deliberately does not take _lock. Waiting for an in-flight matte would make
    shutdown hang behind exactly the case the watchdog exists for; killing the
    helper under a live request ends that request with the EOF path above,
    which is the right answer when the server is going away anyway.
    """
    _reset()


def remove_background(image_bytes: bytes) -> bytes:
    """Whole-foreground cutout as RGBA PNG bytes. Raises if Vision cannot do it.

    Vision reads from disk rather than a pipe: the helper needs a URL to pick up
    EXIF orientation (5.jpeg in the bench is stored rotated), and streaming bytes
    over the same pipe as the protocol would need framing for no gain.
    """
    with tempfile.TemporaryDirectory() as work:
        src = os.path.join(work, "in.png")
        dst = os.path.join(work, "out.png")
        with open(src, "wb") as f:
            f.write(image_bytes)

        with _lock:
            proc = _helper()
            # A helper that hangs -- alive, but never writing its reply line --
            # would block this readline forever, and with it every request
            # queued behind _lock. readline cannot be interrupted, so the
            # watchdog kills the process instead: that closes the pipe and ends
            # the read with EOF, the same way a crash does.
            timed_out = threading.Event()

            def _give_up() -> None:
                timed_out.set()
                proc.kill()

            watchdog = threading.Timer(config.VISION_MATTE_TIMEOUT, _give_up)
            watchdog.start()
            try:
                proc.stdin.write(f"{src}\t{dst}\n")
                proc.stdin.flush()
                reply = proc.stdout.readline()
            except (BrokenPipeError, ValueError) as e:
                _reset()
                raise VisionMatteUnavailable(f"vision matte helper died: {e}") from e
            finally:
                watchdog.cancel()

            # A reply that arrived is a reply, even if the watchdog fired in the
            # same instant; the killed process is picked up by the next call's
            # poll() check. Only an empty read needs the distinction.
            if not reply:
                _reset()
                if timed_out.is_set():
                    raise VisionMatteUnavailable(
                        f"vision matte helper stopped answering and was killed "
                        f"after {config.VISION_MATTE_TIMEOUT}s"
                    )
                raise VisionMatteUnavailable("vision matte helper exited mid-request")

        result = _parse(reply, "reply")
        if not result.get("ok"):
            raise VisionMatteUnavailable(f"vision matte failed: {result.get('error')}")
        with open(dst, "rb") as f:
            return f.read()


def _reset() -> None:
    global _process
    if _process is not None:
        _process.kill()
        _process = None
