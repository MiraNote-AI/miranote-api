"""Tests for the beta's launchd helpers: rotate_logs.sh and watchdog.sh.

Both scripts run for real, against logs in a temporary directory and
against tiny HTTP servers on loopback: one that answers /health, one that
accepts the connection and never answers (a hung backend). launchctl is a
fake on PATH that reports a pid and records every kickstart, so nothing is
loaded or restarted on the machine running the tests.
"""

from __future__ import annotations

import gzip
import http.server
import os
import pathlib
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest

LAUNCHD = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "launchd"
MB = 1024 * 1024
HEALTH = 'INFO:     127.0.0.1:50000 - "GET /health HTTP/1.1" 200 OK\n'
OTHER = 'INFO:     127.0.0.1:50001 - "POST /generate HTTP/1.1" 200 OK\n'


class RotateLogsTest(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)

    def rotate(self, *args):
        subprocess.run(
            ["bash", str(LAUNCHD / "rotate_logs.sh"), str(self.dir),
             "--max-mb", "1", *args],
            check=True, capture_output=True, text=True)

    def gz(self, name):
        with gzip.open(self.dir / name, "rt") as f:
            return f.read()

    def test_rotates_only_logs_over_the_limit_in_place(self):
        big, small = self.dir / "image-service.log", self.dir / "tunnel-service.log"
        small.write_text("small\n")
        # An append-mode writer, as launchd holds the log, keeps writing after
        # the truncate into the same file.
        with open(big, "a") as writer:
            writer.write("x" * (MB + 1))
            writer.flush()
            inode = os.stat(big).st_ino
            self.rotate()
            writer.write("after\n")
        self.assertEqual(big.read_text(), "after\n")
        self.assertEqual(os.stat(big).st_ino, inode)
        self.assertEqual(self.gz("image-service.log.1.gz"), "x" * (MB + 1))
        self.assertEqual(small.read_text(), "small\n")
        self.assertFalse((self.dir / "tunnel-service.log.1.gz").exists())

    def test_keeps_three_generations_newest_first(self):
        log = self.dir / "chat-service.log"
        for gen in "abcd":
            log.write_text(gen * (MB + 1))
            self.rotate("--keep", "3")
        self.assertEqual(self.gz("chat-service.log.1.gz")[0], "d")
        self.assertEqual(self.gz("chat-service.log.2.gz")[0], "c")
        self.assertEqual(self.gz("chat-service.log.3.gz")[0], "b")
        self.assertFalse((self.dir / "chat-service.log.4.gz").exists())

    def test_drop_health_filters_only_health_access_lines(self):
        log = self.dir / "text-service.log"
        n = MB // len(HEALTH) + 1
        log.write_text(HEALTH * n + OTHER + "Traceback: boom\n")
        self.rotate("--drop-health")
        self.assertEqual(self.gz("text-service.log.1.gz"), OTHER + "Traceback: boom\n")
        self.assertEqual(log.read_text(), "")

    def test_drop_health_with_nothing_left_still_rotates(self):
        log = self.dir / "voice-service.log"
        log.write_text(HEALTH * (MB // len(HEALTH) + 1))
        self.rotate("--drop-health")
        self.assertEqual(self.gz("voice-service.log.1.gz"), "")
        self.assertEqual(log.read_text(), "")


class _Healthy(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.paths.append(self.path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *args):
        pass


def _healthy_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Healthy)
    srv.paths = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _hung_server():
    """Accepts, records the request line, never answers."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    lines, held = [], []

    def serve():
        while True:
            try:
                conn, _ = sock.accept()
            except OSError:
                return
            held.append(conn)
            lines.append(conn.recv(4096).split(b"\r\n")[0].decode())

    threading.Thread(target=serve, daemon=True).start()
    return sock, sock.getsockname()[1], lines, held


FAKE_LAUNCHCTL = """#!/bin/bash
# print DOMAIN/LABEL: loaded when $FAKE/LABEL.pid exists; prints its pid
# line unless the file is empty. kickstart -k DOMAIN/LABEL: recorded.
case "$1" in
  print)
    f="$FAKE/${2##*/}.pid"
    [ -f "$f" ] || exit 113
    echo "${2##*/} = {"
    [ -s "$f" ] && printf '\\tpid = %s\\n' "$(cat "$f")"
    echo "}" ;;
  kickstart) echo "$3" >>"$FAKE/kickstarts" ;;
  *) exit 64 ;;
esac
"""


class WatchdogTest(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.healthy, healthy_port = _healthy_server()
        self.addCleanup(self.healthy.server_close)
        self.addCleanup(self.healthy.shutdown)
        sock, hung_port, self.hung_lines, held = _hung_server()
        self.addCleanup(lambda: [c.close() for c in held])
        self.addCleanup(sock.close)

        # The real script and helpers, with the service table pointed at
        # the two test servers.
        scripts = self.dir / "scripts"
        scripts.mkdir()
        shutil.copy(LAUNCHD / "watchdog.sh", scripts)
        table = (LAUNCHD / "beta_services.sh").read_text()
        table += (f'\nBETA_SERVICES=("text:t:main:app:{healthy_port}"'
                  f' "image:i:main:app:{hung_port}")\n')
        (scripts / "beta_services.sh").write_text(table)
        self.script = scripts / "watchdog.sh"

        self.fake = self.dir / "fake"
        self.fake.mkdir()
        launchctl = self.fake / "launchctl"
        launchctl.write_text(FAKE_LAUNCHCTL)
        launchctl.chmod(0o755)
        # A real, long-lived pid: ps reports its age for the grace period.
        for name in ("text", "image"):
            self.set_pid(name, str(os.getpid()))

    def set_pid(self, name, pid):
        (self.fake / f"ai.miranote.beta-{name}.pid").write_text(pid)

    def run_watchdog(self, *args):
        env = dict(os.environ, FAKE=str(self.fake),
                   PATH=f"{self.fake}:{os.environ['PATH']}")
        out = subprocess.run(
            ["bash", str(self.script), str(self.dir / "state"),
             "--timeout", "1", "--grace", "0", *args],
            check=True, capture_output=True, text=True, env=env)
        return out.stdout

    def kickstarts(self):
        f = self.fake / "kickstarts"
        return f.read_text().split() if f.exists() else []

    def test_restarts_a_hung_service_after_k_failures(self):
        out1 = self.run_watchdog("--fails", "3")
        out2 = self.run_watchdog("--fails", "3")
        self.assertEqual(self.kickstarts(), [])
        self.assertIn("image: /health failed", out1)
        self.assertIn("(2/3)", out2)
        out3 = self.run_watchdog("--fails", "3")
        self.assertIn("image: restarted ai.miranote.beta-image", out3)
        self.assertEqual(self.kickstarts(), [f"gui/{os.getuid()}/ai.miranote.beta-image"])
        self.assertNotIn("text", out1 + out2 + out3)
        # The count starts over after a restart.
        self.assertIn("(1/3)", self.run_watchdog("--fails", "3"))

    def test_caps_restarts_per_hour(self):
        for _ in range(3):
            out = self.run_watchdog("--fails", "1", "--max-per-hour", "2")
        self.assertEqual(len(self.kickstarts()), 2)
        self.assertIn("image: NOT restarting, already restarted 2 time(s)", out)

    def test_skips_a_service_inside_its_grace_period(self):
        out = self.run_watchdog("--fails", "1", "--grace", "3600")
        self.assertEqual(out, "")
        self.assertEqual(self.hung_lines, [])
        self.assertEqual(self.kickstarts(), [])

    def test_skips_a_service_that_is_not_loaded_or_has_no_pid(self):
        (self.fake / "ai.miranote.beta-image.pid").unlink()
        self.assertEqual(self.run_watchdog("--fails", "1"), "")
        self.set_pid("image", "")
        self.assertEqual(self.run_watchdog("--fails", "1"), "")
        self.assertEqual(self.kickstarts(), [])

    def test_reports_recovery_and_resets_the_count(self):
        self.run_watchdog("--fails", "3")
        # image now points at the healthy server's port.
        table = self.script.parent / "beta_services.sh"
        port = self.healthy.server_address[1]
        table.write_text(table.read_text() + f'BETA_SERVICES=("image:i:main:app:{port}")\n')
        out = self.run_watchdog("--fails", "3")
        self.assertIn("image: /health answers again after 1 failed check(s)", out)
        self.assertFalse((self.dir / "state" / "image.fails").exists())

    def test_only_ever_calls_loopback_health(self):
        self.run_watchdog("--fails", "1")
        self.assertEqual(self.healthy.paths, ["/health"])
        self.assertEqual(self.hung_lines, ["GET /health HTTP/1.1"])


if __name__ == "__main__":
    unittest.main()
