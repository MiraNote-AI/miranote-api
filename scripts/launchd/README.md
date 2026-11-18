# Beta services under launchd

Everything the beta needs on the Mac runs as a LaunchAgent in the user's
GUI session: it starts at login and launchd restarts it when it dies.

| Job | Runs | Installed by |
| --- | --- | --- |
| `ai.miranote.beta-tunnel` | cloudflared for the five `beta-*.miranote.app` hostnames | `install_tunnel_service.sh` |
| `ai.miranote.beta-text` | text-clean-expand on 127.0.0.1:8001 | `install_backend_services.sh` |
| `ai.miranote.beta-image` | image-generation on 127.0.0.1:8002 | `install_backend_services.sh` |
| `ai.miranote.beta-chat` | chatbot on 127.0.0.1:8003 | `install_backend_services.sh` |
| `ai.miranote.beta-voice` | voice-to-text on 127.0.0.1:8005 | `install_backend_services.sh` |
| `ai.miranote.beta-collections` | photo-collections on 127.0.0.1:8006 | `install_backend_services.sh` |
| `ai.miranote.beta-healthcheck` | one curl at login that records whether all of the above answer | `install_backend_services.sh` |
| `ai.miranote.beta-watchdog` | every 2 min, restarts a backend whose `/health` stopped answering | by hand, see below |
| `ai.miranote.beta-logrotate` | daily at 04:00, rotates every log in `logs/beta` over 10 MB | by hand, see below |

Every server is wrapped in `caffeinate -s`, so the Mac stays awake (on AC
power) for as long as any of them runs. The plists are generated from the
templates in this directory into `~/Library/LaunchAgents`; edit the
template, never the installed copy, and re-run the install script.

## Operating it

`D=gui/$(id -u)` below.

- **Status:** `scripts/status_backends.sh` -- job state, pid, restart
  count, last exit code, loopback and public `/health`. One job in detail:
  `launchctl print $D/ai.miranote.beta-image`.
- **Logs:** `logs/beta/<name>-service.log`, `logs/beta/healthcheck.log`
  (one record per login), `logs/beta/watchdog.log` (every failed check and
  restart), `logs/beta/logrotate.log`. `logs/beta/<name>.log` belongs to the
  hand-started script, not to launchd. Rotated copies are
  `<log>.1.gz` (newest) to `<log>.3.gz`; read one with `gzcat`.
- **Restart one service:** `launchctl kickstart -k $D/ai.miranote.beta-<name>`.
  Image takes about 15 s to answer again, the others a few seconds.
- **Stop one until the next login:** `launchctl bootout $D/ai.miranote.beta-<name>`;
  start it again with
  `launchctl bootstrap $D ~/Library/LaunchAgents/ai.miranote.beta-<name>.plist`.
- **New code:** pull, then `kickstart -k` each service that changed. If
  `vision_bench.swift` changed, re-run `scripts/install_backend_services.sh`
  instead: it rebuilds the helper and restarts the services one at a time.
- **Turn autostart off:** `scripts/disable_backend_services.sh`. It boots
  the jobs out and runs `launchctl disable` on each, which survives reboot.
  The plists stay where they are.
- **Back to the manual script:** `scripts/disable_backend_services.sh`,
  then `scripts/start_backends.sh`. While any backend job is loaded,
  `start_backends.sh` and `stop_backends.sh` refuse to run, because killing
  a launchd server only makes launchd start it again.
- **Back to launchd:** `scripts/install_backend_services.sh`. Run it with
  `--preflight` first after changing a template: that loads the same jobs on
  port + 10000, waits for `/health` and removes them, without touching the
  live servers.

## Watchdog and log rotation

`watchdog.sh` (api #102) calls each backend's loopback `/health` with a
10 s limit. After 3 failed checks in a row it runs `launchctl kickstart -k`
on that job. It leaves a service alone while it is not loaded, while
launchd has no pid for it, and for 180 s after its process started. It
restarts each service at most twice an hour, because every start of image,
voice and collections makes paid probe calls; past that it only logs. It
calls nothing but `/health`. Counters live in `logs/beta/watchdog-state/`;
delete that directory to reset them.

`rotate_logs.sh` (api #101) copies every `logs/beta/*.log` over 10 MB to
`<log>.1.gz`, without the `GET /health` access lines that answered 200,
and truncates the original in place. launchd and the hand-started scripts
both write the logs in append mode, so nothing is restarted. Three
generations are kept.

Both run from this checkout and are installed by hand. The flags in the
plist templates set the limits; edit the template and install again to
change them. Tests: `python3 -m pytest tests/test_beta_launchd.py` runs
both scripts against fake logs, fake servers and a fake `launchctl`.

Install, from the repository root (re-running it is an upgrade):

```bash
D=gui/$(id -u)
S=$PWD/scripts/launchd
L=$PWD/logs/beta
A=~/Library/LaunchAgents
for j in watchdog logrotate; do
  sed -e "s|__SCRIPTS__|$S|g" -e "s|__LOGS__|$L|g" \
    $S/ai.miranote.beta-$j.plist.template >$A/ai.miranote.beta-$j.plist
  plutil -lint $A/ai.miranote.beta-$j.plist
  launchctl bootout $D/ai.miranote.beta-$j 2>/dev/null
  launchctl bootstrap $D $A/ai.miranote.beta-$j.plist
done
```

Check: `launchctl print $D/ai.miranote.beta-watchdog`, and run either one
now with `launchctl kickstart $D/ai.miranote.beta-logrotate`.

Uninstall:

```bash
D=gui/$(id -u)
for j in watchdog logrotate; do
  launchctl bootout $D/ai.miranote.beta-$j
  rm ~/Library/LaunchAgents/ai.miranote.beta-$j.plist
done
```

Going back to the manual script needs nothing here: the watchdog skips
a job that is not loaded.

## Limits

- Nothing starts until someone logs in. FileVault is on, so after any
  reboot -- power loss, a crash, an update -- the beta is down until the
  password is typed at the Mac.
- The health check records; it does not alert. The watchdog restarts a
  hung server, but no one hears about it outside `watchdog.log`, and after
  two restarts in an hour a server that keeps hanging stays down until
  the hour is up.
- A line a service writes in the instant between rotation's copy and its
  truncate is lost.
