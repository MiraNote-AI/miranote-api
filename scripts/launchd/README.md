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

Every server is wrapped in `caffeinate -s`, so the Mac stays awake (on AC
power) for as long as any of them runs. The plists are generated from the
templates in this directory into `~/Library/LaunchAgents`; edit the
template, never the installed copy, and re-run the install script.

## Operating it

`D=gui/$(id -u)` below.

- **Status:** `scripts/status_backends.sh` -- job state, pid, restart
  count, last exit code, loopback and public `/health`. One job in detail:
  `launchctl print $D/ai.miranote.beta-image`.
- **Logs:** `logs/beta/<name>-service.log` (appended, never truncated),
  `logs/beta/healthcheck.log` (one record per login). `logs/beta/<name>.log`
  belongs to the hand-started script, not to launchd.
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

## Limits

- Nothing starts until someone logs in. FileVault is on, so after any
  reboot -- power loss, a crash, an update -- the beta is down until the
  password is typed at the Mac.
- The health check records; it does not alert and does not restart
  anything. A server that hangs without exiting is not restarted either:
  launchd only sees exits.
- Logs grow without bound.
