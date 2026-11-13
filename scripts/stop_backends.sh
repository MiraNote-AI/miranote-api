#!/bin/bash
# Stop everything start_backends.sh started (servers + caffeinate).
set -u

API_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$API_ROOT/logs/beta"

# A server launchd runs comes straight back when killed, so killing it here
# would only look like a stop. See start_backends.sh.
for name in text image chat voice collections; do
  if launchctl print "gui/$(id -u)/ai.miranote.beta-$name" >/dev/null 2>&1; then
    echo "launchd runs the backends (ai.miranote.beta-$name is loaded) -- this script will not stop them"
    echo "   stop one for now: launchctl bootout gui/$(id -u)/ai.miranote.beta-<name>"
    echo "   stop all and keep them off: scripts/disable_backend_services.sh"
    exit 0
  fi
done

for port in 8001 8002 8003 8005 8006; do
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null
    echo "port $port: stopped $pids"
  fi
done

if [ -f "$LOGS/caffeinate.pid" ]; then
  kill "$(cat "$LOGS/caffeinate.pid")" 2>/dev/null
  rm -f "$LOGS/caffeinate.pid"
  echo "caffeinate: stopped"
fi
echo "done"
