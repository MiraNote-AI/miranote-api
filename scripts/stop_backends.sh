#!/bin/bash
# Stop everything start_backends.sh started (servers + caffeinate).
set -u

API_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$API_ROOT/logs/beta"

for port in 8001 8002 8003 8005; do
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
