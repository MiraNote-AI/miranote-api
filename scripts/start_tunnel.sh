#!/bin/bash
# Start the Cloudflare tunnel that fronts the four POC backends, and keep the
# Mac awake for as long as it runs. Companion: stop_tunnel.sh.
#
# The tunnel and the backends are deliberately separate lifecycles. Testers open
# the app at unpredictable times, and the tunnel must answer even when no
# backend is up -- a 502 that says "backends are down" is a far better answer
# than a connection that never opens.
#
# Usage: scripts/start_tunnel.sh
set -u

TUNNEL_NAME="miranote-beta"
TUNNEL_CONFIG="$HOME/.cloudflared/miranote.yml"
HOSTS="beta-text beta-image beta-chat beta-voice"

API_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$API_ROOT/logs/beta"
mkdir -p "$LOGS"
PIDFILE="$LOGS/tunnel.pid"

# This machine also runs an unrelated cloudflared tunnel from a different
# Cloudflare account. Every check and every kill below is therefore matched
# against OUR config path, never against "cloudflared" alone.
running_pid() {
  [ -f "$PIDFILE" ] || return 1
  pid="$(cat "$PIDFILE" 2>/dev/null)"
  [ -n "$pid" ] || return 1
  ps -p "$pid" -o command= 2>/dev/null | grep -q "$TUNNEL_CONFIG" || return 1
  echo "$pid"
}

if pid="$(running_pid)"; then
  echo "already running: pid $pid"
  echo "   log: $LOGS/tunnel.log"
  exit 0
fi

if [ ! -f "$TUNNEL_CONFIG" ]; then
  echo "missing $TUNNEL_CONFIG -- see docs/specs/2026-09-03-testflight-beta-deploy-design.md section 4.2"
  exit 1
fi

# --config is not optional. The default ~/.cloudflared/config.yml belongs to the
# other tunnel on this machine, and a cloudflared command that loads it while
# referring to this tunnel by name fails with "Tunnel not found".
echo "== starting tunnel $TUNNEL_NAME"
nohup cloudflared --config "$TUNNEL_CONFIG" tunnel run "$TUNNEL_NAME" \
  >"$LOGS/tunnel.log" 2>&1 &
TUNNEL_PID=$!
echo "$TUNNEL_PID" >"$PIDFILE"

# Tied to the tunnel with -w so it cannot outlive it and hold the Mac awake for
# nothing. start_backends.sh keeps its own caffeinate for the backends.
nohup caffeinate -s -w "$TUNNEL_PID" >/dev/null 2>&1 &
echo "   caffeinate: holding the Mac awake while pid $TUNNEL_PID lives"

echo "== waiting for edge connections"
deadline=$((SECONDS + 60))
until grep -q "Registered tunnel connection" "$LOGS/tunnel.log" 2>/dev/null; do
  if ! ps -p "$TUNNEL_PID" >/dev/null 2>&1; then
    echo "   tunnel exited -- check $LOGS/tunnel.log"
    rm -f "$PIDFILE"
    exit 1
  fi
  if [ $SECONDS -ge $deadline ]; then
    echo "   TIMED OUT waiting for a connection -- check $LOGS/tunnel.log"
    exit 1
  fi
  sleep 2
done
echo "   pid $TUNNEL_PID, $(grep -c 'Registered tunnel connection' "$LOGS/tunnel.log") edge connections"

echo "== public endpoints"
for h in $HOSTS; do
  code="$(curl -s -m 15 -o /dev/null -w '%{http_code}' "https://$h.miranote.app/health" 2>/dev/null)"
  case "$code" in
    200) note="healthy" ;;
    502) note="reachable, but the backend is down -- run scripts/start_backends.sh" ;;
    000) note="no answer -- DNS or TLS problem" ;;
    *)   note="unexpected" ;;
  esac
  printf "   %-32s %s  %s\n" "https://$h.miranote.app" "$code" "$note"
done

echo "== done"
echo "   log:  $LOGS/tunnel.log"
echo "   stop: scripts/stop_tunnel.sh"
