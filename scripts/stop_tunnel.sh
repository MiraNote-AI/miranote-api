#!/bin/bash
# Stop the tunnel start_tunnel.sh started. Its caffeinate is tied to the tunnel
# process and exits on its own.
#
# Usage: scripts/stop_tunnel.sh
set -u

TUNNEL_CONFIG="$HOME/.cloudflared/miranote.yml"
API_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$API_ROOT/logs/beta/tunnel.pid"

if launchctl print "gui/$(id -u)/ai.miranote.beta-tunnel" >/dev/null 2>&1; then
  echo "the launchd service owns this tunnel -- this script will not stop it"
  echo "   stop for now: launchctl bootout gui/$(id -u)/ai.miranote.beta-tunnel"
  echo "   remove:       scripts/uninstall_tunnel_service.sh"
  exit 0
fi

if [ ! -f "$PIDFILE" ]; then
  echo "no pid file -- nothing to stop"
  exit 0
fi

pid="$(cat "$PIDFILE" 2>/dev/null)"
rm -f "$PIDFILE"

if [ -z "$pid" ]; then
  echo "empty pid file -- nothing to stop"
  exit 0
fi

# Confirm the pid is still OUR tunnel before killing it. This machine runs a
# second cloudflared tunnel from another account, and a recycled pid must never
# take it down.
if ! ps -p "$pid" -o command= 2>/dev/null | grep -q "$TUNNEL_CONFIG"; then
  echo "pid $pid is not this tunnel any more -- nothing killed"
  exit 0
fi

kill "$pid" 2>/dev/null
echo "tunnel: stopped $pid"
echo "done"
