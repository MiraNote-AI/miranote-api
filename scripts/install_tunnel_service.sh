#!/bin/bash
# Install the beta tunnel as a launchd job so it survives reboot and restarts
# on crash. Companion: uninstall_tunnel_service.sh.
#
# NOT "cloudflared service install": that writes one machine-wide service
# around the default ~/.cloudflared/config.yml, which on this machine belongs
# to an unrelated tunnel from a different Cloudflare account.
#
# Usage: scripts/install_tunnel_service.sh
set -eu

LABEL="ai.miranote.beta-tunnel"
TUNNEL_CONFIG="$HOME/.cloudflared/miranote.yml"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

API_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$API_ROOT/logs/beta"
TEMPLATE="$API_ROOT/scripts/launchd/$LABEL.plist.template"

CLOUDFLARED="$(command -v cloudflared || true)"
[ -n "$CLOUDFLARED" ] || { echo "cloudflared is not on PATH"; exit 1; }
[ -f "$TUNNEL_CONFIG" ] || { echo "missing $TUNNEL_CONFIG"; exit 1; }
[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE"; exit 1; }

mkdir -p "$LOGS" "$HOME/Library/LaunchAgents"

# Stop a hand-started tunnel first. Two cloudflared processes serving one
# tunnel is confusing rather than loudly broken, so the two ways of starting it
# must never overlap.
if [ -f "$LOGS/tunnel.pid" ]; then
  echo "== stopping the hand-started tunnel first"
  "$API_ROOT/scripts/stop_tunnel.sh"
fi

echo "== writing $PLIST"
sed -e "s|__CLOUDFLARED__|$CLOUDFLARED|g" \
    -e "s|__CONFIG__|$TUNNEL_CONFIG|g" \
    -e "s|__LOGS__|$LOGS|g" \
    "$TEMPLATE" >"$PLIST"

# bootout first so re-running this is an upgrade rather than an error.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/$LABEL"

echo "== waiting for the tunnel to connect"
deadline=$((SECONDS + 60))
until grep -q "Registered tunnel connection" "$LOGS/tunnel-service.log" 2>/dev/null; do
  if [ $SECONDS -ge $deadline ]; then
    echo "   TIMED OUT -- check $LOGS/tunnel-service.log"
    exit 1
  fi
  sleep 2
done

echo "== done"
echo "   the tunnel now starts at login and restarts if it dies"
echo "   log:       $LOGS/tunnel-service.log"
echo "   status:    launchctl print gui/$(id -u)/$LABEL | head"
echo "   restart:   launchctl kickstart -k gui/$(id -u)/$LABEL"
echo "   remove:    scripts/uninstall_tunnel_service.sh"
