#!/bin/bash
# Remove the launchd job installed by install_tunnel_service.sh. The tunnel
# stops with it; start it by hand again with scripts/start_tunnel.sh.
#
# Usage: scripts/uninstall_tunnel_service.sh
set -u

LABEL="ai.miranote.beta-tunnel"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  echo "service: stopped"
else
  echo "service: not loaded"
fi

if [ -f "$PLIST" ]; then
  rm -f "$PLIST"
  echo "plist: removed"
else
  echo "plist: not present"
fi
echo "done"
