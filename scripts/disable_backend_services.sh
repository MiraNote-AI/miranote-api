#!/bin/bash
# Stop the launchd backend jobs and keep them from starting at login, so the
# backends can go back to scripts/start_backends.sh. The tunnel is not
# touched. Companion: install_backend_services.sh, which turns them on again.
#
# The plists stay in ~/Library/LaunchAgents. "launchctl disable" is what keeps
# them from loading at login, and it survives reboot; nothing is deleted, so
# the way back is a re-run of the install script.
#
# Usage: scripts/disable_backend_services.sh     then: scripts/start_backends.sh
set -u

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="gui/$(id -u)"
# shellcheck source=launchd/beta_services.sh
. "$SCRIPTS/launchd/beta_services.sh"

labels=()
for spec in "${BETA_SERVICES[@]}"; do
  beta_parse "$spec"
  labels+=("$(beta_label "$svc_name")")
done
labels+=("$BETA_HEALTHCHECK_LABEL")

for label in "${labels[@]}"; do
  if beta_loaded "$label"; then
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    state="stopped"
  else
    state="was not loaded"
  fi
  launchctl disable "$DOMAIN/$label"
  printf "   %-34s %s, autostart disabled\n" "$label" "$state"
done

echo "done"
echo "   start by hand:  scripts/start_backends.sh"
echo "   back to launchd: scripts/install_backend_services.sh"
