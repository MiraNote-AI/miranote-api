#!/bin/bash
# Show who runs the beta backends and whether they answer: each launchd job's
# state, each port's /health on loopback, and each public hostname through
# the tunnel. Read-only.
#
# Usage: scripts/status_backends.sh
set -u

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
API_ROOT="${MIRANOTE_API_ROOT:-$(cd "$SCRIPTS/.." && pwd)}"
LOGS="$API_ROOT/logs/beta"
DOMAIN="gui/$(id -u)"
# shellcheck source=launchd/beta_services.sh
. "$SCRIPTS/launchd/beta_services.sh"

DISABLED="$(launchctl print-disabled "$DOMAIN" 2>/dev/null)"

job_line() {
  local label=$1 info state pid runs last off=""
  if echo "$DISABLED" | grep -q -E "\"$label\" => (disabled|true)"; then
    off="  [autostart DISABLED]"
  fi
  if ! info="$(launchctl print "$DOMAIN/$label" 2>/dev/null)"; then
    printf "   %-34s not loaded%s\n" "$label" "$off"
    return
  fi
  state="$(echo "$info" | awk -F' = ' '/^\tstate = / { print $2; exit }')"
  pid="$(echo "$info" | awk -F' = ' '/^\tpid = / { print $2; exit }')"
  runs="$(echo "$info" | awk -F' = ' '/^\truns = / { print $2; exit }')"
  last="$(echo "$info" | awk -F' = ' '/^\tlast exit code = / { print $2; exit }')"
  # a job killed by a signal has no exit code, only this
  [ -n "$last" ] || last="$(echo "$info" | awk -F' = ' '/^\tlast terminating signal = / { print $2; exit }')"
  printf "   %-34s %-12s pid %-7s runs %-4s last exit %s%s\n" \
    "$label" "$state" "${pid:--}" "${runs:--}" "${last:--}" "$off"
}

echo "== launchd jobs"
for spec in "${BETA_SERVICES[@]}"; do
  beta_parse "$spec"
  job_line "$(beta_label "$svc_name")"
done
job_line "$BETA_HEALTHCHECK_LABEL"
job_line "$BETA_TUNNEL_LABEL"

echo "== loopback /health"
for spec in "${BETA_SERVICES[@]}"; do
  beta_parse "$spec"
  code="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$svc_port/health")"
  owner="$(lsof -tiTCP:"$svc_port" -sTCP:LISTEN 2>/dev/null | head -1)"
  printf "   %-12s :%-5s %s  listener pid %s\n" "$svc_name" "$svc_port" "$code" "${owner:-none}"
done

echo "== public /health"
for h in $BETA_HOSTS; do
  code="$(curl -s -m 15 -o /dev/null -w '%{http_code}' "https://$h.miranote.app/health")"
  printf "   %-46s %s\n" "https://$h.miranote.app/health" "$code"
done

echo "== logs"
echo "   $LOGS/<name>-service.log   (launchd; <name>.log is the hand-started script's)"
echo "   $LOGS/healthcheck.log      (one record per login)"
