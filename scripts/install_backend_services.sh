#!/bin/bash
# Install the five beta backends as launchd jobs, so they start at login and
# restart when one dies, plus a one-shot job that records their health after
# login. Companions: status_backends.sh, disable_backend_services.sh.
#
# Services are switched over one at a time: the hand-started server on a port
# is stopped, its job is loaded, and the next service is not touched until
# this one answers /health. Each service is down only for its own startup.
# Re-running this is an upgrade: a loaded job is booted out and loaded again,
# which is also how to pick up new code once vision_bench needs rebuilding.
#
# Usage:
#   scripts/install_backend_services.sh              install or upgrade
#   scripts/install_backend_services.sh --preflight  dry run, touches nothing live
#
# --preflight loads the same jobs under *-preflight labels on port + 10000,
# logging to a temporary directory, waits for /health and boots them out
# again. It leaves the live servers and ~/Library/LaunchAgents alone, so it is
# how to learn that a new plist works before the live servers go down.
#
# MIRANOTE_API_ROOT points the jobs at another checkout than the one holding
# this script, e.g. to run it from a worktree against the main checkout.
set -u

MODE="install"
case "${1:-}" in
  "") ;;
  --preflight) MODE="preflight" ;;
  *) echo "usage: $0 [--preflight]"; exit 2 ;;
esac

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
API_ROOT="${MIRANOTE_API_ROOT:-$(cd "$SCRIPTS/.." && pwd)}"
API_ROOT="$(cd "$API_ROOT" && pwd)" || { echo "no such checkout: $API_ROOT"; exit 1; }
LOGS="$API_ROOT/logs/beta"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
BACKEND_TEMPLATE="$SCRIPTS/launchd/ai.miranote.beta-backend.plist.template"
HEALTHCHECK_TEMPLATE="$SCRIPTS/launchd/ai.miranote.beta-healthcheck.plist.template"
PREFLIGHT_OFFSET=10000

# shellcheck source=launchd/beta_services.sh
. "$SCRIPTS/launchd/beta_services.sh"

[ -f "$BACKEND_TEMPLATE" ] || { echo "missing $BACKEND_TEMPLATE"; exit 1; }
[ -f "$HEALTHCHECK_TEMPLATE" ] || { echo "missing $HEALTHCHECK_TEMPLATE"; exit 1; }

# voice-to-text runs ffmpeg by name, and launchd's PATH has no Homebrew in it.
FFMPEG="$(command -v ffmpeg || true)"
if [ -n "$FFMPEG" ]; then
  JOB_PATH="$(dirname "$FFMPEG"):/usr/bin:/bin:/usr/sbin:/sbin"
else
  echo "!! ffmpeg is not on PATH -- voice will fail every recording it re-encodes"
  JOB_PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
fi

echo "== checkout: $API_ROOT"
missing=0
for spec in "${BETA_SERVICES[@]}"; do
  beta_parse "$spec"
  if [ ! -x "$API_ROOT/poc/$svc_dir/.venv/bin/uvicorn" ]; then
    echo "   $svc_name: MISSING poc/$svc_dir/.venv -- see poc/$svc_dir/README.md"
    missing=1
  fi
done
[ "$missing" -eq 0 ] || exit 1
mkdir -p "$LOGS" "$AGENTS"

# Same rule as start_backends.sh: every cutout goes through the Vision helper
# and nothing downgrades to rembg, so build it whenever it is absent or older
# than its source. launchd only restarts what is there; it never rebuilds.
echo "== Apple Vision matte helper"
VISION_DIR="$API_ROOT/poc/image-generation/vision_bench"
VISION_BIN="$VISION_DIR/vision_bench"
if [ -x "$VISION_BIN" ] && [ "$VISION_BIN" -nt "$VISION_DIR/vision_bench.swift" ]; then
  echo "   up to date"
elif [ "$MODE" = "preflight" ]; then
  echo "   !! absent or stale -- the install will build it; this dry run will not"
elif (cd "$API_ROOT/poc/image-generation" \
      && swiftc -O -parse-as-library vision_bench/vision_bench.swift -o vision_bench/vision_bench); then
  echo "   built $VISION_BIN"
else
  echo "   BUILD FAILED -- every /cutout will 503. vision_bench.swift needs macOS 15+."
fi

# render_backend NAME DIR APP PORT LABEL LOG OUT
render_backend() {
  local name=$1 dir=$2 app=$3 port=$4 label=$5 log=$6 out=$7
  local workdir="$API_ROOT/poc/$dir" app_dir_args=""
  if [ "$name" = "chat" ]; then
    app_dir_args="        <string>--app-dir</string>\n        <string>$API_ROOT</string>"
  fi
  awk -v args="$app_dir_args" \
      '/__APP_DIR_ARGS__/ { if (args != "") print args; next } { print }' \
      "$BACKEND_TEMPLATE" \
    | sed -e "s|__LABEL__|$label|g" \
          -e "s|__UVICORN__|$workdir/.venv/bin/uvicorn|g" \
          -e "s|__WORKDIR__|$workdir|g" \
          -e "s|__APP__|$app|g" \
          -e "s|__PORT__|$port|g" \
          -e "s|__API_ROOT__|$API_ROOT|g" \
          -e "s|__PATH__|$JOB_PATH|g" \
          -e "s|__LOG__|$log|g" \
    >"$out"
  plutil -lint "$out" >/dev/null || { echo "   $name: rendered plist is invalid: $out"; return 1; }
}

# wait_healthy PORT SECONDS -- 0 once /health answers 2xx
wait_healthy() {
  local deadline=$((SECONDS + $2))
  until curl -s -f -m 2 -o /dev/null "http://127.0.0.1:$1/health"; do
    [ $SECONDS -ge $deadline ] && return 1
    sleep 0.5
  done
}

# report_service NAME PORT -- the checks start_backends.sh makes once a port answers
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")
report_service() {
  local name=$1 port=$2
  if [ -n "$LAN_IP" ] && curl -s -m 3 -o /dev/null "http://$LAN_IP:$port/health"; then
    echo "      !! ALSO REACHABLE AT $LAN_IP:$port -- not bound to loopback"
  else
    echo "      loopback only"
  fi
  if [ "$name" = "image" ]; then
    if curl -s -m 2 "http://127.0.0.1:$port/health" | grep -q '"vision_ready":true'; then
      echo "      cutout: Apple Vision ready"
    else
      echo "      cutout: !! VISION HELPER DOWN -- /cutout and sticker /generate will 503"
    fi
  fi
}

job_pid() {
  launchctl print "$DOMAIN/$1" 2>/dev/null | awk -F' = ' '/^\tpid = / { print $2 }'
}

# --------------------------------------------------------------------------- #
if [ "$MODE" = "preflight" ]; then
  WORK="$(mktemp -d "${TMPDIR%/}/beta-preflight.XXXXXX")"
  echo "== preflight on port + $PREFLIGHT_OFFSET (plists and logs in $WORK)"
  for spec in "${BETA_SERVICES[@]}"; do
    beta_parse "$spec"
    port=$((svc_port + PREFLIGHT_OFFSET))
    if lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "   port $port is taken -- cannot run the preflight"; exit 1
    fi
  done
  render_hc="$WORK/$BETA_HEALTHCHECK_LABEL.plist"
  sed -e "s|__LOGS__|$WORK|g" "$HEALTHCHECK_TEMPLATE" >"$render_hc"
  plutil -lint "$render_hc" >/dev/null || { echo "   healthcheck plist is invalid"; exit 1; }

  # Every job loaded here is booted out again below, whatever fails.
  labels=()
  failed=0
  for spec in "${BETA_SERVICES[@]}"; do
    beta_parse "$spec"
    label="$(beta_label "$svc_name")-preflight"
    plist="$WORK/$label.plist"
    if ! render_backend "$svc_name" "$svc_dir" "$svc_app" "$((svc_port + PREFLIGHT_OFFSET))" \
           "$label" "$WORK/$svc_name.log" "$plist"; then
      failed=1; break
    fi
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    if ! launchctl bootstrap "$DOMAIN" "$plist"; then
      echo "   $svc_name: bootstrap failed"; failed=1; break
    fi
    labels+=("$label")
  done
  started=$SECONDS

  if [ "$failed" -eq 0 ]; then
    for spec in "${BETA_SERVICES[@]}"; do
      beta_parse "$spec"
      port=$((svc_port + PREFLIGHT_OFFSET))
      if wait_healthy "$port" 300; then
        echo "   $svc_name: healthy on :$port after ~$((SECONDS - started))s" \
             "(pid $(job_pid "$(beta_label "$svc_name")-preflight"))"
        report_service "$svc_name" "$port"
      else
        echo "   $svc_name: NOT HEALTHY on :$port after 300s -- see $WORK/$svc_name.log"
        failed=1
      fi
    done
  fi

  echo "== booting the preflight jobs out"
  for label in ${labels[@]+"${labels[@]}"}; do
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  done
  [ "$failed" -eq 0 ] && echo "== preflight passed" || echo "== preflight FAILED"
  exit "$failed"
fi

# --------------------------------------------------------------------------- #
STAMP="$(date +%Y%m%d-%H%M%S)"
STAGE="$(mktemp -d "${TMPDIR%/}/beta-install.XXXXXX")"

# install_plist RENDERED LABEL -- back up a different existing plist, then
# put the rendered one in place
install_plist() {
  local rendered=$1 label=$2 target="$AGENTS/$2.plist"
  if [ -f "$target" ] && ! cmp -s "$rendered" "$target"; then
    cp -p "$target" "$target.bak.$STAMP"
    echo "   backed up the old plist to $target.bak.$STAMP"
  fi
  cp "$rendered" "$target"
}

# free_port PORT -- stop whatever hand-started server listens there; 0 once free
free_port() {
  local port=$1 pids deadline=$((SECONDS + 30))
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ')
  [ -n "$pids" ] || return 0
  echo "   stopping hand-started pid $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null
  while lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; do
    [ $SECONDS -ge $deadline ] && return 1
    sleep 0.2
  done
}

echo "== switching the backends to launchd, one at a time"
for spec in "${BETA_SERVICES[@]}"; do
  beta_parse "$spec"
  label="$(beta_label "$svc_name")"
  rendered="$STAGE/$label.plist"
  render_backend "$svc_name" "$svc_dir" "$svc_app" "$svc_port" \
    "$label" "$LOGS/$svc_name-service.log" "$rendered" || exit 1

  echo "   $svc_name (:$svc_port)"
  down_at=$SECONDS
  if beta_loaded "$label"; then
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  fi
  if ! free_port "$svc_port"; then
    echo "   !! :$svc_port is still taken after 30s -- stopping here, nothing else changed"
    exit 1
  fi
  install_plist "$rendered" "$label"
  # enable before bootstrap: a job disabled by disable_backend_services.sh
  # refuses to bootstrap until it is enabled again
  launchctl enable "$DOMAIN/$label"
  if ! launchctl bootstrap "$DOMAIN" "$AGENTS/$label.plist"; then
    echo "   !! bootstrap failed -- $svc_name is DOWN. To go back to the script:"
    echo "      scripts/disable_backend_services.sh && scripts/start_backends.sh"
    exit 1
  fi
  if wait_healthy "$svc_port" 300; then
    echo "      healthy under launchd (pid $(job_pid "$label")), down for ~$((SECONDS - down_at))s"
    report_service "$svc_name" "$svc_port"
  else
    echo "   !! not healthy after 300s -- see $LOGS/$svc_name-service.log"
    echo "      stopping here; launchd keeps retrying it. To go back to the script:"
    echo "      scripts/disable_backend_services.sh && scripts/start_backends.sh"
    exit 1
  fi
done

echo "== health check job (runs now, and once at every login)"
rendered="$STAGE/$BETA_HEALTHCHECK_LABEL.plist"
sed -e "s|__LOGS__|$LOGS|g" "$HEALTHCHECK_TEMPLATE" >"$rendered"
plutil -lint "$rendered" >/dev/null || { echo "   healthcheck plist is invalid"; exit 1; }
launchctl bootout "$DOMAIN/$BETA_HEALTHCHECK_LABEL" 2>/dev/null || true
install_plist "$rendered" "$BETA_HEALTHCHECK_LABEL"
launchctl enable "$DOMAIN/$BETA_HEALTHCHECK_LABEL"
launchctl bootstrap "$DOMAIN" "$AGENTS/$BETA_HEALTHCHECK_LABEL.plist" \
  || echo "   !! bootstrap failed"

echo "== public endpoints"
for h in $BETA_HOSTS; do
  code="$(curl -s -m 15 -o /dev/null -w '%{http_code}' "https://$h.miranote.app/health")"
  printf "   %-46s %s\n" "https://$h.miranote.app/health" "$code"
done

echo "== done"
echo "   the backends now start at login and restart if one dies"
echo "   status:   scripts/status_backends.sh"
echo "   restart:  launchctl kickstart -k $DOMAIN/ai.miranote.beta-<name>"
echo "   logs:     $LOGS/<name>-service.log, $LOGS/healthcheck.log"
echo "   disable:  scripts/disable_backend_services.sh"
