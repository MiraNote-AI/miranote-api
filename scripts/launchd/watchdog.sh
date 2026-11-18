#!/bin/bash
# Restart a beta backend that is running but no longer answers. Run every
# few minutes by ai.miranote.beta-watchdog; safe to run by hand.
#
# launchd's KeepAlive only sees a process exit. A deadlock or a blocked event
# loop keeps the port open and the pid alive, so each run calls every
# backend's loopback /health (free: no service makes a model call there) and
# counts consecutive failures. After --fails of them in a row it runs
# launchctl kickstart -k on that one job.
#
# A service is left alone, and its count reset, while it is not loaded (the
# manual-script mode), while launchd has no pid for it (it is between
# restarts) and for --grace seconds after its process started, which covers
# image's model load and the watchdog's own restarts. Every start of image,
# voice and collections makes paid probe calls, so restarts are capped at
# --max-per-hour per service; once the cap is hit the service is only
# reported until the oldest restart is an hour old.
#
# Nothing is printed for a healthy service. Every failure, restart, skipped
# restart and recovery is one line on stdout, which the job sends to
# watchdog.log.
#
# Usage: watchdog.sh STATE_DIR [--fails K] [--grace S] [--max-per-hour N]
#                              [--timeout S]
set -u

usage() {
  echo "usage: $0 STATE_DIR [--fails K] [--grace S] [--max-per-hour N] [--timeout S]"
  exit 2
}

[ $# -ge 1 ] || usage
STATE="$1"; shift
FAILS=3
GRACE=180
MAX_PER_HOUR=2
TIMEOUT=10
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || usage
  case "$1" in
    --fails) FAILS="$2" ;;
    --grace) GRACE="$2" ;;
    --max-per-hour) MAX_PER_HOUR="$2" ;;
    --timeout) TIMEOUT="$2" ;;
    *) usage ;;
  esac
  shift 2
done

# shellcheck source=beta_services.sh
. "$(cd "$(dirname "$0")" && pwd)/beta_services.sh"

mkdir -p "$STATE" || exit 1
DOMAIN="gui/$(id -u)"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "$(stamp) $*"; }

# job_pid LABEL -- the pid launchd runs for the job; empty if it has none.
# Fails when the job is not loaded.
job_pid() {
  local out
  out=$(launchctl print "$DOMAIN/$1" 2>/dev/null) || return 1
  echo "$out" | awk -F' = ' '/^\tpid = / { print $2; exit }'
}

# uptime_s PID -- seconds since the process started (ps etime: [[dd-]hh:]mm:ss)
uptime_s() {
  local t a b c d=0 h=0 m=0 s=0
  t=$(ps -o etime= -p "$1" 2>/dev/null | tr -d ' ') || return 1
  [ -n "$t" ] || return 1
  case "$t" in *-*) d=${t%%-*}; t=${t#*-} ;; esac
  IFS=: read -r a b c <<<"$t"
  if [ -n "$c" ]; then h=$a; m=$b; s=$c; else m=$a; s=$b; fi
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

now=$(date +%s)
for spec in "${BETA_SERVICES[@]}"; do
  beta_parse "$spec"
  label="$(beta_label "$svc_name")"
  fails_file="$STATE/$svc_name.fails"
  restarts_file="$STATE/$svc_name.restarts"

  if ! pid=$(job_pid "$label") || [ -z "$pid" ]; then
    rm -f "$fails_file"; continue
  fi
  up=$(uptime_s "$pid") || { rm -f "$fails_file"; continue; }
  if [ "$up" -lt "$GRACE" ]; then
    rm -f "$fails_file"; continue
  fi

  fails=$(cat "$fails_file" 2>/dev/null || echo 0)
  code=$(curl -s -f -m "$TIMEOUT" -o /dev/null -w '%{http_code}' \
         "http://127.0.0.1:$svc_port/health")
  rc=$?
  if [ "$rc" -eq 0 ]; then
    if [ "$fails" -gt 0 ]; then
      say "$svc_name: /health answers again after $fails failed check(s)"
    fi
    rm -f "$fails_file"; continue
  fi
  # Say how it failed: a hung server, one that stopped listening and one
  # that answers with an error each need a different look in its log.
  case "$rc" in
    7) why="connection refused" ;;
    22) why="HTTP $code" ;;
    28) why="no answer in ${TIMEOUT}s" ;;
    *) why="curl exit $rc" ;;
  esac

  fails=$((fails + 1))
  echo "$fails" >"$fails_file"
  say "$svc_name: /health failed on :$svc_port ($fails/$FAILS), pid $pid up ${up}s: $why"
  [ "$fails" -ge "$FAILS" ] || continue

  # Restarts in the last hour, one epoch second per line.
  recent=$(awk -v cutoff=$((now - 3600)) '$1 > cutoff' "$restarts_file" 2>/dev/null)
  count=$(printf '%s' "$recent" | grep -c . )
  if [ "$count" -ge "$MAX_PER_HOUR" ]; then
    say "$svc_name: NOT restarting, already restarted $count time(s) in the last hour"
    continue
  fi
  if launchctl kickstart -k "$DOMAIN/$label"; then
    say "$svc_name: restarted $label (kickstart -k)"
  else
    say "$svc_name: kickstart -k $label FAILED"
  fi
  { [ -n "$recent" ] && echo "$recent"; echo "$now"; } >"$restarts_file"
  rm -f "$fails_file"
done
exit 0
