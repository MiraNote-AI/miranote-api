#!/bin/bash
# Rotate the beta's launchd logs in place, without restarting anything. Run
# daily by ai.miranote.beta-logrotate; safe to run by hand at any time.
#
# Every *.log in LOG_DIR over MAX_MB is copied aside and truncated
# (copy-truncate). launchd opens StandardOutPath in append mode, so the
# running job keeps writing to the same file, now from offset 0, and needs no
# signal. A line written between the copy and the truncate is lost; that is
# the price of not restarting anything.
#
# Generations are <log>.1.gz (newest) to <log>.KEEP.gz; the oldest is dropped.
# With --drop-health the uvicorn access lines for GET /health are left out of
# the copy: the health checks and the watchdog write one every few minutes
# per service and they carry nothing worth keeping.
#
# Usage: rotate_logs.sh LOG_DIR [--max-mb N] [--keep N] [--drop-health]
set -u -o pipefail

usage() { echo "usage: $0 LOG_DIR [--max-mb N] [--keep N] [--drop-health]"; exit 2; }

[ $# -ge 1 ] || usage
LOG_DIR="$1"; shift
MAX_MB=10
KEEP=3
DROP_HEALTH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-mb) [ $# -ge 2 ] || usage; MAX_MB="$2"; shift 2 ;;
    --keep) [ $# -ge 2 ] || usage; KEEP="$2"; shift 2 ;;
    --drop-health) DROP_HEALTH=1; shift ;;
    *) usage ;;
  esac
done
[ -d "$LOG_DIR" ] || { echo "no such directory: $LOG_DIR"; exit 1; }

# uvicorn: INFO:     127.0.0.1:52345 - "GET /health HTTP/1.1" 200 OK
HEALTH_LINE='"GET /health HTTP/[0-9.]*" '
LIMIT=$((MAX_MB * 1024 * 1024))
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

status=0
for log in "$LOG_DIR"/*.log; do
  [ -f "$log" ] || continue
  size=$(wc -c <"$log" | tr -d ' ')
  [ "$size" -gt "$LIMIT" ] || continue

  # Copy first: a failed copy leaves the log and every generation as it was.
  # grep exits 1 when every line was a health line; that is not an error.
  if [ "$DROP_HEALTH" -eq 1 ]; then
    { grep -v -e "$HEALTH_LINE" "$log" || [ $? -eq 1 ]; } | gzip >"$log.new.gz"
  else
    gzip -c "$log" >"$log.new.gz"
  fi
  if [ $? -ne 0 ]; then
    rm -f "$log.new.gz"
    echo "$(stamp) $log: copy FAILED, left as it was"
    status=1
    continue
  fi

  rm -f "$log.$KEEP.gz"
  i=$KEEP
  while [ "$i" -gt 1 ]; do
    prev=$((i - 1))
    [ -f "$log.$prev.gz" ] && mv "$log.$prev.gz" "$log.$i.gz"
    i=$prev
  done
  mv "$log.new.gz" "$log.1.gz"
  : >"$log"
  echo "$(stamp) $log: rotated at $size bytes"
done
exit "$status"
