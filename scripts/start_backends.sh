#!/bin/bash
# Start all four POC backends for the phone beta, bound to 0.0.0.0 so
# real devices on the same Wi-Fi can reach them (127.0.0.1 -- the uvicorn
# default -- accepts connections from this Mac only). Keeps the Mac awake
# while they run. Companion: stop_backends.sh; phone-side steps live in
# miranote-ios/docs/RUN_ON_YOUR_PHONE.md.
#
# Usage: scripts/start_backends.sh
set -u

API_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$API_ROOT/logs/beta"
mkdir -p "$LOGS"

# The four POCs have independent venvs and no package in common, so the repo
# root goes on sys.path for all of them. PYTHONPATH rather than --app-dir:
# uvicorn inserts app_dir at sys.path[0] *instead of* the working directory,
# so passing it would break the three services launched as "main:app".
export PYTHONPATH="$API_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# name : working dir (under poc/) : uvicorn app spec : port
SERVICES=(
  "text:text-clean-expand:main:app:8001"
  "image:image-generation:main:app:8002"
  "chat:chatbot:poc.chatbot.main:app:8003"
  "voice:voice-to-text:main:app:8005"
)

echo "== stopping anything already on the beta ports"
for spec in "${SERVICES[@]}"; do
  port="${spec##*:}"
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "   port $port: killing $pids"
    kill $pids 2>/dev/null
  fi
done
sleep 2

echo "== starting services (logs in $LOGS)"
for spec in "${SERVICES[@]}"; do
  name="${spec%%:*}"
  rest="${spec#*:}"
  dir="${rest%%:*}"
  rest="${rest#*:}"
  app="${rest%:*}"
  port="${spec##*:}"
  cd "$API_ROOT/poc/$dir"
  if [ ! -x .venv/bin/uvicorn ]; then
    echo "   $name: MISSING .venv -- see poc/$dir/README.md, skipping"
    continue
  fi
  if [ "$name" = "chat" ]; then
    # chatbot imports as a package from the repo root
    nohup .venv/bin/uvicorn --app-dir "$API_ROOT" "$app" \
      --host 0.0.0.0 --port "$port" >"$LOGS/$name.log" 2>&1 &
  else
    nohup .venv/bin/uvicorn "$app" \
      --host 0.0.0.0 --port "$port" >"$LOGS/$name.log" 2>&1 &
  fi
  echo "$!" >"$LOGS/$name.pid"
  echo "   $name: pid $! -> :$port"
done

# Keep the Mac from sleeping while servers run (display may still sleep).
if ! pgrep -f "caffeinate -s" >/dev/null; then
  nohup caffeinate -s >/dev/null 2>&1 &
  echo "$!" >"$LOGS/caffeinate.pid"
  echo "   caffeinate: pid $! (Mac stays awake on AC power)"
fi

echo "== waiting for health (image preloads ML models -- up to 5 min cold)"
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")
for spec in "${SERVICES[@]}"; do
  name="${spec%%:*}"
  port="${spec##*:}"
  [ -f "$LOGS/$name.pid" ] || continue
  deadline=$((SECONDS + 300))
  until curl -s -m 2 -o /dev/null "http://localhost:$port/health"; do
    if [ $SECONDS -ge $deadline ]; then
      echo "   $name: TIMED OUT -- check $LOGS/$name.log"
      continue 2
    fi
    sleep 3
  done
  # Verify the 0.0.0.0 binding from the network side, not just loopback.
  if [ -n "$LAN_IP" ] && curl -s -m 3 -o /dev/null "http://$LAN_IP:$port/health"; then
    echo "   $name: healthy (reachable at $LAN_IP:$port)"
  else
    echo "   $name: healthy on loopback but NOT via LAN -- check macOS firewall"
  fi
done

echo "== done"
echo "   Phones on this Wi-Fi connect to: $(scutil --get LocalHostName 2>/dev/null || hostname).local"
echo "   (fallback if mDNS is blocked: $LAN_IP)"
echo "   Stop everything: scripts/stop_backends.sh"
