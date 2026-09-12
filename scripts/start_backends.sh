#!/bin/bash
# Start all four POC backends for the beta, bound to loopback. Keeps the Mac
# awake while they run. Companion: stop_backends.sh.
#
# Loopback is deliberate and is the whole security posture of this deployment:
# the Cloudflare tunnel is the only way in, and the tunnel reaches the services
# over localhost. Binding 0.0.0.0 would put four services that spend API
# credits on every Wi-Fi the Mac ever joins, with no gate in front of them.
# The tunnel itself is started separately by scripts/start_tunnel.sh.
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
      --host 127.0.0.1 --port "$port" >"$LOGS/$name.log" 2>&1 &
  else
    nohup .venv/bin/uvicorn "$app" \
      --host 127.0.0.1 --port "$port" >"$LOGS/$name.log" 2>&1 &
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
  # Prove the binding is loopback-only by checking from the LAN address, where
  # the answer must be silence. A reachable service here is the bug, not the
  # unreachable one.
  if [ -n "$LAN_IP" ] && curl -s -m 3 -o /dev/null "http://$LAN_IP:$port/health"; then
    echo "   $name: healthy BUT ALSO REACHABLE AT $LAN_IP:$port -- it is not"
    echo "      bound to loopback; stop it before starting the tunnel"
  else
    echo "   $name: healthy on loopback only"
  fi
done

echo "== done"
echo "   Testers reach these through the tunnel, not over Wi-Fi:"
echo "      https://beta-text.miranote.app    -> :8001"
echo "      https://beta-image.miranote.app   -> :8002"
echo "      https://beta-chat.miranote.app    -> :8003"
echo "      https://beta-voice.miranote.app   -> :8005"
echo "   Tunnel: scripts/start_tunnel.sh (separate lifecycle, start it too)"
echo "   Stop everything: scripts/stop_backends.sh"
