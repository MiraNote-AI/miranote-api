# Sourced, not run. The launchd side of the beta backends: one job per
# service, labelled ai.miranote.beta-<name> after the public hostname it
# serves (beta-text.miranote.app -> ai.miranote.beta-text).
#
# Used by install_backend_services.sh, status_backends.sh and
# disable_backend_services.sh. The table is the same one start_backends.sh
# carries; keep the two in step.

# name : working dir (under poc/) : uvicorn app spec : port
BETA_SERVICES=(
  "text:text-clean-expand:main:app:8001"
  "image:image-generation:main:app:8002"
  "chat:chatbot:poc.chatbot.main:app:8003"
  "voice:voice-to-text:main:app:8005"
  "collections:photo-collections:main:app:8006"
)

BETA_HOSTS="beta-text beta-image beta-chat beta-voice beta-collections"
BETA_HEALTHCHECK_LABEL="ai.miranote.beta-healthcheck"
BETA_TUNNEL_LABEL="ai.miranote.beta-tunnel"

# Split one BETA_SERVICES entry into svc_name, svc_dir, svc_app, svc_port.
# The app spec itself contains a colon, so the port is taken from the end.
beta_parse() {
  svc_name="${1%%:*}"
  local rest="${1#*:}"
  svc_dir="${rest%%:*}"
  rest="${rest#*:}"
  svc_app="${rest%:*}"
  svc_port="${1##*:}"
}

beta_label() { echo "ai.miranote.beta-$1"; }

# True when launchd has the job loaded (running or waiting to restart).
beta_loaded() { launchctl print "gui/$(id -u)/$1" >/dev/null 2>&1; }
