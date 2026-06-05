#!/usr/bin/env bash
# start-all.sh -- spin up the four MiraNote POC servers concurrently for
# local dev. Ctrl-C stops them all.
#
# Skips any POC whose .venv or .env is missing (with a setup hint) so a
# partial environment still works -- you can run just the POCs you've
# configured. Uses --reload so edits are picked up live.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefix each service's output with a coloured [tag] so a single terminal
# stays readable. Args: tag, ANSI-colour-code.
prefix() {
  sed -u "s/^/$(printf '\033[%sm[%s]\033[0m' "$2" "$1") /"
}

# Args: tag colour venv-dir start-cwd uvicorn-extra-args module port
start() {
  local tag=$1 colour=$2 venv_dir=$3 start_cwd=$4 extra=$5 module=$6 port=$7
  local venv="$venv_dir/.venv"
  local short="${venv_dir#$REPO_ROOT/}"

  if [[ ! -x "$venv/bin/python3" ]]; then
    printf '\033[%sm[%s]\033[0m  skipped (no .venv).  setup: cd %s && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cp .env.example .env\n' \
      "$colour" "$tag" "$short"
    return
  fi
  if [[ ! -f "$venv_dir/.env" ]]; then
    printf '\033[%sm[%s]\033[0m  skipped (no .env).  cd %s && cp .env.example .env, then fill LLM_API_KEY\n' \
      "$colour" "$tag" "$short"
    return
  fi

  printf '\033[%sm[%s]\033[0m  starting on port %s\n' "$colour" "$tag" "$port"
  # exec replaces the subshell with python so the backgrounded PID we
  # leak to $! is python itself -- pkill -P $$ on Ctrl-C then signals
  # python directly, which tears down its reload-worker children.
  # Word-split $extra deliberately so "--reload --reload-dir x" becomes
  # multiple uvicorn args.
  # shellcheck disable=SC2086
  (
    cd "$start_cwd"
    exec env PYTHONPATH="$REPO_ROOT" "$venv/bin/python3" -m uvicorn "$module" --port "$port" $extra
  ) 2>&1 | prefix "$tag" "$colour" &
}

cleanup() {
  echo
  echo "stopping all servers..."
  pkill -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

#     tag        colour  venv-dir                                start-cwd                                 uvicorn-extra-args                       module                  port
start text       36      "$REPO_ROOT/poc/text-clean-expand"      "$REPO_ROOT/poc/text-clean-expand"        "--reload"                               main:app                8001
start voice      33      "$REPO_ROOT/poc/voice-to-text"          "$REPO_ROOT/poc/voice-to-text"            "--reload"                               main:app                8000
start chat       35      "$REPO_ROOT/poc/chatbot"                "$REPO_ROOT"                              "--reload --reload-dir poc/chatbot"      poc.chatbot.main:app    8003
start retrieval  32      "$REPO_ROOT/poc/retrieval"              "$REPO_ROOT"                              "--reload --reload-dir poc/retrieval"    poc.retrieval.main:app  8004

cat <<EOF

  Unified UI:  http://localhost:8001/
  Press Ctrl-C to stop all servers.

EOF
wait
