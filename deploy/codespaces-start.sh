#!/usr/bin/env bash
set -euo pipefail
# Keep the forwarded port private unless you deliberately change its visibility.
if [[ -n "${CODESPACE_NAME:-}" && -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]]; then
  export OPENJEV_PUBLIC_ORIGIN="https://${CODESPACE_NAME}-7860.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
fi
if ! python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz')" >/dev/null 2>&1; then
  nohup openjev serve --host 0.0.0.0 --port 7860 >/tmp/openjev.log 2>&1 </dev/null &
fi
