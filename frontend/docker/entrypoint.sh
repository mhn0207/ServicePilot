#!/bin/sh
set -eu

cat >/usr/share/nginx/html/runtime-config.js <<EOF
window.__SERVICEPILOT_CONFIG__ = window.__ZHIYING_CONFIG__ = {
  servicepilotApiUrl: "${PYTHON_API_URL:-/api/servicepilot}",
  zhiyingApiUrl: "${PYTHON_API_URL:-/api/zhiying}"
};
EOF

exec nginx -g "daemon off;"
