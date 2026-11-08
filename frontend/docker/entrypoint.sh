#!/bin/sh
set -eu

cat >/usr/share/nginx/html/runtime-config.js <<EOF
window.__ZHIYING_CONFIG__ = {
  zhiyingApiUrl: "${PYTHON_API_URL:-/api/zhiying}"
};
EOF

exec nginx -g "daemon off;"
