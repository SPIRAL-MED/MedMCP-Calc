#!/bin/bash
#
# inference.sh - Main Entry Script for Running the Medmcp-Calc Benchmark
#
# Description:
#   Launches the agent with configurable proxy and runtime settings.
#   Proxy is disabled by default; enable it if you need external API access
#   through a proxy server (e.g., Clash).
#
# Usage:
#   ./inference.sh [additional arguments]



set -euo pipefail


# =============================================================================
# PROXY SETUP
# =============================================================================

# Proxy settings (commonly used with Clash, etc.)
readonly ENABLE_PROXY="${ENABLE_PROXY:-false}"
readonly PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
readonly PROXY_PORT="${PROXY_PORT:-7890}"
readonly PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"
readonly NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}"


if [[ "${ENABLE_PROXY}" == "true" ]]; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export all_proxy="${PROXY_URL}"
    export no_proxy="${NO_PROXY}"
    echo "[INFO] Proxy enabled: ${PROXY_URL}"
else
    echo "[INFO] Proxy disabled"
fi


# =============================================================================
# MAIN
# =============================================================================

python inference.py \
    "$@"