#!/bin/bash
set -e

# Ensure cache directory exists and has correct permissions
# Run as root to fix permissions, then switch to appuser
if [ "$(id -u)" = "0" ]; then
    mkdir -p /home/appuser/.cache/huggingface
    chown -R appuser:appuser /home/appuser/.cache 2>/dev/null || true
    chmod -R 755 /home/appuser/.cache 2>/dev/null || true
    # Switch to appuser and execute command
    exec gosu appuser "$@"
else
    # Already running as appuser, just ensure directory exists
    mkdir -p /home/appuser/.cache/huggingface 2>/dev/null || true
    exec "$@"
fi





