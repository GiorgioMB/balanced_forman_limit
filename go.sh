#!/bin/bash
# usage: bash go.sh <script> [args]  -- runs with hard 115s limit, output to log, kills stragglers
timeout -k 3 115 "$@" > /tmp/go.log 2>&1
pkill -9 -f "python3 [r]un_" 2>/dev/null
cat /tmp/go.log | tail -5
