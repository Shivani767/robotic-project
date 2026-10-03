#!/usr/bin/env bash
# Start one live Gazebo evidence run inside the container, detached, so the
# caller can poll instead of blocking (episodes take a couple of minutes).
#
# Usage:
#   scripts/gz_run.sh start  <container-command>
#   scripts/gz_run.sh status
#   scripts/gz_run.sh fetch  <host-path>
#   scripts/gz_run.sh stop
set -euo pipefail

cd "$(dirname "$0")/.."

# Kill by exact process name. A -f pattern would match the very shell that
# is running this script (its command line contains the pattern), killing the
# script itself with SIGTERM before it ever starts anything.
CLEAN='pkill -x python3 2>/dev/null || true; pkill -x gz 2>/dev/null || true; pkill -x ruby 2>/dev/null || true; sleep 2'

case "${1:-}" in
  start)
    shift
    docker exec rc-gz bash -lc "$CLEAN"
    # Truncate first: status greps for the DONE marker, and a stale log from the
    # previous run would make a fresh run look finished before it started.
    docker exec rc-gz bash -lc ': > /tmp/gz_run.log'
    docker exec -d rc-gz bash -lc "$* > /tmp/gz_run.log 2>&1; echo DONE >> /tmp/gz_run.log"
    echo "started"
    ;;
  status)
    # Prints "running <pid>" or "done". A missing log means nothing started.
    docker exec rc-gz bash -lc \
      'if [ ! -f /tmp/gz_run.log ]; then echo no-log; elif grep -q "^DONE$" /tmp/gz_run.log; then echo done; else echo running; fi'
    ;;
  fetch)
    docker exec rc-gz bash -lc 'cat /tmp/gz_run.log' > "$2"
    ;;
  stop)
    docker exec rc-gz bash -lc "$CLEAN"
    echo "stopped"
    ;;
  *)
    echo "usage: $0 start|status|fetch|stop" >&2
    exit 2
    ;;
esac
