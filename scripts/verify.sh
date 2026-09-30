#!/usr/bin/env bash
# The project's single verification entry point.
#
# One definition of "verified", used locally and by CI. The rule this script
# exists to enforce: **check exit codes, not output text.**
#
# That rule is not a style preference. For several rounds this project reported
# "ruff clean" while a B905 sat above the summary line, because the checks were
# piped through `tail -n 1` -- which prints "All checks passed!" as the last
# line of a run that also printed a violation. A human-readable summary is a
# claim by the tool; an exit code is a fact. Every check below is run without a
# pipe, its status captured, and the script exits non-zero if any failed.
#
# Usage:
#   scripts/verify.sh                 # everything
#   scripts/verify.sh --quick         # skip the slow experiment scripts
#   scripts/verify.sh --json OUT.json  # machine-readable record of every check
#
# Exit codes: 0 all passed, 1 at least one check failed, 2 the script itself
# could not run (missing interpreter).

set -uo pipefail

QUICK=0
JSON_OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --json) JSON_OUT="${2:-}"; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "$0")/.." || exit 2
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

RESULTS=""
OVERALL=0

# Preflight: fail as an ENVIRONMENT problem, not as a code problem.
#
# Without this, a checkout with no .venv falls back to the system python3, every
# pytest call dies with "No module named pytest", and the script reports
# VERIFICATION FAILED -- which reads as "the code is broken" when the code is
# fine and the environment was never set up. Conflating those two is the same
# error class this project exists to catch, committed in its own checker.
preflight() {
  local missing=0
  for tool in pytest mypy ruff; do
    "$PY" -c "import $tool" 2>/dev/null || missing=1
  done
  "$PY" -c "import robot_ci" 2>/dev/null || {
    echo "robot_ci is not importable by $PY" >&2; missing=1; }
  [ "$missing" -eq 0 ] && return 0
  cat >&2 <<EOF

This checkout has no usable development environment.

  interpreter : $("$PY" -V 2>&1)  ($PY)

That is an environment problem, not a failing check, so it is reported as one
rather than as VERIFICATION FAILED. To set one up:

  python3 -m venv .venv
  .venv/bin/pip install -e ".[dev]"

Then re-run. Exit 2 means the script could not run; exit 1 means a check failed.

EOF
  exit 2
}
preflight

# Advisory checks report, and are recorded in the JSON, but do not fail the
# run. They exist so a known deviation is visible on every run rather than
# discovered later. Promote one to blocking by calling run_check_blocking.
ADVISORY=0

run_advisory() {
  local name="$1"; shift
  local log; log="$(mktemp)"
  "$@" >"$log" 2>&1
  local status=$?
  local detail; detail="$(tail -n 1 "$log" | tr '\n' ' ' | cut -c1-160)"
  if [ "$status" -eq 0 ]; then
    printf '  \033[32mPASS\033[0m  %-34s %s\n' "$name" "$detail"
  else
    printf '  \033[33mWARN\033[0m  %-34s (advisory, non-blocking) %s\n' "$name" "$detail"
  fi
  ADVISORY=1
  RESULTS="${RESULTS}${name}|${status}|advisory: ${detail}"$'\n'
  rm -f "$log"
}

run_check() {
  local name="$1"; shift
  local log; log="$(mktemp)"
  "$@" >"$log" 2>&1
  local status=$?
  # The tail is for the reader; the status is the verdict.
  local detail; detail="$(tail -n 2 "$log" | tr '\n' ' ' | cut -c1-160)"
  if [ "$status" -eq 0 ]; then
    printf '  \033[32mPASS\033[0m  %-34s %s\n' "$name" "$detail"
  else
    printf '  \033[31mFAIL\033[0m  %-34s exit=%s %s\n' "$name" "$status" "$detail"
    sed -n '$p' "$log" | sed 's/^/          /' >&2
    OVERALL=1
  fi
  RESULTS="${RESULTS}${name}|${status}|${detail}"$'\n'
  rm -f "$log"
}

echo "ROBOT CI verification"
echo "  interpreter : $("$PY" -V 2>&1)"
echo "  commit      : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  date        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

echo "static analysis"
run_check "ruff (src, tests, experiments)" "$PY" -m ruff check src/robot_ci tests experiments
# Advisory, not blocking. `ruff check` (the linter) passes on all of this;
# `ruff format` is a separate tool that had never been run here until this
# script was written, and it reports 40 files that the formatter would restyle.
# That is a real deviation and is recorded on every run, but reformatting 40
# files in the same commit as the baseline tag would bury the audit trail, so it
# is fixed separately and recorded here in the meantime.
run_advisory "ruff format --check"          "$PY" -m ruff format --check src/robot_ci
run_check "mypy (src/robot_ci)"            "$PY" -m mypy src/robot_ci
echo

echo "tests"
run_check "unit"                          "$PY" -m pytest tests/unit -q
run_check "integration"                    "$PY" -m pytest tests/integration -q
run_check "end to end"                    "$PY" -m pytest tests/e2e -q
echo

echo "environment"
run_check "robot-ci validate"             "$PY" -m robot_ci.cli.main validate
run_check "package imports without mujoco" "$PY" -m pytest tests/unit/test_simulation_contract.py -q
echo

if [ "$QUICK" -eq 0 ]; then
  echo "research experiments (slow; these regenerate the recorded results)"
  run_check "benchmark suite"    "$PY" experiments/run_benchmark_suite.py
  run_check "regression grid"    "$PY" experiments/rq2_conditional_vs_aggregate.py
  echo
fi

if [ -n "$JSON_OUT" ]; then
  {
    printf '{"commit": "%s", "date": "%s", "overall": %s, "checks": [\n' \
      "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OVERALL"
    local_first=1
    printf '%s' "$RESULTS" | while IFS='|' read -r n s d; do
      [ -z "$n" ] && continue
      [ "$local_first" -eq 0 ] && printf ',\n'
      local_first=0
      printf '    {"name": "%s", "exit": %s, "detail": "%s"}' "$n" "$s" "$d"
    done
    printf '\n  ]}\n'
  } > "$JSON_OUT"
  echo "record written to $JSON_OUT"
fi

echo
if [ "$OVERALL" -eq 0 ]; then
  printf '\033[32mVERIFICATION PASSED\033[0m  every check exited 0\n'
else
  printf '\033[31mVERIFICATION FAILED\033[0m  see the FAIL lines above\n'
fi
exit "$OVERALL"
