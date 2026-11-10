#!/usr/bin/env bash
#
# Bisects the batched-decode regression on feature/performance: batched ITL
# went from ~17 ms to ~35 ms (p99 up to ~91 ms) and batched throughput roughly
# halved, while single-request and long-prompt numbers improved.
#
# Checks out each commit between the last known-good engine (d27040f, which is
# base-engine plus the new benchmark) and HEAD of feature/performance, runs
# tests/test_engine_benchmark.py on it RUNS times, and prints a summary of
# the batched numbers per run. The first commit whose batched ITL / throughput
# jumps is the one to profile next with scripts/profile_decode_step.py.
#
# The benchmark asserts a 2.0x batched/single speedup floor, so slow commits
# may report a test failure. That's expected; the numbers are still captured.
#
# Run on the GPU box, from the repo root, with a clean working tree (untracked
# files are fine) and the .venv set up. Returns to the starting branch when
# done, even if interrupted.
#
# Usage:  ./scripts/bisect_batched_regression.sh [commit ...]
#
#   With no arguments, bisects the default commit list below.
#
# Environment:
#   RUNS        runs per commit (default 2)
#   BISECT_OUT  log directory (default /tmp/bisect)
#   PYTHON      interpreter used to run pytest (default .venv/bin/python if it
#               exists, else python)
#
set -euo pipefail

DEFAULT_COMMITS=(d27040f 200dcb0 9e8c5cc e6d0428 4b2f3d1 6c34124)

main() {
    cd "$(git rev-parse --show-toplevel)"

    local runs="${RUNS:-2}"
    local out="${BISECT_OUT:-/tmp/bisect}"
    local python="${PYTHON:-}"
    if [[ -z "$python" ]]; then
        if [[ -x .venv/bin/python ]]; then python=.venv/bin/python; else python=python; fi
    fi

    local commits=("$@")
    if [[ ${#commits[@]} -eq 0 ]]; then commits=("${DEFAULT_COMMITS[@]}"); fi

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "error: working tree has uncommitted changes; commit or stash them first" >&2
        exit 1
    fi
    for c in "${commits[@]}"; do
        if ! git cat-file -e "$c^{commit}" 2>/dev/null; then
            echo "error: commit $c not found (try: git fetch origin feature/performance)" >&2
            exit 1
        fi
    done

    # Return to wherever we started, branch name if on one, else the commit.
    local start
    start="$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)"
    trap 'git checkout -q "'"$start"'" && echo "restored $start"' EXIT

    mkdir -p "$out"
    echo "logs: $out   runs per commit: $runs   python: $python"

    for c in "${commits[@]}"; do
        git checkout -q "$c"
        for run in $(seq 1 "$runs"); do
            local log="$out/$c-$run.log"
            echo "=== $c run $run ($(git log -1 --format=%s "$c"))"
            # Don't let a failed speedup assert stop the bisect.
            "$python" -m pytest tests/test_engine_benchmark.py -s -q > "$log" 2>&1 || true
            grep -A3 "^batched (" "$log" || echo "  (no batched report; see $log)"
            grep "^batched / single" "$log" || true
        done
    done

    summarize "$out" "$runs" "${commits[@]}"
}

# One row per (commit, run) with the batched numbers pulled from the report.
summarize() {
    local out="$1" runs="$2"
    shift 2

    echo
    printf '%-9s %3s  %13s  %12s  %14s  %8s\n' \
        commit run "ITL mean ms" "ITL p99 ms" "throughput t/s" speedup
    for c in "$@"; do
        for run in $(seq 1 "$runs"); do
            local log="$out/$c-$run.log"
            awk -v c="$c" -v run="$run" '
                /^batched \(/            { in_batched = 1; next }
                in_batched && /ITL mean/ { itl = $(NF-3); p99 = $(NF-1) }
                in_batched && /Throughput/ { tput = $(NF-1); in_batched = 0 }
                /^batched \/ single/     { speedup = $NF }
                END {
                    printf "%-9s %3s  %13s  %12s  %14s  %8s\n", c, run,
                        itl ? itl : "-", p99 ? p99 : "-",
                        tput ? tput : "-", speedup ? speedup : "-"
                }
            ' "$log"
        done
    done
}

# Wrapped in main and exited on the same line so bash has the whole script
# parsed before the checkouts start, even if this file changes on disk.
main "$@"; exit $?
