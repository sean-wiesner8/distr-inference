#!/usr/bin/env bash
#
# Profiles decode steps on the last good commit and the regressed code, so
# their per-step breakdowns can be compared side by side.
#
# The bisect (scripts/bisect_batched_regression.sh) put the batched-decode
# regression at e6d0428; 9e8c5cc is the last commit with fast batched ITL.
# This runs scripts/profile_decode_step.py on each commit at each batch size
# and saves the report plus a Perfetto trace. Batch 1 is the control: it
# didn't regress, so whatever grows only at batch 8 is the suspect.
#
# Compare across the logs: wall vs GPU ms per step (a gap that grows means
# the GPU is idle waiting on the host), syncs / copies / launches per step,
# and which ops moved in the top CPU and GPU tables. Traces open in
# https://ui.perfetto.dev.
#
# Run on the GPU box, from the repo root, with a clean working tree (untracked
# files are fine) and the .venv set up. Returns to the starting branch when
# done, even if interrupted.
#
# Usage:  ./scripts/profile_regression.sh [commit ...]
#
#   With no arguments, profiles 9e8c5cc (good) and feature/performance (bad).
#
# Environment:
#   BATCH_SIZES   space-separated batch sizes (default "8 1")
#   PROFILE_OUT   output directory (default /tmp/profile)
#   PROFILE_ARGS  extra arguments for profile_decode_step.py, e.g. "--stack"
#   PYTHON        interpreter (default .venv/bin/python if it exists, else python)
#
set -euo pipefail

DEFAULT_COMMITS=(9e8c5cc feature/performance)

main() {
    cd "$(git rev-parse --show-toplevel)"

    local batch_sizes="${BATCH_SIZES:-8 1}"
    local out="${PROFILE_OUT:-/tmp/profile}"
    local python="${PYTHON:-}"
    if [[ -z "$python" ]]; then
        if [[ -x .venv/bin/python ]]; then python=.venv/bin/python; else python=python; fi
    fi
    local extra=()
    read -r -a extra <<< "${PROFILE_ARGS:-}"

    local commits=("$@")
    if [[ ${#commits[@]} -eq 0 ]]; then commits=("${DEFAULT_COMMITS[@]}"); fi

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "error: working tree has uncommitted changes; commit or stash them first" >&2
        exit 1
    fi

    # Resolve every ref to a commit up front: a branch name has to mean the
    # same commit no matter what is checked out when its turn comes.
    local shas=() sha
    for c in "${commits[@]}"; do
        if ! sha="$(git rev-parse --short --verify -q "$c^{commit}")"; then
            echo "error: commit $c not found (try: git fetch origin)" >&2
            exit 1
        fi
        shas+=("$sha")
    done

    local start
    start="$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)"
    trap 'git checkout -q "'"$start"'" && echo "restored $start"' EXIT

    mkdir -p "$out"
    echo "output: $out   batch sizes: $batch_sizes   python: $python"

    local logs=()
    for i in "${!shas[@]}"; do
        local sha="${shas[$i]}"
        git checkout -q "$sha"
        for bs in $batch_sizes; do
            local name="$sha-bs$bs"
            echo "=== ${commits[$i]} ($sha) batch $bs: $(git log -1 --format=%s "$sha")"
            if "$python" scripts/profile_decode_step.py --batch-size "$bs" \
                    --trace "$out/$name.json" "${extra[@]}" > "$out/$name.log" 2>&1; then
                logs+=("$out/$name.log")
            else
                echo "  profiler failed; see $out/$name.log"
            fi
        done
    done

    echo
    echo "reports:"
    printf '  %s\n' "${logs[@]}"
    echo "view all:  cat ${logs[*]}"
}

# Wrapped in main and exited on the same line so bash has the whole script
# parsed before the checkouts start, even if this file changes on disk.
main "$@"; exit $?
