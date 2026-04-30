#!/usr/bin/env bash
set -euo pipefail

# Run all Cramer-Short tests (TypeScript + Python, unit + integration + e2e)
# Usage: bash scripts/run-all-tests.sh [--skip-e2e]

# Guard: prevent accidental sourcing (which would kill the caller's shell via exit)
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "ERROR: This script must be executed, not sourced." >&2
  echo "Usage: bash scripts/run-all-tests.sh [--skip-e2e]" >&2
  return 1 2>/dev/null || exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_E2E=false
if [[ "${1:-}" == "--skip-e2e" ]]; then
  SKIP_E2E=true
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Cramer-Short Full Test Suite"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─── Track results ──────────────────────────────────────────
TS_PASS=0
TS_FAIL=0
PY_PASS=0
PY_FAIL=0
TOTAL_EXIT=0

# ─── TypeScript Unit Tests ──────────────────────────────────
echo "── TypeScript Unit Tests ────────────────────────────────"
if bun test --ignore 'src/**/*.integration.test.ts' --ignore 'src/**/*.e2e.test.ts' 2>&1; then
  TS_PASS=$((TS_PASS + 1))
else
  TS_FAIL=$((TS_FAIL + 1))
  TOTAL_EXIT=1
fi
echo ""

# ─── TypeScript Integration Tests ───────────────────────────
echo "── TypeScript Integration Tests ─────────────────────────"
if bun run test:integration 2>&1; then
  TS_PASS=$((TS_PASS + 1))
else
  TS_FAIL=$((TS_FAIL + 1))
  TOTAL_EXIT=1
fi
echo ""

# ─── TypeScript E2E Tests ───────────────────────────────────
if [[ "$SKIP_E2E" == false ]]; then
  echo "── TypeScript E2E Tests ─────────────────────────────────"
  if bun run test:e2e 2>&1; then
    TS_PASS=$((TS_PASS + 1))
  else
    TS_FAIL=$((TS_FAIL + 1))
    TOTAL_EXIT=1
  fi
  echo ""
else
  echo "── TypeScript E2E Tests: SKIPPED (--skip-e2e was passed) ─────────────"
  echo ""
fi

# ─── Python Tests ───────────────────────────────────────────
echo "── Python Tests (research/) ─────────────────────────────"

# Use conda environment as specified in research/README.md
CONDA_ENV="cramer-research"
CONDA_SH=""

# Locate conda.sh — check common install locations
for candidate in \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  if [[ -f "$candidate" ]]; then
    CONDA_SH="$candidate"
    break
  fi
done

if [[ -z "$CONDA_SH" ]]; then
  echo "ERROR: conda.sh not found. Cannot run Python tests." >&2
  echo "Expected at ~/anaconda3/etc/profile.d/conda.sh or similar." >&2
  PY_FAIL=$((PY_FAIL + 1))
  TOTAL_EXIT=1
else
  source "$CONDA_SH"
  conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "ERROR: conda environment '${CONDA_ENV}' not found." >&2
    echo "Create it: conda env create -f environment-research.yml" >&2
    PY_FAIL=$((PY_FAIL + 1))
    TOTAL_EXIT=1
  }
fi

if [[ $PY_FAIL -eq 0 ]]; then
  # Install package in editable mode within the conda environment
  pip install -e "$REPO_ROOT/research" --quiet 2>&1

  if python -m pytest research/tests -v 2>&1; then
    PY_PASS=$((PY_PASS + 1))
  else
    PY_FAIL=$((PY_FAIL + 1))
    TOTAL_EXIT=1
  fi
fi
echo ""

# ─── Summary ────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  Results Summary"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  TypeScript suites:  $TS_PASS passed, $TS_FAIL failed"
echo "  Python suites:      $PY_PASS passed, $PY_FAIL failed"
echo ""

if [[ "$SKIP_E2E" == true ]]; then
  echo "  (E2E tests were skipped — omit --skip-e2e to include)"
  echo ""
fi

if [[ $TOTAL_EXIT -eq 0 ]]; then
  echo "  All tests passed."
else
  echo "  Some tests failed."
fi

exit $TOTAL_EXIT
