#!/usr/bin/env bash
set -euo pipefail

# Run all Cramer-Short tests (TypeScript + Python, unit + integration + e2e)
# Usage: bash scripts/run-all-tests.sh [--no-e2e]

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NO_E2E=false
if [[ "${1:-}" == "--no-e2e" ]]; then
  NO_E2E=true
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
if bun test 2>&1; then
  TS_PASS=$((TS_PASS + 1))
else
  TS_FAIL=$((TS_FAIL + 1))
  TOTAL_EXIT=1
fi
echo ""

# ─── TypeScript Integration Tests ───────────────────────────
echo "── TypeScript Integration Tests ─────────────────────────"
if RUN_INTEGRATION=1 bun test 2>&1; then
  TS_PASS=$((TS_PASS + 1))
else
  TS_FAIL=$((TS_FAIL + 1))
  TOTAL_EXIT=1
fi
echo ""

# ─── TypeScript E2E Tests ───────────────────────────────────
if [[ "$NO_E2E" == false ]]; then
  echo "── TypeScript E2E Tests ─────────────────────────────────"
  # Run e2e tests with extended timeout (some take 5+ min each)
  if RUN_E2E=1 bun test --timeout 600000 2>&1; then
    TS_PASS=$((TS_PASS + 1))
  else
    TS_FAIL=$((TS_FAIL + 1))
    TOTAL_EXIT=1
  fi
  echo ""
else
  echo "── TypeScript E2E Tests: SKIPPED (pass --no-e2e) ──────────"
  echo ""
fi

# ─── Python Tests ───────────────────────────────────────────
echo "── Python Tests (research/) ─────────────────────────────"
cd "$REPO_ROOT/research"

# Ensure the package is installed in editable mode
if ! python -c "import research" 2>/dev/null; then
  echo "Installing cramer-research in editable mode..."
  pip install -e "$REPO_ROOT/research" >/dev/null 2>&1
fi

if PYTHONPATH="$REPO_ROOT" pytest -v 2>&1; then
  PY_PASS=$((PY_PASS + 1))
else
  PY_FAIL=$((PY_FAIL + 1))
  TOTAL_EXIT=1
fi
cd "$REPO_ROOT"
echo ""

# ─── Summary ────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  Results Summary"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  TypeScript tests:  $TS_PASS passed, $TS_FAIL failed"
echo "  Python tests:      $PY_PASS passed, $PY_FAIL failed"
echo ""

if [[ "$NO_E2E" == true ]]; then
  echo "  (E2E tests were skipped — pass without --no-e2e to include)"
  echo ""
fi

if [[ $TOTAL_EXIT -eq 0 ]]; then
  echo "  All tests passed."
else
  echo "  Some tests failed."
fi

exit $TOTAL_EXIT
