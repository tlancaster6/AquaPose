#!/usr/bin/env bash
# Shell wrapper for the AquaPose import boundary and structural rule checker.
#
# Invoked by pre-commit with the list of changed Python files as arguments.
# Passes all arguments through to the Python checker.
set -euo pipefail
python tools/import_boundary_checker.py "$@"
