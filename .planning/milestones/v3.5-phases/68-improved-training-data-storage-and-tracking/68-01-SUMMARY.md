---
phase: 68-improved-training-data-storage-and-tracking
plan: 01
subsystem: database
tags: [sqlite, training-data, dedup, provenance]

# Dependency graph
requires: []
provides:
  - SampleStore class for SQLite-backed training data management
  - Content-hash dedup with source-priority upsert
  - Tag-based querying with AND/OR semantics
  - Exclude/include soft-delete and hard remove with cascade
  - Augmentation lineage tracking via parent_id
  - Provenance history on all mutations
affects: [68-02, 68-03, 68-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [sqlite-wal-store, content-hash-dedup, json-tag-filtering]

key-files:
  created:
    - src/aquapose/training/store_schema.py
    - src/aquapose/training/store.py
    - tests/unit/training/test_store.py
  modified:
    - src/aquapose/training/__init__.py

key-decisions:
  - "JSON columns for tags, provenance, metadata (sqlite json_each for filtering)"
  - "PRAGMA user_version for schema versioning with forward-compat check"
  - "Source priority: manual(2) > corrected(1) > pseudo(0) for dedup upsert"

patterns-established:
  - "SampleStore context manager pattern with lazy _connect()"
  - "Content-hash dedup with source-priority upsert on import"
  - "Cascade operations: upsert deletes children files+rows, exclude/include propagates tags"

requirements-completed: [STORE-01, STORE-02]

# Metrics
duration: 4min
completed: 2026-03-06
---

# Phase 68 Plan 01: SampleStore Summary

**SQLite-backed SampleStore with content-hash dedup, source-priority upsert, provenance tracking, and tag-based query filtering via json_each**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-06T17:28:02Z
- **Completed:** 2026-03-06T17:31:58Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- SampleStore class with full CRUD: import, query, get, count, exclude, include, remove
- Content-hash dedup with source-priority upsert (manual > corrected > pseudo)
- Augmentation lineage via parent_id with cascade on upsert, exclude, include, and remove
- Provenance JSON array tracking import and replace history
- Tag-based querying with AND semantics for include, OR semantics for exclude
- min_confidence filter that passes samples without confidence key (manual/corrected assumed high quality)
- 22 comprehensive unit tests all passing

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests for SampleStore** - `a23a15f` (test)
2. **GREEN: Implement SampleStore** - `49855c1` (feat)

## Files Created/Modified
- `src/aquapose/training/store_schema.py` - SQL DDL, SCHEMA_VERSION, SOURCE_PRIORITY constants
- `src/aquapose/training/store.py` - SampleStore class with all CRUD operations
- `tests/unit/training/test_store.py` - 22 unit tests covering all operations
- `src/aquapose/training/__init__.py` - Added SampleStore export

## Decisions Made
- JSON columns for tags, provenance, and metadata with sqlite json_each for tag filtering
- PRAGMA user_version for schema versioning; raises error if DB is newer than code
- Source priority dict: manual=2, corrected=1, pseudo=0; equal priority triggers upsert
- WAL mode + busy_timeout=5000ms for concurrent access safety

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- SampleStore API complete and tested, ready for Plan 02 (CLI import/query commands)
- Schema supports datasets and models tables for Plans 03 and 04

## Self-Check: PASSED

All 4 files found. Both commit hashes verified.

---
*Phase: 68-improved-training-data-storage-and-tracking*
*Completed: 2026-03-06*
