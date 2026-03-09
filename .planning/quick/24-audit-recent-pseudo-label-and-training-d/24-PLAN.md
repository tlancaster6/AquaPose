---
phase: 24-audit-recent-pseudo-label-and-training-d
plan: 24
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/quick/24-audit-recent-pseudo-label-and-training-d/24-AUDIT-REPORT.md
autonomous: true
requirements: [AUDIT-01]

must_haves:
  truths:
    - "Report covers all 21 source files in src/aquapose/training/"
    - "Each improvement item has a concrete LOE estimate (small/medium/large with approximate time)"
    - "Each improvement item has an ROI assessment (impact vs effort)"
    - "Report distinguishes structural/architectural issues from localized code quality issues"
    - "User can use report to decide which improvements to pursue"
  artifacts:
    - path: ".planning/quick/24-audit-recent-pseudo-label-and-training-d/24-AUDIT-REPORT.md"
      provides: "Prioritized audit findings with LOE and ROI"
      min_lines: 80
  key_links: []
---

<objective>
Audit the src/aquapose/training/ module (~8400 LOC across 21 files) and its tests (~5800 LOC) for code quality improvements. Produce a prioritized report where each finding includes level-of-effort and return-on-investment so the user can decide which improvements to pursue.

Purpose: The training module has grown rapidly across phases 66-73 (pseudo-labels, SampleStore, Label Studio integration, elastic deformation, CLI commands). Fast iteration has likely left behind structural debt, duplication, inconsistent patterns, and test gaps worth addressing before the next round of feature work.

Output: 24-AUDIT-REPORT.md with prioritized findings
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@.claude/skills/coding/elegance-review.md

Source files to audit (read ALL of these):
@src/aquapose/training/store.py (909 LOC - largest, SQLite sample store)
@src/aquapose/training/pseudo_label_cli.py (1302 LOC - largest CLI)
@src/aquapose/training/data_cli.py (850 LOC - data management CLI)
@src/aquapose/training/store_schema.py (65 LOC - schema definitions)
@src/aquapose/training/pseudo_labels.py (539 LOC - pseudo-label generation)
@src/aquapose/training/coco_convert.py (596 LOC - COCO format conversion)
@src/aquapose/training/cli.py (464 LOC - training CLI)
@src/aquapose/training/elastic_deform.py (378 LOC - elastic augmentation)
@src/aquapose/training/geometry.py (354 LOC - geometric utilities)
@src/aquapose/training/run_manager.py (343 LOC - training run management)
@src/aquapose/training/select_diverse_subset.py (330 LOC - diverse subset selection)
@src/aquapose/training/prep.py (319 LOC - data preparation)
@src/aquapose/training/labelstudio_import.py (314 LOC - Label Studio import)
@src/aquapose/training/datasets.py (270 LOC - dataset utilities)
@src/aquapose/training/coco_interchange.py (228 LOC - COCO interchange)
@src/aquapose/training/labelstudio_export.py (232 LOC - Label Studio export)
@src/aquapose/training/compare.py (218 LOC - model comparison)
@src/aquapose/training/common.py (186 LOC - shared utilities)
@src/aquapose/training/yolo_pose.py (112 LOC - YOLO pose format)
@src/aquapose/training/yolo_obb.py (108 LOC - YOLO OBB format)
@src/aquapose/training/yolo_seg.py (108 LOC - YOLO segmentation format)
@src/aquapose/training/__init__.py (136 LOC - module exports)

Test files to review for coverage gaps:
@tests/unit/training/test_store.py (996 LOC)
@tests/unit/training/test_pseudo_label_cli.py (687 LOC)
@tests/unit/training/test_data_cli.py (1334 LOC)
@tests/unit/training/test_pseudo_labels.py (734 LOC)
@tests/unit/training/test_elastic_deform.py (269 LOC)
@tests/unit/training/test_elastic_deform_cli.py (28 LOC)
@tests/unit/training/test_run_manager.py (377 LOC)
@tests/unit/training/test_coco_interchange.py (263 LOC)
@tests/unit/training/test_common.py (191 LOC)
@tests/unit/training/test_geometry.py (305 LOC)
@tests/unit/training/test_compare.py (184 LOC)
@tests/unit/training/test_labelstudio.py (251 LOC)
@tests/unit/training/test_training_cli.py (174 LOC)
@tests/unit/training/test_yolo_pose.py (23 LOC)
@tests/unit/training/test_yolo_seg.py (23 LOC)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Audit training module and produce prioritized report</name>
  <files>.planning/quick/24-audit-recent-pseudo-label-and-training-d/24-AUDIT-REPORT.md</files>
  <action>
Read ALL 21 source files in src/aquapose/training/ and ALL 15 test files in tests/unit/training/. Apply the repo-scope elegance review checklist from .claude/skills/coding/elegance-review.md, focusing on:

**Structural analysis:**
- Duplicated logic across modules that should be consolidated (e.g., repeated YOLO label parsing, path resolution patterns, CLI boilerplate)
- Modules that have become catch-alls and should be split (especially pseudo_label_cli.py at 1302 LOC and store.py at 909 LOC)
- Missing shared abstractions that multiple modules reinvent independently
- Inconsistent patterns across files (error handling, naming, data flow conventions)
- Dead code, orphaned utilities, unused exports

**Interface and API analysis:**
- Leaky abstractions where callers know too much about internal state
- Inconsistent return types or error signaling across similar functions
- Functions that do too much (the "and" test)

**Test coverage analysis:**
- Compare test file sizes against source sizes to identify undertested modules (especially: yolo_pose at 23 LOC test for 112 LOC source, yolo_seg similarly, elastic_deform_cli at 28 LOC test for CLI commands)
- Missing edge case coverage
- Test patterns that could be improved (e.g., excessive mocking hiding real bugs)

**Data flow analysis:**
- Unnecessary mutation where transformation pipelines would be clearer
- State held in broader scope than necessary
- Data threaded through many layers to reach one consumer

Produce 24-AUDIT-REPORT.md with this structure:

```markdown
# Training Module Audit Report

## Summary
[Brief overview: total findings, breakdown by category, overall health assessment]

## Findings

### Category: [Structural / Interface / Test Coverage / Data Flow / Code Quality]

#### Finding N: [Title]
- **Files:** [specific files involved]
- **Issue:** [what the problem is, with specific line references or code snippets]
- **Impact:** [what goes wrong if not fixed -- bugs, maintenance burden, confusion]
- **LOE:** [Small (<30min) / Medium (30-90min) / Large (2-4hr)] with brief justification
- **ROI:** [High / Medium / Low] -- ratio of impact to effort
- **Recommendation:** [specific action to take]

[Repeat for each finding]

## Priority Matrix

| # | Finding | LOE | ROI | Category |
|---|---------|-----|-----|----------|

## Recommended Execution Order
[Group findings into batches the user might want to tackle together, ordered by ROI]
```

Important guidelines:
- Be specific: cite file names, function names, line ranges. Vague findings are useless.
- Be honest about ROI: not everything is worth fixing. Some technical debt is acceptable in research code.
- Distinguish "should fix before next feature round" from "nice to have someday."
- Do NOT implement any changes -- this is audit only. The user will decide what to pursue.
- Focus on the training module specifically, not the broader codebase.
  </action>
  <verify>
    <automated>test -f .planning/quick/24-audit-recent-pseudo-label-and-training-d/24-AUDIT-REPORT.md && wc -l .planning/quick/24-audit-recent-pseudo-label-and-training-d/24-AUDIT-REPORT.md | awk '{if ($1 >= 80) print "PASS: "$1" lines"; else print "FAIL: only "$1" lines"}'</automated>
  </verify>
  <done>24-AUDIT-REPORT.md exists with prioritized findings covering all 21 source files, each finding has LOE and ROI ratings, and the report includes a priority matrix and recommended execution order. User can use this to instruct which improvements to pursue.</done>
</task>

</tasks>

<verification>
- Report file exists at expected path
- Report covers structural, interface, test coverage, and data flow dimensions
- Each finding has LOE and ROI ratings
- Priority matrix summarizes all findings
- No code changes were made (audit only)
</verification>

<success_criteria>
- Complete audit report covering src/aquapose/training/ (21 files, ~8400 LOC)
- Every finding has specific file/function references, LOE estimate, and ROI assessment
- User can read the priority matrix and immediately decide which improvements to pursue
</success_criteria>

<output>
After completion, create `.planning/quick/24-audit-recent-pseudo-label-and-training-d/24-SUMMARY.md`
</output>
