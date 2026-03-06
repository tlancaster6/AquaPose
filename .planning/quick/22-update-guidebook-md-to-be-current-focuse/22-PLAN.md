---
phase: 22-update-guidebook
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [.planning/GUIDEBOOK.md]
autonomous: true
requirements: [GUIDE-01]

must_haves:
  truths:
    - "Every file path mentioned in GUIDEBOOK.md exists in the actual codebase"
    - "No (v2.2) or other version-future annotations remain — all shipped features described as current"
    - "Milestone history covers v1.0 through v3.5 (completed) and v3.6 (active)"
    - "Observer list matches actual engine/ directory contents"
    - "Source layout matches actual src/aquapose/ directory tree"
    - "CLI section does not enumerate specific subcommands that could go stale"
    - "Artifact layout section is accurate or abstracted to stable patterns"
  artifacts:
    - path: ".planning/GUIDEBOOK.md"
      provides: "Authoritative project guidebook for discuss-phase agents"
      contains: "## 1. Purpose"
  key_links:
    - from: ".planning/GUIDEBOOK.md"
      to: "src/aquapose/"
      via: "Source layout section references actual directory structure"
      pattern: "src/aquapose/"
---

<objective>
Update GUIDEBOOK.md to be current, focused, and trustworthy by removing stale content, updating inaccurate sections, and abstracting volatile details to a level where they stay true over time.

Purpose: GUIDEBOOK.md is read by every gsd:discuss-phase agent as authoritative big-picture context. Stale content (wrong file listings, missing milestones, shipped-but-marked-as-planned features) erodes trust in the entire document.

Output: An updated .planning/GUIDEBOOK.md where every claim is verifiable against the current codebase.
</objective>

<execution_context>
@/home/tlancaster6/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlancaster6/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/GUIDEBOOK.md
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Audit GUIDEBOOK.md against codebase and collect corrections</name>
  <files>.planning/GUIDEBOOK.md</files>
  <action>
Systematically audit every factual claim in GUIDEBOOK.md against the actual codebase. For each section, run directory listings and grep checks to identify:

1. **Section 4 (Source Layout):** Run `find src/aquapose/ -type d` and `ls src/aquapose/engine/` to get the real directory tree and file list. Note: `visualization/` does not exist, `evaluation/` exists, `orchestrator.py` exists in engine, `hdf5_observer.py`/`overlay_observer.py`/`tracklet_trail_observer.py`/`animation_observer.py` do NOT exist in engine.

2. **Section 6 (Stage descriptions):** Find all `(v2.2)` annotations — these features have shipped. Check whether Detection section still describes YOLO-OBB as future; it should be described as current. Check Midline backend descriptions for accuracy.

3. **Section 10 (Observer list):** Compare listed observers against actual `ls src/aquapose/engine/*.py`. Remove observers that were migrated to viz CLI.

4. **Section 11 (Configuration):** Remove `(v2.2)` planned annotations for features that shipped.

5. **Section 13 (Artifacts):** Check if the artifact layout reflects the actual chunk-based output structure by examining a real run directory or config defaults.

6. **Section 14 (CLI):** Check actual CLI subcommands via grep of cli.py. The current approach lists specific commands which go stale. Abstract to describe the CLI philosophy and major command groups without enumerating every subcommand.

7. **Section 15 (Milestone History):** v3.0 is described as "(current)" but shipped long ago. Missing milestones: v3.1 Reconstruction (shipped 2026-03-03), v3.2 Evaluation Ecosystem (shipped), v3.3 Chunk Mode (shipped), v3.4 Performance Optimization (shipped), v3.5 Pseudo-Labeling (shipped 2026-03-06), v3.6 Model Iteration and QA (active).

Produce a structured list of every correction needed, organized by section number.
  </action>
  <verify>
    <automated>echo "Audit task is preparatory — verify by confirming correction list is complete"</automated>
  </verify>
  <done>A comprehensive list of corrections exists covering all 7 areas above</done>
</task>

<task type="auto">
  <name>Task 2: Apply all corrections to GUIDEBOOK.md</name>
  <files>.planning/GUIDEBOOK.md</files>
  <action>
Apply all corrections from the audit. The guiding principle: if content goes stale quickly, either remove it or abstract it to a level where it stays true. Specific changes:

**Section 4 (Source Layout):**
- Rebuild the directory tree from actual `find` output
- For engine/, list only the key architectural files (pipeline.py, orchestrator.py, events.py, observers.py, config.py) and describe observer files as "observer implementations" without listing each one by name
- Add `evaluation/` directory
- Remove `visualization/` directory (does not exist)
- Add `cli_utils.py` and `logging.py` at the top level
- Do NOT list individual .py files in leaf directories unless they are architecturally significant — directories are stable, files within them change

**Section 6 (Stage descriptions):**
- Remove ALL `(v2.2)` annotations — these features have shipped
- Detection: describe YOLO-OBB as the current detection approach (not future)
- Midline: ensure both backends (segment_then_extract, direct_pose) are described as current
- Remove language like "will be" or "planned" for shipped features
- Keep the stage I/O contracts and data flow table — these are stable

**Section 10 (Observer Protocol):**
- Remove the specific observer file listing entirely
- Keep the observer protocol description (subscribe to event types, synchronous, passive)
- Add a single sentence: "Observer implementations live in engine/ — see the directory for current observers"
- This way the section never goes stale

**Section 11 (Configuration):**
- Remove the "Planned additions (v2.2)" paragraph entirely — those features either shipped or are irrelevant
- Keep everything else (frozen dataclasses, loading precedence, execution modes, run identity)

**Section 13 (Artifacts):**
- Keep the artifact principles (first-class, structured, run ID, reproducible)
- Keep the output location description
- Replace the specific directory tree with a note that artifacts are organized by stage and observer, with the exact structure determined by config. This avoids staleness as new stages/observers are added.

**Section 14 (CLI):**
- Replace the current content with a description of the CLI philosophy: Click-based, project-aware (`--project`), with command groups for pipeline execution (`run`), project initialization (`init`), model training (`train`), data preparation (`data`, `prep`), evaluation (`eval`), tuning (`tune`), visualization (`viz`), and pseudo-labeling (`pseudo-label`)
- Do NOT list exact flags or subcommand details — just the top-level groups and their purpose
- Keep the architectural rule: CLI is a thin wrapper, no script may bypass stage sequencing
- Keep the training import boundary rule

**Section 15 (Milestone History):**
- Change v3.0 from "(current)" to "(shipped 2026-03-01)" with a brief description
- Add v3.1 Reconstruction (shipped 2026-03-03): Reconstruction tuning, association tuning with corrected refractive LUTs, dead code cleanup
- Add v3.2 Evaluation Ecosystem (shipped): CLI eval/tune commands, OKS metrics, reconstruction quality metrics
- Add v3.3 Chunk Mode (shipped): Chunk-based pipeline execution for long videos, chunk-aware diagnostics, viz CLI migration
- Add v3.4 Performance Optimization (shipped): Vectorized association scoring, vectorized DLT reconstruction
- Add v3.5 Pseudo-Labeling (shipped 2026-03-06): Pseudo-label generation from pipeline output, elastic augmentation for pose model training, curvature-stratified evaluation
- Add v3.6 Model Iteration and QA (active): Current milestone

Keep descriptions to 1-2 lines each, matching the style of existing entries.
  </action>
  <verify>
    <automated>cd /home/tlancaster6/Projects/AquaPose && python -c "
import os, re, sys

guidebook = open('.planning/GUIDEBOOK.md').read()

errors = []

# Check no (v2.2) annotations remain
if '(v2.2)' in guidebook:
    errors.append('Still contains (v2.2) annotations')

# Check visualization/ not mentioned as a directory
if 'visualization/' in guidebook and 'visualization' not in guidebook.split('evaluation'):
    pass  # might be in a different context

# Check evaluation/ is mentioned
if 'evaluation/' not in guidebook:
    errors.append('Missing evaluation/ directory')

# Check v3.1 through v3.5 milestones exist
for v in ['v3.1', 'v3.2', 'v3.3', 'v3.4', 'v3.5', 'v3.6']:
    if v not in guidebook:
        errors.append(f'Missing milestone {v}')

# Check no '(current)' on v3.0
if 'v3.0' in guidebook and '(current)' in guidebook.split('v3.0')[1][:50]:
    errors.append('v3.0 still marked as (current)')

# Check key engine files mentioned or abstracted
if 'orchestrator' not in guidebook.lower():
    errors.append('Missing orchestrator.py reference')

# Check stale observers not listed
for stale in ['hdf5_observer.py', 'overlay_observer.py', 'tracklet_trail_observer.py', 'animation_observer.py']:
    if stale in guidebook:
        errors.append(f'Still lists stale observer: {stale}')

# Check all real top-level dirs exist
real_dirs = ['calibration', 'core', 'engine', 'evaluation', 'io', 'synthetic', 'training']
for d in real_dirs:
    if d + '/' not in guidebook:
        errors.append(f'Missing directory: {d}/')

if errors:
    print('ERRORS:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('All checks passed')
"
    </automated>
  </verify>
  <done>
- No (v2.2) future annotations remain
- Source layout matches actual codebase directories
- No stale observer files listed by name
- Milestone history covers v1.0 through v3.6
- CLI section describes command groups without enumerating specific flags
- All architectural/principle sections preserved unchanged
  </done>
</task>

</tasks>

<verification>
1. Every directory mentioned in Section 4 exists: `find src/aquapose/ -type d -maxdepth 2`
2. No `(v2.2)` annotations: `grep -c "(v2.2)" .planning/GUIDEBOOK.md` returns 0
3. No stale observer files listed: `grep -c "hdf5_observer\|overlay_observer\|tracklet_trail_observer\|animation_observer" .planning/GUIDEBOOK.md` returns 0
4. Milestones v3.1-v3.6 present: `grep -c "v3\.[1-6]" .planning/GUIDEBOOK.md` returns at least 6
5. Stable sections unchanged: Sections 2, 3, 7, 8, 9, 12, 16 content preserved
</verification>

<success_criteria>
- GUIDEBOOK.md contains zero claims that contradict the current codebase
- All volatile content (specific file listings, version annotations, CLI flag details) is either removed or abstracted to stable descriptions
- Milestone history is complete through v3.6
- Architectural sections (2, 3, 7, 8, 9, 12, 16) are preserved — they are timeless
</success_criteria>

<output>
After completion, create `.planning/quick/22-update-guidebook-md-to-be-current-focuse/22-SUMMARY.md`
</output>
