Review code for elegance at the specified scope. The user may pass a scope as an argument: `conversation` (default), `commits`, or `repo`.

Scope argument: $ARGUMENTS

If no scope is provided or the argument is empty, default to `conversation`.

---

# Scope: conversation

Review the most recent implementation from this conversation in **high detail**. Apply every check below thoroughly, reading the code line by line. Suggest concrete revisions and implement them.

## How to identify what to review

Look at the files you created or edited most recently in this conversation. That's the scope.

---

# Scope: commits

Review the last significant task reflected in recent git history, in **medium detail**. Focus on the checks most likely to catch real problems — skip nitpicks.

## How to identify what to review

1. Run `git log --oneline -20` to find the most recent cluster of related commits.
2. Run `git diff <before>..HEAD` (or appropriate range) to get the actual changes.
3. Review those diffs — not the full files, just the changes and enough surrounding context to evaluate them.

---

# Scope: repo

Review the entire `src/` directory (or equivalent project root) at **low detail** for big-picture pain points. Don't go line by line — look for structural and architectural issues.

## How to identify what to review

1. Map out the directory structure and module boundaries.
2. Skim key files: entry points, core modules, shared utilities, type definitions.
3. Focus on the forest, not the trees.

## What to look for at repo scope

Instead of the full checklist below, focus only on:
- Duplicated logic across modules that should be consolidated
- Inconsistent patterns (error handling, naming, data flow) across different parts of the codebase
- Modules or files that have become catch-alls and should be split
- Missing shared abstractions that multiple modules are reinventing independently
- Dependency issues: circular imports, modules that know too much about each other
- Dead code, orphaned utilities, or unused exports

Skip to the **Output** section after this — the detailed checklist below is not needed at repo scope.

---

# Detailed Checklist (conversation and commits scopes only)

## 1. Reuse Check

- Did you reimplement something that already exists in the codebase? Search for existing utilities, helpers, constants, and shared data structures that overlap with what you wrote.
- Did you reimplement standard library functionality? (e.g., hand-rolled parsing, manual iteration where builtins suffice)
- Did you introduce a new dependency when an existing one already covers the need?

## 2. Structural Review

- **The "and" test:** If describing any function requires "and," it should probably be split.
- Are there unnecessary abstraction layers that add complexity without value?
- Are you solving the same kind of problem differently than elsewhere in the codebase? Match existing patterns.
- Are there magic numbers, hardcoded values, or string literals that should be named constants or config?
- Is there overly defensive code (excessive null checks, broad try/catches) that masks real bugs instead of handling genuine edge cases?

## 3. Interface & API Design

- Do function names clearly communicate behavior without requiring the reader to inspect the implementation?
- Are return types consistent? (Don't mix null returns, thrown exceptions, and sentinel values for the same kind of failure.)
- Do parameter signatures match conventions used elsewhere in the codebase?
- Are callers forced to know implementation details they shouldn't? (Leaky abstractions.)

## 4. Data Flow & State

- Is there unnecessary mutation where a transformation pipeline would be clearer?
- Is data being threaded through multiple layers just to reach one consumer? This suggests a structural problem.
- Is state held in a broader scope than necessary?
- Check variable usage: eliminate unnecessary temporaries, but also break up dense one-liners that sacrifice readability.

## 5. Maintainability & Future-Proofing

- Will this obviously break when requirements generalize? (e.g., hardcoding two cases when there will clearly be N.)
- Are non-obvious decisions explained with "why" comments?
- Do error messages contain enough context for someone to debug the issue months from now?
- Does this change introduce test gaps?

## 6. Codebase Coherence

- Does this follow the project's established error handling strategy?
- Does this follow the project's file organization and module boundary conventions?
- Is the style (naming, formatting, idioms) consistent with the surrounding code?

## 7. Extraction Opportunities

- Are there pieces of this implementation that would be useful elsewhere? Extract them as reusable functions or utilities.
- If you wrote a workaround for a recurring problem, should you instead build a proper reusable solution?

---

# Output

After reviewing, do one of:
- **No changes needed:** Briefly state why the code is already clean.
- **Minor improvements:** List and implement them directly.
- **Significant rework warranted:** Explain the reasoning, then implement the revised version.

For `repo` scope, present findings as a prioritized list of the most impactful improvements, with specific file/module references. Don't try to fix everything — identify the top 3-5 things that would most improve the codebase.
