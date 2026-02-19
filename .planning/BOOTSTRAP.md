## Bootstrap Scaffolding

These are the remaining scaffolding steps after the user has run `bootstrap.py` and `/gsd:new-project`. Perform them as part of the first phase of work. Delete this file after completing all steps.

### Context

By this point, the following already exist:

- `src/<package>/__init__.py` and `py.typed` (created by `bootstrap.py`)
- `tests/unit/test_smoke.py` (from the template, package name already replaced)
- `docs/conf.py` and `docs/contributing.md` (from the template, author/URLs already replaced)
- `.planning/PROJECT.md` (created by `/gsd:new-project`)
- `.claude/rules/` with code-style, source-code, and workflow rules
- `.secrets.baseline` (from the template)
- `README.md` (from `README_TEMPLATE.md`, with TODOs to fill in)

`bootstrap.py` has already replaced the package name, author name, and GitHub username across all template files. Version is managed via `importlib.metadata` (single source of truth in `pyproject.toml`).

### 1. Create the source package structure

Expand `src/<package>/` with subpackages as the project's architecture requires. Follow the conventions in `.claude/rules/source-code.md` for `__init__.py` files and docstrings.

Common subpackages (recommended but not required):

| Directory | Purpose |
|-----------|---------|
| `core/` | Core domain logic |
| `config/` | Configuration, shared types, dataclasses |
| `io/` | File I/O, serialization, loaders |
| `utils/` | General-purpose helpers |

### 2. Create the test directories

Add test directories alongside the existing `tests/unit/`:

```
tests/
├── unit/           # One test file per source module
├── integration/    # Cross-module tests
└── e2e/            # End-to-end tests
```

Create whichever test directories the project actually needs.

### 3. Create docs scaffolding

`docs/conf.py` and `docs/contributing.md` already exist. Add:

- `docs/index.md` — landing page with sphinx-design grid cards linking to API reference, contributing guide, and any other sections. Include a toctree. Follow the MyST markdown + Furo grid-card pattern.
- `docs/api/index.rst` — autodoc toctree, one `.rst` file per source subpackage.
- `docs/Makefile` and `docs/make.bat` (standard Sphinx quickstart output).

### 4. Create CODE_OF_CONDUCT.md

Standard Contributor Covenant v2.1. Update the enforcement contact URL to point to the project's GitHub issues.

### 5. Update pyproject.toml metadata

URLs and author are already set by `bootstrap.py`. Update the remaining fields:

- `description`, `keywords`, `classifiers`
- `dependencies` (actual project deps)
- Uncomment `[project.scripts]` if the project has a CLI

### 6. Write the README

Fill in the TODO sections in `README.md` (description, quick start example, docs link).

### 7. Regenerate secrets baseline

The template ships a `.secrets.baseline` but source files have changed. Regenerate it:

```bash
hatch run pre-commit run detect-secrets --all-files
```

### 8. Create CLAUDE.md

Create this last — after the source structure, docs, and pyproject.toml are finalized so the content is accurate rather than speculative.

Keep it under 100 lines. This is injected into every conversation's system prompt.

Contents:

- **Project**: What it does (1-2 sentences)
- **Quick start**: How to set up the dev environment (`hatch env create`, etc.)
- **Commands**: Build, test, lint, typecheck, run (copy-pasteable)
- **Architecture**: Key directories and entry points
- **Domain conventions**: Project-specific terminology or rules (if any)

Do NOT duplicate content already in `.claude/rules/` (code style, workflow). Do NOT hardcode environment details (OS, shell, paths) — Claude Code detects these automatically.

### 9. Verify

```bash
hatch run test                        # tests pass
hatch run lint                        # lint passes
hatch run typecheck                   # type check passes
hatch run pre-commit run --all-files  # hooks pass
```
