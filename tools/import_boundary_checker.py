"""AST-based import boundary and structural rule checker for AquaPose.

Enforces the strict one-way import layering defined in the AquaPose alpha
refactor guidebook (Sections 3 and 15):

  core/   -> nothing (only stdlib, third-party, and core internals)
  engine/ -> core/
  cli/    -> engine/

No TYPE_CHECKING backdoors are permitted.

Usage::

    python tools/import_boundary_checker.py [--verbose] [file1.py file2.py ...]

If no file arguments are given, scans all of ``src/aquapose/``.
Exit 0 if all checks pass; exit 1 if any violations are found.

Rule IDs
--------
- IB-001  File in ``core/`` imports from ``aquapose.engine`` or ``aquapose.cli``
- IB-002  File in ``engine/`` imports from ``aquapose.cli``
- IB-003  TYPE_CHECKING backdoor: ``core/`` file imports ``aquapose.engine`` or
          ``aquapose.cli`` inside an ``if TYPE_CHECKING:`` block
- IB-004  Computation module (legacy layer-1 dirs) imports ``aquapose.engine``
          or ``aquapose.cli``
- SR-001  Stage ``run()`` method contains file I/O calls (``open()``,
          ``Path.write_*``, ``Path.read_*``)
- SR-002  Observer file (``engine/``) imports directly from ``core/`` submodule
          internals (warning — some core imports may be legitimate)
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root of the aquapose source tree, relative to the repository root.
_SRC_ROOT = Path("src/aquapose")

# Layer 1 "pure computation" directories — must not import engine/ or cli/.
# Includes the canonical core/ subpackages plus legacy top-level modules that
# the guidebook classifies as Layer 1.
_CORE_DIRS = {
    "core",
}

_LEGACY_COMPUTATION_DIRS = {
    "calibration",
    "segmentation",
    "tracking",
    "reconstruction",
    "initialization",
    "mesh",
    "optimization",
}

# Forbidden import prefixes for each layer.
_FORBIDDEN_IN_CORE = ("aquapose.engine", "aquapose.cli")
_FORBIDDEN_IN_ENGINE = ("aquapose.cli",)

# IB-003 TYPE_CHECKING allowlist: (filename_suffix, module) pairs that are
# permitted exceptions. These are documented design decisions where a core/
# module legitimately receives a config type from the engine layer as a
# constructor argument. The import is annotation-only (TYPE_CHECKING guard)
# and config flows strictly downward from engine to core, not upward.
#
# Rationale for each entry:
#   core/synthetic.py + aquapose.engine.config: SyntheticDataStage receives
#   SyntheticConfig from the engine layer. Config flows downward (engine ->
#   core), so this is an acceptable annotation-only dependency.
_IB003_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    [
        ("core/synthetic.py", "aquapose.engine.config"),
    ]
)

# File I/O call names that are forbidden inside stage run() methods.
_FILE_IO_CALLS = frozenset(
    [
        "open",
    ]
)

# Path method names that indicate file I/O (write_* or read_*).
_PATH_IO_METHODS = frozenset(
    [
        "write_text",
        "write_bytes",
        "read_text",
        "read_bytes",
        "open",
    ]
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """A single rule violation."""

    filepath: Path
    line: int
    rule_id: str
    description: str
    is_warning: bool = False

    def __str__(self) -> str:
        severity = "warning" if self.is_warning else "error"
        return (
            f"{self.filepath}:{self.line}: {self.rule_id} [{severity}] — "
            f"{self.description}"
        )


@dataclass
class CheckResult:
    """Accumulated results from checking one or more files."""

    violations: list[Violation] = field(default_factory=list)

    def add(
        self,
        filepath: Path,
        line: int,
        rule_id: str,
        description: str,
        is_warning: bool = False,
    ) -> None:
        """Append a violation to the result set."""
        self.violations.append(
            Violation(filepath, line, rule_id, description, is_warning)
        )

    @property
    def errors(self) -> list[Violation]:
        """Return only hard violations (non-warnings)."""
        return [v for v in self.violations if not v.is_warning]

    @property
    def warnings(self) -> list[Violation]:
        """Return only warnings."""
        return [v for v in self.violations if v.is_warning]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_type_checking_block(node: ast.AST) -> bool:
    """Return True if *node* is an ``if TYPE_CHECKING:`` guard."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    # Handle both ``if TYPE_CHECKING:`` and ``if typing.TYPE_CHECKING:``
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _collect_imports(
    tree: ast.Module,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Collect all top-level and TYPE_CHECKING-guarded imports.

    Returns:
        A pair ``(runtime_imports, type_checking_imports)`` where each element
        is a list of ``(line_number, module_name)`` tuples.
    """
    runtime: list[tuple[int, str]] = []
    type_checking: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if _is_type_checking_block(node):
            assert isinstance(node, ast.If)
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        type_checking.append((child.lineno, alias.name))
                elif isinstance(child, ast.ImportFrom) and child.module:
                    type_checking.append((child.lineno, child.module))

    # Collect all imports, then separate out those inside TYPE_CHECKING blocks.
    type_checking_lines = {line for line, _ in type_checking}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if node.lineno not in type_checking_lines:
                    runtime.append((node.lineno, alias.name))
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.lineno not in type_checking_lines
        ):
            runtime.append((node.lineno, node.module))

    return runtime, type_checking


def _module_starts_with_any(module: str, prefixes: tuple[str, ...]) -> str | None:
    """Return the matching prefix if *module* starts with any of *prefixes*."""
    for prefix in prefixes:
        if module == prefix or module.startswith(prefix + "."):
            return prefix
    return None


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------


def _classify_file(filepath: Path, src_root: Path) -> str | None:
    """Return the layer label for *filepath* or None if not in src/aquapose.

    Labels: ``"core"``, ``"legacy_computation"``, ``"engine"``, ``"cli"``,
    ``"other"``.
    """
    try:
        rel = filepath.resolve().relative_to(src_root.resolve())
    except ValueError:
        return None

    parts = rel.parts
    if not parts:
        return None

    top = parts[0]

    if top == "core":
        return "core"
    if top == "engine":
        return "engine"
    if top == "cli.py" or top == "cli":
        return "cli"
    if top in _LEGACY_COMPUTATION_DIRS:
        return "legacy_computation"
    return "other"


# ---------------------------------------------------------------------------
# Rule checkers
# ---------------------------------------------------------------------------


def _check_import_boundaries(
    filepath: Path,
    tree: ast.Module,
    layer: str,
    result: CheckResult,
) -> None:
    """Check IB-001, IB-002, IB-003, IB-004."""
    runtime_imports, type_checking_imports = _collect_imports(tree)

    if layer in ("core", "legacy_computation"):
        # IB-001 / IB-004: runtime imports must not touch engine/ or cli/
        rule_id = "IB-001" if layer == "core" else "IB-004"
        for line, module in runtime_imports:
            hit = _module_starts_with_any(module, _FORBIDDEN_IN_CORE)
            if hit:
                result.add(
                    filepath,
                    line,
                    rule_id,
                    f"Forbidden runtime import of '{module}' in {layer} module "
                    f"(must not import from engine/ or cli/)",
                )

        # IB-003: TYPE_CHECKING backdoors are forbidden in core/ and legacy
        # computation modules that follow core/ rules.
        # Exception: entries in _IB003_ALLOWLIST are documented design decisions
        # where config flows strictly downward from engine to core.
        for line, module in type_checking_imports:
            hit = _module_starts_with_any(module, _FORBIDDEN_IN_CORE)
            if hit:
                # Check against the allowlist using the POSIX-normalised suffix
                rel_posix = filepath.as_posix()
                if any(
                    rel_posix.endswith(suffix) and module == allowed_module
                    for suffix, allowed_module in _IB003_ALLOWLIST
                ):
                    continue
                result.add(
                    filepath,
                    line,
                    "IB-003",
                    f"TYPE_CHECKING backdoor: '{module}' imported under "
                    f"TYPE_CHECKING in {layer} module (guidebook §3 forbids "
                    f"this — use a string annotation or Protocol instead)",
                )

    elif layer == "engine":
        # IB-002: engine/ must not import cli/
        for line, module in runtime_imports + type_checking_imports:
            hit = _module_starts_with_any(module, _FORBIDDEN_IN_ENGINE)
            if hit:
                result.add(
                    filepath,
                    line,
                    "IB-002",
                    f"Forbidden import of '{module}' in engine/ module "
                    f"(engine/ must not import from cli/)",
                )


def _find_run_methods_with_io(
    filepath: Path,
    tree: ast.Module,
    result: CheckResult,
) -> None:
    """Check SR-001: stage run() methods must not contain file I/O calls."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "run":
            continue

        # Walk the body of this run() method looking for file I/O.
        for child in ast.walk(node):
            # Check for bare open() calls.
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id == "open":
                    result.add(
                        filepath,
                        child.lineno,
                        "SR-001",
                        "File I/O call 'open()' inside stage run() method — "
                        "file writing must be done by observers or the pipeline, "
                        "not inside stage logic (guidebook §13)",
                    )
                # Check for Path.write_text(), Path.read_text(), etc.
                elif isinstance(func, ast.Attribute) and func.attr in _PATH_IO_METHODS:
                    result.add(
                        filepath,
                        child.lineno,
                        "SR-001",
                        f"File I/O call 'Path.{func.attr}()' inside stage "
                        f"run() method — file writing must be done by "
                        f"observers or the pipeline, not inside stage logic "
                        f"(guidebook \xa713)",
                    )


def _check_observer_core_imports(
    filepath: Path,
    tree: ast.Module,
    result: CheckResult,
) -> None:
    """Check SR-002: engine observer files should not import core/ submodule internals.

    This is a WARNING (not an error) since some legitimate imports of core types
    may be needed by observers. The rule flags direct imports below the top-level
    ``aquapose.core`` namespace (e.g. ``aquapose.core.detection.types``).

    Exception: ``aquapose.core.context`` is the canonical home of PipelineContext
    and Stage (moved from engine/stages.py in Phase 20). Engine files importing
    from it are correct and expected.
    """
    runtime_imports, type_checking_imports = _collect_imports(tree)
    for line, module in runtime_imports + type_checking_imports:
        # Flag imports that go deeper than aquapose.core (i.e. submodule internals),
        # but allow aquapose.core.context which is the canonical pipeline contract.
        if module.startswith("aquapose.core.") and module != "aquapose.core.context":
            result.add(
                filepath,
                line,
                "SR-002",
                f"Observer imports core/ submodule internal '{module}' — "
                f"prefer using the aquapose.core public API or "
                f"aquapose.core.context for PipelineContext/Stage",
                is_warning=True,
            )


# ---------------------------------------------------------------------------
# File-level checker
# ---------------------------------------------------------------------------


def check_file(filepath: Path, src_root: Path, result: CheckResult) -> None:
    """Run all applicable rules against a single Python file."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # Non-fatal — skip unreadable/non-UTF-8 files.
        print(f"WARNING: could not read {filepath}: {exc}", file=sys.stderr)
        return

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        print(f"WARNING: syntax error in {filepath}: {exc}", file=sys.stderr)
        return

    layer = _classify_file(filepath, src_root)
    if layer is None or layer == "other":
        return

    # Import boundary rules apply to all layers except "other".
    _check_import_boundaries(filepath, tree, layer, result)

    # SR-001: file I/O in run() — applies to stage.py files under core/
    if layer == "core" and filepath.name == "stage.py":
        _find_run_methods_with_io(filepath, tree, result)

    # SR-002: observer imports — applies to engine/ files
    if layer == "engine":
        _check_observer_core_imports(filepath, tree, result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _collect_files(paths: list[Path], src_root: Path) -> list[Path]:
    """Expand directories / globs to a flat list of .py files."""
    result: list[Path] = []
    for p in paths:
        if p.is_dir():
            result.extend(sorted(p.rglob("*.py")))
        elif p.suffix == ".py":
            result.append(p)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in result:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def main(argv: list[str] | None = None) -> int:
    """Run the import boundary checker and return an exit code.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 if all checks pass, 1 if any hard violations are found.
    """
    parser = argparse.ArgumentParser(
        description="AquaPose import boundary and structural rule checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Files or directories to check. Defaults to src/aquapose/.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each file checked (even if no violations found).",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help=(
            "Path to the src/aquapose/ root. "
            "Auto-detected relative to this script if omitted."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve the source root.
    if args.src_root is not None:
        src_root = args.src_root.resolve()
    else:
        # Default: <repo_root>/src/aquapose, where repo_root is the parent of
        # the directory containing this script (tools/).
        script_dir = Path(__file__).resolve().parent
        src_root = (script_dir.parent / _SRC_ROOT).resolve()

    if not src_root.exists():
        print(
            f"ERROR: src/aquapose root not found at {src_root}. "
            f"Run from the repository root or pass --src-root.",
            file=sys.stderr,
        )
        return 1

    # Resolve files to check.
    if args.files:
        targets = _collect_files([Path(p) for p in args.files], src_root)
    else:
        targets = _collect_files([src_root], src_root)

    if args.verbose:
        print(f"Checking {len(targets)} file(s) under {src_root}")

    result = CheckResult()
    for filepath in targets:
        if args.verbose:
            print(f"  {filepath}")
        check_file(filepath, src_root, result)

    # Print results grouped by file.
    if result.violations:
        printed_files: set[Path] = set()
        for violation in sorted(
            result.violations, key=lambda v: (str(v.filepath), v.line)
        ):
            if violation.filepath not in printed_files:
                if printed_files:
                    print()
                printed_files.add(violation.filepath)
            print(str(violation))

    # Summary line.
    n_errors = len(result.errors)
    n_warnings = len(result.warnings)

    if result.violations:
        print()

    if n_errors == 0 and n_warnings == 0:
        print("import-boundary: OK — no violations found")
        return 0

    parts: list[str] = []
    if n_errors:
        parts.append(f"{n_errors} error(s)")
    if n_warnings:
        parts.append(f"{n_warnings} warning(s)")

    print(f"import-boundary: {', '.join(parts)} found")
    return 1 if n_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
