"""Tests for the training CLI group and subcommands."""

from __future__ import annotations

import ast
from pathlib import Path

from click.testing import CliRunner

from aquapose.cli import cli


def test_train_help_lists_yolo_obb() -> None:
    """aquapose train --help should list the yolo-obb subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0, result.output
    assert "yolo-obb" in result.output


def test_train_help_does_not_list_removed_commands() -> None:
    """aquapose train --help must not list the removed unet subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0, result.output
    assert "unet" not in result.output


def test_train_yolo_obb_help_shows_expected_flags() -> None:
    """aquapose train yolo-obb --help should show all required flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "yolo-obb", "--help"])
    assert result.exit_code == 0, result.output
    expected_flags = [
        "--data-dir",
        "--tag",
        "--epochs",
        "--device",
        "--val-split",
        "--batch-size",
        "--imgsz",
        "--model",
        "--weights",
    ]
    for flag in expected_flags:
        assert flag in result.output, f"Expected flag {flag!r} not found in help output"
    assert "--output-dir" not in result.output, "--output-dir should be removed"
    assert "--config" not in result.output, "--config should be removed"


def _get_training_module_imports(filepath: Path) -> list[tuple[int, str]]:
    """Extract all import module names from a Python file via AST.

    Args:
        filepath: Path to the Python source file.

    Returns:
        List of (line_number, module_name) tuples.
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def test_training_modules_do_not_import_engine() -> None:
    """training/ modules must not import from aquapose.engine or aquapose.cli."""
    training_dir = Path("src/aquapose/training")
    forbidden = ("aquapose.engine", "aquapose.cli")
    # cli_utils is allowed -- it's a CLI utility, not engine logic
    allowed = ("aquapose.cli_utils",)
    violations: list[str] = []

    for py_file in sorted(training_dir.glob("*.py")):
        imports = _get_training_module_imports(py_file)
        for line, module in imports:
            for prefix in forbidden:
                if module == prefix or module.startswith(prefix + "."):
                    violations.append(
                        f"{py_file}:{line}: forbidden import of {module!r}"
                    )
            # But allow cli_utils
            for allow in allowed:
                if module == allow or module.startswith(allow + "."):
                    # Remove from violations if it was added
                    key = f"{py_file}:{line}: forbidden import of {module!r}"
                    if key in violations:
                        violations.remove(key)

    assert not violations, "Import boundary violations found:\n" + "\n".join(violations)


def test_train_help_lists_seg_and_pose() -> None:
    """aquapose train --help should list the seg and pose subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0, result.output
    assert "seg" in result.output
    assert "pose" in result.output


def test_train_seg_help_shows_expected_flags() -> None:
    """aquapose train seg --help should show all required flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "seg", "--help"])
    assert result.exit_code == 0, result.output
    expected_flags = [
        "--data-dir",
        "--tag",
        "--epochs",
        "--batch-size",
        "--device",
        "--val-split",
        "--imgsz",
        "--model",
        "--weights",
    ]
    for flag in expected_flags:
        assert flag in result.output, (
            f"Expected flag {flag!r} not found in seg help output"
        )
    assert "--output-dir" not in result.output, "--output-dir should be removed"
    assert "--config" not in result.output, "--config should be removed"


def test_train_pose_help_shows_expected_flags() -> None:
    """aquapose train pose --help should show all required flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "pose", "--help"])
    assert result.exit_code == 0, result.output
    expected_flags = [
        "--data-dir",
        "--tag",
        "--epochs",
        "--batch-size",
        "--device",
        "--val-split",
        "--imgsz",
        "--model",
        "--weights",
    ]
    for flag in expected_flags:
        assert flag in result.output, (
            f"Expected flag {flag!r} not found in pose help output"
        )
    assert "--output-dir" not in result.output, "--output-dir should be removed"
    assert "--config" not in result.output, "--config should be removed"


def test_train_help_lists_compare() -> None:
    """aquapose train --help should list the compare subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0, result.output
    assert "compare" in result.output


def test_compare_help_shows_expected_flags() -> None:
    """aquapose train compare --help should show --model-type, --csv."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "compare", "--help"])
    assert result.exit_code == 0, result.output
    expected_flags = ["--model-type", "--csv"]
    for flag in expected_flags:
        assert flag in result.output, (
            f"Expected flag {flag!r} not found in compare help"
        )
    assert "--config" not in result.output, "--config should be removed"
