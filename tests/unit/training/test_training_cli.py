"""Tests for the training CLI group and subcommands."""

from __future__ import annotations

import ast
from pathlib import Path

from click.testing import CliRunner

from aquapose.cli import cli


def test_train_help_lists_unet() -> None:
    """aquapose train --help should list the unet subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0, result.output
    assert "unet" in result.output


def test_train_help_lists_yolo_obb() -> None:
    """aquapose train --help should list the yolo-obb subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0, result.output
    assert "yolo-obb" in result.output


def test_train_help_lists_pose() -> None:
    """aquapose train --help should list the pose subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0, result.output
    assert "pose" in result.output


def test_train_unet_help_shows_expected_flags() -> None:
    """aquapose train unet --help should show all required flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "unet", "--help"])
    assert result.exit_code == 0, result.output
    expected_flags = [
        "--data-dir",
        "--output-dir",
        "--epochs",
        "--device",
        "--val-split",
        "--batch-size",
        "--lr",
        "--patience",
        "--num-workers",
    ]
    for flag in expected_flags:
        assert flag in result.output, f"Expected flag {flag!r} not found in help output"


def test_train_yolo_obb_help_shows_expected_flags() -> None:
    """aquapose train yolo-obb --help should show all required flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "yolo-obb", "--help"])
    assert result.exit_code == 0, result.output
    expected_flags = [
        "--data-dir",
        "--output-dir",
        "--epochs",
        "--device",
        "--val-split",
        "--batch-size",
        "--imgsz",
        "--model-size",
    ]
    for flag in expected_flags:
        assert flag in result.output, f"Expected flag {flag!r} not found in help output"


def test_train_pose_help_shows_expected_flags() -> None:
    """aquapose train pose --help should show all required flags."""
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "pose", "--help"])
    assert result.exit_code == 0, result.output
    expected_flags = [
        "--data-dir",
        "--output-dir",
        "--epochs",
        "--device",
        "--val-split",
        "--batch-size",
        "--lr",
        "--patience",
        "--num-workers",
        "--backbone-weights",
        "--unfreeze",
        "--n-keypoints",
    ]
    for flag in expected_flags:
        assert flag in result.output, f"Expected flag {flag!r} not found in help output"


def test_all_subcommands_share_consistent_flags() -> None:
    """All three subcommands must share --data-dir, --output-dir, --epochs, --device, --val-split."""
    runner = CliRunner()
    shared_flags = ["--data-dir", "--output-dir", "--epochs", "--device", "--val-split"]
    subcommands = ["unet", "yolo-obb", "pose"]

    for subcmd in subcommands:
        result = runner.invoke(cli, ["train", subcmd, "--help"])
        assert result.exit_code == 0, f"{subcmd} help failed: {result.output}"
        for flag in shared_flags:
            assert flag in result.output, (
                f"Shared flag {flag!r} missing from 'train {subcmd} --help'"
            )


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
    violations: list[str] = []

    for py_file in sorted(training_dir.glob("*.py")):
        imports = _get_training_module_imports(py_file)
        for line, module in imports:
            for prefix in forbidden:
                if module == prefix or module.startswith(prefix + "."):
                    violations.append(
                        f"{py_file}:{line}: forbidden import of {module!r}"
                    )

    assert not violations, "Import boundary violations found:\n" + "\n".join(violations)
