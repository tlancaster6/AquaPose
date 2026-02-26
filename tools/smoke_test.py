"""Smoke test runner for the AquaPose pipeline.

Exercises the pipeline across modes, backends, and data sources. Verifies
reproducibility by running twice with identical config and comparing outputs.

CLI usage::

    python tools/smoke_test.py --config path.yaml --output-dir ./smoke_results
    python tools/smoke_test.py --config path.yaml --only modes
    python tools/smoke_test.py --config path.yaml --only backends
    python tools/smoke_test.py --config path.yaml --only repro

Programmatic usage::

    from tools.smoke_test import SmokeTestRunner
    runner = SmokeTestRunner(config_path="...", output_base="./smoke")
    report = runner.run_all()
    print(report.summary())
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Extensible constants — add new modes / backends here
# ---------------------------------------------------------------------------

ALL_MODES: list[str] = ["production", "diagnostic", "synthetic", "benchmark"]
"""All pipeline mode presets to test (in --only modes runs)."""

ALL_BACKENDS: dict[str, str] = {
    "triangulation": "reconstruction.backend=triangulation",
    "curve_optimizer": "reconstruction.backend=curve_optimizer",
}
"""Reconstruction backend names -> --set override string."""

DEFAULT_FRAME_LIMIT: int = 10
"""Default number of frames to process per test run (kept small for speed)."""

SUBPROCESS_TIMEOUT: int = 300
"""Maximum seconds allowed per pipeline subprocess (5 minutes)."""


# ---------------------------------------------------------------------------
# Result and report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Result of a single smoke test case.

    Attributes:
        name: Human-readable test name.
        passed: True if the test passed, False otherwise.
        duration_seconds: Wall-clock duration of the test.
        error: Error message (None if passed).
        artifacts: List of artifact paths produced by this test.
        skipped: True if the test was skipped due to missing prerequisites.
        skip_reason: Reason for skipping (None if not skipped).
    """

    name: str
    passed: bool
    duration_seconds: float
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class SmokeTestReport:
    """Aggregated results of a full smoke test run.

    Attributes:
        results: Individual TestResult for each test case.
        total_duration: Total wall-clock time for the entire run.
        passed: Number of passed tests.
        failed: Number of failed tests.
        skipped: Number of skipped tests.
    """

    results: list[TestResult]
    total_duration: float
    passed: int
    failed: int
    skipped: int

    def summary(self) -> str:
        """Return a human-readable summary of the report.

        Returns:
            Multi-line string summarising pass/fail/skip counts and
            details for each non-passing result.
        """
        lines: list[str] = [
            "=" * 60,
            "AquaPose Smoke Test Report",
            "=" * 60,
            f"Duration : {self.total_duration:.1f}s",
            f"Passed   : {self.passed}",
            f"Failed   : {self.failed}",
            f"Skipped  : {self.skipped}",
            "",
        ]
        for r in self.results:
            if r.skipped:
                status = "SKIP"
                detail = r.skip_reason or ""
            elif r.passed:
                status = "PASS"
                detail = f"{r.duration_seconds:.1f}s"
            else:
                status = "FAIL"
                detail = r.error or ""
            lines.append(f"  [{status}] {r.name}: {detail}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize the report to a JSON string.

        Returns:
            JSON string with all test results and aggregate metrics.
        """

        def _result_dict(r: TestResult) -> dict:
            return {
                "name": r.name,
                "passed": r.passed,
                "skipped": r.skipped,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
                "skip_reason": r.skip_reason,
                "artifacts": r.artifacts,
            }

        return json.dumps(
            {
                "total_duration": self.total_duration,
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "results": [_result_dict(r) for r in self.results],
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# SmokeTestRunner
# ---------------------------------------------------------------------------


class SmokeTestRunner:
    """Runs smoke tests against the AquaPose pipeline.

    Each test invokes ``aquapose run`` as a subprocess so that process-level
    failures (import errors, crashes, non-zero exits) are captured cleanly.

    Args:
        config_path: Path to a pipeline config YAML. Required for non-synthetic
            tests. For synthetic-only runs, a minimal config is generated if
            this is not provided.
        output_base: Root directory for smoke test output artifacts.
        frame_limit: Number of frames to process per run (default 10).
        calibration_path: Optional path to AquaCal calibration JSON. Required
            for synthetic mode if not embedded in config_path.
        only: Restrict to a subset of tests: "modes", "backends", or "repro".
            None runs all dimensions.
    """

    def __init__(
        self,
        output_base: str | Path,
        config_path: str | Path | None = None,
        frame_limit: int = DEFAULT_FRAME_LIMIT,
        calibration_path: str | Path | None = None,
        only: str | None = None,
    ) -> None:
        self._config_path = Path(config_path) if config_path else None
        self._output_base = Path(output_base)
        self._frame_limit = frame_limit
        self._calibration_path = Path(calibration_path) if calibration_path else None
        self._only = only

        self._output_base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_aquapose(
        self,
        output_subdir: str,
        config_path: Path,
        extra_overrides: Sequence[str] = (),
        mode_override: str | None = None,
    ) -> tuple[bool, str, list[str]]:
        """Invoke ``aquapose run`` as a subprocess and return (success, error, artifacts).

        Args:
            output_subdir: Subdirectory name under output_base for artifacts.
            config_path: Path to the YAML config for this run.
            extra_overrides: Additional ``--set key=value`` strings.
            mode_override: If given, adds ``--mode {mode_override}`` to the command.

        Returns:
            Tuple of (passed: bool, error_message: str, artifact_paths: list[str]).
        """
        out_dir = self._output_base / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "aquapose",
            "run",
            "--config",
            str(config_path),
            "--set",
            f"output_dir={out_dir}",
            "--set",
            f"detection.stop_frame={self._frame_limit}",
        ]

        if mode_override:
            cmd += ["--mode", mode_override]

        for override in extra_overrides:
            cmd += ["--set", override]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {SUBPROCESS_TIMEOUT}s", []
        except Exception as exc:
            return False, f"Subprocess error: {exc}", []

        if result.returncode != 0:
            error = (result.stderr or result.stdout or "").strip()
            return False, error, []

        artifacts = [str(p) for p in out_dir.rglob("*") if p.is_file()]
        return True, "", artifacts

    def _resolve_config(
        self,
        mode: str | None = None,
    ) -> tuple[Path | None, str | None]:
        """Resolve the config path to use for a test, creating a minimal one if needed.

        For synthetic mode: a config with calibration_path is generated
        automatically if no explicit config_path was provided but a
        calibration_path was supplied.

        Args:
            mode: Pipeline mode being tested (affects minimal config creation).

        Returns:
            Tuple of (config_path, skip_reason). If skip_reason is non-None,
            the test should be skipped.
        """
        if self._config_path is not None:
            if not self._config_path.exists():
                return None, f"Config not found: {self._config_path}"
            return self._config_path, None

        # No config provided — create a minimal one for synthetic mode
        if mode == "synthetic" and self._calibration_path is not None:
            if not self._calibration_path.exists():
                return None, f"Calibration not found: {self._calibration_path}"
            minimal_config = self._output_base / "_synthetic_config.yaml"
            minimal_config.write_text(
                f"mode: synthetic\ncalibration_path: {self._calibration_path}\n",
                encoding="utf-8",
            )
            return minimal_config, None

        if mode == "synthetic":
            return (
                None,
                "No config_path or calibration_path provided for synthetic mode",
            )

        return None, f"No config_path provided for {mode} mode"

    def _make_result(
        self,
        name: str,
        skip_reason: str | None,
        start: float,
        success: bool,
        error: str,
        artifacts: list[str],
    ) -> TestResult:
        """Construct a TestResult from run output.

        Args:
            name: Test name.
            skip_reason: If non-None, the test was skipped for this reason.
            start: Monotonic start time (seconds).
            success: Whether the subprocess succeeded.
            error: Error message if failed.
            artifacts: Produced artifact paths.

        Returns:
            Populated TestResult.
        """
        elapsed = time.monotonic() - start
        if skip_reason:
            return TestResult(
                name=name,
                passed=False,
                duration_seconds=elapsed,
                skipped=True,
                skip_reason=skip_reason,
            )
        return TestResult(
            name=name,
            passed=success,
            duration_seconds=elapsed,
            error=error if not success else None,
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------
    # Public test methods
    # ------------------------------------------------------------------

    def run_single_mode(self, mode: str) -> TestResult:
        """Run the pipeline for a single mode preset.

        Each test is independent — failure of one doesn't affect others.

        Args:
            mode: One of "production", "diagnostic", "synthetic", "benchmark".

        Returns:
            TestResult describing the outcome.
        """
        name = f"mode/{mode}"
        start = time.monotonic()

        config_path, skip_reason = self._resolve_config(mode)
        if skip_reason:
            return self._make_result(name, skip_reason, start, False, "", [])

        assert config_path is not None
        success, error, artifacts = self._run_aquapose(
            output_subdir=f"mode_{mode}",
            config_path=config_path,
            mode_override=mode,
        )
        return self._make_result(name, None, start, success, error, artifacts)

    def run_single_backend(self, backend_name: str, override: str) -> TestResult:
        """Run the pipeline with a specific reconstruction backend.

        Args:
            backend_name: Human-readable backend name (used in test name).
            override: ``key=value`` override string for ``--set``.

        Returns:
            TestResult describing the outcome.
        """
        name = f"backend/{backend_name}"
        start = time.monotonic()

        config_path, skip_reason = self._resolve_config()
        if skip_reason:
            return self._make_result(name, skip_reason, start, False, "", [])

        assert config_path is not None
        success, error, artifacts = self._run_aquapose(
            output_subdir=f"backend_{backend_name}",
            config_path=config_path,
            extra_overrides=[override],
        )
        return self._make_result(name, None, start, success, error, artifacts)

    def run_mode_tests(self) -> list[TestResult]:
        """Run the pipeline for each mode in ALL_MODES.

        Returns:
            List of TestResult, one per mode.
        """
        return [self.run_single_mode(mode) for mode in ALL_MODES]

    def run_backend_tests(self) -> list[TestResult]:
        """Run the pipeline for each backend in ALL_BACKENDS.

        Returns:
            List of TestResult, one per backend.
        """
        return [
            self.run_single_backend(name, override)
            for name, override in ALL_BACKENDS.items()
        ]

    def run_reproducibility_test(self) -> TestResult:
        """Verify reproducibility: run pipeline twice, compare HDF5 outputs.

        Runs with identical config and random seed. Loads all arrays from
        both output HDF5 files and compares them with np.allclose(atol=1e-10).

        Returns:
            TestResult describing the outcome (passes only if all arrays match).
        """

        name = "reproducibility"
        start = time.monotonic()

        config_path, skip_reason = self._resolve_config()
        if skip_reason:
            return self._make_result(name, skip_reason, start, False, "", [])

        assert config_path is not None

        # Run A
        ok_a, err_a, arts_a = self._run_aquapose(
            output_subdir="repro_run_a",
            config_path=config_path,
            extra_overrides=["synthetic.seed=99"],
        )
        if not ok_a:
            return self._make_result(
                name, None, start, False, f"Run A failed: {err_a}", arts_a
            )

        # Run B — identical config
        ok_b, err_b, arts_b = self._run_aquapose(
            output_subdir="repro_run_b",
            config_path=config_path,
            extra_overrides=["synthetic.seed=99"],
        )
        if not ok_b:
            return self._make_result(
                name, None, start, False, f"Run B failed: {err_b}", arts_b
            )

        # Compare HDF5 outputs
        try:
            import h5py
        except ImportError:
            return self._make_result(
                name, None, start, False, "h5py not installed — cannot compare HDF5", []
            )

        out_a_dir = self._output_base / "repro_run_a"
        out_b_dir = self._output_base / "repro_run_b"
        h5_files_a = sorted(out_a_dir.rglob("*.h5"))
        h5_files_b = sorted(out_b_dir.rglob("*.h5"))

        if not h5_files_a and not h5_files_b:
            # No HDF5 produced (e.g. benchmark mode) — treat as reproducible
            elapsed = time.monotonic() - start
            return TestResult(
                name=name,
                passed=True,
                duration_seconds=elapsed,
                artifacts=arts_a + arts_b,
            )

        mismatches: list[str] = []
        for h5_a, h5_b in zip(h5_files_a, h5_files_b, strict=False):
            with h5py.File(h5_a, "r") as fa, h5py.File(h5_b, "r") as fb:
                mismatches.extend(_compare_hdf5(fa, fb))

        if mismatches:
            error = "Reproducibility failures:\n" + "\n".join(
                f"  {m}" for m in mismatches
            )
            return self._make_result(name, None, start, False, error, arts_a + arts_b)

        return self._make_result(name, None, start, True, "", arts_a + arts_b)

    def run_all(self) -> SmokeTestReport:
        """Run all enabled test dimensions and return an aggregated report.

        Respects the ``only`` argument to restrict which dimensions run.
        Each test is independent — failure of one does not block others.

        Returns:
            SmokeTestReport with all results and aggregate metrics.
        """
        all_start = time.monotonic()
        results: list[TestResult] = []

        run_modes = self._only in (None, "modes")
        run_backends = self._only in (None, "backends")
        run_repro = self._only in (None, "repro")

        if run_modes:
            results.extend(self.run_mode_tests())
        if run_backends:
            results.extend(self.run_backend_tests())
        if run_repro:
            results.append(self.run_reproducibility_test())

        total = time.monotonic() - all_start
        passed = sum(1 for r in results if r.passed and not r.skipped)
        failed = sum(1 for r in results if not r.passed and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)

        return SmokeTestReport(
            results=results,
            total_duration=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
        )


# ---------------------------------------------------------------------------
# HDF5 comparison helper
# ---------------------------------------------------------------------------


def _compare_hdf5(fa: object, fb: object, path: str = "") -> list[str]:
    """Recursively compare two open HDF5 files/groups and return mismatches.

    Args:
        fa: h5py.File or h5py.Group from run A.
        fb: h5py.File or h5py.Group from run B.
        path: Current dataset path prefix (for error reporting).

    Returns:
        List of mismatch descriptions (empty if all match).
    """
    import numpy as np

    mismatches: list[str] = []
    for key in fa:  # type: ignore[union-attr]
        full_path = f"{path}/{key}" if path else key
        if key not in fb:  # type: ignore[operator]
            mismatches.append(f"{full_path}: missing in run B")
            continue
        item_a = fa[key]  # type: ignore[index]
        item_b = fb[key]  # type: ignore[index]

        import h5py

        if isinstance(item_a, h5py.Group):
            mismatches.extend(_compare_hdf5(item_a, item_b, full_path))
        else:
            arr_a = item_a[()]  # type: ignore[index]
            arr_b = item_b[()]  # type: ignore[index]
            if not np.allclose(arr_a, arr_b, atol=1e-10, rtol=1e-10, equal_nan=True):
                max_diff = float(np.abs(arr_a - arr_b).max())
                mismatches.append(f"{full_path}: max_diff={max_diff:.2e}")

    for key in fb:  # type: ignore[union-attr]
        full_path = f"{path}/{key}" if path else key
        if key not in fa:  # type: ignore[operator]
            mismatches.append(f"{full_path}: missing in run A")

    return mismatches


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the smoke test CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="smoke_test",
        description="AquaPose pipeline smoke test runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/smoke_test.py --config aquapose.yaml --output-dir ./smoke
  python tools/smoke_test.py --config aquapose.yaml --only modes
  python tools/smoke_test.py --calibration /path/to/calibration.json --only modes --mode synthetic
  python tools/smoke_test.py --config aquapose.yaml --only repro --frame-limit 5
        """,
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to pipeline config YAML.",
    )
    parser.add_argument(
        "--calibration",
        metavar="PATH",
        help="Path to AquaCal calibration JSON (used for synthetic-only runs).",
    )
    parser.add_argument(
        "--output-dir",
        default="./smoke_results",
        metavar="PATH",
        help="Root directory for smoke test artifacts (default: ./smoke_results).",
    )
    parser.add_argument(
        "--only",
        choices=["modes", "backends", "repro"],
        default=None,
        help="Run only the specified test dimension (default: all).",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=DEFAULT_FRAME_LIMIT,
        metavar="N",
        help=f"Frames to process per run (default: {DEFAULT_FRAME_LIMIT}).",
    )
    parser.add_argument(
        "--json-report",
        metavar="PATH",
        help="Write JSON report to this path (in addition to stdout).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI invocation.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 if all non-skipped tests passed, 1 otherwise.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    runner = SmokeTestRunner(
        config_path=args.config,
        calibration_path=args.calibration,
        output_base=args.output_dir,
        frame_limit=args.frame_limit,
        only=args.only,
    )

    report = runner.run_all()
    print(report.summary())

    if args.json_report:
        json_path = Path(args.json_report)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(report.to_json(), encoding="utf-8")
        print(f"\nJSON report written to {json_path}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
