# Copyright (c) 2026 PitchAI. All rights reserved.
"""Reject validation suppressions, source exclusions, and fail-open wiring."""

from __future__ import annotations

import hashlib
import sys
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pitchai_quality.source_files import NON_SOURCE_DIRECTORY_NAMES, REPOSITORY_ROOT, iter_python_files
from pitchai_quality.strict_policy import (
    EXPECTED_BASEDPYRIGHT_CORE,
    EXPECTED_BUILD_SYSTEM,
    EXPECTED_GATES,
    EXPECTED_PROJECT,
    EXPECTED_PYLINT,
    EXPECTED_RUFF,
    EXPECTED_SEMGREP_SHA256,
    EXPECTED_SEMGREPIGNORE_SHA256,
    EXPECTED_WORKFLOW_SHA256,
    INLINE_BYPASS,
    REQUIRED_RUNNER_FRAGMENTS,
    REQUIRED_WORKFLOW_FRAGMENTS,
    PolicyConfigurationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pitchai_quality.strict_policy import TomlValue

_QUALITY_ROOT = REPOSITORY_ROOT / "quality"
_CONFIG_PATHS = (
    _QUALITY_ROOT / "pyproject.toml",
    _QUALITY_ROOT / ".semgrep.yml",
    _QUALITY_ROOT / ".semgrepignore",
    REPOSITORY_ROOT / ".github" / "workflows" / "python-strict.yml",
)


@dataclass(frozen=True)
class Violation:
    """One validation bypass with source location and explanation."""

    path: Path
    line: int
    reason: str


def _table(value: TomlValue | None, *, name: str) -> Mapping[str, TomlValue]:
    if not isinstance(value, dict):
        raise PolicyConfigurationError(name, "a TOML table")
    return value


def _string_sequence(value: TomlValue | None, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PolicyConfigurationError(name, "a list of strings")
    return tuple(cast("list[str]", value))


def _line_number(path: Path, fragment: str) -> int:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if fragment in line:
            return line_number
    return 1


def _inline_violations() -> list[Violation]:
    violations: list[Violation] = []
    python_files = iter_python_files((REPOSITORY_ROOT,))
    for path in (*python_files, *_CONFIG_PATHS):
        if not path.is_file():
            violations.append(Violation(path, 1, "required quality file is missing"))
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if INLINE_BYPASS.search(line):
                violations.append(Violation(path, line_number, "inline validation suppression is forbidden"))
    return violations


def _ruff_violations(path: Path, tool: Mapping[str, TomlValue]) -> list[Violation]:
    ruff = _table(tool.get("ruff"), name="tool.ruff")
    if ruff == EXPECTED_RUFF:
        return []
    return [Violation(path, _line_number(path, "[tool.ruff]"), "Ruff policy must match the complete strict profile")]


def _local_typing_path_violations(path: Path, key: str, value: TomlValue) -> list[Violation]:
    if key == "stubPath":
        if not isinstance(value, str):
            return [Violation(path, _line_number(path, key), "BasedPyright stubPath must be a local directory")]
        raw_paths = (value,)
    else:
        raw_paths = _string_sequence(value, name=f"tool.basedpyright.{key}")

    violations: list[Violation] = []
    for raw_path in raw_paths:
        resolved = (_QUALITY_ROOT / raw_path).resolve(strict=False)
        if not resolved.is_relative_to(REPOSITORY_ROOT) or not resolved.is_dir():
            violations.append(
                Violation(path, _line_number(path, key), f"BasedPyright {key} must resolve to a repository directory"),
            )
    return violations


def _basedpyright_violations(path: Path, tool: Mapping[str, TomlValue]) -> list[Violation]:
    config = _table(tool.get("basedpyright"), name="tool.basedpyright")
    violations: list[Violation] = []
    for key, expected in EXPECTED_BASEDPYRIGHT_CORE.items():
        if config.get(key) != expected:
            violations.append(Violation(path, _line_number(path, key), f"BasedPyright {key} strict value changed"))
    allowed_keys = {*EXPECTED_BASEDPYRIGHT_CORE, "stubPath", "extraPaths"}
    for key, value in config.items():
        if key.startswith("report") and value != "error":
            violations.append(Violation(path, _line_number(path, key), f"BasedPyright diagnostic {key} must be error"))
        elif key not in allowed_keys and not key.startswith("report"):
            violations.append(Violation(path, _line_number(path, key), f"BasedPyright option {key} is not approved"))
    for key in ("stubPath", "extraPaths"):
        if key in config:
            violations.extend(_local_typing_path_violations(path, key, config[key]))
    return violations


def _pylint_violations(path: Path, tool: Mapping[str, TomlValue]) -> list[Violation]:
    pylint = _table(tool.get("pylint"), name="tool.pylint")
    if pylint == EXPECTED_PYLINT:
        return []
    return [
        Violation(
            path,
            _line_number(path, "[tool.pylint.main]"),
            "Pylint policy must match the complete strict profile",
        ),
    ]


def _pyproject_violations() -> list[Violation]:
    path = _QUALITY_ROOT / "pyproject.toml"
    project = cast("dict[str, TomlValue]", tomllib.loads(path.read_text(encoding="utf-8")))
    tool = _table(project.get("tool"), name="tool")
    violations = [
        *_ruff_violations(path, tool),
        *_basedpyright_violations(path, tool),
        *_pylint_violations(path, tool),
    ]
    expected_tool_names = {"setuptools", "uv", "ruff", "basedpyright", "pylint"}
    if set(tool) != expected_tool_names:
        violations.append(Violation(path, _line_number(path, "[tool."), "quality tool tables changed without review"))
    if tool.get("setuptools") != {"packages": ["pitchai_quality"]}:
        violations.append(Violation(path, _line_number(path, "[tool.setuptools]"), "quality package source changed"))
    if tool.get("uv") != {"package": True}:
        violations.append(Violation(path, _line_number(path, "[tool.uv]"), "quality project must remain packaged"))
    if project.get("project") != EXPECTED_PROJECT:
        violations.append(
            Violation(path, _line_number(path, "[project]"), "locked quality dependencies or entry point changed"),
        )
    if project.get("build-system") != EXPECTED_BUILD_SYSTEM:
        violations.append(Violation(path, _line_number(path, "[build-system]"), "locked quality build system changed"))
    if set(project) != {"project", "build-system", "tool"}:
        violations.append(Violation(path, 1, "unexpected quality project configuration table"))
    return violations


def _semgrep_violations() -> list[Violation]:
    ignore_path = _QUALITY_ROOT / ".semgrepignore"
    allowed = {f"{name}/" for name in NON_SOURCE_DIRECTORY_NAMES}
    violations: list[Violation] = []
    for line_number, raw_line in enumerate(ignore_path.read_text(encoding="utf-8").splitlines(), start=1):
        entry = raw_line.strip()
        if entry and not entry.startswith("#") and entry not in allowed:
            violations.append(Violation(ignore_path, line_number, f"Semgrep excludes a source path: {entry}"))
    config_path = _QUALITY_ROOT / ".semgrep.yml"
    if _sha256(config_path) != EXPECTED_SEMGREP_SHA256:
        violations.append(Violation(config_path, 1, "Semgrep rules changed without an explicit policy review"))
    if _sha256(ignore_path) != EXPECTED_SEMGREPIGNORE_SHA256:
        violations.append(Violation(ignore_path, 1, "Semgrep ignore policy changed"))
    for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("exclude:"):
            violations.append(Violation(config_path, line_number, "Semgrep rule-level source exclusion is forbidden"))
    return violations


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wiring_violations() -> list[Violation]:
    runner_path = _QUALITY_ROOT / "pitchai_quality" / "check.py"
    policy_path = _QUALITY_ROOT / "pitchai_quality" / "strict_policy.py"
    runner = runner_path.read_text(encoding="utf-8") + policy_path.read_text(encoding="utf-8")
    violations: list[Violation] = []
    violations.extend(
        Violation(runner_path, 1, f"aggregate runner omits required gate {gate}")
        for gate in EXPECTED_GATES
        if f'"{gate}"' not in runner
    )
    for fragment, reason in REQUIRED_RUNNER_FRAGMENTS:
        if fragment not in runner:
            violations.append(Violation(runner_path, 1, reason))

    workflow_path = REPOSITORY_ROOT / ".github" / "workflows" / "python-strict.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    if _sha256(workflow_path) != EXPECTED_WORKFLOW_SHA256:
        violations.append(Violation(workflow_path, 1, "strict CI workflow changed without an explicit policy review"))
    violations.extend(
        Violation(workflow_path, 1, f"strict workflow omits `{fragment}`")
        for fragment in REQUIRED_WORKFLOW_FRAGMENTS
        if fragment not in workflow
    )
    if "continue-on-error" in workflow:
        violations.append(
            Violation(
                workflow_path,
                _line_number(workflow_path, "continue-on-error"),
                "CI may not ignore gate failures",
            ),
        )
    return violations


def main() -> int:
    """Validate that every strict gate is complete and fail-closed.

    Returns:
        Zero when the quality policy is intact; otherwise one.
    """
    violations = [*_inline_violations(), *_pyproject_violations(), *_semgrep_violations(), *_wiring_violations()]
    if violations:
        for violation in violations:
            relative = (
                violation.path.relative_to(REPOSITORY_ROOT)
                if violation.path.is_relative_to(REPOSITORY_ROOT)
                else violation.path
            )
            sys.stderr.write(f"{relative}:{violation.line}: {violation.reason}\n")
        return 1
    sys.stdout.write("ok no_validation_bypasses\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
