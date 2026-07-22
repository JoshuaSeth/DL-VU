# Copyright (c) 2026 PitchAI. All rights reserved.
"""Immutable values that define the repository Python quality policy."""

from __future__ import annotations

import re
from datetime import date, datetime, time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

type TomlScalar = str | int | float | bool | date | datetime | time
type TomlValue = TomlScalar | list[TomlValue] | dict[str, TomlValue]


class PolicyConfigurationError(TypeError):
    """A malformed value in the immutable strict quality policy."""

    def __init__(self, name: str, expected: str) -> None:
        """Initialize a policy-shape failure."""
        message = f"{name} must be {expected}"
        super().__init__(message)


INLINE_BYPASS = re.compile(
    r"#\s*(?:noqa\b|type:\s*ignore\b|ruff:\s*(?:ignore|noqa)\b|"
    r"pylint:\s*(?:disable(?:-next)?|skip-file)\b|nosem(?:grep)?\b|pyright:\s*)",
    re.IGNORECASE,
)
EXPECTED_GATES = (
    "no-validation-bypasses",
    "nested-event-loops",
    "no-vague-signatures",
    "no-single-use-one-line-functions",
    "no-pure-wrapper-functions",
    "no-dense-inline-comprehensions",
    "ruff",
    "basedpyright",
    "pylint",
    "semgrep",
)
EXPECTED_NON_SOURCE_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".semgrep",
        ".venv",
        ".vscode",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    },
)
EXPECTED_RUFF: Mapping[str, TomlValue] = {
    "line-length": 120,
    "target-version": "py312",
    "preview": True,
    "indent-width": 4,
    "lint": {
        "select": ["ALL"],
        "preview": True,
        "fixable": ["ALL"],
        "unfixable": [],
        "mccabe": {"max-complexity": 20},
        "pydocstyle": {"convention": "google"},
        "flake8-annotations": {"mypy-init-return": True},
        "flake8-type-checking": {"quote-annotations": True, "strict": True},
        "pylint": {"max-nested-blocks": 4, "max-locals": 30},
    },
    "format": {"docstring-code-format": False},
}
EXPECTED_BASEDPYRIGHT_CORE: Mapping[str, TomlValue] = {
    "typeCheckingMode": "strict",
    "failOnWarnings": True,
    "include": [".."],
    "pythonVersion": "3.12",
    "pythonPlatform": "All",
    "venvPath": ".",
    "venv": ".venv",
    "allowedUntypedLibraries": [],
    "reportAny": "error",
    "reportExplicitAny": "error",
    "reportInvalidCast": "error",
    "reportImplicitRelativeImport": "error",
    "reportPrivateLocalImportUsage": "error",
    "reportUnusedParameter": "error",
    "reportImplicitAbstractClass": "error",
    "reportInvalidAbstractMethod": "error",
    "reportIncompatibleUnannotatedOverride": "error",
    "reportUnannotatedClassAttribute": "error",
}
EXPECTED_PYLINT: Mapping[str, TomlValue] = {
    "main": {
        "jobs": 0,
        "reports": "no",
        "load-plugins": [
            "pylint.extensions.broad_try_clause",
            "pylint.extensions.overlapping_exceptions",
        ],
    },
    "similarities": {"min-similarity-lines": 6},
    "format": {"max-line-length": 120, "max-module-lines": 250},
}
EXPECTED_PROJECT: Mapping[str, TomlValue] = {
    "name": "pitchai-repository-quality",
    "version": "1.0.0",
    "description": "Fail-closed PitchAI Python quality gate",
    "requires-python": ">=3.12,<3.13",
    "dependencies": [
        "anyio==4.13.0",
        "basedpyright==1.39.8",
        "pylint==4.0.5",
        "ruff==0.15.17",
        "semgrep==1.166.0",
    ],
    "scripts": {"check": "pitchai_quality.check:main"},
}
EXPECTED_BUILD_SYSTEM: Mapping[str, TomlValue] = {
    "requires": ["setuptools==80.10.2"],
    "build-backend": "setuptools.build_meta",
}
EXPECTED_SEMGREP_SHA256 = "b3aa9b18b0993999899d45d704c44b516dd50e42634f00379d39ea5ddd1a647e"
EXPECTED_SEMGREPIGNORE_SHA256 = "5ac837fad6ac50281eec4bafdbd801f53b86229d6b9d7789ad677a2c707630e1"
EXPECTED_WORKFLOW_SHA256 = "443489610b07c40e4307d595484a09237d0b3a1bfa10f6eec67a0ace816708ea"
RUFF_ARGUMENTS = (
    "check",
    "--no-cache",
    "--ignore-noqa",
    "--no-respect-gitignore",
    "--no-force-exclude",
    "--select",
    "ALL",
    "--config",
    "quality/pyproject.toml",
)
SEMGREP_ARGUMENTS = (
    "--error",
    "--strict",
    "--disable-nosem",
    "--no-git-ignore",
    "--x-ignore-semgrepignore-files",
    "--max-target-bytes=0",
    "--timeout=0",
    "--timeout-threshold=0",
    "--disable-version-check",
    "--metrics=off",
)
REQUIRED_RUNNER_FRAGMENTS = (
    ("completed.returncode", "aggregate runner must propagate tool return codes"),
    ("if failures:", "aggregate runner must fail when any gate fails"),
    ("return 1", "aggregate runner must return nonzero on failure"),
    ("--ignore-noqa", "Ruff must ignore inline suppression comments"),
    ("--no-respect-gitignore", "Ruff must ignore Git source exclusions"),
    ("--no-force-exclude", "Ruff must not force source exclusions"),
    ("--warnings", "BasedPyright warnings must fail"),
    ("--fail-under=10", "Pylint score floor must be 10"),
    ("--disable-nosem", "Semgrep must ignore suppression comments"),
    ("--no-git-ignore", "Semgrep must ignore Git source exclusions"),
    ("--x-ignore-semgrepignore-files", "Semgrep must ignore repository source exclusions"),
    ("--max-target-bytes=0", "Semgrep may not skip large source files"),
    ("--timeout=0", "Semgrep may not skip source files on timeout"),
    ("--strict", "Semgrep configuration warnings must fail"),
)
REQUIRED_WORKFLOW_FRAGMENTS = (
    "permissions:\n  contents: read",
    "uv sync --project quality --python 3.12 --frozen",
    "uv run --project quality --python 3.12 --frozen check",
)
