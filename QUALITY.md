# Python Quality Gate

Run the repository's complete fail-closed Python gate with:

```sh
uv run --project quality --python 3.12 --frozen check
```

The aggregate runs the anti-bypass policy, all five PitchAI Python preference
checkers, Ruff with every rule selected, BasedPyright in strict mode with fatal
warnings, full Pylint with a score floor of 10 and a 250-line module ceiling,
and Semgrep `ERROR` rules. It checks every repository Python and stub file,
including tests and tools. Only generated caches, virtual environments, build
outputs, and other non-source directories are omitted from discovery.

The read-only GitHub Actions workflow runs for every pull request and every
push branch. It intentionally has no branch-name filter, so repositories whose
default branch is `main`, `master`, `staging`, or another project-specific name
cannot silently skip the gate.

A nonzero result is an enforcement result, not permission to narrow coverage.
Repair violations with real dependencies or stubs and explicit boundary
architecture. Do not add inline suppressions, diagnostic downgrades, source
exclusions, ignored failures, or checker wrappers that hide a tool result.
