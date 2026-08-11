# Full test suite for `hpcp` (pytest + doctests)

Date: 2026-08-10  
Status: Approved design (pending implementation plan)

## Problem

`hpcp` is a large single-module CLI (`hpcp.py`, ~4k lines, ~89 top-level functions and 2 classes). Automated testing today is thin:

- Doctests exist only on `format_bytes` and `trim_paths`.
- Pytest coverage is effectively limited to `-co` / `--content_only` (`tests/test_content_only.py`).

There is no systematic suite for the public API surface (listing, identity/hash, copy, remove, CLI args, path/FS helpers, imaging/loop/dd, Windows/GUI smoke).

## Goal

Build a **maintainable full public-API test suite** that:

1. Uses **doctests** for pure/helper functions with stable examples (docstring-only edits in `hpcp.py`).
2. Uses **pytest** for filesystem, concurrency, CLI, imaging, and anything needing fixtures or mocks.
3. Covers **everything public** in `hpcp.py`, including Windows/GUI paths, with environment skips where the host cannot run them.
4. Prefers **real** loopback/image fixtures for imaging paths; skips when tools or privileges are missing.
5. Does **not** change production logic — bugs found during testing are **reported**, not patched in `hpcp.py` as part of this work.

No hard line-coverage percentage. Success is broad happy-path + key edge-case coverage across the public surface.

## Non-goals

- Refactoring or fixing production logic in `hpcp.py` (docstring-only doctest additions are allowed).
- Guaranteeing imaging/dd tests run in every CI environment (they must skip cleanly instead).
- Adding cloud Windows or privileged runners (markers should allow that later).
- Exhaustive branch coverage of every orchestrator path (`hpcp`, `process_*`, `main`).
- Parallel pytest (`-n`) as a v1 requirement.

## Constraints (hard rules)

| Rule | Detail |
|------|--------|
| No logic changes | Do not alter `hpcp.py` behavior to make tests pass. |
| Docstrings OK | New/expanded doctests in docstrings only. |
| Report bugs | Record incorrect behavior vs README/docs/intent; do not “fix” product code. |
| Keep existing tests | Retain `tests/test_content_only.py` (light cleanup only if needed for shared fixtures). |
| Real imaging fixtures | Prefer real loop/image when available; skip otherwise. |
| Windows/GUI on Linux | Import/structure/smoke only; skip real GUI/Windows behavior unless that OS is present. |

## Approach

**Feature-layered pytest + helper doctests** (chosen over per-function test modules or scenario-only integration).

Rejected alternatives:

- **One pytest file per public function** — explicit but high boilerplate; awkward for orchestrators.
- **Scenario-first integration only** — strong where it runs, weak isolation and uneven public-API coverage.

## Architecture

**Single entrypoint:** `pytest`, configured to also run doctests from `hpcp.py` (`--doctest-modules` via `pytest.ini` or equivalent light config).

**Proposed layout:**

```text
tests/
  conftest.py              # shared fixtures + markers
  test_content_only.py     # keep
  test_helpers.py          # pytest where doctests are not enough
  test_listing.py          # get_file_list*, exclude, listing edges
  test_identity_hash.py    # hash_file, is_file_identical, hash_size / full_hash
  test_copy.py             # copy_file*, serial/parallel, metadata vs content
  test_remove.py           # delete_*, remove_extra_*, process_remove
  test_cli_args.py         # get_args flag matrix / interactions
  test_paths_fs.py         # expand_paths, dest resolution, free space, fs helpers
  test_imaging.py          # loop/partition/dd — real fixtures + skips
  test_windows_gui.py      # import/structure/smoke; skip unless Windows / display
docs/superpowers/specs/
  2026-08-10-hpcp-test-suite-design.md   # this document
tests/BUGS.md              # findings that must not be fixed in product code during this work
```

**Markers:** `linux`, `windows`, `root`, `loop`, `gui` — used for skip conditions and optional manual runs.

## Doctest strategy

Prioritize pure helpers with deterministic I/O. Candidates (add where examples stay short and stable):

- Already present: `format_bytes`, `trim_paths`
- Likely additions: `format_time`, `natural_sort`, `is_excluded` / `format_exclude`, `get_timestamp_precision`, other pure path/string helpers that do not touch the filesystem or privileges

Do **not** force doctests onto side-effecting, privilege-heavy, or highly environment-dependent APIs — those belong in pytest.

Docstring edits must not change runtime behavior.

## Pytest fixtures and data flow

From `tests/conftest.py`:

- Temp source/dest trees (files, dirs, symlinks, empty, nested)
- Reset of module globals touched by tests (`CONTENT_ONLY`, `NO_CREATE_DIR`, and any others tests mutate) so state does not leak
- `sys.argv` restore helper for `get_args` tests
- Optional `loop_image` (or equivalent): create a small file-backed loop image when `losetup` and privileges exist; otherwise `pytest.skip` with a clear reason

**Call style:** Prefer invoking public functions directly (`copy_file`, `get_file_list`, `get_args`, …). Use subprocess / `main` only for a few CLI smoke cases. Avoid testing private helpers unless needed to diagnose a public failure.

**Isolation:** Each test owns temps and restores globals/`sys.argv`.

## Coverage map

| Area | Style | Depth |
|------|--------|--------|
| Pure helpers | Doctests + light pytest | Happy + edge |
| Listing / identical / hash | Pytest + temps | Happy + key edges (`hash_size=0`, full_hash, symlinks) |
| Copy serial/parallel / dir metadata / content_only | Pytest | Happy + keep existing content_only tests |
| Remove / remove_extra | Pytest | Happy + safety edges (empty lists, exclude) |
| CLI `get_args` | Pytest argv matrix | Major flags and a few interactions |
| Paths / dest / free space / fs helpers | Pytest | Happy + missing-path edges |
| Imaging / loop / partitions / dd | Real fixtures | Happy where possible; skip if no root/tools |
| Windows / GUI | Smoke / import | Skip unless Windows (GUI may need display) |
| Orchestrators (`hpcp`, `process_*`, `main`) | Few integration smokes | Not every branch |

## Bug reporting policy

When observed behavior disagrees with README/docs/intent:

1. Do **not** change `hpcp.py` logic.
2. Record in `tests/BUGS.md`: function/symbol, repro steps, observed vs expected, and how the suite treats it.
3. Prefer asserting **documented intended** behavior. If that would fail on a known bug, use `xfail` (non-strict) or skip with a bug-id reference so the suite stays green and informative.

Unexpected exceptions from code under test fail the test (and may add a bug note). Do not catch-and-pass. Missing binaries or permission errors on optional imaging paths → skip with reason.

## Verification

Default (unprivileged Linux):

```bash
pytest -q
```

Optional / manual privileged imaging:

```bash
pytest -q -m loop
# or documented root invocation as needed
```

**Done when:**

- Suite runs clean on a normal Linux dev box without root (skips documented via markers/reasons).
- Doctests are collected and run via pytest.
- Each coverage-map area has at least smoke/happy coverage as specified.
- `tests/BUGS.md` exists (may start empty or with any findings from writing tests).
- No production logic changes landed in `hpcp.py` beyond docstring doctests.

## Implementation notes (for the plan, not product code)

- Add light pytest config at repo root (`pytest.ini` or equivalent).
- Optionally declare `pytest` as a dev dependency if the project gains a place for it; otherwise document `pip install pytest` in the plan/README test section only if needed — prefer minimal packaging churn.
- Reuse patterns from `tests/test_content_only.py` (argv restore, temp dirs, global toggles).
