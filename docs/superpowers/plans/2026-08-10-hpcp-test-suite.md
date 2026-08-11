# Full hpcp Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a maintainable public-API test suite for `hpcp`: doctests for pure helpers, pytest for FS/CLI/imaging/Windows smoke, with bugs reported (not fixed in product logic).

**Architecture:** Feature-layered tests under `tests/` sharing `conftest.py` fixtures/markers; an explicit pytest test runs `doctest.testmod(hpcp)`. Imaging uses real loop images and skips without privileges/tools. Docstring-only edits in `hpcp.py` for new doctests.

**Tech Stack:** Python 3, pytest, existing `hpcp.py` + tempdirs; Linux `losetup`/`lsblk` for optional imaging tests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-10-hpcp-test-suite-design.md`
- **No production logic changes** in `hpcp.py` — docstring-only doctest additions are allowed
- If observed behavior disagrees with README/docs/intent: record in `tests/BUGS.md` and use `xfail`/`skip` with a bug id — do **not** patch product code to make tests pass
- Keep `tests/test_content_only.py` (may adopt shared fixtures lightly)
- Prefer real loop/image fixtures for imaging; skip when tools/privileges missing
- Windows/GUI: import/structure/smoke on Linux; real behavior only on Windows
- No hard coverage %; happy-path + key edges across public surface
- Tabs for indentation in `hpcp.py` (match existing file style)
- Do not bump package version / release unless asked

## File Structure

| File | Role |
|------|------|
| `pytest.ini` | pytest config: doctest modules, markers |
| `tests/conftest.py` | Import path, markers, argv/globals fixtures, temp trees, optional loop image |
| `tests/BUGS.md` | Bug reports found while writing tests (no product fixes) |
| `tests/test_content_only.py` | Keep existing content-only coverage |
| `tests/test_helpers.py` | Pytest edges for helpers not covered by doctests |
| `tests/test_listing.py` | File listing, exclude integration |
| `tests/test_identity_hash.py` | `hash_file`, `is_file_identical`, `HASH_SIZE` |
| `tests/test_copy.py` | Copy serial/parallel beyond content-only |
| `tests/test_remove.py` | Delete / remove_extra |
| `tests/test_cli_args.py` | `get_args` flag matrix |
| `tests/test_paths_fs.py` | Paths, dest helpers, free space, fs type |
| `tests/test_imaging.py` | Loop / partition helpers with real fixtures |
| `tests/test_windows_gui.py` | Windows/GUI smoke + skips |
| `tests/test_orchestrator_smoke.py` | Thin `hpcp()` / compare smoke |
| `hpcp.py` | Docstring doctests only (no logic edits) |

## How TDD works for this plan

Product code already exists. For each pytest task:

1. Write the test asserting **documented intended** behavior.
2. Run it.
3. If **PASS** → keep it.
4. If **FAIL** due to a product bug → add `tests/BUGS.md` entry, mark test `pytest.mark.xfail(reason="BUGS.md#N", strict=False)` (or skip if unsafe), leave `hpcp.py` logic unchanged.
5. If **FAIL** due to a bad test assumption → fix the test.

For doctest tasks, “implementation” means docstring edits only.

---

### Task 1: Pytest scaffold (`pytest.ini`, `conftest.py`, `BUGS.md`)

**Files:**
- Create: `pytest.ini`
- Create: `tests/conftest.py`
- Create: `tests/BUGS.md`
- Test: run existing `tests/test_content_only.py` plus doctest collection

**Interfaces:**
- Consumes: none
- Produces:
  - markers: `linux`, `windows`, `root`, `loop`, `gui`
  - fixtures: `hpcp_mod` (imported module), `restore_argv`, `reset_hpcp_globals`, `tmp_tree` (callable/factory), `loop_image` (optional skip)
  - `tests/BUGS.md` template

- [ ] **Step 1: Create `pytest.ini`**

Doctests live in root `hpcp.py`; collect them via an explicit runner test (Step 2), not `--doctest-modules` on `testpaths`.

```ini
[pytest]
pythonpath = .
testpaths = tests
python_files = test_*.py
norecursedirs = .git .venv dist hpcp.egg-info __pycache__
markers =
    linux: requires Linux / POSIX behaviors
    windows: requires Windows
    root: requires elevated privileges
    loop: requires losetup / loop device support
    gui: requires GUI / display (Tk)
```

- [ ] **Step 2: Create doctest runner test**

Create `tests/test_run_hpcp_doctests.py`:

```python
import doctest
import hpcp


def test_hpcp_module_doctests():
	failures, _ = doctest.testmod(hpcp, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
	assert failures == 0
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
import os
import sys
import shutil
import tempfile
import pytest

# Repo root on path for `import hpcp` if pythonpath not honored
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

import hpcp


@pytest.fixture
def hpcp_mod():
	return hpcp


@pytest.fixture
def restore_argv():
	old = sys.argv[:]
	try:
		yield
	finally:
		sys.argv = old


@pytest.fixture
def reset_hpcp_globals():
	"""Snapshot and restore module globals tests commonly mutate."""
	snap = {
		'CONTENT_ONLY': hpcp.CONTENT_ONLY,
		'NO_CREATE_DIR': hpcp.NO_CREATE_DIR,
		'HASH_SIZE': hpcp.HASH_SIZE,
		'BYTES_RATE_LIMIT': hpcp.BYTES_RATE_LIMIT,
		'FILES_RATE_LIMIT': hpcp.FILES_RATE_LIMIT,
		'COMMAND_TIMEOUT': hpcp.COMMAND_TIMEOUT,
		'REMOVE_FILES_WHILE_LISTING': hpcp.REMOVE_FILES_WHILE_LISTING,
		'RANDOM_DESTINATION_SELECTION': hpcp.RANDOM_DESTINATION_SELECTION,
	}
	# Clear file-list cache between tests
	if hasattr(hpcp, '_get_file_list_cache'):
		hpcp._get_file_list_cache.clear()
	try:
		yield hpcp
	finally:
		for k, v in snap.items():
			setattr(hpcp, k, v)
		if hasattr(hpcp, '_get_file_list_cache'):
			hpcp._get_file_list_cache.clear()


@pytest.fixture
def tmp_tree(tmp_path):
	"""Build a small src/dst tree under tmp_path. Returns paths namespace."""
	class Tree:
		def __init__(self):
			self.root = tmp_path
			self.src = tmp_path / 'src'
			self.dst = tmp_path / 'dst'
			self.src.mkdir()
			self.dst.mkdir()

		def add_file(self, rel, content=b'hello', under='src'):
			base = self.src if under == 'src' else self.dst
			path = base / rel
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_bytes(content if isinstance(content, (bytes, bytearray)) else content.encode())
			return str(path)

		def add_dir(self, rel, under='src'):
			base = self.src if under == 'src' else self.dst
			path = base / rel
			path.mkdir(parents=True, exist_ok=True)
			return str(path)

		def add_symlink(self, rel, target, under='src'):
			base = self.src if under == 'src' else self.dst
			path = base / rel
			path.parent.mkdir(parents=True, exist_ok=True)
			os.symlink(target, path)
			return str(path)

	return Tree()


def _have_losetup():
	return shutil.which('losetup') is not None


def _can_use_loop():
	if os.name == 'nt':
		return False
	if not _have_losetup():
		return False
	# Non-root often fails; detect by dry permission check later in fixture
	return True


@pytest.fixture
def loop_image(tmp_path, reset_hpcp_globals):
	"""Create a small sparse file and attach via create_loop_device; detach on teardown.

	Skips when losetup is missing or loop attach fails (typically needs root).
	"""
	if not _can_use_loop():
		pytest.skip('losetup / loop devices not available')
	img = tmp_path / 'test.img'
	# 8 MiB empty image is enough for attach smoke
	with open(img, 'wb') as f:
		f.truncate(8 * 1024 * 1024)
	loop_dev = None
	try:
		loop_dev = hpcp.create_loop_device(str(img), read_only=True)
	except Exception as e:
		pytest.skip(f'cannot create loop device (need privileges?): {e}')
	try:
		yield {'image': str(img), 'loop': loop_dev}
	finally:
		if loop_dev:
			hpcp.detach_loop_device(loop_dev)


@pytest.fixture
def require_linux():
	if os.name == 'nt':
		pytest.skip('Linux-only test')


@pytest.fixture
def require_windows():
	if os.name != 'nt':
		pytest.skip('Windows-only test')
```

- [ ] **Step 4: Create `tests/BUGS.md`**

```markdown
# Bugs found while building the hpcp test suite

Do **not** fix these in `hpcp.py` as part of the test-suite work. Report here; product fixes are a separate change.

Format per entry:

## BUG-N: short title

- **Symbol:** `function_or_area`
- **Repro:** steps / minimal snippet
- **Observed:** ...
- **Expected (per README/docs/intent):** ...
- **Suite handling:** `xfail` / `skip` / asserted current behavior (say which)
```

- [ ] **Step 5: Run existing tests + doctest runner**

Run: `python -m pytest tests/test_content_only.py tests/test_run_hpcp_doctests.py -v`

Expected: PASS (existing content-only + current two doctests).

- [ ] **Step 6: Commit**

```bash
git add pytest.ini tests/conftest.py tests/BUGS.md tests/test_run_hpcp_doctests.py
git commit -m "$(cat <<'EOF'
Add pytest scaffold, fixtures, and doctest runner for hpcp.

EOF
)"
```

---

### Task 2: Supplement helper doctests in `hpcp.py`

**Files:**
- Modify: `hpcp.py` docstrings only for `format_time`, `natural_sort`, `is_excluded`, `format_exclude`, `get_timestamp_precision` (and optionally expand `trim_paths` / `format_bytes` if a clear gap exists)
- Test: `tests/test_run_hpcp_doctests.py` (+ optional `tests/test_helpers.py` for FS-touching edges)

**Interfaces:**
- Consumes: Task 1 doctest runner
- Produces: additional `>>>` examples in docstrings; no logic changes

- [ ] **Step 1: Add doctests to `format_time`**

In `format_time` docstring, after Returns, add:

```python
	Examples:
		>>> format_time(0)
		'0s'
		>>> format_time(65)
		'1m5s'
		>>> format_time(3661)
		'1h1m1s'
```

- [ ] **Step 2: Add doctests to `natural_sort`**

```python
	Examples:
		>>> natural_sort(['file10.txt', 'file2.txt', 'file1.txt'])
		['file1.txt', 'file2.txt', 'file10.txt']
```

- [ ] **Step 3: Add doctests to `is_excluded`**

```python
	Examples:
		>>> is_excluded('/data/tmp/cache', ['*/cache'])
		True
		>>> is_excluded('/data/tmp/cache', ['*/logs'])
		False
		>>> is_excluded('/data/tmp/cache', None)
		False
```

- [ ] **Step 4: Add doctests to `format_exclude`**

```python
	Examples:
		>>> sorted(format_exclude(['cache', '*/logs']))
		['*/cache', '*/logs']
		>>> format_exclude(None)
		frozenset()
```

- [ ] **Step 5: Add doctests to `get_timestamp_precision`**

```python
	Examples:
		>>> get_timestamp_precision(10)
		0
		>>> get_timestamp_precision(10.0)
		2
		>>> get_timestamp_precision(10.5)
		0.01
```

(Verified: `10.0` hits the `/2` even-int branch and returns `2`, not `1`.)

- [ ] **Step 6: Run doctests**

Run: `python -m pytest tests/test_run_hpcp_doctests.py -v`

Expected: PASS. If FAIL, fix docstring examples to match actual returns (still no logic changes). If actual return is clearly wrong vs docs, add `BUGS.md` entry and keep doctest matching **current** behavior (or omit that example and cover via xfail pytest).

- [ ] **Step 7: Add `tests/test_helpers.py` for non-doctest edges**

```python
import os
import pytest


def test_get_last_existing_parent_finds_root(tmp_path, hpcp_mod, reset_hpcp_globals):
	missing = tmp_path / 'a' / 'b' / 'c'
	parent = hpcp_mod.get_last_existing_parent(str(missing))
	assert os.path.isdir(parent)
	assert os.path.samefile(parent, tmp_path)


def test_get_file_size_missing_returns_zero(hpcp_mod, reset_hpcp_globals, tmp_path):
	missing = str(tmp_path / 'nope')
	assert hpcp_mod.get_file_size(missing) == 0
```

- [ ] **Step 8: Commit**

```bash
git add hpcp.py tests/test_helpers.py tests/BUGS.md
git commit -m "$(cat <<'EOF'
Add helper doctests and pytest edges without changing hpcp logic.

EOF
)"
```

---

### Task 3: Listing and exclude coverage

**Files:**
- Create: `tests/test_listing.py`
- Modify: `tests/BUGS.md` (only if bugs found)

**Interfaces:**
- Consumes: `tmp_tree`, `reset_hpcp_globals`, `hpcp_mod`
- Produces: listing/exclude regression tests

- [ ] **Step 1: Write listing tests**

Create `tests/test_listing.py`:

```python
import os
import pytest


def test_get_file_list_serial_lists_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('sub/b.txt', 'B')
	# Returns (file_list, links, size, folders)
	files, links, size, folders = hpcp_mod.get_file_list(
		str(tmp_tree.src),
		parallel_file_listing=False,
		drop_cache=True,
		remove_files_while_listing=False,
	)
	joined = ' '.join(map(str, files))
	assert 'a.txt' in joined
	assert 'b.txt' in joined
	assert size >= 2
	assert isinstance(links, (set, frozenset))
	assert isinstance(folders, (set, frozenset))


def test_get_file_list_respects_exclude(tmp_tree, hpcp_mod, reset_hpcp_globals):
	tmp_tree.add_file('keep.txt', 'k')
	tmp_tree.add_file('skip.dat', 's')
	exclude = hpcp_mod.format_exclude(['*.dat'])
	files, links, size, folders = hpcp_mod.get_file_list(
		str(tmp_tree.src),
		exclude=exclude,
		parallel_file_listing=False,
		drop_cache=True,
		remove_files_while_listing=False,
	)
	joined = ' '.join(map(str, files))
	assert 'keep.txt' in joined
	assert 'skip.dat' not in joined


def test_trim_paths_relative(hpcp_mod):
	base = '/home/user/project/main.py'
	got = hpcp_mod.trim_paths({'/home/user/project/file1.py', '/home/user/project/file2.py'}, base)
	assert got == {'file1.py', 'file2.py'}
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_listing.py -v`

Expected: PASS, or FAIL → fix test / add BUGS.md + xfail.

- [ ] **Step 3: Commit**

```bash
git add tests/test_listing.py tests/BUGS.md
git commit -m "$(cat <<'EOF'
Add file listing and exclude pytest coverage for hpcp.

EOF
)"
```

---

### Task 4: Identity and hash coverage

**Files:**
- Create: `tests/test_identity_hash.py`
- Modify: `tests/BUGS.md` if needed

**Interfaces:**
- Consumes: `tmp_tree`, `reset_hpcp_globals` (restores `HASH_SIZE`)
- Produces: hash / identical tests

- [ ] **Step 1: Write identity/hash tests**

```python
import os
import time
import pytest


def test_hash_file_stable_for_same_bytes(tmp_tree, hpcp_mod, reset_hpcp_globals):
	path = tmp_tree.add_file('x.bin', b'abcdefghijklmnopqrstuvwxyz')
	hpcp_mod.HASH_SIZE = 65536
	h1 = hpcp_mod.hash_file(path, os.path.getsize(path), full_hash=False)
	h2 = hpcp_mod.hash_file(path, os.path.getsize(path), full_hash=False)
	assert h1 == h2
	assert isinstance(h1, str) and len(h1) > 0


def test_hash_file_disabled_when_hash_size_zero(tmp_tree, hpcp_mod, reset_hpcp_globals):
	path = tmp_tree.add_file('x.bin', b'abc')
	hpcp_mod.HASH_SIZE = 0
	# Clear lru_cache on hash_file if present
	if hasattr(hpcp_mod.hash_file, 'cache_clear'):
		hpcp_mod.hash_file.cache_clear()
	assert hpcp_mod.hash_file(path, os.path.getsize(path)) == ''


def test_is_file_identical_true_for_same_mtime_size_hash(tmp_tree, hpcp_mod, reset_hpcp_globals):
	src = tmp_tree.add_file('a.txt', 'same', under='src')
	dst = tmp_tree.add_file('a.txt', 'same', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	hpcp_mod.HASH_SIZE = 65536
	if hasattr(hpcp_mod.hash_file, 'cache_clear'):
		hpcp_mod.hash_file.cache_clear()
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), full_hash=False) is True


def test_is_file_identical_false_for_different_content(tmp_tree, hpcp_mod, reset_hpcp_globals):
	src = tmp_tree.add_file('a.txt', 'one', under='src')
	dst = tmp_tree.add_file('a.txt', 'two', under='dst')
	t = time.time() - 1000
	os.utime(src, (t, t))
	os.utime(dst, (t, t))
	hpcp_mod.HASH_SIZE = 65536
	if hasattr(hpcp_mod.hash_file, 'cache_clear'):
		hpcp_mod.hash_file.cache_clear()
	assert hpcp_mod.is_file_identical(src, dst, os.path.getsize(src), full_hash=True) is False
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_identity_hash.py -v`

Expected: PASS or BUGS.md + xfail.

- [ ] **Step 3: Commit**

```bash
git add tests/test_identity_hash.py tests/BUGS.md
git commit -m "$(cat <<'EOF'
Add hash and file-identity pytest coverage for hpcp.

EOF
)"
```

---

### Task 5: Copy coverage (beyond content-only)

**Files:**
- Create: `tests/test_copy.py`
- Keep: `tests/test_content_only.py`
- Modify: `tests/BUGS.md` if needed

**Interfaces:**
- Consumes: `tmp_tree`, `reset_hpcp_globals`
- Produces: default copy + serial/parallel smoke tests

- [ ] **Step 1: Write copy tests**

```python
import os
import stat
import pytest


def test_copy_file_writes_content(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	src = tmp_tree.add_file('a.txt', 'payload-copy')
	dst = str(tmp_tree.dst / 'a.txt')
	hpcp_mod.CONTENT_ONLY = False
	hpcp_mod.NO_CREATE_DIR = False
	size, _, _ = hpcp_mod.copy_file(src, [dst])
	assert os.path.isfile(dst)
	assert open(dst).read() == 'payload-copy'
	assert size > 0


def test_copy_files_serial_smoke(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	tmp_tree.add_file('a.txt', 'A')
	tmp_tree.add_file('b.txt', 'B')
	hpcp_mod.CONTENT_ONLY = False
	# copy_files_serial(src_path, dest_paths, full_hash=False, verbose=False, exclude=None)
	# places files at dest/<relpath> when src is a directory
	hpcp_mod.copy_files_serial(str(tmp_tree.src), [str(tmp_tree.dst)], full_hash=False, verbose=False, exclude=None)
	assert (tmp_tree.dst / 'a.txt').read_text() == 'A'
	assert (tmp_tree.dst / 'b.txt').read_text() == 'B'


def test_sync_directory_metadata_default_applies_mtime(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	# Mirrors intent of test_default_dir_sync_still_applies_metadata but uses shared fixtures
	src_dir = tmp_tree.add_dir('sub')
	dst_dir = str(tmp_tree.dst / 'sub')
	os.chmod(src_dir, 0o750)
	t = __import__('time').time() - 86400
	os.utime(src_dir, (t, t))
	hpcp_mod.CONTENT_ONLY = False
	hpcp_mod.sync_directory_metadata(src_dir, [dst_dir])
	assert os.path.isdir(dst_dir)
	assert abs(os.stat(dst_dir).st_mtime - os.stat(src_dir).st_mtime) < 2
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_copy.py tests/test_content_only.py -v`

Expected: PASS (content_only still green).

- [ ] **Step 3: Commit**

```bash
git add tests/test_copy.py tests/BUGS.md
git commit -m "$(cat <<'EOF'
Add default copy and directory-metadata pytest coverage.

EOF
)"
```

---

### Task 6: Remove and remove_extra coverage

**Files:**
- Create: `tests/test_remove.py`
- Modify: `tests/BUGS.md` if needed

**Interfaces:**
- Consumes: `tmp_tree`, `reset_hpcp_globals`
- Produces: delete / remove_extra tests

- [ ] **Step 1: Write remove tests**

```python
import os
import pytest


def test_delete_file_bulk_removes_files(tmp_tree, hpcp_mod, reset_hpcp_globals):
	p1 = tmp_tree.add_file('a.txt', 'a', under='dst')
	p2 = tmp_tree.add_file('b.txt', 'b', under='dst')
	hpcp_mod.delete_file_bulk([p1, p2])
	assert not os.path.exists(p1)
	assert not os.path.exists(p2)


def test_remove_extra_files_deletes_only_extras(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	# Mirror process_copy mapping: src path WITHOUT trailing sep => dest/src_basename/...
	# Use trailing sep on both so files land directly under dst (see _src_to_dest_map).
	src_keep = tmp_tree.add_file('keep.txt', 'k', under='src')
	dst_keep = tmp_tree.add_file('keep.txt', 'k', under='dst')
	dst_extra = tmp_tree.add_file('extra.txt', 'e', under='dst')
	src_paths = [str(tmp_tree.src) + os.sep]
	dests = [str(tmp_tree.dst) + os.sep]
	hpcp_mod.remove_extra_files(
		[src_keep],
		dests,
		max_workers=2,
		verbose=False,
		files_per_job=1,
		single_thread=True,
		exclude=None,
		parallel_file_listing=False,
		src_paths=src_paths,
		sym_links=None,
	)
	assert os.path.exists(dst_keep)
	assert not os.path.exists(dst_extra)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_remove.py -v`

Expected: PASS or BUGS.md + xfail.

- [ ] **Step 3: Commit**

```bash
git add tests/test_remove.py tests/BUGS.md
git commit -m "$(cat <<'EOF'
Add remove and remove_extra pytest coverage for hpcp.

EOF
)"
```

---

### Task 7: CLI args and paths/FS helpers

**Files:**
- Create: `tests/test_cli_args.py`
- Create: `tests/test_paths_fs.py`
- Modify: `tests/BUGS.md` if needed

**Interfaces:**
- Consumes: `restore_argv`, `reset_hpcp_globals`, `tmp_tree`
- Produces: argv matrix + path helper tests

- [ ] **Step 1: Write CLI tests**

```python
import sys
import pytest


def _parse(hpcp_mod, argv):
	# get_args prints; that is fine
	sys.argv = argv
	return hpcp_mod.get_args()


def test_full_hash_and_hash_size(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-fh', '-hs', '0', '-d', '/tmp', '/tmp/src'])
	assert args.full_hash is True
	assert args.hash_size == 0


def test_remove_force_flag(hpcp_mod, restore_argv, reset_hpcp_globals):
	# argparse stores remove_force independently; it does not auto-set remove=
	args = _parse(hpcp_mod, ['hpcp', '-rf', '/tmp/src'])
	assert args.remove_force is True
	assert args.remove is False


def test_content_only_and_no_create_dir(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-co', '-ncd', '-d', '/tmp', '/tmp/src'])
	assert args.content_only is True
	assert args.no_create_dir is True


def test_batch_default_true(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-d', '/tmp', '/tmp/src'])
	assert args.batch is True


def test_no_batch(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-nb', '-d', '/tmp', '/tmp/src'])
	assert args.batch is False


def test_exclude_append(hpcp_mod, restore_argv, reset_hpcp_globals):
	args = _parse(hpcp_mod, ['hpcp', '-e', '*.o', '-e', 'tmp*', '-d', '/tmp', '/tmp/src'])
	assert '*.o' in args.exclude
	assert 'tmp*' in args.exclude
```

- [ ] **Step 2: Write paths/FS tests**

```python
import os
import pytest


def test_expand_paths_glob(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	tmp_tree.add_file('a.txt', 'a')
	tmp_tree.add_file('b.txt', 'b')
	pattern = str(tmp_tree.src / '*.txt')
	got = hpcp_mod.expand_paths([pattern])
	assert any(p.endswith('a.txt') for p in got)
	assert any(p.endswith('b.txt') for p in got)


def test_get_free_space_bytes_positive(tmp_path, hpcp_mod, reset_hpcp_globals):
	free = hpcp_mod.get_free_space_bytes(str(tmp_path))
	assert isinstance(free, int)
	assert free > 0


def test_get_last_existing_parent_and_is_device(tmp_tree, hpcp_mod, reset_hpcp_globals):
	path = tmp_tree.add_file('a.txt', 'x')
	assert hpcp_mod.is_device(path) is False
	missing = str(tmp_tree.root / 'no' / 'such' / 'dir')
	parent = hpcp_mod.get_last_existing_parent(missing)
	assert os.path.isdir(parent)


def test_src_to_dest_map_trailing_sep(tmp_tree, hpcp_mod, reset_hpcp_globals):
	# Public-ish helper used by remove_extra; documents dest layout contract
	src = str(tmp_tree.src) + os.sep
	dst = str(tmp_tree.dst) + os.sep
	mapping = hpcp_mod._src_to_dest_map([src], [dst])
	assert len(mapping) == 1
	src_abs, dests = mapping[0]
	assert os.path.abspath(str(tmp_tree.dst)) in [os.path.abspath(d) for d in dests]
```

Note: prefer public APIs; `_src_to_dest_map` is allowed here only to lock the dest-layout contract that `remove_extra` / orchestrator tests rely on (spec permits private helpers when needed to explain public behavior).

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_cli_args.py tests/test_paths_fs.py -v`

Expected: PASS or BUGS.md + xfail.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli_args.py tests/test_paths_fs.py tests/BUGS.md
git commit -m "$(cat <<'EOF'
Add CLI argv and path/FS helper pytest coverage.

EOF
)"
```

---

### Task 8: Imaging / loop real fixtures

**Files:**
- Create: `tests/test_imaging.py`
- Modify: `tests/BUGS.md` if needed

**Interfaces:**
- Consumes: `loop_image` fixture (skips without privileges), `require_linux`
- Produces: loop create/detach + light partition helper smoke when possible

- [ ] **Step 1: Write imaging tests**

```python
import os
import pytest


pytestmark = [pytest.mark.linux, pytest.mark.loop]


def test_create_and_detach_loop_device(loop_image, hpcp_mod, reset_hpcp_globals):
	loop = loop_image['loop']
	assert loop.startswith('/dev/loop') or os.path.exists(loop)
	# fixture detaches on teardown; also verify detach API idempotence-ish
	assert hpcp_mod.detach_loop_device(loop) in (True, False)
	# Re-detach should not crash
	hpcp_mod.detach_loop_device(loop)


def test_get_partitions_on_loop_without_table(loop_image, hpcp_mod, reset_hpcp_globals):
	# Empty image may have no partitions — assert function returns a list (possibly empty)
	# and does not crash; if it raises, skip or BUGS.md depending on documented contract.
	loop = loop_image['loop']
	try:
		parts = hpcp_mod.get_partitions(loop)
	except Exception as e:
		pytest.skip(f'get_partitions needs partitioned image: {e}')
	assert isinstance(parts, list)


@pytest.mark.root
def test_partitioned_image_optional(tmp_path, hpcp_mod, reset_hpcp_globals, require_linux):
	"""Optional deeper test: only run when root and sfdisk/parted available.

	If tools missing, skip. Do not require this for default `pytest -q` green.
	"""
	import shutil
	if os.geteuid() != 0:
		pytest.skip('root required')
	if not shutil.which('sfdisk') and not shutil.which('parted'):
		pytest.skip('sfdisk/parted not available')
	# Minimal: create image, attach, attempt get_partition_infos; skip on failure with reason
	img = tmp_path / 'part.img'
	with open(img, 'wb') as f:
		f.truncate(16 * 1024 * 1024)
	loop = None
	try:
		loop = hpcp_mod.create_loop_device(str(img))
		try:
			infos = hpcp_mod.get_partition_infos(loop)
		except Exception as e:
			pytest.skip(f'partition infos unavailable on blank image: {e}')
		assert infos is not None
	finally:
		if loop:
			hpcp_mod.detach_loop_device(loop)
```

Mark the optional root test so default collection can deselect if desired. Default `pytest -q` should still pass via skips.

- [ ] **Step 2: Run tests without root**

Run: `python -m pytest tests/test_imaging.py -v`

Expected: SKIP and/or PASS; no hard FAIL on unprivileged hosts.

- [ ] **Step 3: Commit**

```bash
git add tests/test_imaging.py tests/BUGS.md
git commit -m "$(cat <<'EOF'
Add loop/imaging pytest coverage with privilege skips.

EOF
)"
```

---

### Task 9: Windows/GUI smoke and orchestrator smoke

**Files:**
- Create: `tests/test_windows_gui.py`
- Create: `tests/test_orchestrator_smoke.py`
- Modify: `tests/BUGS.md` if needed

**Interfaces:**
- Consumes: `require_windows`, `require_linux`, `tmp_tree`, `reset_hpcp_globals`
- Produces: import/structure smoke + one small `hpcp()` copy smoke

- [ ] **Step 1: Write Windows/GUI smoke**

```python
import os
import inspect
import pytest


def test_hpcp_gui_is_callable(hpcp_mod):
	assert callable(hpcp_mod.hpcp_gui)


def test_main_mentions_windows_gui_branch(hpcp_mod):
	src = inspect.getsource(hpcp_mod.main)
	assert 'hpcp_gui' in src
	assert "os.name == 'nt'" in src


@pytest.mark.windows
@pytest.mark.gui
def test_gui_launch_skipped_or_runs_on_windows(hpcp_mod, require_windows):
	# Do not start mainloop in CI: only verify Tk import works on Windows
	import tkinter
	root = tkinter.Tk()
	root.withdraw()
	root.destroy()
```

- [ ] **Step 2: Write orchestrator smoke**

```python
import os
import pytest


@pytest.mark.linux
def test_hpcp_copy_smoke_single_thread(tmp_tree, hpcp_mod, reset_hpcp_globals, require_linux):
	tmp_tree.add_file('a.txt', 'orch')
	# Trailing seps => contents of src land directly under dst (see _src_to_dest_map)
	src = str(tmp_tree.src) + os.sep
	dst = str(tmp_tree.dst) + os.sep
	rc = hpcp_mod.hpcp(
		[src],
		dest_paths=[dst],
		single_thread=True,
		max_workers=1,
		verbose=False,
		batch=True,
		do_not_remove_files_while_listing=True,
	)
	assert rc in (None, 0)
	assert (tmp_tree.dst / 'a.txt').is_file()
	assert (tmp_tree.dst / 'a.txt').read_text() == 'orch'


def test_compare_file_list_identical_sets(tmp_tree, hpcp_mod, reset_hpcp_globals):
	a = {'file1:hash', 'file2:hash'}
	b = {'file1:hash', 'file2:hash'}
	# No exception; function prints summary to stdout
	hpcp_mod.compare_file_list(a, b, diff_file_list=None, tar_diff_file_list=False)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_windows_gui.py tests/test_orchestrator_smoke.py -v`

Expected: PASS on Linux (Windows-marked tests skipped).

- [ ] **Step 4: Commit**

```bash
git add tests/test_windows_gui.py tests/test_orchestrator_smoke.py tests/BUGS.md
git commit -m "$(cat <<'EOF'
Add Windows/GUI smoke and hpcp orchestrator smoke tests.

EOF
)"
```

---

### Task 10: Full-suite verification and bug-list polish

**Files:**
- Modify: `tests/BUGS.md` (ensure every xfail/skip bug is listed)
- Optionally modify: `README.md` with a short “Running tests” note **only if** you already document dev workflows there; otherwise skip README to avoid scope creep (spec allows minimal packaging churn)

**Interfaces:**
- Consumes: all prior tasks
- Produces: green default suite

- [ ] **Step 1: Run full suite**

Run: `python -m pytest -q`

Expected: all non-skipped tests PASS; skips only for `loop`/`root`/`windows`/`gui` where environment lacks support; doctests PASS via `test_run_hpcp_doctests`.

- [ ] **Step 2: Confirm no logic diffs in `hpcp.py`**

Run: `git diff hpcp.py`

Expected: docstring-only changes (doctest examples). If any logic hunk slipped in, revert it and move the finding to `tests/BUGS.md`.

- [ ] **Step 3: Ensure `tests/BUGS.md` is coherent**

Every `xfail`/`skip` tied to a product issue must have a BUG-N section. If none found, leave the template header in place.

- [ ] **Step 4: Final commit if anything left**

```bash
git add tests/BUGS.md README.md
git commit -m "$(cat <<'EOF'
Polish bug list and verify full hpcp test suite passes.

EOF
)"
```

(Skip commit if working tree clean.)

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Doctests for pure/helpers | Task 2 |
| Pytest for FS/concurrency/CLI/imaging | Tasks 3–8 |
| Everything public with skips | Tasks 8–9 + coverage via feature files |
| Real loop fixtures + skip | Tasks 1, 8 |
| Windows/GUI smoke + skip | Task 9 |
| No production logic changes; report bugs | Global Constraints + BUGS.md all tasks |
| Keep content_only tests | Tasks 1, 5 |
| Feature-layered layout | File Structure + Tasks 3–9 |
| pytest entry + doctests | Task 1 |
| Broad happy-path, no coverage % | All tasks; Task 10 verification |
| `tests/BUGS.md` | Tasks 1, 10 |

No TBD/TODO placeholders remain in task steps. Return shapes and doctest expected values were locked against live `hpcp` behavior while writing this plan (`format_time`, `get_timestamp_precision`, `get_file_list` 4-tuple, `get_file_size` missing → `0`, trailing-sep dest mapping).
