# Content-only copy (`-co`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in `-co` / `--content_only` so hpcp copies file contents and still creates directories, but never syncs mode, owner, or timestamps.

**Architecture:** Mirror the existing `NO_CREATE_DIR` pattern: module global `CONTENT_ONLY`, set from argparse via `hpcp(...)`, then branch in `copy_file` and `sync_directory_metadata`. Posix new copies use `cp -f` instead of `cp -af`; non-posix uses `shutil.copy` instead of `copy2`.

**Tech Stack:** Python 3 single-file module (`hpcp.py`), `argparse`, `cp` on posix, `pytest` for new tests against temp dirs.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-31-content-only-copy-design.md`
- Stay single-file for runtime code; tests may live under `tests/`
- Do not change `-nds` / `-ncd` semantics
- Do not bump package version / release unless asked
- Default (no `-co`) must preserve today’s full metadata sync
- `-co` must still create missing destination directories (unlike `-ncd`)
- Tabs for indentation in `hpcp.py` (match existing file style)

## File Structure

| File | Role |
|---|---|
| `hpcp.py` | Global, CLI, `hpcp()` param wiring, `copy_file`, `sync_directory_metadata` |
| `tests/test_content_only.py` | New pytest coverage for content-only behavior |
| `README.md` | Document `-co` in the flags listing |

---

### Task 1: Flag plumbing (`CONTENT_ONLY` + CLI + `hpcp()` wiring)

**Files:**
- Modify: `hpcp.py` (global near line 137, `get_args` near `-ncd`, `hpcp()` signature/body, recursive `hpcp(` call ~3777, `main()` call ~4032)
- Create: `tests/test_content_only.py`
- Test: `tests/test_content_only.py`

**Interfaces:**
- Consumes: existing `NO_CREATE_DIR` wiring pattern
- Produces: module global `CONTENT_ONLY: bool` (default `False`); `hpcp(..., content_only: bool = False)`; argparse dest `content_only`

- [ ] **Step 1: Write the failing test for CLI flag**

Create `tests/test_content_only.py`:

```python
import os
import sys

import pytest

# Import sibling module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hpcp


def test_content_only_flag_parses():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-co', '-d', '/tmp', '/tmp/src']
		args = hpcp.get_args()
		assert args.content_only is True
	finally:
		sys.argv = old


def test_content_only_default_off():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-d', '/tmp', '/tmp/src']
		args = hpcp.get_args()
		assert args.content_only is False
	finally:
		sys.argv = old
```

If `get_args()` fails for unrelated parse reasons, keep argv shaped like real CLI usage (`-d DEST SRC`) and assert only on `content_only`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_content_only.py::test_content_only_flag_parses tests/test_content_only.py::test_content_only_default_off -v`

Expected: FAIL because `content_only` is missing on the namespace, or `-co` is an unrecognized argument.

- [ ] **Step 3: Add global and argparse**

In `hpcp.py`, next to `NO_CREATE_DIR`:

```python
NO_CREATE_DIR = False
CONTENT_ONLY = False
ERRORS = []
```

In `get_args()`, immediately after the `-ncd` argument definition:

```python
parser.add_argument('-ncd','--no_create_dir', action='store_true', help='Ignore any destination folder that does not already exist. ( Will still copy if dest is a file )')
parser.add_argument('-co','--content_only', action='store_true', help='Content-only copy: do not sync mode, owner, or timestamps on files or directories. Still create missing destination directories using filesystem-default permissions (so parent ACL/setgid/setuid inherit).')
```

- [ ] **Step 4: Wire through `hpcp()` and `main()`**

Add parameter to `hpcp(...)` signature next to `no_create_dir`:

```python
no_create_dir = False, content_only = False, command_timeout_limit = 0,
```

Inside `hpcp()`, with the other globals:

```python
global NO_CREATE_DIR
global CONTENT_ONLY
global REMOVE_FILES_WHILE_LISTING
NO_CREATE_DIR = no_create_dir
CONTENT_ONLY = content_only
```

Pass through the recursive dd-mode `hpcp(` call (~3777):

```python
target_file_system = target_file_system, no_create_dir = no_create_dir, content_only = content_only, command_timeout_limit = command_timeout_limit,
```

In `main()`:

```python
target_file_system = args.target_file_system, no_create_dir = args.no_create_dir, content_only = args.content_only, command_timeout_limit = args.command_timeout_limit,
```

- [ ] **Step 5: Run tests and make sure they pass**

Run: `python -m pytest tests/test_content_only.py::test_content_only_flag_parses tests/test_content_only.py::test_content_only_default_off -v`

Expected: PASS

Also smoke-check: `python3 hpcp.py -h` shows `-co, --content_only`.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_content_only.py
git commit -m "$(cat <<'EOF'
Add -co/--content_only flag plumbing.

Wire CONTENT_ONLY global through argparse and hpcp(); behavior changes come next.
EOF
)"
```

---

### Task 2: Content-only file copy (`copy_file`)

**Files:**
- Modify: `hpcp.py` (`copy_file`, ~1800–1936)
- Modify: `tests/test_content_only.py`
- Test: `tests/test_content_only.py`

**Interfaces:**
- Consumes: `hpcp.CONTENT_ONLY`
- Produces: when `CONTENT_ONLY` is True — identical files left untouched; new copies via `cp -f` / `shutil.copy` (no mode/owner/times from source)

- [ ] **Step 1: Write failing behavioral tests**

Append to `tests/test_content_only.py`:

```python
import stat
import tempfile
import time


def _unique_mode_pair():
	# Prefer modes that differ from typical umask results
	return 0o640, 0o600


def test_content_only_new_file_skips_mode_and_mtime():
	src_mode, _ = _unique_mode_pair()
	with tempfile.TemporaryDirectory() as tmp:
		src = os.path.join(tmp, 'src')
		dst_root = os.path.join(tmp, 'dst')
		os.makedirs(src)
		os.makedirs(dst_root)
		src_file = os.path.join(src, 'a.txt')
		dst_file = os.path.join(dst_root, 'a.txt')
		with open(src_file, 'w') as f:
			f.write('hello-content-only')
		os.chmod(src_file, src_mode)
		old_mtime = time.time() - 86400
		os.utime(src_file, (old_mtime, old_mtime))

		hpcp.CONTENT_ONLY = True
		hpcp.NO_CREATE_DIR = False
		try:
			size, _, _ = hpcp.copy_file(src_file, [dst_file])
		finally:
			hpcp.CONTENT_ONLY = False

		assert size > 0
		with open(dst_file) as f:
			assert f.read() == 'hello-content-only'
		st_src = os.stat(src_file)
		st_dst = os.stat(dst_file)
		assert st_dst.st_mode != st_src.st_mode or st_dst.st_mtime != st_src.st_mtime
		# Stronger check: mode must not be forced to source mode
		assert stat.S_IMODE(st_dst.st_mode) != stat.S_IMODE(st_src.st_mode) or abs(st_dst.st_mtime - st_src.st_mtime) > 1


def test_content_only_identical_file_leaves_dest_metadata():
	with tempfile.TemporaryDirectory() as tmp:
		src = os.path.join(tmp, 'src')
		dst_root = os.path.join(tmp, 'dst')
		os.makedirs(src)
		os.makedirs(dst_root)
		src_file = os.path.join(src, 'a.txt')
		dst_file = os.path.join(dst_root, 'a.txt')
		payload = 'same-bytes'
		for path in (src_file, dst_file):
			with open(path, 'w') as f:
				f.write(payload)
		os.chmod(src_file, 0o640)
		os.chmod(dst_file, 0o600)
		src_mtime = time.time() - 86400
		dst_mtime = time.time() - 3600
		os.utime(src_file, (src_mtime, src_mtime))
		os.utime(dst_file, (dst_mtime, dst_mtime))
		before = os.stat(dst_file)

		hpcp.CONTENT_ONLY = True
		try:
			size, _, _ = hpcp.copy_file(src_file, [dst_file], full_hash=True)
		finally:
			hpcp.CONTENT_ONLY = False

		after = os.stat(dst_file)
		assert size == 0  # identical short-circuit returns 0 copied size
		assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
		assert int(after.st_mtime) == int(before.st_mtime)


def test_default_identical_file_still_syncs_metadata():
	"""Sanity: without CONTENT_ONLY, identical path still applies metadata."""
	with tempfile.TemporaryDirectory() as tmp:
		src_file = os.path.join(tmp, 'a.txt')
		dst_file = os.path.join(tmp, 'b.txt')
		for path in (src_file, dst_file):
			with open(path, 'w') as f:
				f.write('same')
		os.chmod(src_file, 0o640)
		os.chmod(dst_file, 0o600)
		t = time.time() - 86400
		os.utime(src_file, (t, t))
		os.utime(dst_file, (time.time() - 10, time.time() - 10))

		hpcp.CONTENT_ONLY = False
		hpcp.copy_file(src_file, [dst_file], full_hash=True)
		st_src = os.stat(src_file)
		st_dst = os.stat(dst_file)
		assert stat.S_IMODE(st_dst.st_mode) == stat.S_IMODE(st_src.st_mode)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_content_only.py::test_content_only_new_file_skips_mode_and_mtime tests/test_content_only.py::test_content_only_identical_file_leaves_dest_metadata -v`

Expected: FAIL — current code uses `cp -af` / metadata sync, so dest mode/mtime match source (identical path) or new file gets source mode.

- [ ] **Step 3: Implement `copy_file` branches**

At the top of `copy_file`, ensure the global is visible:

```python
global RANDOM_DESTINATION_SELECTION
global NO_CREATE_DIR
global CONTENT_ONLY
```

Replace the identical-file metadata block (~1827–1836) with:

```python
if os.path.exists(dest) and (not os.path.islink(src_path)) and is_file_identical(src_path, dest,src_size,full_hash):
	if not CONTENT_ONLY:
		st = os.stat(src_path,follow_symlinks=False)
		shutil.copystat(src_path, dest,follow_symlinks=False)
		if os.name == 'posix':
			os.chown(dest, st.st_uid, st.st_gid)
			os.chmod(dest, st.st_mode)
		os.utime(dest, (st.st_atime, st.st_mtime),follow_symlinks=False)
	endTime = time.monotonic()
	return 0, endTime - start_time , symLinks
```

Replace the posix / non-posix copy commands (~1882–1886) with:

```python
if os.name == 'posix':
	cp_flags = ["-f"] if CONTENT_ONLY else ["-af"]
	run_command_in_multicmd_with_path_check(["cp", *cp_flags, "--sparse=always", src_path, dest_path],quiet=True,strict=True)
	copiedSize = get_file_size(dest_path)
else:
	if CONTENT_ONLY:
		shutil.copy(src_path, dest_path, follow_symlinks=False)
	else:
		shutil.copy2(src_path, dest_path, follow_symlinks=False)
	copied = True
	break
```

Replace the non-sparse fallback `cp -af` (~1900) with:

```python
if os.name == 'posix':
	cp_flags = ["-f"] if CONTENT_ONLY else ["-af"]
	run_command_in_multicmd_with_path_check(["cp", *cp_flags, src_path, dest_path],quiet=True,strict=True)
```

Leave the Windows `xcopy` fallback as-is (spec non-goal).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_content_only.py -v`

Expected: PASS for Task 1 + Task 2 tests. If `test_content_only_new_file_skips_mode_and_mtime` flakes because umask happens to equal `0o640`, change source mode to `0o707` or assert only that `st_mtime` was not forced from source.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_content_only.py
git commit -m "$(cat <<'EOF'
Skip file metadata sync when -co/--content_only is set.

Use cp -f / shutil.copy for new files and leave identical destinations untouched.
EOF
)"
```

---

### Task 3: Content-only directory metadata (`sync_directory_metadata`)

**Files:**
- Modify: `hpcp.py` (`sync_directory_metadata`, ~2439–2461)
- Modify: `tests/test_content_only.py`
- Test: `tests/test_content_only.py`

**Interfaces:**
- Consumes: `hpcp.CONTENT_ONLY`
- Produces: dirs still created via `makedirs`; no `copystat`/`chown`/`utime` when `CONTENT_ONLY`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_content_only_creates_dir_without_source_metadata():
	with tempfile.TemporaryDirectory() as tmp:
		src_dir = os.path.join(tmp, 'src', 'sub')
		dst_dir = os.path.join(tmp, 'dst', 'sub')
		os.makedirs(src_dir)
		os.chmod(src_dir, 0o750)
		t = time.time() - 86400
		os.utime(src_dir, (t, t))

		hpcp.CONTENT_ONLY = True
		try:
			count, _, _ = hpcp.sync_directory_metadata(src_dir, [dst_dir])
		finally:
			hpcp.CONTENT_ONLY = False

		assert count == 1
		assert os.path.isdir(dst_dir)
		st_src = os.stat(src_dir)
		st_dst = os.stat(dst_dir)
		assert stat.S_IMODE(st_dst.st_mode) != stat.S_IMODE(st_src.st_mode) or abs(st_dst.st_mtime - st_src.st_mtime) > 1


def test_content_only_setgid_inheritance():
	"""New dir under -co should inherit parent setgid when FS supports it."""
	with tempfile.TemporaryDirectory() as tmp:
		parent = os.path.join(tmp, 'dst')
		os.makedirs(parent)
		mode = os.stat(parent).st_mode | stat.S_ISGID
		os.chmod(parent, mode)
		if not (os.stat(parent).st_mode & stat.S_ISGID):
			pytest.skip('filesystem does not honor setgid on directories')

		src_dir = os.path.join(tmp, 'src', 'child')
		dst_dir = os.path.join(parent, 'child')
		os.makedirs(src_dir)
		# Source deliberately without setgid
		os.chmod(src_dir, 0o755)

		hpcp.CONTENT_ONLY = True
		try:
			hpcp.sync_directory_metadata(src_dir, [dst_dir])
		finally:
			hpcp.CONTENT_ONLY = False

		assert os.stat(dst_dir).st_mode & stat.S_ISGID


def test_default_dir_sync_still_applies_metadata():
	with tempfile.TemporaryDirectory() as tmp:
		src_dir = os.path.join(tmp, 'src', 'sub')
		dst_dir = os.path.join(tmp, 'dst', 'sub')
		os.makedirs(src_dir)
		os.chmod(src_dir, 0o750)
		t = time.time() - 86400
		os.utime(src_dir, (t, t))

		hpcp.CONTENT_ONLY = False
		hpcp.sync_directory_metadata(src_dir, [dst_dir])
		st_src = os.stat(src_dir)
		st_dst = os.stat(dst_dir)
		assert abs(st_dst.st_mtime - st_src.st_mtime) < 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_content_only.py::test_content_only_creates_dir_without_source_metadata tests/test_content_only.py::test_content_only_setgid_inheritance -v`

Expected: FAIL — current `sync_directory_metadata` applies source mode/mtime (and may clear setgid inheritance by chmod via copystat).

- [ ] **Step 3: Implement directory skip**

In `sync_directory_metadata`:

```python
def sync_directory_metadata(src_path, dest_paths):
	global CONTENT_ONLY
	dest_paths = _validate_paths(src_path, dest_paths)
	if dest_paths is None:
		return 0, 0 , set(), frozenset()
	start_time = time.monotonic()
	if os.path.islink(src_path):
		return 0,time.monotonic()-start_time,{src_path: dest_paths}
	if not os.path.isdir(src_path):
		return copy_file(src_path, dest_paths)
	st = os.stat(src_path)
	for dest_path in dest_paths:
		try:
			if not (os.path.exists(dest_path) or os.path.ismount(dest_path)):
				os.makedirs(dest_path, exist_ok=True)
		except FileExistsError:
			eprint(f"Destination path {dest_path} maybe a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...")
		if not CONTENT_ONLY:
			shutil.copystat(src_path, dest_path)
			if os.name == 'posix':
				os.chown(dest_path, st.st_uid, st.st_gid)
			os.utime(dest_path, (st.st_atime, st.st_mtime))
	return 1,time.monotonic()-start_time,frozenset()
```

- [ ] **Step 4: Run full content-only suite**

Run: `python -m pytest tests/test_content_only.py -v`

Expected: PASS (setgid test may skip on filesystems that ignore setgid — that is OK).

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_content_only.py
git commit -m "$(cat <<'EOF'
Skip directory metadata sync under -co/--content_only.

Still create destination dirs so ACL/setgid/setuid inherit from the parent.
EOF
)"
```

---

### Task 4: README + end-to-end smoke

**Files:**
- Modify: `README.md` (usage synopsis ~line 78 and options near `-ncd` ~167)
- Test: manual CLI smoke against temp dirs

**Interfaces:**
- Consumes: finished `-co` behavior from Tasks 1–3
- Produces: documented flag; verified CLI path

- [ ] **Step 1: Update README**

In the synopsis line that lists flags, add `[-co]` near `[-ncd]`.

After the `-ncd` help block, add:

```text
  -co, --content_only   Content-only copy: do not sync mode, owner, or
                        timestamps on files or directories. Still create
                        missing destination directories using filesystem-
                        default permissions (so parent ACL/setgid/setuid
                        inherit).
```

- [ ] **Step 2: End-to-end smoke (CLI)**

```bash
TMP=$(mktemp -d)
mkdir -p "$TMP/src/sub" "$TMP/dst"
echo payload > "$TMP/src/sub/file.txt"
chmod 640 "$TMP/src/sub/file.txt"
chmod 750 "$TMP/src/sub"
# parent setgid on dest
chmod g+s "$TMP/dst"

python3 hpcp.py -co -s "$TMP/src" -d "$TMP/dst"
# or: python3 hpcp.py -co -d "$TMP/dst" "$TMP/src"

# content present
test -f "$TMP/dst/sub/file.txt"
grep -q payload "$TMP/dst/sub/file.txt"
# mode not forced from source (likely differs)
stat -c '%a' "$TMP/src/sub/file.txt"
stat -c '%a' "$TMP/dst/sub/file.txt"
# setgid inherited on new dir when FS supports it
stat -c '%A' "$TMP/dst/sub" | grep -q s || echo "setgid not visible (FS may not support); dir still created"
test -d "$TMP/dst/sub"
rm -rf "$TMP"
```

Expected: file content copied; destination directory exists; source modes not forced onto dest; setgid often present on `dst/sub`.

- [ ] **Step 3: Re-run unit tests**

Run: `python -m pytest tests/test_content_only.py -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Document -co/--content_only in README.
EOF
)"
```

---

## Self-Review (plan vs spec)

| Spec requirement | Task |
|---|---|
| `-co` / `--content_only` opt-in CLI | Task 1 |
| New file copy without mode/owner/times (`cp -f` / `shutil.copy`) | Task 2 |
| Non-sparse fallback also drops `-a` | Task 2 |
| Identical file: leave dest untouched | Task 2 |
| Still create dirs | Task 3 |
| Skip dir `copystat`/`chown`/`utime` | Task 3 |
| setgid/ACL inherit from parent | Task 3 test + Task 4 smoke |
| Default preserve unchanged | Tasks 2–3 default tests |
| Orthogonal to `-ncd`/`-nds` (no semantic change) | No changes to those flags |
| README | Task 4 |
| No version bump | Not in plan |

No placeholders remaining. Names consistently use `CONTENT_ONLY` / `content_only` / `-co` / `--content_only`.
