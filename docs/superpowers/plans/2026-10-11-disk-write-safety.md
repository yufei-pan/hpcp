# Disk-Write Safety and FAT Full Hashes (group B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** close the B1 gaps. Forced removals stop at mounts. A skipped row that another write overwrites through a symlink is reported. Raw writes and identity changes happen only on partitions hpcp holds exclusively, on a locked disk, with a read-back that reaches the device. FAT files are hashed in full, so the timezone tolerance works for every ESP file. Release target: hpcp 9.69.

**Architecture:** all production code stays in `hpcp.py`.
- **Apply engine:** `_apply_rows`, `_apply_section` and `_plan_section` gain a mount-aware delete walk and a re-check of skipped rows by identity.
- **Disk write phase:** `apply_disk_patch` gains an up-front `O_EXCL` check, a hold context manager (`flock` on the disk plus `O_EXCL` on the written partitions), and `O_DIRECT` read-backs in `_hash_area`.
- **FAT hashing:** generation marks full hashes of FAT files with an `F` prefix and writes `format_version` 3. Apply hashes marked rows in full. The timezone tolerance becomes an explicit `tz_tolerant` flag set only for FAT.

**Tech Stack:** Python 3.6+ (single file, tabs), pytest; Linux block-device ioctls (`BLKSSZGET`, `BLKFLSBUF`, `BLKRRPART`), `fcntl.flock`, `O_EXCL` / `O_DIRECT`; losetup, sgdisk, mkfs.* and mount in root tests.

**Spec:** `docs/superpowers/specs/2026-10-11-disk-write-safety-design.md` (approved, commit af8ba62)

## Global Constraints

- Single file: all production code goes in `hpcp.py`; no new modules or packages.
- Tabs for indentation, in `hpcp.py` and in tests.
- Python 3.6 syntax floor: no walrus, no `str.removeprefix`, no `dict | dict`, no `f"{x=}"`, no `namedtuple(..., defaults=)`, no `os.preadv`.
- No new runtime dependencies. `fcntl` and `mmap` are imported inside the functions that need them (hpcp also runs on Windows, where `fcntl` does not exist).
- `eprint("Category: detail")` only for real failures (it records into `ERRORS`; the exit code comes from the category, see `ERROR_TO_RETURNCODE_TABLE`). Warnings and notes use `print(..., file=sys.stderr)` / `print(...)`.
- Patches from 9.65–9.68 (format 1 / 2, unmarked hashes) must apply exactly as before. A new patch without FAT files keeps format 1 (single filesystem) or 2 (disk).
- Full hashing and the timezone tolerance apply to FAT only (`_FAT_TYPES`: fat32, fat16, fat12, fat, vfat, msdos), never exFAT or NTFS.
- Root / loop tests: scratch files only under the test's `tmp_path`; never write to real filesystem paths; leave nothing mounted or attached (unmount / detach in `finally`).
- **Probe safety rule:** any ad-hoc root probe goes in a script file under `/tmp/claude-0` and runs with `bash <file>`. In this harness, `set -euo pipefail` typed at the top of a shell command is ignored; it only works in a script file or a `bash -c` child. Every path variable gets a `: "${VAR:?}"` guard and a prefix check (`case "$D" in /tmp/claude-0/*) ;; *) exit 1;; esac`). `md5sum /usr/lib/os-release` must stay `776b0aac9b17d49a0ac879da9801f51b`.
- Version target 9.69, bumped only in the final task.
- Run tests from the repository root: `python -m pytest tests/ -q` (≈ 2.5 min, root, loop devices; at the start of this plan: 581 passed, 1 skipped — a Windows GUI test).
- Commit messages end with the Co-Authored-By trailer the implementing agent's harness specifies.

## Review Focus

1. A multi-scope apply (mounts below the target) with the new skipped-row re-check must end exactly like the same patch applied as one scope. Covered: Task 2 runs the alias tests on one scope and on two.
2. A mount that appears after the mount table was read must still not be entered by the delete walk (the `st_dev` backstop). Covered: Task 1 unit test with a device number that differs.
3. Holds and the disk lock must be released on every error path, including an exception raised inside the write phase. Covered: Task 3 test that raises inside the hold, then checks an `O_EXCL` open and `flock -n` both succeed.
4. A disk patch whose only changes are on ext4 must stay format 2, readable by 9.68. Covered: Task 6 test of `make_disk_patch_meta` without marks, plus the existing disk round-trip asserting `format_version == '2'`.
5. Direct reads must hash exactly the requested bytes at awkward edges: the 440-byte boot code on a 4096-byte-sector device, and an area whose end is not a multiple of the read chunk. Covered: Task 4 tests.

---

## File Structure

| File | Responsibility |
|---|---|
| `hpcp.py` (modify) | `_remove_tree_around_mounts`, `_mounts_below`, `_apply_rows.remove`, `_resolve_section_conflicts` notes (Task 1); `SectionPlan.skipped_identities`, `_plan_section`, `_apply_section` (Task 2); `DiskHoldError`, `_exclusive_open`, `_exclusive_refusals`, `_write_phase_devices`, `_disk_write_hold`, `_reread_partition_table`, `_run_identity_commands`, `_TOOLS_NEEDING_EXCLUSIVE`, `apply_disk_patch` write phase (Task 3); `_logical_block_size`, `_hash_area_direct`, `_drop_cached_pages`, `_hash_area(direct=)` (Task 4); `_FULL_HASH_MARK`, `_full_marked`, `_is_fat`, `get_entry_state(mark_full=)`, `_states_for_paths(marked=)`, `states_equal(tz_tolerant=)`, `check_patch_target` / `_verify_rows` / `_check_payload` marks, `SectionPlan.tz_tolerant` (Task 5); `compute_changes(fat_paths=)`, `_paths_on_fat`, generation callers, format constants, `make_patch_meta` / `make_disk_patch_meta` (Task 6); version (Task 7) |
| `tests/test_diff_image.py` (modify) | delete-around-mounts, alias re-check, FAT scope tolerance, single-filesystem generation marks, manifest version tests |
| `tests/test_disk_patch.py` (modify) | exclusivity / hold / tool / direct-read / FAT full-hash disk tests; updated timezone tests |
| `tests/test_dd_fs_params.py` (modify) | `test_version_bumped` |
| `README.md` (modify) | patch-image notes |

---

### Task 1: Forced removals delete around mounts

**Files:**
- Modify: `hpcp.py` — add `_remove_tree_around_mounts` and `_mounts_below` just above `def _apply_rows` (≈ line 5140); `_apply_rows` inner `remove` (≈ 5178-5198); `_resolve_section_conflicts` (≈ 5349-5356); `_apply_rows` docstring.
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Produces: `_remove_tree_around_mounts(path: str, mount_points: set|None = None) -> (list[str], list[str])` (kept mount points relative to path, error texts); `_mounts_below(path: str, mount_points: set) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
requires_root = pytest.mark.skipif(os.name == 'nt' or os.geteuid() != 0 or shutil.which('mount') is None,
	reason='needs root and mount')


def test_remove_tree_around_mounts_keeps_mount_points(tmp_path, hpcp_mod):
	"""The walk removes everything on the tree's own filesystem and never enters a directory listed as a mount
	point; the directories above a kept mount stay. Symlinks are unlinked, never followed."""
	top = tmp_path / 'app'
	_write(top / 'a' / 'f', 'x')
	_write(top / 'b' / 'data' / 'keep', 'mounted')
	_write(tmp_path / 'outside' / 'target', 'outside')
	os.symlink(str(tmp_path / 'outside'), str(top / 'link'))
	mount = os.path.realpath(str(top / 'b' / 'data'))
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), {mount})
	assert kept == ['b/data'] and errors == []
	assert (top / 'b' / 'data' / 'keep').read_text() == 'mounted'
	assert not (top / 'a').exists() and not os.path.lexists(str(top / 'link'))
	assert (tmp_path / 'outside' / 'target').read_text() == 'outside'
	assert sorted(os.listdir(str(top))) == ['b'] and os.listdir(str(top / 'b')) == ['data']


def test_remove_tree_around_mounts_leaves_a_mount_point_root_alone(tmp_path, hpcp_mod):
	top = tmp_path / 'esp'
	_write(top / 'EFI' / 'x', 'x')
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), {os.path.realpath(str(top))})
	assert (kept, errors) == (['.'], []) and (top / 'EFI' / 'x').read_text() == 'x'


def test_remove_tree_around_mounts_stops_at_another_device(tmp_path, hpcp_mod, monkeypatch):
	"""Review Focus 2: a mount made after mountinfo was read is still not entered: its directory's st_dev differs."""
	top = tmp_path / 'app'
	_write(top / 'late' / 'keep', 'k')
	_write(top / 'f', 'x')
	late = os.path.realpath(str(top / 'late'))
	real_lstat = os.lstat

	def lstat(path, *args, **kwargs):
		info = real_lstat(path, *args, **kwargs)
		if os.fspath(path) == late:
			values = list(info)
			values[stat.ST_DEV] = info.st_dev + 1
			return os.stat_result(values)
		return info

	monkeypatch.setattr(os, 'lstat', lstat)
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(top), set())
	monkeypatch.undo()
	assert kept == ['late'] and errors == [] and (top / 'late' / 'keep').exists() and not (top / 'f').exists()


def test_mounts_below(tmp_path, hpcp_mod):
	top = os.path.realpath(str(tmp_path / 'app'))
	points = {top + '/data', top + '/x/y', top + 'z', '/elsewhere', top}
	assert hpcp_mod._mounts_below(str(tmp_path / 'app'), points) == ['.', 'data', 'x/y']


@contextlib.contextmanager
def _tmpfs_at(path):
	os.makedirs(str(path), exist_ok=True)
	subprocess.run(['mount', '-t', 'tmpfs', '-o', 'size=4m', 'tmpfs', str(path)], check=True)
	try:
		yield path
	finally:
		subprocess.run(['umount', str(path)], check=False)


@contextlib.contextmanager
def _bind_at(source, path):
	os.makedirs(str(path), exist_ok=True)
	subprocess.run(['mount', '--bind', str(source), str(path)], check=True)
	try:
		yield path
	finally:
		subprocess.run(['umount', str(path)], check=False)


def _app_dropped(hpcp_mod, tmp_path, new, base):
	"""The golden drops app/ (two files) and adds other.txt. Returns the patch volume."""
	_write(base / 'app' / 'one', '1')
	_write(base / 'app' / 'two', '2')
	_write(new / 'other.txt', 'added')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert ('D', 'app/') in _ops(rows)
	return vol


@requires_root
@pytest.mark.root
@pytest.mark.parametrize('kind', ['tmpfs', 'bind'])
def test_forced_removal_stops_at_a_mount(trees, hpcp_mod, tmp_path, capsys, kind):
	"""Spec Part 1 item 1: -fa removes app/ around the filesystem mounted at app/data (a tmpfs, or a bind of a
	directory of the same filesystem): the mounted files survive, the D row fails at verify, the rest applies, and
	a note names the mount before anything changes."""
	new, base = trees
	vol = _app_dropped(hpcp_mod, tmp_path, new, base)
	target = tmp_path / 'target'
	_clone(base, target)
	outside = tmp_path / 'outside'
	_write(outside / 'precious', 'keep me')
	mount = _tmpfs_at(target / 'app' / 'data') if kind == 'tmpfs' else _bind_at(outside, target / 'app' / 'data')
	with mount:
		if kind == 'tmpfs':
			_write(target / 'app' / 'data' / 'precious', 'keep me')
		hpcp_mod.ERRORS.clear()
		summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
		assert (target / 'app' / 'data' / 'precious').read_text() == 'keep me'
		assert os.path.ismount(str(target / 'app' / 'data')) or kind == 'bind'
	out = capsys.readouterr().out
	assert 'Note: forced removal of app/ stops at the mount at app/data; that row will fail verify' in out
	assert summary['failed'] >= 1 and (target / 'other.txt').read_text() == 'added'
	assert not (target / 'app' / 'one').exists() and not (target / 'app' / 'two').exists()
	assert any(error.startswith('Patch verify error: D app/') for error in hpcp_mod.ERRORS)
	assert (outside / 'precious').read_text() == 'keep me'


@requires_root
@pytest.mark.root
def test_forced_type_change_stops_at_a_mount(trees, hpcp_mod, tmp_path):
	"""A forced directory-to-file replacement over a directory holding a mount leaves the mount and fails."""
	new, base = trees
	_write(base / 'app' / 'one', '1')
	_write(new / 'app', 'now a file')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	with _tmpfs_at(target / 'app' / 'data'):
		_write(target / 'app' / 'data' / 'precious', 'keep me')
		hpcp_mod.ERRORS.clear()
		summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
		assert (target / 'app' / 'data' / 'precious').read_text() == 'keep me'
	assert summary['failed'] >= 1 and os.path.isdir(str(target / 'app'))
```

If `contextlib`, `subprocess` or `stat` is not yet imported at the top of `tests/test_diff_image.py`, add it to the imports (the file already imports `contextlib`, `shutil`, `subprocess`; add `import stat` if missing).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "remove_tree_around_mounts or mounts_below or stops_at_a_mount"`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_remove_tree_around_mounts'` / `_mounts_below`; the root tests fail because `shutil.rmtree` empties the mount (`precious` gone) and no note is printed.

- [ ] **Step 3: Implement**

Add just above `def _apply_rows`:

```python
def _mounts_below(path, mount_points):
	"""The mount points at or below path's real path, relative to it ('.' for path itself), sorted."""
	real = os.path.realpath(path)
	inside = real.rstrip(os.sep) + os.sep
	return sorted(os.path.relpath(point, real) for point in mount_points if point == real or point.startswith(inside))

def _remove_tree_around_mounts(path, mount_points=None):
	"""Remove the directory tree at path as shutil.rmtree would, without entering a mount.

	An entry that is a mount point in mountinfo (any entry: bind mounts and hidden mounts too), or a directory whose
	st_dev differs from path's (a mount made after mountinfo was read), is left in place with everything below it,
	and so are the directories above it; path itself too when it is a mount point. Symlinks are unlinked, never
	followed. The walk is iterative and keeps going after an error. Returns (the entries left in place as mount
	points, relative to path; the errors met, as text).
	"""
	if mount_points is None:
		mount_points = {entry['mount_point'] for entry in _mountinfo_entries()}
	top = os.path.realpath(path)
	kept, errors = [], []
	try:
		top_dev = os.lstat(top).st_dev
	except FileNotFoundError:
		return kept, errors
	if top in mount_points:
		return ['.'], errors
	# (directory, its entries already handled): a directory is pushed again after its children, then removed.
	stack = [(top, False)]
	while stack:
		directory, done = stack.pop()
		if done:
			try:
				os.rmdir(directory)
			except FileNotFoundError:
				pass
			except OSError as e:
				# Not empty: a kept mount below it, or an entry whose removal already failed (recorded).
				if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
					errors.append(f'{os.path.relpath(directory, top)}: {e.strerror}')
			continue
		stack.append((directory, True))
		try:
			names = os.listdir(directory)
		except OSError as e:
			errors.append(f'{os.path.relpath(directory, top)}: {e.strerror}')
			continue
		for name in names:
			child = os.path.join(directory, name)
			try:
				info = os.lstat(child)
			except FileNotFoundError:
				continue
			except OSError as e:
				errors.append(f'{os.path.relpath(child, top)}: {e.strerror}')
				continue
			if child in mount_points or (stat.S_ISDIR(info.st_mode) and info.st_dev != top_dev):
				kept.append(os.path.relpath(child, top))
			elif stat.S_ISDIR(info.st_mode):
				stack.append((child, False))
			else:
				try:
					os.unlink(child)
				except FileNotFoundError:
					pass
				except OSError as e:
					errors.append(f'{os.path.relpath(child, top)}: {e.strerror}')
	return sorted(kept), errors
```

In `_apply_rows`, replace the body of the inner `remove` with:

```python
	def remove(row):
		"""Unlink row's target, rmdir it, or (forced rows only) remove its whole tree, never entering a mount."""
		path = guarded(row, removing=True)
		if path is None:
			return
		directory = is_real_dir(path)
		try:
			if directory and row in forced_rows:
				# Recorded before the walk: one that stops partway may already have removed entries, so the
				# repair step must still check the skipped rows.
				removed.append(row.path)
				kept, errors = _remove_tree_around_mounts(path)
				for error in errors:
					eprint(f'Remove directory failed: {row.path}: {error}')
				if kept:
					eprint(f"Remove directory failed: {row.path}: {', '.join(kept[:3])}{' ...' if len(kept) > 3 else ''} "
						'is a mount point; left in place with the directories above it')
				return
			elif directory:
				os.rmdir(path)
			else:
				os.unlink(path)
			removed.append(row.path)
		except FileNotFoundError:
			pass
		except OSError as e:
			eprint(f"Remove {'directory' if directory else 'file'} failed: {row.path}: {e}")
```

In the `_apply_rows` docstring, replace `only a row in forced_rows (a conflict the user forced) may remove a whole tree.` with `only a row in forced_rows (a conflict the user forced) may remove a whole tree, and that removal never enters a mount (see _remove_tree_around_mounts): a tree holding a mount point is left in place and fails verify.`

Replace `_resolve_section_conflicts` with:

```python
def _resolve_section_conflicts(plans, force, extra_lines=()):
	"""One conflict list and one prompt for every section (and extra lines such as raw areas).

	Before them, a note for each conflicting row whose forced removal of a directory would stop at a mount below it
	(see _remove_tree_around_mounts), so the user sees it before confirming.
	"""
	lines = []
	mount_points = None
	for plan in plans:
		conflicts = _section_conflicts(plan)
		where = '' if plan.name == 'root' else f'[{plan.name}] '
		lines += _conflict_lines(conflicts, where)
		for check in conflicts:
			row = check.row
			if not check.found.startswith('d:') or (row.op != 'D' and row.new.startswith('d:')):
				continue
			if mount_points is None:
				mount_points = {entry['mount_point'] for entry in _mountinfo_entries()}
			below = _mounts_below(_key_path(plan.target_root, row.path), mount_points)
			if below:
				first = os.path.normpath(os.path.join(row.path.rstrip('/'), below[0]))
				more = f' (and {len(below) - 1} more)' if len(below) > 1 else ''
				print(f'Note: forced removal of {where}{row.path} stops at the mount at {first}{more}; that row will fail verify')
	lines += list(extra_lines)
	if lines:
		_resolve_conflict_lines(lines, force)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "remove_tree_around_mounts or mounts_below or stops_at_a_mount"` — PASS.
Then: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q` — all pass; `mount | grep -c /tmp/pytest` prints 0.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "fix(diff-image): a forced directory removal never enters a mount; warn before the prompt"
```

---

### Task 2: Skipped rows overwritten through an alias are verified

**Files:**
- Modify: `hpcp.py` — `SectionPlan` (≈ 5317), `_plan_section` (≈ 5338), `_apply_section` (≈ 5371-5407), docstrings of `_apply_section` and `_verify_across_sections`.
- Test: `tests/test_diff_image.py` (new tests; rewrite `test_skipped_rows_whose_names_differ_only_by_case_from_a_write_are_verified`)

**Interfaces:**
- Consumes: `_entry_identity(path)`, `_identities(target_root, rows)` (existing).
- Produces: `SectionPlan` gains the field `skipped_identities` (`{row: (st_dev, st_ino, file type) | None}` for the skipped A / M rows, recorded at planning), placed after `case_insensitive`. Task 5 appends `tz_tolerant` after it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
def _alias_overwrite(hpcp_mod, tmp_path, new, base):
	"""The golden adds y/f with the content base's m/x/f has, and changes m/x/f. On a node where y -> m/x, A y/f is
	skipped (already there) and M m/x/f then rewrites that same file. Returns (volume, make_target)."""
	(new / 'm' / 'x').mkdir(parents=True)
	(base / 'm' / 'x').mkdir(parents=True)
	(base / 'y').mkdir()
	(new / 'y').mkdir()
	_write(base / 'm' / 'x' / 'f', 'old')
	_write(new / 'y' / 'f', 'old')
	_write(new / 'm' / 'x' / 'f', 'new')
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('A', 'y/f'), ('M', 'm/x/f')}

	def make_target(name):
		target = tmp_path / name
		_clone(base, target)
		(target / 'y').rmdir()
		os.symlink(os.path.join('m', 'x'), str(target / 'y'))
		return target

	return vol, make_target


def test_skipped_row_overwritten_through_an_alias_fails_at_verify(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""Spec Part 1 item 2 / Review Focus 1: A y/f is skipped, M m/x/f rewrites the same entry through y -> m/x. The
	skipped row's entry changed, so it is verified and fails, on one scope and across two (m mounted below)."""
	new, base = trees
	vol, make_target = _alias_overwrite(hpcp_mod, tmp_path, new, base)
	summaries = []
	for name, mounts in (('single', []), ('scoped', ['m'])):
		target = make_target(name)
		summaries.append(_apply_on_scopes(hpcp_mod, monkeypatch, vol, target, mounts))
		assert (target / 'm' / 'x' / 'f').read_text() == 'new'
		assert [error for error in hpcp_mod.ERRORS if error.startswith('Patch verify error: A y/f:')]
	assert summaries[0] == summaries[1] and summaries[0]['failed'] == 1


def test_skipped_directory_changed_through_an_alias_fails_at_verify(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""A skipped directory row keeps its inode when another row changes its mode through an alias: skipped
	directories are always verified once something was written."""
	new, base = trees
	for tree in (new, base):
		(tree / 'm' / 'x' / 'd').mkdir(parents=True)
		(tree / 'y').mkdir()
	(new / 'y' / 'd').mkdir()
	os.chmod(str(new / 'y' / 'd'), 0o755)
	os.chmod(str(base / 'm' / 'x' / 'd'), 0o755)
	os.chmod(str(new / 'm' / 'x' / 'd'), 0o700)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert {('A', 'y/d/'), ('M', 'm/x/d/')} <= _ops(rows)
	target = tmp_path / 'target'
	_clone(base, target)
	(target / 'y').rmdir()
	os.symlink(os.path.join('m', 'x'), str(target / 'y'))
	summary = _apply_on_scopes(hpcp_mod, monkeypatch, vol, target, [])
	assert summary['failed'] == 1
	assert [error for error in hpcp_mod.ERRORS if error.startswith('Patch verify error: A y/d/:')]


def test_unchanged_skipped_files_are_not_verified_again(trees, hpcp_mod, tmp_path, monkeypatch):
	"""An ordinary apply: skipped files whose entry did not change are not verified again (no extra hashing); the
	written rows are, and the apply ends failed 0."""
	new, base = trees
	for index in range(20):
		_write(new / f'same{index}', 's')
	_write(new / 'changed', 'new')
	_write(base / 'changed', 'old')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(new, target)
	_write(target / 'changed', 'old')
	verified = []
	real_verify_rows = hpcp_mod._verify_rows

	def verify_rows(rows, *args, **kwargs):
		verified.extend(row.path for row in rows)
		return real_verify_rows(rows, *args, **kwargs)

	monkeypatch.setattr(hpcp_mod, '_verify_rows', verify_rows)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert summary['failed'] == 0 and summary['skipped'] == 20
	assert verified == ['changed']
```

Replace the existing `test_skipped_rows_whose_names_differ_only_by_case_from_a_write_are_verified` with:

```python
def test_skipped_case_siblings_on_a_case_sensitive_target_are_not_verified(trees, hpcp_mod, tmp_path, monkeypatch):
	"""On a case-sensitive target EFI/X and EFI/x are separate files: writing EFI/X leaves EFI/x's entry unchanged, so
	the skipped EFI/x is not verified again, and verify passes. (On vfat the write lands on EFI/x's entry, which then
	fails loudly: test_write_over_a_skipped_case_sibling_on_vfat_fails_at_verify.)"""
	new, base = trees
	_write(new / 'other', 'same')
	vol = _case_siblings(hpcp_mod, tmp_path, new, base)
	target = tmp_path / 'target'
	_clone(base, target)
	_write(target / 'EFI' / 'x', 'old loader')
	shutil.copy2(str(new / 'other'), str(target / 'other'))
	verified = []
	real_verify_rows = hpcp_mod._verify_rows

	def verify_rows(rows, *args, **kwargs):
		verified.extend(row.path for row in rows)
		return real_verify_rows(rows, *args, **kwargs)

	monkeypatch.setattr(hpcp_mod, '_verify_rows', verify_rows)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (summary['modified'], summary['skipped'], summary['failed']) == (1, 2, 0)
	assert verified == ['EFI/X']
	assert (target / 'EFI' / 'X').read_text() == 'NEW loader' and (target / 'EFI' / 'x').read_text() == 'old loader'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "alias_fails_at_verify or not_verified_again or case_sensitive_target_are_not_verified"`
Expected: the two alias tests FAIL (`failed == 0`, no `Patch verify error: A y/...`); `test_unchanged_skipped_files...` may pass already; `test_skipped_case_siblings_on_a_case_sensitive_target...` FAILS (`verified == ['EFI/X', 'EFI/x']`).

- [ ] **Step 3: Implement**

Change `SectionPlan` and its comment to:

```python
# metadata False: a content-only section (see states_equal). case_insensitive: the target looks names
# up case-insensitively (FAT, exFAT, HFS+ but not HFSX; see _has_case_insensitive_names). skipped_identities:
# {row: _entry_identity} of the A / M rows planned as skipped, recorded while planning (see _apply_section).
SectionPlan = namedtuple('SectionPlan', 'name volume_roots target_root rows checks metadata case_insensitive skipped_identities')
```

Replace `_plan_section` with:

```python
def _plan_section(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers, metadata=True, case_insensitive=False):
	"""Check one section's rows against target_root; nothing changes. The entries of the rows found already done are
	recorded, so a later write that reaches one of them (through a symlinked directory, or a name differing only by
	case) is seen."""
	checks = check_patch_target(rows, target_root, compare_hash, full_hash, max_workers, metadata, case_insensitive)
	skipped = [check.row for check in checks if check.action == 'skip' and check.row.op != 'D']
	return SectionPlan(name, list(volume_roots), target_root, rows, checks, metadata, case_insensitive,
		_identities(target_root, skipped))
```

In `_apply_section`, replace the docstring paragraph that begins `Verify also covers each skipped A / M row whose name differs only by case` (three lines) with:

```
	Once anything was written, removed or restored, verify also covers each skipped A / M row whose entry is no longer
	the one planning found (plan.skipped_identities: another row's write reached it through a symlinked directory, or
	through a name differing only by case on a case-insensitive filesystem, recognised or not) and every skipped
	directory row (a directory changed in place keeps its inode). Such a contradictory patch then fails loudly.
```

and replace these lines:

```python
	written = {}
	for row in todo + restored:
		if row.op != 'D':
			written.setdefault(_fold_name(row.path.rstrip('/')), set()).add(row.path.rstrip('/'))
	restored_rows = set(restored)
	siblings = [row for row in skipped if row not in restored_rows
		and written.get(_fold_name(row.path.rstrip('/')), set()) - {row.path.rstrip('/')}]
	verified, passed = todo + restored + siblings, []
	failed = _verify_rows(verified, plan.target_root, compare_hash, full_hash, max_workers, plan.metadata, passed=passed)
	if outcome is not None:
		outcome.update(removed=bool(removed), restored=restored, verified=verified, deletes=[row for row in passed if row.op == 'D'],
			identities=_identities(plan.target_root, passed))
```

with:

```python
	restored_rows = set(restored)
	changed = []
	if todo or restored:
		changed = [row for row in skipped if row not in restored_rows and (row.new.startswith('d:')
			or _entry_identity(_key_path(plan.target_root, row.path)) != plan.skipped_identities.get(row))]
	verified, passed = todo + restored + changed, []
	failed = _verify_rows(verified, plan.target_root, compare_hash, full_hash, max_workers, plan.metadata, passed=passed)
	if outcome is not None:
		# Skipped rows not verified here keep planning's entry, so the final pass across sections sees a later change.
		verified_rows = set(verified)
		identities = {row: plan.skipped_identities.get(row) for row in skipped if row not in verified_rows}
		identities.update(_identities(plan.target_root, passed))
		outcome.update(removed=bool(removed), restored=restored, verified=verified, deletes=[row for row in passed if row.op == 'D'],
			identities=identities)
```

In the `_verify_across_sections` docstring, replace `Every other A / M row that passed is lstat'ed and verified again only when its entry (_entry_identity, recorded right after its verify) changed or is gone.` with `Every other A / M row that passed, or that its section skipped without verifying, is lstat'ed and verified again only when its entry (_entry_identity, recorded right after its verify, or while planning) changed or is gone.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "alias_fails_at_verify or not_verified_again or case_sensitive_target_are_not_verified or case_sibling or two_scopes or restore"` — PASS.
Then: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q` — all pass, including `test_write_over_a_skipped_case_sibling_on_vfat_fails_at_verify` (vfat root test) three runs in a row: `python -m pytest tests/test_diff_image.py -q -k vfat` ×3.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "fix(diff-image): verify skipped rows whose entry another write reached through an alias"
```

---

### Task 3: Exclusive holds and a locked disk during the disk write phase

**Files:**
- Modify: `hpcp.py` — add the helpers below just above `def apply_disk_patch` (≈ 6714); `apply_disk_patch` planning (after `skipped_identity = ...`, ≈ 6788) and write phase (≈ 6827-6851); its docstring.
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: `_fs_identity_commands`, `_identity_command_list`, `_settle_partitions`, `run_command_in_multicmd_with_path_check` (existing). Task 4's `_hash_area(..., direct=True)` is not available yet; this task keeps the existing `_hash_area` calls and Task 4 changes them.
- Produces: `DiskHoldError(OSError)`; `_exclusive_open(device) -> int`; `_exclusive_refusals({number: device}) -> list[str]`; `_write_phase_devices(raw_plan, changes, layout, skipped_identity) -> {number: device}`; `_disk_write_hold(disk, devices, timeout=10)` (context manager yielding `{device: fd or None}`); `_run_identity_commands(commands, holds)`; `_TOOLS_NEEDING_EXCLUSIVE` (frozenset of tool names).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
def test_exclusive_refusals_name_busy_partitions(hpcp_mod, monkeypatch):
	opened = []
	real_open = os.open

	def fake_open(path, flags, *args):
		opened.append((path, bool(flags & os.O_EXCL)))
		if path == '/dev/fake2':
			raise OSError(errno.EBUSY, 'Device or resource busy')
		return real_open(os.devnull, os.O_RDONLY)

	monkeypatch.setattr(hpcp_mod.os, 'open', fake_open)
	refusals = hpcp_mod._exclusive_refusals({1: '/dev/fake1', 2: '/dev/fake2'})
	monkeypatch.undo()
	assert refusals == ['partition 2 is mounted, active swap or held (LVM, md, dm), in this or another mount namespace']
	assert opened == [('/dev/fake1', True), ('/dev/fake2', True)]


def test_write_phase_devices(hpcp_mod):
	IC = hpcp_mod.IdentityChange
	layout = {'parts': {1: {'device': '/dev/p1'}, 2: {'device': '/dev/p2'}, 3: {'device': '/dev/p3'}, 4: {'device': '/dev/p4'}}}
	raw_plan = [{'name': 'part2', 'device': '/dev/p2', 'action': 'apply'}, {'name': 'part4', 'device': '/dev/p4', 'action': 'skip'},
		{'name': 'mbr', 'device': '/dev/disk', 'action': 'apply'}]
	changes = [IC(3, 'fs_label', 'a', 'b'), IC(1, 'name', 'x', 'y'), IC(4, 'fs_uuid', 'u', 'v'), IC(None, 'guid', 'g', 'h')]
	devices = hpcp_mod._write_phase_devices(raw_plan, changes, layout, {(4, 'fs_uuid')})
	assert devices == {2: '/dev/p2', 3: '/dev/p3'}


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_partition_held_by_another_process_is_refused_up_front(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 2 item 1: partition 2 (its raw area changes) is held O_EXCL by another process, which hpcp's own
	mount-table check cannot see: refused before anything changes."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	before = _sha(target.read_bytes())
	with _attached(target) as loop:
		holder = subprocess.Popen([sys.executable, '-c',
			'import os, sys, time; fd = os.open(sys.argv[1], os.O_RDONLY | os.O_EXCL); print("held", flush=True); time.sleep(60)',
			loop + 'p2'], stdout=subprocess.PIPE, text=True)
		try:
			assert holder.stdout.readline().strip() == 'held'
			with pytest.raises(RuntimeError, match='^' + re.escape('Disk in use: partition 2 is mounted, active swap or held (LVM, md, dm), '
					'in this or another mount namespace; nothing applied')):
				_apply(hpcp_mod, patch, loop)
		finally:
			holder.kill()
			holder.wait()
	assert _sha(target.read_bytes()) == before


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
@pytest.mark.skipif(shutil.which('unshare') is None, reason='needs unshare')
def test_partition_mounted_in_another_namespace_is_refused_up_front(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Partition 3 (its label changes) is mounted only inside another mount namespace: hpcp's mountinfo does not list
	it, but the kernel refuses the exclusive open, so the apply is refused before anything changes."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	mount_point = tmp_path / 'elsewhere'
	mount_point.mkdir()
	with _attached(target) as loop:
		child = subprocess.Popen(['unshare', '-m', '--propagation', 'private', 'sh', '-c',
			'mount "$1" "$2" && echo ready && exec sleep 60', 'sh', loop + 'p3', str(mount_point)],
			stdout=subprocess.PIPE, text=True)
		try:
			assert child.stdout.readline().strip() == 'ready'
			assert not os.path.ismount(str(mount_point))
			with pytest.raises(RuntimeError, match=r'^Disk in use: partition 3 is mounted, active swap or held'):
				_apply(hpcp_mod, patch, loop)
		finally:
			child.kill()
			child.wait()


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_write_phase_holds_partitions_and_locks_the_disk(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Spec Part 2 item 2: during the raw writes the disk is flock'ed (flock -n fails) and the written partition is
	held exclusively (another O_EXCL open, as mount would make, fails); afterwards both are free again and the patch
	applied."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	seen = {}
	real_write_area = hpcp_mod._write_area

	def write_area(device, payload, offset, length):
		real_write_area(device, payload, offset, length)
		if device.endswith('p2'):
			seen['flock'] = subprocess.run(['flock', '-n', device[:-2], 'true']).returncode
			try:
				os.close(os.open(device, os.O_RDONLY | os.O_EXCL))
				seen['excl'] = 0
			except OSError as e:
				seen['excl'] = e.errno

	monkeypatch.setattr(hpcp_mod, '_write_area', write_area)
	with _attached(target) as loop:
		summary = _apply(hpcp_mod, patch, loop)
		assert subprocess.run(['flock', '-n', loop, 'true']).returncode == 0
		os.close(os.open(loop + 'p2', os.O_RDONLY | os.O_EXCL))
	assert summary['failed'] == 0 and summary['raw_written'] == 2
	assert seen['flock'] != 0 and seen['excl'] == errno.EBUSY


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_a_hold_lost_before_the_write_phase_stops_raw_areas_and_identity(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Something took partition 2 after the up-front check (simulated): raw areas, boot code and identity changes are
	not applied, the apply fails loudly (Disk in use), and a re-run can finish."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	calls = []
	real_exclusive_open = hpcp_mod._exclusive_open

	def exclusive_open(device):
		calls.append(device)
		if len(calls) > 2 and device.endswith('p2'):
			raise OSError(errno.EBUSY, 'Device or resource busy')
		return real_exclusive_open(device)

	monkeypatch.setattr(hpcp_mod, '_exclusive_open', exclusive_open)
	with _attached(target) as loop:
		with open(loop + 'p2', 'rb') as f:
			raw_before = f.read(1024)
		summary = _apply(hpcp_mod, patch, loop)
		with open(loop + 'p2', 'rb') as f:
			assert f.read(1024) == raw_before
		assert subprocess.run(['flock', '-n', loop, 'true']).returncode == 0
	assert summary['failed'] >= 1 and summary['raw_written'] == 0 and summary['identity_changed'] == 0
	assert any(error.startswith('Disk in use: ') and 'not applied' in error for error in hpcp_mod.ERRORS)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_disk_write_hold_releases_on_an_error_inside(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 3: an exception inside the hold still releases the partition holds and the disk lock."""
	old = tmp_path / 'old.img'
	_make_golden(old)
	with _attached(old) as loop:
		with pytest.raises(ValueError):
			with hpcp_mod._disk_write_hold(loop, [loop + 'p2', loop + 'p3']) as holds:
				assert set(holds) == {loop + 'p2', loop + 'p3'}
				assert subprocess.run(['flock', '-n', loop, 'true']).returncode != 0
				raise ValueError('boom')
		assert subprocess.run(['flock', '-n', loop, 'true']).returncode == 0
		for device in (loop + 'p2', loop + 'p3'):
			os.close(os.open(device, os.O_RDONLY | os.O_EXCL))


_FULL_UUID = '11111111-2222-3333-4444-555555555555'
# (fs_type, mkfs argv, identity tool, UUID as blkid shows it, label)
_IDENTITY_TOOL_CASES = [
	('ext4', ['mkfs.ext4', '-q', '-F'], 'tune2fs', _FULL_UUID, 'newlabel'),
	('xfs', ['mkfs.xfs', '-q', '-f'], 'xfs_admin', _FULL_UUID, 'newlabel'),
	('vfat', ['mkfs.vfat'], 'fatlabel', 'ABCD-1234', 'NEWLABEL'),
	('btrfs', ['mkfs.btrfs', '-q', '-f'], 'btrfstune', _FULL_UUID, 'newlabel'),
	('exfat', ['mkfs.exfat'], 'exfatlabel', 'ABCD-1234', 'newlabel'),
	('swap', ['mkswap'], 'swaplabel', _FULL_UUID, 'newlabel'),
]


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
@pytest.mark.parametrize('fs_type, mkfs, tool, uuid, label', _IDENTITY_TOOL_CASES)
def test_identity_tools_work_while_the_partition_is_held(tmp_path, hpcp_mod, reset_hpcp_globals, fs_type, mkfs, tool, uuid, label):
	"""Spec Part 2 item 2: every identity tool hpcp runs sets the UUID and label of a partition hpcp holds
	exclusively on a locked disk (a tool that cannot is listed in _TOOLS_NEEDING_EXCLUSIVE and runs with the hold
	dropped for its command). Skipped when the filesystem's tools are not installed."""
	if shutil.which(mkfs[0]) is None or shutil.which(tool) is None:
		pytest.skip(f'needs {mkfs[0]} and {tool}')
	image = tmp_path / 'disk.img'
	with open(str(image), 'wb') as f:
		f.truncate(400 << 20)
	_run('sgdisk', '-n1:2048:+300M', image)
	with _attached(image) as loop:
		part = loop + 'p1'
		_run(*(mkfs + [part]))
		commands = hpcp_mod._fs_identity_commands(fs_type, part, uuid=uuid, label=label)
		if commands is None:
			pytest.skip(f'hpcp sets no identity on {fs_type}')
		with hpcp_mod._disk_write_hold(loop, [part]) as holds:
			hpcp_mod._run_identity_commands(commands, holds)
		hpcp_mod._settle_partitions(loop, [1])
		found = subprocess.run(['blkid', '-p', '-o', 'export', part], capture_output=True, text=True).stdout
	assert f'UUID={uuid}' in found and f'LABEL={label}' in found
```

At the top of `tests/test_disk_patch.py`, add `import errno` and `import sys` if missing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "exclusive or write_phase or hold or identity_tools"`
Expected: FAIL — the new helpers do not exist; the held / other-namespace tests apply instead of refusing.

- [ ] **Step 3: Implement**

Add just above `def apply_disk_patch`:

```python
_BLKRRPART = 0x125f
# Identity tools that fail while hpcp holds their partition O_EXCL (found by
# test_identity_tools_work_while_the_partition_is_held); they run with that partition's hold dropped.
_TOOLS_NEEDING_EXCLUSIVE = frozenset()

class DiskHoldError(OSError):
	"""The write phase could not lock the disk or hold a partition (see _disk_write_hold)."""

def _exclusive_open(device):
	"""A read-only O_EXCL descriptor of block device device. Raises OSError (EBUSY) when it is mounted, active swap or
	held (LVM, md, dm, another exclusive opener) in any mount namespace: the kernel answers for all of them."""
	return os.open(device, os.O_RDONLY | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0))

def _exclusive_refusals(devices):
	"""'partition N is ...' for each {number: device} that cannot be opened exclusively right now (opened, then
	closed at once)."""
	refusals = []
	for number, device in sorted(devices.items()):
		try:
			os.close(_exclusive_open(device))
		except OSError as e:
			if e.errno == errno.EBUSY:
				refusals.append(f'partition {number} is mounted, active swap or held (LVM, md, dm), in this or another mount namespace')
			else:
				refusals.append(f'partition {number} cannot be opened exclusively ({e.strerror})')
	return refusals

def _write_phase_devices(raw_plan, changes, layout, skipped_identity):
	"""{number: device} of the partitions the write phase writes: a raw area to write, or a filesystem UUID or label
	to set. GPT fields and the MBR boot code are written through the whole disk, which the write phase locks."""
	devices = {}
	for entry in raw_plan:
		if entry['action'] != 'skip' and entry['name'] != 'mbr':
			devices[int(entry['name'][len('part'):])] = entry['device']
	for change in changes:
		if change.number is not None and change.field in ('fs_uuid', 'fs_label') and (change.number, change.field) not in skipped_identity:
			devices[change.number] = layout['parts'][change.number]['device']
	return devices

def _reread_partition_table(disk_fd, disk):
	"""Ask the kernel to re-read disk's partition table (BLKRRPART on disk_fd; `partx -u` when the kernel refuses, a
	partition being in use). Quiet: udev settling afterwards shows the outcome."""
	try:
		import fcntl
		fcntl.ioctl(disk_fd, _BLKRRPART)
		return
	except (ImportError, OSError):
		pass
	if check_path('partx'):
		error_count = len(ERRORS)
		try:
			run_command_in_multicmd_with_path_check(['partx', '-u', disk], quiet=True, strict=False)
		finally:
			del ERRORS[error_count:]

@contextlib.contextmanager
def _disk_write_hold(disk, devices, timeout=10):
	"""Hold disk for the write phase; yields {device: descriptor} for devices (None once dropped, see
	_run_identity_commands).

	flock(LOCK_EX) on the whole disk (systemd's block-device locking convention: udev postpones its events for a disk
	locked so, and so does not re-read the partition table mid-phase), retried for up to timeout seconds while udev
	holds its shared lock; then an O_EXCL open of every device, kept to the end, so nothing can mount or claim them
	meanwhile. Raises DiskHoldError when the lock or a hold cannot be taken (what was taken is released). On exit the
	holds are closed first (the kernel cannot re-read a table whose partitions are open exclusively), then the
	partition table is re-read, then the lock released.
	"""
	import fcntl
	disk_fd = os.open(disk, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
	holds = {}
	try:
		deadline = time.time() + timeout
		while True:
			try:
				fcntl.flock(disk_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
				break
			except OSError as e:
				if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN) or time.time() >= deadline:
					raise DiskHoldError(e.errno, f'cannot lock {disk} ({e.strerror})')
				time.sleep(0.1)
		for device in devices:
			try:
				holds[device] = _exclusive_open(device)
			except OSError as e:
				raise DiskHoldError(e.errno, f'{device} was taken after the check ({e.strerror})')
		yield holds
	finally:
		for fd in holds.values():
			if fd is not None:
				os.close(fd)
		_reread_partition_table(disk_fd, disk)
		# Closing the descriptor releases the lock.
		os.close(disk_fd)

def _run_identity_commands(commands, holds):
	"""Run the identity commands while the write phase holds their partitions. A tool in _TOOLS_NEEDING_EXCLUSIVE
	opens its device exclusively itself: the hold of every held device its command names is dropped for that one
	command and taken again after it (a warning when that fails; the remaining commands still run)."""
	for command in commands:
		dropped = []
		if command[0] in _TOOLS_NEEDING_EXCLUSIVE:
			for device, fd in list(holds.items()):
				if fd is not None and device in command:
					os.close(fd)
					holds[device] = None
					dropped.append(device)
		run_command_in_multicmd_with_path_check(command, strict=False)
		for device in dropped:
			try:
				holds[device] = _exclusive_open(device)
			except OSError as e:
				print(f'Warning: cannot hold {device} again after {command[0]} ({e.strerror})', file=sys.stderr)
```

In `apply_disk_patch`, right after `skipped_identity = {(number, field) for number, fields, _ in unsupported for field in fields}`, add:

```python
	# The kernel answers for every mount namespace and for hidden mounts, which _partition_in_use cannot see.
	write_devices = _write_phase_devices(raw_plan, changes, layout, skipped_identity)
	exclusive = _exclusive_refusals(write_devices)
	if exclusive:
		raise RuntimeError('Disk in use: ' + '; '.join(exclusive) + '; nothing applied')
```

Replace the write phase, from `pending = [entry for entry in raw_plan if entry['action'] != 'skip'] if stopped is None else []` through the block ending `_settle_partitions(disk, sorted(layout['parts']))` that follows the identity commands, with:

```python
	pending = [entry for entry in raw_plan if entry['action'] != 'skip'] if stopped is None else []
	attempted = []
	raw_stopped = None
	hold_error = None
	if stopped is None and (pending or commands):
		try:
			# The disk is locked (udev leaves it alone) and every partition written here is held exclusively
			# (nothing can mount it) from the first raw write to the last identity command.
			with _disk_write_hold(disk, sorted(write_devices.values())) as holds:
				# Partition areas first, the MBR boot code last. A write that fails stops the rest and the identity
				# changes: boot code pointing at a core image that is not there would leave the node unbootable.
				# So does one that raised no error but does not hold the new content when read back.
				for entry in sorted(pending, key=lambda entry: entry['name'] == 'mbr'):
					attempted.append(entry)
					try:
						_write_area(entry['device'], entry['payload'], 0, entry['size'])
						if _hash_area(entry['device'], 0, entry['size']) != entry['new']:
							entry['error'] = 'does not hold the new content after writing'
					except OSError as e:
						entry['error'] = str(e)
					if entry.get('error'):
						raw_stopped = entry['name']
						break
				if raw_stopped is None:
					_run_identity_commands(commands, holds)
		except DiskHoldError as e:
			hold_error = e.strerror
			failed += 1
			eprint(f'Disk in use: {hold_error}; raw areas, the boot code and identity changes were not applied; fix the cause and re-run')
	if stopped is None:
		# The holds are released and the partition table re-read: wait until udev shows every partition again.
		_settle_partitions(disk, sorted(layout['parts']))
```

Then change `applied = ([change for change in changes if (change.number, change.field) not in skipped_identity] if stopped is None and raw_stopped is None else [])` to use `if stopped is None and raw_stopped is None and hold_error is None else []`.

In the `apply_disk_patch` docstring, after `Everything is checked before anything changes: ... the identity values.` add: `A partition the patch writes raw or sets a UUID / label on must open exclusively (O_EXCL) during planning, or nothing is applied (Disk in use): that also catches mounts in another mount namespace. During the raw writes and identity commands the whole disk is flock'ed and those partitions held exclusively; a hold that cannot be taken then stops the raw areas, boot code and identity changes like a failed section.`

- [ ] **Step 4: Run the tests and fill `_TOOLS_NEEDING_EXCLUSIVE`**

Run: `python -m pytest tests/test_disk_patch.py -q -k "exclusive or write_phase or hold or identity_tools"`.
If a parametrized `test_identity_tools_work_while_the_partition_is_held[...]` fails because its tool refuses the held partition (EBUSY / "in use" in its output), add the tool's name (e.g. `'btrfstune'`; for ext4 the first command is the filesystem check from `_FS_FIX_COMMANDS`, e.g. `'e2fsck'`) to `_TOOLS_NEEDING_EXCLUSIVE`, re-run, and name the listed tools in the commit message. Do not weaken the test. If a case fails for a reason unrelated to the hold — reproduce it without `_disk_write_hold`, e.g. the tool rejects the value's format — mark only that case `pytest.mark.xfail(reason=...)` with the exact tool error, and report it as a pre-existing issue (outside B1).
Then: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q` — all pass; `losetup -a | grep -c /tmp/pytest` and `mount | grep -c /tmp/pytest` print 0.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd-diff): hold written partitions exclusively on a locked disk; refuse partitions in use in any namespace"
```

---

### Task 4: Read-backs that reach the device

**Files:**
- Modify: `hpcp.py` — add `_BLKSSZGET`, `_BLKFLSBUF`, `_logical_block_size`, `_hash_area_direct`, `_drop_cached_pages` just above `def _hash_area` (≈ 5983); `_hash_area` signature and body; the two read-back calls in `apply_disk_patch` (Task 3's write loop and the verify loop `for entry in attempted:`).
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Produces: `_hash_area(path, offset=0, length=None, direct=False) -> str`; `_hash_area_direct(path, offset, length) -> str | None` (None when O_DIRECT is refused).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
requires_root_loop = pytest.mark.skipif(os.name == 'nt' or os.geteuid() != 0 or shutil.which('losetup') is None,
	reason='needs root and losetup')


@contextlib.contextmanager
def _plain_loop(image, sector_size=None):
	argv = ['losetup', '--find', '--show'] + (['--sector-size', str(sector_size)] if sector_size else []) + [str(image)]
	loop = subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()
	try:
		yield loop
	finally:
		subprocess.run(['losetup', '-d', loop], capture_output=True)


@requires_root_loop
@pytest.mark.root
@pytest.mark.loop
def test_direct_read_back_sees_the_device_not_the_cache(tmp_path, hpcp_mod):
	"""Spec Part 2 item 3: after a write through the loop device, the backing file is changed underneath; the direct
	read hashes what the device now holds (a buffered read may still return the cached page)."""
	image = tmp_path / 'disk.img'
	with open(str(image), 'wb') as f:
		f.truncate(8 << 20)
	payload = tmp_path / 'payload'
	payload.write_bytes(b'A' * 8192)
	with _plain_loop(image) as loop:
		hpcp_mod._write_area(loop, str(payload), 4096, 8192)
		assert hpcp_mod._hash_area(loop, 4096, 8192, direct=True) == _sha(b'A' * 8192)
		with open(str(image), 'r+b') as f:
			f.seek(4096)
			f.write(b'B' * 8192)
			f.flush()
			os.fsync(f.fileno())
		assert hpcp_mod._hash_area(loop, 4096, 8192, direct=True) == _sha(b'B' * 8192)


@requires_root_loop
@pytest.mark.root
@pytest.mark.loop
@pytest.mark.parametrize('sector_size', [512, 4096])
def test_direct_read_of_the_boot_code_on_any_sector_size(tmp_path, hpcp_mod, sector_size):
	"""Review Focus 5: the 440-byte boot code is read as its enclosing logical block and cut back to 440 bytes; an
	area ending short of a whole read chunk is hashed exactly."""
	image = tmp_path / 'disk.img'
	data = bytes(range(256)) * (5 << 12)
	image.write_bytes(data)
	with _plain_loop(image, sector_size) as loop:
		assert hpcp_mod._hash_area(loop, 0, 440, direct=True) == _sha(data[:440])
		assert hpcp_mod._hash_area(loop, 1000, 777, direct=True) == _sha(data[1000:1777])
		length = (5 << 20) - 3 * 4096
		assert hpcp_mod._hash_area(loop, 4096, length, direct=True) == _sha(data[4096:4096 + length])


def test_direct_read_falls_back_when_o_direct_is_refused(tmp_path, hpcp_mod, monkeypatch):
	"""Where O_DIRECT is refused (EINVAL), the cached pages are dropped and the area is read normally."""
	path = tmp_path / 'file'
	path.write_bytes(b'0123456789' * 1000)
	dropped = []
	real_open = os.open

	def refusing_open(name, flags, *args):
		if flags & getattr(os, 'O_DIRECT', 0):
			raise OSError(errno.EINVAL, 'Invalid argument')
		return real_open(name, flags, *args)

	monkeypatch.setattr(os, 'open', refusing_open)
	monkeypatch.setattr(hpcp_mod, '_drop_cached_pages', lambda name: dropped.append(name))
	assert hpcp_mod._hash_area(str(path), 10, 5, direct=True) == _sha(b'01234')
	assert dropped == [str(path)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "direct_read"`
Expected: FAIL — `TypeError: _hash_area() got an unexpected keyword argument 'direct'`.

- [ ] **Step 3: Implement**

Add just above `def _hash_area`:

```python
_BLKSSZGET = 0x1268
_BLKFLSBUF = 0x1261

def _logical_block_size(fd):
	"""fd's logical block size (BLKSSZGET); 4096 when it is not a block device or unknown, which suits every size."""
	try:
		import fcntl
		size = struct.unpack('i', fcntl.ioctl(fd, _BLKSSZGET, b'\0' * 4))[0]
	except (ImportError, OSError):
		return 4096
	return size if size > 0 else 4096

def _hash_area_direct(path, offset, length):
	"""sha256 of length bytes of path from offset, read past the page cache (O_DIRECT); None when O_DIRECT is refused.

	O_DIRECT needs the buffer, the file offset and the read size aligned to the logical block size: the reads cover
	the enclosing blocks into a page-aligned buffer (mmap), and only the requested bytes are hashed. Raises OSError
	when path ends before offset + length.
	"""
	flag = getattr(os, 'O_DIRECT', 0)
	if not flag:
		return None
	import mmap
	try:
		fd = os.open(path, os.O_RDONLY | flag | getattr(os, 'O_CLOEXEC', 0))
	except OSError as e:
		if e.errno == errno.EINVAL:
			return None
		raise
	buffer = None
	try:
		block = _logical_block_size(fd)
		chunk = max(block, _AREA_CHUNK - _AREA_CHUNK % block)
		buffer = mmap.mmap(-1, chunk)
		end = offset + length
		position = offset - offset % block
		digest = hashlib.sha256()
		while position < end:
			os.lseek(fd, position, os.SEEK_SET)
			try:
				count = os.readv(fd, [buffer])
			except OSError as e:
				if e.errno == errno.EINVAL and position == offset - offset % block:
					return None
				raise
			if count <= 0:
				break
			first, last = max(offset - position, 0), min(count, end - position)
			if last > first:
				with memoryview(buffer) as view:
					digest.update(view[first:last])
			position += count
		if position < end:
			raise OSError(errno.EIO, f'{path} ends before byte {end}')
		return digest.hexdigest()
	finally:
		if buffer is not None:
			buffer.close()
		os.close(fd)

def _drop_cached_pages(path):
	"""Ask the kernel to forget path's cached pages: BLKFLSBUF for a block device, then posix_fadvise DONTNEED."""
	try:
		fd = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
	except OSError:
		return
	try:
		try:
			import fcntl
			if stat.S_ISBLK(os.fstat(fd).st_mode):
				fcntl.ioctl(fd, _BLKFLSBUF, 0)
		except (ImportError, OSError):
			pass
		if hasattr(os, 'posix_fadvise'):
			try:
				os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
			except OSError:
				pass
	finally:
		os.close(fd)
```

Change `_hash_area`'s signature and docstring, and add the direct branch at the start of its body:

```python
def _hash_area(path, offset=0, length=None, direct=False):
	"""sha256 hex digest of length bytes of path from offset (to the end when length is None).

	direct (with a length): read past the page cache (_hash_area_direct), so a read-back sees what the device holds;
	where O_DIRECT is refused, the cached pages are dropped first (_drop_cached_pages) and the area read normally.
	Raises OSError when path ends before offset + length.
	"""
	if direct and length is not None:
		found = _hash_area_direct(path, offset, length)
		if found is not None:
			return found
		_drop_cached_pages(path)
	digest = hashlib.sha256()
```

(The rest of `_hash_area` stays as it is.)

In `apply_disk_patch`, change both read-backs to `direct=True`: in the write loop `if _hash_area(entry['device'], 0, entry['size'], direct=True) != entry['new']:` and in the verify loop `if _hash_area(entry['device'], 0, entry['size'], direct=True) == entry['new']:`. Update the write loop's comment to `So does one that raised no error but does not hold the new content when read back past the page cache.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py -q -k "direct_read or hash_area or raw"` — PASS.
Then: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q` — all pass; nothing left attached.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd-diff): read raw areas back past the page cache"
```

---

### Task 5: FAT-only timezone tolerance and full-hash marks on apply

**Files:**
- Modify: `hpcp.py` — constants after `_SINGLE_FS_FORMAT_VERSION` (≈ 3790); `states_equal` (≈ 3874); `get_entry_state` (≈ 3953); `_states_chunk`, `_states_for_paths` (≈ 4279-4307); `_check_payload` (≈ 4554-4557); `check_patch_target` (≈ 4692-4710); `_verify_rows` (≈ 5278-5305); `SectionPlan`, `_plan_section`, `_apply_section`, `_restore_across_sections`, `_verify_across_sections`; `apply_patch_tree` and `apply_disk_patch` `_plan_section` calls; `_is_fat` after `_FAT_TYPES` (≈ 6250).
- Test: `tests/test_disk_patch.py` (update the three group A timezone tests), `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `SectionPlan` with `skipped_identities` (Task 2).
- Produces: `_FULL_HASH_MARK = 'F'`; `_full_marked(state) -> bool`; `_is_fat(fs_type) -> bool`; `get_entry_state(path, full_hash=False, read_xattrs=True, mark_full=False)`; `_states_for_paths(paths, full_hash=False, read_xattrs=True, max_workers=1, marked=frozenset())`; `states_equal(a, b, compare_hash=True, metadata=True, full_hash=False, tz_tolerant=False)`; `check_patch_target(..., case_insensitive=False, tz_tolerant=False)`; `_verify_rows(..., passed=None, tz_tolerant=False)`; `_plan_section(..., case_insensitive=False, tz_tolerant=False)`; `SectionPlan` field `tz_tolerant` (last).

- [ ] **Step 1: Write the failing tests**

In `tests/test_disk_patch.py`, replace `test_content_only_states_tolerate_timezone_shaped_mtime_offsets` with:

```python
def test_fat_states_tolerate_timezone_shaped_mtime_offsets(hpcp_mod):
	"""Review Focus 3 (group A), FAT only (B1 decision 1): a FAT time read under two timezone settings differs by whole
	15-minute steps (at most 24 h); with the same size and hash that is the same file. Any other offset, another hash,
	exFAT / NTFS (content-only but no tolerance) and comparisons without hashes still count the difference."""
	eq = hpcp_mod.states_equal
	t = 1717000000 * 10**9

	def state(mtime_ns, digest='aaaa'):
		return f'f:10:{mtime_ns}:100755:0:0:{digest}:-'

	hour = 3600 * 10**9
	fat = dict(metadata=False, tz_tolerant=True)
	assert eq(state(t), state(t + 2 * hour), **fat)
	assert eq(state(t), state(t - 5 * hour - 30 * 60 * 10**9), **fat)
	assert eq(state(t), state(t + 2 * hour + 10**9), **fat)
	assert not eq(state(t), state(t + 7 * 60 * 10**9), **fat)
	assert not eq(state(t), state(t + 25 * hour), **fat)
	assert not eq(state(t), state(t + 2 * hour, 'bbbb'), **fat)
	assert not eq(state(t), state(t + 2 * hour), metadata=False)
	assert not eq(state(t), state(t + 2 * hour))
	assert not eq(state(t), state(t + 2 * hour), compare_hash=False, **fat)
	assert not eq(state(t, '-'), state(t + 2 * hour, '-'), **fat)
```

Replace `test_timezone_tolerance_needs_a_hash_of_the_whole_file` with:

```python
def test_timezone_tolerance_needs_a_hash_of_the_whole_file(hpcp_mod, reset_hpcp_globals):
	"""Final review I2 (group A): past HASH_SIZE only a file's tail is hashed, so a head edit with a timezone-shaped
	offset would pass. The offset is tolerated only when the hash covers the whole file: size at most HASH_SIZE,
	a full-hash mark ('F'), or full_hash."""
	eq = hpcp_mod.states_equal
	hpcp_mod.HASH_SIZE = 16
	t = 1717000000 * 10**9
	hour = 3600 * 10**9

	def state(size, mtime_ns, digest='aaaa'):
		return f'f:{size}:{mtime_ns}:100644:0:0:{digest}:-'

	fat = dict(metadata=False, tz_tolerant=True)
	assert eq(state(16, t), state(16, t + hour), **fat)
	assert not eq(state(17, t), state(17, t + hour), **fat)
	assert eq(state(17, t), state(17, t + hour), full_hash=True, **fat)
	assert eq(state(17, t, 'Faaaa'), state(17, t + hour, 'Faaaa'), **fat)
	assert eq(state(17, t), state(17, t + 10**9), **fat)
```

In `test_head_edit_with_a_timezone_shaped_offset_is_a_conflict`, change the `check_patch_target` call to `hpcp_mod.check_patch_target(rows, str(target), full_hash=full_hash, metadata=False, tz_tolerant=True)`.

Append to `tests/test_disk_patch.py`:

```python
def test_marked_states_hash_the_whole_file(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""get_entry_state(mark_full=True) hashes the whole file and marks it 'F' (unless full_hash already hashes every file
	in full); _full_marked recognises it."""
	hpcp_mod.HASH_SIZE = 16
	path = tmp_path / 'f'
	path.write_bytes(b'head' + b'x' * 60)
	marked = hpcp_mod.get_entry_state(str(path), mark_full=True)
	digest = marked.split(':')[6]
	assert digest == 'F' + hpcp_mod.hash_file.__wrapped__(str(path), 64, True)
	assert hpcp_mod._full_marked(marked) and not hpcp_mod._full_marked(hpcp_mod.get_entry_state(str(path)))
	assert hpcp_mod.get_entry_state(str(path), full_hash=True, mark_full=True).split(':')[6] == digest[1:]
	assert not hpcp_mod._full_marked('d:40755:0:0:-') and not hpcp_mod._full_marked('-')


def test_marked_rows_compare_the_node_file_hashed_in_full(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 3 item 3: rows whose states carry the mark compare the node's file hashed in full: on a FAT section a
	timezone-shifted identical copy is already done, a head edit (same size, tail unchanged) is a conflict, and verify
	of an identical copy passes."""
	hpcp_mod.HASH_SIZE = 4096
	base, target = tmp_path / 'base', tmp_path / 'target'
	body = b'h' * 4096 + b't' * 4096
	_write(base / 'same', body)
	_write(base / 'edited', body)
	_write(target / 'same', body, mtime=T0 + 3600)
	_write(target / 'edited', b'EDIT' + body[4:], mtime=T0 + 3600)
	state = lambda path: hpcp_mod.get_entry_state(str(path), mark_full=True)
	R = hpcp_mod.ChangeRow
	rows = [R('A', 'same', '-', state(base / 'same')), R('D', 'edited', state(base / 'edited'), '-')]
	checks = hpcp_mod.check_patch_target(rows, str(target), metadata=False, tz_tolerant=True)
	assert [(check.row.path, check.action) for check in checks] == [('same', 'skip'), ('edited', 'conflict')]
	assert hpcp_mod._verify_rows([rows[0]], str(target), metadata=False, tz_tolerant=True) == 0
```

Append to `tests/test_diff_image.py`:

```python
@pytest.mark.parametrize('fs_type, conflict', [('vfat', False), ('exfat', True)])
def test_timezone_tolerance_applies_on_fat_scopes_only(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch, fs_type, conflict):
	"""B1 decision 1: the node's copy of a changed file is the base's content with its mtime an hour later (a FAT time
	read under another timezone). On a vfat scope the M row applies; on exFAT (which records UTC offsets) the shift is
	a real difference: a conflict."""
	new, base = trees
	_write(new / 'boot.cfg', 'new')
	_write(base / 'boot.cfg', 'old')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	os.utime(str(target / 'boot.cfg'), (T0 + 3600, T0 + 3600))
	_fake_mounts(hpcp_mod, monkeypatch, (target, fs_type))
	if conflict:
		with pytest.raises(RuntimeError, match='^Patch conflict'):
			hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	else:
		summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
		assert summary['modified'] == 1 and summary['failed'] == 0
		assert (target / 'boot.cfg').read_text() == 'new'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q -k "timezone or marked or tolerance"`
Expected: FAIL — `states_equal() got an unexpected keyword argument 'tz_tolerant'`, `get_entry_state() got an unexpected keyword argument 'mark_full'`, and the exFAT case applies instead of conflicting.

- [ ] **Step 3: Implement**

After `_SINGLE_FS_FORMAT_VERSION = 1` add:

```python
# A hash over the whole file in a patch hashed by its tail (manifest full_hash 0): 'F' + the hex digest. Files on FAT
# carry it (see compute_changes); digests are lowercase hex, so the mark cannot be one.
_FULL_HASH_MARK = 'F'

def _full_marked(state):
	"""True when state is a file state whose hash carries the full-hash mark (_FULL_HASH_MARK)."""
	if not state.startswith('f:'):
		return False
	fields = state.split(':')
	return len(fields) == 8 and fields[6].startswith(_FULL_HASH_MARK)
```

Right after the `_FAT_TYPES = (...)` line add:

```python
def _is_fat(fs_type):
	"""True for the FAT family: local-time timestamps (the timezone tolerance) and full hashes (_FULL_HASH_MARK)."""
	return fs_type in _FAT_TYPES
```

Replace `states_equal`'s signature, the last paragraph of its docstring (from `With metadata False a whole number of 15-minute steps`), and its final `return`:

```python
def states_equal(a, b, compare_hash=True, metadata=True, full_hash=False, tz_tolerant=False):
	"""True when two entry states describe the same entry.

	Directories ignore mtime; files allow the timestamp-precision tolerance; an xattr
	digest of '-' on either side is not compared; '?' never equals anything.
	With compare_hash False neither the content hash nor the xattr digest is compared
	(both come from the same hasher). With metadata False mode, owner and xattrs are not
	compared: on FAT, exFAT and NTFS they come from mount options, not the filesystem.
	With tz_tolerant (a FAT section or scope: FAT stores local times) a whole number of 15-minute steps (at most 24
	hours) between file mtimes is tolerated when the hashes match and cover the whole file: a FAT time read under
	another timezone setting. The hashes cover the whole file when its size is at most HASH_SIZE, when they carry the
	full-hash mark (_FULL_HASH_MARK), or with full_hash; past HASH_SIZE only the file's tail is hashed, and a head
	edit would pass. NTFS stores UTC times and exFAT records the offset: no tolerance there.
	"""
```

```python
	# FAT stores local times: read under another timezone setting the same file shows a whole-step offset. Only
	# FAT comparisons whose hashes confirmed the whole content accept it.
	return (tz_tolerant and compare_hash and info_a['hash'] not in ('', '-')
		and (full_hash or info_a['hash'].startswith(_FULL_HASH_MARK) or info_a['size'] <= HASH_SIZE)
		and _tz_shift_close(info_a['mtime_ns'], info_b['mtime_ns']))
```

`get_entry_state`: change the signature to `def get_entry_state(path, full_hash=False, read_xattrs=True, mark_full=False):`, add to its docstring `mark_full: hash a regular file in full and mark the hash (_FULL_HASH_MARK), unless full_hash already hashes every file in full.`, and replace the hashing line with:

```python
				# hash_file is lru_cached and would return a stale hash for a same-size rewrite.
				file_hash = hash_file.__wrapped__(path, st.st_size, full_hash or mark_full) or '-'
				if mark_full and not full_hash and file_hash != '-':
					file_hash = _FULL_HASH_MARK + file_hash
```

`_states_chunk` and `_states_for_paths`:

```python
def _states_chunk(paths, full_hash, read_xattrs, hash_size, marked=frozenset()):
	"""Worker body for _states_for_paths: states plus the errors recorded while computing them.

	hash_size is the caller's HASH_SIZE; a worker that did not fork from the caller
	(spawn / forkserver) would otherwise hash with the module default.
	"""
	global HASH_SIZE
	HASH_SIZE = hash_size
	error_start = len(ERRORS)
	states = {path: get_entry_state(path, full_hash, read_xattrs, path in marked) for path in paths}
	errors = ERRORS[error_start:]
	del ERRORS[error_start:]
	return states, errors

def _states_for_paths(paths, full_hash=False, read_xattrs=True, max_workers=1, marked=frozenset()):
	"""{path: state} for every path; parallel chunks on a ProcessPoolExecutor for large sets. Paths in marked are
	hashed in full with the mark (see get_entry_state)."""
	paths = list(paths)
	if max_workers <= 1 or len(paths) < 256:
		return {path: get_entry_state(path, full_hash, read_xattrs, path in marked) for path in paths}
	chunk_size = max(64, len(paths) // (max_workers * 4) + 1)
	states = {}
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = []
		for index in range(0, len(paths), chunk_size):
			chunk = paths[index:index + chunk_size]
			futures.append(executor.submit(_states_chunk, chunk, full_hash, read_xattrs, HASH_SIZE,
				frozenset(path for path in chunk if path in marked)))
		for future in concurrent.futures.as_completed(futures):
			chunk_states, chunk_errors = future.result()
			states.update(chunk_states)
			ERRORS.extend(chunk_errors)
	return states
```

`_check_payload`: replace the `states = _states_for_paths(...)` line with:

```python
	marked = {copy for row, copy in copies.items() if _full_marked(row.new)}
	states = _states_for_paths(set(copies.values()), full_hash=full_hash, read_xattrs=False, max_workers=max_workers, marked=marked)
```

`check_patch_target`: change the signature to `def check_patch_target(rows, target_root, compare_hash=True, full_hash=False, max_workers=1, metadata=True, case_insensitive=False, tz_tolerant=False):`, add to its docstring `Rows whose states carry the full-hash mark (_FULL_HASH_MARK) are compared with the target's file hashed in full and marked. tz_tolerant: a FAT section (see states_equal).`, and replace the `states = ...` and `same = ...` lines with:

```python
	marked = {paths[row.path] for row in rows if _full_marked(row.old) or _full_marked(row.new)}
	states = _states_for_paths(set(paths.values()), full_hash=full_hash, read_xattrs=metadata, max_workers=max_workers, marked=marked)
	same = functools.partial(states_equal, compare_hash=compare_hash, metadata=metadata, full_hash=full_hash, tz_tolerant=tz_tolerant)
```

`_verify_rows`: change the signature to `def _verify_rows(rows, target_root, compare_hash=True, full_hash=False, max_workers=1, metadata=True, written_rows=(), passed=None, tz_tolerant=False):`, add to its docstring `Marked rows are read hashed in full (see check_patch_target); tz_tolerant: a FAT section.`, replace the `states = ...` line with:

```python
	marked = {paths[name] for name, row in final.items() if _full_marked(row.old) or _full_marked(row.new)}
	states = _states_for_paths(set(paths.values()), full_hash=full_hash, read_xattrs=metadata, max_workers=max_workers, marked=marked)
```

and the comparison with `ok = found == '-' if expected == '-' else states_equal(found, expected, compare_hash, metadata, full_hash, tz_tolerant)`.

`SectionPlan`: append ` tz_tolerant` to the field string and `tz_tolerant: a FAT section (see states_equal).` to its comment. `_plan_section`: signature `def _plan_section(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers, metadata=True, case_insensitive=False, tz_tolerant=False):`; pass `tz_tolerant=tz_tolerant` to `check_patch_target`; append `tz_tolerant` as the last `SectionPlan` argument.

Pass `tz_tolerant=plan.tz_tolerant` to every `_verify_rows` call in `_apply_section`, `_restore_across_sections` and `_verify_across_sections`.

`apply_patch_tree`: in the `plans.append(_plan_section(...))` call add `tz_tolerant=_is_fat(fs_type)` after the `_has_case_insensitive_names(fs_type, source)` argument. `apply_disk_patch`: in its `_plan_section(...)` call add `tz_tolerant=_is_fat(fs_type)` after `_has_case_insensitive_names(fs_type, part['device'])`.

In `apply_patch_tree`'s docstring, change `on FAT, exFAT or NTFS by content only,` to `on FAT, exFAT or NTFS by content only (on FAT with the timezone tolerance, see states_equal),`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q -k "timezone or marked or tolerance or fat"` — PASS.
Then: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q` — all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py tests/test_diff_image.py
git commit -m "fix(diff-image): timezone tolerance on FAT only; marked rows hash the node's file in full"
```

---

### Task 6: FAT files hashed in full at generation; format version 3

**Files:**
- Modify: `hpcp.py` — format constants (≈ 3788-3790); `compute_changes` (≈ 4309-4334); add `_paths_on_fat` after `_compare_metadata` (≈ 6217); `process_get_diff_image` (≈ 5651-5653); `_partition_changes` (≈ 6467-6490) and its caller in `process_get_disk_diff_image` (≈ 6586); `make_patch_meta` (≈ 4475); `make_disk_patch_meta` (≈ 6515).
- Test: `tests/test_diff_image.py`, `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: `_FULL_HASH_MARK`, `_full_marked`, `_is_fat`, `_states_for_paths(marked=)`, `states_equal(tz_tolerant=)` (Task 5); `_mount_scopes`, `_mountinfo_entries` (existing).
- Produces: `compute_changes(new_map, base_map, full_hash=False, max_workers=..., read_xattrs=True, metadata=True, fat_paths=frozenset())`; `_paths_on_fat(paths, root, entries) -> set`; `DIFF_FORMAT_VERSION = 3`, `_DISK_FORMAT_VERSION = 2`, `_FULL_HASH_FORMAT_VERSION = 3`; `_partition_changes(..., metadata=True, fat=False)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
def test_fat_paths_are_hashed_in_full_and_compared_like_fat(trees, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 3 item 1: a key whose path on either side is on FAT is hashed in full on both sides (marked), compared
	by content only and with the timezone tolerance; other keys keep the tail hash and the plain comparison."""
	hpcp_mod.HASH_SIZE = 16
	new, base = trees
	body = b'h' * 64
	for tree in (new, base):
		_write(tree / 'esp' / 'shifted', body)
		_write(tree / 'root' / 'shifted', body)
	os.utime(str(new / 'esp' / 'shifted'), (T0 + 3600, T0 + 3600))
	os.utime(str(new / 'root' / 'shifted'), (T0 + 3600, T0 + 3600))
	_write(new / 'esp' / 'edited', b'EDIT' + body[4:])
	_write(base / 'esp' / 'edited', body)
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	fat = {path for path in list(new_map.values()) + list(base_map.values()) if os.sep + 'esp' + os.sep in path}
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1, fat_paths=fat)
	by_path = {row.path: row for row in rows}
	assert set(by_path) == {'esp/edited', 'root/shifted'}
	assert hpcp_mod._full_marked(by_path['esp/edited'].old) and hpcp_mod._full_marked(by_path['esp/edited'].new)
	assert not hpcp_mod._full_marked(by_path['root/shifted'].new)


def test_paths_on_fat_follow_the_mount_below_the_root(tmp_path, hpcp_mod, monkeypatch):
	root = tmp_path / 'tree'
	paths = [str(root / 'etc' / 'x'), str(root / 'boot' / 'efi' / 'EFI' / 'x'), str(root / 'boot' / 'efix')]
	entries = [_mount_entry(tmp_path, 'ext4'), _mount_entry(root / 'boot' / 'efi', 'vfat')]
	for path in paths:
		os.makedirs(os.path.dirname(path), exist_ok=True)
	assert hpcp_mod._paths_on_fat(paths, str(root), entries) == {paths[1]}
	assert hpcp_mod._paths_on_fat(paths, str(root), [_mount_entry(tmp_path, 'ext4')]) == set()


def test_format_version_follows_the_full_hash_mark(hpcp_mod):
	R = hpcp_mod.ChangeRow
	plain = [R('A', 'a', '-', 'f:5:1:100644:0:0:aa:-')]
	marked = [R('A', 'a', '-', 'f:5:1:100644:0:0:Faa:-')]
	assert hpcp_mod.make_patch_meta(plain, 'n', 'b', 'root')['format_version'] == 1
	assert hpcp_mod.make_patch_meta(marked, 'n', 'b', 'root')['format_version'] == 3
	assert hpcp_mod.make_disk_patch_meta([(1, plain)], [], 'n', 'b')['format_version'] == 2
	assert hpcp_mod.make_disk_patch_meta([(1, marked)], [], 'n', 'b')['format_version'] == 3


def test_a_newer_reader_accepts_version_3_and_an_older_one_refuses_it(tmp_path, hpcp_mod, monkeypatch):
	path = tmp_path / 'manifest'
	path.write_text('format\thpcp-diff\nformat_version\t3\n')
	assert hpcp_mod.read_manifest(str(path))['format_version'] == '3'
	monkeypatch.setattr(hpcp_mod, 'DIFF_FORMAT_VERSION', 2)
	with pytest.raises(RuntimeError, match='uses format version 3, newer than supported 2'):
		hpcp_mod.read_manifest(str(path))
```

In `test_read_manifest_rejects_foreign_or_newer`, change `format_version\t3` to `format_version\t4`.

Append to `tests/test_disk_patch.py`:

```python
@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_fat_section_files_are_hashed_in_full_and_tolerate_a_timezone_shift(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 3: the golden changes a 256 KiB ESP file. Its rows carry the full-hash mark and the patch is format 3;
	the root section's rows are unmarked. A node whose copy of the old file is identical but an hour later (another
	timezone) applies; a node whose copy differs in its head (same size, same tail) is a conflict."""
	hpcp_mod.HASH_SIZE = 65536  # the CLI's -hs default: the 256 KiB file is past the tail window
	old, new = tmp_path / 'old.img', tmp_path / 'new.img'
	_make_golden(old)
	big = bytes(range(256)) * 1024
	with _attached(old) as loop, _mounted(loop + 'p1') as esp:
		_write(os.path.join(esp, 'EFI', 'BOOT', 'shim.efi'), big)
	shutil.copyfile(str(old), str(new))
	with _attached(new) as loop:
		with _mounted(loop + 'p1') as esp:
			_write(os.path.join(esp, 'EFI', 'BOOT', 'shim.efi'), big[:-4] + b'NEW!')
		with _mounted(loop + 'p3') as root:
			_write(os.path.join(root, 'etc', 'os-release'), b'v2')
	patch = str(tmp_path / 'patch.img')
	with _session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_disk_diff_image([str(new)], [str(old)], patch, mount_points, loop_devices, max_workers=1) == patch
		volume = hpcp_mod._mount_single_fs_image(patch, True, mount_points, loop_devices)
		meta = hpcp_mod.read_manifest(os.path.join(volume, hpcp_mod.DIFF_DIR_NAME, 'manifest'))
		esp_rows = hpcp_mod.read_changes(os.path.join(volume, hpcp_mod.DIFF_DIR_NAME, 'part1.changes.tsv'))
		root_rows = hpcp_mod.read_changes(os.path.join(volume, hpcp_mod.DIFF_DIR_NAME, 'part3.changes.tsv'))
	assert meta['format_version'] == '3'
	assert all(hpcp_mod._full_marked(row.new) for row in esp_rows if row.new.startswith('f:'))
	assert not any(hpcp_mod._full_marked(row.new) for row in root_rows)
	for name, head, expect_conflict in (('shifted.img', b'', False), ('edited.img', b'EDIT', True)):
		target = tmp_path / name
		shutil.copyfile(str(old), str(target))
		with _attached(target) as loop, _mounted(loop + 'p1') as esp:
			shim = os.path.join(esp, 'EFI', 'BOOT', 'shim.efi')
			if head:
				with open(shim, 'r+b') as f:
					f.write(head)
			stat_result = os.stat(shim)
			os.utime(shim, (stat_result.st_atime + 3600, stat_result.st_mtime + 3600))
		if expect_conflict:
			with pytest.raises(RuntimeError, match='^Patch conflict'):
				_apply(hpcp_mod, patch, target)
		else:
			summary = _apply(hpcp_mod, patch, target)
			assert summary['failed'] == 0 and summary['sections']['part1']['modified'] >= 1


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_single_filesystem_generation_marks_rows_under_a_fat_mount(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 3 item 1: diffing a tree with a vfat filesystem mounted below its root marks only the rows under it."""
	hpcp_mod.HASH_SIZE = 16
	esp_image = tmp_path / 'esp.img'
	with open(str(esp_image), 'wb') as f:
		f.truncate(32 << 20)
	_run('mkfs.vfat', esp_image)
	new, base = tmp_path / 'new', tmp_path / 'base'
	_write(os.path.join(str(new), 'etc', 'conf'), b'new-conf-' * 8)
	_write(os.path.join(str(base), 'etc', 'conf'), b'old-conf-' * 8)
	os.makedirs(os.path.join(str(new), 'boot', 'efi'))
	_write(os.path.join(str(base), 'boot', 'efi', 'EFI', 'x.efi'), b'old-efi-' * 8)
	with _plain_loop(esp_image) as loop, _mounted(loop, 'vfat') as esp_mount:
		subprocess.run(['mount', '--bind', esp_mount, os.path.join(str(new), 'boot', 'efi')], check=True)
		try:
			_write(os.path.join(str(new), 'boot', 'efi', 'EFI', 'x.efi'), b'new-efi-' * 8)
			entries = hpcp_mod._mountinfo_entries()
			new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
			fat = hpcp_mod._paths_on_fat(new_map.values(), str(new), entries) | hpcp_mod._paths_on_fat(base_map.values(), str(base), entries)
			rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1, fat_paths=fat)
		finally:
			subprocess.run(['umount', os.path.join(str(new), 'boot', 'efi')], check=False)
	by_path = {row.path: row for row in rows}
	assert hpcp_mod._full_marked(by_path['boot/efi/EFI/x.efi'].new) and hpcp_mod._full_marked(by_path['boot/efi/EFI/x.efi'].old)
	assert not hpcp_mod._full_marked(by_path['etc/conf'].new)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q -k "fat_paths or paths_on_fat or format_version or version_3 or newer or hashed_in_full or marks_rows"`
Expected: FAIL — `compute_changes() got an unexpected keyword argument 'fat_paths'`, `_paths_on_fat` missing, format version 1/2 for marked rows, version 3 rejected as newer.

- [ ] **Step 3: Implement**

Replace the format constants:

```python
DIFF_DIR_NAME = '.hpcp_diff'
DIFF_FORMAT_VERSION = 3
# Single-filesystem patches keep the version hpcp 9.65 reads; disk patches (layout disk) are version 2; a patch whose
# states carry the full-hash mark (_FULL_HASH_MARK, FAT files, hpcp 9.69) is version 3, which 9.68 and older refuse.
_SINGLE_FS_FORMAT_VERSION = 1
_DISK_FORMAT_VERSION = 2
_FULL_HASH_FORMAT_VERSION = 3
```

Replace `compute_changes` with:

```python
def compute_changes(new_map, base_map, full_hash=False, max_workers=multiprocessing.cpu_count(), read_xattrs=True, metadata=True,
		fat_paths=frozenset()):
	"""Rows that turn base_map's tree into new_map's tree, sorted D / A / M.

	With metadata False (see states_equal) a mode, owner or xattr difference alone is no change.
	fat_paths: paths (values of either map) on FAT. A key whose path on either side is one is compared like a FAT
	section: by content only and with the timezone tolerance, its files on both sides hashed in full and marked
	(_FULL_HASH_MARK; unless full_hash already hashes every file in full), so the tolerance holds past HASH_SIZE.
	"""
	keys = set(new_map) | set(base_map)
	fat_keys = {key for key in keys if new_map.get(key) in fat_paths or base_map.get(key) in fat_paths} if fat_paths else set()
	marked = set() if full_hash else {tree[key] for tree in (new_map, base_map) for key in fat_keys if key in tree}
	states = _states_for_paths(set(new_map.values()) | set(base_map.values()), full_hash=full_hash,
		read_xattrs=read_xattrs and metadata, max_workers=max_workers, marked=marked)
	rows = []
	for key in keys:
		new_state = states[new_map[key]] if key in new_map else '-'
		old_state = states[base_map[key]] if key in base_map else '-'
		# A read failure (already recorded) or a special file: a guessed row could delete or add real files.
		if new_state == '?' or old_state == '?':
			continue
		if new_state == '-' and old_state == '-':
			continue
		if old_state == '-':
			rows.append(ChangeRow('A', key, '-', new_state))
		elif new_state == '-':
			rows.append(ChangeRow('D', key, old_state, '-'))
		elif old_state[0] != new_state[0]:
			rows.append(ChangeRow('D', key, old_state, '-'))
			rows.append(ChangeRow('A', key, '-', new_state))
		elif not states_equal(old_state, new_state, metadata=metadata and key not in fat_keys, full_hash=full_hash,
				tz_tolerant=key in fat_keys):
			rows.append(ChangeRow('M', key, old_state, new_state))
	return sort_change_rows(rows)
```

Add after `_compare_metadata`:

```python
def _paths_on_fat(paths, root, entries):
	"""The paths (absolute, under root) on a FAT filesystem: root's own mount, or one mounted below it (_mount_scopes;
	hidden mounts left out). Each path looks up its own parent names, deepest first, as _rows_by_scope does."""
	fat = {prefix: _is_fat(fs_type) for prefix, fs_type, _ in _mount_scopes(root, entries)}
	if not any(fat.values()):
		return set()
	root = os.path.abspath(root)
	result = set()
	for path in paths:
		name = os.path.relpath(path, root).replace(os.sep, '/')
		if name == '.' or name.startswith('../'):
			name = ''
		while name and name not in fat:
			name = name.rpartition('/')[0]
		if fat.get(name, False):
			result.add(path)
	return result
```

`process_get_diff_image`: replace

```python
	metadata = _compare_metadata((new_root, base_root), _mountinfo_entries())
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata)
```

with

```python
	entries = _mountinfo_entries()
	metadata = _compare_metadata((new_root, base_root), entries)
	# Files on FAT on either side (the root itself, or a mount below it such as /boot/efi) are hashed in full.
	fat_paths = _paths_on_fat(new_map.values(), new_root, entries) | _paths_on_fat(base_map.values(), base_root, entries)
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata, fat_paths=fat_paths)
```

`_partition_changes`: signature `def _partition_changes(new_root, old_root, mount_points, live_mounts, exclude, full_hash, max_workers, parallel_file_listing, input_files=(), metadata=True, fat=False):`; add to its docstring `fat: a FAT section, its files hashed in full (see compute_changes).`; replace its `compute_changes` call with:

```python
	fat_paths = frozenset(new_map.values()) | frozenset(base_map.values()) if fat else frozenset()
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata, fat_paths=fat_paths)
```

In `process_get_disk_diff_image`, change the `_partition_changes(...)` call's last argument to `metadata=_has_posix_metadata(new_part['fs_type']), fat=_is_fat(new_part['fs_type']) or _is_fat(old_part['fs_type'])`.

`make_patch_meta`: change `'format_version': _SINGLE_FS_FORMAT_VERSION,` to:

```python
		'format_version': (_FULL_HASH_FORMAT_VERSION if any(_full_marked(row.old) or _full_marked(row.new) for row in rows)
			else _SINGLE_FS_FORMAT_VERSION),
```

`make_disk_patch_meta`: change `meta['format_version'] = DIFF_FORMAT_VERSION` to:

```python
	# Version 3 when a section's rows carry the full-hash mark (see make_patch_meta), else the disk version 2.
	if meta['format_version'] != _FULL_HASH_FORMAT_VERSION:
		meta['format_version'] = _DISK_FORMAT_VERSION
```

- [ ] **Step 4: Run the tests to verify they pass**

`test_disk_patch_generation_records_sections_raw_and_identity` (tests/test_disk_patch.py ≈ 984) builds its golden with `_make_newer`, which changes a file on the vfat ESP (part1): that patch now carries marks, so change its assertion to `assert (meta['layout'], meta['format_version']) == ('disk', '3')` and add, right after it, `assert all(hpcp_mod._full_marked(row.new) for row in hpcp_mod.read_changes(os.path.join(root, '.hpcp_diff', 'part1.changes.tsv')) if row.new.startswith('f:'))`.

Run: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q -k "fat_paths or paths_on_fat or format_version or version_3 or newer or hashed_in_full or marks_rows or records_sections"` — PASS.
Then: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q` — all pass; nothing left mounted or attached.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py tests/test_disk_patch.py
git commit -m "feat(diff-image): hash FAT files in full at generation (F mark); format version 3"
```

---

### Task 7: Docs, version 9.69, full verification

**Files:**
- Modify: `README.md`, `hpcp.py` (`version`, `COMMIT_DATE`), `tests/test_dd_fs_params.py` (`test_version_bumped`)

- [ ] **Step 1: README**

In `README.md`:

1. In the paragraph at ≈ line 90 that begins `Files and symlinks are replaced atomically`, replace `whole trees are removed only for conflicts forced with `-fa`.` with `whole trees are removed only for conflicts forced with `-fa`, and that removal never enters a mounted filesystem (bind mounts included): a directory holding a mount point is left in place with the mount, that row fails verify, and a note names the mount before the prompt.`
2. In the paragraph at ≈ line 92 (single-filesystem patches), replace `On FAT, exFAT and NTFS rows are compared and written by content only, since modes and owners there come from mount options. As in disk sections, a file whose mtime differs only by a timezone offset (a whole number of 15 minutes, at most 24 hours) with the same content (hashed in full: files up to the patch's `-hs` size, or patches made with `-fh`) counts as unchanged, and a replaced FAT file keeps its read-only, hidden and system attributes.` with `On FAT, exFAT and NTFS rows are compared and written by content only, since modes and owners there come from mount options. On FAT, which stores local times, a file whose mtime differs only by a timezone offset (a whole number of 15 minutes, at most 24 hours) with the same content counts as unchanged; files on FAT on either side when the patch is made (including a mount below the source root, such as `/boot/efi`) are hashed in full for this, whatever `-hs` says (a golden whose ESP was a plain directory is hashed by its tail, so there the tolerance covers files up to the `-hs` size only). A replaced FAT file keeps its read-only, hidden and system attributes. A write that lands on an entry another row found already done (through a symlinked directory, or a name differing only by case) is reported by verify.`
3. In the disk-patch bullet at ≈ line 124 that begins `- FAT, exFAT and NTFS sections (the ESP)`, replace `A file whose mtime differs only by a timezone offset (a whole number of 15 minutes, at most 24 hours) with the same content (hashed in full: files up to the patch's `-hs` size, or patches made with `-fh`) counts as unchanged, since FAT stores local times.` with `On FAT sections, files are hashed in full and a file whose mtime differs only by a timezone offset (a whole number of 15 minutes, at most 24 hours) with the same content counts as unchanged, since FAT stores local times (exFAT and NTFS record UTC times and get no such tolerance).`
4. After the bullet at ≈ line 125 that begins `- Filesystem sections are applied first`, add a bullet:

```markdown
- Every partition the patch writes raw or sets a UUID or label on must open exclusively (`O_EXCL`) before anything changes; one that is mounted, active swap or held by LVM/md/dm — in this or another mount namespace — is refused (`Disk in use`). During the raw writes and identity changes hpcp locks the whole disk (`flock`, so udev leaves it alone) and holds those partitions exclusively; raw areas are read back past the page cache.
```

5. After the bullet that begins `- An image file that is already attached to a loop device is refused`, add a bullet:

```markdown
- A patch with files on FAT is written as format version 3 (the full-hash marks); hpcp 9.68 and older refuse it ("newer than supported"). Patches without FAT files keep format 1 or 2.
```

If `python3 hpcp.py -h` output changed (it should not), regenerate the README help block the way the 9.66 release did; otherwise leave it.

- [ ] **Step 2: Version**

```bash
sed -i "s/^version = '9.68'$/version = '9.69'/" hpcp.py
sed -i "s/^COMMIT_DATE = '.*'$/COMMIT_DATE = '$(date +%F)'/" hpcp.py
grep -nE "^version|^COMMIT_DATE" hpcp.py
```

Update `test_version_bumped` in `tests/test_dd_fs_params.py` to expect `'9.69'` and the new `COMMIT_DATE`.

- [ ] **Step 3: Full verification**

Run: `python -m pytest tests/ -q` — all pass, exactly 1 skipped.
Then:
- `python3 hpcp.py -V` prints `9.69`;
- `uv run --no-project --python 3.8 python -c "import ast; ast.parse(open('hpcp.py').read(), feature_version=(3,6)); print('ok')"` prints `ok`;
- `losetup -a | grep -c /tmp/pytest` and `mount | grep -c /tmp/pytest` print 0;
- `md5sum /usr/lib/os-release` prints `776b0aac9b17d49a0ac879da9801f51b`. If it differs, STOP and report BLOCKED without trying to repair anything.

- [ ] **Step 4: Commit**

```bash
git add README.md hpcp.py tests/test_dd_fs_params.py
git commit -m "docs: forced removals stop at mounts, exclusive disk writes, FAT full hashes; release 9.69"
```

Releasing (gate review, push, uvpack, cas-toolbox pin) is out of scope for this plan.
