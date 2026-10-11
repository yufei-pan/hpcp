# Patch Apply Repair and Case Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make hpcp patch apply correct on any filesystem — case-only renames included — by re-checking skipped writes after deletes, replacing the 9.67 refusals, and finishing the FAT/HFS details, released as 9.68.

**Architecture:** All production code is in the single module `hpcp.py` (Diff Image section for patch apply/generation, Disk Patch section for disk sections). `_apply_rows` gains a repair step that restores skipped A/M rows a removal took away; `_verify_rows` accepts a deleted name that is the same entry as a written name differing only by case. `check_patch_target` matches names with a per-character fold on known case-insensitive targets. Single-filesystem apply splits rows into one plan per mount scope (target and mounts below it), each content-only and/or case-insensitive by its filesystem type. The 9.67 refusals and case-folding generation are removed.

**Tech Stack:** Python ≥ 3.6 stdlib (`fcntl` lazily for FAT ioctls), pytest; root + loop tests use `losetup`, `mkfs.vfat`, `mkfs.hfsplus`, `mount`.

**Spec:** `docs/superpowers/specs/2026-10-11-apply-repair-and-case-handling-design.md`

## Global Constraints

- Single file: all production code goes in `hpcp.py`; no new modules or packages.
- Tabs for indentation, in `hpcp.py` and in tests.
- Python 3.6 syntax floor: no walrus, no `str.removeprefix`, no `dict | dict`, no `f"{x=}"`.
- No new runtime dependencies. `fcntl` is imported inside the FAT helpers only (hpcp also runs on Windows, where `fcntl` does not exist).
- `eprint("Category: detail")` only for real failures (it records into `ERRORS`; the exit code comes from the category). Warnings and notes use `print(..., file=sys.stderr)` / `print(...)`.
- Patch formats are unchanged (manifest v1 / v2, `changes.tsv`, `disk.tsv`); patches from 9.65, 9.66 and 9.67 must still apply.
- Root / loop tests: scratch files only under the test's `tmp_path` or /tmp; never write to real filesystem paths; any shell probe uses `set -euo pipefail` and refuses empty path variables (a probe with an empty mount-point variable once overwrote `/usr/lib/os-release`). Root tests leave nothing mounted or attached.
- Version target 9.68, bumped only in the final task.
- Run tests from the repository root: `python -m pytest tests/ -q` (expect 1 skipped: a Windows GUI test).
- Commit messages end with the Co-Authored-By trailer the implementing agent's harness specifies.

## Review Focus

1. Re-running a patch on a FAT target after a case-only rename has been applied must skip every row (no deletes, no conflicts). Test: Task 2 `test_esp_case_only_rename_applies_with_the_new_spelling` (second apply), Task 3 `test_case_only_rename_patch_applies_on_a_vfat_image` (second run).
2. A skipped row that must be restored but whose payload is missing from the patch volume must fail loudly (an error and a verify failure), not crash or pass silently. Test: Task 1 `test_restore_without_a_payload_is_a_failure`.
3. The timezone tolerance must never hide a real content change: same-shape offset with a different hash stays a difference. Test: Task 4 `test_content_only_states_tolerate_timezone_shaped_mtime_offsets`.
4. A read-only + hidden + system FAT file must still be replaced, and keep those attributes. Test: Task 4 `test_replacing_a_file_on_vfat_keeps_its_fat_attributes`.
5. On a case-insensitive target hpcp does not recognise, a re-run after the case rename must still leave the files correct. Test: Task 3 `test_case_only_rename_is_restored_on_a_vfat_image_reported_as_nfs4` (second run).

## Deviations from the spec (decided while planning)

- `_mount_scopes` (replacing `_path_mounts`) decides which mounts are visible by mountinfo order — a mount listed later on the same or a parent mount point hides earlier ones — instead of comparing device numbers. Same result, and it does not stat paths that may be hidden.
- An entry lost to a forced (`-fa`) type change through a symlink alias cannot be restored (its alias now points at a file); it is reported as a failure (copy error + verify failure), not restored.
- `_verify_rows`: a D row whose path still exists passes when an A / M row's name differs from it only by case (`_fold_name`) and both paths are one entry (same `st_dev` / `st_ino`). Without this, a correct case-only rename on a case-insensitive target hpcp does not recognise would be reported as a verify failure.
- `_fold_name` is introduced in Task 1 (verify uses it); Task 2 uses it for planning.

---

## File Structure

| File | Responsibility |
|---|---|
| `hpcp.py` (modify) | `_apply_rows` repair step, `_holds_type`, `_verify_rows` + `_same_entry`, `_apply_section` summary; `_fold_name`, `_spelled_on_disk`, `check_patch_target`, `_unlisted_children`; `_has_case_insensitive_names` (HFSX); `_partition_changes`; `_mount_scopes`, `_rows_by_scope`, `_compare_metadata`, `apply_patch_tree`, `_preflight_section`, `apply_disk_patch` prompt order; `states_equal` + `_tz_shift_close`; FAT attribute helpers in `_finish_replace`; removal of `_refuse_case_only_renames`, `_CASE_SENSITIVE_TYPES`, `_path_mounts`, `_same_entry_pair`, `_caseless`, `_fold_case_keys`, `_case_folded_names`, `_case_collision`, `import unicodedata`; version |
| `tests/test_diff_image.py` (modify) | single-filesystem apply tests: repair, contradictory alias, scopes, prompt order, vfat root tests; removed-refusal tests deleted |
| `tests/test_disk_patch.py` (modify) | disk-section tests: exact-name generation, planning fold, HFSX, ESP root tests rewritten; removed-refusal tests deleted |
| `tests/test_dd_fs_params.py` (modify) | `test_version_bumped` |
| `README.md` (modify) | patch-image notes (single-filesystem case handling, disk patch FAT/case notes) |
| `docs/superpowers/specs/2026-10-11-apply-repair-and-case-handling-design.md` (modify) | the three deviations above |

---

### Task 1: Repair step, case-aware verify, `restored` count

**Files:**
- Modify: `hpcp.py` — `_apply_rows` (≈ line 5060), `_verify_rows` (≈ 5175), `_apply_section` (≈ 5254), the `Section partN:` print in `apply_disk_patch` (≈ 6648); add `_fold_name` just above `def _under_removed`, `_holds_type` and `_same_entry` just above `def _apply_rows`.
- Test: `tests/test_diff_image.py`, `tests/test_disk_patch.py`

**Interfaces:**
- Produces:
  - `_fold_name(name: str) -> str`
  - `_holds_type(path: str, state: str) -> bool`
  - `_same_entry(path: str, others) -> bool`
  - `_apply_rows(rows, volume_roots, target_root, max_workers=1, verbose=False, full_hash=False, forced_rows=(), metadata=True, skipped=()) -> list[ChangeRow]` (returns the restored rows)
  - `_apply_section(plan, max_workers, verbose, full_hash, compare_hash) -> dict` with keys, in order: `deleted, added, modified, skipped, restored, forced, failed`

- [ ] **Step 1: Delete the two tests the repair step replaces**

In `tests/test_diff_image.py` delete the whole function `test_single_fs_apply_on_case_insensitive_target_verifies_rows_skipped_as_done` (with its `@pytest.mark.parametrize` decorator). In `tests/test_disk_patch.py` delete the whole function `test_content_only_verify_covers_rows_skipped_as_done`. (Both simulate a loss from outside the apply; the repair step covers losses the apply itself causes.)

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
def _restore_scenario(hpcp_mod, tmp_path):
	"""The target spells EFI/BOOT/x.efi, and EFI/Boot is a symlink to BOOT: a lookup of EFI/Boot/x.efi finds the
	same file, as on a case-insensitive filesystem. The patch deletes EFI/BOOT/x.efi and adds EFI/Boot/x.efi with
	the same content (a case-only rename)."""
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'EFI' / 'BOOT' / 'x.efi', 'loader')
	os.symlink('BOOT', str(target / 'EFI' / 'Boot'))
	_write(vol / 'EFI' / 'Boot' / 'x.efi', 'loader')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('D', 'EFI/BOOT/x.efi', state(str(target / 'EFI' / 'BOOT' / 'x.efi')), '-'),
		R('A', 'EFI/Boot/x.efi', '-', state(str(vol / 'EFI' / 'Boot' / 'x.efi')))]
	return target, vol, rows


def test_apply_restores_a_skipped_entry_a_delete_took_away(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""The add is skipped as already done (the file is there under the other spelling), then the delete removes that
	very file. The repair step writes the add again; verify accepts the deleted name because it is the same entry as
	the written one, differing only by case."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol, rows = _restore_scenario(hpcp_mod, tmp_path)
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	assert {check.row.path: check.action for check in plan.checks} == {'EFI/BOOT/x.efi': 'apply', 'EFI/Boot/x.efi': 'skip'}
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary == {'deleted': 1, 'added': 0, 'modified': 0, 'skipped': 0, 'restored': 1, 'forced': 0, 'failed': 0}
	assert hpcp_mod.ERRORS == []
	assert (target / 'EFI' / 'Boot' / 'x.efi').read_text() == 'loader'


def test_contradictory_patch_on_an_aliased_target_fails_at_verify(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""D x/f + A y/f where y is a symlink to x: names that differ by more than case. The write is restored (the content
	is kept) and the delete's verify fails: loud, not silent."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'x' / 'f', 'moved')
	os.symlink('x', str(target / 'y'))
	_write(vol / 'y' / 'f', 'moved')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('D', 'x/f', state(str(target / 'x' / 'f')), '-'), R('A', 'y/f', '-', state(str(vol / 'y' / 'f')))]
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary['restored'] == 1 and summary['failed'] == 1
	assert (target / 'x' / 'f').read_text() == 'moved'
	assert len(hpcp_mod.ERRORS) == 1 and hpcp_mod.ERRORS[0].startswith('Patch verify error: D x/f: expected -')


def test_entry_lost_to_a_forced_type_change_is_reported(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""-fa replacing directory x with a file removes x's tree; y/g, reached through the symlink y -> x and skipped as
	already done, goes with it and cannot be restored (y now points at a file): a failure, not a silent loss."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'x' / 'g', 'g')
	os.symlink('x', str(target / 'y'))
	_write(vol / 'x', 'now a file')
	_write(vol / 'y' / 'g', 'g')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('A', 'x', '-', state(str(vol / 'x'))), R('A', 'y/g', '-', state(str(vol / 'y' / 'g')))]
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	assert {check.row.path: check.action for check in plan.checks} == {'x': 'conflict', 'y/g': 'skip'}
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary['forced'] == 1 and summary['restored'] == 1 and summary['failed'] >= 1
	assert (target / 'x').read_text() == 'now a file'
	assert any('y/g' in error for error in hpcp_mod.ERRORS)


def test_restore_without_a_payload_is_a_failure(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 2: a skipped row's payload is not pre-checked (nothing is written for it normally); when it must be
	restored and the volume lacks it, that is an error and a verify failure."""
	hpcp_mod.HASH_SIZE = 65536
	target, vol, rows = _restore_scenario(hpcp_mod, tmp_path)
	os.unlink(str(vol / 'EFI' / 'Boot' / 'x.efi'))
	plan = hpcp_mod._plan_section('root', rows, [str(vol)], str(target), True, False, 1)
	summary = hpcp_mod._apply_section(plan, 1, False, False, True)
	assert summary['restored'] == 1 and summary['failed'] == 1
	assert any(error.startswith('Incomplete patch: payload for EFI/Boot/x.efi') for error in hpcp_mod.ERRORS)


def test_rerun_after_a_completed_apply_restores_nothing(trees, hpcp_mod, tmp_path):
	new, base = trees
	_write(base / 'gone.txt', 'bye')
	_write(new / 'kept.txt', 'kept')
	_write(base / 'kept.txt', 'kept')
	_write(new / 'add.txt', 'add')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	first = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (first['deleted'], first['added'], first['restored'], first['failed']) == (1, 1, 0, 0)
	again = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (again['deleted'], again['added'], again['restored'], again['failed']) == (0, 0, 0, 0)


def test_fold_name_keeps_names_fat_keeps_apart(hpcp_mod):
	fold = hpcp_mod._fold_name
	assert fold('EFI/Boot/bootx64.efi') == fold('efi/BOOT/BOOTX64.EFI') == 'EFI/BOOT/BOOTX64.EFI'
	# 'ß'.upper() is 'SS' (two characters): kept, so 'Straße' and 'STRASSE' stay two names, as FAT keeps them.
	assert fold('Straße') == 'STRAßE' != fold('STRASSE')
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "restore or contradictory or forced_type_change or fold_name or restores_nothing"`
Expected: FAIL — `KeyError: 'restored'` / `AttributeError: module 'hpcp' has no attribute '_fold_name'`.

- [ ] **Step 4: Add the helpers**

Just above `def _under_removed` add:

```python
def _fold_name(name):
	"""name as a case-insensitive filesystem (FAT, exFAT, HFS+) compares it: each character upper-cased when that
	gives exactly one character, otherwise kept, so 'ß' stays apart from 'SS' as FAT keeps them."""
	return ''.join(char.upper() if len(char.upper()) == 1 else char for char in name)
```

Just above `def _apply_rows` add:

```python
def _holds_type(path, state):
	"""True when path exists (its last component not followed) with the entry type of state: 'd:', 'f:' or 'l:'."""
	try:
		mode = os.lstat(path).st_mode
	except OSError:
		return False
	return {'d': stat.S_ISDIR, 'f': stat.S_ISREG, 'l': stat.S_ISLNK}[state[0]](mode)

def _same_entry(path, others):
	"""True when path and one of others are one entry: same st_dev and st_ino (last components not followed)."""
	try:
		info = os.lstat(path)
	except OSError:
		return False
	for other in others:
		if other == path:
			continue
		try:
			found = os.lstat(other)
		except OSError:
			continue
		if (found.st_dev, found.st_ino) == (info.st_dev, info.st_ino):
			return True
	return False
```

- [ ] **Step 5: Replace `_apply_rows`**

Replace the whole function with:

```python
def _apply_rows(rows, volume_roots, target_root, max_workers=1, verbose=False, full_hash=False, forced_rows=(), metadata=True,
		skipped=()):
	"""Carry out checked rows: deletions, then directories, files, symlinks, directory metadata.

	Containment of every target path is re-checked immediately before each write, and the
	target root itself is never removed. A directory is removed with rmdir, so an entry that
	appeared after the check survives (and verify reports the row); only a row in
	forced_rows (a conflict the user forced) may remove a whole tree. Files and symlinks are
	built at a temporary sibling and renamed over the target (see _replace_files), so an
	existing file or symlink is never removed before its replacement exists. Only an entry
	a rename cannot replace is removed first: a directory where a file or symlink goes, or
	a file or symlink where a directory goes.
	skipped: the A / M rows check_patch_target found already done. When the removals took anything
	away, each one whose target is now missing or of another type (a delete that reached it through a
	name differing only by case, or through a symlinked directory) is written again like any row.
	Returns those restored rows.
	With metadata False (a FAT / exFAT / NTFS target) only content and file times are
	written: no mode, owner, xattrs or directory metadata (new directories get the target's
	default mode).
	"""
	root_norm = os.path.normpath(target_root)
	forced_rows = set(forced_rows)
	removed = []

	def guarded(row, removing=False):
		"""Target path for row, or None (with an error) when it must not be touched."""
		path = _key_path(target_root, row.path)
		if removing and os.path.normpath(path) == root_norm:
			eprint(f'Patch format error: {row.path} would remove the target root; skipped')
			return None
		if row.path != './' and not _inside_root(target_root, path):
			eprint(f'Patch format error: {row.path} resolves outside {target_root}; skipped')
			return None
		return path

	def is_real_dir(path):
		return os.path.isdir(path) and not os.path.islink(path)

	def remove(row):
		"""Unlink row's target, rmdir it, or (forced rows only) remove its whole tree."""
		path = guarded(row, removing=True)
		if path is None:
			return
		directory = is_real_dir(path)
		try:
			if directory and row in forced_rows:
				shutil.rmtree(path)
			elif directory:
				os.rmdir(path)
			else:
				os.unlink(path)
			removed.append(row.path)
		except FileNotFoundError:
			pass
		except OSError as e:
			eprint(f"Remove {'directory' if directory else 'file'} failed: {row.path}: {e}")

	def remove_other_type(row):
		"""Remove what a rename cannot replace: a directory where a file or symlink goes, or the reverse."""
		if row.path == './':
			return
		path = guarded(row)
		if path is not None and os.path.lexists(path) and row.new.startswith('d:') != is_real_dir(path):
			remove(row)

	def sync_dir(row):
		"""Copy volume 1's directory metadata onto the target directory, with containment checks."""
		path = guarded(row)
		if path is None:
			return
		if row.path != './' and not is_real_dir(path):
			eprint(f'Patch format error: {row.path} is not a real directory on the target; metadata skipped')
			return
		source = _key_path(volume_roots[0], row.path)
		if row.path != './' and (not is_real_dir(source) or not _inside_root(volume_roots[0], source)):
			eprint(f'Patch format error: directory metadata source for {row.path} is not a real directory in volume 1; skipped')
			return
		try:
			sync_directory_metadata(source, [path])
		except OSError as e:
			eprint(f'Copy failed: {row.path}: {e}')

	deletes = [row for row in rows if row.op == 'D']
	payload = [row for row in rows if row.op in ('A', 'M')]
	# Non-directories first, then directories deepest first.
	for row in sorted(deletes, key=lambda r: (r.path.endswith('/'), -_key_depth(r.path))):
		remove(row)
	for row in payload:
		remove_other_type(row)
	restored = []
	if removed:
		restored = [row for row in skipped if row.op != 'D' and row.path != './'
			and not _holds_type(_key_path(target_root, row.path), row.new)]
		for row in restored:
			remove_other_type(row)
		if restored:
			print(f"Restoring {len(restored)} entr{'y' if len(restored) == 1 else 'ies'} skipped as already done that a removal "
				f"took away, e.g. {restored[0].path}")
		payload = payload + restored
	dir_rows = []
	for row in sorted((r for r in payload if r.new.startswith('d:')), key=lambda r: _key_depth(r.path)):
		path = guarded(row)
		if path is None:
			continue
		try:
			# 0700 until sync_dir sets the payload's metadata; a content-only target keeps its default mode.
			os.makedirs(path, mode=0o700 if metadata else 0o777, exist_ok=True)
		except OSError as e:
			eprint(f'Copy failed: {row.path}: {e}')
			continue
		dir_rows.append(row)
	files = []
	for row in payload:
		if not row.new.startswith('f:'):
			continue
		path = guarded(row)
		if path is None:
			continue
		source = _payload_source(volume_roots, row)
		if source is None:
			eprint(f'Incomplete patch: payload for {row.path} is missing or unsafe in every volume')
			continue
		files.append((row, source, path))
	_replace_files(files, target_root, max_workers, verbose, full_hash, metadata)
	for row in payload:
		if not row.new.startswith('l:') or row.path == './':
			continue
		path = guarded(row, removing=True)
		if path is not None:
			_replace_with_symlink(row, decode_state(row.new)['target'], path, target_root)
	if metadata:
		for row in sorted(dir_rows, key=lambda r: _key_depth(r.path), reverse=True):
			sync_dir(row)
	return restored
```

- [ ] **Step 6: Replace `_verify_rows`**

```python
def _verify_rows(rows, target_root, compare_hash=True, full_hash=False, max_workers=1, metadata=True):
	"""Re-read every applied path and compare with its final expected state (content only with
	metadata False); returns the failure count.

	A D row whose path still exists passes when an A / M row's name differs from it only by case
	(_fold_name) and both paths are one entry (same st_dev / st_ino): a case-only rename on a
	case-insensitive target, whether or not hpcp recognised the filesystem as one.
	"""
	final = {}
	for row in sort_change_rows(rows):
		# D sorts before A, so for a D + A pair on one name the A row decides the final state.
		final[row.path.rstrip('/')] = row
	paths = {name: _key_path(target_root, row.path) for name, row in final.items()}
	states = _states_for_paths(set(paths.values()), full_hash=full_hash, read_xattrs=metadata, max_workers=max_workers)
	written = {}
	for name, row in final.items():
		if row.op != 'D':
			written.setdefault(_fold_name(name), []).append(paths[name])
	failed = 0
	for name, row in final.items():
		expected = '-' if row.op == 'D' else row.new
		found = states[paths[name]]
		ok = found == '-' if expected == '-' else states_equal(found, expected, compare_hash, metadata)
		if not ok and expected == '-' and _same_entry(paths[name], written.get(_fold_name(name), ())):
			ok = True
		if not ok:
			failed += 1
			eprint(f'Patch verify error: {row.op} {row.path}: expected {expected}, found {found}')
	return failed
```

- [ ] **Step 7: Replace `_apply_section`**

```python
def _apply_section(plan, max_workers, verbose, full_hash, compare_hash):
	"""Apply a checked section and verify it (with the rows the repair step restored); returns its summary."""
	conflicts = _section_conflicts(plan)
	todo = _section_todo(plan)
	skipped = [check.row for check in plan.checks if check.action == 'skip' and check.row.op != 'D']
	restored = _apply_rows(todo, plan.volume_roots, plan.target_root, max_workers, verbose, full_hash,
		forced_rows=[check.row for check in conflicts], metadata=plan.metadata, skipped=skipped)
	failed = _verify_rows(todo + restored, plan.target_root, compare_hash, full_hash, max_workers, plan.metadata)
	return {
		'deleted': sum(1 for row in todo if row.op == 'D'),
		'added': sum(1 for row in todo if row.op == 'A'),
		'modified': sum(1 for row in todo if row.op == 'M'),
		'skipped': len(plan.checks) - len(todo) - len(restored),
		'restored': len(restored),
		'forced': len(conflicts),
		'failed': failed,
	}
```

In `apply_disk_patch`, change the per-section print to include the count (summaries faked by tests may lack the key):

```python
		print(f"Section {name}: deleted {item['deleted']}, added {item['added']}, modified {item['modified']}, "
			f"skipped {item['skipped']}, restored {item.get('restored', 0)}, forced {item['forced']}, failed {item['failed']}")
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q`
Expected: all pass, except root tests that compare a whole section summary dict (`test_esp_case_only_rename_keeps_the_files` asserts a dict without `restored`) — that test is rewritten in Task 2; if it is the only failure, note it in your report and continue.

- [ ] **Step 9: Commit**

```bash
git add hpcp.py tests/test_diff_image.py tests/test_disk_patch.py
git commit -m "fix(diff-image): restore skipped writes a removal took away; verify case-only renames by entry"
```

---

### Task 2: Case-insensitive planning, exact-name generation, HFSX

**Files:**
- Modify: `hpcp.py` — `check_patch_target` (≈ 4637), `_unlisted_children` (≈ 4618); add `_spelled_on_disk` just above `def check_patch_target`; `_has_case_insensitive_names` (≈ 5975); `_partition_changes` (≈ 6238) and its call in `process_get_disk_diff_image` (≈ 6357); `apply_disk_patch` (remove the old-patch collision loop ≈ 6505–6512; pass the partition device when planning ≈ 6575); delete `_case_folded_names`, `_fold_case_keys`, `_case_collision`.
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: `_fold_name` (Task 1).
- Produces:
  - `_spelled_on_disk(target_root: str, name: str, listings: dict) -> str | None`
  - `check_patch_target(rows, target_root, compare_hash=True, full_hash=False, max_workers=1, metadata=True, case_insensitive=False)` (signature unchanged)
  - `_has_case_insensitive_names(fs_type: str, device: str = '') -> bool`
  - `_partition_changes(new_root, old_root, mount_points, live_mounts, exclude, full_hash, max_workers, parallel_file_listing, input_files=(), metadata=True)`

- [ ] **Step 1: Delete the folding tests and rewrite the old-patch refusal test**

In `tests/test_disk_patch.py` delete: `_case_changes`, `test_content_only_sections_pair_names_case_insensitively`, `test_names_differing_only_by_case_refuse_content_only_generation`, `test_case_collision_finds_deletes_and_writes_that_differ_only_by_case` (with its parametrize decorator), and `test_case_colliding_rows_are_refused_only_on_case_insensitive_filesystems` (with its parametrize decorator). Keep `_patch_row`. Then add:

```python
@pytest.mark.parametrize('fs_type', ['vfat', 'exfat', 'hfsplus', 'ntfs', 'ext4'])
def test_case_only_rename_rows_pass_the_disk_patch_checks(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, fs_type):
	"""A delete of EFI/BOOT/x.efi and an add of EFI/Boot/x.efi is an ordinary case-only rename: no refusal before the
	target is opened, whatever the section's filesystem."""
	rows = [_patch_row(hpcp_mod, 'D', 'EFI/BOOT/x.efi'), _patch_row(hpcp_mod, 'A', 'EFI/Boot/x.efi')]
	volume, _ = _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch, sections=[('part1', rows)], kinds={1: ('fs', fs_type, 32 << 20)})
	with pytest.raises(AssertionError, match='the target was opened before the patch was checked'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
def test_generation_pairs_names_by_exact_case_in_every_section(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""A case-only rename is a delete plus an add, for content-only (FAT) and POSIX sections alike."""
	hpcp_mod.HASH_SIZE = 65536
	new, base = tmp_path / 'new', tmp_path / 'base'
	_write(base / 'EFI' / 'BOOT' / 'bootx64.efi', 'shim')
	_write(new / 'EFI' / 'Boot' / 'BOOTX64.EFI', 'shim')
	for metadata in (True, False):
		rows = hpcp_mod._partition_changes(str(new), str(base), [], [], None, False, 1, False, metadata=metadata)
		assert {(row.op, row.path) for row in rows} == {('D', 'EFI/BOOT/'), ('D', 'EFI/BOOT/bootx64.efi'),
			('A', 'EFI/Boot/'), ('A', 'EFI/Boot/BOOTX64.EFI')}


def test_spelled_on_disk_reads_the_targets_own_spelling(tmp_path, hpcp_mod):
	_write(tmp_path / 'EFI' / 'Boot' / 'BOOTX64.EFI', 'x')
	listings = {}
	assert hpcp_mod._spelled_on_disk(str(tmp_path), 'efi/BOOT/bootx64.efi', listings) == 'EFI/Boot/BOOTX64.EFI'
	assert hpcp_mod._spelled_on_disk(str(tmp_path), 'EFI/missing', listings) is None


def test_case_insensitive_planning_writes_where_a_differently_spelled_name_is_deleted(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""On FAT the add of EFI/Boot/x.efi finds the old file under the other spelling. Matched case-insensitively, the
	delete of EFI/BOOT/x.efi removes that name first, so the add counts as absent: a write, not an 'already exists'
	conflict. (A case-sensitive test directory holds both spellings to show the matching.)"""
	hpcp_mod.HASH_SIZE = 65536
	target, vol = tmp_path / 'target', tmp_path / 'vol'
	_write(target / 'EFI' / 'BOOT' / 'x.efi', 'old')
	_write(target / 'EFI' / 'Boot' / 'x.efi', 'old')
	_write(vol / 'EFI' / 'Boot' / 'x.efi', 'new content')
	state = hpcp_mod.get_entry_state
	R = hpcp_mod.ChangeRow
	rows = [R('D', 'EFI/BOOT/x.efi', state(str(target / 'EFI' / 'BOOT' / 'x.efi')), '-'),
		R('A', 'EFI/Boot/x.efi', '-', state(str(vol / 'EFI' / 'Boot' / 'x.efi')))]

	def actions(case_insensitive):
		checks = hpcp_mod.check_patch_target(rows, str(target), case_insensitive=case_insensitive)
		return {check.row.path: (check.action, check.reason) for check in checks}

	assert actions(True) == {'EFI/BOOT/x.efi': ('apply', ''), 'EFI/Boot/x.efi': ('apply', '')}
	assert actions(False)['EFI/Boot/x.efi'] == ('conflict', 'already exists on target')


def test_hfsx_is_case_sensitive(tmp_path, hpcp_mod):
	device = tmp_path / 'hfs.img'
	device.write_bytes(b'\0' * 1024 + b'HX' + b'\0' * 510)
	assert hpcp_mod._has_case_insensitive_names('hfsplus', str(device)) is False
	device.write_bytes(b'\0' * 1024 + b'H+' + b'\0' * 510)
	assert hpcp_mod._has_case_insensitive_names('hfsplus', str(device)) is True
	assert hpcp_mod._has_case_insensitive_names('hfsplus', str(tmp_path / 'missing')) is True
	assert hpcp_mod._has_case_insensitive_names('hfsplus') is True
	assert hpcp_mod._has_case_insensitive_names('vfat', str(device)) is True
	assert hpcp_mod._has_case_insensitive_names('ext4', str(device)) is False


@pytest.mark.skipif(shutil.which('mkfs.hfsplus') is None, reason='needs mkfs.hfsplus')
def test_hfsx_made_by_mkfs_is_case_sensitive(tmp_path, hpcp_mod):
	for flags, insensitive in (([], True), (['-s'], False)):
		image = tmp_path / f'hfs{len(flags)}.img'
		with open(str(image), 'wb') as f:
			f.truncate(32 << 20)
		subprocess.run(['mkfs.hfsplus'] + flags + [str(image)], check=True, stdout=subprocess.DEVNULL)
		assert hpcp_mod._has_case_insensitive_names('hfsplus', str(image)) is insensitive
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "exact_case or spelled_on_disk or differently_spelled or hfsx or pass_the_disk_patch_checks"`
Expected: FAIL — `_partition_changes` still folds (rows paired), `_spelled_on_disk` missing, `_has_case_insensitive_names` takes one argument, the disk apply still refuses the D + A pair.

- [ ] **Step 4: Implement planning**

Just above `def check_patch_target` add:

```python
def _spelled_on_disk(target_root, name, listings):
	"""name (a key without its trailing '/') as the target spells it, each component matched with _fold_name;
	None when a component is missing. listings caches {directory: {folded name: name}}."""
	directory, spelled = os.path.normpath(target_root), []
	for part in name.split('/'):
		if directory not in listings:
			try:
				listings[directory] = {_fold_name(entry): entry for entry in os.listdir(directory)}
			except OSError:
				listings[directory] = {}
		entry = listings[directory].get(_fold_name(part))
		if entry is None:
			return None
		spelled.append(entry)
		directory = os.path.join(directory, entry)
	return '/'.join(spelled)
```

Replace `_unlisted_children` with:

```python
def _unlisted_children(key, path, deleted_keys, case_insensitive=False):
	"""Children of directory path whose keys are not in deleted_keys; None when path cannot be listed.

	case_insensitive: deleted_keys are folded (_fold_name) and so is each child's key (a target that looks
	names up case-insensitively may spell an entry as it did before a case-only rename).
	"""
	prefix = '' if key == './' else key
	try:
		names = os.listdir(path)
	except OSError:
		return None
	unlisted = []
	for name in names:
		child = os.path.join(path, name)
		child_key = prefix + name + ('/' if os.path.isdir(child) and not os.path.islink(child) else '')
		if (_fold_name(child_key) if case_insensitive else child_key) not in deleted_keys:
			unlisted.append(child_key)
	return unlisted
```

Replace `check_patch_target` with:

```python
def check_patch_target(rows, target_root, compare_hash=True, full_hash=False, max_workers=1, metadata=True, case_insensitive=False):
	"""Classify every row against target_root as 'apply', 'skip' or 'conflict'.

	D: target == old -> apply; absent, or already the replacing A row's state -> skip.
	A: absent -> apply; == new -> skip. M: == old -> apply; == new -> skip.
	Anything else is a conflict, as is a D directory holding entries the patch does not
	delete or one that cannot be listed.
	A and M rows are judged as the target will be after the applicable D rows: a name a D
	row removes (with or without the trailing '/'), and everything below it, counts as absent.
	metadata False compares content only (see states_equal). case_insensitive (a FAT, exFAT or HFS+
	target, which looks names up case-insensitively) compares names with _fold_name: a D directory's
	entries, the name a D row removes, and the A row that replaces a D row. A case-only rename (D on one
	spelling, A on another) is then a delete and a write; once applied (the target already spells the A
	row's name) both rows are skipped.
	"""
	fold = _fold_name if case_insensitive else (lambda name: name)
	paths = {row.path: _key_path(target_root, row.path) for row in rows}
	states = _states_for_paths(set(paths.values()), full_hash=full_hash, read_xattrs=metadata, max_workers=max_workers)
	same = functools.partial(states_equal, compare_hash=compare_hash, metadata=metadata)
	added_by_name = {fold(row.path.rstrip('/')): row for row in rows if row.op == 'A'}
	deleted_keys = {fold(row.path) for row in rows if row.op == 'D'}
	listings = {}
	checks = {}
	removed_names = set()
	for row in rows:
		if row.op != 'D':
			continue
		name = row.path.rstrip('/')
		found = states[paths[row.path]]
		replacement = added_by_name.get(fold(name))
		if replacement is not None and replacement.path.rstrip('/') != name:
			# A case-only rename: only the target's own spelling shows whether it already happened.
			if _spelled_on_disk(target_root, name, listings) != replacement.path.rstrip('/'):
				replacement = None
		if found == '-':
			checks[row] = PatchCheck(row, 'skip', found, 'already absent')
			continue
		if replacement is not None and same(found, replacement.new):
			checks[row] = PatchCheck(row, 'skip', found, 'already replaced')
			continue
		removed_names.add(fold(name))
		if not same(found, row.old):
			checks[row] = PatchCheck(row, 'conflict', found, 'changed on target')
			continue
		unlisted = _unlisted_children(row.path, paths[row.path], deleted_keys, case_insensitive) if found.startswith('d:') else []
		if unlisted is None:
			checks[row] = PatchCheck(row, 'conflict', found, 'cannot list directory')
		elif unlisted:
			checks[row] = PatchCheck(row, 'conflict', found, 'directory holds entries the patch does not delete')
		else:
			checks[row] = PatchCheck(row, 'apply', found, '')
	for row in rows:
		if row.op == 'D':
			continue
		found = '-' if _under_removed(fold(row.path), removed_names) else states[paths[row.path]]
		if row.op == 'A':
			if found == '-':
				checks[row] = PatchCheck(row, 'apply', found, '')
			elif same(found, row.new):
				checks[row] = PatchCheck(row, 'skip', found, 'already in new state')
			else:
				checks[row] = PatchCheck(row, 'conflict', found, 'already exists on target')
		elif same(found, row.old):
			checks[row] = PatchCheck(row, 'apply', found, '')
		elif same(found, row.new):
			checks[row] = PatchCheck(row, 'skip', found, 'already in new state')
		else:
			checks[row] = PatchCheck(row, 'conflict', found, 'missing on target' if found == '-' else 'changed on target')
	return [checks[row] for row in rows]
```

- [ ] **Step 5: HFSX, exact-name generation, disk apply**

Replace `_has_case_insensitive_names` with:

```python
def _has_case_insensitive_names(fs_type, device=''):
	"""True for a filesystem whose Linux driver looks names up case-insensitively: the FAT family, exFAT, HFS and
	HFS+, except case-sensitive HFSX (an hfsplus volume whose header signature, 2 bytes at offset 1024 of device,
	is 'HX'). An unreadable header counts as case-insensitive."""
	if fs_type not in _CASE_INSENSITIVE_TYPES:
		return False
	if fs_type == 'hfsplus' and device:
		try:
			with open(device, 'rb') as f:
				f.seek(1024)
				return f.read(2) != b'HX'
		except OSError:
			pass
	return True
```

Delete `_case_folded_names`, `_fold_case_keys` and `_case_collision` entirely. Replace `_partition_changes` with:

```python
def _partition_changes(new_root, old_root, mount_points, live_mounts, exclude, full_hash, max_workers, parallel_file_listing,
		input_files=(), metadata=True):
	"""Change rows of one filesystem partition (root to root; lost+found, this run's mounts and
	its input disk-image files left out). metadata False: content-only rows (see states_equal).
	Names are compared by exact case in every section: a case-only rename is a delete plus an add."""
	matched = os.path.abspath(new_root)
	patterns = _mount_point_excludes(mount_points, (new_root, old_root), matched)
	for live_mount in live_mounts:
		if not live_mount:
			continue
		# A live filesystem holds the (empty) directories of this run's other temp mount points,
		# and may hold the disk images this run compares (the old golden image on the new golden).
		live_real = os.path.realpath(live_mount)
		for path in list(mount_points) + list(input_files):
			path_real = os.path.realpath(path)
			if path_real.startswith(live_real.rstrip(os.sep) + os.sep):
				patterns.add(glob.escape(os.path.join(matched, os.path.relpath(path_real, live_real))))
	new_map, base_map = build_root_maps(new_root, old_root, frozenset(exclude or ()) | patterns, max_workers, parallel_file_listing)
	new_map, base_map = _drop_lost_found(new_map), _drop_lost_found(base_map)
	if any(_is_reserved_key(key) for key in new_map):
		raise RuntimeError(f'Diff image error: {new_root} contains {DIFF_DIR_NAME}, which hpcp reserves for patch metadata')
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata)
	_validate_patch_rows(rows)
	return rows
```

In `process_get_disk_diff_image`, the call becomes:

```python
				rows = _partition_changes(new_root, old_root, mount_points, lives, exclude, full_hash, max_workers, parallel_file_listing,
					input_files, metadata=_has_posix_metadata(new_part['fs_type']))
```

In `apply_disk_patch`, delete the loop that starts with the comment `# Patches from older hpcp turned a case-only rename into a D + A pair` (the `for name, _, rows in sections:` loop that calls `_case_collision` and raises `Patch format error: section ... differing only by case`). Keep the `fs_types = {...}` line above it. In the planning loop, pass the partition device:

```python
			plans.append(_plan_section(name, rows, _section_volume_roots(volume_roots, payload_dir), root, compare_hash, full_hash, max_workers,
				_has_posix_metadata(fs_type), _has_case_insensitive_names(fs_type, part['device'])))
```

Update the comment above `_CASE_INSENSITIVE_TYPES` to: `# Filesystems whose Linux drivers look names up case-insensitively (ntfs3 and ntfs-3g use the exact case; hfsplus folds case unless it is HFSX, and keeps owner, group and mode on disk).`

- [ ] **Step 6: Rewrite the two ESP root tests**

In `tests/test_disk_patch.py` replace `test_esp_case_only_rename_keeps_the_files` with:

```python
@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_esp_case_only_rename_applies_with_the_new_spelling(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""EFI/BOOT -> EFI/Boot (and two file renames by case) is a delete plus an add per entry. On the node's vfat ESP,
	matched case-insensitively, the deletes run and the adds write the new spelling; a re-run skips every row."""
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_add_esp_file(old, 'EFI/BOOT/bootx64.efi', b'shim-v1')
	_make_newer(old, new, label=None, raw=False, mbr=False, gpt_name=None)
	listing = _rename_on_esp(new, [('EFI/BOOT', 'EFI/Boot'), ('EFI/Boot/bootx64.efi', 'EFI/Boot/BOOTX64.EFI'),
		('EFI/Boot/grubx64.efi', 'EFI/Boot/GRUBX64.efi')])
	assert listing == ['EFI', 'EFI/Boot', 'EFI/Boot/BOOTX64.EFI', 'EFI/Boot/GRUBX64.efi']
	with _session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_disk_diff_image([str(new)], [str(old)], out, mount_points, loop_devices, max_workers=1) == out
	_, part1 = _read_part3_patch(hpcp_mod, out, 'part1')
	assert {(row.op, row.path) for row in part1} == {('D', 'EFI/BOOT/'), ('D', 'EFI/BOOT/bootx64.efi'), ('D', 'EFI/BOOT/grubx64.efi'),
		('A', 'EFI/Boot/'), ('A', 'EFI/Boot/BOOTX64.EFI'), ('A', 'EFI/Boot/GRUBX64.efi')}
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	summary = _apply(hpcp_mod, out, target)
	assert summary['failed'] == 0 and hpcp_mod.ERRORS == []
	assert summary['sections']['part1'] == {'deleted': 3, 'added': 3, 'modified': 0, 'skipped': 0, 'restored': 0, 'forced': 0, 'failed': 0}
	assert _esp_files(target) == {'EFI/Boot/BOOTX64.EFI': b'shim-v1', 'EFI/Boot/GRUBX64.efi': b'new-efi-binary'}
	again = _apply(hpcp_mod, out, target)
	assert again['failed'] == 0 and again['sections']['part1']['skipped'] == 6
	assert _no_differences(hpcp_mod, tmp_path, new, target)
```

Replace `test_hand_made_case_colliding_delete_and_add_is_refused_before_anything_changes` with:

```python
@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_hand_made_case_only_rename_with_new_content_applies_on_the_esp(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""A patch that deletes EFI/BOOT/grubx64.efi and adds EFI/Boot/grubx64.efi with new content (older hpcp, or written
	by hand): on the vfat ESP the add is planned as a write after the delete, not an 'already exists' conflict."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	with _attached(patch) as loop:
		with _mounted(loop) as root:
			changes = os.path.join(root, '.hpcp_diff', 'part1.changes.tsv')
			(row,) = hpcp_mod.read_changes(changes)
			assert (row.op, row.path) == ('M', 'EFI/BOOT/grubx64.efi')
			R = hpcp_mod.ChangeRow
			hpcp_mod.write_changes(changes, [R('D', row.path, row.old, '-'), R('A', 'EFI/Boot/grubx64.efi', '-', row.new)])
			# The patch volume is case-sensitive: the add's payload lives under its own spelling.
			os.makedirs(os.path.join(root, 'part1', 'EFI', 'Boot'))
			os.rename(os.path.join(root, 'part1', 'EFI', 'BOOT', 'grubx64.efi'), os.path.join(root, 'part1', 'EFI', 'Boot', 'grubx64.efi'))
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	summary = _apply(hpcp_mod, patch, target)
	assert summary['failed'] == 0 and hpcp_mod.ERRORS == []
	assert (summary['sections']['part1']['deleted'], summary['sections']['part1']['added']) == (1, 1)
	files = {name.lower(): data for name, data in _esp_files(target).items()}
	assert files['efi/boot/grubx64.efi'] == b'new-efi-binary'
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q`
Expected: all pass (root tests included). Then run `python -m pytest tests/test_disk_patch.py -q -k "esp_case_only_rename or hand_made_case_only"` three times in a row; all three runs must pass. Check `losetup -a` (apart from snap loops) and `findmnt | grep /tmp/tmp` show nothing left behind.

- [ ] **Step 8: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd-diff): exact names at generation, case-insensitive planning on FAT/exFAT/HFS+, HFSX detection"
```

---

### Task 3: Single-filesystem mount scopes, refusals removed, prompt order, content-only generation

**Files:**
- Modify: `hpcp.py` — replace `_path_mounts` (≈ 5913) with `_mount_scopes`; add `_rows_by_scope` and `_compare_metadata` after it; rewrite `apply_patch_tree` (≈ 5298); `_preflight_section` (≈ 5237); `apply_disk_patch` prompt order (≈ 6578); `process_get_diff_image` (≈ 5398); delete `_refuse_case_only_renames`, `_CASE_SENSITIVE_TYPES`, `_same_entry_pair`, `_caseless`, and `import unicodedata`.
- Test: `tests/test_diff_image.py`, `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: Task 1 (`restored` summary key), Task 2 (`_has_case_insensitive_names(fs_type, device)`).
- Produces:
  - `_mount_scopes(path: str, entries: list) -> list[tuple[str, str, str]]` — `[(key prefix, fstype, source)]`, the target's own scope first with prefix `''`
  - `_rows_by_scope(rows, scopes) -> dict[str, list[ChangeRow]]`
  - `_compare_metadata(roots, entries) -> bool`
  - `apply_patch_tree(...)` summary dict gains `restored`

- [ ] **Step 1: Delete the refusal tests**

In `tests/test_diff_image.py` delete: `_case_conflict`, `_unknown_case_conflict`, `test_single_fs_apply_refuses_case_only_rename_on_case_insensitive_target` (with its parametrize decorator), `test_single_fs_apply_finds_the_target_mount_by_longest_prefix`, `test_single_fs_apply_refuses_case_only_rename_under_a_case_insensitive_mount_below_the_target` (with its parametrize decorator), `_same_entry_conflict`, `test_apply_refuses_to_delete_and_write_one_entry_reached_through_an_alias` (with its parametrize decorator), `test_apply_refuses_to_write_into_a_directory_it_deletes_through_an_alias`, `test_case_only_rename_patch_is_refused_on_a_vfat_image` and `test_case_only_rename_is_refused_on_a_vfat_image_whatever_its_reported_type` (with their decorators). Keep `_case_only_rename`, `_moved_file`, `_fake_mounts`. In `tests/test_disk_patch.py` delete `test_section_preflight_refuses_a_delete_and_a_write_of_one_entry`.

In `tests/test_disk_patch.py`, `test_disk_apply_plans_each_section_by_filesystem_class` stops the apply at the conflict prompt, which now runs after the pre-flight (whose payload check would fail on this hand-made patch). Change its stop point: replace

```python
	monkeypatch.setattr(hpcp_mod, '_resolve_section_conflicts', stop)
```

with

```python
	monkeypatch.setattr(hpcp_mod, '_preflight_section', lambda plan: stop([plan], False))
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_diff_image.py`:

```python
@pytest.mark.parametrize('fs_type', ['vfat', 'exfat', 'hfsplus', 'fuseblk', 'cifs', '', 'ext4', 'nfs4'])
def test_single_fs_case_only_rename_applies_whatever_the_reported_type(trees, hpcp_mod, tmp_path, reset_hpcp_globals,
		monkeypatch, fs_type):
	"""A case-only rename is a delete plus an add; it applies on any filesystem (no refusal, -fa or not)."""
	new, base = trees
	_case_only_rename(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('D', 'EFI/BOOT/'), ('D', 'EFI/BOOT/BOOTX64.EFI'), ('A', 'EFI/Boot/'), ('A', 'EFI/Boot/BOOTX64.EFI'),
		('M', 'EFI/grub.cfg')}
	target = tmp_path / 'target'
	_clone(base, target)
	_fake_mounts(hpcp_mod, monkeypatch, (target, fs_type))
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (summary['deleted'], summary['added'], summary['modified'], summary['failed']) == (2, 2, 1, 0)
	assert _changes(hpcp_mod, new, target) == []


def _mount_entry(point, fs_type, source='none'):
	return {'devnum': (0, 1), 'root': '/', 'mount_point': os.path.realpath(str(point)), 'options': ['rw'],
		'fstype': fs_type, 'source': source}


def test_mount_scopes_follow_mountinfo_order(tmp_path, hpcp_mod):
	"""The deepest visible mount holding the real path is the target's; of two on one mount point the later is on top,
	and a later mount on a parent hides earlier ones below it. Mounts below the target come after, deepest last."""
	scopes = hpcp_mod._mount_scopes
	esp = tmp_path / 'mnt' / 'esp'
	esp.mkdir(parents=True)
	link = tmp_path / 'link'
	os.symlink(str(esp), str(link))
	assert scopes(str(link), [_mount_entry('/', 'ext4'), _mount_entry(tmp_path / 'mnt', 'xfs'), _mount_entry(esp, 'vfat')])[0][1] == 'vfat'
	assert scopes(str(esp), [_mount_entry('/', 'ext4'), _mount_entry(esp, 'ext4'), _mount_entry(esp, 'vfat')])[0][1] == 'vfat'
	# The xfs on mnt, mounted later, hides the vfat at mnt/esp.
	assert scopes(str(esp), [_mount_entry('/', 'ext4'), _mount_entry(esp, 'vfat'), _mount_entry(tmp_path / 'mnt', 'xfs')])[0][1] == 'xfs'
	entries = [_mount_entry(tmp_path, 'ext4', '/dev/sda2'), _mount_entry(tmp_path / 'boot' / 'efi', 'vfat', '/dev/sda1'),
		_mount_entry(tmp_path / 'boot' / 'efi' / 'deeper', 'exfat'), _mount_entry(tmp_path / 'a' / 'b', 'vfat'),
		_mount_entry(tmp_path / 'a', 'tmpfs')]
	assert scopes(str(tmp_path), entries) == [('', 'ext4', '/dev/sda2'), ('a', 'tmpfs', 'none'), ('boot/efi', 'vfat', '/dev/sda1'),
		('boot/efi/deeper', 'exfat', 'none')]
	assert scopes(str(tmp_path), []) == [('', '', '')]


def test_rows_by_scope_uses_the_deepest_mount(hpcp_mod):
	R = hpcp_mod.ChangeRow
	rows = [R('M', './', 'd:40755:0:0:-', 'd:40755:0:0:-'), R('A', 'etc/x', '-', 'f:1:1:100644:0:0:-:-'),
		R('M', 'boot/efi/', 'd:40755:0:0:-', 'd:40755:0:0:-'), R('A', 'boot/efi/EFI/a', '-', 'f:1:1:100644:0:0:-:-'),
		R('A', 'boot/efix', '-', 'f:1:1:100644:0:0:-:-')]
	scopes = [('', 'ext4', ''), ('boot/efi', 'vfat', '')]
	grouped = hpcp_mod._rows_by_scope(rows, scopes)
	assert {scope: [row.path for row in scoped] for scope, scoped in grouped.items()} == {
		'': ['./', 'etc/x', 'boot/efix'], 'boot/efi': ['boot/efi/', 'boot/efi/EFI/a']}


def test_rows_under_a_content_only_mount_below_the_target_are_content_only(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""`-siff store-patch.img -d /`: rows under boot/efi land on the vfat ESP mounted there, whose modes come from
	its mount options (here not those of the store's copy): planned and applied by content only, case-insensitively."""
	new, base = trees
	_write(new / 'etc' / 'conf', 'new conf')
	_write(base / 'etc' / 'conf', 'old conf')
	_write(new / 'boot' / 'efi' / 'EFI' / 'grub.cfg', 'new cfg')
	_write(base / 'boot' / 'efi' / 'EFI' / 'grub.cfg', 'old cfg')
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	for path in (target / 'boot' / 'efi' / 'EFI', target / 'boot' / 'efi' / 'EFI' / 'grub.cfg'):
		os.chmod(str(path), 0o700)
	_fake_mounts(hpcp_mod, monkeypatch, (target, 'ext4'), (target / 'boot' / 'efi', 'vfat'))
	planned = []
	real_plan_section = hpcp_mod._plan_section
	def plan_section(*args, **kwargs):
		planned.append(real_plan_section(*args, **kwargs))
		return planned[-1]
	monkeypatch.setattr(hpcp_mod, '_plan_section', plan_section)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert (summary['modified'], summary['forced'], summary['failed']) == (2, 0, 0)
	assert [(plan.name, plan.metadata, plan.case_insensitive) for plan in planned] == [('root', True, False), ('boot/efi', False, True)]
	assert (target / 'boot' / 'efi' / 'EFI' / 'grub.cfg').read_text() == 'new cfg'
	assert (target / 'etc' / 'conf').read_text() == 'new conf'


def test_contradictory_alias_patch_writes_and_then_fails_at_verify(trees, hpcp_mod, tmp_path):
	"""D x/f + A y/f on a target where y is a symlink to x: not refused; the write is restored after the delete and
	the delete's verify fails (non-zero exit). Where y is a real directory the patch applies cleanly."""
	new, base = trees
	_moved_file(new, base)
	vol = tmp_path / 'vol'
	rows, _ = _patch_from(hpcp_mod, new, base, [vol])
	assert _ops(rows) == {('D', 'x/f'), ('A', 'y/f')}
	target = tmp_path / 'target'
	_clone(base, target)
	(target / 'y').rmdir()
	os.symlink('x', str(target / 'y'))
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(target), force=True, max_workers=1)
	assert summary['restored'] == 1 and summary['failed'] == 1
	assert (target / 'x' / 'f').read_text() == 'moved'
	plain = tmp_path / 'plain'
	_clone(base, plain)
	summary = hpcp_mod.apply_patch_tree([str(vol)], str(plain), max_workers=1)
	assert (summary['deleted'], summary['added'], summary['failed']) == (1, 1, 0)
	assert _changes(hpcp_mod, new, plain) == []


def test_refusals_come_before_the_conflict_prompt(trees, hpcp_mod, tmp_path, reset_hpcp_globals, monkeypatch):
	"""A patch with a conflict and a path that escapes the target through a symlink: refused without asking first."""
	new, base = trees
	_write(new / 'conflict.txt', 'new')
	_write(base / 'conflict.txt', 'old')
	_write(new / 'sub' / 'f', 'f')
	(base / 'sub').mkdir()
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	_write(target / 'conflict.txt', 'local edit')
	outside = tmp_path / 'outside'
	outside.mkdir()
	(target / 'sub').rmdir()
	os.symlink(str(outside), str(target / 'sub'))
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(True))
	def prompt(*args, **kwargs):
		raise AssertionError('asked before refusing')
	monkeypatch.setattr(hpcp_mod.multiCMD, 'input_with_timeout_and_countdown', prompt)
	with pytest.raises(RuntimeError, match='resolve outside the target through symlinks'):
		hpcp_mod.apply_patch_tree([str(vol)], str(target), max_workers=1)
	assert list(outside.iterdir()) == []


def test_single_fs_generation_compares_content_only_on_fat(tmp_path, hpcp_mod):
	new, base = tmp_path / 'new', tmp_path / 'base'
	new.mkdir()
	base.mkdir()
	compare = hpcp_mod._compare_metadata
	assert compare((str(new), str(base)), [_mount_entry('/', 'ext4')]) is True
	assert compare((str(new), str(base)), [_mount_entry('/', 'ext4'), _mount_entry(new, 'vfat')]) is False
	assert compare((str(new), str(base)), [_mount_entry('/', 'ext4'), _mount_entry(base, 'exfat')]) is False
```

Also append these root tests to `tests/test_diff_image.py`:

```python
def _vfat_rename_images(hpcp_mod, tmp_path):
	"""golden.img holds EFI/Boot/BOOTX64.EFI, node.img EFI/BOOT/BOOTX64.EFI (vfat); returns (node, patch) for
	-si golden.img -di node.img."""
	golden, node, out = str(tmp_path / 'golden.img'), str(tmp_path / 'node.img'), str(tmp_path / 'patch.img')
	for image, name in ((golden, 'Boot'), (node, 'BOOT')):
		with open(image, 'wb') as f:
			f.truncate(32 << 20)
		subprocess.run(['mkfs.vfat', image], check=True, stdout=subprocess.DEVNULL)
		_change_image(hpcp_mod, image, lambda root, name=name: _write(os.path.join(root, 'EFI', name, 'BOOTX64.EFI'), 'loader'))
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [golden], [], node, out, mount_points, loop_devices, max_workers=1) == out
	return node, out


def _vfat_boot_listing(hpcp_mod, image):
	"""(names under EFI, content of EFI/<boot dir>/BOOTX64.EFI) of a vfat image."""
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(image, True, mount_points, loop_devices)
		names = os.listdir(os.path.join(root, 'EFI'))
		with open(os.path.join(root, 'EFI', names[0], 'BOOTX64.EFI')) as f:
			return names, f.read()


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.vfat') is None, reason='needs mkfs.vfat')
@pytest.mark.root
@pytest.mark.loop
def test_case_only_rename_patch_applies_on_a_vfat_image(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 1: the rename applies with the new spelling, and a re-run skips every row."""
	hpcp_mod.HASH_SIZE = 65536
	node, out = _vfat_rename_images(hpcp_mod, tmp_path)
	for run in (1, 2):
		with _mount_session(hpcp_mod) as (mount_points, loop_devices):
			summary = hpcp_mod.process_load_diff_image([out], [], [], node, mount_points, loop_devices, max_workers=1)
		assert summary['failed'] == 0 and summary['restored'] == 0 and summary['forced'] == 0
		assert (summary['deleted'], summary['added']) == ((2, 2) if run == 1 else (0, 0))
	assert _vfat_boot_listing(hpcp_mod, node) == (['Boot'], 'loader')


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.vfat') is None, reason='needs mkfs.vfat')
@pytest.mark.root
@pytest.mark.loop
def test_case_only_rename_is_restored_on_a_vfat_image_reported_as_nfs4(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review Focus 5: a filesystem type can hide case-insensitivity (NFS from a case-insensitive server). Planned
	case-sensitively, the adds are skipped as present and the deletes remove those files; the repair step writes them
	again and verify accepts the deleted names (same entries). A re-run stays correct."""
	hpcp_mod.HASH_SIZE = 65536
	node, out = _vfat_rename_images(hpcp_mod, tmp_path)
	monkeypatch.setattr(hpcp_mod, '_mount_scopes', lambda path, entries: [('', 'nfs4', '')])
	for run in (1, 2):
		with _mount_session(hpcp_mod) as (mount_points, loop_devices):
			summary = hpcp_mod.process_load_diff_image([out], [], [], node, mount_points, loop_devices, max_workers=1)
		assert summary['failed'] == 0 and hpcp_mod.ERRORS == []
		if run == 1:
			assert summary['restored'] == 2
		assert _vfat_boot_listing(hpcp_mod, node) == (['Boot'], 'loader')
```

If the nfs4 test reports permission errors from `cp -af` on vfat (the POSIX-mode copy), report it in your fix report; do not loosen the assertions.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "whatever_the_reported_type or mount_scopes or rows_by_scope or content_only_mount_below or contradictory_alias_patch or before_the_conflict_prompt or compares_content_only_on_fat or vfat_image"`
Expected: FAIL — refusals still raised, `_mount_scopes` / `_rows_by_scope` / `_compare_metadata` missing.

- [ ] **Step 4: Implement scopes, the new `apply_patch_tree`, generation and prompt order**

Replace `_path_mounts` with:

```python
def _mount_scopes(path, entries):
	"""[(key prefix, fstype, source)] for path: first ('', ...) for the mount holding path's real path, then one per
	mount below it, deepest last. A mount listed later (in mountinfo order) on the same or a parent mount point hides
	earlier ones, as on the system; hidden mounts are left out. fstype and source '' when no mount holds path."""
	visible = []
	for entry in entries:
		inside = entry['mount_point'].rstrip(os.sep) + os.sep
		visible = [other for other in visible if not (other['mount_point'].rstrip(os.sep) + os.sep).startswith(inside)]
		visible.append(entry)
	real = os.path.realpath(path)
	real_inside = real.rstrip(os.sep) + os.sep
	holder, depth, below = None, -1, []
	for entry in visible:
		inside = entry['mount_point'].rstrip(os.sep) + os.sep
		if real_inside.startswith(inside):
			if len(inside) > depth:
				holder, depth = entry, len(inside)
		elif inside.startswith(real_inside):
			below.append((os.path.relpath(entry['mount_point'], real), entry['fstype'], entry.get('source', '')))
	below.sort(key=lambda scope: (scope[0].count('/'), scope[0]))
	own = ('', holder['fstype'], holder.get('source', '')) if holder else ('', '', '')
	return [own] + below

def _rows_by_scope(rows, scopes):
	"""{key prefix: rows}: each row under the deepest scope prefix it lies in ('' for the target's own)."""
	prefixes = sorted((prefix for prefix, _, _ in scopes if prefix), key=len, reverse=True)
	grouped = {}
	for row in rows:
		name = row.path.rstrip('/')
		scope = next((prefix for prefix in prefixes if name == prefix or name.startswith(prefix + '/')), '')
		grouped.setdefault(scope, []).append(row)
	return grouped

def _compare_metadata(roots, entries):
	"""False when any of roots sits on a filesystem whose modes and owners come from mount options (FAT, exFAT,
	NTFS): a single-filesystem patch between such trees compares content only (see states_equal)."""
	return all(_has_posix_metadata(_mount_scopes(root, entries)[0][1]) for root in roots)
```

Delete `_refuse_case_only_renames`, `_CASE_SENSITIVE_TYPES` (with its comment), `_same_entry_pair`, `_caseless`, and the line `import unicodedata`.

In `_preflight_section`, delete the last block (`pair = _same_entry_pair(plan)` and the `raise` that follows). Its docstring stays `Refusals -fa cannot override, raised before anything changes.`

Replace `apply_patch_tree` with:

```python
def apply_patch_tree(volume_roots, target_root, force=False, max_workers=1, verbose=False, volume_names=None, store_note=False):
	"""Apply a patch whose volumes are mounted (or unpacked) at volume_roots onto directory target_root.

	Returns {'deleted', 'added', 'modified', 'skipped', 'restored', 'forced', 'failed'}. Raises RuntimeError
	for a malformed or incomplete patch, or when conflicts are not overridden.
	Another process modifying the target concurrently during an apply is out of scope
	(residual race between the containment checks and the writes).
	Files and symlinks are replaced atomically (a temporary sibling renamed over the target),
	so running processes keep the old file and every target ends either old or new.
	Paths are followed through symlinked directories that resolve inside target_root, for
	deletions and writes alike; a patch that would delete or write through a symlink leading
	outside it is refused before anything changes, even when forced (and before any conflict prompt).
	Only conflicts the user forced may have a whole directory tree removed.
	An interrupted apply leaves new directories 0700 and root-owned.
	A hard kill mid-apply can leave new directories with default metadata (it is applied
	only after every write); the re-run then reports those A rows as conflicts and needs -fa.
	Rows are planned per mount (the target's own, and each filesystem mounted below it, from
	/proc/self/mountinfo): on FAT, exFAT or NTFS by content only, on FAT, exFAT or HFS+ (not HFSX) with
	names matched case-insensitively (see check_patch_target). A write skipped as already done that a
	delete then takes away (a case-insensitive filesystem hpcp does not recognise, a symlinked directory
	alias) is written again and counted as restored (see _apply_rows).
	"""
	meta, rows, volume_roots = load_patch_volumes(volume_roots, volume_names)
	if store_note and meta.get('layout') == 'store':
		example = rows[0].path if rows else ''
		print(f"Note: this patch uses image-store layout; paths such as {example!r} are resolved under {target_root}")
	scopes = _mount_scopes(target_root, _mountinfo_entries())
	grouped = _rows_by_scope(rows, scopes)
	with _patch_hash_settings(meta) as (compare_hash, full_hash):
		plans = [_plan_section(prefix or 'root', grouped.get(prefix, []), volume_roots, target_root, compare_hash, full_hash,
				max_workers, _has_posix_metadata(fs_type), _has_case_insensitive_names(fs_type, source))
			for prefix, fs_type, source in scopes if not prefix or prefix in grouped]
		for plan in plans:
			_preflight_section(plan)
		_resolve_section_conflicts(plans, force)
		summaries = [_apply_section(plan, max_workers, verbose, full_hash, compare_hash) for plan in plans]
	summary = {key: sum(part[key] for part in summaries) for key in summaries[0]}
	print('Patch summary: ' + ', '.join(f'{key} {value}' for key, value in summary.items()))
	return summary
```

In `apply_disk_patch`, move the pre-flight before the prompt: replace

```python
		_resolve_section_conflicts(plans, force, extra)
		for plan in plans:
			_preflight_section(plan)
```

with

```python
		for plan in plans:
			_preflight_section(plan)
		_resolve_section_conflicts(plans, force, extra)
```

In `process_get_diff_image`, right before `rows = compute_changes(...)`, add the classification and pass it:

```python
	# Modes and owners on FAT, exFAT and NTFS come from mount options: compare content only when either side is one.
	metadata = _compare_metadata((new_root, base_root), _mountinfo_entries())
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata)
```

(replacing the existing `rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers)` line).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q`
Expected: all pass. Run the two vfat root tests three times; check nothing is left mounted / attached.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_diff_image.py tests/test_disk_patch.py
git commit -m "fix(diff-image): plan single-filesystem patches per mount, drop the case and same-entry refusals, refuse before prompting"
```

---

### Task 4: FAT timestamps across timezones, FAT attributes

**Files:**
- Modify: `hpcp.py` — `states_equal` (≈ 3866); add `_tz_shift_close` after `_mtime_ns_close`; add FAT attribute helpers just above `def _finish_replace`; `_finish_replace`.
- Test: `tests/test_disk_patch.py` (unit), `tests/test_diff_image.py` (root vfat)

**Interfaces:**
- Produces:
  - `_tz_shift_close(a_ns: int, b_ns: int) -> bool`
  - `_fat_attributes(path: str) -> int | None`, `_set_fat_attributes(path: str, attributes: int, key: str) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
def test_content_only_states_tolerate_timezone_shaped_mtime_offsets(hpcp_mod):
	"""Review Focus 3: a FAT time read under two timezone settings differs by whole 15-minute steps (at most 24 h);
	with the same size and hash that is the same file. Any other offset, another hash, POSIX sections and
	comparisons without hashes still count the difference."""
	eq = hpcp_mod.states_equal
	t = 1717000000 * 10**9

	def state(mtime_ns, digest='aaaa'):
		return f'f:10:{mtime_ns}:100755:0:0:{digest}:-'

	hour = 3600 * 10**9
	assert eq(state(t), state(t + 2 * hour), metadata=False)
	assert eq(state(t), state(t - 5 * hour - 30 * 60 * 10**9), metadata=False)
	assert eq(state(t), state(t + 2 * hour + 10**9), metadata=False)
	assert not eq(state(t), state(t + 7 * 60 * 10**9), metadata=False)
	assert not eq(state(t), state(t + 25 * hour), metadata=False)
	assert not eq(state(t), state(t + 2 * hour, 'bbbb'), metadata=False)
	assert not eq(state(t), state(t + 2 * hour))
	assert not eq(state(t), state(t + 2 * hour), compare_hash=False, metadata=False)
	assert not eq(state(t, '-'), state(t + 2 * hour, '-'), metadata=False)


def test_fat_attributes_are_none_off_fat(tmp_path, hpcp_mod):
	path = tmp_path / 'f'
	path.write_text('x')
	assert hpcp_mod._fat_attributes(str(path)) is None
	assert hpcp_mod._fat_attributes(str(tmp_path / 'missing')) is None
```

Append to `tests/test_diff_image.py`:

```python
def _fat_attrs(path, value=None):
	"""Read (and first set, when value is given) the FAT attribute byte of a file on vfat."""
	import fcntl
	fd = os.open(str(path), os.O_RDONLY)
	try:
		if value is not None:
			fcntl.ioctl(fd, 0x40047211, struct.pack('I', value))
		return struct.unpack('I', fcntl.ioctl(fd, 0x80047210, b'\0' * 4))[0]
	finally:
		os.close(fd)


@requires_root_and_image_tools
@pytest.mark.skipif(shutil.which('mkfs.vfat') is None, reason='needs mkfs.vfat')
@pytest.mark.root
@pytest.mark.loop
def test_replacing_a_file_on_vfat_keeps_its_fat_attributes(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 4: a read-only + hidden + system file on the node's FAT target is replaced (new content) and keeps
	those attributes."""
	hpcp_mod.HASH_SIZE = 65536
	golden, node, out = str(tmp_path / 'golden.img'), str(tmp_path / 'node.img'), str(tmp_path / 'patch.img')
	for image, data in ((golden, 'new loader'), (node, 'old loader')):
		with open(image, 'wb') as f:
			f.truncate(32 << 20)
		subprocess.run(['mkfs.vfat', image], check=True, stdout=subprocess.DEVNULL)
		_change_image(hpcp_mod, image, lambda root, data=data: _write(os.path.join(root, 'EFI', 'x.efi'), data))
	# read-only (0x01) | hidden (0x02) | system (0x04) | archive (0x20)
	_change_image(hpcp_mod, node, lambda root: _fat_attrs(os.path.join(root, 'EFI', 'x.efi'), 0x27))
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [golden], [], node, out, mount_points, loop_devices, max_workers=1) == out
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		summary = hpcp_mod.process_load_diff_image([out], [], [], node, mount_points, loop_devices, max_workers=1)
	assert summary['failed'] == 0 and summary['modified'] == 1
	seen = {}

	def read(root):
		path = os.path.join(root, 'EFI', 'x.efi')
		seen['attrs'] = _fat_attrs(path)
		with open(path) as f:
			seen['data'] = f.read()

	_change_image(hpcp_mod, node, read)
	assert seen['data'] == 'new loader' and seen['attrs'] & 0x07 == 0x07
```

Make sure `import struct` is present at the top of `tests/test_diff_image.py` (add it to the imports if it is missing).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q -k "timezone_shaped or fat_attributes"`
Expected: FAIL — shifted states unequal; `_fat_attributes` missing; attributes lost after replace.

- [ ] **Step 3: Implement**

After `_mtime_ns_close` add:

```python
_TZ_STEP_NS = 15 * 60 * 10**9
_TZ_MAX_NS = 24 * 3600 * 10**9

def _tz_shift_close(a_ns, b_ns):
	"""True when a_ns and b_ns are a whole number of 15-minute steps apart (at most 24 hours), within the usual
	timestamp tolerance: one FAT timestamp read under two timezone settings (sys_tz, tz=, time_offset=)."""
	shift = int(round((a_ns - b_ns) / _TZ_STEP_NS)) * _TZ_STEP_NS
	return 0 < abs(shift) <= _TZ_MAX_NS and _mtime_ns_close(a_ns, b_ns + shift)
```

In `states_equal`, replace the final line `return _mtime_ns_close(info_a['mtime_ns'], info_b['mtime_ns'])` with:

```python
	if _mtime_ns_close(info_a['mtime_ns'], info_b['mtime_ns']):
		return True
	# FAT stores local times: read under another timezone setting the same file shows a whole-step offset. Only
	# content-only comparisons whose hashes confirmed the content accept it.
	return (not metadata and compare_hash and info_a['hash'] not in ('', '-')
		and _tz_shift_close(info_a['mtime_ns'], info_b['mtime_ns']))
```

Add to the `states_equal` docstring, after its last sentence: `With metadata False a whole number of 15-minute steps (at most 24 hours) between file mtimes is tolerated when the hashes match: a FAT time read under another timezone setting.`

Just above `def _finish_replace` add:

```python
_FAT_IOCTL_GET_ATTRIBUTES = 0x80047210
_FAT_IOCTL_SET_ATTRIBUTES = 0x40047211
# Read-only, hidden, system and archive: the attribute bits a regular file carries.
_FAT_FILE_ATTRIBUTES = 0x27

def _fat_attributes(path):
	"""The FAT attribute bits of an existing file on vfat / msdos, else None (missing file, another
	filesystem, or a platform without fcntl)."""
	try:
		import fcntl
		fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
	except (ImportError, OSError):
		return None
	try:
		return struct.unpack('I', fcntl.ioctl(fd, _FAT_IOCTL_GET_ATTRIBUTES, b'\0' * 4))[0] & _FAT_FILE_ATTRIBUTES
	except OSError:
		return None
	finally:
		os.close(fd)

def _set_fat_attributes(path, attributes, key):
	"""Set path's FAT attribute bits; a failure is a warning, never an error."""
	try:
		import fcntl
		fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
		try:
			fcntl.ioctl(fd, _FAT_IOCTL_SET_ATTRIBUTES, struct.pack('I', attributes))
		finally:
			os.close(fd)
	except (ImportError, OSError) as e:
		print(f'Warning: cannot keep the FAT attributes of {key}: {e}', file=sys.stderr)
```

In `_finish_replace`, replace the line `os.replace(tmp, dest)` with:

```python
		# A FAT file keeps its read-only / hidden / system bits: the replacement would otherwise get the defaults.
		attributes = None if metadata else _fat_attributes(dest)
		os.replace(tmp, dest)
		if attributes is not None and attributes != _fat_attributes(dest):
			_set_fat_attributes(dest, attributes, row.path)
```

and add to its docstring: `On a content-only target (metadata False) an existing FAT file's attribute bits are kept.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q`
Expected: all pass; the vfat attribute root test passes three runs in a row; nothing left mounted / attached.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py tests/test_diff_image.py
git commit -m "fix(diff-image): tolerate FAT timezone offsets in content-only sections; keep FAT attributes on replace"
```

---

### Task 5: Docs, spec deviations, version 9.68, full verification

**Files:**
- Modify: `README.md`, `hpcp.py` (`version`, `COMMIT_DATE`), `tests/test_dd_fs_params.py` (`test_version_bumped`), `docs/superpowers/specs/2026-10-11-apply-repair-and-case-handling-design.md`

- [ ] **Step 1: README**

In `README.md`, replace the paragraph that begins `A single-filesystem patch pairs names by exact case, so a case-only rename` with:

```markdown
A single-filesystem patch pairs names by exact case, so a case-only rename (`EFI/BOOT` to `EFI/Boot`) is a delete plus an add, and the target ends with the new spelling. Rows are planned per filesystem: the target's own and each one mounted below it (for example the ESP at `/boot/efi` when applying to `/`). On FAT, exFAT and HFS/HFS+ (not case-sensitive HFSX) names are matched case-insensitively, so such a rename applies, with or without a content change, and a re-run skips it. On FAT, exFAT and NTFS rows are compared and written by content only, since modes and owners there come from mount options. When a delete removes an entry whose add or update was skipped as already done (a case-insensitive filesystem hpcp does not recognise, or a symlinked directory alias), the apply writes it again and counts it as `restored`.
```

Replace the bullet that begins `- FAT, exFAT and NTFS sections (the ESP) are compared and applied by content only` with:

```markdown
- FAT, exFAT and NTFS sections (the ESP) are compared and applied by content only: their permissions and owners come from mount options, so differences there are ignored. A file whose mtime differs only by a timezone offset (a whole number of 15 minutes, at most 24 hours) with the same content counts as unchanged, since FAT stores local times. A replaced FAT file keeps its read-only, hidden and system attributes. Names are generated by exact case, so a case-only rename (`EFI/BOOT` to `EFI/Boot`) is carried and the node ends with the golden's spelling; FAT, exFAT and HFS/HFS+ sections match names case-insensitively when applying (case-sensitive HFSX is detected). HFS and HFS+ sections compare owners and modes too (hfsplus keeps them on disk).
```

If the help text changed (it should not), regenerate the README help block the way the 9.66 release did; otherwise leave it.

- [ ] **Step 2: Spec deviations**

In `docs/superpowers/specs/2026-10-11-apply-repair-and-case-handling-design.md`:
- In Part 2 item 3, replace the sentence about `_path_mounts` picking the entry by device number with: `_mount_scopes` (replacing `_path_mounts`) takes the mounts in mountinfo order — a mount listed later on the same or a parent mount point hides earlier ones — and picks the deepest visible mount holding the path, so a mount hidden by a later mount on a parent is not chosen.
- In the Testing list, replace the bullet `Forced type change (-fa) over an aliased skipped row: the row is restored.` with `Forced type change (-fa) over an aliased skipped row: the loss is reported (copy error and verify failure); it cannot be restored once the alias points at a file.`
- In Part 1, after item 1, add: `Verify (_verify_rows): a D row whose path still exists passes when an A / M row's name differs from it only by case (_fold_name) and both paths are one entry (same st_dev / st_ino), so a correct case-only rename on a case-insensitive filesystem hpcp does not recognise is not reported as a failure.`

- [ ] **Step 3: Version**

```bash
sed -i "s/^version = '9.67'$/version = '9.68'/" hpcp.py
sed -i "s/^COMMIT_DATE = '.*'$/COMMIT_DATE = '$(date +%F)'/" hpcp.py
grep -nE "^version|^COMMIT_DATE" hpcp.py
```

Update `test_version_bumped` in `tests/test_dd_fs_params.py` to expect `'9.68'` and the new `COMMIT_DATE`.

- [ ] **Step 4: Full verification**

Run: `python -m pytest tests/ -q`
Expected: all pass (1 skipped). Then `python3 hpcp.py -V` prints `9.68`; `uv run --no-project --python 3.8 python -c "import ast; ast.parse(open('hpcp.py').read(), feature_version=(3,6)); print('ok')"` prints `ok`; `losetup -a` (apart from snap loops) and `findmnt | grep /tmp/tmp` show nothing left behind; `md5sum /usr/lib/os-release` is unchanged from before the work.

- [ ] **Step 5: Commit**

```bash
git add README.md hpcp.py tests/test_dd_fs_params.py docs/superpowers/specs/2026-10-11-apply-repair-and-case-handling-design.md
git commit -m "docs: case renames, content-only FAT rules and the repair step; release 9.68"
```

Releasing (gate review, push, uvpack, cas-toolbox pin) is out of scope for this plan.
