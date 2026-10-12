# Exclude anchoring, inputs kept out of patches, cheaper payload re-check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Patch generation matches `-e` inside each mounted filesystem, never ships its own inputs, and reads the payload a second time only for files that may have changed. It also closes the `/x/out` test hazard and the B2 leftovers (hpcp 9.71).

**Architecture:**
- **Exclude patterns.** Patterns for a mounted source are rewritten per side by putting that side's escaped root path in front of each one (`_rooted_excludes`). The existing pruning listing then matches them exactly as `/` + the path inside the filesystem, on both sides (`build_rooted_maps`).
- **Hpcp's own paths.** This run's mount points and inputs are expressed once, as `/`-rooted relative paths (`_inside_excludes`), and rooted per side.
- **Payload re-check.** The scan keeps each new-side file's `lstat` values, taken before the hash read, plus the scan's start time (`compute_changes(scan=...)`). `_check_payload` hashes again only the copies that `_rows_to_recheck` names. It decides with mountinfo super options and the filesystem type, through `_mount_scope_entries`.

**Tech Stack:** Python ≥ 3.6 single-file `hpcp.py`, pytest; root tests use loop devices.

**Spec:** `docs/superpowers/specs/2026-10-12-exclude-inputs-recheck-design.md` (approved; corrected in 1b2f396: live `-dd -diff`
partitions are bind-mounted at temporary paths, so there is no `/boot/efi/...` compatibility break).

## Global Constraints

- One file, `hpcp.py`. Indentation is tabs. It must import on Windows, so new code must not need `/proc`.
- Python 3.6 floor: no `time.time_ns()`, no walrus, no `dataclasses`. Use `int(time.time() * 1000000000)` for wall-clock ns.
- Tests are pytest functions with no classes. Real-filesystem tests carry `@pytest.mark.root` and `@pytest.mark.loop` plus the file's `requires_root_and_image_tools` / `requires_root_and_disk_tools` skip marker. Scratch goes only under `tmp_path`, and nothing may be left mounted or attached.
- **Probe safety rule:** any ad-hoc root probe goes in a script file run with `bash <file>`. `set -euo pipefail` typed at the top of a shell command is ignored in this harness. Every path variable gets a `: "${VAR:?}"` guard and a `/tmp/claude-0/` prefix check, and devices are only `/dev/loop*` backed by scratch images.
- Full suite: `python -m pytest tests/ -q` (about 3.5 minutes, as root). Never run two full suites at once, because they race for loop devices.
- Version 9.71. `COMMIT_DATE` is the day the work is finished. No PyPI publish (`uvpack`) without the user's go.
- Patch formats are unchanged.
- Directory sources (`-diff DIR …`) and copy-mode `-e` match exactly as before.

## Review Focus

1. **Scan stats from worker processes** (≥ 256 files, `max_workers > 1`): they must come back from the pool, or every copy is hashed again silently. Pinned by `test_payload_recheck_uses_scan_stats_from_worker_processes` (Task 4).
2. **A symlinked new root** (`-diff LINK -d BASE`): the scan lists the target and the copy is taken through the link. Stats are keyed by row key, so an unchanged file is trusted and a rewritten one is caught. Pinned by `test_payload_recheck_through_a_symlinked_new_root` (Task 4).
3. **A FAT ESP mounted below a live directory source** (`-diff /` with `/boot/efi`): its files are re-hashed even when the root filesystem is trusted. Pinned by `test_payload_recheck_hashes_files_on_an_untrusted_mount_below_the_source` (Task 4).
4. **A mounted root whose path has glob characters, or the root `/` itself**: rooting must escape the prefix, and for `/` it must change nothing. Pinned by `test_rooted_excludes_match_like_the_rooted_path` (Task 1).
5. **An input given through a symlink** (`-d LINK` where LINK resolves inside the new directory): it is still found, by real path. Pinned by `test_base_directory_given_through_a_symlink_inside_the_new_directory_stays_out` (Task 2).

---

### Task 1: `-e` matched inside each mounted filesystem

**Files:**
- Modify: `hpcp.py`:
  - `build_root_maps` (~4323): add `build_rooted_maps` after it;
  - `_mount_point_excludes` (~5892): replace it with `_inside_excludes` + `_rooted_excludes`;
  - `process_get_diff_image` (~5933-5948);
  - `_partition_changes` (~6958-6977).
- Test: `tests/test_diff_image.py`, `tests/test_disk_patch.py`

**Interfaces:**
- Produces:
  - `_inside_excludes(paths, roots) -> set[str]`: glob-escaped `os.sep + relpath` for each path strictly inside a root, compared by real path.
  - `_rooted_excludes(patterns, root) -> frozenset[str]`.
  - `build_rooted_maps(new_root, base_root, exclude=None, max_workers=..., parallel_file_listing=False) -> (new_map, base_map)`.

  Task 2 uses `_inside_excludes`.
- Removes: `_mount_point_excludes`. Its only callers are the two functions changed here.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_diff_image.py`)

```python
def test_rooted_excludes_match_like_the_rooted_path(hpcp_mod):
	"""Spec Part 1 / Review Focus 4: a pattern rooted at a mount matches a real path exactly when it matches '/' + the
	path inside; the mount's own path never matches a pattern's '*' part, and is escaped."""
	import fnmatch
	for root in ('/tmp/tmpab12', '/m/[x]*?', '/m/[x]*?/'):
		rooted = hpcp_mod._rooted_excludes(['/var/log/*', '*/tmp/*', '*/cache'], root)
		real = os.path.abspath(root).rstrip(os.sep)
		for inside, excluded in (('/var/log/a', True), ('/var/logs', False), ('/tmp/b', True), ('/a/cache', True),
				('/a', False), ('/var/log', False)):
			assert any(fnmatch.fnmatch(real + inside, pattern) for pattern in rooted) is excluded, (root, inside)
	assert hpcp_mod._rooted_excludes(['/var/log/*', '*/tmp/*'], '/') == frozenset(['/var/log/*', '*/tmp/*'])


def test_inside_excludes_key_paths_by_their_place_in_a_root(tmp_path, hpcp_mod):
	root, other = tmp_path / 'root', tmp_path / 'other'
	(root / 'a' / '[b]').mkdir(parents=True)
	other.mkdir()
	paths = [str(root / 'a' / '[b]'), str(other), str(root)]
	assert hpcp_mod._inside_excludes(paths, [str(root)]) == {'/a/[[]b]'}


def test_image_source_excludes_are_rooted_in_the_image(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 1: for -si, '/var/log/*' means var/log inside the image, on both sides; '*/var/log/*' still works; and
	'*/tmp/*' or '/tmp/*' no longer match the (temporary) path the image is mounted at, which excluded the whole image."""
	new, base = trees
	image_root = tmp_path / 'image_root'
	_write(image_root / 'keep.txt', 'k')
	_write(image_root / 'var' / 'log' / 'a.log', 'log')
	_write(image_root / 'tmp' / 'b', 'b')
	_write(base / 'var' / 'log' / 'old.log', 'old')
	_write(base / 'tmp' / 'c', 'c')
	image = tmp_path / 'new.img'
	image.write_bytes(b'')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): image_root})
	for patterns in (['/var/log/*', '*/tmp/*'], ['*/var/log/*', '/tmp/*']):
		rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [],
			exclude=frozenset(patterns), max_workers=1)
		ops = _ops(rows)
		assert ('A', 'keep.txt') in ops, patterns
		assert not [path for _, path in ops if path.startswith(('var/log/', 'tmp/')) and path not in ('var/log/', 'tmp/')], patterns


def test_exclude_file_patterns_are_rooted_like_e(trees, hpcp_mod, tmp_path, monkeypatch):
	"""-x patterns go through format_exclude with -e ones and are rooted the same way."""
	new, base = trees
	image_root = tmp_path / 'image_root'
	_write(image_root / 'keep.txt', 'k')
	_write(image_root / 'var' / 'log' / 'a.log', 'log')
	image = tmp_path / 'new.img'
	image.write_bytes(b'')
	listing = tmp_path / 'exclude.txt'
	listing.write_text('/var/log/*\n')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): image_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [],
		exclude=hpcp_mod.format_exclude(None, str(listing)), max_workers=1)
	assert ('A', 'keep.txt') in _ops(rows) and ('A', 'var/log/a.log') not in _ops(rows)


def test_directory_source_excludes_still_match_real_paths(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 1 item 4: a directory source keeps copy mode's matching: real paths, not '/' + the path inside."""
	new, base = trees
	_write(new / 'keep.txt', 'k')
	_write(new / 'var' / 'log' / 'a.log', 'log')
	real = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [],
		exclude=frozenset([str(new / 'var' / 'log') + '/*']), max_workers=1)
	assert ('A', 'keep.txt') in _ops(real) and ('A', 'var/log/a.log') not in _ops(real)
	rooted = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(base)], None, str(tmp_path / 'p2.img'), [], [],
		exclude=frozenset(['/var/log/*']), max_workers=1)
	assert ('A', 'var/log/a.log') in _ops(rooted)


def test_rooted_maps_skip_excluded_directories_on_both_sides(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Spec Part 1 item 6: an excluded directory is skipped, not walked, on the new and the base side."""
	new, base = tmp_path / 'mnt_new', tmp_path / 'mnt_base'
	for root in (new, base):
		_write(root / 'var' / 'log' / 'deep' / 'a.log', 'log')
		_write(root / 'keep.txt', 'k')
	walked = []
	real = hpcp_mod.get_file_list_serial
	def spy(root, *args, **kwargs):
		walked.append(root)
		return real(root, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, 'get_file_list_serial', spy)
	new_map, base_map = hpcp_mod.build_rooted_maps(str(new), str(base), frozenset(['/var/log']), max_workers=1, parallel_file_listing=False)
	for entry_map in (new_map, base_map):
		assert 'var/' in entry_map and 'keep.txt' in entry_map
		assert not [key for key in entry_map if key.startswith('var/log')]
	log_dir = os.path.join('var', 'log')
	assert not [root for root in walked if root.endswith(log_dir) or log_dir + os.sep in root]
```

Append to `tests/test_disk_patch.py` (after `test_input_disk_image_on_a_live_partition_stays_out_of_the_patch`):

```python
@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_disk_patch_exclude_patterns_are_rooted_in_each_partition(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 1: '/var/log/*' applies to the root partition, '/EFI/x/*' to the ESP, and '*/tmp/*' to tmp/ inside a
	partition only (it used to match the temporary mount path and exclude every partition)."""
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_make_newer(old, new, label=None, raw=False, mbr=False, gpt_name=None)
	with _attached(new) as loop:
		with _mounted(loop + 'p3') as root:
			_write(os.path.join(root, 'var', 'log', 'syslog'), b'noise')
			_write(os.path.join(root, 'tmp', 'scratch'), b'scratch')
		with _mounted(loop + 'p1') as esp:
			_write(os.path.join(esp, 'EFI', 'x', 'skip.efi'), b'skip')
	with _session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_disk_diff_image([str(new)], [str(old)], out, mount_points, loop_devices,
			exclude=frozenset(['/var/log/*', '/EFI/x/*', '*/tmp/*']), max_workers=1) == out
	_, part3 = _read_part3_patch(hpcp_mod, out)
	_, part1 = _read_part3_patch(hpcp_mod, out, section='part1')
	paths3, paths1 = {row.path for row in part3}, {row.path for row in part1}
	assert 'var/log/' in paths3 and 'var/log/syslog' not in paths3
	assert 'tmp/' in paths3 and 'tmp/scratch' not in paths3
	assert ('M', 'etc/os-release') in {(row.op, row.path) for row in part3}
	assert 'EFI/x/' in paths1 and 'EFI/x/skip.efi' not in paths1
	assert any(path.endswith('grubx64.efi') for path in paths1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "rooted or exclude_file or directory_source_excludes or inside_excludes"`
Expected: FAIL.
- `_rooted_excludes`, `_inside_excludes` and `build_rooted_maps` are missing (AttributeError).
- `test_image_source_excludes_are_rooted_in_the_image` fails on its first assert: `*/tmp/*` matches the image's `/tmp/...` path, so `keep.txt` is gone.
- `test_directory_source_excludes_still_match_real_paths` already passes. It pins the unchanged behaviour.

Then, as root: `python -m pytest tests/test_disk_patch.py -q -k rooted_in_each_partition`.
Expected: FAIL. `*/tmp/*` excludes every partition, so `process_get_disk_diff_image` returns `None` instead of `out`.

- [ ] **Step 3: Implement**

In `hpcp.py`, replace the whole `_mount_point_excludes` function with:

```python
def _inside_excludes(paths, roots):
	"""Rooted exclude patterns (matched against '/' + a path inside a listed tree; see _rooted_excludes), one for each
	of paths that lies strictly inside one of roots: its path relative to that root, compared by real path. Used for
	this run's mount points and inputs, so both listings leave them out. A root itself is never excluded."""
	patterns = set()
	for path in paths:
		path_real = os.path.realpath(path)
		for root in roots:
			root_real = os.path.realpath(root)
			if path_real.startswith(root_real.rstrip(os.sep) + os.sep):
				patterns.add(glob.escape(os.sep + os.path.relpath(path_real, root_real)))
	return patterns

def _rooted_excludes(patterns, root):
	"""patterns written for '/' + a path inside the tree at root, as patterns for the real paths under root.

	root's own path, glob-escaped, goes in front of each, so a real path matches exactly when its '/'-rooted path does:
	'/var/log/*' under /tmp/tmpab12 becomes '/tmp/tmpab12/var/log/*', and '*/tmp/*' matches a tmp/ inside the tree,
	never the tree's own (temporary) mount path. Nothing changes for root '/'."""
	prefix = glob.escape(os.path.abspath(root).rstrip(os.sep))
	return frozenset(prefix + pattern for pattern in patterns)
```

After `build_root_maps`, add:

```python
def build_rooted_maps(new_root, base_root, exclude=None, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""Entry maps for two mounted filesystems compared root to root ('root' layout: -si images, -dd -diff sections),
	exclude matched against '/' + the path inside each filesystem (_rooted_excludes). Each side is listed with its own
	rooted patterns, so an excluded directory is skipped, not walked, on both sides."""
	exclude = frozenset(exclude or ())
	new_map = build_entry_map(new_root, '', None, _rooted_excludes(exclude, new_root), None, max_workers, parallel_file_listing)
	base_map = build_entry_map(base_root, '', None, _rooted_excludes(exclude, base_root), None, max_workers, parallel_file_listing)
	return new_map, base_map
```

In `process_get_diff_image`, replace this block (from `exclude = frozenset(exclude or ()) | _mount_point_excludes(` through the closing `base_map = _drop_lost_found(base_map)`):

```python
	exclude = frozenset(exclude or ()) | _mount_point_excludes(mount_points, (new_root, base_listed), os.path.abspath(new_root))
	if layout == 'store':
		new_map, base_map = build_store_maps(new_root, base_root, exclude, max_workers, parallel_file_listing)
		new_base = os.sep
	else:
		new_map, base_map = build_root_maps(new_root, base_root, exclude, max_workers, parallel_file_listing)
		new_base = new_root
		if src_image or dest_image:
			# On both sides: `-si golden.img -d /` must not delete or replace the live root's own lost+found/.
			new_map = _drop_lost_found(new_map)
			base_map = _drop_lost_found(base_map)
```

with:

```python
	# This run's mount points inside a listed tree stay out of both listings.
	internal = _inside_excludes(mount_points, (new_root, base_listed))
	if src_image:
		# A mounted source: patterns are matched against '/' + the path inside each filesystem.
		new_map, base_map = build_rooted_maps(new_root, base_root, frozenset(exclude or ()) | internal, max_workers, parallel_file_listing)
	else:
		# A directory source: patterns are matched against its real paths, as in copy mode.
		exclude = frozenset(exclude or ()) | _rooted_excludes(internal, new_root)
		if layout == 'store':
			new_map, base_map = build_store_maps(new_root, base_root, exclude, max_workers, parallel_file_listing)
		else:
			new_map, base_map = build_root_maps(new_root, base_root, exclude, max_workers, parallel_file_listing)
	new_base = os.sep if layout == 'store' else new_root
	if layout == 'root' and (src_image or dest_image):
		# On both sides: `-si golden.img -d /` must not delete or replace the live root's own lost+found/.
		new_map = _drop_lost_found(new_map)
		base_map = _drop_lost_found(base_map)
```

In `_partition_changes`, update the docstring's first sentence and replace the listing block. The docstring becomes:

```python
	"""Change rows of one filesystem partition (root to root; lost+found, this run's mounts and
	its input disk-image files left out). exclude is matched against '/' + the path inside the partition's filesystem
	(build_rooted_maps). metadata False: content-only rows (see states_equal).
	fat: a FAT section, its files hashed in full (see compute_changes).
	Names are compared by exact case in every section: a case-only rename is a delete plus an add."""
```

Replace everything from `matched = os.path.abspath(new_root)` through the `new_map, base_map = build_root_maps(...)` line with:

```python
	internal = _inside_excludes(mount_points, (new_root, old_root))
	for live_mount in live_mounts:
		if live_mount:
			# A live filesystem holds the (empty) directories of this run's other temp mount points,
			# and may hold the disk images this run compares (the old golden image on the new golden).
			internal |= _inside_excludes(list(mount_points) + list(input_files), (live_mount,))
	new_map, base_map = build_rooted_maps(new_root, old_root, frozenset(exclude or ()) | internal, max_workers, parallel_file_listing)
```

Run `grep -n "_mount_point_excludes" hpcp.py tests/*.py`. Expected: no output.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "rooted or exclude_file or directory_source_excludes or inside_excludes or lost_and_found or ignores_its_own or ignores_the_store"`
Then, as root: `python -m pytest tests/test_disk_patch.py -q -k "rooted_in_each_partition or live_partition or input_disk_image or pairs_names"`
Expected: PASS.

- [ ] **Step 5: Run both affected files in full**

Run (as root): `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py -q`
Expected: all pass, with no new skips. Check that nothing is left behind: `losetup -a | grep -v snapd` prints nothing.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_diff_image.py tests/test_disk_patch.py
git commit -m "fix(diff): match -e inside each mounted filesystem for -si and -dd -diff sections; skip excluded dirs on both sides"
```

---

### Task 2: inputs inside a directory side stay out of both listings

**Files:**
- Modify: `hpcp.py`, in `process_get_diff_image`: the `internal = _inside_excludes(...)` line Task 1 added.
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `_inside_excludes(paths, roots)` from Task 1.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_diff_image.py`)

```python
def test_di_image_inside_the_new_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 2: `-diff / -di /root/golden-v1.img` must not install the base image on every node."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	image = new / 'golden-v1.img'
	image.write_bytes(b'image bytes')
	store_root = tmp_path / 'store_root'
	store_root.mkdir()
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): store_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [], str(image), str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert any(row.path.endswith('/a.txt') for row in rows)
	assert not [row for row in rows if 'golden-v1.img' in row.path]


def test_si_image_inside_the_base_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""`-diff -si /new.img -d /`: the image file on the base side is no D row."""
	new, base = trees
	image_root = tmp_path / 'image_root'
	_write(image_root / 'a.txt', 'a')
	image = base / 'new.img'
	image.write_bytes(b'image bytes')
	_fake_image_mounts(hpcp_mod, monkeypatch, {str(image): image_root})
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [], [str(image)], [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert ('A', 'a.txt') in _ops(rows)
	assert not [row for row in rows if 'new.img' in row.path]


def test_base_directory_inside_the_new_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""`-diff / -d /mnt/oldroot` must not add the whole old tree under mnt/oldroot/."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	oldroot = new / 'mnt' / 'oldroot'
	_write(oldroot / 'old.txt', 'old')
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(oldroot)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert ('A', 'a.txt') in _ops(rows) and ('A', 'mnt/') in _ops(rows)
	assert not [row for row in rows if row.path.startswith('mnt/oldroot')]


def test_new_directory_inside_the_base_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""The base listing must not see the new tree inside it and delete it on every node."""
	new, base = trees
	inner = base / 'srv' / 'golden'
	_write(inner / 'a.txt', 'a')
	_write(base / 'keep.txt', 'k')
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(inner)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert ('A', 'a.txt') in _ops(rows) and ('D', 'keep.txt') in _ops(rows)
	assert not [row for row in rows if row.path.startswith('srv/golden')]


def test_base_directory_given_through_a_symlink_inside_the_new_directory_stays_out(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 5: inputs are compared by real path, so a symlinked spelling of the base is still found."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	oldroot = new / 'mnt' / 'oldroot'
	_write(oldroot / 'old.txt', 'old')
	link = tmp_path / 'oldroot-link'
	os.symlink(str(oldroot), str(link))
	rows = _rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(link)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert ('A', 'a.txt') in _ops(rows)
	assert not [row for row in rows if row.path.startswith('mnt/oldroot')]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "stays_out"`
Expected: all five FAIL. Each asserts that an input path is absent from the rows, and it is present.

- [ ] **Step 3: Implement**

In `process_get_diff_image`, replace the two lines Task 1 added:

```python
	# This run's mount points inside a listed tree stay out of both listings.
	internal = _inside_excludes(mount_points, (new_root, base_listed))
```

with:

```python
	# This run's mount points, and its inputs inside a directory being compared (the -si / -di image files, the base
	# and new directories), stay out of both listings: `-diff / -di /root/golden-v1.img` must not ship the image.
	internal = _inside_excludes(mount_points, (new_root, base_listed))
	directories = [root for root, is_dir in ((new_root, new_is_dir), (base_root, not base_is_image)) if is_dir]
	images_in = [os.path.expanduser(path) for path in (images[0] if src_image else None, dest_image) if path]
	internal |= _inside_excludes(images_in + directories, directories)
```

`images` is defined only when `src_image` is set. The conditional expression above reads it only then.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "stays_out or ignores_its_own or ignores_the_store or lost_and_found or rooted"`
Expected: PASS.

- [ ] **Step 5: Run the file in full**

Run (as root): `python -m pytest tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "fix(diff): keep -si/-di images and the base/new directories out of a patch when they lie inside a compared directory"
```

---

### Task 3: housekeeping — `/x` test paths and the B2 leftovers

**Files:**
- Modify: `tests/test_disk_patch.py`, the stubbed generation tests (~1172-1316).
- Modify: `hpcp.py`:
  - the btrfs warning (~1910);
  - `_lsblk_disk_rows` (~832);
  - the raw-size loop in `process_get_disk_diff_image` (~7100-7106).
- Modify: `tests/test_dd_fs_params.py:398`.
- Test: `tests/test_disk_patch.py`, `tests/test_dd_fs_params.py`.

**Interfaces:**
- Produces (tests, used by Task 4):
  - `_fake_disk_paths(tmp_path) -> (new, old, out)`;
  - `_fake_generation(hpcp_mod, monkeypatch, tmp_path, new_parts, old_parts, in_use=None)`, which takes `tmp_path` and keys its layouts by the `_fake_disk_paths` disks.

- [ ] **Step 1: Move every `/x` path to `tmp_path`, and write the failing tests**

In `tests/test_disk_patch.py`, add this helper before `_fake_generation`:

```python
def _fake_disk_paths(tmp_path):
	"""(new disk, old disk, patch output) under tmp_path for stubbed generations: a stub that regresses can create or
	mount something only there, never under a real directory such as /x."""
	return str(tmp_path / 'new.img'), str(tmp_path / 'old.img'), str(tmp_path / 'out.img')
```

Change `_fake_generation`'s signature to `_fake_generation(hpcp_mod, monkeypatch, tmp_path, new_parts, old_parts, in_use=None)`, and change its first docstring line to `"""Stub the disk side of process_get_disk_diff_image for the disks of _fake_disk_paths(tmp_path).`. Its layouts become:

```python
	new, old, _ = _fake_disk_paths(tmp_path)
	layouts = {new: _layout(new_parts), old: _layout(old_parts)}
```

Every `_fake_generation(hpcp_mod, monkeypatch, …)` call passes `tmp_path` as its third argument. Every call of `process_get_disk_diff_image(['/x/new'], ['/x/old'], …)` takes its paths from `new, old, out = _fake_disk_paths(tmp_path)`: `([new], [old], out, …)`. A test whose output was already `str(tmp_path / 'patch.img')` keeps that output. Add `tmp_path` to the signature of each test that lacks it. Concretely:
- `test_unmountable_filesystem_is_compared_as_raw_with_a_warning(tmp_path, hpcp_mod, monkeypatch, capsys)`: use `[new], [old], out`.
- `test_in_use_raw_area_is_refused_before_anything_is_hashed`:
  - parametrize becomes `@pytest.mark.parametrize('busy, side', [('/dev/new2', 'new'), ('/dev/old2', 'old')])`;
  - the signature gains `tmp_path`;
  - the body uses `disk = {'new': new, 'old': old}[side]` and `match=rf'^Disk in use: partition 2 of {re.escape(disk)} is held by another device; a raw area cannot be copied while it is in use'`.
- `test_active_swap_and_unchanged_raw_areas_are_not_refused`: add `tmp_path`; use `[new], [old], out`.
- `test_unmountable_in_use_filesystem_is_refused_as_a_raw_area`: add `tmp_path`; use `match=rf'^Disk in use: partition 1 of {re.escape(new)} is held by another device'`.
- `test_raw_area_too_large_for_one_volume_suggests_a_dis_that_passes`: add `tmp_path`; use `[new], [old], out`.
- `test_raw_areas_are_stored_before_section_payloads_largest_first` and `test_generation_classifies_sections_by_filesystem_type`: `_fake_generation(hpcp_mod, monkeypatch, tmp_path, parts('new'), parts('old'))` and `[new], [old]`, keeping their `out`.

Then add the guard test:

```python
def test_stubbed_generation_tests_use_no_real_directory():
	"""The /x/out hazard: a stub that regressed once let a test create and mount /x/out on the real filesystem."""
	with open(__file__, encoding='utf-8') as f:
		assert ("'" + '/x' + '/') not in f.read()
```

Append to `tests/test_disk_patch.py` (next to `test_probe_identity_reads_the_encoded_udev_values`):

```python
def test_probe_identity_falls_back_to_the_plain_uuid(hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""A blkid that prints ID_FS_UUID without ID_FS_UUID_ENC still gives the UUID (get_partition_details reads identity
	only through _probe_identity)."""
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check',
		lambda argv, **kwargs: ['ID_FS_UUID=ABCD-1234', 'ID_FS_TYPE=vfat'])
	assert hpcp_mod._probe_identity('/dev/x1') == ('vfat', 'ABCD-1234', '')


def test_lsblk_disk_rows_refuses_empty_output(hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', lambda argv, **kwargs: ['', '  '])
	with pytest.raises(RuntimeError, match=r'^Disk error: lsblk printed nothing for /dev/sdz$'):
		hpcp_mod._lsblk_disk_rows('/dev/sdz', 'NAME,SIZE')
```

In `tests/test_dd_fs_params.py`, the assertion at ~398 becomes:

```python
	assert '/dev/loop0p2' in err and f'set {_BTRFS_UUID} with btrfstune -U once no attached device carries {_BTRFS_UUID}' in err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py -q -k "plain_uuid or empty_output or no_real_directory or new_uuid_without_temp_fsid"`
Expected:
- `test_lsblk_disk_rows_refuses_empty_output` FAILs (IndexError).
- `test_write_partition_info_gives_btrfs_a_new_uuid_without_temp_fsid` FAILs, because the old wording is still in the code.
- `test_probe_identity_falls_back_to_the_plain_uuid` passes. It pins existing code that had no test.
- `test_stubbed_generation_tests_use_no_real_directory` passes once Step 1's conversion is complete. If it fails, a `/x/` path is left.

- [ ] **Step 3: Implement**

In `_lsblk_disk_rows`, replace `return rows[0], [row for row in rows[1:] if row[-1] == 'part']` with:

```python
	if not rows:
		raise RuntimeError(f'Disk error: lsblk printed nothing for {disk}')
	return rows[0], [row for row in rows[1:] if row[-1] == 'part']
```

In `write_partition_info`'s btrfs branch, the warning's second f-string becomes:

```python
						f"{target_partition} gets a new btrfs UUID; set {fs_uuid} with btrfstune -U once no attached device carries {fs_uuid}", file=sys.stderr)
```

In `process_get_disk_diff_image`, replace:

```python
	# Raw areas first, before any filesystem section is diffed: their sizes are checked before anything is
	# hashed, and an area too large for -dis is refused before the sections' file diffs when the raw areas alone
	# force more than one volume (otherwise below, once the sections' payload is known).
	for number in raw_numbers:
		new_size, old_size = new_layout['parts'][number]['size'], old_layout['parts'][number]['size']
		if new_size != old_size:
			raise RuntimeError(f"Disk layout mismatch: partition {number} is a raw area and its size changed ({old_size} vs {new_size}); re-clone instead")
	for number in raw_numbers:
```

with:

```python
	# Raw areas first, before any filesystem section is diffed (compare_disk_layouts already refused one whose size
	# changed, before anything was hashed): an area too large for -dis is refused before the sections' file diffs when
	# the raw areas alone force more than one volume (otherwise below, once the sections' payload is known).
	for number in raw_numbers:
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py -q -k "plain_uuid or empty_output or no_real_directory or temp_fsid or raw_size_change or in_use or unmountable or too_large or largest_first or classifies_sections"`
Expected: PASS.

- [ ] **Step 5: Run the affected files in full**

Run (as root): `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py tests/test_dd_fs_params.py
git commit -m "fix: stubbed generation tests under tmp_path; clear error on empty lsblk output; btrfs warning wording; drop dead raw-size loop"
```

---

### Task 4: payload re-check only where a source may have changed

**Files:**
- Modify `hpcp.py`:
  - `_mountinfo_entries` (~6525): add `super_options`;
  - `_mount_scopes` (~6641): split out `_mount_scope_entries`;
  - `get_entry_state` (~3999), `_states_chunk` and `_states_for_paths` (~4329-4361), `compute_changes` (~4363): collect scan stats;
  - add `_scan_stat`, `_wall_clock_ns`, `_TRUSTED_CTIME_FS`, `_RACY_WINDOW_NS` and `_rows_to_recheck` before `_check_payload`;
  - `_check_payload` (~4594), `_build_payload` (~4651), `build_patch_tree` (~4682);
  - `process_get_diff_image` (~5965 and the `build_patch_tree` call);
  - `_partition_changes` and its caller in `process_get_disk_diff_image`.
- Test: `tests/test_diff_image.py`, `tests/test_disk_patch.py`. Two existing `_build_payload` stubs need `**kwargs`.

**Interfaces:**
- Produces:
  - `compute_changes(..., scan=None)`: fills `scan['started_ns']` (int) and `scan['stats']` (`{key: (st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)}`).
  - `build_patch_tree(rows, new_base, volume_roots, meta, max_workers=1, verbose=False, scan=None)`.
  - `_build_payload(..., full_hash=False, scan=None)`.
  - `_check_payload(..., new_base=None, scan=None)`.
  - `_partition_changes(..., fat=False, scan=None)`.
  - `_mount_scope_entries(path, entries) -> [(prefix, entry or None)]`.
  - `_rows_to_recheck(rows, new_base, scan) -> list[ChangeRow]`.
  - mountinfo entries gain `'super_options': list[str]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_diff_image.py`)

```python
def _one_mount(hpcp_mod, monkeypatch, fs_type='ext4', options=('rw',), super_options=('rw',)):
	"""mountinfo holding every path on one mount at / (no ids: a test double is never hidden)."""
	entry = {'devnum': (0, 99), 'root': '/', 'mount_point': '/', 'options': list(options), 'fstype': fs_type,
		'source': '/dev/x', 'super_options': list(super_options)}
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry)])


def _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, change=lambda: None, max_workers=1, scan=True, volume='vol'):
	"""Scan new vs base (keeping the scan stats unless scan is False), change() the sources, then build the patch.
	Returns the payload copies _check_payload hashed again, relative to the volume."""
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	kept = {} if scan else None
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=max_workers, scan=kept)
	change()
	hashed = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		paths = list(paths)
		hashed.extend(paths)
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	vol = tmp_path / volume
	vol.mkdir()
	hpcp_mod.build_patch_tree(rows, str(new), [str(vol)], hpcp_mod.make_patch_meta(rows, 'n', 'b', 'root'),
		max_workers=max_workers, scan=kept)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', real)
	return sorted(os.path.relpath(path, str(vol)).replace(os.sep, '/') for path in hashed)


def test_mountinfo_entries_keep_super_options(tmp_path, hpcp_mod):
	path = tmp_path / 'mountinfo'
	path.write_text('36 35 98:0 / /mnt/a rw,noatime master:1 - ext4 /dev/sda1 ro,errors=continue\n'
		'37 35 98:1 / /mnt/b ro - vfat /dev/sda2\n')
	entries = hpcp_mod._mountinfo_entries(str(path))
	assert entries[0]['options'] == ['rw', 'noatime'] and entries[0]['super_options'] == ['ro', 'errors=continue']
	assert entries[1]['super_options'] == []


def test_compute_changes_keeps_the_scan_stats_of_new_files(trees, hpcp_mod):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_write(base / 'b.txt', 'b')
	before = int(time.time() * 1000000000)
	scan = {}
	hpcp_mod.compute_changes(*hpcp_mod.build_root_maps(str(new), str(base), max_workers=1), max_workers=1, scan=scan)
	st = os.lstat(str(new / 'a.txt'))
	assert scan['stats'] == {'a.txt': (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)}
	assert before <= scan['started_ns'] <= int(time.time() * 1000000000)


def test_payload_recheck_skips_a_read_only_filesystem(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 3 item 1: a source on a read-only superblock cannot change; its copies are not read again."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch, options=('ro',), super_options=('ro',))
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == []


@pytest.mark.parametrize('options', [('rw',), ('ro',)])
def test_payload_recheck_catches_a_rewrite_that_keeps_size_and_mtime(trees, hpcp_mod, tmp_path, monkeypatch, options):
	"""Only ctime shows this rewrite; a read-only bind of a writable filesystem (mount ro, superblock rw) is live."""
	new, base = trees
	_write(new / 'live.log', 'version 1')
	_one_mount(hpcp_mod, monkeypatch, options=options)
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	with pytest.raises(RuntimeError, match=_CHANGED_WHILE_BUILT):
		_scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, lambda: _write(new / 'live.log', 'version 2'))


def test_payload_recheck_trusts_an_unchanged_file_outside_the_racy_window(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_write(new / 'sub' / 'b.txt', 'b')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == []


def test_payload_recheck_hashes_a_file_changed_within_the_racy_window(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Written a moment before the scan: a second write in the same timestamp tick would keep its ctime."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == ['a.txt']


@pytest.mark.parametrize('fs_type', ['vfat', 'exfat', 'ntfs3', 'fuseblk', 'nfs4', 'cifs', 'zfs'])
def test_payload_recheck_hashes_files_on_an_untrusted_filesystem(trees, hpcp_mod, tmp_path, monkeypatch, fs_type):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch, fs_type=fs_type)
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == ['a.txt']


def test_payload_recheck_without_scan_stats_hashes_every_copy(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, scan=False) == ['a.txt']


def test_payload_recheck_hashes_a_metadata_change_and_keeps_the_copy(trees, hpcp_mod, tmp_path, monkeypatch):
	"""A chmod moves ctime: the copy is hashed again, matches its row, and the patch is written."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, lambda: os.chmod(str(new / 'a.txt'), 0o600)) == ['a.txt']
	assert (tmp_path / 'vol' / '.hpcp_diff' / 'manifest').exists()


def test_payload_recheck_hashes_files_on_an_untrusted_mount_below_the_source(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 3: `-diff /` with a FAT ESP at /boot/efi: only the ESP's files are read again."""
	new, base = trees
	_write(new / 'root.txt', 'r')
	_write(new / 'esp' / 'boot.efi', 'e')
	entries = [{'devnum': (0, 1), 'root': '/', 'mount_point': '/', 'options': ['rw'], 'fstype': 'ext4', 'source': '/dev/a',
			'super_options': ['rw']},
		{'devnum': (0, 2), 'root': '/', 'mount_point': os.path.realpath(str(new / 'esp')), 'options': ['rw'], 'fstype': 'vfat',
			'source': '/dev/b', 'super_options': ['rw']}]
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: [dict(entry) for entry in entries])
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch) == ['esp/boot.efi']


def test_payload_recheck_uses_scan_stats_from_worker_processes(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 1: with >= 256 paths the scan runs in a process pool; its stats must come back from the workers."""
	new, base = trees
	for index in range(300):
		_write(new / 'many' / f'f{index:03}.txt', f'file {index:03}')
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	assert _scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, max_workers=2) == []
	with pytest.raises(RuntimeError, match=r'^Diff image error: 1 file\(s\) changed while the patch was built \(e\.g\. many/f007\.txt\)'):
		_scan_and_build(hpcp_mod, new, base, tmp_path, monkeypatch, lambda: _write(new / 'many' / 'f007.txt', 'FILE 007'),
			max_workers=2, volume='vol2')


def test_payload_recheck_through_a_symlinked_new_root(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 2: the scan lists the link's target, the copy goes through the link; stats are keyed by row."""
	new, base = trees
	_write(new / 'live.log', 'version 1')
	link = tmp_path / 'link'
	os.symlink(str(new), str(link))
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	assert _scan_and_build(hpcp_mod, link, base, tmp_path, monkeypatch) == []
	with pytest.raises(RuntimeError, match=_CHANGED_WHILE_BUILT):
		_scan_and_build(hpcp_mod, link, base, tmp_path, monkeypatch, lambda: _write(new / 'live.log', 'version 2'), volume='vol2')


def test_directory_source_generation_passes_the_scan_to_the_payload_check(trees, hpcp_mod, tmp_path, monkeypatch):
	"""process_get_diff_image hands compute_changes' scan to build_patch_tree (without it every copy is read again)."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	vol = tmp_path / 'vol'
	vol.mkdir()
	monkeypatch.setattr('tempfile.tempdir', str(tmp_path))
	monkeypatch.setattr(hpcp_mod, '_claim_volume_names', lambda output, count: [output])
	monkeypatch.setattr(hpcp_mod, 'create_image', lambda *args, **kwargs: [str(vol)])
	_one_mount(hpcp_mod, monkeypatch)
	monkeypatch.setattr(hpcp_mod, '_RACY_WINDOW_NS', 0)
	rechecked = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		paths = list(paths)
		if kwargs.get('read_xattrs', True) is False:
			rechecked.extend(paths)
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	out = str(tmp_path / 'p.img')
	assert hpcp_mod.process_get_diff_image([str(new)], None, [str(base)], None, out, [], [], max_workers=1) == out
	assert rechecked == [] and (vol / '.hpcp_diff' / 'manifest').exists()


@requires_root_and_image_tools
@pytest.mark.root
@pytest.mark.loop
def test_image_source_payload_is_not_read_again(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 3 item 1 on a real read-only loop mount: a -si image's payload copies are not hashed a second time."""
	new, base = trees
	_write(new / 'a.txt', 'a' * 1000)
	_write(new / 'sub' / 'b.txt', 'b')
	image = str(tmp_path / 'new.img')
	rc = hpcp_mod.hpcp([str(new)], dest_image=image, single_thread=True, max_workers=1, batch=True, do_not_remove_files_while_listing=True)
	assert rc in (None, 0)
	rechecked = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		paths = list(paths)
		if kwargs.get('read_xattrs', True) is False:
			rechecked.extend(paths)
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	out = str(tmp_path / 'patch.img')
	with _mount_session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_diff_image([], [image], [str(base)], None, out, mount_points, loop_devices, max_workers=1) == out
	assert rechecked == []
```

`tests/test_diff_image.py` imports neither `time` nor `tempfile` at the top. Add `import time` to its imports. `tempfile.tempdir` is patched through the string form above, so `tempfile` needs no import.

Append to `tests/test_disk_patch.py` (after `test_raw_areas_are_stored_before_section_payloads_largest_first`):

```python
def test_disk_generation_passes_each_sections_scan_to_its_payload(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Each section's compute_changes scan reaches that section's _build_payload (Spec Part 3 item 4)."""
	parts = lambda side: {1: _part(fs_type='vfat', device=f'/dev/{side}1'), 3: _part(fs_type='ext4', device=f'/dev/{side}3')}
	_fake_generation(hpcp_mod, monkeypatch, tmp_path, parts('new'), parts('old'))
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda device, read_only, mount_points, entries: str(tmp_path / device[5:]) + os.sep)
	monkeypatch.setattr(hpcp_mod, '_release_mount', lambda mount_point, mount_points: None)
	monkeypatch.setattr(hpcp_mod, '_live_mount', lambda *args: (None, False))
	def changes(new_root, *args, scan=None, **kwargs):
		scan['marker'] = new_root
		return _one_row(hpcp_mod)
	monkeypatch.setattr(hpcp_mod, '_partition_changes', changes)
	monkeypatch.setattr(hpcp_mod, '_claim_volume_names', lambda output, count: [output])
	monkeypatch.setattr(hpcp_mod, 'create_image', lambda *args, **kwargs: [str(tmp_path / 'vol')])
	seen = []
	monkeypatch.setattr(hpcp_mod, '_build_payload', lambda rows, new_root, roots, *args, **kwargs: seen.append((new_root, kwargs['scan']['marker'])))
	monkeypatch.setattr(hpcp_mod, '_store_raw', lambda *args: None)
	monkeypatch.setattr(hpcp_mod, 'write_disk_patch_metadata', lambda *args: None)
	monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
	new, old, out = _fake_disk_paths(tmp_path)
	assert hpcp_mod.process_get_disk_diff_image([new], [old], out, [], [], max_workers=1) == out
	assert seen == [(root, root) for root in (str(tmp_path / 'new1') + os.sep, str(tmp_path / 'new3') + os.sep)]
```

This test uses `_fake_disk_paths` and `_fake_generation(hpcp_mod, monkeypatch, tmp_path, …)` from Task 3.

In the same file, the two existing stubs `lambda rows, new_root, roots, *args: …` of `_build_payload` (in `test_raw_areas_are_stored_before_section_payloads_largest_first` and `test_generation_classifies_sections_by_filesystem_type`) become `lambda rows, new_root, roots, *args, **kwargs: …`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "payload_recheck or super_options or scan_stats or passes_the_scan"`
Expected: FAIL.
- `compute_changes` and `build_patch_tree` reject the `scan` argument (TypeError).
- `test_mountinfo_entries_keep_super_options` hits a KeyError on `super_options`.

- [ ] **Step 3: Implement**

`_mountinfo_entries`: add a `super_options` key to the appended dict, after `'source': …`:

```python
					'super_options': fields[separator + 3].split(',') if len(fields) > separator + 3 else [],
```

`_mount_scopes`: replace the whole function with these two:

```python
def _mount_scope_entries(path, entries):
	"""[(key prefix, mountinfo entry or None)] for path: first ('', ...) for the visible mount holding path's real path
	(the deepest; None when no mount holds it), then one per visible mount below it, deepest last. A mount path lookup
	does not reach, or that another is stacked on, is left out (see _hidden_mounts), as on the system."""
	hidden = _hidden_mounts(entries)
	visible = [entry for index, entry in enumerate(entries) if index not in hidden]
	real = os.path.realpath(path)
	real_inside = real.rstrip(os.sep) + os.sep
	holder, depth, below = None, -1, []
	for entry in visible:
		inside = entry['mount_point'].rstrip(os.sep) + os.sep
		if real_inside.startswith(inside):
			if len(inside) >= depth:
				holder, depth = entry, len(inside)
		elif inside.startswith(real_inside):
			below.append((os.path.relpath(entry['mount_point'], real), entry))
	below.sort(key=lambda scope: (scope[0].count('/'), scope[0]))
	return [('', holder)] + below

def _mount_scopes(path, entries):
	"""[(key prefix, fstype, source)] for path (see _mount_scope_entries); fstype and source '' when no mount holds
	path. fstype is mountinfo's; see _scope_fs_type."""
	return [(prefix, entry['fstype'], entry.get('source', '')) if entry else (prefix, '', '')
		for prefix, entry in _mount_scope_entries(path, entries)]
```

`get_entry_state`:
- The signature becomes `def get_entry_state(path, full_hash=False, read_xattrs=True, mark_full=False, stats=None):`.
- Add to its docstring: `stats: a dict; a regular file's _scan_stat values, from the lstat taken before the file is read, are stored under path.`
- In the `if stat.S_ISREG(st.st_mode):` branch, insert before the `# hash_file is lru_cached…` comment:

```python
			if stats is not None:
				stats[path] = _scan_stat(st)
```

Replace `_states_chunk` and `_states_for_paths` with:

```python
def _states_chunk(paths, full_hash, read_xattrs, hash_size, marked=frozenset(), want_stats=False):
	"""Worker body for _states_for_paths: states, the errors recorded while computing them, and (want_stats) the
	regular files' _scan_stat values ({} otherwise).

	hash_size is the caller's HASH_SIZE; a worker that did not fork from the caller
	(spawn / forkserver) would otherwise hash with the module default.
	"""
	global HASH_SIZE
	HASH_SIZE = hash_size
	error_start = len(ERRORS)
	stats = {} if want_stats else None
	states = {path: get_entry_state(path, full_hash, read_xattrs, path in marked, stats) for path in paths}
	errors = ERRORS[error_start:]
	del ERRORS[error_start:]
	return states, errors, stats or {}

def _states_for_paths(paths, full_hash=False, read_xattrs=True, max_workers=1, marked=frozenset(), stats=None):
	"""{path: state} for every path; parallel chunks on a ProcessPoolExecutor for large sets. Paths in marked are
	hashed in full with the mark (see get_entry_state). stats: a dict filled with each regular file's _scan_stat values."""
	paths = list(paths)
	if max_workers <= 1 or len(paths) < 256:
		return {path: get_entry_state(path, full_hash, read_xattrs, path in marked, stats) for path in paths}
	chunk_size = max(64, len(paths) // (max_workers * 4) + 1)
	states = {}
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = []
		for index in range(0, len(paths), chunk_size):
			chunk = paths[index:index + chunk_size]
			futures.append(executor.submit(_states_chunk, chunk, full_hash, read_xattrs, HASH_SIZE,
				frozenset(path for path in chunk if path in marked), stats is not None))
		for future in concurrent.futures.as_completed(futures):
			chunk_states, chunk_errors, chunk_stats = future.result()
			states.update(chunk_states)
			ERRORS.extend(chunk_errors)
			if stats is not None:
				stats.update(chunk_stats)
	return states
```

`compute_changes`:
- The signature gains `scan=None` after `fat_paths=frozenset()`.
- Add to its docstring: `scan: a dict, filled for the payload re-check (_rows_to_recheck): 'started_ns', the wall clock before any state is read, and 'stats', {key: _scan_stat values} of new_map's regular files, from the lstat taken before each was read.`
- Replace the `states = _states_for_paths(...)` statement with:

```python
	stats = None
	if scan is not None:
		scan['started_ns'] = _wall_clock_ns()
		stats = {}
	states = _states_for_paths(set(new_map.values()) | set(base_map.values()), full_hash=full_hash,
		read_xattrs=read_xattrs and metadata, max_workers=max_workers, marked=marked, stats=stats)
	if scan is not None:
		scan['stats'] = {key: stats[path] for key, path in new_map.items() if path in stats}
```

Directly before `def _check_payload`, add:

```python
# Filesystems where every write to a file moves its ctime, which nothing can set back: an unchanged ctime then proves
# the content did not change (outside _RACY_WINDOW_NS). Everything else (FAT, exFAT, NTFS, FUSE, network filesystems)
# is hashed again.
_TRUSTED_CTIME_FS = frozenset(('ext2', 'ext3', 'ext4', 'xfs', 'btrfs', 'f2fs', 'tmpfs'))
# A file written again within the timestamp tick of its last write keeps its ctime ("racily clean"): a file whose
# scan-time ctime is this close to the scan's start, or later, is hashed again.
_RACY_WINDOW_NS = 2 * 1000000000

def _wall_clock_ns():
	"""Wall-clock time in ns (time.time_ns needs Python 3.7)."""
	return int(time.time() * 1000000000)

def _scan_stat(st):
	"""The lstat values the payload re-check compares: device, inode, size, mtime and ctime (ns)."""
	return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

def _rows_to_recheck(rows, new_base, scan):
	"""The rows (A / M file rows with a payload copy) whose copy _check_payload hashes again.

	None on a filesystem whose superblock is read-only (mountinfo super options; it cannot change). On any other,
	a row whose source may have changed since the scan read it: its filesystem is not in _TRUSTED_CTIME_FS, the scan
	(compute_changes' scan dict) has no stat for it, its scan-time ctime lies within _RACY_WINDOW_NS of the scan's
	start or later, or its lstat now differs from the scan's. Filesystems are found per row as _rows_by_scope groups
	rows (mounts below new_base included)."""
	scopes = _mount_scope_entries(new_base, _mountinfo_entries())
	entries = dict(scopes)
	scan = scan or {}
	stats, started = scan.get('stats', {}), scan.get('started_ns')
	recheck = []
	for prefix, scope_rows in _rows_by_scope(rows, [(prefix, '', '') for prefix, _ in scopes]).items():
		entry = entries.get(prefix)
		if entry is not None and 'ro' in entry.get('super_options', ()):
			continue
		trusted = entry is not None and entry['fstype'] in _TRUSTED_CTIME_FS
		for row in scope_rows:
			before = stats.get(row.path)
			if not trusted or before is None or started is None or before[4] >= started - _RACY_WINDOW_NS:
				recheck.append(row)
				continue
			try:
				now = _scan_stat(os.lstat(_key_path(new_base, row.path)))
			except OSError:
				now = None
			if now != before:
				recheck.append(row)
	return recheck
```

`_check_payload`:
- The signature becomes `def _check_payload(rows, volume_roots, full_hash=False, max_workers=1, new_base=None, scan=None):`.
- Append to its docstring: `A copy is hashed again only when its source may have changed since the scan read it (_rows_to_recheck; every copy when new_base is None).`
- Replace the two lines `marked = {copy for row, copy in copies.items() …}` and `states = _states_for_paths(set(copies.values()), …)` with:

```python
	recheck = set(_rows_to_recheck(list(copies), new_base, scan)) if new_base is not None else set(copies)
	copies = {row: copy for row, copy in copies.items() if row in recheck}
	marked = {copy for row, copy in copies.items() if _full_marked(row.new)}
	states = _states_for_paths(set(copies.values()), full_hash=full_hash, read_xattrs=False, max_workers=max_workers, marked=marked) if copies else {}
```

`_build_payload`:
- The signature becomes `def _build_payload(rows, new_base, volume_roots, max_workers=1, verbose=False, estimated_size=0, full_hash=False, scan=None):`.
- Its last line becomes `_check_payload(payload, volume_roots, full_hash, max_workers, new_base, scan)`.
- Add to its docstring: `scan: compute_changes' scan dict for these rows (see _rows_to_recheck).`

`build_patch_tree`:

```python
def build_patch_tree(rows, new_base, volume_roots, meta, max_workers=1, verbose=False, scan=None):
	"""Copy the A / M payload of rows from new_base onto the volume roots, then write .hpcp_diff/.
	scan: compute_changes' scan dict for rows (see _rows_to_recheck)."""
	_build_payload(rows, new_base, volume_roots, max_workers, verbose, meta.get('payload_bytes', 0),
		str(meta.get('full_hash', 0)) == '1', scan=scan)
	write_patch_metadata(volume_roots, rows, meta)
```

`process_get_diff_image` (two edits):
- `rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata, fat_paths=fat_paths)` becomes:

```python
	scan = {}
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata, fat_paths=fat_paths, scan=scan)
```

- `build_patch_tree(rows, new_base, volume_roots, meta, max_workers=max_workers, verbose=verbose)` becomes `build_patch_tree(rows, new_base, volume_roots, meta, max_workers=max_workers, verbose=verbose, scan=scan)`.

`_partition_changes`:
- The signature's last line becomes `input_files=(), metadata=True, fat=False, scan=None):`.
- Add `scan: compute_changes' scan dict (see _rows_to_recheck).` to its docstring.
- Its `compute_changes(...)` call gains `, scan=scan`.

`process_get_disk_diff_image`, in the section loop:

```python
				scan = {}
				rows = _partition_changes(new_root, old_root, mount_points, lives, exclude, full_hash, max_workers, parallel_file_listing,
					input_files, metadata=_has_posix_metadata(new_part['fs_type']),
					fat=_is_fat(new_part['fs_type']) or _is_fat(old_part['fs_type']), scan=scan)
```

The next change in that loop is `sections.append((number, new_root, rows))`, which becomes `sections.append((number, new_root, rows, scan))`. Further down:
- `section_rows = [(number, rows) for number, _, rows in sections]` becomes `section_rows = [(number, rows) for number, _, rows, _ in sections]`.
- The payload loop becomes:

```python
	for number, new_root, rows, scan in sections:
		section_roots = [os.path.join(volume_root, f'part{number}') for volume_root in volume_roots]
		for root in section_roots:
			os.makedirs(root, exist_ok=True)
		_build_payload(rows, new_root, section_roots, max_workers, verbose, _payload_bytes(rows), full_hash, scan=scan)
```

Run `grep -n "for number, _, rows in sections\|for number, new_root, rows in sections" hpcp.py`. Expected: no output.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "payload or super_options or scan_stats or passes_the_scan or states_for_paths or compute_changes"`
Then: `python -m pytest tests/test_disk_patch.py -q -k "scan_to_its_payload or largest_first or classifies_sections"`
Then, as root: `python -m pytest tests/test_diff_image.py -q -k image_source_payload_is_not_read_again`
Expected: PASS.

- [ ] **Step 5: Run the affected files in full**

Run (as root): `python -m pytest tests/test_diff_image.py tests/test_disk_patch.py tests/test_dd_fs_params.py -q`
Expected: all pass. Afterwards `losetup -a | grep -v snapd` prints nothing.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_diff_image.py tests/test_disk_patch.py
git commit -m "perf(diff): re-hash payload copies only when the source may have changed (read-only skip, stat check with racy window)"
```

---

### Task 5: README, version 9.71, full suite

**Files:**
- Modify: `README.md`, at lines ~84, ~118 and ~129.
- Modify: `hpcp.py`, at lines ~195-197 (`version`, `COMMIT_DATE`).

- [ ] **Step 1: README edits**

On line ~84, after the sentence `A file that changes between the comparison and the payload copy (on a live source) stops generation before any manifest is written; exclude it with \`-e\` or generate from an offline copy.`, append:

```markdown
 For a directory source, `-e` and `-x` patterns are matched against its real paths, as when copying; for a `-si` image they are matched against `/` plus the path inside the image's filesystem, so `-e '/var/log/*'` leaves out `var/log` there (and `*/tmp/*` only a `tmp/` inside it). This run's input images (`-si`, `-di`) and the base or new directory are left out of the patch when they lie inside a directory being compared, so `hpcp -diff / -di /root/golden-v1.img` does not install the old image on every node. A payload copy is read again to check it only when its source may have changed since the comparison: never on a read-only filesystem (a `-si` image, or a golden mounted read-only), and on a live one only when the file's inode, size, mtime or change time moved, it was changed within 2 seconds of the comparison, or it lies on a filesystem whose change time hpcp does not rely on (anything but ext2/3/4, xfs, btrfs, f2fs and tmpfs).
```

On line ~118, replace `` `-e` patterns are matched against the temporary paths hpcp mounts each partition at, so start them with `*/` (e.g. `-e '*/var/log/*'`). `` with:

```markdown
`-e` patterns are matched against `/` plus the path inside each partition's filesystem: `-e '/var/log/*'` leaves out `var/log` on the root partition and `-e '/EFI/Linux/*'` that directory on the ESP (a separate `/var` partition holds `log/` at its root, so there the pattern is `/log/*`); `*/…` patterns apply on every partition.
```

On line ~129, replace `and so does a raw-area size change when generating;` with:

```markdown
and so does, when generating, a size change of a partition that is raw by its type (one that falls back to raw because it cannot be mounted is checked once that is found);
```

- [ ] **Step 2: Version**

In `hpcp.py`, set `version = '9.71'`, and set `COMMIT_DATE` to the output of `date +%F` on the day this task runs.

- [ ] **Step 3: Full suite**

Run (as root, with no other hpcp suite running; check with `pgrep -af pytest`): `python -m pytest tests/ -q`
Expected: all pass, with the skip count unchanged (1 skipped).
Then check:
- `losetup -a | grep -v snapd` prints nothing;
- `dmsetup ls` lists only the host's own devices;
- `md5sum /usr/lib/os-release` prints `776b0aac9b17d49a0ac879da9801f51b`. If it differs, STOP and report; do not repair anything.

- [ ] **Step 4: Commit**

```bash
git add README.md hpcp.py
git commit -m "docs: -e inside mounted filesystems, inputs left out, payload re-check; release 9.71"
```
