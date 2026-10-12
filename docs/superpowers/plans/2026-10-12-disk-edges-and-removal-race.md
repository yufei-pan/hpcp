# Disk edge cases, the forced-removal race, and C1 polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `-dd` sees only a disk's own partitions, `-dd -diff` keeps this run's paths out on btrfs subvolume roots and binds, applies plan automounted ESPs as FAT, FAT attributes survive on every architecture, a forced removal can never act outside the target, and the C1 leftovers are closed (hpcp 9.72).

**Architecture:** Each fix is local to one helper:
- `_lsblk_disk_rows` parses `lsblk -P` pairs and filters partitions by `PKNAME`.
- `_live_excludes` maps paths through mountinfo roots.
- `_trigger_automounts` runs before `apply_patch_tree` plans its scopes.
- `_fat_ioctl_numbers` gives the ioctl numbers per architecture.
- `_remove_tree_around_mounts` reaches its top's parent by a descriptor walk from the target root.
- The polish items touch `process_compare_file_list`, `_rows_to_recheck` and `build_rooted_maps`.

**Tech Stack:** Python ≥ 3.6 single-file `hpcp.py`, pytest; root tests use loop devices (and btrfs-progs where present).

**Spec:** `docs/superpowers/specs/2026-10-12-disk-edges-and-removal-race-design.md` (approved, 0529c2f)

## Global Constraints

- One file, `hpcp.py`. Indentation is tabs. Python 3.6 floor: no walrus, no `time.time_ns`, no `dataclasses`. It must import on Windows, so new module-level code must not need `/proc` (`import platform` is fine).
- `lsblk -P` and the `KNAME` / `PKNAME` columns exist since util-linux 2.23 (CentOS 7). Do not use `lsblk -J` (2.27+).
- Tests are pytest functions with no classes. Real-device tests carry `@pytest.mark.root` and `@pytest.mark.loop` plus the file's `requires_root_and_disk_tools` / `requires_root_and_image_tools` skip marker. Scratch goes only under `tmp_path`, and nothing may be left mounted, attached or assembled.
- **Probe safety rule:** any ad-hoc root probe goes in a script file under `/tmp/claude-0/` run with `bash <file>`. `set -euo pipefail` typed at the top of a shell command is ignored in this harness. Every path variable gets a `: "${VAR:?}"` guard and a `case "$D" in /tmp/claude-0/*) ;; *) exit 1;; esac` check. Devices are only `/dev/loop*` backed by scratch images, or md/dm devices built on them and stopped or removed in a `trap`. Never touch the host's `pve-root` / `pve-swap`, the snap loop devices, or another session's processes. `/usr/lib/os-release` must keep md5 `776b0aac9b17d49a0ac879da9801f51b`.
- Full suite: `python -m pytest tests/ -q` (about 3.5 minutes, as root). Never run two full suites at once; check `pgrep -af "python.*pytest"` first.
- Version 9.72. `COMMIT_DATE` is the day the work is finished. No PyPI publish without the user's go.
- Patch formats are unchanged. Directory sources and copy-mode `-e` match exactly as before.

## Review Focus

1. **A disk whose lsblk `NAME` differs from its `KNAME`** (dm/multipath `/dev/mapper/mpatha` vs `/dev/dm-0`, a by-id path): its partitions must still be found. Pinned by the mpath and by-id cases of `test_partition_paths_come_from_lsblk_full_names` (Task 1).
2. **An lsblk value with a special character** (`-P` writes it as `\xNN`): it must decode back. Pinned by `test_lsblk_pairs_decode_escapes` (Task 1).
3. **A live mount of the same device that is this run's own temp mount**: it must never be taken as a live mount to map through. Pinned by the own-mount case of `test_live_excludes_map_paths_through_mount_roots` (Task 2).
4. **Descriptors on every exit path of the forced removal**: none leak, including refusals. Pinned by `test_forced_removal_refusal_leaks_no_descriptors` (Task 5).
5. **An automount the patch does not write into** (`/net`, NFS homes): it is never triggered. Pinned by `test_apply_leaves_unrelated_automounts_alone` (Task 3).

---

### Task 1: a disk's own partitions (`lsblk -P`, `KNAME` / `PKNAME`)

**Files:**
- Modify: `hpcp.py`:
  - `_lsblk_disk_rows` (~832-843): add `_LSBLK_PAIR_RE` and `_lsblk_pairs` just above it;
  - the docstrings of `get_largest_partition`, `get_partitions` and `get_target_partition` (wording only).
- Test: `tests/test_disk_patch.py` (fake-lsblk tests ~2968-3011; new tests next to them)

**Interfaces:**
- Produces:
  - `_lsblk_pairs(line) -> dict`;
  - `_lsblk_disk_rows(disk, columns) -> (list[str], list[list[str]])` (same return shape as today: the asked columns' values, disk row first).
- Consumes: `_udev_decode(value)` (exists, ~6313: decodes `\xNN` escapes to bytes, then UTF-8 with surrogateescape).

- [ ] **Step 1: Capture real md output (root probe, guarded)**

Write `/tmp/claude-0/c2-md-probe/probe.sh` and run `bash /tmp/claude-0/c2-md-probe/probe.sh /tmp/claude-0/c2-md-probe`:

```bash
#!/bin/bash
# C2 probe: lsblk -P of a loop disk that is a member of a partitioned md RAID1.
set -euo pipefail
D="${1:?scratch dir required}"
case "$D" in /tmp/claude-0/*) ;; *) echo "refusing: $D" >&2; exit 1;; esac
MD="/dev/md/hpcp-probe-$$"
A="" ; B="" ; MADE=""
cleanup() { set +e; if [ -n "$MADE" ]; then mdadm --stop "$MD" >/dev/null 2>&1; fi; for L in "$A" "$B"; do [ -n "$L" ] && losetup -d "$L"; done; rm -f -- "$D/a.img" "$D/b.img"; }
trap cleanup EXIT
mkdir -p -- "$D"
truncate -s 96M "$D/a.img" "$D/b.img"
A=$(losetup --find --show "$D/a.img"); : "${A:?}"; case "$A" in /dev/loop*) ;; *) exit 1;; esac
B=$(losetup --find --show "$D/b.img"); : "${B:?}"; case "$B" in /dev/loop*) ;; *) exit 1;; esac
mdadm --create "$MD" --level=1 --raid-devices=2 --metadata=1.2 --run "$A" "$B" >/dev/null 2>&1; MADE=1
sgdisk -n1:0:+16M -n2:0:+16M "$MD" >/dev/null
partprobe "$MD" 2>/dev/null || true
udevadm settle 2>/dev/null || true
echo "== lsblk -P of member $A"
lsblk -p -nb -P -o NAME,SIZE,KNAME,PKNAME,TYPE "$A"
echo done
```

Put the output in your report. Use its shape (TYPE `raid1`, PKNAME values, md partition names) in the md test case of Step 2. If it differs from the case written there, follow the real output and say so in the report. Afterwards `cat /proc/mdstat` must show no `hpcp-probe` array, and `losetup -a | grep -v snapd` must print nothing.

- [ ] **Step 2: Write the failing tests**

In `tests/test_disk_patch.py`, replace the parametrize block and body of `test_partition_paths_come_from_lsblk_full_names`, and the body of `test_target_partition_uses_lsblk_full_names`, with:

```python
def _lsblk_p(name, size, kname, pkname, kind):
	"""One `lsblk -p -nb -P` line."""
	return f'NAME="{name}" SIZE="{size}" KNAME="{kname}" PKNAME="{pkname}" TYPE="{kind}"'


@pytest.mark.parametrize('disk, lsblk, partitions, largest', [
	# An LVM LV spanning both partitions: larger than either, its name ends in '2', and it is not a partition.
	('/dev/cciss/c0d0', [_lsblk_p('/dev/cciss/c0d0', 1000, '/dev/cciss!c0d0', '', 'disk'),
		_lsblk_p('/dev/mapper/vg-data2', 990, '/dev/dm-3', '/dev/cciss!c0d0p1', 'lvm'),
		_lsblk_p('/dev/cciss/c0d0p1', 400, '/dev/cciss!c0d0p1', '/dev/cciss!c0d0', 'part'),
		_lsblk_p('/dev/cciss/c0d0p2', 600, '/dev/cciss!c0d0p2', '/dev/cciss!c0d0', 'part')],
		['/dev/cciss/c0d0p1', '/dev/cciss/c0d0p2'], '/dev/cciss/c0d0p2'),
	('/dev/disk/by-id/ata-DISK', [_lsblk_p('/dev/sda', 1000, '/dev/sda', '', 'disk'),
		_lsblk_p('/dev/sda1', 400, '/dev/sda1', '/dev/sda', 'part'), _lsblk_p('/dev/sda2', 600, '/dev/sda2', '/dev/sda', 'part'),
		_lsblk_p('/dev/mapper/luks-2', 584, '/dev/dm-0', '/dev/sda2', 'crypt')], ['/dev/sda1', '/dev/sda2'], '/dev/sda2'),
	('/dev/loop7', [_lsblk_p('/dev/loop7', 1000, '/dev/loop7', '', 'loop'),
		_lsblk_p('/dev/mapper/hpcp-holder-1', 900, '/dev/dm-1', '/dev/loop7p1', 'dm'),
		_lsblk_p('/dev/loop7p1', 900, '/dev/loop7p1', '/dev/loop7', 'part')], ['/dev/loop7p1'], '/dev/loop7p1'),
	# dm multipath: NAME (/dev/mapper/...) differs from KNAME (/dev/dm-N); parents are kernel names.
	('/dev/mapper/mpatha', [_lsblk_p('/dev/mapper/mpatha', 1000, '/dev/dm-0', '', 'mpath'),
		_lsblk_p('/dev/mapper/mpatha1', 400, '/dev/dm-1', '/dev/dm-0', 'part'),
		_lsblk_p('/dev/mapper/mpatha2', 600, '/dev/dm-2', '/dev/dm-0', 'part')],
		['/dev/mapper/mpatha1', '/dev/mapper/mpatha2'], '/dev/mapper/mpatha2'),
	# A whole-disk filesystem image: no partitions, so the disk itself is the largest.
	('/dev/loop8', [_lsblk_p('/dev/loop8', 1000, '/dev/loop8', '', 'loop')], [], '/dev/loop8'),
	# A disk that is a member of a partitioned md array: the array's partitions are TYPE part, but not the disk's.
	('/dev/loop9', [_lsblk_p('/dev/loop9', 1000, '/dev/loop9', '', 'loop'),
		_lsblk_p('/dev/md127', 990, '/dev/md127', '/dev/loop9', 'raid1'),
		_lsblk_p('/dev/md127p1', 400, '/dev/md127p1', '/dev/md127', 'part'),
		_lsblk_p('/dev/md127p2', 500, '/dev/md127p2', '/dev/md127', 'part')], [], '/dev/loop9'),
	# An md array on partition 1, partitioned itself: only the disk's own partitions count.
	('/dev/sdb', [_lsblk_p('/dev/sdb', 1000, '/dev/sdb', '', 'disk'),
		_lsblk_p('/dev/sdb1', 600, '/dev/sdb1', '/dev/sdb', 'part'),
		_lsblk_p('/dev/md126', 590, '/dev/md126', '/dev/sdb1', 'raid1'),
		_lsblk_p('/dev/md126p1', 580, '/dev/md126p1', '/dev/md126', 'part'),
		_lsblk_p('/dev/sdb2', 400, '/dev/sdb2', '/dev/sdb', 'part')], ['/dev/sdb1', '/dev/sdb2'], '/dev/sdb1'),
])
def test_partition_paths_come_from_lsblk_full_names(hpcp_mod, reset_hpcp_globals, monkeypatch, disk, lsblk, partitions, largest):
	"""lsblk -p prints each node's full path (cciss, by-id and kpartx alike), so those are used as they are. lsblk
	also lists every device stacked on the disk's partitions and those devices' own partitions (md127p1): only TYPE
	part rows whose parent (PKNAME) is the disk itself (its KNAME) are partitions."""
	argvs = []
	def fake_lsblk(argv, **kwargs):
		argvs.append(argv)
		return list(lsblk)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake_lsblk)
	assert hpcp_mod.get_partitions(disk) == partitions
	assert hpcp_mod.get_largest_partition(disk) == largest
	assert all('-p' in argv and '-P' in argv and argv[argv.index('-o') + 1] == 'NAME,SIZE,KNAME,PKNAME,TYPE' for argv in argvs)


def test_target_partition_uses_lsblk_full_names(hpcp_mod, reset_hpcp_globals, monkeypatch):
	argvs = []
	def fake_lsblk(argv, **kwargs):
		argvs.append(argv)
		# A dm holder of partition 1 whose name ends in '2' comes before the partitions, as lsblk lists it.
		return ['NAME="/dev/cciss/c0d0" KNAME="/dev/cciss!c0d0" PKNAME="" TYPE="disk"',
			'NAME="/dev/mapper/hpcp-holder-2" KNAME="/dev/dm-0" PKNAME="/dev/cciss!c0d0p1" TYPE="dm"',
			'NAME="/dev/cciss/c0d0p1" KNAME="/dev/cciss!c0d0p1" PKNAME="/dev/cciss!c0d0" TYPE="part"',
			'NAME="/dev/cciss/c0d0p2" KNAME="/dev/cciss!c0d0p2" PKNAME="/dev/cciss!c0d0" TYPE="part"']
	monkeypatch.setattr(hpcp_mod, 'create_loop_device', lambda image: '/dev/cciss/c0d0')
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake_lsblk)
	assert hpcp_mod.get_target_partition('/tmp/not-a-device.img', '2') == ('/dev/cciss/c0d0p2', '/dev/cciss/c0d0')
	assert '-P' in argvs[0] and argvs[0][argvs[0].index('-o') + 1] == 'NAME,KNAME,PKNAME,TYPE'


def test_lsblk_pairs_decode_escapes(hpcp_mod):
	"""Review Focus 2: lsblk -P writes special characters as \\xNN escapes."""
	assert hpcp_mod._lsblk_pairs('NAME="/dev/mapper/a\\x20b" SIZE="7" PKNAME=""') == {'NAME': '/dev/mapper/a b', 'SIZE': '7', 'PKNAME': ''}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "lsblk_full_names or target_partition_uses or lsblk_pairs"`
Expected: FAIL. `_lsblk_pairs` is missing, and the `-l` split parser cannot read pairs lines.

- [ ] **Step 4: Implement**

Replace `_lsblk_disk_rows` with:

```python
_LSBLK_PAIR_RE = re.compile(r'([A-Z][A-Z0-9:_-]*)="((?:[^"\\]|\\.)*)"')

def _lsblk_pairs(line):
	"""{column: value} of one `lsblk -P` line (KEY="value" pairs; lsblk writes special characters as \\xNN)."""
	return {key: _udev_decode(value) for key, value in _LSBLK_PAIR_RE.findall(line)}

def _lsblk_disk_rows(disk, columns):
	"""(disk row, partition rows) of `lsblk -p -nb -P -o <columns>,KNAME,PKNAME,TYPE disk`, each row the list of the
	asked columns' values.

	lsblk is asked for full paths (-p): it names each node itself (cciss, by-id and kpartx nodes alike), so no path
	is built here from the kernel names. The first row is the disk itself. lsblk also lists every device stacked on
	the disk's partitions (LVM, LUKS and dm-linear devices, md arrays) and those devices' own partitions (md127p1 of a
	partitioned md array built on the disk): a row is a partition of the disk only when its TYPE is part and its
	parent (PKNAME) is the disk itself (its KNAME). -P prints KEY="value" pairs, so the disk's empty PKNAME cannot
	shift the columns.
	"""
	names = columns.split(',')
	lines = run_command_in_multicmd_with_path_check(['lsblk', '-p', '-nb', '-P', '-o', columns + ',KNAME,PKNAME,TYPE', disk], strict=True)
	rows = [_lsblk_pairs(line) for line in lines if line.strip()]
	if not rows:
		raise RuntimeError(f'Disk error: lsblk printed nothing for {disk}')
	disk_row = rows[0]
	partitions = [row for row in rows[1:] if row.get('TYPE') == 'part' and row.get('PKNAME') == disk_row.get('KNAME')]
	return [disk_row.get(name, '') for name in names], [[row.get(name, '') for name in names] for row in partitions]
```

In the docstrings of `get_largest_partition`, `get_partitions` and `get_target_partition`, change "holders of its partitions are not partitions" (or "a holder lsblk also lists") to "devices stacked on its partitions, and their own partitions, are not partitions". `get_target_partition`'s loop `for row in partitions: if row[0].endswith(partition_name)` is unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py -q -k "lsblk or target_partition or partition_paths or largest"`
Then, as root: `python -m pytest tests/test_disk_patch.py -q -k "real_loop_disk or device_mapper_holder"`
Expected: PASS. The real dm-holder test still takes only the loop's own partitions.

- [ ] **Step 6: Run the dd and disk files in full**

Run (as root): `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py tests/test_dd_errors.py -q`
Expected: all pass. `losetup -a | grep -v snapd` prints nothing, and `dmsetup ls` lists only `pve-root` / `pve-swap`.

- [ ] **Step 7: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd): partitions are TYPE part rows whose lsblk parent is the disk itself (lsblk -P pairs)"
```

---

### Task 2: live mounts that show part of a filesystem (btrfs subvolumes, binds)

**Files:**
- Modify: `hpcp.py`:
  - add `_live_excludes` before `_partition_changes` (~7102);
  - `_partition_changes`: its signature and live loop;
  - the section loop of `process_get_disk_diff_image` (~7255-7259): drop `lives`, pass the devices.
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Produces:
  - `_live_excludes(root, device, paths, entries) -> set[str]`;
  - `_partition_changes(new_root, old_root, mount_points, devices, exclude, full_hash, max_workers, parallel_file_listing, input_files=(), metadata=True, fat=False, scan=None)`. The fourth parameter changes from `live_mounts` to `devices`: `(new_device, old_device)`, or `()` when there are none.
- Consumes: `_entry_matches(entry, devnum, device)`, `_device_number(device)`, `_inside_excludes(paths, roots)`, `_mountinfo_entries()` (all existing).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_disk_patch.py` after `test_input_disk_image_on_a_live_partition_stays_out_of_the_patch`)

```python
def _mount_entry(mount_point, root='/', devnum=(8, 3), source='/dev/sda3'):
	return {'mount_point': mount_point, 'root': root, 'devnum': devnum, 'source': source, 'fstype': 'btrfs',
		'options': ['rw'], 'super_options': ['rw']}


@pytest.mark.parametrize('own_root, live, paths, expected', [
	# Fedora: subvolume /root mounted at /, hpcp's own mount shows the top level.
	('/', [('/', '/root')], ['/nx/golden-v1.img', '/nx/tmp/tmpab'], {'/root/nx/golden-v1.img', '/root/nx/tmp/tmpab'}),
	# hpcp's own mount shows subvolume /root itself (a default subvolume set to it).
	('/root', [('/', '/root')], ['/nx/golden-v1.img'], {'/nx/golden-v1.img'}),
	# A bind of a subdirectory; a path outside every live mount of the device maps to nothing.
	('/', [('/nx-srv', '/data')], ['/nx-srv/x.img', '/nx-elsewhere/y.img'], {'/data/x.img'}),
	# The whole filesystem live (the case before 9.72).
	('/', [('/nx-live', '/')], ['/nx-live/a.img'], {'/a.img'}),
	# A live path outside what hpcp's own mount shows.
	('/home', [('/', '/root')], ['/nx/x.img'], set()),
	# Review Focus 3: this run's own temp mount of the same device is no live mount to map through.
	('/', [('/nx-run/tmpzz', '/')], ['/nx-run/tmpzz', '/nx-run/tmpzz/a.img'], set()),
])
def test_live_excludes_map_paths_through_mount_roots(hpcp_mod, monkeypatch, own_root, live, paths, expected):
	"""Spec Part 1 item 2: a path below a live mount M showing root R is R + its path below M in the filesystem's own
	terms; it is excluded relative to the root hpcp's own mount shows. Paths are under /nx* (nonexistent), so
	realpath leaves them as they are."""
	import glob as _glob
	own = '/nx-own'
	entries = ([_mount_entry(own, own_root)] + [_mount_entry(point, root) for point, root in live]
		+ [_mount_entry('/nx-other', '/', devnum=(7, 1), source='/dev/loop1')])
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (8, 3))
	assert hpcp_mod._live_excludes(own, '/dev/sda3', paths, entries) == {_glob.escape(path) for path in expected}


def test_live_excludes_without_an_own_mount_entry_is_empty(hpcp_mod, monkeypatch):
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (8, 3))
	assert hpcp_mod._live_excludes('/nx-not-mounted', '/dev/sda3', ['/nx/a.img'], [_mount_entry('/', '/root')]) == set()


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
@pytest.mark.skipif(shutil.which('mkfs.btrfs') is None or shutil.which('btrfs') is None, reason='needs btrfs-progs')
def test_input_disk_image_under_a_live_btrfs_subvolume_stays_out_of_the_patch(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Spec Part 1 item 2: a Fedora-style root (btrfs subvolume 'root' mounted live) holding the old golden image and
	this run's temp directories: none of them lands in the patch."""
	if not hpcp_mod._btrfs_temp_fsid_supported():
		pytest.skip('mounting the old copy next to the live one (same fsid) needs btrfs temp_fsid (kernel 6.7+)')
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	with open(str(old), 'wb') as f:
		f.truncate(320 << 20)
	_run('sgdisk', '-n1:2048:+300M', '-t1:8300', '-c1:root', old)
	with _attached(old) as loop:
		_run('mkfs.btrfs', '-q', '-f', loop + 'p1')
		with _mounted(loop + 'p1') as top:
			_run('btrfs', 'subvolume', 'create', os.path.join(top, 'root'))
			_write(os.path.join(top, 'root', 'etc', 'os-release'), b'v1')
	shutil.copyfile(str(old), str(new))
	with _attached(new) as loop:
		with _mounted(loop + 'p1', options='subvol=root') as live_root:
			_write(os.path.join(live_root, 'etc', 'os-release'), b'v2')
			inside = os.path.join(live_root, 'root', 'golden-v1.img')
			os.makedirs(os.path.dirname(inside))
			_run('cp', '--sparse=always', old, inside)
			temp_dir = os.path.join(live_root, 'tmp')
			os.makedirs(temp_dir)
			monkeypatch.setattr(tempfile, 'tempdir', temp_dir)
			with _session(hpcp_mod) as (mount_points, loop_devices):
				assert hpcp_mod.process_get_disk_diff_image([loop], [inside], out, mount_points, loop_devices, max_workers=1) == out
			monkeypatch.setattr(tempfile, 'tempdir', None)
	_, part1 = _read_part3_patch(hpcp_mod, out, section='part1')
	paths = {row.path for row in part1}
	assert ('M', 'root/etc/os-release') in {(row.op, row.path) for row in part1}
	assert 'root/root/golden-v1.img' not in paths
	assert [path for path in paths if path.startswith('root/tmp/') and path != 'root/tmp/'] == []
```

The new disk's partition is live only as subvolume `root`, so hpcp mounts the partition itself (read-only, its default subvolume = the top level) next to the live read-write mount. If this kernel refuses that second mount, `process_get_disk_diff_image` falls back to comparing p1 as a raw area, and the test fails on `_read_part3_patch` (no `part1` section). Report that as BLOCKED with the mount error; do not weaken the test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "live_excludes"`
Expected: FAIL (AttributeError: `_live_excludes`).
Then, as root: `python -m pytest tests/test_disk_patch.py -q -k live_btrfs_subvolume`
Expected: FAIL. `root/root/golden-v1.img`, or `root/tmp/tmp…/` entries, are in the patch.

- [ ] **Step 3: Implement**

Add before `_partition_changes`:

```python
def _live_excludes(root, device, paths, entries):
	"""Rooted exclude patterns (see _rooted_excludes) for the paths (this run's mount points and input disk images)
	that a live mount of device shows, as they lie in what hpcp's own mount of device at root shows.

	A live mount M of the device may show the whole filesystem, a btrfs subvolume (Fedora mounts / from subvolume
	root) or a bound directory: its mountinfo root R says which. A path P below M is R + P's path below M in the
	filesystem's own terms; it is excluded as its path below the root R0 hpcp's own mount shows (mountinfo root of
	root's mount), when it lies there. Entries of other devices, hpcp's own mount and this run's other mounts (the
	paths themselves) are not live mounts. Empty when root is not a mount point in entries."""
	root_real = os.path.realpath(root)
	own = next((entry for entry in entries if entry['mount_point'] == root_real), None)
	if own is None or not device:
		return set()
	own_root = own['root']
	devnum = _device_number(device)
	reals = [os.path.realpath(path) for path in paths]
	skip = set(reals) | {root_real}
	patterns = set()
	for entry in entries:
		point = entry['mount_point']
		if point in skip or not _entry_matches(entry, devnum, device):
			continue
		point_inside = point.rstrip(os.sep) + os.sep
		for real in reals:
			if not real.startswith(point_inside):
				continue
			inside = os.path.normpath(os.path.join(entry['root'], os.path.relpath(real, point)))
			if own_root != os.sep and not inside.startswith(own_root.rstrip(os.sep) + os.sep):
				continue
			patterns.add(glob.escape(os.sep + os.path.relpath(inside, own_root)))
	return patterns
```

In `_partition_changes`:
- Change its signature's fourth parameter from `live_mounts` to `devices`.
- Add to its docstring: `devices: (new device, old device) of the partition, whose live mounts may show this run's paths (see _live_excludes); () when there are none.`
- Replace:

```python
	internal = _inside_excludes(mount_points, (new_root, old_root))
	for live_mount in live_mounts:
		if live_mount:
			# A live filesystem holds the (empty) directories of this run's other temp mount points,
			# and may hold the disk images this run compares (the old golden image on the new golden).
			internal |= _inside_excludes(list(mount_points) + list(input_files), (live_mount,))
```

with:

```python
	internal = _inside_excludes(mount_points, (new_root, old_root))
	# A live mount of the partition (the whole filesystem, a btrfs subvolume or a bound directory) holds the (empty)
	# directories of this run's other temp mount points, and may hold the disk images this run compares (the old
	# golden image on the new golden). mountinfo is read now: hpcp's own mounts of both sides exist.
	entries = _mountinfo_entries() if devices else []
	for root, device in zip((new_root, old_root), devices):
		internal |= _live_excludes(root, device, list(mount_points) + list(input_files), entries)
```

In `process_get_disk_diff_image`'s section loop, delete the line
`lives = [_live_mount(_device_number(part['device']), entries, part['device'])[0] for part in (new_part, old_part)]`
and pass `(new_part['device'], old_part['device'])` where `lives` was passed to `_partition_changes`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py -q -k "live_excludes or pairs_names or classifies_sections or largest_first or scan_to_its_payload"`
Then, as root: `python -m pytest tests/test_disk_patch.py -q -k "live_btrfs_subvolume or live_partition or input_disk_image or rooted_in_each_partition"`
Expected: PASS. The existing ext4 live-partition tests are unchanged.

- [ ] **Step 5: Run the file in full**

Run (as root): `python -m pytest tests/test_disk_patch.py -q`
Expected: all pass. No mounts are left under `/tmp` (`findmnt -rn | grep -c /tmp/tmp` prints 0), and no extra loop devices remain.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd-diff): keep this run's paths out through every live mount (btrfs subvolumes, binds), mapped by mountinfo roots"
```

---

### Task 3: automounts before an apply plans its scopes; generation-order pin

**Files:**
- Modify: `hpcp.py`:
  - add `_trigger_automount` and `_trigger_automounts` before `apply_patch_tree` (~5913);
  - `apply_patch_tree`: its scope planning and its docstring.
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Produces:
  - `_trigger_automount(mount_point) -> None`;
  - `_trigger_automounts(target_root, rows, entries) -> list` (mountinfo entries).
- Consumes: `_mount_scopes(path, entries)`, `_rows_by_scope(rows, scopes)`, `_mountinfo_entries()`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_diff_image.py`)

```python
def _automount_entries(target, triggered, net=False):
	"""mountinfo for an ext4 / with an automounted ESP at target/boot/efi (autofs; vfat stacked on it once triggered),
	and optionally an untouched automount at target/net."""
	real = os.path.realpath(str(target))
	esp = os.path.join(real, 'boot', 'efi')
	entries = [
		{'mount_id': 20, 'parent_id': 1, 'devnum': (8, 2), 'root': '/', 'mount_point': '/', 'options': ['rw'],
			'fstype': 'ext4', 'source': '/dev/sda2', 'super_options': ['rw']},
		{'mount_id': 30, 'parent_id': 20, 'devnum': (0, 50), 'root': '/', 'mount_point': esp, 'options': ['rw'],
			'fstype': 'autofs', 'source': 'systemd-1', 'super_options': ['rw']},
	]
	if net:
		entries.append({'mount_id': 40, 'parent_id': 20, 'devnum': (0, 51), 'root': '/', 'mount_point': os.path.join(real, 'net'),
			'options': ['rw'], 'fstype': 'autofs', 'source': 'auto.net', 'super_options': ['rw']})
	if triggered:
		entries.append({'mount_id': 31, 'parent_id': 30, 'devnum': (8, 1), 'root': '/', 'mount_point': esp, 'options': ['rw'],
			'fstype': 'vfat', 'source': '/dev/sda1', 'super_options': ['rw']})
	return entries


def _esp_patch(hpcp_mod, tmp_path):
	"""A patch adding boot/efi/EFI/x.efi; returns (volume, target), the target holding boot/efi/ and net/."""
	new, base = tmp_path / 'new', tmp_path / 'base'
	_write(new / 'boot' / 'efi' / 'EFI' / 'x.efi', 'efi')
	(base / 'boot' / 'efi').mkdir(parents=True)
	for tree in (new, base):
		(tree / 'net').mkdir()
	vol = tmp_path / 'vol'
	_patch_from(hpcp_mod, new, base, [vol])
	target = tmp_path / 'target'
	_clone(base, target)
	return str(vol), target


def _apply_until_preflight(hpcp_mod, monkeypatch, volume, target, state, net=False):
	"""apply_patch_tree with faked mountinfo (see _automount_entries) up to the preflight; returns [(scope name,
	metadata, tz_tolerant)] of the planned sections and the automounts triggered."""
	triggered, planned = [], []
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: _automount_entries(target, state['mounted'], net))
	def trigger(mount_point):
		triggered.append(mount_point)
		state['mounted'] = state['mounts_when_triggered']
	monkeypatch.setattr(hpcp_mod, '_trigger_automount', trigger)
	real_plan = hpcp_mod._plan_section
	def plan(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers, metadata=True, case_insensitive=False, tz_tolerant=False):
		planned.append((name, metadata, tz_tolerant))
		return real_plan(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers, metadata, case_insensitive, tz_tolerant)
	monkeypatch.setattr(hpcp_mod, '_plan_section', plan)
	class Stop(Exception):
		pass
	def stop(plan):
		raise Stop()
	monkeypatch.setattr(hpcp_mod, '_preflight_section', stop)
	with pytest.raises(Stop):
		hpcp_mod.apply_patch_tree([volume], str(target), max_workers=1)
	return planned, triggered


def test_apply_triggers_the_automounted_esp_it_writes_into(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 1 item 3: the ESP at boot/efi is an untriggered automount (autofs); once triggered it is planned as
	the FAT filesystem it holds (content only, timezone tolerance)."""
	volume, target = _esp_patch(hpcp_mod, tmp_path)
	state = {'mounted': False, 'mounts_when_triggered': True}
	planned, triggered = _apply_until_preflight(hpcp_mod, monkeypatch, volume, target, state)
	assert triggered == [os.path.join(os.path.realpath(str(target)), 'boot', 'efi')]
	assert ('boot/efi/', False, True) in planned


def test_apply_refuses_an_automount_that_cannot_be_mounted(trees, hpcp_mod, tmp_path, monkeypatch):
	volume, target = _esp_patch(hpcp_mod, tmp_path)
	state = {'mounted': False, 'mounts_when_triggered': False}
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: _automount_entries(target, False))
	monkeypatch.setattr(hpcp_mod, '_trigger_automount', lambda mount_point: None)
	with pytest.raises(RuntimeError, match=r'^Patch error: .*boot/efi is an automount \(autofs\) that could not be mounted; mount it and re-run$'):
		hpcp_mod.apply_patch_tree([volume], str(target), max_workers=1)
	assert not (target / 'boot' / 'efi' / 'EFI').exists()


def test_apply_leaves_unrelated_automounts_alone(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Review Focus 5: an automount the patch does not write into (net/) is never triggered."""
	volume, target = _esp_patch(hpcp_mod, tmp_path)
	state = {'mounted': False, 'mounts_when_triggered': True}
	planned, triggered = _apply_until_preflight(hpcp_mod, monkeypatch, volume, target, state, net=True)
	assert triggered == [os.path.join(os.path.realpath(str(target)), 'boot', 'efi')]


def test_trigger_automount_enters_the_mount_point(tmp_path, hpcp_mod, monkeypatch):
	"""A plain stat of an automount point does not trigger it; a lookup through it (mount_point/.) does."""
	seen = []
	real_stat = os.stat
	monkeypatch.setattr(hpcp_mod.os, 'stat', lambda path, *args, **kwargs: seen.append(path) or real_stat(path, *args, **kwargs))
	hpcp_mod._trigger_automount(str(tmp_path))
	hpcp_mod._trigger_automount(str(tmp_path / 'missing'))
	assert seen == [os.path.join(str(tmp_path), '.'), os.path.join(str(tmp_path / 'missing'), '.')]


def test_generation_reads_mountinfo_after_its_listing_entered_any_automount(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 1 item 3: generation needs no trigger, since its listing walks into an automount before mountinfo is
	read (FAT detection then sees the mounted ESP). Pin that order."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	order = []
	real_maps, real_entries = hpcp_mod.build_root_maps, hpcp_mod._mountinfo_entries
	monkeypatch.setattr(hpcp_mod, 'build_root_maps', lambda *args, **kwargs: order.append('listing') or real_maps(*args, **kwargs))
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: order.append('mountinfo') or real_entries(*args, **kwargs))
	_rows_before_mkfs(hpcp_mod, monkeypatch, [str(new)], None, [str(base)], None, str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert 'mountinfo' in order and order.index('listing') < order.index('mountinfo')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "automount"`
Expected: the trigger tests fail (AttributeError: `_trigger_automount`). The generation-order test passes, because it pins existing behaviour.

- [ ] **Step 3: Implement**

Add before `apply_patch_tree`:

```python
def _trigger_automount(mount_point):
	"""Enter mount_point so that its automount (autofs) mounts what it stands for: a lookup through it (mount_point/.),
	since a plain stat of the mount point itself does not trigger one. A failure is left to the caller's new look at
	mountinfo."""
	try:
		os.stat(os.path.join(mount_point, '.'))
	except OSError:
		pass

def _trigger_automounts(target_root, rows, entries):
	"""mountinfo entries after triggering each visible automount (autofs) at or below target_root that rows write into
	(rows _rows_by_scope groups under it, the target's own scope included); entries as given when there is none.
	Other automounts (/net, NFS home directories) are left alone."""
	scopes = _mount_scopes(target_root, entries)
	grouped = _rows_by_scope(rows, scopes)
	real = os.path.realpath(target_root)
	points = [os.path.join(real, prefix) if prefix else real
		for prefix, fs_type, _ in scopes if fs_type == 'autofs' and prefix in grouped]
	for point in points:
		_trigger_automount(point)
	return _mountinfo_entries() if points else entries
```

In `apply_patch_tree`, replace:

```python
	scopes = _mount_scopes(target_root, _mountinfo_entries())
	grouped = _rows_by_scope(rows, scopes)
```

with:

```python
	# An automounted filesystem the patch writes into (an ESP systemd mounts on demand) is mounted first, so it is
	# planned as what it holds; one that stays an automount is refused before anything changes.
	scopes = _mount_scopes(target_root, _trigger_automounts(target_root, rows, _mountinfo_entries()))
	grouped = _rows_by_scope(rows, scopes)
	real = os.path.realpath(target_root)
	for prefix, fs_type, _ in scopes:
		if fs_type == 'autofs' and prefix in grouped:
			point = os.path.join(real, prefix) if prefix else real
			raise RuntimeError(f'Patch error: {point} is an automount (autofs) that could not be mounted; mount it and re-run')
```

Append to `apply_patch_tree`'s docstring: `An automount (autofs) the patch writes into is triggered before the scopes are planned (see _trigger_automounts); one that cannot be mounted is refused before anything changes.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "automount or scope or case_only or fat"`
Expected: PASS.

- [ ] **Step 5: Run the file in full**

Run (as root): `python -m pytest tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "fix(siff): trigger the automounts a patch writes into before planning scopes; refuse one that stays autofs"
```

---

### Task 4: FAT attribute ioctl numbers per architecture

**Files:**
- Modify: `hpcp.py`: add `import platform` to the imports (alphabetical, after `import pathlib`); the FAT ioctl constants (~5172-5173).
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Produces: `_fat_ioctl_numbers(machine=None) -> (int, int)`. Existing users of `_FAT_IOCTL_GET_ATTRIBUTES` / `_FAT_IOCTL_SET_ATTRIBUTES` are unchanged.

- [ ] **Step 1: Write the failing test** (append to `tests/test_diff_image.py`)

```python
@pytest.mark.parametrize('machine, numbers', [
	('x86_64', (0x80047210, 0x40047211)), ('i686', (0x80047210, 0x40047211)), ('aarch64', (0x80047210, 0x40047211)),
	('armv7l', (0x80047210, 0x40047211)), ('riscv64', (0x80047210, 0x40047211)), ('s390x', (0x80047210, 0x40047211)),
	('loongarch64', (0x80047210, 0x40047211)),
	('ppc64le', (0x40047210, 0x80047211)), ('ppc64', (0x40047210, 0x80047211)), ('ppc', (0x40047210, 0x80047211)),
	('mips', (0x40047210, 0x80047211)), ('mips64', (0x40047210, 0x80047211)), ('mipsel', (0x40047210, 0x80047211)),
	('sparc64', (0x40047210, 0x80047211)), ('alpha', (0x40047210, 0x80047211)), ('parisc64', (0x40047210, 0x80047211)),
])
def test_fat_ioctl_numbers_follow_the_architecture(hpcp_mod, machine, numbers):
	"""Spec Part 1 item 4: _IOR('r', 0x10, __u32) / _IOW('r', 0x11, __u32). alpha, mips, powerpc and sparc put
	_IOC_READ (2) and _IOC_WRITE (4) at bit 29, and parisc swaps the generic read and write bits: either way the two
	numbers come out the other way round."""
	assert hpcp_mod._fat_ioctl_numbers(machine) == numbers
	assert (hpcp_mod._FAT_IOCTL_GET_ATTRIBUTES, hpcp_mod._FAT_IOCTL_SET_ATTRIBUTES) == hpcp_mod._fat_ioctl_numbers()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_diff_image.py -q -k fat_ioctl_numbers`
Expected: FAIL (AttributeError: `_fat_ioctl_numbers`).

- [ ] **Step 3: Implement**

Replace the two constant lines:

```python
_FAT_IOCTL_GET_ATTRIBUTES = 0x80047210
_FAT_IOCTL_SET_ATTRIBUTES = 0x40047211
```

with:

```python
def _fat_ioctl_numbers(machine=None):
	"""(FAT_IOCTL_GET_ATTRIBUTES, FAT_IOCTL_SET_ATTRIBUTES) for machine (platform.machine() by default):
	_IOR('r', 0x10, __u32) and _IOW('r', 0x11, __u32). alpha, mips, powerpc and sparc put _IOC_READ (2) and
	_IOC_WRITE (4) at bit 29, and parisc swaps the generic read and write bits: on all of them the two numbers come
	out the other way round."""
	machine = (platform.machine() if machine is None else machine).lower()
	if machine.startswith(('alpha', 'mips', 'ppc', 'powerpc', 'sparc', 'parisc')):
		return 0x40047210, 0x80047211
	return 0x80047210, 0x40047211

_FAT_IOCTL_GET_ATTRIBUTES, _FAT_IOCTL_SET_ATTRIBUTES = _fat_ioctl_numbers()
```

Add `import platform` after `import pathlib` in the import block.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "fat_ioctl or fat_attributes or attributes"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "fix(fat): FAT attribute ioctl numbers per architecture (alpha, mips, powerpc, sparc, parisc)"
```

---

### Task 5: forced removals reach their top by a descriptor walk from the target root

**Files:**
- Modify: `hpcp.py`:
  - `_remove_tree_around_mounts` (~5400-5520);
  - its caller in `_apply_rows`' `remove()` (~5576);
  - add `_open_below` before `_remove_tree_around_mounts`.
- Test: `tests/test_diff_image.py` (next to the `test_remove_tree_around_mounts_*` tests, ~3106-3270)

**Interfaces:**
- Produces:
  - `_open_below(root_real, rel, flags) -> int` (a directory descriptor);
  - `_remove_tree_around_mounts(path, mount_points=None, target_root=None) -> (list[str], list[str])`. `target_root` None means `/`, which keeps every existing two-argument call working.

- [ ] **Step 1: Write the failing tests** (append after `test_remove_tree_around_mounts_path_fallback_matches_the_fd_walk`)

```python
def _victim_tree(tmp_path):
	"""target/a/b/victim/ with a file, and outside/a/b/victim/ with a file the removal must never touch."""
	for top in ('target', 'outside'):
		_write(tmp_path / top / 'a' / 'b' / 'victim' / 'f.txt', top)
	return tmp_path / 'target', tmp_path / 'outside'


def test_forced_removal_refuses_a_parent_resolving_outside_the_target(tmp_path, hpcp_mod):
	"""Spec Part 2: a directory above the top swapped for a symlink to outside before the removal resolves its parent:
	the parent's real path lies outside the target, so nothing is removed."""
	target, outside = _victim_tree(tmp_path)
	shutil.rmtree(str(target / 'a'))
	os.symlink(str(outside / 'a'), str(target / 'a'))
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(target / 'a' / 'b' / 'victim'), set(), target_root=str(target))
	assert kept == [] and errors and 'outside' in errors[0]
	assert (outside / 'a' / 'b' / 'victim' / 'f.txt').exists()


def test_forced_removal_refuses_a_parent_swapped_after_it_was_resolved(tmp_path, hpcp_mod, monkeypatch):
	"""Spec Part 2: the swap comes right after the removal resolved its parent; the descriptor walk (O_NOFOLLOW at every
	component) meets the symlink and refuses, so nothing outside the target is removed."""
	target, outside = _victim_tree(tmp_path)
	parent = os.path.realpath(str(target / 'a' / 'b'))
	real_realpath = os.path.realpath
	def realpath(path, *args, **kwargs):
		result = real_realpath(path, *args, **kwargs)
		if path == str(target / 'a' / 'b') and not os.path.islink(str(target / 'a')):
			os.rename(str(target / 'a'), str(target / 'a-old'))
			os.symlink(str(outside / 'a'), str(target / 'a'))
		return result
	monkeypatch.setattr(hpcp_mod.os.path, 'realpath', realpath)
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(target / 'a' / 'b' / 'victim'), set(), target_root=str(target))
	monkeypatch.setattr(hpcp_mod.os.path, 'realpath', real_realpath)
	assert parent.startswith(real_realpath(str(target)))
	assert kept == [] and errors
	assert (outside / 'a' / 'b' / 'victim' / 'f.txt').exists()
	assert (target / 'a-old' / 'b' / 'victim' / 'f.txt').exists()


def test_forced_removal_follows_a_symlinked_directory_inside_the_target(tmp_path, hpcp_mod):
	target = tmp_path / 'target'
	_write(target / 'real' / 'sub' / 'victim' / 'f.txt', 'x')
	os.symlink('real', str(target / 'link'))
	kept, errors = hpcp_mod._remove_tree_around_mounts(str(target / 'link' / 'sub' / 'victim'), set(), target_root=str(target))
	assert (kept, errors) == ([], [])
	assert not (target / 'real' / 'sub' / 'victim').exists() and (target / 'real' / 'sub').is_dir()


@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason='needs /proc/self/fd')
def test_forced_removal_refusal_leaks_no_descriptors(tmp_path, hpcp_mod):
	"""Review Focus 4: every descriptor the walk opened is closed, on refusals too."""
	target, outside = _victim_tree(tmp_path)
	before = len(os.listdir('/proc/self/fd'))
	(target / 'a' / 'b' / 'victim2').mkdir()
	hpcp_mod._remove_tree_around_mounts(str(target / 'a' / 'b' / 'victim2'), set(), target_root=str(target))
	shutil.rmtree(str(target / 'a'))
	os.symlink(str(outside / 'a'), str(target / 'a'))
	hpcp_mod._remove_tree_around_mounts(str(target / 'a' / 'b' / 'victim'), set(), target_root=str(target))
	hpcp_mod._remove_tree_around_mounts(str(target / 'missing' / 'x'), set(), target_root=str(target))
	assert len(os.listdir('/proc/self/fd')) == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "forced_removal"`
Expected: FAIL. `target_root` is an unexpected keyword (TypeError). Without it, the outside victim would be removed.

- [ ] **Step 3: Implement**

Add before `_remove_tree_around_mounts`:

```python
def _open_below(root_real, rel, flags):
	"""A descriptor of root_real/rel reached one component at a time from a descriptor of root_real, each opened with
	flags (O_NOFOLLOW | O_DIRECTORY): a component that is a symlink or no directory raises OSError (ELOOP, ENOTDIR)."""
	fd = os.open(root_real, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0))
	try:
		for part in ([] if rel in ('', '.') else rel.split(os.sep)):
			next_fd = os.open(part, flags, dir_fd=fd)
			os.close(fd)
			fd = next_fd
	except BaseException:
		os.close(fd)
		raise
	return fd
```

Change `_remove_tree_around_mounts`:
- Its signature becomes `def _remove_tree_around_mounts(path, mount_points=None, target_root=None):`.
- Add to its docstring: `target_root: the tree the removal must stay in (the apply's target; '/' when None). The top's parent is resolved once and must lie inside it; with the descriptor functions it is then reached from a descriptor of target_root one component at a time with O_NOFOLLOW, and the top is examined, entered and removed relative to its parent's descriptor. A directory above the top swapped for a symlink after the apply checked the path therefore refuses the removal instead of acting outside the target (this runs as root on trees other users may write to).`
- Replace everything from `# The parent is resolved, the last component never:` through the line `flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0)` with:

```python
	# The parent is resolved once, the last component never: a symlink there must not be followed to its target.
	stripped = path.rstrip(os.sep) or os.sep
	name = os.path.basename(stripped)
	if not name:
		return [], ['.: refusing to remove a filesystem root']
	root_real = os.path.realpath(target_root or os.sep)
	parent_real = os.path.realpath(os.path.dirname(stripped))
	top = os.path.join(parent_real, name)
	if parent_real != root_real and not parent_real.startswith(root_real.rstrip(os.sep) + os.sep):
		return [], [f'..: resolves outside {root_real}; not removed']
	if not _DIR_FD_WALK:
		try:
			top_info = os.lstat(top)
		except FileNotFoundError:
			return [], []
		except OSError as e:
			return [], [f'.: {e.strerror}']
		if not stat.S_ISDIR(top_info.st_mode):
			return [], ['.: not a directory (a symlink?)']
		if top in mount_points:
			return ['.'], []
		try:
			parent_dev = os.lstat(parent_real).st_dev
		except OSError as e:
			return [], [f'..: {e.strerror}']
		if parent_dev != top_info.st_dev:
			return ['.'], []
		return _remove_tree_by_path(top, top_info.st_dev, mount_points)
	kept, errors = [], []
	flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0)
	try:
		parent_fd = _open_below(root_real, os.path.relpath(parent_real, root_real), flags)
	except FileNotFoundError:
		return [], []
	except OSError as e:
		return [], [f'..: {e.strerror} (a directory above it changed while it was being removed); not removed']
	try:
		try:
			top_info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
		except FileNotFoundError:
			return [], []
		except OSError as e:
			return [], [f'.: {e.strerror}']
		if not stat.S_ISDIR(top_info.st_mode):
			return [], ['.: not a directory (a symlink?)']
		top_dev = top_info.st_dev
		if top in mount_points:
			return ['.'], []
		# A mount made on path itself after mountinfo was read (or a btrfs subvolume) is on another device than its parent.
		if os.fstat(parent_fd).st_dev != top_dev:
			return ['.'], []
```

The rest of the function, from `def open_dir(` to the final `return sorted(kept), errors`, moves one indentation level in, into this `try:` block, with three changes:
1. `enter(top, None, top, top_info)` becomes `enter(name, parent_fd, top, top_info)`.
2. In the frame-pop branch, `os.rmdir(frame[2])` (the top, removed by path) becomes `os.rmdir(name, dir_fd=parent_fd)`.
3. Close the outer `try:` with:

```python
	finally:
		os.close(parent_fd)
```

The inner `try: ... finally: for frame in stack: os.close(frame[0])` stays as it is, inside.

In `_apply_rows`' `remove()`, change `kept, errors = _remove_tree_around_mounts(path)` to `kept, errors = _remove_tree_around_mounts(path, target_root=target_root)`. `target_root` is a parameter of `_apply_rows` and is visible in the closure.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "forced_removal or remove_tree_around_mounts or forced or rmtree or mount_point"`
Expected: PASS. Every existing `_remove_tree_around_mounts` test is unchanged, including the by-path fallback test and the swapped-directory test.

- [ ] **Step 5: Run the file in full**

Run (as root): `python -m pytest tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "fix(siff): a forced removal reaches its top by a descriptor walk from the target root (no path race above it)"
```

---

### Task 6: polish — rooted `-e` in compare mode, holder lookup by dict, `-e /`, scan-stat pin

**Files:**
- Modify: `hpcp.py`:
  - `process_compare_file_list` (~8220-8245);
  - the `holder()` closure in `_rows_to_recheck` (~4663);
  - `build_rooted_maps` (~4330).
- Test: `tests/test_diff_image.py`

**Interfaces:**
- Consumes: `build_rooted_maps`, `_rooted_excludes`, `build_store_maps`, `build_entry_map`, `_trim_prefix` (existing).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_diff_image.py`)

```python
def _compare_rows(hpcp_mod, monkeypatch, *args, **kwargs):
	captured = {}
	monkeypatch.setattr(hpcp_mod, 'report_changes', lambda rows, *rest, **options: captured.setdefault('rows', rows))
	hpcp_mod.process_compare_file_list(*args, **kwargs)
	return captured['rows']


def test_compare_mode_roots_exclude_patterns_for_image_sources(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Spec Part 3 item 1: -si image sources (and a -di image base) are listed with rooted matching in compare mode,
	as in -diff -si; '*/tmp/*' no longer matches the temporary mount path."""
	image_root, store_root = tmp_path / 'image_root', tmp_path / 'store_root'
	_write(image_root / 'changed.txt', 'new content')
	_write(store_root / 'changed.txt', 'old')
	_write(image_root / 'var' / 'log' / 'a.log', 'new log')
	_write(store_root / 'var' / 'log' / 'old.log', 'old log')
	rows = _compare_rows(hpcp_mod, monkeypatch, [str(image_root)], [str(store_root)], max_workers=1,
		exclude=frozenset(['/var/log/*', '*/tmp/*']), dest_image='store.img', src_image_roots=[str(image_root)])
	# Something is compared at all (before 9.72, '*/tmp/*' matched both temp mount paths and excluded everything) ...
	assert ('M', 'changed.txt') in _ops(rows)
	# ... and '/var/log/*' left out what is under var/log on both sides.
	assert not [row for row in rows if row.path.startswith('var/log/') and row.path != 'var/log/']


def test_compare_mode_roots_exclude_patterns_against_directory_bases(trees, hpcp_mod, tmp_path, monkeypatch):
	"""Without -di, an image source's patterns are rooted on both sides, as -diff -si -d DIR does (C1 Part 1 item 3)."""
	image_root, base = tmp_path / 'image_root', tmp_path / 'base_dir'
	_write(image_root / 'var' / 'log' / 'a.log', 'new log')
	_write(base / 'var' / 'log' / 'old.log', 'old log')
	_write(image_root / 'changed.txt', 'new content')
	_write(base / 'changed.txt', 'old')
	rows = _compare_rows(hpcp_mod, monkeypatch, [str(image_root) + os.sep], [str(base) + os.sep], max_workers=1,
		exclude=frozenset(['/var/log/*']), src_image_roots=[str(image_root) + os.sep])
	assert any(row.op == 'M' and row.path.endswith('changed.txt') for row in rows)
	assert not [row for row in rows if 'var/log/' in row.path and not row.path.endswith('var/log/')]


def test_compare_mode_keeps_real_path_matching_for_directory_sources(trees, hpcp_mod, tmp_path, monkeypatch):
	new, base = trees
	_write(new / 'var' / 'log' / 'a.log', 'log')
	rows = _compare_rows(hpcp_mod, monkeypatch, [str(new) + os.sep], [str(base) + os.sep], max_workers=1,
		exclude=frozenset(['/var/log/*']))
	assert any(row.path.endswith('var/log/a.log') for row in rows)


def test_exclude_matching_the_root_empties_both_sides(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 3 item 3: '-e /' (a pattern matching the filesystem root itself) lists nothing on either side."""
	new, base = tmp_path / 'mnt_new', tmp_path / 'mnt_base'
	for root in (new, base):
		_write(root / 'a.txt', 'a')
	new_map, base_map = hpcp_mod.build_rooted_maps(str(new), str(base), frozenset([os.sep]), max_workers=1)
	assert set(new_map) == {'./'} == set(base_map)
	new_map, base_map = hpcp_mod.build_rooted_maps(str(new), str(base), frozenset(['/a.txt']), max_workers=1)
	assert set(new_map) == {'./'} == set(base_map)


def test_recheck_lookup_does_not_scan_every_mount_per_directory(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Spec Part 3 item 2: with thousands of mounts the lookup walks up each directory's real path instead of scanning
	every mount (5000 directories x 5000 mounts took seconds)."""
	new = tmp_path / 'new'
	rows, stats = [], {}
	for index in range(5000):
		path = _write(new / f'd{index:04}' / 'f.txt', 'x')
		key = f'd{index:04}/f.txt'
		rows.append(hpcp_mod.ChangeRow('A', key, '-', hpcp_mod.get_entry_state(path, False, False)))
		stats[key] = hpcp_mod._scan_stat(os.lstat(path))
	entries = [{'devnum': (0, 1), 'root': '/', 'mount_point': '/', 'options': ['rw'], 'fstype': 'ext4', 'source': '/dev/x',
		'super_options': ['rw']}]
	entries += [{'devnum': (0, 2 + index), 'root': '/', 'mount_point': f'/nx-mnt/m{index}', 'options': ['rw'], 'fstype': 'ext4',
		'source': '/dev/y', 'super_options': ['rw']} for index in range(5000)]
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda *args, **kwargs: entries)
	monkeypatch.setattr(hpcp_mod, '_racy_window_ns', lambda: 0)
	start = time.monotonic()
	assert hpcp_mod._rows_to_recheck(rows, str(new), {'stats': stats, 'started_ns': int(time.time() * 1000000000)}) == []
	assert time.monotonic() - start < 2.0


def test_compute_changes_collects_scan_stats_for_new_side_paths_only(trees, hpcp_mod, monkeypatch):
	"""C1 parked gap: base-side paths are never stat-collected (memory grows with the changes, not the tree)."""
	new, base = trees
	_write(new / 'a.txt', 'a')
	_write(base / 'b.txt', 'b')
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	asked = []
	real = hpcp_mod._states_for_paths
	def spy(paths, *args, **kwargs):
		asked.append(set(kwargs.get('stat_paths') or ()))
		return real(paths, *args, **kwargs)
	monkeypatch.setattr(hpcp_mod, '_states_for_paths', spy)
	hpcp_mod.compute_changes(new_map, base_map, max_workers=1, scan={})
	assert asked == [set(new_map.values())]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_diff_image.py -q -k "compare_mode or matching_the_root or recheck_lookup or new_side_paths_only"`
Expected:
- The two rooted compare-mode tests fail: `*/tmp/*` excludes everything (no `M changed.txt` row), or `/var/log/*` is
  not rooted (`var/log` files show up).
- `test_exclude_matching_the_root_empties_both_sides` fails, because the temp-mounted side still lists `a.txt`.
- The lookup test fails on the time bound, since the old holder scans every mount.
- The directory-source test and the scan-stat pin pass: they pin existing behaviour.

- [ ] **Step 3: Implement**

**Compare mode.** In `process_compare_file_list`, replace the `if dest_image: ... else: ...` listing block with:

```python
	if dest_image:
		for src in src_paths:
			if src in src_image_roots:
				# An image source and the -di image store, both mounted: patterns match '/' + the path inside each.
				src_new, src_base = build_rooted_maps(src, dests[0], exclude, max_workers, parallel_file_listing)
			else:
				src_new, src_base = build_store_maps(src, dests[0], exclude=exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing)
			new_map.update(src_new)
			base_map.update(src_base)
	else:
		# With image sources, patterns match '/' + the path inside each tree on both sides (as -diff -si -d DIR does).
		rooted = bool(src_image_roots)
		for src in src_paths:
			src_exclude = _rooted_excludes(exclude or (), src) if src in src_image_roots else exclude
			new_map.update(build_entry_map(src, _trim_prefix(src), exclude=src_exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
		for dest in dests:
			dest_exclude = _rooted_excludes(exclude or (), dest) if rooted else exclude
			base_map.update(build_entry_map(dest, _trim_prefix(dest), exclude=dest_exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
```

**Holder lookup.** In `_rows_to_recheck`, replace the `holder()` closure body that scans `visible` per directory:

```python
		if directory not in holders:
			real_directory = os.path.realpath(directory).rstrip(os.sep)
			inside, found, depth = real_directory + os.sep, None, -1
			for point, entry in visible:
				if inside.startswith(point) and len(point) >= depth:
					found, depth = entry, len(point)
			holders[directory] = (found, real_directory)
```

with a walk up the real path's parents:

```python
		if directory not in holders:
			real_directory = os.path.realpath(directory).rstrip(os.sep)
			# The deepest visible mount holding it: its own real path or the nearest parent that is a mount point
			# ('' stands for '/'), looked up in points, so the cost grows with the depth, not with the mounts.
			key, found = real_directory, None
			while True:
				if key in points:
					found = points[key]
					break
				if not key:
					break
				key = key.rpartition(os.sep)[0]
			holders[directory] = (found, real_directory)
```

If `visible` has no other user left in `_rows_to_recheck`, keep building `points` from the visible entries directly and drop the unused `visible` list. Update the docstring sentence about the lookup if it mentions scanning.

**`-e /`.** In `build_rooted_maps`, replace the two `build_entry_map` lines with:

```python
	# A pattern matching the filesystem root itself ('-e /'): each side's own root is excluded too, so nothing is
	# listed on either side (a '/' base already was, by its walker's check of the root).
	whole = any(fnmatch.fnmatch(os.sep, pattern) for pattern in exclude)
	def rooted(root):
		patterns = _rooted_excludes(exclude, root)
		return patterns | {glob.escape(os.path.abspath(root).rstrip(os.sep)) or os.sep} if whole else patterns
	new_map = build_entry_map(new_root, '', None, rooted(new_root), None, max_workers, parallel_file_listing)
	base_map = build_entry_map(base_root, '', None, rooted(base_root), None, max_workers, parallel_file_listing)
```

The scan-stat pin needs no code change.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_diff_image.py -q -k "compare or rooted or recheck or payload or matching_the_root or new_side_paths_only or file_list"`
Expected: PASS. The C1 re-check tests (file mount, mount below the source, store layout) give the same answers.

- [ ] **Step 5: Run the affected files in full**

Run (as root): `python -m pytest tests/test_diff_image.py tests/test_file_lists.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_diff_image.py
git commit -m "fix: rooted -e for image sources in compare mode; recheck lookup by mount-point dict; -e / empties both sides; pin scan stats to the new side"
```

---

### Task 7: docs, version 9.72, full suite

**Files:**
- Modify:
  - `docs/superpowers/specs/2026-10-12-exclude-inputs-recheck-design.md` (lines ~83-84, ~187, ~191);
  - `README.md` (lines ~84, ~92, ~118, plus the Disk Imaging Feature Note ~56);
  - `hpcp.py` (`version`, `COMMIT_DATE`, ~195-197);
  - `tests/test_dd_fs_params.py` (`test_version_bumped`).

- [ ] **Step 1: C1 spec version ranges**

In `docs/superpowers/specs/2026-10-12-exclude-inputs-recheck-design.md`:
- In Part 1 item 7, `(` + "`-diff` first shipped in 9.67, so only 9.67–9.70 are affected" + `)` becomes "(`-diff` first shipped in 9.65, so only 9.65–9.70 are affected)".
- In the next line, "matched in 9.67–9.70" becomes "matched in 9.65–9.70".
- In Compatibility, "Patches made by 9.67–9.70 apply as before" becomes "Patches made by 9.65–9.70 apply as before", and "matched in 9.67–9.70 and" becomes "matched in 9.65–9.70 and".

- [ ] **Step 2: README**

1. In line ~84, replace `never on a read-only filesystem (a \`-si\` image, or a golden mounted read-only)` with `on a read-only filesystem (a \`-si\` image, or a golden mounted read-only) only when the file's stat moved`.
2. In line ~92, after the sentence `Rows are planned per filesystem: the target's own and each one mounted below it (for example the ESP at \`/boot/efi\` when applying to \`/\`).`, insert:
   ` An automounted filesystem the patch writes into (an ESP that systemd mounts on demand, shown as autofs until something enters it) is mounted first, so it is planned as what it holds; one that cannot be mounted stops the apply before anything changes. A file written by one filesystem's rows and then removed, through a symlinked directory, by a later filesystem's delete is reported by verify (failed), not written again; and an entry replaced twice by contradictory rows of one patch can in principle reuse an inode number, which verify then cannot tell apart (both need patches that contradict themselves).`
3. In line ~118 (the `-dd -diff` paragraph), after the `-e` sentence, insert:
   ` This run's temporary directories and its disk-image files are left out wherever a live mount of a partition shows them: the whole filesystem, a btrfs subvolume mounted as the root (Fedora's \`root\`), or a bound directory.`
4. In the Disk Imaging Feature Note, at the end of the paragraph that starts `` `-dd --disk_dump` mode differs ``, append:
   ` A disk's partitions are the ones whose parent is the disk itself: devices stacked on them (md arrays, LVM, dm-crypt) and the partitions of such devices (a partitioned md array built on the disk) are never taken for the disk's own.`

- [ ] **Step 3: Version**

In `hpcp.py`, set `version = '9.72'` and `COMMIT_DATE` to the output of `date +%F`. In `tests/test_dd_fs_params.py::test_version_bumped`, set `'9.72'` and the same date.

- [ ] **Step 4: Full suite**

Check that no other suite is running (`pgrep -af "python.*pytest"`), then run as root: `python -m pytest tests/ -q`
Expected: all pass. The skip count is 1, or 2 if the btrfs subvolume test skips for lack of `temp_fsid` (it must not skip on this host).
Then check:
- `losetup -a | grep -v snapd` prints nothing;
- `dmsetup ls` lists only `pve-root` / `pve-swap`;
- `cat /proc/mdstat` lists no hpcp array;
- `md5sum /usr/lib/os-release` prints `776b0aac9b17d49a0ac879da9801f51b`. If it differs, STOP and report; do not repair anything.

- [ ] **Step 5: Commit**

```bash
git add README.md hpcp.py tests/test_dd_fs_params.py docs/superpowers/specs/2026-10-12-exclude-inputs-recheck-design.md
git commit -m "docs: automounts, live subvolume roots, stacked partitions, known limitations; C1 version ranges; release 9.72"
```
