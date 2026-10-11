# Disk-Level Patch Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `hpcp -dd -diff [PATCH] NEW_DISK -d OLD_DISK` write one patch image for a whole GPT disk (per-partition sections, raw areas, identity changes) and make `hpcp -siff PATCH -d TARGET_DISK` apply it to an offline or live node disk.

**Architecture:** Builds on the shipped phase-1 Diff Image section of `hpcp.py`. A new "Disk Patch" section adds format v2 (manifest `raw`/`disk` entries, `disk.tsv`), disk-layout reading and matching, raw-area hashing/writing, live-disk detection (mountinfo, swaps, holders), identity commands, and the generate/apply wrappers. Phase-1 apply is refactored into per-section plan → shared conflict prompt → apply/verify steps so a disk patch runs one phase-1 section per filesystem partition.

**Tech Stack:** Python 3 stdlib (`hashlib`, `contextlib` added to imports), existing hpcp helpers (`get_partition_infos`, `create_loop_device`, `create_image`, copy engine, phase-1 diff units), `sgdisk`, `blkid`, `losetup`, `mount`, filesystem label tools; pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-dd-diff-image-design.md` (phase 1: `docs/superpowers/specs/2026-10-10-diff-image-design.md`)

## Global Constraints

- **Single-file module.** All production code goes in `hpcp.py`. No new modules or packages.
- **Tabs for indentation** in `hpcp.py` and `tests/`.
- **Python 3.6 syntax floor**: no walrus, no `str.removeprefix`, no `dict | dict`, no `f"{x=}"`.
- **No new runtime dependencies**; stdlib additions only (`hashlib` and `contextlib` imported at module top).
- **Errors vs warnings.** `eprint("Category: detail")` only for real failures (it records into `ERRORS`; exit code comes from the text before the first `:`). Warnings and notes use `print(..., file=sys.stderr)` / `print(...)`.
- **New error categories:** `Disk layout mismatch` 248, `Raw area error` 249, `Identity change error` 250, `Disk in use` 251.
- **Formats:** disk patches write manifest `format_version 2` with `layout disk`; single-filesystem patches keep writing `format_version 1` (so hpcp 9.65 can still apply them). `read_manifest` accepts versions up to 2. Section names `part<N>` (payload dir = name, change list `part<N>.changes.tsv`); raw names `part<N>` or `mbr`, payload files `raw/<bare name>`; `disk.tsv` header `#hpcp-disk<TAB>1`.
- **GPT only** (`sgdisk`). Partition pairing by number; same count, type GUIDs, filesystem types and sector size; raw areas need equal sizes.
- **Every new external binary goes into `_binCalled`** (hpcp.py ≈ line 422) so `check_path()` resolves it: `e2label`, `fatlabel`, `ntfslabel`, `btrfstune`, `f2fslabel`.
- **Phase-1 behaviour must not change**: every existing test in `tests/` keeps passing (except the explicitly updated expectations named in Tasks 1 and 9).
- **Version target:** `9.66`, bumped only in the final task.
- Run tests from the repository root with `python -m pytest tests/ -q`. Root/loop tests are marked `root` and `loop` and skip when not root or tools are missing; this machine runs as root, so they execute.
- Commits end with the `Co-Authored-By` trailer the implementing agent's harness attribution specifies.

## Review Focus

1. **Partition numbers ≥ 10** (`loop0p1` vs `loop0p11`): name-suffix matching picks the wrong device; the kernel's partition number must decide. Test: Task 2 `test_partition_devices_use_kernel_numbers`.
2. **A live golden partition with other filesystems mounted below it** (`/boot/efi`, `/proc`, a tmpfs): their contents must not leak into the root section, and their mount-point directories must not become deletions. Test: Task 7 `test_live_partition_hides_nested_mounts`.
3. **Node-specific identities** (a node's filesystem UUID differs from the golden base): a warning, not a refusal, and the new value is still set. Test: Task 5 `test_plan_identity_warns_when_target_differs_from_base`.
4. **A damaged raw payload inside the patch image** (bit rot, truncated file): must be refused before anything on the target changes. Tests: Task 3 `test_raw_payload_rejects_damaged_or_escaping_files`, Task 8 `test_damaged_raw_payload_is_refused_up_front`.
5. **Re-running an interrupted apply** (raw area or identity already new): must skip what is done, not conflict. Tests: Task 3 `test_plan_raw_actions`, Task 5 `test_plan_identity_skips_values_already_new`, Task 8 `test_disk_apply_round_trip_and_rerun`.

## Deviations from the spec (decided while planning)

- A filesystem section whose partition is **live-mounted read-only** is refused up front (`Disk in use`) instead of a forceable conflict — forcing could not write to a read-only mount.
- Live partitions are reached through a **non-recursive bind mount** of their live mount (generation read-only via `remount,bind,ro`, apply read-write), so filesystems mounted below them stay out of view; this is how dd mode already reaches live partitions.
- Filesystem sections are applied in **partition-number order with the ESP last** ("root first" cannot be identified reliably from GPT data).
- In `-dd -diff`, a `-diff` value that names an **existing path** (disk image or block device) is treated as the source path argparse swallowed (phase 1 only rescued directories); the output must not exist anyway.
- Code units: the spec's `_read_mbr_bootcode` is dropped (`_hash_area` / `_copy_area` / `_write_area` take an offset and length), and `_live_mounts(device, mountinfo_path)` / `_partition_in_use(part, …)` become `_mountinfo_entries` + `_live_mount` + `_partition_in_use(device, devnum, entries, swaps)` so mountinfo is read once per run.
- Filesystem type, UUID and label are read with an uncached `blkid -p` (`_probe_identity`): the cached `blkid` in `get_partition_details` can return a value it read under 2 s ago, which would fail the verify step right after an identity change. `get_partition_details`' own blkid call becomes quiet: exit 2 (no filesystem, e.g. BIOS boot) was recorded as an error, in dd mode too.
- Raw areas are re-hashed in the final verify pass (after the identity commands), not straight after each write, so an `sgdisk` GPT rewrite that disturbed sector 0 would be caught.

---

## File Structure

| File | Responsibility |
|---|---|
| `hpcp.py` (modify) | Format v2 (`read_manifest`/`write_manifest`, `disk.tsv`), phase-1 refactors (`_load_patch`, section plan/apply, `_build_payload`, `_payload_allocation`, `_claim_volume_names`, `_ro_mount_options`, `_fs_identity_commands` used by `write_partition_info`), new "Disk Patch" section, CLI wiring, error codes, `_binCalled`, imports |
| `tests/test_disk_patch.py` (create) | All new tests: format v2, layouts, raw areas, live detection, identity, multi-section apply, root+loop GPT fixtures for generation/apply/CLI |
| `tests/test_diff_image.py` (modify) | Two updated expectations (format version 3 is "newer"; the `-dd` validation case) |
| `tests/test_dd_fs_params.py` (modify) | Version pin in `test_version_bumped` (final task) |
| `README.md` (modify) | "Disk patches" subsection, regenerated help block |

**Where new code goes:** format and phase-1 refactors edit the existing functions in place (the Diff Image section, `#%% ---- Diff Image ----`). Everything new for disks goes into a new section that starts with `#%% ---- Disk Patch ----` and sits **immediately before `#%% ---- Remove Extra ----`** (i.e. after `_validate_diff_args`). Tasks append to the end of that section.

---

### Task 1: Format v2 — manifest raw/disk entries and `disk.tsv`

**Files:**
- Modify: `hpcp.py` (`DIFF_FORMAT_VERSION` ≈ line 3725, `write_manifest` / `read_manifest` ≈ 3970–4022, `make_patch_meta` ≈ 4295; new code at the end of the Diff Image section, just above `_logically_excluded`)
- Create: `tests/test_disk_patch.py`
- Modify: `tests/test_diff_image.py` (`test_read_manifest_rejects_foreign_or_newer`)

**Interfaces:**
- Produces:
  - `DIFF_FORMAT_VERSION = 2` (newest readable), `_SINGLE_FS_FORMAT_VERSION = 1` (written by `make_patch_meta`)
  - `read_manifest(source) -> dict` with `meta['sections']` (list of `(name, payload_dir, changes_file)`) and `meta['raws']` (list of `(name, file, old_sha256, new_sha256, size:int)`); `write_manifest(target, meta)` writes both
  - `DiskRow = namedtuple('DiskRow', 'kind number field old new')` (all `str`; `number` is `'-'` for disk rows)
  - `write_disk_tsv(target, rows)`, `read_disk_tsv(source) -> list[DiskRow]`
  - `_SECTION_NAME_RE` (`part[1-9][0-9]*`), `_DISK_FIELDS`, `_PART_FIELDS`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_disk_patch.py`:

```python
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import pytest


T0 = 1700000000  # whole even second (FAT keeps 2 s mtimes)


def _write(path, data=b'x', mtime=T0):
	path = str(path)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'wb') as f:
		f.write(data if isinstance(data, bytes) else data.encode('utf-8', 'surrogateescape'))
	os.utime(path, (mtime, mtime))
	return path


_A, _B, _C, _D = 'a' * 64, 'b' * 64, 'c' * 64, 'd' * 64


def _disk_meta():
	return {'format': 'hpcp-diff', 'format_version': 2, 'layout': 'disk', 'patch_id': 'abc', 'volume': '1/1',
		'sections': [('part1', 'part1', 'part1.changes.tsv'), ('part3', 'part3', 'part3.changes.tsv')],
		'raws': [('part2', 'raw/part2.img', _A, _B, 1048576), ('mbr', 'raw/mbr-bootcode.bin', _C, _D, 440)],
		'disk': 'disk.tsv'}


def test_manifest_v2_round_trips_raw_and_disk_entries(tmp_path, hpcp_mod):
	meta = _disk_meta()
	path = tmp_path / 'manifest'
	hpcp_mod.write_manifest(str(path), meta)
	got = hpcp_mod.read_manifest(str(path))
	assert got['layout'] == 'disk' and got['format_version'] == '2' and got['disk'] == 'disk.tsv'
	assert got['sections'] == meta['sections']
	assert got['raws'] == meta['raws']


_DISK_HEAD = 'format\thpcp-diff\nformat_version\t2\nlayout\tdisk\ndisk\tdisk.tsv\n'


@pytest.mark.parametrize('line', [
	f'raw\tpart2\traw/../x\t{_A}\t{_B}\t1',
	f'raw\tpart2\tpart2.img\t{_A}\t{_B}\t1',
	f'raw\tswap\traw/x.img\t{_A}\t{_B}\t1',
	f'raw\tpart0\traw/x.img\t{_A}\t{_B}\t1',
	f'raw\tpart2\traw/x.img\tzz\t{_B}\t1',
	f'raw\tpart2\traw/x.img\t{_A}\t{_B}\t-1',
	'section\tpart1\tother\tpart1.changes.tsv',
	'section\tpart1\tpart1\tchanges.tsv',
	'section\troot\t.\tchanges.tsv',
	'raws\tx',
	f'raw\tpart2\traw/a.img\t{_A}\t{_B}\t1\nraw\tpart2\traw/b.img\t{_A}\t{_B}\t1',
	'section\tpart1\tpart1\tpart1.changes.tsv\nsection\tpart1\tpart1\tpart1.changes.tsv',
])
def test_disk_manifest_rejects_bad_entries(tmp_path, hpcp_mod, line):
	path = tmp_path / 'manifest'
	path.write_text(_DISK_HEAD + line + '\n')
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.read_manifest(str(path))


def test_raw_and_disk_entries_need_layout_disk(tmp_path, hpcp_mod):
	path = tmp_path / 'manifest'
	path.write_text(f'format\thpcp-diff\nformat_version\t2\nraw\tpart2\traw/p.img\t{_A}\t{_B}\t1\n')
	with pytest.raises(RuntimeError, match='layout disk'):
		hpcp_mod.read_manifest(str(path))
	path.write_text('format\thpcp-diff\nformat_version\t1\nlayout\tdisk\ndisk\tdisk.tsv\n')
	with pytest.raises(RuntimeError, match='format version 1'):
		hpcp_mod.read_manifest(str(path))
	path.write_text('format\thpcp-diff\nformat_version\t2\nlayout\tdisk\n')
	with pytest.raises(RuntimeError, match='disk table'):
		hpcp_mod.read_manifest(str(path))


def test_single_filesystem_patches_still_write_format_version_1(hpcp_mod):
	assert hpcp_mod.make_patch_meta([], 'n', 'b', 'root')['format_version'] == 1


def test_disk_tsv_round_trip(tmp_path, hpcp_mod):
	R = hpcp_mod.DiskRow
	rows = [
		R('disk', '-', 'guid', 'G-OLD', 'G-NEW'),
		R('disk', '-', 'sector_size', '512', '512'),
		R('part', '1', 'name', 'EFI\tSystem', 'EFI System 2'),
		R('part', '3', 'fs_label', 'root', 'root2'),
		R('part', '12', 'kind', 'raw', 'raw'),
	]
	path = tmp_path / 'disk.tsv'
	hpcp_mod.write_disk_tsv(str(path), rows)
	assert path.read_text().splitlines()[0] == '#hpcp-disk\t1'
	assert hpcp_mod.read_disk_tsv(str(path)) == rows


@pytest.mark.parametrize('body', [
	'part\t1\tcolour\ta\tb\n',
	'part\t0\tname\ta\tb\n',
	'disk\t1\tguid\ta\tb\n',
	'disk\t-\tname\ta\tb\n',
	'part\t1\tname\ta\n',
	'part\t1\tname\ta\\q\tb\n',
])
def test_disk_tsv_rejects_bad_rows(tmp_path, hpcp_mod, body):
	path = tmp_path / 'disk.tsv'
	path.write_text('#hpcp-disk\t1\n' + body)
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.read_disk_tsv(str(path))
	path.write_text('#hpcp-disk\t2\n')
	with pytest.raises(RuntimeError, match='newer'):
		hpcp_mod.read_disk_tsv(str(path))
```

In `tests/test_diff_image.py`, `test_read_manifest_rejects_foreign_or_newer`, change the line

```python
	path.write_text('format\thpcp-diff\nformat_version\t2\n')
```

to

```python
	path.write_text('format\thpcp-diff\nformat_version\t3\n')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q`
Expected: FAIL — `KeyError: 'raws'` / `AttributeError: module 'hpcp' has no attribute 'DiskRow'`, and the v2 manifest rejected as "newer".

- [ ] **Step 3: Implement**

Replace the line `DIFF_FORMAT_VERSION = 1` with:

```python
DIFF_FORMAT_VERSION = 2
# Single-filesystem patches keep the version hpcp 9.65 reads; disk patches (layout disk) are version 2.
_SINGLE_FS_FORMAT_VERSION = 1
```

In `make_patch_meta`, change `'format_version': DIFF_FORMAT_VERSION,` to `'format_version': _SINGLE_FS_FORMAT_VERSION,`.

Replace `write_manifest` and `read_manifest` with:

```python
_SECTION_NAME_RE = re.compile(r'part[1-9][0-9]*')
_SHA256_RE = re.compile(r'[0-9a-f]{64}')

def write_manifest(target, meta):
	"""Write meta as key/value TSV.

	'sections' is a list of (name, payload_dir, changes_file) and 'raws' a list of
	(name, file, old_sha256, new_sha256, size); both are written as repeatable lines.
	"""
	with open(target, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		for key, value in meta.items():
			if key not in ('sections', 'raws'):
				f.write(f'{key}\t{_escape_field(str(value))}\n')
		for name, payload_dir, changes_file in meta.get('sections', []):
			f.write(f'section\t{_escape_field(name)}\t{_escape_field(payload_dir)}\t{_escape_field(changes_file)}\n')
		for raw in meta.get('raws', []):
			f.write('raw\t' + '\t'.join(_escape_field(str(field)) for field in raw) + '\n')

def _check_raw_entry(fields):
	"""Validate one manifest raw line (name, file, old, new, size); returns it with size as int."""
	name, raw_file, old_sha, new_sha, size = fields
	bare = raw_file[len('raw/'):] if raw_file.startswith('raw/') else ''
	if not (name == 'mbr' or _SECTION_NAME_RE.fullmatch(name)):
		raise ValueError(f'bad raw area name {name!r}')
	if not bare or '/' in bare or bare in ('.', '..'):
		raise ValueError(f'bad raw payload file {raw_file!r}')
	if not (_SHA256_RE.fullmatch(old_sha) and _SHA256_RE.fullmatch(new_sha)):
		raise ValueError('bad raw area hash')
	if not size.isdigit():
		raise ValueError(f'bad raw area size {size!r}')
	return (name, raw_file, old_sha, new_sha, int(size))

def read_manifest(source):
	"""Read a manifest. Values are strings; 'sections' and 'raws' are lists of tuples.

	Unknown keys are kept and ignored by callers. Raw entries and the disk table are only
	valid in a layout disk manifest (format_version 2).
	"""
	meta = {'sections': [], 'raws': []}
	seen_keys = set()
	with open(source, 'r', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		for line in f:
			line = line.rstrip('\n')
			if not line or line.startswith('#'):
				continue
			try:
				fields = [_unescape_field(field) for field in line.split('\t')]
			except ValueError as e:
				raise RuntimeError(f'Patch format error: {source}: {e}')
			if fields[0] == 'section' and len(fields) == 4:
				name, payload_dir, changes_file = fields[1:]
				try:
					if not name or '/' in name:
						raise ValueError('bad section name')
					if payload_dir != '.':
						_check_key(payload_dir)
					if not changes_file or '/' in changes_file or changes_file in ('.', '..'):
						raise ValueError('bad changes file')
				except ValueError:
					raise RuntimeError(f'Patch format error: {source} has an unsafe section line {line!r}')
				meta['sections'].append(tuple(fields[1:]))
			elif fields[0] == 'raw' and len(fields) == 6:
				try:
					meta['raws'].append(_check_raw_entry(fields[1:]))
				except ValueError as e:
					raise RuntimeError(f'Patch format error: {source} has an unsafe raw line {line!r}: {e}')
			elif fields[0] not in ('section', 'raw') and len(fields) == 2:
				key, value = fields
				if key in ('sections', 'raws'):
					raise RuntimeError(f'Patch format error: {source} uses reserved key {key}')
				if key in seen_keys:
					raise RuntimeError(f'Patch format error: {source} has a duplicate key {key!r}')
				seen_keys.add(key)
				meta[key] = value
			else:
				raise RuntimeError(f'Patch format error: {source} has a malformed line {line!r}')
	if meta.get('format') != 'hpcp-diff':
		raise RuntimeError(f'Patch format error: {source} is not an hpcp-diff manifest')
	try:
		format_version = int(meta.get('format_version', ''))
	except ValueError:
		raise RuntimeError(f'Patch format error: {source} has a bad format_version')
	if format_version > DIFF_FORMAT_VERSION:
		raise RuntimeError(f'Patch format error: {source} uses format version {format_version}, newer than supported {DIFF_FORMAT_VERSION}')
	if meta.get('layout') == 'disk':
		if format_version < 2:
			raise RuntimeError(f'Patch format error: {source} has layout disk but format version {format_version}')
		for name, payload_dir, changes_file in meta['sections']:
			if not _SECTION_NAME_RE.fullmatch(name) or payload_dir != name or changes_file != name + '.changes.tsv':
				raise RuntimeError(f'Patch format error: {source} has a bad disk section {name!r}')
		for kind, names in (('section', [section[0] for section in meta['sections']]), ('raw area', [raw[0] for raw in meta['raws']])):
			if len(set(names)) != len(names):
				raise RuntimeError(f'Patch format error: {source} names a {kind} twice')
		disk_file = meta.get('disk', '')
		if not disk_file or '/' in disk_file or disk_file in ('.', '..'):
			raise RuntimeError(f'Patch format error: {source} has no valid disk table entry')
	elif meta['raws'] or 'disk' in meta:
		raise RuntimeError(f'Patch format error: {source} has raw areas or a disk table outside layout disk')
	return meta
```

Immediately after `read_manifest` (still in the Diff Image section) add:

```python
DISK_HEADER = '#hpcp-disk'
DISK_VERSION = 1
DiskRow = namedtuple('DiskRow', 'kind number field old new')
_DISK_FIELDS = ('guid', 'sector_size')
_PART_FIELDS = ('kind', 'type_guid', 'unique_guid', 'name', 'attrs', 'size', 'fs_type', 'fs_uuid', 'fs_label', 'fs_params')

def write_disk_tsv(target, rows):
	"""Write disk.tsv: escaped kind / number / field / old / new rows."""
	with open(target, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		f.write(f'{DISK_HEADER}\t{DISK_VERSION}\n')
		f.write('#kind\tnumber\tfield\told\tnew\n')
		for row in rows:
			f.write('\t'.join(_escape_field(str(value)) for value in row) + '\n')

def read_disk_tsv(source):
	"""Read disk.tsv into DiskRows. Raises RuntimeError('Patch format error: ...')."""
	rows = []
	with open(source, 'r', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		header = f.readline().rstrip('\n').split('\t')
		if len(header) != 2 or header[0] != DISK_HEADER or not header[1].isdigit():
			raise RuntimeError(f'Patch format error: {source} is not an hpcp disk table')
		if int(header[1]) > DISK_VERSION:
			raise RuntimeError(f'Patch format error: {source} has disk table version {header[1]}, newer than supported {DISK_VERSION}')
		for line_number, line in enumerate(f, start=2):
			line = line.rstrip('\n')
			if not line or line.startswith('#'):
				continue
			try:
				fields = [_unescape_field(field) for field in line.split('\t')]
			except ValueError as e:
				raise RuntimeError(f'Patch format error: {source} line {line_number}: {e}')
			if len(fields) != 5:
				raise RuntimeError(f'Patch format error: {source} line {line_number} is malformed')
			kind, number, field = fields[:3]
			ok = ((kind == 'disk' and number == '-' and field in _DISK_FIELDS) or
				(kind == 'part' and number.isdigit() and int(number) > 0 and field in _PART_FIELDS))
			if not ok:
				raise RuntimeError(f'Patch format error: {source} line {line_number} has an unknown row {kind} {number} {field}')
			rows.append(DiskRow(*fields))
	return rows

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py tests/test_diff_image.py
git commit -m "feat(dd-diff): manifest v2 raw and disk entries and disk.tsv"
```

---

### Task 2: Disk layouts — read, compare, match

**Files:**
- Modify: `hpcp.py` (new `#%% ---- Disk Patch ----` section immediately before `#%% ---- Remove Extra ----`; `get_partition_details` ≈ line 1671)
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: `get_partition_infos(device)` / `get_partition_details(...)` (existing, both `lru_cache`d), `get_partitions(disk)` (existing), `DiskRow` (Task 1).
- Produces:
  - `_ESP_TYPE_GUID = 'C12A7328-F81F-11D2-BA4B-00A0C93EC93B'`
  - `_partition_devices(disk, sys_block_path='/sys/class/block') -> {int: '/dev/…'}`
  - `_probe_identity(device) -> (fs_type, fs_uuid, fs_label)` (uncached `blkid -p`; nothing left in `ERRORS`)
  - `read_disk_layout(device) -> {'disk': {'guid', 'sector_size', 'size'}, 'parts': {int: {'type_guid', 'unique_guid', 'name', 'attrs', 'size', 'fs_type', 'fs_uuid', 'fs_label', 'fs_params', 'device'}}}`
  - `_default_kind(part) -> 'fs' | 'swap' | 'raw'`
  - `compare_disk_layouts(old, new, kinds=None) -> (problems: list[str], rows: list[DiskRow])`
  - `match_target_layout(disk_rows, target_layout) -> list[str]`
  - Test helpers (in `tests/test_disk_patch.py`): `requires_root_and_disk_tools`, `_run`, `_attached`, `_mounted`, `_make_golden`, `_make_newer`, `_session`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
def _part(**overrides):
	part = {'type_guid': '0FC63DAF-8483-4772-8E79-3D69D8477DE4', 'unique_guid': 'U1', 'name': 'root', 'attrs': '0000000000000000',
		'size': 41943040, 'fs_type': 'ext4', 'fs_uuid': 'F1', 'fs_label': 'root', 'fs_params': 'block_size=4096', 'device': '/dev/x3'}
	part.update(overrides)
	return part


def _layout(parts, guid='DISK-GUID', sector_size=512):
	return {'disk': {'guid': guid, 'sector_size': sector_size, 'size': 100 << 20}, 'parts': parts}


def test_partition_devices_use_kernel_numbers(tmp_path, hpcp_mod, monkeypatch):
	"""Review Focus 1: loop0p1 and loop0p11 must not be confused by name suffix."""
	names = {'loop0p1': '1', 'loop0p11': '11', 'loop0p2': '2'}
	for name, number in names.items():
		(tmp_path / name).mkdir()
		(tmp_path / name / 'partition').write_text(number + '\n')
	monkeypatch.setattr(hpcp_mod, 'get_partitions', lambda disk: ['/dev/' + name for name in names])
	got = hpcp_mod._partition_devices('/dev/loop0', sys_block_path=str(tmp_path))
	assert got == {1: '/dev/loop0p1', 11: '/dev/loop0p11', 2: '/dev/loop0p2'}


def test_compare_disk_layouts_identical(hpcp_mod):
	old = _layout({1: _part(fs_type='vfat', type_guid=hpcp_mod._ESP_TYPE_GUID), 3: _part()})
	problems, rows = hpcp_mod.compare_disk_layouts(old, old)
	assert problems == []
	fields = {(row.kind, row.number, row.field) for row in rows}
	assert ('disk', '-', 'guid') in fields and ('part', '3', 'fs_label') in fields and ('part', '1', 'kind') in fields
	assert all(row.old == row.new for row in rows)


@pytest.mark.parametrize('new_parts, sector_size, expected', [
	({1: _part(), 3: _part()}, 512, 'partition numbers'),
	({1: _part(type_guid='OTHER')}, 512, 'type_guid'),
	({1: _part(fs_type='xfs')}, 512, 'fs_type'),
	({1: _part()}, 4096, 'sector size'),
	({1: _part(fs_type='', size=2048)}, 512, 'raw area'),
])
def test_compare_disk_layouts_refusals(hpcp_mod, new_parts, sector_size, expected):
	old = _layout({1: _part(fs_type='' if expected == 'raw area' else 'ext4', size=1024)}, sector_size=512)
	new = _layout(new_parts, sector_size=sector_size)
	problems, _ = hpcp_mod.compare_disk_layouts(old, new)
	assert any(expected in problem for problem in problems), problems


def test_compare_disk_layouts_allows_filesystem_size_and_identity_changes(hpcp_mod):
	old = _layout({3: _part(size=1 << 30)})
	new = _layout({3: _part(size=2 << 30, fs_label='root2', unique_guid='U2')}, guid='DISK2')
	problems, rows = hpcp_mod.compare_disk_layouts(old, new, kinds={3: 'fs'})
	assert problems == []
	changed = {(row.number, row.field): (row.old, row.new) for row in rows if row.old != row.new}
	assert changed == {('-', 'guid'): ('DISK-GUID', 'DISK2'), ('3', 'size'): (str(1 << 30), str(2 << 30)),
		('3', 'fs_label'): ('root', 'root2'), ('3', 'unique_guid'): ('U1', 'U2')}


def test_match_target_layout(hpcp_mod):
	base = _layout({1: _part(fs_type='', size=1048576), 3: _part()})
	_, rows = hpcp_mod.compare_disk_layouts(base, base, kinds={1: 'raw', 3: 'fs'})
	assert hpcp_mod.match_target_layout(rows, _layout({1: _part(fs_type='', size=1048576), 3: _part(size=8 << 30, fs_uuid='NODE')})) == []
	assert any('size' in p for p in hpcp_mod.match_target_layout(rows, _layout({1: _part(fs_type='', size=2097152), 3: _part()})))
	assert any('fs_type' in p for p in hpcp_mod.match_target_layout(rows, _layout({1: _part(fs_type='', size=1048576), 3: _part(fs_type='xfs')})))
	assert any('partition numbers' in p for p in hpcp_mod.match_target_layout(rows, _layout({3: _part()})))
	assert any('sector size' in p for p in hpcp_mod.match_target_layout(rows, _layout({1: _part(fs_type='', size=1048576), 3: _part()}, sector_size=4096)))


# ---- root + loop fixtures (GPT disk images) ----

_DISK_TOOLS = ('losetup', 'sgdisk', 'mkfs.vfat', 'mkfs.ext4', 'mkswap', 'mount', 'umount', 'lsblk', 'blkid', 'e2label')
requires_root_and_disk_tools = pytest.mark.skipif(
	os.name == 'nt' or os.geteuid() != 0 or any(shutil.which(tool) is None for tool in _DISK_TOOLS),
	reason='disk patch tests need root and losetup/sgdisk/mkfs.vfat/mkfs.ext4/mkswap/mount/lsblk/blkid/e2label')


def _run(*argv):
	subprocess.run([str(arg) for arg in argv], check=True, capture_output=True)


@contextlib.contextmanager
def _attached(image, read_only=False):
	argv = ['losetup', '--partscan', '--find', '--show'] + (['--read-only'] if read_only else []) + [str(image)]
	loop = subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()
	try:
		for _ in range(50):
			if os.path.exists(loop + 'p1'):
				break
			time.sleep(0.1)
		yield loop
	finally:
		subprocess.run(['losetup', '-d', loop], capture_output=True)


@contextlib.contextmanager
def _mounted(device, fstype=None):
	mount_point = tempfile.mkdtemp()
	_run(*(['mount'] + (['-t', fstype] if fstype else []) + [device, mount_point]))
	try:
		yield mount_point
	finally:
		subprocess.run(['umount', mount_point], capture_output=True)
		os.rmdir(mount_point)


def _make_golden(path):
	"""96 MiB GPT disk: p1 ESP (vfat), p2 BIOS boot (raw bytes), p3 root (ext4, label root), p4 swap."""
	with open(str(path), 'wb') as f:
		f.truncate(96 << 20)
	_run('sgdisk', '-n1:2048:+32M', '-t1:EF00', '-c1:ESP', '-n2:0:+1M', '-t2:EF02', '-c2:bios',
		'-n3:0:+40M', '-t3:8300', '-c3:root', '-n4:0:+8M', '-t4:8200', '-c4:swap', path)
	with _attached(path) as loop:
		_run('mkfs.vfat', loop + 'p1')
		_run('mkfs.ext4', '-q', '-F', '-L', 'root', loop + 'p3')
		_run('mkswap', '-L', 'swap', loop + 'p4')
		with open(loop + 'p2', 'r+b') as f:
			f.write(b'GRUBCORE' * 128)
		with _mounted(loop + 'p1') as esp:
			_write(os.path.join(esp, 'EFI', 'BOOT', 'grubx64.efi'), b'old-efi')
		with _mounted(loop + 'p3') as root:
			_write(os.path.join(root, 'etc', 'os-release'), b'v1')
			_write(os.path.join(root, 'etc', 'keep.conf'), b'same')
			_write(os.path.join(root, 'etc', 'obsolete.conf'), b'gone soon')
			os.makedirs(os.path.join(root, 'mnt'))
	with open(str(path), 'r+b') as f:
		f.write(b'\xeb\x63OLD-BOOTCODE')


def _make_newer(old, new, label='root2', raw=True, mbr=True, gpt_name='ESP-v2', files=True, extra_partition=False):
	"""Copy old to new, then change what is asked: files on p1/p3, p2 bytes, p3 label, p1 GPT name, MBR boot code."""
	shutil.copyfile(str(old), str(new))
	if gpt_name:
		_run('sgdisk', f'-c1:{gpt_name}', new)
	if extra_partition:
		_run('sgdisk', '-n5:0:+4M', '-t5:8300', '-c5:extra', new)
	with _attached(new) as loop:
		if files:
			with _mounted(loop + 'p1') as esp:
				_write(os.path.join(esp, 'EFI', 'BOOT', 'grubx64.efi'), b'new-efi-binary')
			with _mounted(loop + 'p3') as root:
				_write(os.path.join(root, 'etc', 'os-release'), b'v2')
				_write(os.path.join(root, 'usr', 'bin', 'tool'), b'added')
				os.unlink(os.path.join(root, 'etc', 'obsolete.conf'))
		if raw:
			with open(loop + 'p2', 'r+b') as f:
				f.write(b'GRUBCOR2' * 128)
		if label:
			_run('e2label', loop + 'p3', label)
	if mbr:
		with open(str(new), 'r+b') as f:
			f.write(b'\xeb\x63NEW-BOOTCODE')


@contextlib.contextmanager
def _session(hpcp_mod):
	mount_points, loop_devices = [], []
	try:
		yield mount_points, loop_devices
	finally:
		hpcp_mod.clean_up(mount_points, loop_devices)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_read_disk_layout_of_a_gpt_image(tmp_path, hpcp_mod, reset_hpcp_globals):
	image = tmp_path / 'golden.img'
	_make_golden(image)
	with _attached(image, read_only=True) as loop:
		layout = hpcp_mod.read_disk_layout(loop)
	parts = layout['parts']
	assert sorted(parts) == [1, 2, 3, 4]
	assert layout['disk']['sector_size'] == 512 and layout['disk']['guid']
	assert (parts[1]['fs_type'], parts[1]['name'], parts[1]['type_guid'].upper()) == ('vfat', 'ESP', hpcp_mod._ESP_TYPE_GUID)
	assert parts[2]['fs_type'] == '' and parts[2]['size'] == 1 << 20
	assert (parts[3]['fs_type'], parts[3]['fs_label']) == ('ext4', 'root')
	assert (parts[4]['fs_type'], parts[4]['fs_label']) == ('swap', 'swap')
	assert parts[3]['device'] == loop + 'p3'
	assert [hpcp_mod._default_kind(parts[n]) for n in (1, 2, 3, 4)] == ['fs', 'raw', 'fs', 'swap']
	# blkid finding nothing on the BIOS boot partition is an answer, not an error.
	assert hpcp_mod.ERRORS == []


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_read_disk_layout_sees_a_label_changed_a_moment_ago(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""blkid's cache may serve a value it read under 2 s ago; the verify step must see the disk."""
	image = tmp_path / 'golden.img'
	_make_golden(image)
	with _attached(image) as loop:
		assert hpcp_mod.read_disk_layout(loop)['parts'][3]['fs_label'] == 'root'
		_run('e2label', loop + 'p3', 'fresh')
		assert hpcp_mod.read_disk_layout(loop)['parts'][3]['fs_label'] == 'fresh'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "partition_devices or disk_layout or match_target or label_changed"`
Expected: FAIL with `AttributeError` for `_partition_devices` / `compare_disk_layouts` / `read_disk_layout`.

- [ ] **Step 3: Implement**

Immediately before the line `#%% ---- Remove Extra ----`, insert the new section:

```python
#%% ---- Disk Patch ----
_ESP_TYPE_GUID = 'C12A7328-F81F-11D2-BA4B-00A0C93EC93B'

def _partition_devices(disk, sys_block_path='/sys/class/block'):
	"""{partition number: device path} for disk, using the kernel's partition number, not the name suffix."""
	result = {}
	for path in get_partitions(disk):
		name = os.path.basename(path)
		try:
			with open(os.path.join(sys_block_path, name, 'partition')) as f:
				result[int(f.read().strip())] = path
		except (OSError, ValueError):
			continue
	return result

def _fs_params_text(params):
	"""Stable text form of probe_fs_params() output for disk.tsv."""
	return ';'.join(f'{key}={(params or {})[key]}' for key in sorted(params or {}))

def _probe_identity(device):
	"""(fs_type, fs_uuid, fs_label) of device from a low-level `blkid -p` probe ('' when nothing is found).

	No blkid cache, so a label or UUID this run just changed reads back as it is on disk; exact
	keys, so UUID_SUB (btrfs) and LABEL_FATBOOT (FAT) never stand in for UUID and LABEL.
	"""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'export', device], quiet=True)
	finally:
		# blkid exits 2 when it finds nothing (BIOS boot and other raw partitions).
		del ERRORS[error_count:]
	values = {}
	for line in output:
		key, sep, value = line.strip().partition('=')
		if sep:
			# -o export escapes shell-unsafe characters (a space in a label) with a backslash.
			values[key] = re.sub(r'\\(.)', r'\1', value)
	return values.get('TYPE', ''), values.get('UUID', ''), values.get('LABEL', '')

def read_disk_layout(device):
	"""Fresh GPT + filesystem identity read of block device.

	Returns {'disk': {'guid', 'sector_size', 'size'}, 'parts': {number: {'type_guid',
	'unique_guid', 'name', 'attrs', 'size', 'fs_type', 'fs_uuid', 'fs_label', 'fs_params',
	'device'}}}. The partition caches are cleared first: loop device names are reused.
	Filesystem type, UUID and label come from _probe_identity, not the cached blkid read.
	"""
	get_partition_infos.cache_clear()
	get_partition_details.cache_clear()
	infos = get_partition_infos(device)
	devices = _partition_devices(device)
	disk, parts = {}, {}
	for key, info in infos.items():
		if 'disk_identifier' in info:
			disk = {'guid': info['disk_identifier'], 'sector_size': info['sector_size'], 'size': info['size']}
			continue
		number = int(key)
		if number not in devices:
			raise RuntimeError(f'Disk layout mismatch: the kernel shows no device for partition {number} of {device}; re-read its partition table (partprobe) first')
		fs_type, fs_uuid, fs_label = _probe_identity(devices[number])
		parts[number] = {
			'type_guid': info['partition_guid_code'], 'unique_guid': info['unique_partition_guid'],
			'name': info['partition_name'], 'attrs': info['partition_attrs'], 'size': info['size'],
			'fs_type': fs_type, 'fs_uuid': fs_uuid, 'fs_label': fs_label,
			'fs_params': _fs_params_text(info.get('fs_params')), 'device': devices[number],
		}
	return {'disk': disk, 'parts': parts}

def _default_kind(part):
	"""'swap' for swap, 'fs' when blkid found a filesystem, else 'raw' (generation may downgrade fs to raw)."""
	if part['fs_type'] == 'swap':
		return 'swap'
	return 'fs' if part['fs_type'] else 'raw'

def compare_disk_layouts(old, new, kinds=None):
	"""(problems, rows): why a patch from old to new is impossible, and the disk.tsv rows.

	Partition numbers, type GUIDs, filesystem types and the sector size must match; a raw
	area must keep its size. kinds maps partition number -> 'fs' / 'swap' / 'raw'
	(default: _default_kind of the new partition).
	"""
	problems = []
	kinds = kinds or {}
	if old['disk'].get('sector_size') != new['disk'].get('sector_size'):
		problems.append(f"sector size {old['disk'].get('sector_size')} vs {new['disk'].get('sector_size')}")
	if sorted(old['parts']) != sorted(new['parts']):
		problems.append(f"partition numbers {sorted(old['parts'])} vs {sorted(new['parts'])}")
		return problems, []
	rows = [DiskRow('disk', '-', 'guid', str(old['disk'].get('guid', '')), str(new['disk'].get('guid', ''))),
		DiskRow('disk', '-', 'sector_size', str(old['disk'].get('sector_size', '')), str(new['disk'].get('sector_size', '')))]
	for number in sorted(new['parts']):
		old_part, new_part = old['parts'][number], new['parts'][number]
		kind = kinds.get(number) or _default_kind(new_part)
		for field in ('type_guid', 'fs_type'):
			if old_part[field] != new_part[field]:
				problems.append(f'partition {number} {field} {old_part[field]!r} vs {new_part[field]!r}')
		if kind == 'raw' and old_part['size'] != new_part['size']:
			problems.append(f"partition {number} is a raw area and its size changed ({old_part['size']} vs {new_part['size']})")
		rows.append(DiskRow('part', str(number), 'kind', kind, kind))
		for field in _PART_FIELDS[1:]:
			rows.append(DiskRow('part', str(number), field, str(old_part[field]), str(new_part[field])))
	return problems, rows

def match_target_layout(disk_rows, target):
	"""Why target cannot take a patch with these disk.tsv rows (empty when it matches its base)."""
	problems = []
	parts = {}
	for row in disk_rows:
		if row.kind == 'disk' and row.field == 'sector_size' and str(target['disk'].get('sector_size')) != row.old:
			problems.append(f"sector size {target['disk'].get('sector_size')}, patch expects {row.old}")
		elif row.kind == 'part':
			parts.setdefault(int(row.number), {})[row.field] = row
	if sorted(parts) != sorted(target['parts']):
		problems.append(f"partition numbers {sorted(target['parts'])}, patch expects {sorted(parts)}")
		return problems
	for number, fields in sorted(parts.items()):
		missing = [field for field in _PART_FIELDS if field not in fields]
		if missing:
			raise RuntimeError(f"Patch format error: the disk table lacks {', '.join(missing)} for partition {number}")
		part = target['parts'][number]
		for field in ('type_guid', 'fs_type'):
			if str(part[field]) != fields[field].old:
				problems.append(f'partition {number} {field} is {part[field]!r}, patch expects {fields[field].old!r}')
		if fields['kind'].old == 'raw' and str(part['size']) != fields['size'].old:
			problems.append(f"partition {number} is a raw area of size {part['size']}, patch expects {fields['size'].old}")
	return problems

```

In `get_partition_details` (≈ line 1671), make the blkid read quiet and drop what it records — `blkid` exits 2 on a partition without a filesystem (BIOS boot, raw data), which is an answer, not a failure (dd mode reported a spurious `Task return code error` for every such partition). Replace

```python
	result = run_command_in_multicmd_with_path_check(["blkid", '-o', 'export', target_partition])
```

with

```python
	# blkid exits 2 on a partition without a filesystem (BIOS boot, raw data): an answer, not an error.
	error_count = len(ERRORS)
	result = run_command_in_multicmd_with_path_check(["blkid", '-o', 'export', target_partition], quiet=True)
	del ERRORS[error_count:]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py tests/test_dd_errors.py -q`
Expected: all pass (the root tests run as root).

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "feat(dd-diff): read, compare and match GPT disk layouts; quiet blkid on raw partitions"
```

---

### Task 3: Raw areas

**Files:**
- Modify: `hpcp.py` (add `import hashlib` to the import block after `import glob`; Disk Patch section)
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: `_inside_root(root, path)` (phase 1).
- Produces:
  - `_MBR_BOOTCODE_SIZE = 440`
  - `_hash_area(path, offset=0, length=None) -> str` (sha256 hex; `OSError` when shorter)
  - `_copy_area(source, dest, offset, length) -> str` (sha256 of the copied bytes)
  - `_write_area(target, payload, offset, length) -> None` (fsync'd)
  - `_raw_payload(volume_roots, raw_file, new_sha, size) -> str | None`
  - `_plan_raw(old_sha, new_sha, found_sha) -> 'skip' | 'apply' | 'conflict'`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
import hashlib as _hashlib


def _sha(data):
	return _hashlib.sha256(data).hexdigest()


def test_hash_area_ranges_and_short_files(tmp_path, hpcp_mod):
	path = tmp_path / 'disk'
	path.write_bytes(b'0123456789' * 1000)
	assert hpcp_mod._hash_area(str(path)) == _sha(b'0123456789' * 1000)
	assert hpcp_mod._hash_area(str(path), 10, 5) == _sha(b'01234')
	with pytest.raises(OSError):
		hpcp_mod._hash_area(str(path), 9990, 20)


def test_copy_and_write_area_keep_bytes_outside_the_range(tmp_path, hpcp_mod):
	source = tmp_path / 'source'
	source.write_bytes(b'N' * 440 + b'SIGNATURE' + b'\x55\xaa')
	payload = tmp_path / 'payload'
	assert hpcp_mod._copy_area(str(source), str(payload), 0, 440) == _sha(b'N' * 440)
	target = tmp_path / 'target'
	target.write_bytes(b'O' * 440 + b'KEEPTHIS!' + b'\x55\xaa')
	hpcp_mod._write_area(str(target), str(payload), 0, 440)
	assert target.read_bytes() == b'N' * 440 + b'KEEPTHIS!' + b'\x55\xaa'
	with pytest.raises(OSError):
		hpcp_mod._write_area(str(target), str(payload), 0, 441)


def test_raw_payload_rejects_damaged_or_escaping_files(tmp_path, hpcp_mod):
	"""Review Focus 4: a payload whose bytes do not hash to the recorded value is never used."""
	data = b'GRUBCOR2' * 128
	v1, v2 = tmp_path / 'v1', tmp_path / 'v2'
	_write(v1 / 'raw' / 'part2.img', b'GRUBCOR2' * 127 + b'BITROT!!')
	_write(v2 / 'raw' / 'part2.img', data)
	assert hpcp_mod._raw_payload([str(v1), str(v2)], 'raw/part2.img', _sha(data), len(data)) == str(v2 / 'raw' / 'part2.img')
	assert hpcp_mod._raw_payload([str(v1)], 'raw/part2.img', _sha(data), len(data)) is None
	outside = tmp_path / 'outside'
	_write(outside / 'part2.img', data)
	v3 = tmp_path / 'v3'
	v3.mkdir()
	os.symlink(str(outside), str(v3 / 'raw'))
	assert hpcp_mod._raw_payload([str(v3)], 'raw/part2.img', _sha(data), len(data)) is None


def test_plan_raw_actions(hpcp_mod):
	"""Review Focus 5: an area already holding the new content is skipped on a re-run."""
	assert hpcp_mod._plan_raw(_A, _B, _B) == 'skip'
	assert hpcp_mod._plan_raw(_A, _B, _A) == 'apply'
	assert hpcp_mod._plan_raw(_A, _B, _C) == 'conflict'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "area or raw_payload or plan_raw"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute '_hash_area'`.

- [ ] **Step 3: Implement**

Add `import hashlib` to the import block (after `import glob`).

Append to the Disk Patch section:

```python
_MBR_BOOTCODE_SIZE = 440
_AREA_CHUNK = 4 << 20

def _hash_area(path, offset=0, length=None):
	"""sha256 hex digest of length bytes of path from offset (to the end when length is None).

	Raises OSError when path ends before offset + length.
	"""
	digest = hashlib.sha256()
	remaining = length
	with open(path, 'rb') as f:
		f.seek(offset)
		while remaining is None or remaining > 0:
			chunk = f.read(_AREA_CHUNK if remaining is None else min(_AREA_CHUNK, remaining))
			if not chunk:
				break
			digest.update(chunk)
			if remaining is not None:
				remaining -= len(chunk)
	if remaining:
		raise OSError(errno.EIO, f'{path} ends before byte {offset + length}')
	return digest.hexdigest()

def _copy_area(source, dest, offset, length):
	"""Copy length bytes of source from offset into the new file dest (fsync'd); returns their sha256."""
	digest = hashlib.sha256()
	remaining = length
	with open(source, 'rb') as src, open(dest, 'wb') as out:
		src.seek(offset)
		while remaining > 0:
			chunk = src.read(min(_AREA_CHUNK, remaining))
			if not chunk:
				raise OSError(errno.EIO, f'{source} ends before byte {offset + length}')
			out.write(chunk)
			digest.update(chunk)
			remaining -= len(chunk)
		out.flush()
		os.fsync(out.fileno())
	return digest.hexdigest()

def _write_area(target, payload, offset, length):
	"""Write the first length bytes of payload into target at offset, then fsync.

	Bytes of target outside [offset, offset + length) are left as they are (the MBR keeps its
	disk signature, partition entries and boot signature).
	"""
	remaining = length
	with open(payload, 'rb') as src, open(target, 'r+b') as out:
		out.seek(offset)
		while remaining > 0:
			chunk = src.read(min(_AREA_CHUNK, remaining))
			if not chunk:
				raise OSError(errno.EIO, f'{payload} is shorter than {length} bytes')
			out.write(chunk)
			remaining -= len(chunk)
		out.flush()
		os.fsync(out.fileno())

def _raw_payload(volume_roots, raw_file, new_sha, size):
	"""Path of raw_file on a volume when it is a regular file inside that volume with the recorded
	size and sha256; None when no volume holds an intact copy."""
	for volume_root in volume_roots:
		path = os.path.join(volume_root, raw_file)
		if (_inside_root(volume_root, path) and os.path.isfile(path) and not os.path.islink(path)
				and os.path.getsize(path) == size and _hash_area(path) == new_sha):
			return path
	return None

def _plan_raw(old_sha, new_sha, found_sha):
	"""'skip' when the area already holds the new content, 'apply' when it holds the old, else 'conflict'."""
	if found_sha == new_sha:
		return 'skip'
	if found_sha == old_sha:
		return 'apply'
	return 'conflict'

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "feat(dd-diff): hash, copy and write raw disk areas"
```

---

### Task 4: Live-disk detection

**Files:**
- Modify: `hpcp.py` (Disk Patch section)
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Produces:
  - `_unescape_mount(text) -> str` (kernel octal escapes such as `\040`)
  - `_mountinfo_entries(mountinfo_path='/proc/self/mountinfo') -> list[dict]` with keys `devnum` (`(major, minor)`), `root`, `mount_point`, `options` (list), `fstype`, `source`
  - `_device_number(device) -> (major, minor)`
  - `_entry_matches(entry, devnum, device) -> bool` (by device number, or by source path for btrfs-style anonymous device numbers)
  - `_live_mount(devnum, entries, device='') -> (mount_point | None, read_only: bool)` — only mounts whose filesystem root is `/`
  - `_swap_devices(swaps_path='/proc/swaps') -> set[str]` (realpaths)
  - `_partition_in_use(device, devnum, entries, swaps, sys_block_path='/sys/class/block') -> str` (`'mounted'`, `'active swap'`, `'held by another device'` or `''`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
_MOUNTINFO = '\n'.join([
	'36 25 8:3 / / rw,relatime shared:1 - ext4 /dev/sda3 rw',
	'37 36 8:1 / /boot/efi rw,relatime shared:2 - vfat /dev/sda1 rw',
	'38 36 8:3 /srv /mnt/bind rw,relatime shared:1 - ext4 /dev/sda3 rw',
	'39 36 8:5 / /mnt/with\\040space ro,relatime - ext4 /dev/sda5 ro',
	'40 36 0:50 / /btr rw,relatime - btrfs /dev/sda6 rw',
]) + '\n'


def test_mountinfo_parsing_and_live_mounts(tmp_path, hpcp_mod):
	path = tmp_path / 'mountinfo'
	path.write_text(_MOUNTINFO)
	entries = hpcp_mod._mountinfo_entries(str(path))
	assert len(entries) == 5
	assert entries[3]['mount_point'] == '/mnt/with space' and entries[3]['devnum'] == (8, 5)
	assert hpcp_mod._live_mount((8, 3), entries) == ('/', False)
	assert hpcp_mod._live_mount((8, 5), entries) == ('/mnt/with space', True)
	assert hpcp_mod._live_mount((8, 9), entries) == (None, False)
	# btrfs reports an anonymous device number; the source path still identifies it
	assert hpcp_mod._live_mount((8, 6), entries, '/dev/sda6') == ('/btr', False)
	assert hpcp_mod._mountinfo_entries(str(tmp_path / 'missing')) == []


def test_partition_in_use_reasons(tmp_path, hpcp_mod):
	mountinfo = tmp_path / 'mountinfo'
	mountinfo.write_text(_MOUNTINFO)
	swaps = tmp_path / 'swaps'
	swaps.write_text('Filename\tType\tSize\tUsed\tPriority\n/dev/sda4 partition 8388604 0 -2\n')
	sys_block = tmp_path / 'sys'
	(sys_block / 'sda7' / 'holders' / 'dm-0').mkdir(parents=True)
	(sys_block / 'sda8' / 'holders').mkdir(parents=True)
	entries = hpcp_mod._mountinfo_entries(str(mountinfo))
	swap_set = hpcp_mod._swap_devices(str(swaps))
	assert swap_set == {os.path.realpath('/dev/sda4')}
	check = lambda device, devnum: hpcp_mod._partition_in_use(device, devnum, entries, swap_set, str(sys_block))
	assert check('/dev/sda1', (8, 1)) == 'mounted'
	assert check('/dev/sda4', (8, 4)) == 'active swap'
	assert check('/dev/sda7', (8, 7)) == 'held by another device'
	assert check('/dev/sda8', (8, 8)) == ''
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "mountinfo or in_use"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute '_mountinfo_entries'`.

- [ ] **Step 3: Implement**

Append to the Disk Patch section:

```python
def _unescape_mount(text):
	"""Undo the octal escapes (\\040 for a space, …) the kernel uses in mountinfo and /proc/swaps."""
	return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), text)

def _mountinfo_entries(mountinfo_path='/proc/self/mountinfo'):
	"""Parsed mountinfo lines (an empty list when the file cannot be read)."""
	entries = []
	try:
		with open(mountinfo_path, encoding='utf-8', errors='surrogateescape') as f:
			for line in f:
				fields = line.split()
				if '-' not in fields or len(fields) < 7:
					continue
				separator = fields.index('-')
				major, _, minor = fields[2].partition(':')
				entries.append({
					'devnum': (int(major), int(minor)), 'root': _unescape_mount(fields[3]),
					'mount_point': _unescape_mount(fields[4]), 'options': fields[5].split(','),
					'fstype': fields[separator + 1] if len(fields) > separator + 1 else '',
					'source': _unescape_mount(fields[separator + 2]) if len(fields) > separator + 2 else '',
				})
	except (OSError, ValueError):
		pass
	return entries

def _device_number(device):
	"""(major, minor) of a block device node."""
	st = os.stat(device)
	return (os.major(st.st_rdev), os.minor(st.st_rdev))

def _entry_matches(entry, devnum, device):
	"""A mountinfo entry belongs to device: same device number, or (btrfs and other filesystems with
	anonymous device numbers) the same source device path."""
	if entry['devnum'] == devnum:
		return True
	return bool(device) and entry['source'].startswith('/') and os.path.realpath(entry['source']) == os.path.realpath(device)

def _live_mount(devnum, entries, device=''):
	"""(mount point, read_only) of the mount that shows device's whole filesystem (root '/'), or (None, False)."""
	for entry in entries:
		if entry['root'] == '/' and _entry_matches(entry, devnum, device):
			return entry['mount_point'], 'ro' in entry['options']
	return None, False

def _swap_devices(swaps_path='/proc/swaps'):
	"""Real paths of the active swap areas."""
	result = set()
	try:
		with open(swaps_path, encoding='utf-8', errors='surrogateescape') as f:
			next(f, None)
			for line in f:
				fields = line.split()
				if fields:
					result.add(os.path.realpath(_unescape_mount(fields[0])))
	except OSError:
		pass
	return result

def _partition_in_use(device, devnum, entries, swaps, sys_block_path='/sys/class/block'):
	"""Why device is in use ('' when it is not): mounted, active swap, or held by LVM / md / dm."""
	if any(_entry_matches(entry, devnum, device) for entry in entries):
		return 'mounted'
	if os.path.realpath(device) in swaps:
		return 'active swap'
	try:
		if os.listdir(os.path.join(sys_block_path, os.path.basename(os.path.realpath(device)), 'holders')):
			return 'held by another device'
	except OSError:
		pass
	return ''

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "feat(dd-diff): detect mounted, swapped and held partitions"
```

---

### Task 5: Identity commands and planning, error codes

**Files:**
- Modify: `hpcp.py` (`ERROR_TO_RETURNCODE_TABLE`, `_binCalled` ≈ line 422, `write_partition_info` ≈ 1740–1941, Disk Patch section)
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: `_FS_FIX_COMMANDS` (existing), `DiskRow` (Task 1).
- Produces:
  - `_FAT_TYPES`, `_IDENTITY_FIELDS = ('unique_guid', 'name', 'attrs', 'fs_uuid', 'fs_label')`
  - `_fs_identity_commands(fs_type, device, uuid=None, label=None) -> list[list[str]] | None` (`None` = unchanged; `label=''` clears)
  - `_gpt_identity_commands(disk, number=None, unique_guid=None, name=None, attrs=None, disk_guid=None) -> list[list[str]]`
  - `IdentityChange = namedtuple('IdentityChange', 'number field current new')` (`number` is `None` for the disk GUID)
  - `_plan_identity(disk_rows, layout, in_use) -> (changes, refusals: list[str], warnings: list[str])`
  - `_identity_command_list(changes, layout, disk_device) -> (commands, unsupported)` where `unsupported` is a list of `(number, fields_tuple, text)`; filesystem commands come before GPT commands
  - Error codes 248–251

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
def test_fs_identity_commands_match_dd_mode_uuid_commands(hpcp_mod):
	"""dd mode's delayed UUID commands now come from this table; the argv must not change."""
	dev, u = '/dev/x1', '1234-ABCD'
	cmd = hpcp_mod._fs_identity_commands
	assert cmd('ext4', dev, uuid=u) == [['e2fsck', '-f', '-y', dev], ['tune2fs', '-U', u, dev]]
	assert cmd('xfs', dev, uuid=u) == [['xfs_admin', '-U', u, dev]]
	assert cmd('exfat', dev, uuid=u) == [['exfatlabel', '-i', dev, u]]
	assert cmd('udf', dev, uuid=u) == [['udflabel', '--uuid', u, dev]]
	assert cmd('jfs', dev, uuid=u) == [['jfs_tune', '-U', u, dev]]
	assert cmd('reiserfs', dev, uuid=u) == [['reiserfstune', '-u', u, dev]]
	assert cmd('swap', dev, uuid=u) == [['swaplabel', '-U', u, dev]]


def test_fs_identity_commands_labels_and_unsupported(hpcp_mod):
	dev = '/dev/x1'
	cmd = hpcp_mod._fs_identity_commands
	assert cmd('ext4', dev, label='root2') == [['e2label', dev, 'root2']]
	assert cmd('xfs', dev, label='') == [['xfs_admin', '-L', '--', dev]]
	assert cmd('vfat', dev, uuid='ab12-cd34', label='ESP') == [['fatlabel', '-i', dev, 'AB12CD34'], ['fatlabel', dev, 'ESP']]
	assert cmd('swap', dev, label='swap2') == [['swaplabel', '-L', 'swap2', dev]]
	assert cmd('ntfs', dev, label='DATA') == [['ntfslabel', dev, 'DATA']]
	assert cmd('ntfs', dev, uuid='X') is None
	assert cmd('minix', dev, label='x') is None
	assert cmd('ext4', dev) == []


def test_gpt_identity_commands(hpcp_mod):
	assert hpcp_mod._gpt_identity_commands('/dev/sda', 2, unique_guid='G', name='', attrs='8000000000000000') == [
		['sgdisk', '--partition-guid=2:G', '/dev/sda'],
		['sgdisk', '--change-name=2:', '/dev/sda'],
		['sgdisk', '--attributes=2:=:8000000000000000', '/dev/sda']]
	assert hpcp_mod._gpt_identity_commands('/dev/sda', disk_guid='D') == [['sgdisk', '--disk-guid=D', '/dev/sda']]


def _identity_rows(hpcp_mod, **changes):
	"""disk.tsv rows for partitions 1 (fs, vfat), 2 (raw) and 3 (fs, ext4) with the given (number, field) -> (old, new)."""
	R = hpcp_mod.DiskRow
	rows = [R('disk', '-', 'guid', 'D1', changes.get('disk_guid', 'D1')), R('disk', '-', 'sector_size', '512', '512')]
	for number, kind, fs_type in (('1', 'fs', 'vfat'), ('2', 'raw', ''), ('3', 'fs', 'ext4')):
		rows.append(R('part', number, 'kind', kind, kind))
		rows.append(R('part', number, 'fs_type', fs_type, fs_type))
		for field in ('unique_guid', 'name', 'attrs', 'fs_uuid', 'fs_label'):
			old, new = changes.get(f'p{number}_{field}', ('v', 'v'))
			rows.append(R('part', number, field, old, new))
	return rows


def _target(**current):
	parts = {}
	for number, fs_type in ((1, 'vfat'), (2, ''), (3, 'ext4')):
		parts[number] = {'unique_guid': 'v', 'name': 'v', 'attrs': 'v', 'fs_uuid': 'v', 'fs_label': 'v', 'fs_type': fs_type, 'device': f'/dev/x{number}'}
		for key, value in current.items():
			if key.startswith(f'p{number}_'):
				parts[number][key[3:]] = value
	return {'disk': {'guid': current.get('disk_guid', 'D1')}, 'parts': parts}


def test_plan_identity_applies_changes_and_refuses_in_use_partitions(hpcp_mod):
	rows = _identity_rows(hpcp_mod, p3_fs_label=('v', 'root2'), p1_name=('v', 'ESP2'), disk_guid='D2')
	changes, refusals, warnings = hpcp_mod._plan_identity(rows, _target(), {1: '', 2: '', 3: ''})
	assert {(c.number, c.field, c.new) for c in changes} == {(3, 'fs_label', 'root2'), (1, 'name', 'ESP2'), (None, 'guid', 'D2')}
	assert refusals == [] and warnings == []
	_, refusals, _ = hpcp_mod._plan_identity(rows, _target(), {1: '', 2: '', 3: 'mounted'})
	assert refusals == ['partition 3 is mounted and its fs_label changes']


def test_plan_identity_warns_when_target_differs_from_base(hpcp_mod):
	"""Review Focus 3: a node-specific value is a warning, and the new value is still set."""
	rows = _identity_rows(hpcp_mod, p3_fs_uuid=('v', 'NEW-UUID'))
	changes, refusals, warnings = hpcp_mod._plan_identity(rows, _target(p3_fs_uuid='NODE-UUID'), {})
	assert [(c.number, c.field, c.current, c.new) for c in changes] == [(3, 'fs_uuid', 'NODE-UUID', 'NEW-UUID')]
	assert refusals == [] and len(warnings) == 1 and 'NODE-UUID' in warnings[0]


def test_plan_identity_skips_values_already_new(hpcp_mod):
	"""Review Focus 5: a re-run after an interruption changes nothing that is already set."""
	rows = _identity_rows(hpcp_mod, p3_fs_label=('v', 'root2'), disk_guid='D2')
	changes, _, warnings = hpcp_mod._plan_identity(rows, _target(p3_fs_label='root2', disk_guid='D2'), {3: 'mounted'})
	assert changes == [] and warnings == []


def test_plan_identity_ignores_filesystem_fields_of_raw_areas(hpcp_mod):
	rows = _identity_rows(hpcp_mod, p2_fs_uuid=('v', 'LVM-PV'), p2_name=('v', 'bios2'))
	changes, _, _ = hpcp_mod._plan_identity(rows, _target(), {})
	assert [(c.number, c.field) for c in changes] == [(2, 'name')]


def test_identity_command_list_groups_and_flags_unsupported(hpcp_mod):
	IC = hpcp_mod.IdentityChange
	layout = {'disk': {}, 'parts': {1: {'fs_type': 'ntfs', 'device': '/dev/x1'}, 3: {'fs_type': 'ext4', 'device': '/dev/x3'}}}
	changes = [IC(None, 'guid', 'D1', 'D2'), IC(3, 'fs_label', 'root', 'root2'), IC(3, 'name', 'root', 'root-v2'),
		IC(1, 'fs_uuid', 'A', 'B')]
	commands, unsupported = hpcp_mod._identity_command_list(changes, layout, '/dev/x')
	assert commands == [['e2label', '/dev/x3', 'root2'], ['sgdisk', '--disk-guid=D2', '/dev/x'], ['sgdisk', '--change-name=3:root-v2', '/dev/x']]
	assert [(number, fields) for number, fields, _ in unsupported] == [(1, ('fs_uuid',))]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "identity"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute '_fs_identity_commands'`.

- [ ] **Step 3: Implement**

In `ERROR_TO_RETURNCODE_TABLE`, after `'Entry state error': 247,` add:

```python
	'Disk layout mismatch': 248,
	'Raw area error': 249,
	'Identity change error': 250,
	'Disk in use': 251,
```

In `_binCalled`, change the last line `'reiserfstune', 'debugreiserfs', 'swaplabel'}` to:

```python
			  'reiserfstune', 'debugreiserfs', 'swaplabel', 'e2label', 'fatlabel', 'ntfslabel', 'btrfstune', 'f2fslabel'}
```

Append to the Disk Patch section:

```python
_FAT_TYPES = ('fat32', 'fat16', 'fat12', 'fat', 'vfat', 'msdos')
_IDENTITY_FIELDS = ('unique_guid', 'name', 'attrs', 'fs_uuid', 'fs_label')
IdentityChange = namedtuple('IdentityChange', 'number field current new')

def _fs_identity_commands(fs_type, device, uuid=None, label=None):
	"""argv lists that set a filesystem's UUID and / or label on an unmounted device.

	uuid / label None leave the value alone; label '' clears it. Returns None when fs_type
	cannot have a requested value set without recreating it. dd mode's delayed UUID
	commands (write_partition_info) come from this table too.
	"""
	commands = []
	if uuid:
		if fs_type in ('ext2', 'ext3', 'ext4'):
			# tune2fs -U on a metadata_csum filesystem refuses unless it was freshly checked.
			commands += [_FS_FIX_COMMANDS[fs_type] + [device], ['tune2fs', '-U', uuid, device]]
		elif fs_type == 'xfs':
			commands.append(['xfs_admin', '-U', uuid, device])
		elif fs_type == 'btrfs':
			commands.append(['btrfstune', '-f', '-U', uuid, device])
		elif fs_type in _FAT_TYPES:
			commands.append(['fatlabel', '-i', device, uuid.replace('-', '').upper()])
		elif fs_type == 'exfat':
			commands.append(['exfatlabel', '-i', device, uuid])
		elif fs_type == 'udf':
			commands.append(['udflabel', '--uuid', uuid, device])
		elif fs_type == 'jfs':
			commands.append(['jfs_tune', '-U', uuid, device])
		elif fs_type == 'reiserfs':
			commands.append(['reiserfstune', '-u', uuid, device])
		elif fs_type == 'swap':
			commands.append(['swaplabel', '-U', uuid, device])
		else:
			return None
	if label is not None:
		if fs_type in ('ext2', 'ext3', 'ext4'):
			commands.append(['e2label', device, label])
		elif fs_type == 'xfs':
			commands.append(['xfs_admin', '-L', label or '--', device])
		elif fs_type == 'btrfs':
			commands.append(['btrfs', 'filesystem', 'label', device, label])
		elif fs_type in _FAT_TYPES:
			commands.append(['fatlabel', device, label])
		elif fs_type == 'exfat':
			commands.append(['exfatlabel', device, label])
		elif fs_type == 'ntfs':
			commands.append(['ntfslabel', device, label])
		elif fs_type == 'f2fs':
			commands.append(['f2fslabel', device, label])
		elif fs_type == 'udf':
			commands.append(['udflabel', device, label])
		elif fs_type == 'jfs':
			commands.append(['jfs_tune', '-L', label, device])
		elif fs_type == 'reiserfs':
			commands.append(['reiserfstune', '-l', label, device])
		elif fs_type == 'swap':
			commands.append(['swaplabel', '-L', label, device])
		else:
			return None
	return commands

def _gpt_identity_commands(disk, number=None, unique_guid=None, name=None, attrs=None, disk_guid=None):
	"""sgdisk argv lists for a partition's unique GUID / name / attributes and the disk GUID."""
	commands = []
	if disk_guid:
		commands.append(['sgdisk', f'--disk-guid={disk_guid}', disk])
	if number is not None:
		if unique_guid:
			commands.append(['sgdisk', f'--partition-guid={number}:{unique_guid}', disk])
		if name is not None:
			commands.append(['sgdisk', f'--change-name={number}:{name}', disk])
		if attrs:
			commands.append(['sgdisk', f'--attributes={number}:=:{attrs}', disk])
	return commands

def _plan_identity(disk_rows, layout, in_use):
	"""Identity values the patch changes on this target: (changes, refusals, warnings).

	Values already new are skipped (re-runs). A target value that differs from the patch
	base is a warning, not a conflict: node identities may legitimately differ. A change on
	a partition that is in use is a refusal. The disk GUID only rewrites the GPT headers and
	is changed even on a live disk. Raw areas carry their own filesystem identity.
	"""
	kinds = {row.number: row.new for row in disk_rows if row.kind == 'part' and row.field == 'kind'}
	changes, refusals, warnings = [], [], []
	for row in disk_rows:
		if row.kind == 'disk':
			if row.field != 'guid':
				continue
			number, current = None, str(layout['disk'].get('guid', ''))
		else:
			if row.field not in _IDENTITY_FIELDS:
				continue
			if kinds.get(row.number) == 'raw' and row.field in ('fs_uuid', 'fs_label'):
				continue
			number = int(row.number)
			current = str(layout['parts'][number][row.field])
		if row.old == row.new or current == row.new:
			continue
		where = 'disk' if number is None else f'partition {number}'
		if current != row.old:
			warnings.append(f'{where} {row.field} is {current!r}, the patch base had {row.old!r}; setting {row.new!r}')
		if number is not None and in_use.get(number):
			refusals.append(f'partition {number} is {in_use[number]} and its {row.field} changes')
		changes.append(IdentityChange(number, row.field, current, row.new))
	return changes, refusals, warnings

def _identity_command_list(changes, layout, disk_device):
	"""(commands, unsupported) for planned identity changes.

	Filesystem UUID / label commands come first (per partition, UUID before label), then GPT
	and disk GUID commands. unsupported lists (number, fields, text) that cannot be set.
	"""
	fs_fields, gpt_commands = {}, []
	for change in changes:
		if change.number is None:
			gpt_commands += _gpt_identity_commands(disk_device, disk_guid=change.new)
		elif change.field in ('fs_uuid', 'fs_label'):
			fs_fields.setdefault(change.number, {})[change.field] = change.new
		else:
			gpt_commands += _gpt_identity_commands(disk_device, number=change.number, **{change.field: change.new})
	commands, unsupported = [], []
	for number, fields in sorted(fs_fields.items()):
		part = layout['parts'][number]
		fs_commands = _fs_identity_commands(part['fs_type'], part['device'], uuid=fields.get('fs_uuid'), label=fields.get('fs_label'))
		if fs_commands is None:
			names = tuple(sorted(fields))
			unsupported.append((number, names, f"partition {number} ({part['fs_type'] or 'unknown'}) cannot have its {' / '.join(names)} set"))
		else:
			commands += fs_commands
	return commands + gpt_commands, unsupported

```

In `write_partition_info`, replace each delayed UUID command with the shared table (the argv stays identical):

- ext branch — replace the two lines
  ```python
  					delayed_commands.append(_FS_FIX_COMMANDS[fs_type] + [target_partition])
  					delayed_commands.append(['tune2fs', '-U', fs_uuid, target_partition])
  ```
  with
  ```python
  					delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))
  ```
- xfs branch: `delayed_commands.append(['xfs_admin', '-U', fs_uuid, target_partition])` → `delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))`
- exfat branch: `delayed_commands.append(['exfatlabel', '-i', target_partition, fs_uuid])` → same replacement
- udf branch: `delayed_commands.append(['udflabel', '--uuid', fs_uuid, target_partition])` → same replacement
- jfs branch: `delayed_commands.append(['jfs_tune', '-U', fs_uuid, target_partition])` → same replacement
- reiserfs branch: `delayed_commands.append(['reiserfstune', '-u', fs_uuid, target_partition])` → same replacement
- swap branch: `delayed_commands.append(['swaplabel', '-U', fs_uuid, target_partition])` → same replacement

Leave the surrounding comments in place.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py tests/test_dd_errors.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "feat(dd-diff): plan and build filesystem and GPT identity changes"
```

---

### Task 6: Section-based apply (refactor of phase-1 apply)

**Files:**
- Modify: `hpcp.py` (add `import contextlib` after `import concurrent.futures`; `load_patch_volumes`, `_resolve_conflicts`, `apply_patch_tree` in the Diff Image section)
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: phase-1 `check_patch_target`, `_rows_escaping_target`, `_payload_source`, `_apply_rows`, `_verify_rows`, `read_changes`, `_validate_patch_rows`.
- Produces:
  - `_load_patch(volume_roots, volume_names=None) -> (meta, sections, ordered_roots)` with `sections = [(name, payload_dir, rows)]`
  - `load_patch_volumes(...)` unchanged signature/return (single-filesystem only)
  - `_section_volume_roots(volume_roots, payload_dir) -> list[str]`
  - `_conflict_lines(checks, prefix='') -> list[(detail, short)]`, `_resolve_conflict_lines(lines, force)`, `_resolve_conflicts(conflicts, force)` (unchanged behaviour)
  - `SectionPlan = namedtuple('SectionPlan', 'name volume_roots target_root rows checks')`
  - `_plan_section(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers) -> SectionPlan`
  - `_resolve_section_conflicts(plans, force, extra_lines=())`
  - `_preflight_section(plan)` (raises before anything changes)
  - `_apply_section(plan, max_workers, verbose, full_hash, compare_hash) -> summary dict`
  - `_patch_hash_settings(meta)` context manager yielding `(compare_hash, full_hash)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
class _FakeStdin:
	def __init__(self, tty):
		self._tty = tty

	def isatty(self):
		return self._tty

	def close(self):
		pass


def _two_sections(hpcp_mod, tmp_path):
	"""Two partition-like trees with one modified file each; targets have a local edit on that file."""
	hpcp_mod.HASH_SIZE = 65536
	plans_input = []
	for name, filename in (('part1', 'efi.bin'), ('part3', 'os-release')):
		new, base, target = tmp_path / f'{name}-new', tmp_path / f'{name}-base', tmp_path / f'{name}-target'
		_write(new / filename, f'{name} new content')
		_write(base / filename, f'{name} old')
		shutil.copytree(str(base), str(target), symlinks=True)
		_write(target / filename, f'{name} local edit')
		new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
		rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1)
		volume = tmp_path / 'vol'
		shutil.copytree(str(new), str(volume / name))
		plans_input.append((name, rows, [str(volume / name)], str(target), filename))
	return plans_input


def test_sections_share_one_conflict_prompt(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	sections = _two_sections(hpcp_mod, tmp_path)
	plans = [hpcp_mod._plan_section(name, rows, roots, target, True, False, 1) for name, rows, roots, target, _ in sections]
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(False))
	with pytest.raises(RuntimeError, match='Patch conflict: 2 conflicting path'):
		hpcp_mod._resolve_section_conflicts(plans, False)
	out = capsys.readouterr().out
	assert "[part1] M 'efi.bin'" in out and "[part3] M 'os-release'" in out
	calls = []
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(True))
	monkeypatch.setattr(hpcp_mod.multiCMD, 'input_with_timeout_and_countdown', lambda timeout, prompt='': calls.append(prompt) or 'y')
	hpcp_mod._resolve_section_conflicts(plans, False, extra_lines=[('raw part2: content differs', 'raw part2: content differs')])
	assert len(calls) == 1
	assert 'Warning: forcing raw part2' in capsys.readouterr().err
	for plan in plans:
		hpcp_mod._preflight_section(plan)
	summaries = [hpcp_mod._apply_section(plan, 1, False, False, True) for plan in plans]
	assert [s['forced'] for s in summaries] == [1, 1] and all(s['failed'] == 0 for s in summaries)
	for name, _, _, target, filename in sections:
		assert open(os.path.join(target, filename)).read() == f'{name} new content'


def test_load_patch_returns_sections_and_single_filesystem_loader_refuses_disk_patches(tmp_path, hpcp_mod, reset_hpcp_globals):
	R = hpcp_mod.ChangeRow
	volume = tmp_path / 'vol'
	diff_dir = volume / '.hpcp_diff'
	diff_dir.mkdir(parents=True)
	rows = [R('A', 'a.txt', '-', 'f:1:1:100644:0:0:-:-')]
	hpcp_mod.write_changes(str(diff_dir / 'part1.changes.tsv'), rows)
	hpcp_mod.write_disk_tsv(str(diff_dir / 'disk.tsv'), [])
	meta = dict(_disk_meta(), patch_id='p1', sections=[('part1', 'part1', 'part1.changes.tsv')], raws=[])
	hpcp_mod.write_manifest(str(diff_dir / 'manifest'), meta)
	got_meta, sections, roots = hpcp_mod._load_patch([str(volume)])
	assert got_meta['layout'] == 'disk' and sections == [('part1', 'part1', rows)] and roots == [str(volume)]
	with pytest.raises(RuntimeError, match='disk patches'):
		hpcp_mod.load_patch_volumes([str(volume)])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "sections or load_patch"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute '_plan_section'`.

- [ ] **Step 3: Implement**

Add `import contextlib` after `import concurrent.futures`.

Replace `load_patch_volumes` with `_load_patch` plus a wrapper (the body up to `ordered = …` is unchanged):

```python
def _load_patch(volume_roots, volume_names=None):
	"""Read and cross-check every volume's manifest.

	Returns (meta of volume 1, [(section name, payload dir, rows)], volume roots ordered by
	volume number). volume_names (image paths, parallel to volume_roots) only improve error messages.
	"""
	names = list(volume_names) if volume_names else [str(root) for root in volume_roots]
	metas = []
	for volume_root, name in zip(volume_roots, names):
		manifest = os.path.join(volume_root, DIFF_DIR_NAME, 'manifest')
		if not os.path.isfile(manifest):
			raise RuntimeError(f'Incomplete patch: {name} has no {DIFF_DIR_NAME}/manifest')
		metas.append(read_manifest(manifest))
	for name, meta in zip(names, metas):
		if not meta.get('patch_id'):
			raise RuntimeError(f'Incomplete patch: {name} has no patch_id')
	patch_ids = sorted({meta['patch_id'] for meta in metas})
	if len(patch_ids) != 1:
		raise RuntimeError(f'Incomplete patch: volumes belong to different patches ({", ".join(patch_ids)})')
	by_number = {}
	total = None
	for volume_root, name, meta in zip(volume_roots, names, metas):
		number, _, count = str(meta.get('volume', '')).partition('/')
		try:
			number, count = int(number), int(count)
		except ValueError:
			raise RuntimeError(f'Patch format error: {name} has a bad volume field {meta.get("volume")!r}')
		if total is None:
			total = count
		elif count != total:
			raise RuntimeError(f'Incomplete patch: volumes disagree on the volume count ({total} vs {count})')
		by_number[number] = (volume_root, name, meta)
	missing = [number for number in range(1, total + 1) if number not in by_number]
	if missing:
		if 1 in by_number:
			expected = ', '.join(_split_dest_image_name(by_number[1][1], number - 1) for number in missing)
		else:
			expected = ', '.join(f'volume {number}' for number in missing)
		raise RuntimeError(f'Incomplete patch: missing volume(s) {expected} of {total}')
	ordered = [by_number[number] for number in range(1, total + 1)]
	meta = ordered[0][2]
	sections = []
	for name, payload_dir, changes_file in meta['sections']:
		rows = read_changes(os.path.join(ordered[0][0], DIFF_DIR_NAME, changes_file))
		_validate_patch_rows(rows)
		sections.append((name, payload_dir, rows))
	return meta, sections, [entry[0] for entry in ordered]

def load_patch_volumes(volume_roots, volume_names=None):
	"""Single-filesystem patch: (meta, rows, volume roots ordered by volume number)."""
	meta, sections, ordered = _load_patch(volume_roots, volume_names)
	if meta.get('layout') == 'disk' or meta['sections'] != [('root', '.', 'changes.tsv')]:
		raise RuntimeError(f"Patch format error: unsupported sections {meta['sections']}; disk patches apply to a whole disk (-siff PATCH -d DISK)")
	return meta, sections[0][2], ordered

def _section_volume_roots(volume_roots, payload_dir):
	"""Where a section's payload lives on each volume."""
	return list(volume_roots) if payload_dir == '.' else [os.path.join(volume_root, payload_dir) for volume_root in volume_roots]
```

Replace `_resolve_conflicts` with:

```python
def _conflict_lines(checks, prefix=''):
	"""(detail, short) text per conflicting check, as printed in the conflict list and the forcing warnings."""
	lines = []
	for check in checks:
		expected = '-' if check.row.op == 'A' else check.row.old
		lines.append((f"{prefix}{check.row.op} {check.row.path!r}: {check.reason} (expected {expected}, found {check.found})",
			f"{prefix}{check.row.op} {check.row.path}: {check.reason}"))
	return lines

def _resolve_conflict_lines(lines, force):
	"""Print conflicts, then prompt (default abort), abort without a tty, or force with warnings."""
	print(f"{len(lines)} path(s) on the target differ from the state this patch expects:")
	for detail, _ in lines:
		print(f"  {detail}")
	if not force:
		if not sys.stdin.isatty():
			raise RuntimeError(f'Patch conflict: {len(lines)} conflicting path(s) and stdin is not a tty; nothing applied (use -fa to force)')
		answer = multiCMD.input_with_timeout_and_countdown(60, 'Apply anyway? [y/N]')
		if (answer or '').strip().lower() not in ('y', 'yes'):
			raise RuntimeError(f'Patch conflict: {len(lines)} conflicting path(s); nothing applied')
	for _, short in lines:
		print(f"Warning: forcing {short}", file=sys.stderr)

def _resolve_conflicts(conflicts, force):
	"""Phase-1 entry point: one section's conflicting checks."""
	_resolve_conflict_lines(_conflict_lines(conflicts), force)
```

Replace `apply_patch_tree` (keep its docstring text unchanged) with the section helpers followed by the new body:

```python
SectionPlan = namedtuple('SectionPlan', 'name volume_roots target_root rows checks')

@contextlib.contextmanager
def _patch_hash_settings(meta):
	"""Use the patch's hash size while checking and verifying; yields (compare_hash, full_hash)."""
	global HASH_SIZE
	compare_hash = meta.get('hash_algo') == _hash_algo_name()
	if not compare_hash:
		print(f"Warning: patch was hashed with {meta.get('hash_algo')}, this system uses {_hash_algo_name()}; hashes and xattr digests are not compared", file=sys.stderr)
	full_hash = meta.get('full_hash') == '1'
	try:
		patch_hash_size = int(meta.get('hash_size', HASH_SIZE))
	except ValueError:
		raise RuntimeError(f"Patch format error: bad hash_size {meta.get('hash_size')!r}")
	saved_hash_size = HASH_SIZE
	HASH_SIZE = patch_hash_size
	try:
		yield compare_hash, full_hash
	finally:
		HASH_SIZE = saved_hash_size

def _plan_section(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers):
	"""Check one section's rows against target_root; nothing changes."""
	return SectionPlan(name, list(volume_roots), target_root, rows,
		check_patch_target(rows, target_root, compare_hash, full_hash, max_workers))

def _section_conflicts(plan):
	return [check for check in plan.checks if check.action == 'conflict']

def _section_todo(plan):
	return [check.row for check in plan.checks if check.action != 'skip']

def _resolve_section_conflicts(plans, force, extra_lines=()):
	"""One conflict list and one prompt for every section (and extra lines such as raw areas)."""
	lines = []
	for plan in plans:
		lines += _conflict_lines(_section_conflicts(plan), '' if plan.name == 'root' else f'[{plan.name}] ')
	lines += list(extra_lines)
	if lines:
		_resolve_conflict_lines(lines, force)

def _preflight_section(plan):
	"""Refusals -fa cannot override, raised before anything changes."""
	todo = _section_todo(plan)
	# Not forceable: the deletes would run and the writes be refused, leaving the patch half applied.
	escaping = _rows_escaping_target(todo, plan.target_root)
	if escaping:
		for row in escaping:
			print(f"  {row.op} {row.path!r}: resolves outside {plan.target_root} through a symlink")
		raise RuntimeError(f'Patch conflict: {len(escaping)} path(s) resolve outside the target through symlinks, e.g. {escaping[0].path}; nothing applied')
	unsafe = [row for row in todo if row.op in ('A', 'M') and row.new.startswith('f:') and _payload_source(plan.volume_roots, row) is None]
	if unsafe:
		raise RuntimeError(f'Incomplete patch: payload missing or unsafe for {len(unsafe)} file(s), e.g. {unsafe[0].path}; nothing applied')

def _apply_section(plan, max_workers, verbose, full_hash, compare_hash):
	"""Apply a checked section and verify it; returns its summary."""
	conflicts = _section_conflicts(plan)
	todo = _section_todo(plan)
	_apply_rows(todo, plan.volume_roots, plan.target_root, max_workers, verbose, full_hash,
		forced_rows=[check.row for check in conflicts])
	failed = _verify_rows(todo, plan.target_root, compare_hash, full_hash, max_workers)
	return {
		'deleted': sum(1 for row in todo if row.op == 'D'),
		'added': sum(1 for row in todo if row.op == 'A'),
		'modified': sum(1 for row in todo if row.op == 'M'),
		'skipped': len(plan.checks) - len(todo),
		'forced': len(conflicts),
		'failed': failed,
	}

def apply_patch_tree(volume_roots, target_root, force=False, max_workers=1, verbose=False, volume_names=None, store_note=False):
	"""Apply a patch whose volumes are mounted (or unpacked) at volume_roots onto directory target_root.

	Returns {'deleted', 'added', 'modified', 'skipped', 'forced', 'failed'}. Raises RuntimeError
	for a malformed or incomplete patch, or when conflicts are not overridden.
	Another process modifying the target concurrently during an apply is out of scope
	(residual race between the containment checks and the writes).
	Files and symlinks are replaced atomically (a temporary sibling renamed over the target),
	so running processes keep the old file and every target ends either old or new.
	Paths are followed through symlinked directories that resolve inside target_root, for
	deletions and writes alike; a patch that would delete or write through a symlink leading
	outside it is refused before anything changes, even when forced.
	Only conflicts the user forced may have a whole directory tree removed.
	An interrupted apply leaves new directories 0700 and root-owned.
	A hard kill mid-apply can leave new directories with default metadata (it is applied
	only after every write); the re-run then reports those A rows as conflicts and needs -fa.
	"""
	meta, rows, volume_roots = load_patch_volumes(volume_roots, volume_names)
	if store_note and meta.get('layout') == 'store':
		example = rows[0].path if rows else ''
		print(f"Note: this patch uses image-store layout; paths such as {example!r} are resolved under {target_root}")
	with _patch_hash_settings(meta) as (compare_hash, full_hash):
		plan = _plan_section('root', rows, volume_roots, target_root, compare_hash, full_hash, max_workers)
		_resolve_section_conflicts([plan], force)
		_preflight_section(plan)
		summary = _apply_section(plan, max_workers, verbose, full_hash, compare_hash)
	print('Patch summary: ' + ', '.join(f'{key} {value}' for key, value in summary.items()))
	return summary
```

(The docstring above is the current one, unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q`
Expected: all pass — every phase-1 apply test stays green.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "refactor(diff): apply patches section by section with one shared conflict prompt"
```

---

### Task 7: Generating a disk patch

**Files:**
- Modify: `hpcp.py` (Diff Image section: `build_patch_tree`, `_mount_single_fs_image`, `process_get_diff_image`; Disk Patch section)
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: Tasks 1–5, phase-1 `build_root_maps`, `_drop_lost_found`, `_mount_point_excludes`, `compute_changes`, `_validate_patch_rows`, `make_patch_meta`, `_payload_bytes`, `create_image`, `_plan_dest_images`, `_probe_fs_type`.
- Produces:
  - `_build_payload(rows, new_base, volume_roots, max_workers=1, verbose=False, estimated_size=0)`; `build_patch_tree` keeps its signature
  - `_payload_allocation(rows) -> int`, `_claim_volume_names(output, count) -> list[str]`, `_ro_mount_options(device) -> str`
  - `_attach_disk(path, read_only, loop_devices) -> str` (block device)
  - `_mount_partition(device, read_only, mount_points, entries) -> str | None` (mount point + `os.sep`)
  - `_release_mount(mount_point, mount_points)`
  - `_partition_changes(new_root, old_root, mount_points, live_mount, exclude, full_hash, max_workers, parallel_file_listing) -> list[ChangeRow]`
  - `_is_identity_change(row, kinds) -> bool`
  - `make_disk_patch_meta(sections, raws, new_name, base_name, full_hash=False) -> dict` (`sections` = `[(number, rows)]`, `raws` = `[(name, file, old_sha, new_sha, size)]`)
  - `write_disk_patch_metadata(volume_roots, sections, disk_rows, meta)`
  - `_store_raw(volume_roots, raw_file, device, size, new_sha) -> str`
  - `process_get_disk_diff_image(src_paths, dest_paths, output, mount_points, loop_devices, exclude=None, full_hash=False, dest_image_size=0, max_workers=…, parallel_file_listing=False, verbose=False) -> str | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_disk_patch_generation_records_sections_raw_and_identity(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_make_newer(old, new)
	with _session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_disk_diff_image([str(new)], [str(old)], out, mount_points, loop_devices, max_workers=1) == out
	with _session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(out, True, mount_points, loop_devices)
		meta = hpcp_mod.read_manifest(os.path.join(root, '.hpcp_diff', 'manifest'))
		assert (meta['layout'], meta['format_version']) == ('disk', '2')
		assert [section[0] for section in meta['sections']] == ['part1', 'part3']
		assert sorted(raw[0] for raw in meta['raws']) == ['mbr', 'part2']
		rows = hpcp_mod.read_disk_tsv(os.path.join(root, '.hpcp_diff', 'disk.tsv'))
		changed = {(row.number, row.field): (row.old, row.new) for row in rows if row.old != row.new}
		assert changed[('3', 'fs_label')] == ('root', 'root2') and changed[('1', 'name')] == ('ESP', 'ESP-v2')
		part3 = hpcp_mod.read_changes(os.path.join(root, '.hpcp_diff', 'part3.changes.tsv'))
		assert {(r.op, r.path) for r in part3} >= {('M', 'etc/os-release'), ('A', 'usr/bin/tool'), ('D', 'etc/obsolete.conf')}
		assert not any(r.path.startswith('lost+found') for r in part3)
		with open(os.path.join(root, 'part3', 'etc', 'os-release'), 'rb') as f:
			assert f.read() == b'v2'
		with open(os.path.join(root, 'raw', 'part2.img'), 'rb') as f:
			assert f.read(16) == b'GRUBCOR2GRUBCOR2'


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_disk_patch_generation_without_differences_creates_nothing(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', tmp_path / 'patch.img'
	_make_golden(old)
	shutil.copyfile(str(old), str(new))
	with _session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_disk_diff_image([str(new)], [str(old)], str(out), mount_points, loop_devices, max_workers=1) is None
	assert not out.exists()


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_disk_patch_generation_refuses_layout_changes(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', tmp_path / 'patch.img'
	_make_golden(old)
	_make_newer(old, new, extra_partition=True)
	with _session(hpcp_mod) as (mount_points, loop_devices):
		with pytest.raises(RuntimeError, match='Disk layout mismatch'):
			hpcp_mod.process_get_disk_diff_image([str(new)], [str(old)], str(out), mount_points, loop_devices, max_workers=1)
	assert not out.exists()


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_live_partition_hides_nested_mounts(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 2: a filesystem mounted below a live partition stays out of that partition's section."""
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_make_newer(old, new, label=None, raw=False, mbr=False, gpt_name=None)
	with _attached(new, read_only=False) as loop:
		with _mounted(loop + 'p3') as live_root:
			nested = os.path.join(live_root, 'mnt')
			_run('mount', '-t', 'tmpfs', 'tmpfs', nested)
			try:
				_write(os.path.join(nested, 'leak.txt'), b'must not appear')
				with _session(hpcp_mod) as (mount_points, loop_devices):
					hpcp_mod.process_get_disk_diff_image([loop], [str(old)], out, mount_points, loop_devices, max_workers=1)
			finally:
				subprocess.run(['umount', nested], capture_output=True)
	with _session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(out, True, mount_points, loop_devices)
		part3 = hpcp_mod.read_changes(os.path.join(root, '.hpcp_diff', 'part3.changes.tsv'))
	paths = {row.path for row in part3}
	assert 'mnt/leak.txt' not in paths and 'mnt/' not in paths
	assert ('M', 'etc/os-release') in {(row.op, row.path) for row in part3}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "generation or nested"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute 'process_get_disk_diff_image'`.

- [ ] **Step 3: Implement the phase-1 refactors**

Split `build_patch_tree` (Diff Image section) into `_build_payload` plus the metadata write:

```python
def _build_payload(rows, new_base, volume_roots, max_workers=1, verbose=False, estimated_size=0):
	"""Copy the A / M payload of rows from new_base onto the volume roots and check it landed.

	Every map key is relpath(path, new_base), so new_base is '/' for store layout and the
	new tree's root for root layout.
	"""
	for row in rows:
		if _is_reserved_key(row.path):
			raise RuntimeError(f'Diff image error: {row.path} uses the reserved name {DIFF_DIR_NAME}')
	payload = [row for row in rows if row.op in ('A', 'M')]
	dir_keys = set()
	for row in payload:
		dir_keys.update(_parent_keys(row.path))
		if row.new.startswith('d:'):
			dir_keys.add(row.path)
	for key in sorted(dir_keys, key=_key_depth):
		for volume_root in volume_roots:
			os.makedirs(_key_path(volume_root, key), exist_ok=True)
	files = [_key_path(new_base, row.path) for row in payload if row.new.startswith('f:')]
	if files:
		copy_file_list_parallel(files, [], new_base, list(volume_roots), max(1, max_workers), verbose=verbose,
			estimated_size=estimated_size)
	for row in payload:
		if row.new.startswith('l:'):
			_place_symlink(decode_state(row.new)['target'], _key_path(volume_roots[0], row.path))
	# Directories last and deepest first, so later writes cannot disturb their mtimes.
	for key in sorted(dir_keys, key=_key_depth, reverse=True):
		source = _key_path(new_base, key)
		if os.path.islink(source) and os.path.isdir(source):
			# A symlinked root is listed from its target (build_entry_map); take its metadata there too.
			source = os.path.realpath(source)
		sync_directory_metadata(source, [_key_path(volume_root, key) for volume_root in volume_roots])
	# copy_file_list_parallel reports failures only on stderr and skips full destinations, so check the result here.
	missing = _missing_payload(payload, volume_roots)
	if missing:
		raise RuntimeError(f'Diff image error: payload incomplete for {len(missing)} file(s), e.g. {missing[0]}; patch metadata not written')

def build_patch_tree(rows, new_base, volume_roots, meta, max_workers=1, verbose=False):
	"""Copy the A / M payload of rows from new_base onto the volume roots, then write .hpcp_diff/."""
	_build_payload(rows, new_base, volume_roots, max_workers, verbose, meta.get('payload_bytes', 0))
	write_patch_metadata(volume_roots, rows, meta)
```

Add next to `make_patch_meta`:

```python
def _payload_allocation(rows):
	"""Bytes a patch volume needs for rows: change list, allocation (not apparent size) and an inode each."""
	size = 2 * sum(len(row.path) + len(row.old) + len(row.new) + 8 for row in rows)
	for row in rows:
		if row.op in ('A', 'M'):
			# Allocation, not apparent size, plus the inode each entry needs (512 bytes on xfs).
			size += 512
			if row.new.startswith('f:'):
				size += _align_up(max(decode_state(row.new)['size'], 1))
			else:
				size += _IMAGE_BLOCK
	return size

def _claim_volume_names(output, count):
	"""Create every planned volume file exclusively (O_EXCL, O_NOFOLLOW); on a clash remove what this call created."""
	claimed = []
	for name in [_split_dest_image_name(output, i) for i in range(count)]:
		try:
			os.close(os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
		except FileExistsError:
			for done in claimed:
				try:
					os.unlink(done)
				except OSError:
					pass
			raise RuntimeError(f'Diff image error: output {name} already exists')
		claimed.append(name)
	return claimed
```

In `process_get_diff_image`, replace the inline sizing (from `change_list_bytes = …` through the `for row in rows:` loop that builds `init_size`) with `init_size = _payload_allocation(rows)`, and replace the inline claim loop (`claimed = []` … `claimed.append(name)`) with `_claim_volume_names(output, count)`.

Add next to `_probe_fs_type`:

```python
def _ro_mount_options(device):
	"""Read-only mount options that also work for cloned (duplicate-UUID) xfs and dirty ext3/4 journals."""
	return {'xfs': 'ro,nouuid,norecovery', 'ext3': 'ro,noload', 'ext4': 'ro,noload'}.get(_probe_fs_type(device), 'ro')
```

and in `_mount_single_fs_image` replace the line building `options` from the dict with `options = _ro_mount_options(target)` (keep its comment).

- [ ] **Step 4: Implement the disk generation units**

Append to the Disk Patch section:

```python
def _attach_disk(path, read_only, loop_devices):
	"""Block device for path: itself when it is one, else a loop device over the disk-image file."""
	path = os.path.abspath(os.path.expanduser(path))
	if pathlib.Path(path).resolve().is_block_device():
		return path
	if not os.path.isfile(path):
		raise RuntimeError(f'Invalid arguments: {path} is not a block device or a disk image file')
	loop_device = create_loop_device(path, read_only=read_only)
	loop_devices.append(loop_device)
	return loop_device

def _release_mount(mount_point, mount_points):
	"""Unmount one of this run's temp mounts and remove its directory (clean_up handles anything left)."""
	path = mount_point.rstrip(os.sep)
	if os.path.ismount(path):
		run_command_in_multicmd_with_path_check(['umount', path], strict=False)
	if not os.path.ismount(path):
		try:
			os.rmdir(path)
		except OSError:
			pass
		if path in mount_points:
			mount_points.remove(path)

def _mount_partition(device, read_only, mount_points, entries):
	"""Mount point (with a trailing separator) showing exactly device's filesystem, or None.

	A partition that is already mounted is bind-mounted (non-recursively) from its live mount,
	so filesystems mounted below it stay out of view; otherwise it is mounted directly
	(read-only with _ro_mount_options). Nothing is recorded in ERRORS: a partition the kernel
	cannot mount is an expected answer, and the caller decides what it means.
	"""
	live, _ = _live_mount(_device_number(device), entries, device)
	mount_point = tempfile.mkdtemp()
	mount_points.append(mount_point)
	error_count = len(ERRORS)
	if live:
		run_command_in_multicmd_with_path_check(['mount', '--bind', live, mount_point], quiet=True)
		if read_only and os.path.ismount(mount_point):
			# Best effort: generation only reads through this bind mount.
			run_command_in_multicmd_with_path_check(['mount', '-o', 'remount,bind,ro', mount_point], quiet=True)
	elif read_only:
		run_command_in_multicmd_with_path_check(['mount', '-o', _ro_mount_options(device), device, mount_point], quiet=True)
	else:
		run_command_in_multicmd_with_path_check(['mount', device, mount_point], quiet=True)
	del ERRORS[error_count:]
	if not os.path.ismount(mount_point):
		_release_mount(mount_point, mount_points)
		return None
	return mount_point + os.sep

def _partition_changes(new_root, old_root, mount_points, live_mount, exclude, full_hash, max_workers, parallel_file_listing):
	"""Change rows of one filesystem partition (root to root, lost+found and this run's mounts left out)."""
	matched = os.path.abspath(new_root)
	patterns = _mount_point_excludes(mount_points, (new_root, old_root), matched)
	if live_mount:
		# The live filesystem holds the (empty) directories of this run's other temp mount points.
		live_real = os.path.realpath(live_mount)
		for mount_point in mount_points:
			mount_real = os.path.realpath(mount_point)
			if mount_real.startswith(live_real.rstrip(os.sep) + os.sep):
				patterns.add(glob.escape(os.path.join(matched, os.path.relpath(mount_real, live_real))))
	new_map, base_map = build_root_maps(new_root, old_root, frozenset(exclude or ()) | patterns, max_workers, parallel_file_listing)
	new_map, base_map = _drop_lost_found(new_map), _drop_lost_found(base_map)
	if any(_is_reserved_key(key) for key in new_map):
		raise RuntimeError(f'Diff image error: {new_root} contains {DIFF_DIR_NAME}, which hpcp reserves for patch metadata')
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers)
	_validate_patch_rows(rows)
	return rows

def _is_identity_change(row, kinds):
	"""A disk.tsv row whose value the patch would set on a target."""
	if row.old == row.new:
		return False
	if row.kind == 'disk':
		return row.field == 'guid'
	if row.field not in _IDENTITY_FIELDS:
		return False
	return not (kinds.get(int(row.number)) == 'raw' and row.field in ('fs_uuid', 'fs_label'))

def make_disk_patch_meta(sections, raws, new_name, base_name, full_hash=False):
	"""Manifest fields for a disk patch; sections = [(number, rows)], raws = [(name, file, old, new, size)]."""
	meta = make_patch_meta([row for _, rows in sections for row in rows], new_name, base_name, 'disk', full_hash)
	meta['format_version'] = DIFF_FORMAT_VERSION
	meta['sections'] = [(f'part{number}', f'part{number}', f'part{number}.changes.tsv') for number, _ in sections]
	meta['raws'] = list(raws)
	meta['disk'] = 'disk.tsv'
	meta['raw_bytes'] = sum(raw[4] for raw in raws)
	return meta

def write_disk_patch_metadata(volume_roots, sections, disk_rows, meta):
	"""Write every change list and disk.tsv on every volume first, then each volume's manifest (volume k/n)."""
	total = len(volume_roots)
	for volume_root in volume_roots:
		diff_dir = os.path.join(volume_root, DIFF_DIR_NAME)
		os.makedirs(diff_dir, exist_ok=True)
		for number, rows in sections:
			write_changes(os.path.join(diff_dir, f'part{number}.changes.tsv'), rows)
		write_disk_tsv(os.path.join(diff_dir, meta['disk']), disk_rows)
	for index, volume_root in enumerate(volume_roots, start=1):
		volume_meta = dict(meta)
		volume_meta['volume'] = f'{index}/{total}'
		write_manifest(os.path.join(volume_root, DIFF_DIR_NAME, 'manifest'), volume_meta)

def _store_raw(volume_roots, raw_file, device, size, new_sha):
	"""Copy a raw area (the first size bytes of device) into raw_file on a volume with room."""
	for volume_root in volume_roots:
		if shutil.disk_usage(volume_root).free > size + (1 << 20):
			dest = os.path.join(volume_root, raw_file)
			os.makedirs(os.path.dirname(dest), exist_ok=True)
			if _copy_area(device, dest, 0, size) != new_sha:
				raise RuntimeError(f'Raw area error: {raw_file} changed while it was copied; patch metadata not written')
			return dest
	raise RuntimeError(f'Diff image error: no patch volume has room for {raw_file} ({format_bytes(size)}B); use a larger -dis')

def process_get_disk_diff_image(src_paths, dest_paths, output, mount_points, loop_devices, exclude=None,
		full_hash=False, dest_image_size=0, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False, verbose=False):
	"""Write a disk patch image that turns a disk laid out like the old disk (-d) into the new disk.

	Both disks are GPT block devices or disk-image files and are only read. Returns the
	patch path, or None when the disks do not differ (no image is created).
	"""
	sources, bases = expand_paths(src_paths or []), expand_paths(dest_paths or [])
	if len(sources) != 1 or len(bases) != 1:
		raise RuntimeError(f'Invalid arguments: -dd -diff needs exactly one new disk and one old disk (-d), got {src_paths} and {dest_paths}')
	new_name, base_name = os.path.basename(sources[0]), os.path.basename(bases[0])
	if not output or output == 'auto':
		output = f'DIFF_{new_name}_TO_{base_name}_{int(time.time())}.img'
	output = os.path.abspath(os.path.expanduser(output))
	if os.path.lexists(output):
		raise RuntimeError(f'Diff image error: output {output} already exists')
	new_disk = _attach_disk(sources[0], True, loop_devices)
	old_disk = _attach_disk(bases[0], True, loop_devices)
	new_layout, old_layout = read_disk_layout(new_disk), read_disk_layout(old_disk)
	problems, _ = compare_disk_layouts(old_layout, new_layout)
	if problems:
		raise RuntimeError('Disk layout mismatch: ' + '; '.join(problems) + '; re-clone instead')
	entries = _mountinfo_entries()
	kinds, sections, raws = {}, [], []
	for number in sorted(new_layout['parts']):
		new_part, old_part = new_layout['parts'][number], old_layout['parts'][number]
		kind = _default_kind(new_part)
		if kind == 'fs':
			new_root = _mount_partition(new_part['device'], True, mount_points, entries)
			old_root = _mount_partition(old_part['device'], True, mount_points, entries) if new_root else None
			if new_root and old_root:
				live, _ = _live_mount(_device_number(new_part['device']), entries, new_part['device'])
				rows = _partition_changes(new_root, old_root, mount_points, live, exclude, full_hash, max_workers, parallel_file_listing)
				_release_mount(old_root, mount_points)
				if rows:
					# Kept mounted: its payload is copied once the patch image exists.
					sections.append((number, new_root, rows))
				else:
					_release_mount(new_root, mount_points)
			else:
				for root in (new_root, old_root):
					if root:
						_release_mount(root, mount_points)
				kind = 'raw'
		if kind == 'raw':
			if new_part['size'] != old_part['size']:
				raise RuntimeError(f"Disk layout mismatch: partition {number} is a raw area and its size changed ({old_part['size']} vs {new_part['size']}); re-clone instead")
			old_sha, new_sha = _hash_area(old_part['device'], 0, old_part['size']), _hash_area(new_part['device'], 0, new_part['size'])
			if old_sha != new_sha:
				raws.append((f'part{number}', new_part['device'], new_part['size'], old_sha, new_sha))
		kinds[number] = kind
	old_mbr, new_mbr = _hash_area(old_disk, 0, _MBR_BOOTCODE_SIZE), _hash_area(new_disk, 0, _MBR_BOOTCODE_SIZE)
	if old_mbr != new_mbr:
		raws.append(('mbr', new_disk, _MBR_BOOTCODE_SIZE, old_mbr, new_mbr))
	_, disk_rows = compare_disk_layouts(old_layout, new_layout, kinds)
	for row in disk_rows:
		if row.field == 'fs_params' and row.old != row.new:
			print(f"Warning: partition {row.number} was created with different mkfs parameters ({row.old} -> {row.new}); a patch cannot change them", file=sys.stderr)
	identity = [row for row in disk_rows if _is_identity_change(row, kinds)]
	raw_entries = [(name, 'raw/mbr-bootcode.bin' if name == 'mbr' else f'raw/{name}.img', old_sha, new_sha, size)
		for name, _, size, old_sha, new_sha in raws]
	section_rows = [(number, rows) for number, _, rows in sections]
	meta = make_disk_patch_meta(section_rows, raw_entries, new_name, base_name, full_hash)
	print(f"Disk changes: {len(sections)} filesystem section(s), {len(raws)} raw area(s), {len(identity)} identity value(s); "
		f"{meta['added']} added, {meta['modified']} modified, {meta['deleted']} deleted")
	if not sections and not raws and not identity:
		print(f"No differences between {sources[0]} and {bases[0]}; no patch image created.")
		return None
	init_size = (sum(_payload_allocation(rows) for _, rows in section_rows) + sum(_align_up(raw[2]) for raw in raws)
		+ 64 * 1024 * (len(sections) + 1))
	count, volume_size = _plan_dest_images(init_size, dest_image_size)
	largest_raw = max((raw[2] for raw in raws), default=0)
	# Same usable-space estimate as _plan_dest_images uses for split volumes.
	if count > 1 and largest_raw > int(volume_size * 0.90) - (16 << 20):
		needed = _align_up(int((largest_raw + (16 << 20)) / 0.90) + 1)
		raise RuntimeError(f'Diff image error: a raw area of {format_bytes(largest_raw)}B must fit in one patch volume; use -dis {format_bytes(needed)}B or larger')
	if os.geteuid() != 0:
		print("Warning: creating a patch image usually needs root privileges (loop devices, mkfs, mount)", file=sys.stderr)
	target_mount_point = tempfile.mkdtemp()
	mount_points.append(target_mount_point)
	_claim_volume_names(output, count)
	volume_roots = create_image(output, target_mount_point, loop_devices, [], mount_points, max_workers=max_workers,
		dest_image_size=dest_image_size, init_size=init_size)
	for number, new_root, rows in sections:
		section_roots = [os.path.join(volume_root, f'part{number}') for volume_root in volume_roots]
		for root in section_roots:
			os.makedirs(root, exist_ok=True)
		_build_payload(rows, new_root, section_roots, max_workers, verbose, _payload_bytes(rows))
	for (name, device, size, old_sha, new_sha), entry in zip(raws, raw_entries):
		_store_raw(volume_roots, entry[1], device, size, new_sha)
	write_disk_patch_metadata(volume_roots, section_rows, disk_rows, meta)
	extra = f" (+{len(volume_roots) - 1} more volume(s))" if len(volume_roots) > 1 else ''
	print(f"Disk patch image written to {output}{extra}")
	return output

```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q`
Expected: all pass; afterwards `losetup -a` and `findmnt | grep /tmp/tmp` show nothing left from the run.

- [ ] **Step 6: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "feat(dd-diff): generate disk patch images with per-partition sections and raw areas"
```

---

### Task 8: Applying a disk patch

**Files:**
- Modify: `hpcp.py` (Disk Patch section)
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: Tasks 1–7, phase-1 section helpers from Task 6.
- Produces:
  - `_refuse_if_image_in_use(image, entries)` (raises `Disk in use`)
  - `apply_disk_patch(volume_roots, target, mount_points, loop_devices, force=False, max_workers=1, verbose=False, volume_names=None) -> dict` with keys `sections` (name → phase-1 summary), `raw_written`, `raw_skipped`, `identity_changed`, `forced`, `failed`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
def _make_patch(hpcp_mod, tmp_path, **newer):
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_make_newer(old, new, **newer)
	with _session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_disk_diff_image([str(new)], [str(old)], out, mount_points, loop_devices, max_workers=1) == out
	return old, new, out


def _apply(hpcp_mod, patch, target, force=False):
	with _session(hpcp_mod) as (mount_points, loop_devices):
		volume = hpcp_mod._mount_single_fs_image(patch, True, mount_points, loop_devices)
		return hpcp_mod.apply_disk_patch([volume], str(target), mount_points, loop_devices, force=force, max_workers=1)


def _no_differences(hpcp_mod, tmp_path, new, target):
	with _session(hpcp_mod) as (mount_points, loop_devices):
		return hpcp_mod.process_get_disk_diff_image([str(new)], [str(target)], str(tmp_path / 'check.img'),
			mount_points, loop_devices, max_workers=1) is None


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_disk_apply_round_trip_and_rerun(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Success criterion + Review Focus 5."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	summary = _apply(hpcp_mod, patch, target)
	assert summary['failed'] == 0 and summary['raw_written'] == 2 and summary['identity_changed'] == 2
	assert _no_differences(hpcp_mod, tmp_path, new, target)
	again = _apply(hpcp_mod, patch, target)
	assert again['failed'] == 0 and again['raw_written'] == 0 and again['identity_changed'] == 0
	assert all(s['added'] + s['modified'] + s['deleted'] == 0 for s in again['sections'].values())


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_disk_apply_refuses_a_different_layout(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	_run('sgdisk', '-n5:0:+4M', '-t5:8300', target)
	before = _sha(target.read_bytes())
	with pytest.raises(RuntimeError, match='Disk layout mismatch'):
		_apply(hpcp_mod, patch, target)
	assert _sha(target.read_bytes()) == before


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_live_disk_identity_change_on_a_mounted_partition_is_refused(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	with _attached(target) as loop:
		with _mounted(loop + 'p3') as live_root:
			with pytest.raises(RuntimeError, match='Disk in use: .*partition 3 is mounted'):
				_apply(hpcp_mod, patch, loop)
			with open(os.path.join(live_root, 'etc', 'os-release'), 'rb') as f:
				assert f.read() == b'v1'


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_live_disk_files_apply_through_the_live_mount(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, patch = _make_patch(hpcp_mod, tmp_path, label=None, raw=False, mbr=False, gpt_name=None)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	with _attached(target) as loop:
		with _mounted(loop + 'p3') as live_root:
			summary = _apply(hpcp_mod, patch, loop)
			assert summary['failed'] == 0
			with open(os.path.join(live_root, 'etc', 'os-release'), 'rb') as f:
				assert f.read() == b'v2'
			assert os.path.ismount(live_root)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_image_target_attached_and_mounted_elsewhere_is_refused(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	with _attached(target) as loop:
		with _mounted(loop + 'p1'):
			with pytest.raises(RuntimeError, match='Disk in use'):
				_apply(hpcp_mod, patch, target)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_damaged_raw_payload_is_refused_up_front(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Review Focus 4."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	with _attached(patch) as loop:
		with _mounted(loop) as root:
			with open(os.path.join(root, 'raw', 'part2.img'), 'r+b') as f:
				f.write(b'BITROT!!')
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	before = _sha(target.read_bytes())
	with pytest.raises(RuntimeError, match='Incomplete patch: raw payload'):
		_apply(hpcp_mod, patch, target)
	assert _sha(target.read_bytes()) == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "apply or live_disk or attached or damaged"`
Expected: FAIL with `AttributeError: module 'hpcp' has no attribute 'apply_disk_patch'`.

- [ ] **Step 3: Implement**

Append to the Disk Patch section:

```python
def _refuse_if_image_in_use(image, entries):
	"""Refuse an image-file target that is attached to a loop device with something mounted from it."""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['losetup', '-j', image], quiet=True)
	finally:
		del ERRORS[error_count:]
	for line in output:
		loop = line.partition(':')[0].strip()
		if not loop:
			continue
		for device in [loop] + get_partitions(loop):
			try:
				devnum = _device_number(device)
			except OSError:
				continue
			if any(_entry_matches(entry, devnum, device) for entry in entries):
				raise RuntimeError(f'Disk in use: {image} is attached as {loop} and {device} is mounted; unmount it first or apply to the device')

def apply_disk_patch(volume_roots, target, mount_points, loop_devices, force=False, max_workers=1, verbose=False, volume_names=None):
	"""Apply a disk patch (layout disk) to target: a GPT block device or disk-image file.

	Everything is checked before anything changes: the layout, which partitions are in use,
	every section's paths, the raw areas and the identity values. Filesystem sections are
	applied first (partition order, ESP last), then raw partition areas, the MBR boot code
	and identity changes; then everything is verified.
	Returns {'sections', 'raw_written', 'raw_skipped', 'identity_changed', 'forced', 'failed'}.
	"""
	meta, sections, volume_roots = _load_patch(volume_roots, volume_names)
	if meta.get('layout') != 'disk':
		raise RuntimeError('Invalid arguments: this is a single-filesystem patch; apply it to a directory or a filesystem image')
	disk_rows = read_disk_tsv(os.path.join(volume_roots[0], DIFF_DIR_NAME, meta['disk']))
	target = os.path.abspath(os.path.expanduser(target))
	entries = _mountinfo_entries()
	if not pathlib.Path(target).resolve().is_block_device():
		_refuse_if_image_in_use(target, entries)
	disk = _attach_disk(target, False, loop_devices)
	layout = read_disk_layout(disk)
	problems = match_target_layout(disk_rows, layout)
	if problems:
		raise RuntimeError('Disk layout mismatch: ' + '; '.join(problems) + '; nothing applied')
	swaps = _swap_devices()
	in_use = {number: _partition_in_use(part['device'], _device_number(part['device']), entries, swaps)
		for number, part in layout['parts'].items()}
	changes, refusals, warnings = _plan_identity(disk_rows, layout, in_use)
	raw_plan = []
	for name, raw_file, old_sha, new_sha, size in meta['raws']:
		number = None if name == 'mbr' else int(name[len('part'):])
		device = disk if number is None else layout['parts'][number]['device']
		found = _hash_area(device, 0, size)
		action = _plan_raw(old_sha, new_sha, found)
		payload = None
		if action != 'skip':
			if number is not None and in_use.get(number):
				refusals.append(f'partition {number} is {in_use[number]} and its raw area changes')
			payload = _raw_payload(volume_roots, raw_file, new_sha, size)
			if payload is None:
				raise RuntimeError(f'Incomplete patch: raw payload {raw_file} is missing or damaged; nothing applied')
		raw_plan.append({'name': name, 'device': device, 'size': size, 'payload': payload, 'action': action,
			'found': found, 'new': new_sha})
	for name, _, _ in sections:
		part = layout['parts'][int(name[len('part'):])]
		live, live_read_only = _live_mount(_device_number(part['device']), entries, part['device'])
		if live and live_read_only:
			refusals.append(f'partition {name[len("part"):]} is mounted read-only at {live}')
	if refusals:
		raise RuntimeError('Disk in use: ' + '; '.join(refusals) + '; apply this patch offline')
	commands, unsupported = _identity_command_list(changes, layout, disk)
	if unsupported and not force:
		raise RuntimeError('Identity change error: ' + '; '.join(text for _, _, text in unsupported) + '; nothing applied (use -fa to apply the rest)')
	for _, _, text in unsupported:
		print(f'Warning: {text}; skipped', file=sys.stderr)
	for text in warnings:
		print(f'Warning: {text}', file=sys.stderr)
	skipped_identity = {(number, field) for number, fields, _ in unsupported for field in fields}
	with _patch_hash_settings(meta) as (compare_hash, full_hash):
		plans = []
		for name, payload_dir, rows in sections:
			part = layout['parts'][int(name[len('part'):])]
			root = _mount_partition(part['device'], False, mount_points, entries)
			if root is None:
				raise RuntimeError(f"Disk layout mismatch: partition {name[len('part'):]} ({part['fs_type']}) cannot be mounted read-write; nothing applied")
			plans.append(_plan_section(name, rows, _section_volume_roots(volume_roots, payload_dir), root, compare_hash, full_hash, max_workers))
		raw_conflicts = [entry for entry in raw_plan if entry['action'] == 'conflict']
		extra = [(f"raw {entry['name']}: content matches neither the patch base nor the new state (found {entry['found'][:12]})",
			f"raw {entry['name']}: content differs") for entry in raw_conflicts]
		_resolve_section_conflicts(plans, force, extra)
		for plan in plans:
			_preflight_section(plan)
		summaries = {}
		order = sorted(plans, key=lambda plan: (layout['parts'][int(plan.name[len('part'):])]['type_guid'].upper() == _ESP_TYPE_GUID,
			int(plan.name[len('part'):])))
		for plan in order:
			summaries[plan.name] = _apply_section(plan, max_workers, verbose, full_hash, compare_hash)
			_release_mount(plan.target_root, mount_points)
	failed = sum(summary['failed'] for summary in summaries.values())
	pending = [entry for entry in raw_plan if entry['action'] != 'skip']
	# Partition areas first, the MBR boot code last.
	for entry in sorted(pending, key=lambda entry: entry['name'] == 'mbr'):
		try:
			_write_area(entry['device'], entry['payload'], 0, entry['size'])
		except OSError as e:
			entry['error'] = str(e)
	for command in commands:
		run_command_in_multicmd_with_path_check(command, strict=False)
	# Verify after the identity commands: sgdisk rewriting the GPT must not have undone a raw write.
	written = 0
	for entry in pending:
		reason = entry.get('error')
		if not reason:
			try:
				if _hash_area(entry['device'], 0, entry['size']) == entry['new']:
					written += 1
					continue
				reason = 'does not hold the new content after writing'
			except OSError as e:
				reason = str(e)
		failed += 1
		eprint(f"Raw area error: {entry['name']}: {reason}")
	changed = 0
	applied = [change for change in changes if (change.number, change.field) not in skipped_identity]
	if applied:
		after = read_disk_layout(disk)
		for change in applied:
			current = str(after['disk'].get('guid', '')) if change.number is None else str(after['parts'][change.number][change.field])
			if current == change.new:
				changed += 1
			else:
				failed += 1
				where = 'disk' if change.number is None else f'partition {change.number}'
				eprint(f'Identity change error: {where} {change.field} is {current!r}, expected {change.new!r}')
	summary = {
		'sections': summaries,
		'raw_written': written,
		'raw_skipped': sum(1 for entry in raw_plan if entry['action'] == 'skip'),
		'identity_changed': changed,
		'forced': sum(summary['forced'] for summary in summaries.values()) + len(raw_conflicts),
		'failed': failed,
	}
	print(f"Disk patch summary: {len(summaries)} section(s), raw written {written}, raw skipped {summary['raw_skipped']}, "
		f"identity changed {changed}, forced {summary['forced']}, failed {failed}")
	return summary

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py -q`
Expected: all pass; `losetup -a` and `findmnt | grep /tmp/tmp` show nothing left from the run.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "feat(dd-diff): apply disk patches to offline and live disks"
```

---

### Task 9: CLI wiring

**Files:**
- Modify: `hpcp.py` (`_validate_diff_args`, `process_load_diff_image`, `hpcp()` dispatch ≈ line 6097, `get_args` help for `-diff` / `-siff`)
- Modify: `tests/test_diff_image.py` (`test_hpcp_rejects_bad_patch_arguments` — the `-dd` case)
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Consumes: `process_get_disk_diff_image`, `apply_disk_patch`, `_attach_disk`.
- Produces:
  - `_validate_diff_args(...)` accepts `dd` with `-diff` (exactly one positional, one `-d`, no `-si` / `-di`) and with `-siff`
  - `_patch_layout(volume_roots) -> str` (the first readable manifest's `layout`, default `'root'`)
  - `process_load_diff_image(..., dd=False)` dispatches disk patches to `apply_disk_patch`

- [ ] **Step 1: Write the failing tests**

In `tests/test_diff_image.py`, `test_hpcp_rejects_bad_patch_arguments`, replace the case

```python
		(dict(src_path=[new], dest_paths=[base], get_diff_image='auto', dd=True), '-dd'),
```

with

```python
		(dict(src_path=[new], dest_paths=[base], get_diff_image='auto', dd=True, src_image=['x.img']), '-dd'),
```

Append to `tests/test_disk_patch.py`:

```python
def test_validate_dd_diff_arguments(tmp_path, hpcp_mod):
	v = hpcp_mod._validate_diff_args
	assert v('out.img', None, False, ['/dev/sdb'], None, ['old.img'], None, dd=True) == ('out.img', ['/dev/sdb'])
	existing = tmp_path / 'new.img'
	existing.write_bytes(b'')
	assert v(str(existing), None, False, [], None, ['old.img'], None, dd=True) == ('auto', [str(existing)])
	for kwargs, message in (
		(dict(src_path=['a', 'b'], dest_paths=['old.img']), 'exactly one new disk'),
		(dict(src_path=['a'], dest_paths=[]), 'exactly one old disk'),
		(dict(src_path=['a'], dest_paths=['old.img'], dest_image='x.img'), 'no -si / -di'),
	):
		with pytest.raises(RuntimeError, match=message):
			v('auto', None, False, kwargs['src_path'], None, kwargs['dest_paths'], kwargs.get('dest_image'), dd=True)
	assert v(None, ['p.img'], False, [], None, ['/dev/sdb'], None, dd=True) == (None, [])


def test_patch_layout_reads_the_manifest(tmp_path, hpcp_mod):
	volume = tmp_path / 'vol'
	(volume / '.hpcp_diff').mkdir(parents=True)
	assert hpcp_mod._patch_layout([str(volume)]) == 'root'
	hpcp_mod.write_manifest(str(volume / '.hpcp_diff' / 'manifest'), _disk_meta())
	assert hpcp_mod._patch_layout([str(volume)]) == 'disk'


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_hpcp_cli_generates_and_applies_a_disk_patch(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'p.img')
	_make_golden(old)
	_make_newer(old, new)
	assert hpcp_mod.hpcp([str(new)], dest_paths=[str(old)], get_diff_image=out, dd=True, max_workers=1) == 0
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	assert hpcp_mod.hpcp([], dest_paths=[str(target)], load_diff_image=[out], max_workers=1) == 0
	assert _no_differences(hpcp_mod, tmp_path, new, target)
	assert hpcp_mod.hpcp([], dest_paths=[str(tmp_path)], load_diff_image=[out], max_workers=1) == 241
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q -k "validate or patch_layout or cli or bad_patch"`
Expected: FAIL — `-dd` is still rejected by `_validate_diff_args`, `_patch_layout` is missing.

- [ ] **Step 3: Implement**

In `_validate_diff_args`, replace

```python
	mode = '-siff' if load_diff_image else '-diff'
	if dd:
		raise RuntimeError(f'Invalid arguments: {mode} with -dd is not supported yet (dd-mode patches are planned)')
```

with

```python
	mode = '-siff' if load_diff_image else '-diff'
```

and, immediately before the existing line `if get_diff_image not in (None, 'auto') and os.path.isdir(os.path.expanduser(get_diff_image)):`, insert:

```python
	if dd:
		if src_image or dest_image:
			raise RuntimeError('Invalid arguments: -dd -diff takes the new disk as a path and the old disk with -d (no -si / -di)')
		if get_diff_image not in (None, 'auto') and os.path.lexists(os.path.expanduser(get_diff_image)):
			# The output must not exist, so an existing path is the new disk argparse took as the patch name.
			print(f"Note: -diff value {get_diff_image} exists; using it as the new disk and generating the patch name")
			src_path = [get_diff_image] + src_path
			get_diff_image = 'auto'
		if len(src_path) != 1:
			raise RuntimeError(f'Invalid arguments: -dd -diff needs exactly one new disk (a path), got {len(src_path)}')
		if len(dest_paths) != 1:
			raise RuntimeError(f'Invalid arguments: -dd -diff needs exactly one old disk (-d), got {len(dest_paths)}')
		return get_diff_image, src_path
```

Add before `process_load_diff_image`:

```python
def _patch_layout(volume_roots):
	"""layout of the first readable manifest among the volumes ('root' when none can be read)."""
	for volume_root in volume_roots:
		try:
			return read_manifest(os.path.join(volume_root, DIFF_DIR_NAME, 'manifest')).get('layout', 'root')
		except (OSError, RuntimeError):
			continue
	return 'root'
```

Replace `process_load_diff_image` with:

```python
def process_load_diff_image(load_diff_image, src_paths, dest_paths, dest_image, mount_points, loop_devices,
		force=False, max_workers=multiprocessing.cpu_count(), verbose=False, dd=False):
	"""Mount every -siff volume and apply the patch: a disk patch to -d DISK (or one path), a
	single-filesystem patch to -di IMG, -d DIR, or a single positional path."""
	images = expand_paths(load_diff_image)
	if not images:
		raise RuntimeError(f'Incomplete patch: no patch image found for {load_diff_image}')
	volume_roots = [_mount_single_fs_image(image, True, mount_points, loop_devices) for image in images]
	volume_names = [os.path.abspath(image) for image in images]
	if _patch_layout(volume_roots) == 'disk':
		if dest_image:
			raise RuntimeError('Invalid arguments: disk patches take the target disk with -d (or a single path), not -di')
		targets = expand_paths(dest_paths) if dest_paths else expand_paths(src_paths or [])
		if len(targets) != 1 or os.path.isdir(targets[0]):
			raise RuntimeError(f'Invalid arguments: a disk patch needs exactly one target disk (block device or disk image file), got {dest_paths or src_paths}')
		return apply_disk_patch(volume_roots, targets[0], mount_points, loop_devices, force=force, max_workers=max_workers,
			verbose=verbose, volume_names=volume_names)
	if dd:
		raise RuntimeError('Invalid arguments: -dd with -siff needs a disk patch; this is a single-filesystem patch')
	if dest_image:
		target_root = _mount_single_fs_image(dest_image, False, mount_points, loop_devices)
		store_note = False
	else:
		targets = expand_paths(dest_paths) if dest_paths else expand_paths(src_paths or [])
		if len(targets) != 1 or not os.path.isdir(targets[0]):
			raise RuntimeError(f'Invalid arguments: -siff needs exactly one existing target directory, got {dest_paths or src_paths}')
		target_root = os.path.abspath(targets[0])
		store_note = target_root != os.path.abspath(os.sep)
	return apply_patch_tree(volume_roots, target_root, force=force, max_workers=max_workers, verbose=verbose,
		volume_names=volume_names, store_note=store_note)
```

In `hpcp()`, in the patch-image dispatch block, pass `dd=dd` to `process_load_diff_image(...)`, and replace the generation lines

```python
			process_get_diff_image(src_path, src_image, dest_paths, dest_image, get_diff_image, mount_points, loop_devices,
				exclude=exclude, full_hash=full_hash, dest_image_size=dest_image_size, max_workers=workers,
				parallel_file_listing=parallel_file_listing, verbose=verbose)
			raise RuntimeError("Exiting after generating patch image.")
```

with

```python
			if dd:
				process_get_disk_diff_image(src_path, dest_paths, get_diff_image, mount_points, loop_devices,
					exclude=exclude, full_hash=full_hash, dest_image_size=dest_image_size, max_workers=workers,
					parallel_file_listing=parallel_file_listing, verbose=verbose)
				raise RuntimeError("Exiting after generating disk patch image.")
			process_get_diff_image(src_path, src_image, dest_paths, dest_image, get_diff_image, mount_points, loop_devices,
				exclude=exclude, full_hash=full_hash, dest_image_size=dest_image_size, max_workers=workers,
				parallel_file_listing=parallel_file_listing, verbose=verbose)
			raise RuntimeError("Exiting after generating patch image.")
```

In `get_args`, append to the `-diff` help text: `' With -dd, compares whole GPT disks: the new disk as the path and the old disk with -d.'`; append to the `-siff` help text: `' A disk patch (made with -dd -diff) takes the target disk with -d.'`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py tests/test_cli_args.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py tests/test_diff_image.py
git commit -m "feat(cli): wire -dd -diff and disk-patch -siff into hpcp"
```

---

### Task 10: Docs, version, full verification

**Files:**
- Modify: `hpcp.py` (`version`, `COMMIT_DATE`)
- Modify: `tests/test_dd_fs_params.py` (`test_version_bumped`)
- Modify: `README.md`

- [ ] **Step 1: Bump the version and date**

```bash
sed -i "s/^version = '9.65'$/version = '9.66'/" hpcp.py
sed -i "s/^COMMIT_DATE = '.*'$/COMMIT_DATE = '$(date +%F)'/" hpcp.py
grep -nE "^version|^COMMIT_DATE" hpcp.py
grep -n "9.65\|COMMIT_DATE\|version" tests/test_dd_fs_params.py | head
```

Update `test_version_bumped` in `tests/test_dd_fs_params.py` to expect `'9.66'` and the new `COMMIT_DATE`.

- [ ] **Step 2: Add the README subsection**

In `README.md`, at the end of the "Patch Images Feature Note" section (before the next `##` heading), add:

````markdown
### Disk patches (`-dd`)

`hpcp -dd -diff [PATCH.img] NEW_DISK -d OLD_DISK` patches whole GPT disks — for rolling a new golden image out to nodes cloned from the previous one. Both disks are block devices or disk-image files and are only read. The patch holds one section per filesystem partition (the same change list and payload as a single-filesystem patch), raw copies of areas that cannot be patched file by file when they changed (BIOS-boot and unknown-filesystem partitions, the MBR boot code), and the identity values that changed (filesystem UUID / label, GPT partition GUID / name / attributes, disk GUID). Swap contents are never carried.

`hpcp -siff PATCH.img -d TARGET_DISK` applies it to a node disk, offline or live:

- The target must have the base's layout: the same partition numbers, type GUIDs and filesystem types (sizes may differ, except raw areas). Layout changes need a re-clone.
- On a live disk, mounted partitions are patched through their live mount. Raw writes and identity changes to a partition that is mounted, active swap or held by LVM/md are refused before anything changes — apply such patches offline.
- Filesystem sections are applied first (ESP last), then raw areas, the MBR boot code, and identity changes; everything is verified afterwards, and a re-run skips what is already done.
- A node whose UUID or label differs from the golden base gets a warning and the new value.

````

- [ ] **Step 3: Regenerate the README help block**

```bash
python3 - <<'EOF'
import os, subprocess
env = dict(os.environ, COLUMNS='80')
help_text = subprocess.run(['python3', 'hpcp.py', '-h'], capture_output=True, text=True, env=env, check=True).stdout
help_text = help_text.split('\nFound bins:')[0].rstrip('\n') + '\n'
readme = open('README.md').read()
marker = '```bash\n$ hpcp -h\n'
start = readme.index(marker) + len(marker)
end = readme.index('```', start)
open('README.md', 'w').write(readme[:start] + help_text + readme[end:])
EOF
grep -c "Found bins" README.md
```

Expected: `0` (the machine-specific epilog is cut off), and the help block shows the new `-diff` / `-siff` text.

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (root/loop tests execute as root). Then `losetup -a` and `findmnt | grep /tmp/tmp` show nothing left from the run, and `python3 hpcp.py -V` prints `9.66`.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py README.md tests/test_dd_fs_params.py
git commit -m "docs: document disk patches and release 9.66"
```

Releasing (push, uvpack, cas-toolbox pin) is out of scope for this plan.
