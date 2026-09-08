# dd-mode Filesystem Parameter Mirroring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In dd mode, recreate each destination filesystem with the source's geometry and feature set instead of the building host's `mkfs` defaults, so clones stay mountable and bootable on the system they came from.

**Architecture:** Two module-level registries keyed by `fs_type` — `_FS_PARAM_PROBES` maps a filesystem to a function that reads its parameters from the source partition, `_FS_MKFS_BUILDERS` maps it to a function that turns those parameters into `mkfs` arguments. `get_partition_details()` stores the probe result under a new `fs_params` key; `write_partition_info()` splices the built arguments into each existing `mkfs` command. Every layer degrades to today's behavior on failure.

**Tech Stack:** Python 3 stdlib only (`subprocess` via the existing `multiCMD` wrapper, `struct` for two superblock readers), pytest for tests. External probe tools are called through the existing `run_command_in_multicmd_with_path_check()` / `_binPaths` mechanism.

**Spec:** `docs/superpowers/specs/2026-09-08-dd-fs-param-mirroring-design.md`

## Global Constraints

- **Single-file module.** All production code goes in `hpcp.py`. No new modules, no packages — the tool must stay copy-transportable and importable standalone.
- **Tabs for indentation** in `hpcp.py` and in `tests/`, matching the existing files.
- **Probing must never break a copy.** Every probe returns `{}` on any failure; every builder returns `[]` on unknown or empty input. No probe or builder may raise.
- **Never mirror absolute size-dependent counts.** Mirror ratios and percentages (`-i` bytes-per-inode, `-m` reserved percent) so `-ddr` resizes stay valid. Never pass source inode counts, block counts, xfs log size, or xfs agcount.
- **Mirroring is on by default**; `-nfp` / `--no_fs_param_mirror` restores today's behavior exactly.
- **Every new external binary must be added to `_binCalled`** (hpcp.py:388) so `check_path()` resolves it and `--help` reports it as found/missing.
- **Version target:** `9.59` (hpcp.py:126). Bump only in the final task.
- Run tests with `python -m pytest tests/ -v` from the `hpcp/` directory.

---

## File Structure

| File | Responsibility |
|---|---|
| `hpcp.py` (modify) | All production code: probes, builders, the two registries, the `fs_params` key, `write_partition_info()` integration, `_run_mkfs_with_fallback()`, the `-nfp` flag |
| `tests/test_dd_fs_params.py` (create) | Parser/builder unit tests against recorded tool output, plus the root-only live round-trip test |
| `README.md` (modify) | Document `-nfp` and the mirroring behavior |

New code in `hpcp.py` is inserted as one contiguous section placed **immediately after `_FS_FIX_UNSUPPORTED`** (hpcp.py:727) and before `def fix_fs`, so the filesystem tables live together. `write_partition_info()` and `get_partition_details()` are edited in place.

---

### Task 1: CLI flag and module global

**Files:**
- Modify: `hpcp.py:138` (globals), `hpcp.py:3644` (`get_args`), `hpcp.py:3720` (`hpcp()` signature and globals), `hpcp.py:4124` (`main()` call site)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: nothing
- Produces: module global `MIRROR_FS_PARAMS` (bool, default `True`); `args.no_fs_param_mirror` (bool, default `False`); `hpcp(..., no_fs_param_mirror=False)` keyword

- [ ] **Step 1: Write the failing test**

Create `tests/test_dd_fs_params.py`:

```python
import os
import sys

import pytest

# Import sibling module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hpcp


def test_no_fs_param_mirror_flag_parses():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-nfp', '-dd', '/tmp/src.img', '/tmp/dest.img']
		args = hpcp.get_args()
		assert args.no_fs_param_mirror is True
	finally:
		sys.argv = old


def test_no_fs_param_mirror_default_off():
	old = sys.argv
	try:
		sys.argv = ['hpcp', '-dd', '/tmp/src.img', '/tmp/dest.img']
		args = hpcp.get_args()
		assert args.no_fs_param_mirror is False
	finally:
		sys.argv = old


def test_mirror_fs_params_global_defaults_true():
	assert hpcp.MIRROR_FS_PARAMS is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'no_fs_param_mirror'` and `AttributeError: module 'hpcp' has no attribute 'MIRROR_FS_PARAMS'`

- [ ] **Step 3: Add the global**

In `hpcp.py`, immediately after `CONTENT_ONLY = False` (line 138):

```python
MIRROR_FS_PARAMS = True
```

- [ ] **Step 4: Add the argparse flag**

In `get_args()`, immediately after the `-ddr` / `--dd_resize` argument (hpcp.py:3683):

```python
	parser.add_argument('-nfp','--no_fs_param_mirror', action='store_true', help='Do not mirror source filesystem parameters (geometry, features) in -dd mode. Create destination filesystems with mkfs defaults, preserving only label and UUID.')
```

- [ ] **Step 5: Thread it through `hpcp()`**

Add `no_fs_param_mirror = False` to the `hpcp()` signature, in the same keyword group as `content_only` (hpcp.py:3724):

```python
			exit_not_enough_space = False,do_not_remove_files_while_listing = False,no_fs_param_mirror = False):
```

Add the global declaration next to `global CONTENT_ONLY` (hpcp.py:3733):

```python
	global MIRROR_FS_PARAMS
```

And the assignment next to `CONTENT_ONLY = content_only` (hpcp.py:3736):

```python
	MIRROR_FS_PARAMS = not no_fs_param_mirror
```

- [ ] **Step 6: Pass it from `main()`**

In `main()`, extend the final argument line of the `hpcp(...)` call (hpcp.py:4132):

```python
			 exit_not_enough_space = args.exit_not_enough_space,do_not_remove_files_while_listing = args.do_not_remove_files_while_listing,
			 no_fs_param_mirror = args.no_fs_param_mirror)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): add -nfp/--no_fs_param_mirror flag and MIRROR_FS_PARAMS global"
```

---

### Task 2: Registry scaffolding and the `fs_params` key

Empty registries mean this task is behavior-neutral: `probe_fs_params()` returns `{}` for everything and `build_mkfs_params()` returns `[]`. Later tasks fill the tables.

**Files:**
- Modify: `hpcp.py:727` (insert new section after `_FS_FIX_UNSUPPORTED`), `hpcp.py:810` (`get_partition_details`)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `MIRROR_FS_PARAMS` from Task 1
- Produces:
  - `_FS_PARAM_PROBES: dict[str, Callable[[str], dict]]`
  - `_FS_MKFS_BUILDERS: dict[str, Callable[[dict], list]]`
  - `probe_fs_params(target_partition: str, fs_type: str) -> dict`
  - `build_mkfs_params(fs_type: str, fs_params: dict) -> list[str]`
  - `get_partition_details()` result gains key `'fs_params'` (dict, `{}` when unavailable)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`:

```python
def test_probe_fs_params_unknown_type_returns_empty():
	assert hpcp.probe_fs_params('/dev/null', 'no_such_fs') == {}


def test_probe_fs_params_never_raises_on_probe_error():
	def boom(device):
		raise RuntimeError('probe exploded')
	hpcp._FS_PARAM_PROBES['test_boom_fs'] = boom
	try:
		assert hpcp.probe_fs_params('/dev/null', 'test_boom_fs') == {}
	finally:
		del hpcp._FS_PARAM_PROBES['test_boom_fs']


def test_build_mkfs_params_unknown_type_returns_empty_list():
	assert hpcp.build_mkfs_params('no_such_fs', {'block_size': 4096}) == []


def test_build_mkfs_params_empty_params_returns_empty_list():
	assert hpcp.build_mkfs_params('ext4', {}) == []


def test_build_mkfs_params_never_raises_on_builder_error():
	def boom(params):
		raise RuntimeError('builder exploded')
	hpcp._FS_MKFS_BUILDERS['test_boom_fs'] = boom
	try:
		assert hpcp.build_mkfs_params('test_boom_fs', {'a': 1}) == []
	finally:
		del hpcp._FS_MKFS_BUILDERS['test_boom_fs']


def test_partition_details_dict_has_fs_params_key():
	# get_partition_details builds this key set; assert the contract without running sgdisk.
	import inspect
	src = inspect.getsource(hpcp.get_partition_details)
	assert "'fs_params'" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute 'probe_fs_params'`

- [ ] **Step 3: Add the registries and dispatch**

Insert in `hpcp.py` immediately after `_FS_FIX_UNSUPPORTED = {'udf', 'bfs'}` (line 727):

```python
#%% -- Filesystem Parameter Mirroring --
# Probe a source filesystem for the parameters mkfs would otherwise pick by
# default (geometry + features), so a dd-mode clone is not silently rebuilt
# with the building host's defaults. Registries are filled in below.
_FS_PARAM_PROBES = {}
_FS_MKFS_BUILDERS = {}

def probe_fs_params(target_partition, fs_type):
	"""
	Read the creation parameters of an existing filesystem.

	Args:
		target_partition (str): Path to the partition holding the filesystem.
		fs_type (str): Filesystem type as reported by blkid (e.g. 'ext4').

	Returns:
		dict: Opaque, filesystem-specific parameters for the matching builder in
			_FS_MKFS_BUILDERS. Empty dict when the type is unknown, the probe tool
			is missing, or parsing fails. This function never raises: a failed
			probe must degrade to default mkfs behaviour, not abort a copy.
	"""
	probe = _FS_PARAM_PROBES.get(fs_type)
	if not probe:
		return {}
	try:
		return probe(target_partition) or {}
	except Exception as e:
		eprint(f"FS param probe warning: Could not read {fs_type} parameters from {target_partition}: {e}")
		return {}

def build_mkfs_params(fs_type, fs_params):
	"""
	Translate probed filesystem parameters into mkfs arguments.

	Args:
		fs_type (str): Filesystem type as reported by blkid.
		fs_params (dict): The dict returned by probe_fs_params for the same type.

	Returns:
		list: Extra argv fragments to splice into the mkfs command, or an empty
			list when the type is unknown, params are empty, or building fails.
			This function never raises.
	"""
	if not fs_params:
		return []
	builder = _FS_MKFS_BUILDERS.get(fs_type)
	if not builder:
		return []
	try:
		return builder(fs_params) or []
	except Exception as e:
		eprint(f"FS param build warning: Could not build {fs_type} mkfs parameters: {e}")
		return []
```

- [ ] **Step 4: Populate `fs_params` in `get_partition_details`**

In `get_partition_details()`, add `'fs_params': {}` to the `rtnDic` initialiser (hpcp.py:839):

```python
	rtnDic = {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '', 'partition_attrs': '', 'fs_type': '', 'fs_uuid': '', 'fs_label': '', 'size': 0, 'fs_params': {}}
```

Then, in the same function, after the `for line in result:` loop that parses `blkid -o export` and **before** the `if loop_device:` detach block, add:

```python
	if rtnDic['fs_type']:
		rtnDic['fs_params'] = probe_fs_params(target_partition, rtnDic['fs_type'])
```

This runs while the loop device is still attached, so no extra attach/detach cycle is needed.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): add fs param probe/builder registries and fs_params partition key"
```

---

### Task 3: `mkfs` fallback helper

`run_command_in_multicmd_with_path_check()` returns output lines and discards the return code, so mirrored-argument failures cannot currently be detected. This helper adds that, using the same `return_object=True` pattern as `detach_loop_device()` (hpcp.py:3568).

**Files:**
- Modify: `hpcp.py` (new function in the filesystem-parameter section from Task 2)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `_FS_PARAM_PROBES` section location from Task 2
- Produces: `_run_mkfs_with_fallback(base_command: list, param_args: list, target_partition: str, fs_type: str) -> bool` — returns `True` if the filesystem was created (with or without mirrored params), `False` if both attempts failed

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`:

```python
class _FakeTask:
	def __init__(self, returncode, stderr=None):
		self.returncode = returncode
		self.stderr = stderr or []


def test_mkfs_fallback_uses_mirrored_params_when_they_succeed(monkeypatch):
	calls = []

	def fake_run(commands, **kwargs):
		calls.append(list(commands[0]))
		return [_FakeTask(0)]

	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	ok = hpcp._run_mkfs_with_fallback(['mkfs', '-t', 'ext4'], ['-b', '1024'], '/dev/fake1', 'ext4')
	assert ok is True
	assert len(calls) == 1
	assert calls[0] == ['mkfs', '-t', 'ext4', '-b', '1024', '/dev/fake1']


def test_mkfs_fallback_retries_without_params_on_failure(monkeypatch):
	calls = []

	def fake_run(commands, **kwargs):
		calls.append(list(commands[0]))
		# First attempt (with mirrored params) fails, second succeeds.
		return [_FakeTask(1, ['invalid block size'])] if len(calls) == 1 else [_FakeTask(0)]

	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	ok = hpcp._run_mkfs_with_fallback(['mkfs', '-t', 'ext4'], ['-b', '1024'], '/dev/fake1', 'ext4')
	assert ok is True
	assert len(calls) == 2
	assert calls[0] == ['mkfs', '-t', 'ext4', '-b', '1024', '/dev/fake1']
	assert calls[1] == ['mkfs', '-t', 'ext4', '/dev/fake1']


def test_mkfs_fallback_reports_failure_when_both_attempts_fail(monkeypatch):
	def fake_run(commands, **kwargs):
		return [_FakeTask(1, ['no such device'])]

	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	assert hpcp._run_mkfs_with_fallback(['mkfs', '-t', 'ext4'], ['-b', '1024'], '/dev/fake1', 'ext4') is False


def test_mkfs_fallback_single_attempt_when_no_params(monkeypatch):
	calls = []

	def fake_run(commands, **kwargs):
		calls.append(list(commands[0]))
		return [_FakeTask(0)]

	monkeypatch.setattr(hpcp.multiCMD, 'run_commands', fake_run)
	assert hpcp._run_mkfs_with_fallback(['mkfs.xfs'], [], '/dev/fake1', 'xfs') is True
	assert calls == [['mkfs.xfs', '/dev/fake1']]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_run_mkfs_with_fallback'`

- [ ] **Step 3: Write the helper**

Add to the filesystem-parameter section of `hpcp.py`:

```python
def _run_mkfs_with_fallback(base_command, param_args, target_partition, fs_type):
	"""
	Run mkfs with mirrored source parameters, falling back to defaults if rejected.

	Mirrored parameters can be legitimately invalid on a resized destination (a
	1 KiB block size on a filesystem grown past 16 TiB, FAT32 on a partition too
	small for it). A rejected parameter set must degrade to a default filesystem,
	not abort the copy.

	Args:
		base_command (list): The mkfs command without the target partition.
		param_args (list): Mirrored parameter arguments, possibly empty.
		target_partition (str): Partition to create the filesystem on.
		fs_type (str): Filesystem type, for messages.

	Returns:
		bool: True if the filesystem was created by either attempt.
	"""
	def _attempt(command):
		resolved = [_binPaths.get(command[0], command[0])] + list(command[1:])
		tasks = multiCMD.run_commands([resolved], timeout=COMMAND_TIMEOUT, max_threads=1, return_object=True)
		if not tasks:
			return 1, ''
		rc = getattr(tasks[0], 'returncode', 1)
		stderr = getattr(tasks[0], 'stderr', None) or []
		return rc, (stderr[-1].strip() if stderr else '')

	if param_args:
		rc, err = _attempt(list(base_command) + list(param_args) + [target_partition])
		if rc == 0:
			return True
		eprint(f"FS param warning: mkfs rejected mirrored {fs_type} parameters {' '.join(param_args)} on {target_partition}: {err}")
		eprint(f"FS param warning: Retrying with {fs_type} defaults. The destination filesystem will not match the source exactly.")
	rc, err = _attempt(list(base_command) + [target_partition])
	if rc != 0:
		eprint(f"Create fs error: Failed to create {fs_type} on {target_partition}: {err}")
		return False
	return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): add _run_mkfs_with_fallback for mirrored mkfs parameters"
```

---

### Task 4: Wire mirroring into `write_partition_info`

Still behavior-neutral — the registries are empty until Task 5 — but this is the integration point every later task depends on.

**Files:**
- Modify: `hpcp.py:926-1133` (`write_partition_info`)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `build_mkfs_params()` (Task 2), `_run_mkfs_with_fallback()` (Task 3), `MIRROR_FS_PARAMS` (Task 1)
- Produces: no new symbols; `write_partition_info()` now applies mirrored parameters to every `mkfs` branch

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`:

```python
def test_write_partition_info_applies_mirrored_params(monkeypatch):
	seen = {}

	def fake_target(image, partition_name):
		return '/dev/fake1', None

	def fake_run_cmd(command, **kwargs):
		return ['']

	def fake_mkfs(base_command, param_args, target_partition, fs_type):
		seen['base'] = list(base_command)
		seen['params'] = list(param_args)
		seen['target'] = target_partition
		return True

	monkeypatch.setattr(hpcp, 'get_target_partition', fake_target)
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', fake_run_cmd)
	monkeypatch.setattr(hpcp, '_run_mkfs_with_fallback', fake_mkfs)
	monkeypatch.setattr(hpcp, 'MIRROR_FS_PARAMS', True)
	monkeypatch.setitem(hpcp._FS_MKFS_BUILDERS, 'ext4', lambda p: ['-b', '1024'])

	infos = {'2': {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '',
				   'partition_attrs': '', 'fs_type': 'ext4', 'fs_uuid': '', 'fs_label': 'BOOTFS',
				   'size': 0, 'fs_params': {'block_size': 1024}}}
	hpcp.write_partition_info('/dev/fakeimg', infos, '2')

	assert seen['params'] == ['-b', '1024']
	assert seen['target'] == '/dev/fake1'
	assert seen['base'][:3] == ['mkfs', '-t', 'ext4']
	assert '/dev/fake1' not in seen['base']


def test_write_partition_info_skips_mirroring_when_disabled(monkeypatch):
	seen = {}

	def fake_mkfs(base_command, param_args, target_partition, fs_type):
		seen['params'] = list(param_args)
		return True

	monkeypatch.setattr(hpcp, 'get_target_partition', lambda image, name: ('/dev/fake1', None))
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', lambda command, **kwargs: [''])
	monkeypatch.setattr(hpcp, '_run_mkfs_with_fallback', fake_mkfs)
	monkeypatch.setattr(hpcp, 'MIRROR_FS_PARAMS', False)
	monkeypatch.setitem(hpcp._FS_MKFS_BUILDERS, 'ext4', lambda p: ['-b', '1024'])

	infos = {'2': {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '',
				   'partition_attrs': '', 'fs_type': 'ext4', 'fs_uuid': '', 'fs_label': '',
				   'size': 0, 'fs_params': {'block_size': 1024}}}
	hpcp.write_partition_info('/dev/fakeimg', infos, '2')

	assert seen['params'] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k write_partition_info -v`
Expected: FAIL — `_run_mkfs_with_fallback` is never called, so `seen` stays empty and the test raises `KeyError: 'params'`

- [ ] **Step 3: Compute the parameter arguments once**

In `write_partition_info()`, inside `if partition_infos[partition_name]['fs_type']:`, immediately after the three existing `fs_type` / `fs_label` / `fs_uuid` assignments (hpcp.py:973-975):

```python
			fs_params = partition_infos[partition_name].get('fs_params') or {}
			param_args = build_mkfs_params(fs_type, fs_params) if MIRROR_FS_PARAMS else []
			if param_args:
				print(f"Mirroring source {fs_type} parameters onto {target_partition}: {' '.join(param_args)}")
```

- [ ] **Step 4: Replace every mkfs invocation with the fallback helper**

In each branch of `write_partition_info()` that builds a `command` list and creates a filesystem, replace this shape:

```python
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
```

with:

```python
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
```

The helper appends `target_partition` itself, so the `command.append(target_partition)` line is removed in each case. Apply to every filesystem branch: `ext4/ext3/ext2`, `btrfs`, `xfs`, `f2fs`, `ntfs`, the FAT group, `exfat`, `hfsplus/hfs`, `udf`, `jfs`, `reiserfs`, `ufs`, `bfs`, `swap`, and the trailing generic `else` branch.

Two branches need extra care:

- **`minix`** currently builds `command = ['mkfs.minix',target_partition]` in one statement. Change it to `command = ['mkfs.minix']` followed by `_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)`.
- **`ufs`** builds `command = ['newfs', '-t']`. Keep the argument order — the helper appends only the target partition.

Leave the `zfs`, `cramfs`, `iso9660`, and `gpt` branches untouched; they do not create filesystems.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 15 passed

- [ ] **Step 6: Verify no behavior change with empty registries**

Run: `python3 hpcp.py -h | head -5`
Expected: help prints without error (module still imports and parses cleanly)

- [ ] **Step 7: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): apply mirrored fs parameters in write_partition_info"
```

---
### Task 5: ext2 / ext3 / ext4 probe and builder

**Files:**
- Modify: `hpcp.py` (filesystem-parameter section), `hpcp.py:388` (`_binCalled`)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `probe_fs_params()` / `build_mkfs_params()` dispatch (Task 2)
- Produces: `_probe_ext(device) -> dict` with keys `features` (list of str), `block_size`, `inode_size`, `inode_count`, `block_count`, `reserved_block_count` (all int); `_build_ext(params) -> list[str]`; registry entries for `ext2`, `ext3`, `ext4`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`. `_DUMPE2FS_OUTPUT` is real `dumpe2fs -h` output from an ext4 filesystem created with `mkfs.ext4 -b 1024 -I 128 -O ^metadata_csum,^64bit,^dir_index -m 0`:

```python
_DUMPE2FS_OUTPUT = """dumpe2fs 1.47.2 (1-Jan-2025)
Filesystem volume name:   BOOTFS
Last mounted on:          /tmp/tmp.BJpz2N4pR0
Filesystem UUID:          d0bac943-fd97-4462-bc04-0ba9a3097027
Filesystem magic number:  0xEF53
Filesystem revision #:    1 (dynamic)
Filesystem features:      has_journal ext_attr resize_inode orphan_file filetype extent flex_bg sparse_super large_file huge_file dir_nlink extra_isize
Filesystem flags:         signed_directory_hash
Default mount options:    user_xattr acl
Filesystem state:         clean
Inode count:              32768
Block count:              524288
Reserved block count:     0
Free blocks:              501204
Free inodes:              32754
First block:              1
Block size:               1024
Fragment size:            1024
Blocks per group:         8192
Inode size:               128
""".splitlines()


def test_probe_ext_parses_geometry_and_features(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DUMPE2FS_OUTPUT)
	params = hpcp._probe_ext('/dev/fake2')
	assert params['block_size'] == 1024
	assert params['inode_size'] == 128
	assert params['inode_count'] == 32768
	assert params['block_count'] == 524288
	assert params['reserved_block_count'] == 0
	assert 'has_journal' in params['features']
	assert '64bit' not in params['features']
	assert 'metadata_csum' not in params['features']


def test_build_ext_mirrors_geometry():
	args = hpcp._build_ext({'block_size': 1024, 'inode_size': 128})
	assert args[:4] == ['-b', '1024', '-I', '128']


def test_build_ext_negates_absent_curated_features():
	args = hpcp._build_ext({'features': ['has_journal', 'extent']})
	features = args[args.index('-O') + 1].split(',')
	# Present features are emitted plain...
	assert 'has_journal' in features
	assert 'extent' in features
	# ...and every curated feature the source lacks is explicitly negated,
	# because a plain -O list is merged with the mke2fs.conf defaults.
	assert '^64bit' in features
	assert '^metadata_csum' in features
	assert '^dir_index' in features


def test_build_ext_filters_runtime_state_features():
	args = hpcp._build_ext({'features': ['has_journal', 'needs_recovery', 'journal_dev']})
	features = args[args.index('-O') + 1].split(',')
	assert 'needs_recovery' not in features
	assert 'journal_dev' not in features
	assert '^needs_recovery' not in features


def test_build_ext_mirrors_inode_ratio_not_absolute_count():
	# 524288 blocks * 1024 bytes / 32768 inodes = 16384 bytes per inode
	args = hpcp._build_ext({'block_size': 1024, 'block_count': 524288, 'inode_count': 32768})
	assert args[args.index('-i') + 1] == '16384'
	assert '-N' not in args


def test_build_ext_mirrors_reserved_percentage():
	args = hpcp._build_ext({'block_count': 524288, 'reserved_block_count': 0})
	assert args[args.index('-m') + 1] == '0.00'
	args = hpcp._build_ext({'block_count': 200000, 'reserved_block_count': 10000})
	assert args[args.index('-m') + 1] == '5.00'


def test_ext_registered_for_all_three_types():
	for fs_type in ('ext2', 'ext3', 'ext4'):
		assert hpcp._FS_PARAM_PROBES[fs_type] is hpcp._probe_ext
		assert hpcp._FS_MKFS_BUILDERS[fs_type] is hpcp._build_ext
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k ext -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_probe_ext'`

- [ ] **Step 3: Write the probe and builder**

Add to the filesystem-parameter section of `hpcp.py`:

```python
# Features mke2fs may enable from mke2fs.conf regardless of the source. A plain
# -O list is *merged* with those defaults, so every curated feature the source
# lacks has to be negated explicitly or it silently comes back.
_EXT_CURATED_FEATURES = ('has_journal', 'ext_attr', 'resize_inode', 'dir_index', 'filetype', 'extent',
						 '64bit', 'flex_bg', 'metadata_csum', 'metadata_csum_seed', 'sparse_super',
						 'large_file', 'huge_file', 'dir_nlink', 'extra_isize', 'orphan_file',
						 'fast_commit', 'casefold', 'project', 'quota', 'verity', 'encrypt',
						 'bigalloc', 'inline_data', 'ea_inode', 'mmp', 'stable_inodes', 'uninit_bg',
						 'sparse_super2', 'meta_bg')
# Runtime state, not creation parameters. Never hand these to mke2fs.
_EXT_RUNTIME_FEATURES = {'needs_recovery', 'orphan_present', 'has_snapshot', 'journal_dev', 'shared_blocks'}

def _probe_ext(device):
	"""Read ext2/3/4 geometry and features from dumpe2fs."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['dumpe2fs', '-h', device], quiet=True):
		key, sep, value = line.partition(':')
		if not sep:
			continue
		key = key.strip().lower()
		value = value.strip()
		if key == 'filesystem features':
			params['features'] = value.split()
		elif key == 'block size':
			params['block_size'] = int(value)
		elif key == 'inode size':
			params['inode_size'] = int(value)
		elif key == 'inode count':
			params['inode_count'] = int(value)
		elif key == 'block count':
			params['block_count'] = int(value)
		elif key == 'reserved block count':
			params['reserved_block_count'] = int(value)
	return params

def _build_ext(params):
	"""Translate probed ext parameters into mke2fs arguments."""
	args = []
	if params.get('block_size'):
		args.extend(['-b', str(params['block_size'])])
	if params.get('inode_size'):
		args.extend(['-I', str(params['inode_size'])])
	src_features = [f for f in params.get('features', []) if f not in _EXT_RUNTIME_FEATURES]
	if src_features:
		feature_opts = list(src_features) + ['^' + f for f in _EXT_CURATED_FEATURES if f not in src_features]
		args.extend(['-O', ','.join(feature_opts)])
	block_count = params.get('block_count')
	block_size = params.get('block_size')
	inode_count = params.get('inode_count')
	if block_count and block_size and inode_count:
		# Mirror the inode ratio, never the absolute count: -ddr can resize the partition.
		args.extend(['-i', str(max(1024, (block_count * block_size) // inode_count))])
	reserved = params.get('reserved_block_count')
	if block_count and reserved is not None:
		args.extend(['-m', f"{reserved * 100.0 / block_count:.2f}"])
	return args

_FS_PARAM_PROBES.update({'ext2': _probe_ext, 'ext3': _probe_ext, 'ext4': _probe_ext})
_FS_MKFS_BUILDERS.update({'ext2': _build_ext, 'ext3': _build_ext, 'ext4': _build_ext})
```

- [ ] **Step 4: Register the probe binary**

Add `'dumpe2fs'` to the `_binCalled` set (hpcp.py:388), in the group with `tune2fs`:

```python
			  'tune2fs', 'dumpe2fs', 'xfs_admin', 'exfatlabel', 'udflabel', 'jfs_tune', 'reiserfstune', 'swaplabel'}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 22 passed

- [ ] **Step 6: Verify against a real filesystem**

```bash
truncate -s 512M /tmp/ext_probe.img
mkfs.ext4 -q -F -b 1024 -I 128 -O ^metadata_csum,^64bit,^dir_index -m 0 /tmp/ext_probe.img
python3 -c "
import hpcp
p = hpcp._probe_ext('/tmp/ext_probe.img')
print(p)
print(hpcp._build_ext(p))
"
rm -f /tmp/ext_probe.img
```
Expected: `block_size` 1024, `inode_size` 128, and a `-O` list containing `^64bit`, `^metadata_csum`, `^dir_index`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): mirror ext2/3/4 geometry, features, inode ratio and reserved percentage"
```

---

### Task 6: xfs probe and builder

**Files:**
- Modify: `hpcp.py` (filesystem-parameter section), `hpcp.py:388` (`_binCalled`)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `probe_fs_params()` / `build_mkfs_params()` dispatch (Task 2)
- Produces: `_probe_xfs(device) -> dict` — a dict of section name (`'meta-data'`, `'data'`, `'naming'`, `'log'`, `'realtime'`) to a dict of `key -> str value`; `_build_xfs(params) -> list[str]`; registry entry for `xfs`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`. `_XFS_INFO_OUTPUT` is real `xfs_info` output from a filesystem created with `mkfs.xfs -i size=1024 -n size=8192 -m reflink=0`:

```python
_XFS_INFO_OUTPUT = """meta-data=/dev/loop8p3           isize=1024   agcount=4, agsize=40127 blks
         =                       sectsz=512   attr=2, projid32bit=1
         =                       crc=1        finobt=1, sparse=1, rmapbt=1
         =                       reflink=0    bigtime=1 inobtcount=1 nrext64=1
         =                       exchange=0   metadir=0
data     =                       bsize=4096   blocks=160507, imaxpct=25
         =                       sunit=0      swidth=0 blks
naming   =version 2              bsize=8192   ascii-ci=0, ftype=1, parent=0
log      =internal log           bsize=4096   blocks=16384, version=2
         =                       sectsz=512   sunit=0 blks, lazy-count=1
realtime =none                   extsz=4096   blocks=0, rtextents=0
""".splitlines()


def test_probe_xfs_parses_sections(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _XFS_INFO_OUTPUT)
	params = hpcp._probe_xfs('/dev/fake3')
	assert params['meta-data']['isize'] == '1024'
	assert params['meta-data']['sectsz'] == '512'
	assert params['meta-data']['reflink'] == '0'
	assert params['meta-data']['crc'] == '1'
	# bsize appears in three sections; each must land in its own bucket.
	assert params['data']['bsize'] == '4096'
	assert params['naming']['bsize'] == '8192'
	assert params['log']['bsize'] == '4096'
	assert params['data']['imaxpct'] == '25'


def test_build_xfs_mirrors_geometry_and_features(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _XFS_INFO_OUTPUT)
	args = hpcp._build_xfs(hpcp._probe_xfs('/dev/fake3'))
	assert args[args.index('-b') + 1] == 'size=4096'
	assert args[args.index('-s') + 1] == 'size=512'
	assert args[args.index('-i') + 1] == 'size=1024,sparse=1,projid32bit=1,nrext64=1,maxpct=25'
	assert args[args.index('-n') + 1] == 'size=8192,ftype=1'
	assert args[args.index('-m') + 1] == 'crc=1,finobt=1,rmapbt=1,reflink=0,bigtime=1,inobtcount=1'


def test_build_xfs_omits_size_dependent_params(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _XFS_INFO_OUTPUT)
	args = hpcp._build_xfs(hpcp._probe_xfs('/dev/fake3'))
	joined = ' '.join(args)
	# Log size and agcount scale with filesystem size and must never be mirrored.
	assert 'agcount' not in joined
	assert 'logdev' not in joined
	assert '-l' not in args


def test_build_xfs_skips_missing_keys():
	args = hpcp._build_xfs({'data': {'bsize': '4096'}})
	assert args == ['-b', 'size=4096']


def test_xfs_registered():
	assert hpcp._FS_PARAM_PROBES['xfs'] is hpcp._probe_xfs
	assert hpcp._FS_MKFS_BUILDERS['xfs'] is hpcp._build_xfs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k xfs -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_probe_xfs'`

- [ ] **Step 3: Write the probe and builder**

Add to the filesystem-parameter section of `hpcp.py`:

```python
def _probe_xfs(device):
	"""
	Read xfs geometry and features from xfs_info.

	xfs_info groups values into sections (meta-data / data / naming / log /
	realtime) and reuses key names across them - `bsize` is the block size under
	`data` but the directory block size under `naming` - so values are bucketed
	by section rather than flattened.
	"""
	params = {}
	section = ''
	for line in run_command_in_multicmd_with_path_check(['xfs_info', device], quiet=True):
		if not line.strip():
			continue
		if not line[0].isspace():
			section = line.split('=')[0].strip()
		if not section:
			continue
		bucket = params.setdefault(section, {})
		for token in line.replace(',', ' ').split():
			key, sep, value = token.partition('=')
			if key and sep and value:
				bucket[key] = value
	return params

def _build_xfs(params):
	"""Translate probed xfs parameters into mkfs.xfs arguments."""
	meta = params.get('meta-data', {})
	data = params.get('data', {})
	naming = params.get('naming', {})
	args = []
	if data.get('bsize'):
		args.extend(['-b', 'size=' + data['bsize']])
	if meta.get('sectsz'):
		args.extend(['-s', 'size=' + meta['sectsz']])
	inode_opts = []
	if meta.get('isize'):
		inode_opts.append('size=' + meta['isize'])
	for key in ('sparse', 'projid32bit', 'nrext64'):
		if key in meta:
			inode_opts.append(f'{key}={meta[key]}')
	if data.get('imaxpct'):
		inode_opts.append('maxpct=' + data['imaxpct'])
	if inode_opts:
		args.extend(['-i', ','.join(inode_opts)])
	naming_opts = []
	if naming.get('bsize'):
		naming_opts.append('size=' + naming['bsize'])
	if 'ftype' in naming:
		naming_opts.append('ftype=' + naming['ftype'])
	if naming_opts:
		args.extend(['-n', ','.join(naming_opts)])
	meta_opts = [f'{key}={meta[key]}' for key in ('crc', 'finobt', 'rmapbt', 'reflink', 'bigtime', 'inobtcount') if key in meta]
	if meta_opts:
		args.extend(['-m', ','.join(meta_opts)])
	# Log size and agcount are deliberately not mirrored: both scale with fs size.
	return args

_FS_PARAM_PROBES['xfs'] = _probe_xfs
_FS_MKFS_BUILDERS['xfs'] = _build_xfs
```

- [ ] **Step 4: Register the probe binary**

Add `'xfs_info'` to `_binCalled` alongside `xfs_repair`:

```python
			  'e2fsck', 'btrfs', 'xfs_repair', 'xfs_info', 'fsck.f2fs', 'ntfsfix', 'fsck.fat', 'fsck.exfat', 'fsck.hfsplus',
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 27 passed

- [ ] **Step 6: Verify the built arguments are accepted by mkfs.xfs**

```bash
truncate -s 627M /tmp/xfs_probe.img
mkfs.xfs -q -f -i size=1024 -n size=8192 -m reflink=0 /tmp/xfs_probe.img
python3 -c "
import hpcp, subprocess
p = hpcp._probe_xfs('/tmp/xfs_probe.img')
args = hpcp._build_xfs(p)
print(args)
subprocess.run(['mkfs.xfs', '-q', '-f'] + args + ['/tmp/xfs_probe.img'], check=True)
"
xfs_db -r -c 'sb 0' -c p /tmp/xfs_probe.img | egrep '^inodesize|^dirblklog|^features_ro_compat'
rm -f /tmp/xfs_probe.img
```
Expected: `mkfs.xfs` exits 0; `inodesize = 1024` and `dirblklog = 1` are preserved.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): mirror xfs geometry and v5 feature flags"
```

---

### Task 7: btrfs probe and builder

**Files:**
- Modify: `hpcp.py` (filesystem-parameter section)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `probe_fs_params()` / `build_mkfs_params()` dispatch (Task 2)
- Produces: `_probe_btrfs(device) -> dict` with keys `nodesize` (int), `sectorsize` (int), `csum_type` (str), `incompat_flags` (list of str), `compat_ro_flags` (list of str); `_build_btrfs(params) -> list[str]`; registry entry for `btrfs`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`. `_BTRFS_SUPER_OUTPUT` is real `btrfs inspect-internal dump-super` output, abridged to the parsed fields:

```python
_BTRFS_SUPER_OUTPUT = """superblock: bytenr=65536, device=/dev/fake4
---------------------------------------------------------
csum_type		0 (crc32c)
csum_size		4
bytenr			65536
flags			0x1
			( WRITTEN )
magic			_BHRfS_M [match]
label			ROOTFS
sectorsize		4096
nodesize		4096
leafsize (deprecated)	4096
stripesize		4096
num_devices		1
compat_flags		0x0
compat_ro_flags		0x3
			( FREE_SPACE_TREE |
			  FREE_SPACE_TREE_VALID )
incompat_flags		0x341
			( MIXED_BACKREF |
			  EXTENDED_IREF |
			  SKINNY_METADATA |
			  NO_HOLES )
""".splitlines()


def test_probe_btrfs_parses_super(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _BTRFS_SUPER_OUTPUT)
	params = hpcp._probe_btrfs('/dev/fake4')
	assert params['nodesize'] == 4096
	assert params['sectorsize'] == 4096
	assert params['csum_type'] == 'crc32c'
	assert 'NO_HOLES' in params['incompat_flags']
	assert 'EXTENDED_IREF' in params['incompat_flags']
	assert 'FREE_SPACE_TREE' in params['compat_ro_flags']
	# The WRITTEN flag belongs to `flags`, not to the feature flag blocks.
	assert 'WRITTEN' not in params['incompat_flags']
	assert 'WRITTEN' not in params['compat_ro_flags']


def test_build_btrfs_mirrors_geometry_and_features(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _BTRFS_SUPER_OUTPUT)
	args = hpcp._build_btrfs(hpcp._probe_btrfs('/dev/fake4'))
	assert args[args.index('-n') + 1] == '4096'
	assert args[args.index('-s') + 1] == '4096'
	assert args[args.index('--csum') + 1] == 'crc32c'
	features = args[args.index('-O') + 1].split(',')
	assert 'no-holes' in features
	assert 'extref' in features
	assert 'skinny-metadata' in features
	assert 'free-space-tree' in features
	# Absent curated features are negated so mkfs defaults cannot reintroduce them.
	assert '^raid56' in features
	assert '^block-group-tree' in features
	# Unmappable / non-creation flags are dropped entirely.
	assert 'MIXED_BACKREF' not in features
	assert 'FREE_SPACE_TREE_VALID' not in features


def test_build_btrfs_negates_features_when_source_has_none_mappable():
	args = hpcp._build_btrfs({'incompat_flags': ['MIXED_BACKREF'], 'compat_ro_flags': []})
	features = args[args.index('-O') + 1].split(',')
	assert all(f.startswith('^') for f in features)


def test_btrfs_registered():
	assert hpcp._FS_PARAM_PROBES['btrfs'] is hpcp._probe_btrfs
	assert hpcp._FS_MKFS_BUILDERS['btrfs'] is hpcp._build_btrfs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k btrfs -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_probe_btrfs'`

- [ ] **Step 3: Write the probe and builder**

Add to the filesystem-parameter section of `hpcp.py`:

```python
# btrfs superblock flag names -> the feature names mkfs.btrfs -O accepts.
# Flags with no mkfs equivalent (MIXED_BACKREF, BIG_METADATA, DEFAULT_SUBVOL,
# FREE_SPACE_TREE_VALID, COMPRESS_*) are intentionally absent and get dropped.
_BTRFS_FLAG_TO_MKFS = {
	'MIXED_GROUPS':     'mixed-bg',
	'EXTENDED_IREF':    'extref',
	'SKINNY_METADATA':  'skinny-metadata',
	'NO_HOLES':         'no-holes',
	'RAID56':           'raid56',
	'RAID1C34':         'raid1c34',
	'ZONED':            'zoned',
	'SIMPLE_QUOTA':     'squota',
	'FREE_SPACE_TREE':  'free-space-tree',
	'BLOCK_GROUP_TREE': 'block-group-tree',
}
_BTRFS_CURATED_FEATURES = tuple(sorted(set(_BTRFS_FLAG_TO_MKFS.values())))

def _probe_btrfs(device):
	"""Read btrfs geometry, checksum type and feature flags from the superblock."""
	params = {'incompat_flags': [], 'compat_ro_flags': []}
	current = None
	for line in run_command_in_multicmd_with_path_check(['btrfs', 'inspect-internal', 'dump-super', device], quiet=True):
		stripped = line.strip()
		if not stripped:
			continue
		fields = stripped.split()
		name = fields[0]
		if name in ('nodesize', 'sectorsize') and len(fields) > 1:
			params[name] = int(fields[1])
			current = None
		elif name == 'csum_type' and len(fields) > 2:
			params['csum_type'] = fields[2].strip('()')
			current = None
		elif name in ('incompat_flags', 'compat_ro_flags'):
			current = name
		elif current:
			# Flag blocks look like "( MIXED_BACKREF |" ... "  NO_HOLES )".
			token = stripped.strip('()| \t')
			if token and token.replace('_', '').isalnum() and token.upper() == token:
				params[current].append(token)
			if stripped.endswith(')'):
				current = None
		else:
			current = None
	return params

def _build_btrfs(params):
	"""Translate probed btrfs parameters into mkfs.btrfs arguments."""
	args = []
	if params.get('nodesize'):
		args.extend(['-n', str(params['nodesize'])])
	if params.get('sectorsize'):
		args.extend(['-s', str(params['sectorsize'])])
	if params.get('csum_type'):
		args.extend(['--csum', params['csum_type']])
	flags = list(params.get('incompat_flags', [])) + list(params.get('compat_ro_flags', []))
	if flags:
		src_features = {_BTRFS_FLAG_TO_MKFS[f] for f in flags if f in _BTRFS_FLAG_TO_MKFS}
		feature_opts = sorted(src_features) + ['^' + f for f in _BTRFS_CURATED_FEATURES if f not in src_features]
		args.extend(['-O', ','.join(feature_opts)])
	return args

_FS_PARAM_PROBES['btrfs'] = _probe_btrfs
_FS_MKFS_BUILDERS['btrfs'] = _build_btrfs
```

`btrfs` is already in `_binCalled`; no change needed there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 31 passed

- [ ] **Step 5: Verify the built arguments are accepted by mkfs.btrfs**

```bash
truncate -s 300M /tmp/btrfs_probe.img
mkfs.btrfs -q -f -n 4096 -s 4096 /tmp/btrfs_probe.img
python3 -c "
import hpcp, subprocess
args = hpcp._build_btrfs(hpcp._probe_btrfs('/tmp/btrfs_probe.img'))
print(args)
subprocess.run(['mkfs.btrfs', '-q', '-f'] + args + ['/tmp/btrfs_probe.img'], check=True)
"
rm -f /tmp/btrfs_probe.img
```
Expected: `mkfs.btrfs` exits 0 with the full `-O` list including `^` negations.

- [ ] **Step 6: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): mirror btrfs nodesize, sectorsize, csum type and features"
```

---

### Task 8: FAT probe and builder — the ESP fix

`blkid -o export` reports `TYPE=vfat` for every FAT width, so the `fat32` / `fat16` / `fat12` branches in `write_partition_info()` never fire and `mkfs.vfat` picks the width from the partition size. A 260 MiB FAT32 EFI System Partition is rebuilt as FAT16, which some firmware cannot boot. The width comes from `blkid -p`'s `VERSION` field instead.

**Files:**
- Modify: `hpcp.py` (filesystem-parameter section), `hpcp.py:1019-1031` (remove the dead width branches)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `probe_fs_params()` / `build_mkfs_params()` dispatch (Task 2)
- Produces: `_probe_vfat(device) -> dict` with keys `fat_bits`, `sector_size`, `cluster_size`, `reserved_sectors`, `fat_count` (all int); `_build_vfat(params) -> list[str]`; registry entries for `vfat`, `fat`, `fat12`, `fat16`, `fat32`, `msdos`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`. Both fixtures are real output from a FAT32 filesystem on a 260 MiB partition:

```python
_BLKID_P_FAT32_OUTPUT = """DEVNAME=/dev/loop8p1
LABEL_FATBOOT=ESP
LABEL=ESP
UUID=A4D7-1D90
VERSION=FAT32
FSBLOCKSIZE=512
BLOCK_SIZE=512
TYPE=vfat
USAGE=filesystem
PART_ENTRY_TYPE=c12a7328-f81f-11d2-ba4b-00a0c93ec93b
""".splitlines()

_FSCK_FAT_OUTPUT = """fsck.fat 4.2 (2021-01-31)
Checking we can access the last sector of the filesystem
Boot sector contents:
System ID "mkfs.fat"
Media byte 0xf8 (hard disk)
       512 bytes per logical sector
       512 bytes per cluster
        32 reserved sectors
First FAT starts at byte 16384 (sector 32)
         2 FATs, 32 bit entries
   2097152 bytes per FAT (= 4096 sectors)
""".splitlines()


def _fake_fat_runner(command, **kwargs):
	return _BLKID_P_FAT32_OUTPUT if command[0] == 'blkid' else _FSCK_FAT_OUTPUT


def test_probe_vfat_reads_fat_width_and_geometry(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', _fake_fat_runner)
	params = hpcp._probe_vfat('/dev/fake1')
	assert params['fat_bits'] == 32
	assert params['sector_size'] == 512
	assert params['cluster_size'] == 512
	assert params['reserved_sectors'] == 32
	assert params['fat_count'] == 2


def test_build_vfat_forces_source_fat_width(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check', _fake_fat_runner)
	args = hpcp._build_vfat(hpcp._probe_vfat('/dev/fake1'))
	# Without -F 32 mkfs.vfat picks FAT16 for a partition this size.
	assert args[args.index('-F') + 1] == '32'
	assert args[args.index('-S') + 1] == '512'
	assert args[args.index('-s') + 1] == '1'
	assert args[args.index('-f') + 1] == '2'
	assert args[args.index('-R') + 1] == '32'


def test_build_vfat_computes_sectors_per_cluster():
	args = hpcp._build_vfat({'sector_size': 512, 'cluster_size': 8192})
	assert args[args.index('-s') + 1] == '16'


def test_build_vfat_skips_bad_fat_width():
	assert '-F' not in hpcp._build_vfat({'fat_bits': 64, 'sector_size': 512})


def test_vfat_registered_for_all_fat_aliases():
	for fs_type in ('vfat', 'fat', 'fat12', 'fat16', 'fat32', 'msdos'):
		assert hpcp._FS_PARAM_PROBES[fs_type] is hpcp._probe_vfat
		assert hpcp._FS_MKFS_BUILDERS[fs_type] is hpcp._build_vfat


def test_dead_fat_width_branches_removed():
	# The old -F 16 / -F 12 branches keyed on blkid TYPE could never fire and
	# would now conflict with the probed width.
	import inspect
	src = inspect.getsource(hpcp.write_partition_info)
	assert "'-F', '16'" not in src
	assert "'-F', '12'" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k vfat -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_probe_vfat'`

- [ ] **Step 3: Write the probe and builder**

Add to the filesystem-parameter section of `hpcp.py`:

```python
def _probe_vfat(device):
	"""
	Read FAT width and geometry.

	blkid -o export reports TYPE=vfat for FAT12, FAT16 and FAT32 alike, so the
	width has to come from blkid -p's VERSION field. Without it mkfs.vfat picks
	the width from the partition size and can turn a FAT32 EFI System Partition
	into FAT16 that firmware refuses to boot.
	"""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'export', device], quiet=True):
		key, sep, value = line.partition('=')
		if sep and key.strip() == 'VERSION':
			version = value.strip().upper()
			if version.startswith('FAT') and version[3:].isdigit():
				params['fat_bits'] = int(version[3:])
	for line in run_command_in_multicmd_with_path_check(['fsck.fat', '-nv', device], quiet=True):
		fields = line.split()
		if not fields or not fields[0].isdigit():
			continue
		if 'bytes per logical sector' in line:
			params['sector_size'] = int(fields[0])
		elif 'bytes per cluster' in line:
			params['cluster_size'] = int(fields[0])
		elif 'reserved sectors' in line:
			params['reserved_sectors'] = int(fields[0])
		elif 'FATs,' in line and 'bit entries' in line:
			params['fat_count'] = int(fields[0])
			params.setdefault('fat_bits', int(fields[2]))
	return params

def _build_vfat(params):
	"""Translate probed FAT parameters into mkfs.vfat arguments."""
	args = []
	if params.get('fat_bits') in (12, 16, 32):
		args.extend(['-F', str(params['fat_bits'])])
	sector_size = params.get('sector_size')
	if sector_size:
		args.extend(['-S', str(sector_size)])
		cluster_size = params.get('cluster_size')
		if cluster_size and cluster_size >= sector_size:
			args.extend(['-s', str(cluster_size // sector_size)])
	if params.get('fat_count'):
		args.extend(['-f', str(params['fat_count'])])
	if params.get('reserved_sectors'):
		args.extend(['-R', str(params['reserved_sectors'])])
	return args

for _fat_alias in ('vfat', 'fat', 'fat12', 'fat16', 'fat32', 'msdos'):
	_FS_PARAM_PROBES[_fat_alias] = _probe_vfat
	_FS_MKFS_BUILDERS[_fat_alias] = _build_vfat
del _fat_alias
```

- [ ] **Step 4: Remove the dead width branches**

In `write_partition_info()`, delete these four lines from the FAT branch (hpcp.py:1021-1024):

```python
				if fs_type == 'fat16':
					command.extend(['-F', '16'])
				elif fs_type == 'fat12':
					command.extend(['-F', '12'])
```

The width now always comes from `param_args`, and leaving these would risk passing `-F` twice with conflicting values.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 37 passed

- [ ] **Step 6: Verify the FAT32 round trip**

```bash
truncate -s 260M /tmp/fat_probe.img
mkfs.vfat -F 32 -S 512 -n ESP /tmp/fat_probe.img
python3 -c "
import hpcp, subprocess
args = hpcp._build_vfat(hpcp._probe_vfat('/tmp/fat_probe.img'))
print(args)
subprocess.run(['mkfs.vfat'] + args + ['/tmp/fat_probe.img'], check=True)
"
blkid -p -o export /tmp/fat_probe.img | grep VERSION
rm -f /tmp/fat_probe.img
```
Expected: `VERSION=FAT32` — without the fix, a plain `mkfs.vfat` on this size yields FAT16.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "fix(dd): mirror FAT width and geometry, fixing FAT32 ESP rebuilt as FAT16"
```

---
### Task 9: ntfs, exfat and f2fs probes and builders

**Files:**
- Modify: `hpcp.py` (filesystem-parameter section), `hpcp.py:388` (`_binCalled`)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `probe_fs_params()` / `build_mkfs_params()` dispatch (Task 2)
- Produces: `_probe_ntfs`/`_build_ntfs`, `_probe_exfat`/`_build_exfat`, `_probe_f2fs`/`_build_f2fs`; registry entries for `ntfs`, `exfat`, `f2fs`. Keys: ntfs `cluster_size`, `sector_size` (int); exfat `sector_size`, `cluster_size` (int); f2fs `log_sectorsize`, `segs_per_sec`, `secs_per_zone` (int) and `features` (list of str)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`. All three fixtures are real tool output:

```python
_NTFSINFO_OUTPUT = """Volume Information
	Name of device: /dev/fake5
	Device state: 11
	Volume Name: NT
	Volume State: 1
	Volume Version: 3.1
	Sector Size: 512
	Cluster Size: 8192
	Index Block Size: 4096
	Volume Size in Clusters: 25599
""".splitlines()

_DUMP_EXFAT_OUTPUT = """exfatprogs version : 1.2.9
-------------- Dump Boot sector region --------------
Volume Length(sectors):                  409600
FAT Offset(sector offset):               2048
FAT Length(sectors):                     13
Cluster Heap Offset (sector offset):     4096
Cluster Count:                           1584
Root Cluster (cluster offset):           4
Volume Serial:                           0xebe5bdee
Bytes per Sector:                        512
Sectors per Cluster:                     256

---------------- Show the statistics ----------------
Cluster size:                            131072
""".splitlines()

_DUMP_F2FS_OUTPUT = """Info: Debug level = 1
Info: superblock features = 4 : extra_attr
Info: superblock encrypt level = 0, salt = 00000000000000000000000000000000
magic                         		[0xf2f52010 : 4076150800]
major_ver                     		[0x       1 : 1]
volum_name                    		[F2]
log_sectorsize                		[0x       9 : 9]
log_sectors_per_block         		[0x       3 : 3]
log_blocksize                 		[0x       c : 12]
log_blocks_per_seg            		[0x       9 : 9]
segs_per_sec                  		[0x       2 : 2]
secs_per_zone                 		[0x       1 : 1]
block_count                   		[0x   12c00 : 76800]
""".splitlines()


def test_probe_ntfs_reads_cluster_and_sector_size(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _NTFSINFO_OUTPUT)
	params = hpcp._probe_ntfs('/dev/fake5')
	assert params['cluster_size'] == 8192
	assert params['sector_size'] == 512


def test_build_ntfs():
	args = hpcp._build_ntfs({'cluster_size': 8192, 'sector_size': 512})
	assert args == ['-c', '8192', '-s', '512']


def test_probe_exfat_reads_geometry(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DUMP_EXFAT_OUTPUT)
	params = hpcp._probe_exfat('/dev/fake6')
	assert params['sector_size'] == 512
	assert params['cluster_size'] == 131072


def test_build_exfat():
	args = hpcp._build_exfat({'sector_size': 512, 'cluster_size': 131072})
	assert args == ['-s', '512', '-c', '131072']


def test_probe_f2fs_reads_geometry_and_features(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DUMP_F2FS_OUTPUT)
	params = hpcp._probe_f2fs('/dev/fake7')
	assert params['log_sectorsize'] == 9
	assert params['segs_per_sec'] == 2
	assert params['secs_per_zone'] == 1
	assert params['features'] == ['extra_attr']


def test_build_f2fs_converts_log_sector_size(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DUMP_F2FS_OUTPUT)
	args = hpcp._build_f2fs(hpcp._probe_f2fs('/dev/fake7'))
	assert args[args.index('-w') + 1] == '512'
	assert args[args.index('-s') + 1] == '2'
	assert args[args.index('-z') + 1] == '1'
	assert args[args.index('-O') + 1] == 'extra_attr'


def test_probe_f2fs_handles_no_features(monkeypatch):
	no_features = ['Info: superblock features = 0 : ', 'log_sectorsize                \t\t[0x       9 : 9]']
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: no_features)
	params = hpcp._probe_f2fs('/dev/fake7')
	assert params['features'] == []
	assert '-O' not in hpcp._build_f2fs(params)


def test_ntfs_exfat_f2fs_registered():
	for fs_type, probe, builder in (('ntfs', hpcp._probe_ntfs, hpcp._build_ntfs),
									('exfat', hpcp._probe_exfat, hpcp._build_exfat),
									('f2fs', hpcp._probe_f2fs, hpcp._build_f2fs)):
		assert hpcp._FS_PARAM_PROBES[fs_type] is probe
		assert hpcp._FS_MKFS_BUILDERS[fs_type] is builder
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k "ntfs or exfat or f2fs" -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_probe_ntfs'`

- [ ] **Step 3: Write the probes and builders**

Add to the filesystem-parameter section of `hpcp.py`:

```python
def _probe_ntfs(device):
	"""Read ntfs cluster and sector size from ntfsinfo."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['ntfsinfo', '-m', device], quiet=True):
		key, sep, value = line.partition(':')
		if not sep:
			continue
		key = key.strip().lower()
		value = value.strip()
		if key == 'cluster size' and value.isdigit():
			params['cluster_size'] = int(value)
		elif key == 'sector size' and value.isdigit():
			params['sector_size'] = int(value)
	return params

def _build_ntfs(params):
	"""Translate probed ntfs parameters into mkfs.ntfs arguments."""
	args = []
	if params.get('cluster_size'):
		args.extend(['-c', str(params['cluster_size'])])
	if params.get('sector_size'):
		args.extend(['-s', str(params['sector_size'])])
	return args

def _probe_exfat(device):
	"""Read exfat sector and cluster size from dump.exfat."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['dump.exfat', device], quiet=True):
		key, sep, value = line.partition(':')
		if not sep:
			continue
		key = key.strip().lower()
		value = value.strip()
		if key == 'bytes per sector' and value.isdigit():
			params['sector_size'] = int(value)
		elif key == 'cluster size' and value.isdigit():
			params['cluster_size'] = int(value)
	return params

def _build_exfat(params):
	"""Translate probed exfat parameters into mkfs.exfat arguments."""
	args = []
	if params.get('sector_size'):
		args.extend(['-s', str(params['sector_size'])])
	if params.get('cluster_size'):
		args.extend(['-c', str(params['cluster_size'])])
	return args

# mkfs.f2fs enables no features by default and has no negation syntax, so the
# source's feature list is applied additively rather than as a delta.
def _probe_f2fs(device):
	"""Read f2fs geometry and features from dump.f2fs."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['dump.f2fs', '-d', '1', device], quiet=True):
		if 'superblock features' in line:
			names = line.rpartition(':')[2]
			params['features'] = [n.strip() for n in names.replace(',', ' ').split() if n.strip()]
			continue
		fields = line.split()
		if not fields or '[' not in line or ':' not in line:
			continue
		if fields[0] not in ('log_sectorsize', 'segs_per_sec', 'secs_per_zone'):
			continue
		value = line.rpartition(':')[2].strip().rstrip(']').strip()
		if value.isdigit():
			params[fields[0]] = int(value)
	return params

def _build_f2fs(params):
	"""Translate probed f2fs parameters into mkfs.f2fs arguments."""
	args = []
	if params.get('log_sectorsize'):
		args.extend(['-w', str(1 << params['log_sectorsize'])])
	if params.get('segs_per_sec'):
		args.extend(['-s', str(params['segs_per_sec'])])
	if params.get('secs_per_zone'):
		args.extend(['-z', str(params['secs_per_zone'])])
	if params.get('features'):
		args.extend(['-O', ','.join(params['features'])])
	return args

_FS_PARAM_PROBES.update({'ntfs': _probe_ntfs, 'exfat': _probe_exfat, 'f2fs': _probe_f2fs})
_FS_MKFS_BUILDERS.update({'ntfs': _build_ntfs, 'exfat': _build_exfat, 'f2fs': _build_f2fs})
```

- [ ] **Step 4: Register the probe binaries**

Add `'ntfsinfo'`, `'dump.exfat'`, and `'dump.f2fs'` to `_binCalled`:

```python
			  'e2fsck', 'btrfs', 'xfs_repair', 'xfs_info', 'fsck.f2fs', 'dump.f2fs', 'ntfsfix', 'ntfsinfo',
			  'fsck.fat', 'fsck.exfat', 'dump.exfat', 'fsck.hfsplus',
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 45 passed

- [ ] **Step 6: Verify the built arguments are accepted**

```bash
truncate -s 200M /tmp/ex.img
mkfs.exfat -c 128K /tmp/ex.img >/dev/null
python3 -c "
import hpcp, subprocess
args = hpcp._build_exfat(hpcp._probe_exfat('/tmp/ex.img'))
print(args)
subprocess.run(['mkfs.exfat'] + args + ['/tmp/ex.img'], check=True)
"
rm -f /tmp/ex.img
```
Expected: exits 0 with `['-s', '512', '-c', '131072']`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): mirror ntfs, exfat and f2fs geometry and features"
```

---

### Task 10: udf and reiserfs probes and builders

**Files:**
- Modify: `hpcp.py` (filesystem-parameter section), `hpcp.py:388` (`_binCalled`)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `probe_fs_params()` / `build_mkfs_params()` dispatch (Task 2)
- Produces: `_probe_udf`/`_build_udf` (keys `block_size` int, `udfrev` str), `_probe_reiserfs`/`_build_reiserfs` (keys `block_size` int, `fs_format` str, `hash_function` str); registry entries for `udf` and `reiserfs`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`:

```python
_UDFINFO_OUTPUT = """filename=/dev/fake8
label=UD
uuid=6aa09dfa158beac7
lvid=UD
vid=UD
blocksize=512
udfrev=2.01
integrity=closed
blocks=409600
usedblocks=103
""".splitlines()

_DEBUGREISERFS_OUTPUT = """debugreiserfs 3.6.27
Filesystem state: consistent
Reiserfs super block in block 16 on 0x0 of format 3.6 with standard journal
Count of blocks on the device: 76800
Blocksize: 4096
Hash function used to sort names: "r5"
sb_version: 2
""".splitlines()


def test_probe_udf_reads_blocksize_and_revision(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _UDFINFO_OUTPUT)
	params = hpcp._probe_udf('/dev/fake8')
	assert params['block_size'] == 512
	assert params['udfrev'] == '2.01'


def test_build_udf():
	args = hpcp._build_udf({'block_size': 512, 'udfrev': '2.01'})
	assert args == ['--blocksize=512', '--udfrev=2.01']


def test_probe_reiserfs_reads_blocksize_format_and_hash(monkeypatch):
	monkeypatch.setattr(hpcp, 'run_command_in_multicmd_with_path_check',
						lambda command, **kwargs: _DEBUGREISERFS_OUTPUT)
	params = hpcp._probe_reiserfs('/dev/fake9')
	assert params['block_size'] == 4096
	assert params['fs_format'] == '3.6'
	assert params['hash_function'] == 'r5'


def test_build_reiserfs():
	args = hpcp._build_reiserfs({'block_size': 4096, 'fs_format': '3.6', 'hash_function': 'r5'})
	assert args == ['-b', '4096', '--format', '3.6', '-h', 'r5']


def test_udf_reiserfs_registered():
	assert hpcp._FS_PARAM_PROBES['udf'] is hpcp._probe_udf
	assert hpcp._FS_MKFS_BUILDERS['udf'] is hpcp._build_udf
	assert hpcp._FS_PARAM_PROBES['reiserfs'] is hpcp._probe_reiserfs
	assert hpcp._FS_MKFS_BUILDERS['reiserfs'] is hpcp._build_reiserfs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k "udf or reiserfs" -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_probe_udf'`

- [ ] **Step 3: Write the probes and builders**

Add to the filesystem-parameter section of `hpcp.py`:

```python
def _probe_udf(device):
	"""Read udf block size and revision from udfinfo."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['udfinfo', device], quiet=True):
		key, sep, value = line.partition('=')
		if not sep:
			continue
		key = key.strip()
		value = value.strip()
		if key == 'blocksize' and value.isdigit():
			params['block_size'] = int(value)
		elif key == 'udfrev' and value:
			params['udfrev'] = value
	return params

def _build_udf(params):
	"""Translate probed udf parameters into mkudffs arguments."""
	args = []
	if params.get('block_size'):
		args.append('--blocksize=' + str(params['block_size']))
	if params.get('udfrev'):
		args.append('--udfrev=' + params['udfrev'])
	return args

def _probe_reiserfs(device):
	"""Read reiserfs block size, on-disk format and hash from debugreiserfs."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['debugreiserfs', device], quiet=True):
		stripped = line.strip()
		if stripped.lower().startswith('blocksize:'):
			value = stripped.partition(':')[2].strip()
			if value.isdigit():
				params['block_size'] = int(value)
		elif 'of format' in stripped:
			fields = stripped.split('of format')[1].split()
			if fields:
				params['fs_format'] = fields[0]
		elif 'hash function' in stripped.lower() and '"' in stripped:
			params['hash_function'] = stripped.split('"')[1]
	return params

def _build_reiserfs(params):
	"""Translate probed reiserfs parameters into mkfs.reiserfs arguments."""
	args = []
	if params.get('block_size'):
		args.extend(['-b', str(params['block_size'])])
	if params.get('fs_format'):
		args.extend(['--format', params['fs_format']])
	if params.get('hash_function'):
		args.extend(['-h', params['hash_function']])
	return args

_FS_PARAM_PROBES.update({'udf': _probe_udf, 'reiserfs': _probe_reiserfs})
_FS_MKFS_BUILDERS.update({'udf': _build_udf, 'reiserfs': _build_reiserfs})

# Deliberately not registered, with reasons:
#   jfs   - mkfs.jfs exposes no geometry options; block size is fixed at 4096.
#   minix - see _build_minix; mkfs.minix has no block size option.
#   bfs, ufs, swap - mkfs/newfs/mkswap expose nothing worth mirroring, and the
#                    on-disk headers carry no creation parameters we can act on.
#   zfs, cramfs, iso9660 - hpcp already declines to create these.
# Unregistered types fall through probe_fs_params/build_mkfs_params to {} / [],
# which is exactly today's default-mkfs behaviour.
```

- [ ] **Step 4: Register the probe binaries**

Add `'udfinfo'` and `'debugreiserfs'` to `_binCalled`, in the group with the other tuning tools:

```python
			  'tune2fs', 'dumpe2fs', 'xfs_admin', 'exfatlabel', 'udflabel', 'udfinfo', 'jfs_tune',
			  'reiserfstune', 'debugreiserfs', 'swaplabel'}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 50 passed

- [ ] **Step 6: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): mirror udf and reiserfs creation parameters"
```

---

### Task 11: hfsplus and minix superblock readers

These two have no probe tool that reports creation geometry, but both superblock layouts are stable and documented. Reading them directly with `struct` is more reliable than parsing an `fsck` transcript and needs no extra binary.

**Files:**
- Modify: `hpcp.py:22` (add `import struct`), `hpcp.py` (filesystem-parameter section)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: `probe_fs_params()` / `build_mkfs_params()` dispatch (Task 2)
- Produces: `_probe_hfsplus(device) -> dict` (key `block_size` int), `_build_hfsplus`; `_probe_minix(device) -> dict` (keys `fs_version` int, `name_length` int), `_build_minix`; registry entries for `hfsplus`, `hfs`, `minix`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`. The fixtures are synthesised superblocks matching the verified on-disk offsets:

```python
import struct
import tempfile


def _write_superblock(path, offset, payload):
	with open(path, 'wb') as f:
		f.write(b'\0' * offset)
		f.write(payload)
		f.write(b'\0' * 512)


def test_probe_hfsplus_reads_block_size():
	# HFS+ volume header lives at byte 1024: 'H+' signature, blockSize at +40 big-endian.
	header = bytearray(64)
	struct.pack_into('>H', header, 0, 0x482B)
	struct.pack_into('>H', header, 2, 4)
	struct.pack_into('>I', header, 40, 8192)
	with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
		path = tmp.name
	try:
		_write_superblock(path, 1024, bytes(header))
		assert hpcp._probe_hfsplus(path) == {'block_size': 8192}
	finally:
		os.unlink(path)


def test_probe_hfsplus_rejects_foreign_signature():
	with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
		path = tmp.name
	try:
		_write_superblock(path, 1024, b'\0' * 64)
		assert hpcp._probe_hfsplus(path) == {}
	finally:
		os.unlink(path)


def test_build_hfsplus():
	assert hpcp._build_hfsplus({'block_size': 8192}) == ['-b', '8192']


def test_probe_minix_detects_version_and_name_length():
	# v1/v2 magic sits at 1024+16; v3 magic at 1024+24.
	cases = ((0x137F, 16, 1, 14), (0x138F, 16, 1, 30), (0x2468, 16, 2, 14),
			 (0x2478, 16, 2, 30), (0x4D5A, 24, 3, 60))
	for magic, offset, expected_version, expected_namelen in cases:
		sb = bytearray(64)
		struct.pack_into('<H', sb, offset, magic)
		with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
			path = tmp.name
		try:
			_write_superblock(path, 1024, bytes(sb))
			params = hpcp._probe_minix(path)
			assert params['fs_version'] == expected_version
			assert params['name_length'] == expected_namelen
		finally:
			os.unlink(path)


def test_build_minix_versions():
	assert hpcp._build_minix({'fs_version': 3, 'name_length': 60}) == ['-3']
	assert hpcp._build_minix({'fs_version': 2, 'name_length': 30}) == ['-2', '-n', '30']
	assert hpcp._build_minix({'fs_version': 1, 'name_length': 14}) == ['-1', '-n', '14']


def test_hfsplus_minix_registered():
	assert hpcp._FS_PARAM_PROBES['hfsplus'] is hpcp._probe_hfsplus
	assert hpcp._FS_PARAM_PROBES['hfs'] is hpcp._probe_hfsplus
	assert hpcp._FS_MKFS_BUILDERS['hfsplus'] is hpcp._build_hfsplus
	assert hpcp._FS_PARAM_PROBES['minix'] is hpcp._probe_minix
	assert hpcp._FS_MKFS_BUILDERS['minix'] is hpcp._build_minix
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k "hfsplus or minix" -v`
Expected: FAIL — `AttributeError: module 'hpcp' has no attribute '_probe_hfsplus'`

- [ ] **Step 3: Add the `struct` import**

In `hpcp.py`, add to the stdlib import block so it stays alphabetical (between `stat` on line 21 and `sys` on line 22):

```python
import struct
```

- [ ] **Step 4: Write the probes and builders**

Add to the filesystem-parameter section of `hpcp.py`:

```python
# minix superblock magic -> (version, max filename length). v1/v2 keep the magic
# at byte 16 of the superblock, v3 at byte 24.
_MINIX_MAGICS = {0x137F: (1, 14), 0x138F: (1, 30), 0x2468: (2, 14), 0x2478: (2, 30)}
_MINIX3_MAGIC = 0x4D5A

def _probe_hfsplus(device):
	"""
	Read the hfs+ allocation block size straight from the volume header.

	No fsck.hfsplus output reports it, but the volume header is at a fixed
	offset: signature at byte 1024 ('H+' / 'HX'), blockSize as a big-endian
	uint32 40 bytes into the header.
	"""
	with open(device, 'rb') as f:
		f.seek(1024)
		header = f.read(64)
	if len(header) < 44:
		return {}
	signature = struct.unpack_from('>H', header, 0)[0]
	if signature not in (0x482B, 0x4858):
		return {}
	block_size = struct.unpack_from('>I', header, 40)[0]
	return {'block_size': block_size} if block_size else {}

def _build_hfsplus(params):
	"""Translate probed hfs+ parameters into mkfs.hfsplus arguments."""
	if params.get('block_size'):
		return ['-b', str(params['block_size'])]
	return []

def _probe_minix(device):
	"""Read the minix filesystem version and filename length from the superblock."""
	with open(device, 'rb') as f:
		f.seek(1024)
		superblock = f.read(64)
	if len(superblock) < 32:
		return {}
	magic = struct.unpack_from('<H', superblock, 16)[0]
	if magic in _MINIX_MAGICS:
		version, name_length = _MINIX_MAGICS[magic]
		return {'fs_version': version, 'name_length': name_length}
	if struct.unpack_from('<H', superblock, 24)[0] == _MINIX3_MAGIC:
		return {'fs_version': 3, 'name_length': 60}
	return {}

def _build_minix(params):
	"""
	Translate probed minix parameters into mkfs.minix arguments.

	mkfs.minix exposes no block size option, so a v3 block size cannot be
	mirrored; version and filename length are what it accepts.
	"""
	args = []
	version = params.get('fs_version')
	if version in (1, 2, 3):
		args.append(f'-{version}')
	if version in (1, 2) and params.get('name_length') in (14, 30):
		args.extend(['-n', str(params['name_length'])])
	return args

_FS_PARAM_PROBES.update({'hfsplus': _probe_hfsplus, 'hfs': _probe_hfsplus, 'minix': _probe_minix})
_FS_MKFS_BUILDERS.update({'hfsplus': _build_hfsplus, 'hfs': _build_hfsplus, 'minix': _build_minix})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dd_fs_params.py -v`
Expected: 56 passed

- [ ] **Step 6: Verify against real filesystems**

```bash
truncate -s 200M /tmp/hfs.img && mkfs.hfsplus -b 8192 -v HF /tmp/hfs.img >/dev/null
truncate -s 50M /tmp/mx.img && mkfs.minix -3 /tmp/mx.img >/dev/null
python3 -c "
import hpcp
print(hpcp._probe_hfsplus('/tmp/hfs.img'), hpcp._build_hfsplus(hpcp._probe_hfsplus('/tmp/hfs.img')))
print(hpcp._probe_minix('/tmp/mx.img'), hpcp._build_minix(hpcp._probe_minix('/tmp/mx.img')))
"
rm -f /tmp/hfs.img /tmp/mx.img
```
Expected: `{'block_size': 8192} ['-b', '8192']` and `{'fs_version': 3, 'name_length': 60} ['-3']`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py
git commit -m "feat(dd): mirror hfs+ block size and minix version via superblock reads"
```

---

### Task 12: Live round-trip integration test

**Files:**
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: everything from Tasks 1-11
- Produces: no production symbols; a root-only test that proves an end-to-end dd copy preserves source parameters

- [ ] **Step 1: Write the test**

Append to `tests/test_dd_fs_params.py`:

```python
import shutil
import subprocess

_ROUNDTRIP_TOOLS = ('losetup', 'sgdisk', 'mkfs.vfat', 'mkfs.ext4', 'mkfs.xfs',
					'blkid', 'dumpe2fs', 'xfs_info', 'fsck.fat', 'truncate')

requires_root_and_tools = pytest.mark.skipif(
	os.geteuid() != 0 or any(shutil.which(t) is None for t in _ROUNDTRIP_TOOLS),
	reason='dd round-trip test needs root and losetup/sgdisk/mkfs tools')


def _run(*command):
	return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def _partition_params(image, index, fs_type):
	loop = _run('losetup', '--partscan', '--find', '--show', '--read-only', image).strip()
	try:
		subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
		return hpcp.probe_fs_params(f'{loop}p{index}', fs_type)
	finally:
		subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)


@requires_root_and_tools
def test_dd_roundtrip_preserves_source_fs_params(tmp_path):
	src = str(tmp_path / 'src.img')
	dest = str(tmp_path / 'dest.img')
	_run('truncate', '-s', '1400M', src)
	_run('sgdisk', '--clear',
		 '--new=1:0:+260M', '--typecode=1:ef00', '--change-name=1:EFI System',
		 '--new=2:0:+512M', '--typecode=2:8300', '--change-name=2:boot',
		 '--new=3:0:0', '--typecode=3:8300', '--change-name=3:root', src)

	loop = _run('losetup', '--partscan', '--find', '--show', src).strip()
	try:
		subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
		# Deliberately non-default: a FAT32 ESP small enough that mkfs.vfat would
		# otherwise pick FAT16, an ext4 built the way an older distro would, and
		# an xfs with non-default inode and directory geometry.
		_run('mkfs.vfat', '-F', '32', '-n', 'ESP', f'{loop}p1')
		_run('mkfs.ext4', '-q', '-F', '-b', '1024', '-I', '128',
			 '-O', '^metadata_csum,^64bit,^dir_index', '-m', '0', '-L', 'BOOTFS', f'{loop}p2')
		_run('mkfs.xfs', '-q', '-f', '-i', 'size=1024', '-n', 'size=8192',
			 '-m', 'reflink=0', '-L', 'ROOTFS', f'{loop}p3')
		for index in (1, 2, 3):
			mount_point = str(tmp_path / f'mnt{index}')
			os.makedirs(mount_point, exist_ok=True)
			_run('mount', f'{loop}p{index}', mount_point)
			try:
				os.makedirs(os.path.join(mount_point, 'dir'), exist_ok=True)
				with open(os.path.join(mount_point, 'dir', f'file{index}.txt'), 'w') as f:
					f.write(f'hello-{index}\n')
			finally:
				_run('umount', mount_point)
	finally:
		subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)

	src_fat = _partition_params(src, 1, 'vfat')
	src_ext = _partition_params(src, 2, 'ext4')
	src_xfs = _partition_params(src, 3, 'xfs')

	hpcp_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')
	open(dest, 'wb').close()
	subprocess.run([sys.executable, hpcp_py, '-dd', src, dest],
				   stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900)

	dest_fat = _partition_params(dest, 1, 'vfat')
	dest_ext = _partition_params(dest, 2, 'ext4')
	dest_xfs = _partition_params(dest, 3, 'xfs')

	# The ESP must stay FAT32; plain mkfs.vfat picks FAT16 at this size.
	assert dest_fat['fat_bits'] == src_fat['fat_bits'] == 32
	assert dest_fat['cluster_size'] == src_fat['cluster_size']

	assert dest_ext['block_size'] == src_ext['block_size'] == 1024
	assert dest_ext['inode_size'] == src_ext['inode_size'] == 128
	assert dest_ext['reserved_block_count'] == 0
	assert sorted(dest_ext['features']) == sorted(src_ext['features'])
	for absent in ('64bit', 'metadata_csum', 'dir_index'):
		assert absent not in dest_ext['features']

	assert dest_xfs['meta-data']['isize'] == src_xfs['meta-data']['isize'] == '1024'
	assert dest_xfs['naming']['bsize'] == src_xfs['naming']['bsize'] == '8192'
	assert dest_xfs['meta-data']['reflink'] == src_xfs['meta-data']['reflink'] == '0'


@requires_root_and_tools
def test_dd_roundtrip_uses_defaults_with_no_fs_param_mirror(tmp_path):
	src = str(tmp_path / 'src.img')
	dest = str(tmp_path / 'dest.img')
	_run('truncate', '-s', '700M', src)
	_run('sgdisk', '--clear', '--new=1:0:0', '--typecode=1:8300', '--change-name=1:root', src)

	loop = _run('losetup', '--partscan', '--find', '--show', src).strip()
	try:
		subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
		_run('mkfs.ext4', '-q', '-F', '-b', '1024', '-I', '128', '-L', 'ROOTFS', f'{loop}p1')
	finally:
		subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)

	hpcp_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpcp.py')
	open(dest, 'wb').close()
	subprocess.run([sys.executable, hpcp_py, '-dd', '-nfp', src, dest],
				   stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900)

	dest_ext = _partition_params(dest, 1, 'ext4')
	# -nfp restores today's behaviour: mkfs defaults, not the source's 1 KiB blocks.
	assert dest_ext['block_size'] == 4096
```

- [ ] **Step 2: Run the round-trip test**

Run: `python -m pytest tests/test_dd_fs_params.py -k roundtrip -v`
Expected (as root, tools present): 2 passed. Without root: 2 skipped.

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass, including the pre-existing `tests/test_content_only.py`

- [ ] **Step 4: Commit**

```bash
git add tests/test_dd_fs_params.py
git commit -m "test(dd): add live round-trip test for fs parameter mirroring"
```

---

### Task 13: Documentation and version bump

**Files:**
- Modify: `README.md`, `hpcp.py:126` (version)
- Test: `tests/test_dd_fs_params.py`

**Interfaces:**
- Consumes: everything from Tasks 1-12
- Produces: no new symbols

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dd_fs_params.py`:

```python
def test_version_bumped():
	assert hpcp.version == '9.59'
	assert hpcp.__version__ == hpcp.version
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dd_fs_params.py -k version -v`
Expected: FAIL — `assert '9.58' == '9.59'`

- [ ] **Step 3: Bump the version**

In `hpcp.py:126`:

```python
version = '9.59'
```

- [ ] **Step 4: Document the behavior in README.md**

Add to the flag listing, next to the other `-dd` options:

```markdown
- `-nfp`, `--no_fs_param_mirror` — Do not mirror source filesystem parameters (geometry, features)
  in `-dd` mode. Destination filesystems are created with `mkfs` defaults, preserving only label
  and UUID.
```

And add a short subsection under the disk-dump documentation:

```markdown
### Filesystem parameter mirroring

In `-dd` mode `hpcp` reads each source filesystem's creation parameters and recreates the
destination with the same geometry and feature set, instead of whatever the local `mkfs` defaults
happen to be. This covers ext2/3/4 (block size, inode size, inode ratio, reserved percentage, full
feature set), xfs (block/sector/inode/directory geometry and v5 feature flags), btrfs (nodesize,
sectorsize, checksum type, features), FAT (width, sector and cluster size, FAT count, reserved
sectors), ntfs, exfat, f2fs, udf, reiserfs, hfs+, and minix.

Without it, a clone built on a modern host can be unmountable or unbootable on the system it came
from — for example a FAT32 EFI System Partition recreated as FAT16, or an ext4 `/boot` that gains
`metadata_csum` and `64bit` that its bootloader does not understand.

Size-dependent values are scaled rather than copied, so `-ddr` resizes stay valid: the ext inode
*ratio* and reserved *percentage* are mirrored, never absolute counts. If `mkfs` rejects a mirrored
parameter set, `hpcp` warns and retries with defaults rather than failing the copy. Use `-nfp` to
turn mirroring off entirely.
```

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 6: Verify the CLI end to end**

Run: `python3 hpcp.py -V && python3 hpcp.py -h | grep -A2 no_fs_param_mirror`
Expected: prints `9.59` and the `-nfp` help text

- [ ] **Step 7: Commit**

```bash
git add tests/test_dd_fs_params.py hpcp.py README.md
git commit -m "docs(dd): document fs parameter mirroring and bump hpcp to 9.59"
```

---

## Notes for the executor

- **Do not release.** This plan stops at the version bump. Publishing to PyPI (`uvpack`) and bumping
  the `hpcp==` pin in `cas-toolbox/pyproject.toml` are separate steps the maintainer decides on,
  per the weekly `cas-toolbox` cadence in `CLAUDE.md`.
- **Loop device hygiene.** The round-trip test attaches loop devices. If a test run is interrupted,
  check `losetup -a` and detach leftovers before re-running.
- **A pre-existing, unrelated bug was observed** while verifying this issue: after a dd run the
  read-only *source* loop device stays attached even though `clean_up()` reports detaching the
  destination loop. It is out of scope here — do not fix it as part of this plan.
