# Exact Disk Values, Refusals Before Hashing, Single Probes (group B2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Disk patches and `-dd` copy mode read and write filesystem labels, GPT partition names and exFAT serials exactly; find partitions on cciss-style disks and through by-id paths; refuse before hashing whenever a refusal needs no hash; and probe each disk once per layout read. Release target: hpcp 9.70.

**Architecture:** all production code stays in `hpcp.py`.
- **Exact values:** the identity probe `_probe_identity` switches to `blkid -p -o udev` with an exact decoder (`_udev_decode`). `get_partition_details`, shared by disk patches and `-dd` mode, uses it and parses GPT names between the first and the last quote.
- **Device paths:** a `_dev_path` helper turns kernel names into `/dev` paths for the old lsblk-based helpers.
- **Refusals before hashing:** `apply_disk_patch` and `process_get_disk_diff_image` are reordered so cheap refusals run before any hash.
- **Single probes:** `read_disk_layout` gains `pttype=` and `fs_params=` and reads the GPT without per-partition blkid. It then probes each partition once on its settled device.

**Tech Stack:** Python 3.6+ (single file, tabs), pytest, util-linux `blkid -o udev`, sgdisk, lsblk, exfatprogs, loop devices in root tests.

**Spec:** `docs/superpowers/specs/2026-10-11-disk-values-and-probes-design.md` (approved, commit 9e2e782; Part 2 item 2 corrected in 0f17dc5)

## Global Constraints

- Single file: all production code goes in `hpcp.py`; no new modules or packages.
- Tabs for indentation, in `hpcp.py` and in tests.
- Python 3.6 syntax floor: no walrus, no `str.removeprefix`, no `dict | dict`, no `f"{x=}"`, no `namedtuple(..., defaults=)`.
- No new runtime dependencies.
- `eprint("Category: detail")` only for real failures; warnings and notes use `print(..., file=sys.stderr)` / `print(...)`.
- Patch formats are unchanged (manifest, `changes.tsv`, `disk.tsv`). Patches from 9.65–9.69 apply as before.
- `-dd` copy mode: ordinary ASCII labels and names behave byte-for-byte as before.
- Root / loop tests: scratch files only under the test's `tmp_path`; never write to real filesystem paths; devices only `/dev/loop*` backed by scratch images; leave nothing mounted or attached (unmount / detach in `finally`).
- **Probe safety rule:**
  - Any ad-hoc root probe goes in a script file under `/tmp/claude-0` and runs with `bash <file>`. In this harness, `set -euo pipefail` typed at the top of a shell command is ignored; it only works in a script file or a `bash -c` child.
  - Every path variable gets a `: "${VAR:?}"` guard and a prefix check (`case "$D" in /tmp/claude-0/*) ;; *) exit 1;; esac`).
  - `md5sum /usr/lib/os-release` must stay `776b0aac9b17d49a0ac879da9801f51b`.
- Version target 9.70, bumped only in the final task.
- Run tests from the repository root: `python -m pytest tests/ -q`. At the start of this plan this gives 644 passed, 1 skipped (Windows) and 1 xfailed (the exFAT UUID case, which Task 1 turns into a pass).
- Commit messages end with the Co-Authored-By trailer the implementing agent's harness specifies.

## Review Focus

1. A label containing a backslash, a label that is not valid UTF-8, and an empty label must all round-trip exactly. Covered by Task 1's `_udev_decode` / `_probe_identity` unit tests.
2. A GPT name that is empty, all spaces, or has several apostrophes must be read exactly. Covered by Task 1's `get_partition_details` unit test.
3. The reordering must not lose a refusal or weaken one, and nothing may be mounted or written before a refusal that used to come first. Covered by Task 3's one-test-per-refusal spies and Task 4's tests.
4. `-dd` mode on ordinary ASCII values must be unchanged. Covered by the existing dd round-trip tests, which stay green, plus Task 1's dd test.
5. The settle reduction must not bring back the udev re-read race 9.67 fixed (a loop device attached, then probed at once). Covered by Task 5's root test, which reads a freshly attached loop disk ten times in a row.

---

### Task 1: Exact labels, GPT names and exFAT serials

**Files:**
- Modify: `hpcp.py`:
  - add `_udev_decode` just above `def _probe_identity` (≈ 6129);
  - replace `_probe_identity`'s body;
  - `get_partition_details` (≈ 1680-1748): the name parse and the blkid parse;
  - `_fs_identity_commands` exfat branch (≈ 6300).
- Test: `tests/test_disk_patch.py`, `tests/test_dd_fs_params.py`

**Interfaces:**
- Produces:
  - `_udev_decode(value: str) -> str`;
  - `_probe_identity(device: str) -> (fs_type, fs_uuid, fs_label)`, which now reads `blkid -p -o udev`;
  - `get_partition_details` returns the exact `partition_name`, `fs_type`, `fs_uuid` and `fs_label`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
def test_udev_decode_restores_the_exact_bytes(hpcp_mod):
	"""Review Focus 1: blkid -o udev writes unsafe bytes as \\xNN (a space, a backslash, bytes that are not UTF-8);
	multiCMD decodes undecodable output bytes to the same escape. Both decode to the original bytes."""
	dec = hpcp_mod._udev_decode
	assert dec('röot\\x20=x') == 'röot =x'
	assert dec('a\\x5cb') == 'a\\b'
	assert dec('x\\x5cx41') == 'x\\x41'
	assert dec('a\\xffb') == 'a\udcffb'
	assert dec('') == ''
	assert dec('plain') == 'plain'


def test_probe_identity_reads_the_encoded_udev_values(hpcp_mod, reset_hpcp_globals, monkeypatch):
	output = ['ID_FS_UUID=9134403a-3b3f', 'ID_FS_UUID_ENC=9134403a-3b3f', 'ID_FS_UUID_SUB=sub',
		'ID_FS_LABEL=röot_=x', 'ID_FS_LABEL_ENC=röot\\x20=x', 'ID_FS_LABEL_FATBOOT=BOOT', 'ID_FS_TYPE=ext4',
		'ID_FS_USAGE=filesystem']
	calls = []

	def fake(argv, **kwargs):
		calls.append(argv)
		return output

	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake)
	assert hpcp_mod._probe_identity('/dev/x1') == ('ext4', '9134403a-3b3f', 'röot =x')
	assert calls == [['blkid', '-p', '-o', 'udev', '/dev/x1']]
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', lambda argv, **kwargs: [])
	assert hpcp_mod._probe_identity('/dev/x2') == ('', '', '')


@pytest.mark.parametrize('line, name', [
	("Partition name: 'it's a  name '", "it's a  name "),
	("Partition name: ''", ''),
	("Partition name: '   '", '   '),
	("Partition name: 'a'b'c'", "a'b'c"),
	("Partition name: 'Ünïcode ✓'", 'Ünïcode ✓'),
])
def test_get_partition_details_reads_gpt_names_exactly(hpcp_mod, reset_hpcp_globals, monkeypatch, line, name):
	"""Review Focus 2: sgdisk prints the name raw between quotes (a quote inside it included)."""
	sgdisk_output = ['Partition GUID code: 0FC63DAF-8483-4772-8E79-3D69D8477DE4 (Linux filesystem)',
		'Partition unique GUID: 11111111-1111-1111-1111-111111111111', line,
		'Attribute flags: 0000000000000000', 'Partition size: 2048 sectors (1.0 MiB)']

	def fake(argv, **kwargs):
		if argv[0] == 'sgdisk':
			return sgdisk_output
		if argv[0] == 'blkid':
			return ['ID_FS_TYPE=ext4', 'ID_FS_LABEL_ENC=a\\x3db']
		return ['']

	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake)
	monkeypatch.setattr(hpcp_mod, 'get_target_partition', lambda device, partition: ('/dev/x1', None))
	monkeypatch.setattr(hpcp_mod, 'MIRROR_FS_PARAMS', False)
	hpcp_mod.get_partition_details.cache_clear()
	try:
		details = hpcp_mod.get_partition_details('/dev/x', '1')
	finally:
		hpcp_mod.get_partition_details.cache_clear()
	assert details['partition_name'] == name
	assert (details['fs_type'], details['fs_label']) == ('ext4', 'a=b')


def test_exfat_serial_is_passed_in_exfatlabel_form(hpcp_mod):
	"""exfatlabel -i rejects blkid's XXXX-XXXX form ("invalid serial number") and takes 0xXXXXXXXX."""
	assert hpcp_mod._fs_identity_commands('exfat', '/dev/x1', uuid='ABCD-1234') == [['exfatlabel', '-i', '/dev/x1', '0xABCD1234']]
```

In `tests/test_disk_patch.py`, `test_fs_identity_commands_match_dd_mode_uuid_commands` expects `cmd('exfat', dev, uuid=u) == [['exfatlabel', '-i', dev, u]]`. Change it to `[['exfatlabel', '-i', dev, '0x1234ABCD']]` (u is `'1234-ABCD'`). In `_IDENTITY_TOOL_CASES` (tests/test_disk_patch.py), remove the `pytest.mark.xfail` from the exFAT UUID case, so it must pass.

In `tests/test_dd_fs_params.py`, `test_get_partition_details_skips_probe_when_mirroring_disabled` fakes `blkid_output = ['TYPE=vfat']`. Change it to `blkid_output = ['ID_FS_TYPE=vfat']`.

Append to `tests/test_disk_patch.py` (root round trip through a disk patch):

```python
_AWKWARD_NAME_1 = "it's a  name "
_AWKWARD_NAME_2 = 'Ünïcode ✓'
_AWKWARD_LABEL = 'röot =x'
_HAS_EXFAT = shutil.which('mkfs.exfat') is not None and shutil.which('exfatlabel') is not None


def _make_identity_disk(path, names, label, exfat_serial=None):
	"""64 MiB GPT disk: p1 vfat (name names[0]), p2 ext4 (name names[1], label label), p3 exFAT (when the tools
	exist; serial exfat_serial, label 'dàta')."""
	with open(str(path), 'wb') as f:
		f.truncate(64 << 20)
	argv = ['sgdisk', '-n1:2048:+16M', '-t1:EF00', f'-c1:{names[0]}', '-n2:0:+24M', '-t2:8300', f'-c2:{names[1]}']
	if _HAS_EXFAT:
		argv += ['-n3:0:+16M', '-t3:0700', '-c3:data']
	_run(*(argv + [path]))
	with _attached(path) as loop:
		_run('mkfs.vfat', loop + 'p1')
		_run('mkfs.ext4', '-q', '-F', '-L', label, loop + 'p2')
		if _HAS_EXFAT:
			_run('mkfs.exfat', '-L', 'dàta', loop + 'p3')
			if exfat_serial:
				_run('exfatlabel', '-i', loop + 'p3', exfat_serial)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_disk_patch_writes_awkward_labels_names_and_exfat_serials_exactly(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Spec Part 1: a golden whose GPT names hold an apostrophe, double and trailing spaces and non-ASCII
	characters, whose ext4 label is non-ASCII with a space and '=', and whose exFAT serial differs: the node ends
	with exactly those values."""
	base, golden = tmp_path / 'base.img', tmp_path / 'golden.img'
	_make_identity_disk(base, ('ESP', 'root'), 'ROOT', exfat_serial='0x11112222')
	shutil.copyfile(str(base), str(golden))
	_run('sgdisk', f'-c1:{_AWKWARD_NAME_1}', f'-c2:{_AWKWARD_NAME_2}', golden)
	with _attached(golden) as loop:
		_run('e2label', loop + 'p2', _AWKWARD_LABEL)
		if _HAS_EXFAT:
			_run('exfatlabel', '-i', loop + 'p3', '0xABCD1234')
	patch = str(tmp_path / 'patch.img')
	with _session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_disk_diff_image([str(golden)], [str(base)], patch, mount_points, loop_devices, max_workers=1) == patch
	target = tmp_path / 'node.img'
	shutil.copyfile(str(base), str(target))
	summary = _apply(hpcp_mod, patch, target)
	assert summary['failed'] == 0
	with _attached(target, read_only=True) as loop:
		names = [subprocess.run(['sgdisk', '-i', str(number), loop], capture_output=True, text=True).stdout for number in (1, 2)]
		assert f"Partition name: '{_AWKWARD_NAME_1}'" in names[0] and f"Partition name: '{_AWKWARD_NAME_2}'" in names[1]
		assert hpcp_mod._probe_identity(loop + 'p2')[2] == _AWKWARD_LABEL
		if _HAS_EXFAT:
			assert hpcp_mod._probe_identity(loop + 'p3')[1] == 'ABCD-1234'
```

Append to `tests/test_dd_fs_params.py` (root round trip through `-dd`):

```python
@requires_root_and_tools
def test_dd_copies_awkward_labels_and_gpt_names_exactly(tmp_path):
	"""Spec Part 1 for -dd: a source whose GPT names hold an apostrophe, double and trailing spaces and non-ASCII
	characters, and whose ext4 label is non-ASCII with a space and '=', is copied with exactly those values."""
	name1, name2, label = "it's a  name ", 'Ünïcode ✓', 'röot =x'
	src = str(tmp_path / 'src.img')
	_run('truncate', '-s', '200M', src)
	_run('sgdisk', '--clear', '--new=1:0:+64M', '--typecode=1:ef00', f'--change-name=1:{name1}',
		'--new=2:0:0', '--typecode=2:8300', f'--change-name=2:{name2}', src)
	loop = _run('losetup', '--partscan', '--find', '--show', src).strip()
	try:
		subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
		_run('mkfs.vfat', '-F', '32', '-n', 'ESP', f'{loop}p1')
		_run('mkfs.ext4', '-q', '-F', '-L', label, f'{loop}p2')
	finally:
		subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)
	dest = str(tmp_path / 'dest.img')
	try:
		_run_hpcp_dd([], src, dest)
		loop = _run('losetup', '--partscan', '--find', '--show', '--read-only', dest).strip()
		try:
			subprocess.run(['udevadm', 'settle'], check=False, capture_output=True)
			assert f"Partition name: '{name1}'" in _run('sgdisk', '-i', '1', loop)
			assert f"Partition name: '{name2}'" in _run('sgdisk', '-i', '2', loop)
			assert _run('e2label', f'{loop}p2').rstrip('\n') == label
		finally:
			subprocess.run(['losetup', '-d', loop], check=False, capture_output=True)
	finally:
		_detach_loops_for_image(src)
		_detach_loops_for_image(dest)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py -q -k "udev_decode or encoded_udev or gpt_names_exactly or exfat_serial or dd_mode_uuid or awkward or identity_tools or skips_probe_when_mirroring"`
Expected: FAIL.
- `_udev_decode` is missing.
- `_probe_identity` still runs `-o export`.
- The GPT names are truncated (`it`).
- The exFAT argv is `ABCD-1234`, so the exFAT tool case now fails instead of xfailing.
- The root round trips end with mangled or truncated values.

- [ ] **Step 3: Implement**

Add just above `def _probe_identity`:

```python
_UDEV_HEX_ESCAPE_RE = re.compile(rb'\\x([0-9a-fA-F]{2})')

def _udev_decode(value):
	"""A blkid `-o udev` *_ENC value as text, with the original bytes: blkid writes unsafe bytes (a space, a
	backslash, bytes that are not UTF-8) as \\xNN and the rest as UTF-8, and multiCMD decodes undecodable output
	bytes to the same escape. Bytes that are not UTF-8 are kept with surrogateescape."""
	data = value.encode('utf-8', 'surrogateescape')
	data = _UDEV_HEX_ESCAPE_RE.sub(lambda match: bytes([int(match.group(1), 16)]), data)
	return data.decode('utf-8', 'surrogateescape')
```

Replace `_probe_identity` with:

```python
def _probe_identity(device):
	"""(fs_type, fs_uuid, fs_label) of device from a low-level `blkid -p -o udev` probe, read exactly ('' when
	nothing is found).

	-o udev gives encoded forms (ID_FS_UUID_ENC, ID_FS_LABEL_ENC; see _udev_decode). -o export prints bytes above
	127 in M- notation in every locale, so a label such as 'röot' would read back mangled. No blkid cache, so a
	label or UUID this run just changed reads back as it is on disk; exact keys, so UUID_SUB (btrfs) and
	LABEL_FATBOOT (FAT) never stand in for UUID and LABEL.
	"""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'udev', device], quiet=True)
	finally:
		# blkid exits 2 when it finds nothing (BIOS boot and other raw partitions).
		del ERRORS[error_count:]
	values = {}
	for line in output:
		key, sep, value = line.rstrip('\r\n').partition('=')
		if sep:
			values[key] = value
	uuid = values.get('ID_FS_UUID_ENC', values.get('ID_FS_UUID', ''))
	return values.get('ID_FS_TYPE', ''), _udev_decode(uuid), _udev_decode(values.get('ID_FS_LABEL_ENC', ''))
```

In `get_partition_details`, replace the partition-name branch:

```python
		elif "partition name:" in line.lower():
			# sgdisk prints the name raw between quotes, a quote inside it included: everything between the first
			# and the last quote, spaces kept.
			first, last = line.find("'"), line.rfind("'")
			rtnDic['partition_name'] = line[first + 1:last] if last > first else ''
```

and replace the block from `target_partition, loop_device = get_target_partition(device, partition)` through the `for line in result:` loop that fills `fs_type`, `fs_uuid` and `fs_label` (it ends just before `if rtnDic['fs_type'] and MIRROR_FS_PARAMS:`) with:

```python
	target_partition, loop_device = get_target_partition(device, partition)
	# The same exact, uncached probe disk patches use (_probe_identity): -o export mangled non-ASCII labels, and
	# 'LABEL' in line also matched LABEL_FATBOOT and cut a label at its first '='.
	if target_partition:
		rtnDic['fs_type'], rtnDic['fs_uuid'], rtnDic['fs_label'] = _probe_identity(target_partition)
```

In `_fs_identity_commands`, change the exfat UUID line to:

```python
		elif fs_type == 'exfat':
			# exfatlabel -i takes the serial as 0xXXXXXXXX; blkid shows it as XXXX-XXXX ("invalid serial number").
			commands.append(['exfatlabel', '-i', device, '0x' + uuid.replace('-', '')])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command: PASS.
Run the two root round-trip tests three times in a row: `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py -q -k awkward`.
Then run `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py tests/test_diff_image.py -q`: all pass. Nothing should be left attached: `losetup -a | grep -c /tmp/pytest` prints 0.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py tests/test_dd_fs_params.py
git commit -m "fix(dd-diff, dd): read labels, GPT names and exFAT serials exactly"
```

---

### Task 2: Partition device paths from kernel names

**Files:**
- Modify: `hpcp.py`:
  - add `_dev_path` just above `def get_largest_partition` (≈ 827);
  - `get_largest_partition`, `get_partitions`, `get_target_partition` (≈ 1670-1677);
  - `_partition_devices` (≈ 6063), which reuses `_dev_path`.
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Produces: `_dev_path(kernel_name: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
@pytest.mark.parametrize('disk, lsblk, partitions, largest', [
	('/dev/cciss/c0d0', ['cciss!c0d0 1000', 'cciss!c0d0p1 400', 'cciss!c0d0p2 600'],
		['/dev/cciss/c0d0p1', '/dev/cciss/c0d0p2'], '/dev/cciss/c0d0p2'),
	('/dev/disk/by-id/ata-DISK', ['sda 1000', 'sda1 400', 'sda2 600'], ['/dev/sda1', '/dev/sda2'], '/dev/sda2'),
	('/dev/loop7', ['loop7 1000', 'loop7p1 900'], ['/dev/loop7p1'], '/dev/loop7p1'),
])
def test_partition_paths_come_from_kernel_names(hpcp_mod, reset_hpcp_globals, monkeypatch, disk, lsblk, partitions, largest):
	"""A '!' in a kernel name is a '/' under /dev (cciss), and a disk given by a by-id link still has its
	partitions under /dev, not next to the link."""
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', lambda argv, **kwargs: list(lsblk))
	assert hpcp_mod.get_partitions(disk) == partitions
	assert hpcp_mod.get_largest_partition(disk) == largest


def test_target_partition_maps_cciss_names(hpcp_mod, reset_hpcp_globals, monkeypatch):
	monkeypatch.setattr(hpcp_mod, 'create_loop_device', lambda image: '/dev/cciss/c0d0')
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check',
		lambda argv, **kwargs: ['cciss!c0d0', 'cciss!c0d0p1', 'cciss!c0d0p2'])
	assert hpcp_mod.get_target_partition('/tmp/not-a-device.img', '2') == ('/dev/cciss/c0d0p2', '/dev/cciss/c0d0')


def test_dev_path(hpcp_mod):
	assert hpcp_mod._dev_path('cciss!c0d0p1') == '/dev/cciss/c0d0p1'
	assert hpcp_mod._dev_path('nvme0n1p2') == '/dev/nvme0n1p2'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "kernel_names or cciss_names or dev_path"`
Expected: FAIL. The paths come out as `/dev/cciss/cciss!c0d0p1` and `/dev/disk/by-id/sda1`, and `_dev_path` is missing.

- [ ] **Step 3: Implement**

Add just above `def get_largest_partition`:

```python
def _dev_path(kernel_name):
	"""/dev path of a block device from its kernel (lsblk, sysfs) name: a '!' spells a '/' (cciss!c0d0p1 is
	/dev/cciss/c0d0p1). Never relative to the path the disk was given by (a /dev/disk/by-id link)."""
	return '/dev/' + kernel_name.replace('!', '/')
```

In `get_largest_partition`, change `return os.path.join(os.path.dirname(disk), largest_partition)` to `return _dev_path(largest_partition)`. In `get_partitions`, change the list comprehension to `partitions = [_dev_path(partition.split()[0]) for partition in partitions[1:]]`. In `get_target_partition`, change `target_partition = '/dev/' + part` to `target_partition = _dev_path(part.strip())`. In `_partition_devices`, change `result[int(f.read().strip())] = '/dev/' + child.replace('!', '/')` to `result[int(f.read().strip())] = _dev_path(child)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command: PASS.
Run `python -m pytest tests/test_disk_patch.py tests/test_imaging.py tests/test_dd_fs_params.py -q`: all pass.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd, dd-diff): partition paths from kernel names (cciss, by-id disks)"
```

---

### Task 3: Disk apply refuses before any raw-area hash

**Files:**
- Modify: `hpcp.py`:
  - add `_raw_payload_candidates` and `_raw_payload_present`, and rewrite `_raw_payload` (≈ 6038);
  - change `_write_phase_devices`' first parameter;
  - reorder `apply_disk_patch`, from `changes, refusals, warnings = _plan_identity(...)` through the end of the section-planning loop.
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Produces:
  - `_raw_payload_present(volume_roots, raw_file, size) -> bool`;
  - `_write_phase_devices(raws, changes, layout, skipped_identity) -> {number: device}`, where `raws` is the manifest's raw entries `(name, raw_file, old_sha, new_sha, size)`. It now includes every raw partition with an entry, whether or not its area is already done.

- [ ] **Step 1: Write the failing tests**

In `tests/test_disk_patch.py`:
- **`test_in_use_raw_partition_is_refused_before_it_is_hashed`:** change the last assertion to `assert hashed == []`. The boot code is no longer hashed before the refusal. Change its `_hash_area` lambda to `lambda path, offset=0, length=None, direct=False: hashed.append(path) or answers[path]`.
- **`test_several_in_use_raw_partitions_and_other_refusals_share_one_message`:** change its `_hash_area` lambda the same way.
- **`test_write_phase_devices`:** replace it with:

```python
def test_write_phase_devices(hpcp_mod):
	"""Every raw partition with an entry is written or checked; whether its area is already done is only known after
	hashing, and the exclusive check runs before any hash."""
	IC = hpcp_mod.IdentityChange
	layout = {'parts': {1: {'device': '/dev/p1'}, 2: {'device': '/dev/p2'}, 3: {'device': '/dev/p3'}, 4: {'device': '/dev/p4'}}}
	raws = [('part2', 'raw/part2.img', 'a', 'b', 1), ('part4', 'raw/part4.img', 'a', 'b', 1), ('mbr', 'raw/mbr-bootcode.bin', 'c', 'd', 440)]
	changes = [IC(3, 'fs_label', 'a', 'b'), IC(1, 'name', 'x', 'y'), IC(None, 'guid', 'g', 'h')]
	assert hpcp_mod._write_phase_devices(raws, changes, layout, set()) == {2: '/dev/p2', 3: '/dev/p3', 4: '/dev/p4'}
	assert hpcp_mod._write_phase_devices([], [IC(3, 'fs_label', 'a', 'b')], layout, {(3, 'fs_label')}) == {}
```

Append to `tests/test_disk_patch.py`:

```python
def _refusal_target(hpcp_mod, tmp_path, monkeypatch, sections=(), kinds=None, payload=True, raw_size=1 << 20):
	"""A hand-made disk patch with a changed raw area on partition 2 and a target whose raw areas would be hashed;
	returns (volume, layout, hashed): hashed lists every _hash_area call."""
	raws = [('part2', 'raw/part2.img', _A, _B, raw_size)]
	volume, layout = _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch, sections=sections, raws=raws, kinds=kinds, open_target=True)
	if payload:
		path = os.path.join(volume, 'raw', 'part2.img')
		os.makedirs(os.path.dirname(path))
		with open(path, 'wb') as f:
			f.truncate(raw_size)
	monkeypatch.setattr(hpcp_mod, '_refuse_if_image_in_use', lambda image: None)
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: '/dev/fake')
	monkeypatch.setattr(hpcp_mod, '_partition_table_type', lambda device: 'gpt')
	monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk, name=None, **kwargs: layout)
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: [])
	monkeypatch.setattr(hpcp_mod, '_swap_devices', lambda: set())
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (7, 0))
	monkeypatch.setattr(hpcp_mod, '_partition_in_use', lambda device, devnum, entries, swaps: '')
	monkeypatch.setattr(hpcp_mod, '_exclusive_refusals', lambda devices: [])
	hashed = []
	monkeypatch.setattr(hpcp_mod, '_hash_area', lambda path, offset=0, length=None, direct=False: hashed.append(path) or _A)
	return volume, layout, hashed


def test_read_only_live_section_is_refused_before_any_raw_hash(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	volume, _, hashed = _refusal_target(hpcp_mod, tmp_path, monkeypatch, sections=[('part1', _one_row(hpcp_mod))])
	monkeypatch.setattr(hpcp_mod, '_live_mount', lambda devnum, entries, device='': ('/boot/efi', True) if device == '/dev/fake1' else (None, False))
	with pytest.raises(RuntimeError, match='^' + re.escape('Disk in use: partition 1 is mounted read-only at /boot/efi; apply this patch offline') + '$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	assert hashed == []


def test_unsupported_identity_change_is_refused_before_any_raw_hash(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	kinds = dict(_GOLDEN_KINDS)
	kinds[3] = ('fs', 'ntfs', 40 << 20)
	volume, _, hashed = _refusal_target(hpcp_mod, tmp_path, monkeypatch, kinds=kinds)
	path = os.path.join(volume, '.hpcp_diff', 'disk.tsv')
	rows = [row._replace(new='NEW-UUID') if (row.number, row.field) == ('3', 'fs_uuid') else row for row in hpcp_mod.read_disk_tsv(path)]
	hpcp_mod.write_disk_tsv(path, rows)
	with pytest.raises(RuntimeError, match=r'^Identity change error: .*nothing applied \(use -fa to apply the rest\)$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	assert hashed == []


def test_exclusive_refusal_comes_before_any_raw_hash(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	volume, _, hashed = _refusal_target(hpcp_mod, tmp_path, monkeypatch)
	seen = []
	monkeypatch.setattr(hpcp_mod, '_exclusive_refusals', lambda devices: seen.append(dict(devices)) or ['partition 2 is held'])
	with pytest.raises(RuntimeError, match='^' + re.escape('Disk in use: partition 2 is held; nothing applied') + '$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	assert hashed == [] and seen == [{2: '/dev/fake2'}]


def test_missing_raw_payload_is_refused_before_any_raw_hash(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	volume, _, hashed = _refusal_target(hpcp_mod, tmp_path, monkeypatch, payload=False)
	with pytest.raises(RuntimeError, match='^' + re.escape('Incomplete patch: raw payload raw/part2.img is missing or damaged; nothing applied') + '$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	assert hashed == []


def test_unmountable_section_is_refused_before_any_raw_hash(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	volume, _, hashed = _refusal_target(hpcp_mod, tmp_path, monkeypatch, sections=[('part1', _one_row(hpcp_mod))])
	monkeypatch.setattr(hpcp_mod, '_live_mount', lambda devnum, entries, device='': (None, False))
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda device, read_only, mount_points, entries: None)
	with pytest.raises(RuntimeError, match='^' + re.escape('Disk layout mismatch: partition 1 (vfat) cannot be mounted read-write; nothing applied') + '$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	assert hashed == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "before_any_raw_hash or before_it_is_hashed or write_phase_devices or share_one_message"`
Expected: FAIL. The raw areas are hashed (`hashed == ['/dev/fake2', ...]`), and `_write_phase_devices` still takes raw_plan dicts.

- [ ] **Step 3: Implement**

Replace `_raw_payload` with:

```python
def _raw_payload_candidates(volume_roots, raw_file, size):
	"""Paths of raw_file on the volumes that are regular files inside their volume with the recorded size."""
	for volume_root in volume_roots:
		path = os.path.join(volume_root, raw_file)
		if (_inside_root(volume_root, path) and os.path.isfile(path) and not os.path.islink(path)
				and os.path.getsize(path) == size):
			yield path

def _raw_payload_present(volume_roots, raw_file, size):
	"""True when some volume holds raw_file with the recorded size (checked before anything is hashed; its content
	is checked by _raw_payload)."""
	return any(True for _ in _raw_payload_candidates(volume_roots, raw_file, size))

def _raw_payload(volume_roots, raw_file, new_sha, size):
	"""Path of raw_file on a volume when it is a regular file inside that volume with the recorded
	size and sha256; None when no volume holds an intact copy."""
	return next((path for path in _raw_payload_candidates(volume_roots, raw_file, size) if _hash_area(path) == new_sha), None)
```

Replace `_write_phase_devices` with:

```python
def _write_phase_devices(raws, changes, layout, skipped_identity):
	"""{number: device} of the partitions the write phase may write: every raw partition with an entry in the patch
	(raws: the manifest's raw entries; whether an area is already done is known only after hashing, and the
	exclusive check runs before any hash), and every partition whose filesystem UUID or label the patch sets. GPT
	fields and the MBR boot code are written through the whole disk, which the write phase locks."""
	devices = {}
	for name, _, _, _, _ in raws:
		if name != 'mbr':
			number = int(name[len('part'):])
			devices[number] = layout['parts'][number]['device']
	for change in changes:
		if change.number is not None and change.field in ('fs_uuid', 'fs_label') and (change.number, change.field) not in skipped_identity:
			devices[change.number] = layout['parts'][change.number]['device']
	return devices
```

In `apply_disk_patch`, replace everything from `changes, refusals, warnings = _plan_identity(disk_rows, layout, in_use)` through the end of the `for name, payload_dir, rows in sections:` planning loop (the line that appends `_plan_section(...)` to `plans`) with:

```python
	changes, refusals, warnings = _plan_identity(disk_rows, layout, in_use)
	# Every refusal that needs no content hash comes before the first one: a raw area can be gigabytes.
	raw_refusals = []
	for name, _, _, _, _ in meta['raws']:
		number = None if name == 'mbr' else int(name[len('part'):])
		if number is not None and in_use.get(number):
			# Refused without reading it: a live area (an LVM PV, say) can be hundreds of GB. So whether it
			# already holds the new content (a live re-run) is not known either.
			raw_refusals.append(f'partition {number} is {in_use[number]}; this patch carries its raw area, '
				'which cannot be checked or written while it is in use')
	for name, _, _ in sections:
		part = layout['parts'][int(name[len('part'):])]
		live, live_read_only = _live_mount(_device_number(part['device']), entries, part['device'])
		if live and live_read_only:
			refusals.append(f'partition {name[len("part"):]} is mounted read-only at {live}')
	if raw_refusals:
		done = 'it if it is' if len(raw_refusals) == 1 else 'each one that is'
		raise RuntimeError('Disk in use: ' + '; '.join(refusals + raw_refusals)
			+ f'; apply the patch offline (an offline apply skips {done} already done)')
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
	# The kernel answers for every mount namespace and for hidden mounts, which _partition_in_use cannot see.
	write_devices = _write_phase_devices(meta['raws'], changes, layout, skipped_identity)
	exclusive = _exclusive_refusals(write_devices)
	if exclusive:
		raise RuntimeError('Disk in use: ' + '; '.join(exclusive) + '; nothing applied')
	for name, raw_file, _, _, size in meta['raws']:
		if not _raw_payload_present(volume_roots, raw_file, size):
			raise RuntimeError(f'Incomplete patch: raw payload {raw_file} is missing or damaged; nothing applied')
	with _patch_hash_settings(meta) as (compare_hash, full_hash):
		roots = {}
		for name, _, _ in sections:
			part = layout['parts'][int(name[len('part'):])]
			root = _mount_partition(part['device'], False, mount_points, entries)
			if root is None:
				raise RuntimeError(f"Disk layout mismatch: partition {name[len('part'):]} ({part['fs_type']}) cannot be mounted read-write; nothing applied")
			roots[name] = root
		# Only now the raw areas are read: every refusal that needs no hash has run.
		raw_plan = []
		for name, raw_file, old_sha, new_sha, size in meta['raws']:
			number = None if name == 'mbr' else int(name[len('part'):])
			device = disk if number is None else layout['parts'][number]['device']
			found = _hash_area(device, 0, size)
			action = _plan_raw(old_sha, new_sha, found)
			payload = None
			if action != 'skip':
				payload = _raw_payload(volume_roots, raw_file, new_sha, size)
				if payload is None:
					raise RuntimeError(f'Incomplete patch: raw payload {raw_file} is missing or damaged; nothing applied')
			raw_plan.append({'name': name, 'device': device, 'size': size, 'payload': payload, 'action': action,
				'found': found, 'new': new_sha})
		plans = []
		for name, payload_dir, rows in sections:
			part = layout['parts'][int(name[len('part'):])]
			fs_type = fs_types.get(name[len('part'):], '')
			plans.append(_plan_section(name, rows, _section_volume_roots(volume_roots, payload_dir), roots[name], compare_hash, full_hash, max_workers,
				_has_posix_metadata(fs_type), _has_case_insensitive_names(fs_type, part['device']), tz_tolerant=_is_fat(fs_type)))
```

The code that follows (`raw_conflicts = ...`, `_preflight_section`, `_resolve_section_conflicts`, and the section loop) stays unchanged. In the `apply_disk_patch` docstring, after the sentence that begins `Everything is checked before anything changes`, add: `Every refusal that needs no content hash (in use, read-only live mounts, identity changes that cannot be made, the exclusive check, missing raw payloads, sections that cannot be mounted) comes before the first raw area is read: a raw area can be gigabytes.`

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command: PASS.
Run `python -m pytest tests/test_disk_patch.py tests/test_diff_image.py -q`: all pass, including the root disk round-trip and hold tests.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd-diff): disk apply refuses before reading any raw area"
```

---

### Task 4: Disk patch generation hashes raw areas first and refuses early

**Files:**
- Modify: `hpcp.py`:
  - add `_check_raw_fits_a_volume` just above `def process_get_disk_diff_image` (≈ 6947);
  - restructure the per-partition loop of `process_get_disk_diff_image`;
  - replace its late raw-fit check with a call to the helper.
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Produces: `_check_raw_fits_a_volume(raw_sizes: list[int], init_size: int, dest_image_size: int) -> None`, which raises `Diff image error: ...`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_disk_patch.py`:

```python
def _generation_fakes(hpcp_mod, monkeypatch, tmp_path, new_raw_size=1 << 20, old_raw_size=1 << 20, raw_changes=True):
	"""Fake disks for process_get_disk_diff_image: p1 vfat (a section), p2 raw. Returns calls: the order of
	_hash_area / _partition_changes calls."""
	new = _layout({1: _part(fs_type='vfat', device='/dev/new1'), 2: _part(fs_type='', size=new_raw_size, device='/dev/new2')})
	old = _layout({1: _part(fs_type='vfat', device='/dev/old1'), 2: _part(fs_type='', size=old_raw_size, device='/dev/old2')})
	layouts = {'/dev/newdisk': new, '/dev/olddisk': old}
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: '/dev/newdisk' if path.endswith('new.img') else '/dev/olddisk')
	monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk, name=None, **kwargs: layouts[disk])
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: [])
	monkeypatch.setattr(hpcp_mod, '_swap_devices', lambda: set())
	monkeypatch.setattr(hpcp_mod, '_refuse_raw_in_use', lambda number, disks, entries, swaps: None)
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda device, read_only, mount_points, entries: str(tmp_path))
	monkeypatch.setattr(hpcp_mod, '_release_mount', lambda root, mount_points: None)
	monkeypatch.setattr(hpcp_mod, '_live_mount', lambda devnum, entries, device='': (None, False))
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (7, 0))
	calls = []

	def changes(*args, **kwargs):
		calls.append('section')
		return []

	def hash_area(path, offset=0, length=None, direct=False):
		calls.append(path)
		return 'new' if raw_changes and path.startswith('/dev/new') else 'old'

	monkeypatch.setattr(hpcp_mod, '_partition_changes', changes)
	monkeypatch.setattr(hpcp_mod, '_hash_area', hash_area)
	for name in ('new.img', 'old.img'):
		(tmp_path / name).write_bytes(b'')
	return calls


def test_generation_refuses_a_raw_size_change_before_hashing(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	calls = _generation_fakes(hpcp_mod, monkeypatch, tmp_path, new_raw_size=2 << 20)
	with pytest.raises(RuntimeError, match=r'^Disk layout mismatch: partition 2 is a raw area and its size changed'):
		hpcp_mod.process_get_disk_diff_image([str(tmp_path / 'new.img')], [str(tmp_path / 'old.img')], str(tmp_path / 'p.img'), [], [], max_workers=1)
	assert calls == []


def test_generation_hashes_raw_areas_before_any_section(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	calls = _generation_fakes(hpcp_mod, monkeypatch, tmp_path, raw_changes=False)
	assert hpcp_mod.process_get_disk_diff_image([str(tmp_path / 'new.img')], [str(tmp_path / 'old.img')], str(tmp_path / 'p.img'), [], [], max_workers=1) is None
	assert calls.index('section') > max(calls.index('/dev/old2'), calls.index('/dev/new2'))


def test_generation_refuses_a_raw_area_too_large_for_dis_before_any_section(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Spec Part 2 item 2: the changed raw areas alone force a split and the largest one does not fit a volume."""
	calls = _generation_fakes(hpcp_mod, monkeypatch, tmp_path, new_raw_size=300 << 20, old_raw_size=300 << 20)
	with pytest.raises(RuntimeError, match=r'^Diff image error: a raw area of .* must fit in one patch volume; use -dis \d+MiB or larger$'):
		hpcp_mod.process_get_disk_diff_image([str(tmp_path / 'new.img')], [str(tmp_path / 'old.img')], str(tmp_path / 'p.img'), [], [],
			dest_image_size=100 << 20, max_workers=1)
	assert 'section' not in calls


def test_generation_does_not_refuse_an_unchanged_large_raw_area(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	calls = _generation_fakes(hpcp_mod, monkeypatch, tmp_path, new_raw_size=300 << 20, old_raw_size=300 << 20, raw_changes=False)
	assert hpcp_mod.process_get_disk_diff_image([str(tmp_path / 'new.img')], [str(tmp_path / 'old.img')], str(tmp_path / 'p.img'), [], [],
		dest_image_size=100 << 20, max_workers=1) is None
	assert 'section' in calls
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "generation_refuses or generation_hashes or generation_does_not_refuse"`
Expected: FAIL. The size change is refused only after the p1 section diff, the raw areas are hashed after it, and the early `-dis` refusal is missing.

- [ ] **Step 3: Implement**

Add just above `def process_get_disk_diff_image`:

```python
def _check_raw_fits_a_volume(raw_sizes, init_size, dest_image_size):
	"""Raise 'Diff image error: ...' when the patch needs more than one volume (_plan_dest_images for init_size)
	and its largest raw area, which must fit whole in one volume, is larger than one volume's usable space."""
	if not raw_sizes:
		return
	count, volume_size = _plan_dest_images(init_size, dest_image_size)
	largest_raw = max(raw_sizes)
	# Same usable-space estimate as _plan_dest_images uses for split volumes.
	if count > 1 and largest_raw > int(volume_size * 0.90) - (16 << 20):
		needed = _align_up(int((largest_raw + (16 << 20)) / 0.90) + 1)
		# Whole MiB, rounded up, in a form the -dis parser reads exactly ('M' alone is 10^6 bytes).
		needed_mib = (needed + (1 << 20) - 1) >> 20
		raise RuntimeError(f'Diff image error: a raw area of {format_bytes(largest_raw)}B must fit in one patch volume; use -dis {needed_mib}MiB or larger')
```

In `process_get_disk_diff_image`, replace the block from `kinds, sections, raws = {}, [], []` through the end of the `for number in sorted(new_layout['parts']):` loop (the line `kinds[number] = kind`) with:

```python
	kinds, sections, raws = {}, [], []

	def add_raw(number):
		"""Hash partition number's area on both disks; a changed one becomes a raw entry."""
		new_part, old_part = new_layout['parts'][number], old_layout['parts'][number]
		if new_part['size'] != old_part['size']:
			raise RuntimeError(f"Disk layout mismatch: partition {number} is a raw area and its size changed ({old_part['size']} vs {new_part['size']}); re-clone instead")
		old_sha, new_sha = _hash_area(old_part['device'], 0, old_part['size']), _hash_area(new_part['device'], 0, new_part['size'])
		if old_sha != new_sha:
			raws.append((f'part{number}', new_part['device'], new_part['size'], old_sha, new_sha))

	numbers = sorted(new_layout['parts'])
	raw_numbers = [number for number in numbers if _default_kind(new_layout['parts'][number]) == 'raw']
	# Raw areas first, before any filesystem section is diffed: their sizes are checked before anything is
	# hashed, and an area too large for -dis is refused before the sections' file diffs.
	for number in raw_numbers:
		new_size, old_size = new_layout['parts'][number]['size'], old_layout['parts'][number]['size']
		if new_size != old_size:
			raise RuntimeError(f"Disk layout mismatch: partition {number} is a raw area and its size changed ({old_size} vs {new_size}); re-clone instead")
	for number in raw_numbers:
		add_raw(number)
		kinds[number] = 'raw'
	_check_raw_fits_a_volume([raw[2] for raw in raws], sum(_align_up(raw[2]) for raw in raws) + 64 * 1024, dest_image_size)
	for number in numbers:
		if number in kinds:
			continue
		new_part, old_part = new_layout['parts'][number], old_layout['parts'][number]
		kind = _default_kind(new_part)
		if kind == 'fs':
			new_root = _mount_partition(new_part['device'], True, mount_points, entries)
			old_root = _mount_partition(old_part['device'], True, mount_points, entries)
			if new_root and old_root:
				lives = [_live_mount(_device_number(part['device']), entries, part['device'])[0] for part in (new_part, old_part)]
				rows = _partition_changes(new_root, old_root, mount_points, lives, exclude, full_hash, max_workers, parallel_file_listing,
					input_files, metadata=_has_posix_metadata(new_part['fs_type']), fat=_is_fat(new_part['fs_type']) or _is_fat(old_part['fs_type']))
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
				failed = ' and '.join(side for side, root in (('new', new_root), ('old', old_root)) if not root)
				print(f"Warning: partition {number} ({new_part['fs_type']}) could not be mounted on the {failed} side; comparing it as a raw area", file=sys.stderr)
				_refuse_raw_in_use(number, disks, entries, swaps)
				kind = 'raw'
				add_raw(number)
		kinds[number] = kind
```

Keep the `_partition_changes(...)` call exactly as it is in the current code. If its arguments differ from the block above, the current code wins: copy them unchanged.

Then replace the late raw-fit block (from `count, volume_size = _plan_dest_images(init_size, dest_image_size)` through its `raise RuntimeError(f'Diff image error: a raw area of ...')`) with:

```python
	count, volume_size = _plan_dest_images(init_size, dest_image_size)
	# Again with every section's payload: sections can force the split the raw areas alone did not.
	_check_raw_fits_a_volume([raw[2] for raw in raws], init_size, dest_image_size)
```

The sections' sorted order in the patch is unchanged: `section_rows` and `raw_entries` come from `sections` and `raws` as before. If any code after the loop relied on `sections` or `raws` being in partition order, sort them by number there and say so in the report.

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command: PASS.
Run `python -m pytest tests/test_disk_patch.py -q`: all pass. That includes the root generation tests, whose raws and sections must still be listed in partition order in the manifest.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd-diff): generation hashes raw areas first and refuses a size change or a too-large raw area early"
```

---

### Task 5: One probe per layout read

**Files:**
- Modify: `hpcp.py`:
  - `get_partition_details` and `get_partition_infos` gain `identity=True`;
  - `_settle_partitions` gains `settle=True`;
  - `read_disk_layout` gains `pttype=None` and `fs_params=True`;
  - `apply_disk_patch` passes `pttype` to both `read_disk_layout` calls, and `fs_params=False` to the verify read.
- Test: `tests/test_disk_patch.py`

**Interfaces:**
- Produces:
  - `read_disk_layout(device, name=None, pttype=None, fs_params=True)`;
  - `get_partition_infos(device, identity=True)`;
  - `get_partition_details(device, partition, sector_size=512, identity=True)`;
  - `_settle_partitions(disk, numbers, timeout=10, settle=True)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_disk_patch.py`, every `monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk, name=None: ...)` becomes `lambda disk, name=None, **kwargs: ...`; there are six, found by `grep -n "'read_disk_layout'" tests/test_disk_patch.py`. Replace `test_read_disk_layout_settles_udev_before_reading` with:

```python
def test_read_disk_layout_settles_once_and_probes_each_partition_once(hpcp_mod, monkeypatch):
	"""A node udev is still re-adding would be probed as empty: settle once, then wait for the nodes without settling
	again; the GPT read runs no blkid of its own, and each partition gets one probe on its settled device."""
	calls = []
	monkeypatch.setattr(hpcp_mod, '_partition_table_type', lambda device: calls.append('pttype') or 'gpt')
	monkeypatch.setattr(hpcp_mod, '_udev_settle', lambda: calls.append('settle'))

	def infos(device, identity=True):
		calls.append(('infos', identity))
		return {'x': {'disk_identifier': 'G', 'sector_size': 512, 'size': 1 << 20},
			'1': {'partition_guid_code': 'T', 'unique_partition_guid': 'U', 'partition_name': 'n', 'partition_attrs': '0', 'size': 1 << 20}}

	infos.cache_clear = lambda: None
	monkeypatch.setattr(hpcp_mod, 'get_partition_infos', infos)
	monkeypatch.setattr(hpcp_mod, '_settle_partitions', lambda disk, numbers, timeout=10, settle=True: calls.append(('nodes', settle)) or {1: '/dev/x1'})
	monkeypatch.setattr(hpcp_mod, '_probe_identity', lambda device: calls.append(('probe', device)) or ('ext4', 'u', 'l'))
	monkeypatch.setattr(hpcp_mod, 'probe_fs_params', lambda device, fs_type: calls.append(('params', device)) or {})
	monkeypatch.setattr(hpcp_mod, 'MIRROR_FS_PARAMS', True)
	layout = hpcp_mod.read_disk_layout('/dev/x')
	assert calls == ['pttype', 'settle', ('infos', False), ('nodes', False), ('probe', '/dev/x1'), ('params', '/dev/x1')]
	assert layout['parts'][1]['fs_label'] == 'l' and layout['parts'][1]['device'] == '/dev/x1'
	del calls[:]
	hpcp_mod.read_disk_layout('/dev/x', pttype='gpt', fs_params=False)
	assert calls == ['settle', ('infos', False), ('nodes', False), ('probe', '/dev/x1')]


def test_get_partition_details_without_identity_runs_only_sgdisk(hpcp_mod, reset_hpcp_globals, monkeypatch):
	commands = []

	def fake(argv, **kwargs):
		commands.append(argv[0])
		return ["Partition name: 'n'"] if argv[0] == 'sgdisk' else []

	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake)
	monkeypatch.setattr(hpcp_mod, 'get_target_partition', lambda device, partition: (_ for _ in ()).throw(AssertionError('not called')))
	hpcp_mod.get_partition_details.cache_clear()
	try:
		details = hpcp_mod.get_partition_details('/dev/x', '1', 512, identity=False)
	finally:
		hpcp_mod.get_partition_details.cache_clear()
	assert commands == ['sgdisk'] and details['partition_name'] == 'n' and details['fs_type'] == ''


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_a_fresh_loop_disk_reads_the_same_ten_times(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Review Focus 5: one settle per read must not bring back the udev re-read race 9.67 fixed: attach, read at once,
	repeatedly; the counts show one PTTYPE probe, one settle and one blkid per partition per read."""
	image = tmp_path / 'golden.img'
	_make_golden(image)
	commands = []
	real = hpcp_mod.run_command_in_multicmd_with_path_check

	def counting(argv, **kwargs):
		commands.append(list(argv))
		return real(argv, **kwargs)

	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', counting)
	first = None
	for _ in range(10):
		with _session(hpcp_mod) as (mount_points, loop_devices):
			disk = hpcp_mod._attach_disk(str(image), True, loop_devices)
			del commands[:]
			layout = hpcp_mod.read_disk_layout(disk)
			stripped = {number: {key: value for key, value in part.items() if key != 'device'} for number, part in layout['parts'].items()}
			first = first or stripped
			assert stripped == first
			assert sum(1 for argv in commands if 'PTTYPE' in argv) == 1
			assert sum(1 for argv in commands if argv[:2] == ['udevadm', 'settle']) == 1
			assert sum(1 for argv in commands if argv[:4] == ['blkid', '-p', '-o', 'udev']) == len(layout['parts'])
	assert first[3]['fs_label'] == 'root' and first[1]['fs_type'] == 'vfat'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_disk_patch.py -q -k "settles_once or without_identity or ten_times or read_disk_layout"`
Expected: FAIL. The parameters are missing, the read settles twice, and each read runs two blkid probes per partition.

- [ ] **Step 3: Implement**

`get_partition_details`:
- Change its signature to `def get_partition_details(device, partition, sector_size=512, identity=True):`.
- Add a docstring line: `identity False reads the GPT entry only (one sgdisk call): no partition lookup, blkid or fs-params probe; read_disk_layout probes the settled device itself.`
- Guard the block from `target_partition, loop_device = get_target_partition(...)` to the end of the function, before `return rtnDic`, with `if not identity: return rtnDic` placed before it.

`get_partition_infos`:
- Change its signature to `def get_partition_infos(device, identity=True):`.
- Pass it on: `partition_info[part[0]] = get_partition_details(device, part[0], sector_size, identity)`.

`_settle_partitions`:
- Change its signature to `def _settle_partitions(disk, numbers, timeout=10, settle=True):`.
- Change its first line to `if settle:` followed by the indented `_udev_settle()`.
- Add a docstring line: `settle False skips the udev settle (the caller has just settled).`

Replace `read_disk_layout` with:

```python
def read_disk_layout(device, name=None, pttype=None, fs_params=True):
	"""Fresh GPT + filesystem identity read of block device (name: the disk as messages call it, default device).

	Returns {'disk': {'guid', 'sector_size', 'size'}, 'parts': {number: {'type_guid',
	'unique_guid', 'name', 'attrs', 'size', 'fs_type', 'fs_uuid', 'fs_label', 'fs_params',
	'device'}}}. The partition caches are cleared first: loop device names are reused.
	pttype: the partition-table type the caller already probed (_partition_table_type), so it is not probed twice.
	udev is settled once, the GPT is read with sgdisk alone, then each partition is probed once on its settled
	device (_probe_identity); fs_params False skips the mkfs-parameter probes ('fs_params' is then '': a verify read
	compares identity values only).
	Raises RuntimeError('Disk layout mismatch: ...') unless the disk has a GPT partition table:
	sgdisk reads an MBR disk as if it were GPT, and an identity change would convert it.
	"""
	name = name or device
	if pttype is None:
		pttype = _partition_table_type(device)
	if not pttype:
		raise RuntimeError(f'Disk layout mismatch: {_no_partition_table(name)}; disk patches support GPT disks only')
	if pttype != 'gpt':
		raise RuntimeError(f'Disk layout mismatch: {name} has no GPT partition table (found {pttype}); disk patches support GPT disks only')
	get_partition_infos.cache_clear()
	get_partition_details.cache_clear()
	# A node udev is still re-adding would be probed as empty: settle once, then wait for every node.
	_udev_settle()
	infos = get_partition_infos(device, identity=False)
	devices = _settle_partitions(device, [int(key) for key, info in infos.items() if 'disk_identifier' not in info], settle=False)
	disk, parts = {}, {}
	for key, info in infos.items():
		if 'disk_identifier' in info:
			disk = {'guid': info['disk_identifier'], 'sector_size': info['sector_size'], 'size': info['size']}
			continue
		number = int(key)
		fs_type, fs_uuid, fs_label = _probe_identity(devices[number])
		params = probe_fs_params(devices[number], fs_type) if fs_params and fs_type and MIRROR_FS_PARAMS else {}
		parts[number] = {
			'type_guid': info['partition_guid_code'], 'unique_guid': info['unique_partition_guid'],
			'name': info['partition_name'], 'attrs': info['partition_attrs'], 'size': info['size'],
			'fs_type': fs_type, 'fs_uuid': fs_uuid, 'fs_label': fs_label,
			'fs_params': _fs_params_text(params), 'device': devices[number],
		}
	return {'disk': disk, 'parts': parts}
```

In `apply_disk_patch`, replace:

```python
	if not _partition_table_type(disk):
		raise RuntimeError(f'Invalid arguments: {_no_partition_table(target)}; a disk patch needs a whole GPT disk (block device or disk image)')
	layout = read_disk_layout(disk, name=target)
```

with:

```python
	pttype = _partition_table_type(disk)
	if not pttype:
		raise RuntimeError(f'Invalid arguments: {_no_partition_table(target)}; a disk patch needs a whole GPT disk (block device or disk image)')
	layout = read_disk_layout(disk, name=target, pttype=pttype)
```

and change the verify read `after = read_disk_layout(disk)` to `after = read_disk_layout(disk, pttype=pttype, fs_params=False)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command: PASS.
Run the root test three times: `-k ten_times`.
Run `python -m pytest tests/test_disk_patch.py tests/test_dd_fs_params.py tests/test_imaging.py -q`: all pass. The disk round trips must give identical `disk.tsv` values, including `fs_params` for ext4 and vfat.

- [ ] **Step 5: Commit**

```bash
git add hpcp.py tests/test_disk_patch.py
git commit -m "fix(dd-diff): one partition-table probe, one udev settle and one blkid per partition per layout read"
```

---

### Task 6: Docs, version 9.70, full verification

**Files:**
- Modify: `README.md`; `hpcp.py` (`version`, `COMMIT_DATE`); `tests/test_dd_fs_params.py` (`test_version_bumped`).

- [ ] **Step 1: README**

In `README.md`, after the disk-patch bullet that begins `- A node's filesystem UUID or label that differs from the golden base`, add:

```markdown
- Filesystem labels, GPT partition names and exFAT serials are read and written exactly: non-ASCII characters, spaces (also leading and trailing), `=` and apostrophes included; `-dd` copies keep them the same way. Disk patches made by hpcp 9.67–9.69 for disks with such labels or names recorded them mangled or cut short, so regenerate those patches.
- Every refusal that needs no hash (a partition in use, an identity change that cannot be made, a missing raw payload, a section that cannot be mounted) comes before any raw area is read, and so does a raw-area size change when generating; a raw area too large for `-dis` is refused right after the raw areas are hashed, before the filesystems are compared.
```

If `python3 hpcp.py -h` output changed (it should not), regenerate the README help block the way the 9.66 release did; otherwise leave it.

- [ ] **Step 2: Version**

```bash
sed -i "s/^version = '9.69'$/version = '9.70'/" hpcp.py
sed -i "s/^COMMIT_DATE = '.*'$/COMMIT_DATE = '$(date +%F)'/" hpcp.py
grep -nE "^version|^COMMIT_DATE" hpcp.py
```

Update `test_version_bumped` in `tests/test_dd_fs_params.py` to expect `'9.70'` and the new `COMMIT_DATE`.

- [ ] **Step 3: Full verification**

Run: `python -m pytest tests/ -q`. Expected: all pass with exactly 1 skipped. No xfailed remain, now that the exFAT UUID case passes.
Then check:
- `python3 hpcp.py -V` prints `9.70`;
- `uv run --no-project --python 3.8 python -c "import ast; ast.parse(open('hpcp.py').read(), feature_version=(3,6)); print('ok')"` prints `ok`;
- `losetup -a | grep -c /tmp/pytest` and `mount | grep -c /tmp/pytest` both print 0;
- `md5sum /usr/lib/os-release` prints `776b0aac9b17d49a0ac879da9801f51b`. If it differs, STOP and report BLOCKED without trying to repair anything.

- [ ] **Step 4: Commit**

```bash
git add README.md hpcp.py tests/test_dd_fs_params.py
git commit -m "docs: exact labels, GPT names and exFAT serials; refusals before hashing; release 9.70"
```

Releasing (gate review, push, uvpack, cas-toolbox pin) is out of scope for this plan.
