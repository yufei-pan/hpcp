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
	f'raw\tpart2\traw/x.img\t{_A}\t{_B}\t\u00b2',
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
	'part\t\u00b2\tname\ta\tb\n',
	'part\t\u0663\tname\ta\tb\n',
])
def test_disk_tsv_rejects_bad_rows(tmp_path, hpcp_mod, body):
	path = tmp_path / 'disk.tsv'
	path.write_text('#hpcp-disk\t1\n' + body)
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.read_disk_tsv(str(path))
	path.write_text('#hpcp-disk\t\u00b2\n')
	with pytest.raises(RuntimeError, match='Patch format error'):
		hpcp_mod.read_disk_tsv(str(path))
	path.write_text('#hpcp-disk\t2\n')
	with pytest.raises(RuntimeError, match='newer'):
		hpcp_mod.read_disk_tsv(str(path))


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


def test_settle_partitions_waits_for_late_nodes(tmp_path, hpcp_mod, monkeypatch):
	nodes = {}
	for number in (1, 2):
		nodes[number] = str(tmp_path / f'p{number}')
		(tmp_path / f'p{number}').write_text('')
	calls = []
	def fake(disk):
		calls.append(disk)
		return dict(nodes) if len(calls) > 3 else {1: nodes[1]}
	monkeypatch.setattr(hpcp_mod, '_partition_devices', fake)
	monkeypatch.setattr(hpcp_mod, 'check_path', lambda name: False)
	assert hpcp_mod._settle_partitions('/dev/x', [1, 2]) == nodes
	assert len(calls) == 4


def test_settle_partitions_gives_up(hpcp_mod, monkeypatch):
	monkeypatch.setattr(hpcp_mod, '_partition_devices', lambda disk: {})
	monkeypatch.setattr(hpcp_mod, 'check_path', lambda name: False)
	with pytest.raises(RuntimeError, match=r'Disk layout mismatch.*\[2\]'):
		hpcp_mod._settle_partitions('/dev/x', [2], timeout=0.3)


# ---- root + loop fixtures (GPT disk images) ----

_DISK_TOOLS = ('losetup', 'sgdisk', 'mkfs.vfat', 'mkfs.ext4', 'mkswap', 'mount', 'umount', 'lsblk', 'blkid', 'e2label')
requires_root_and_disk_tools = pytest.mark.skipif(
	os.name == 'nt' or os.geteuid() != 0 or any(shutil.which(tool) is None for tool in _DISK_TOOLS),
	reason='disk patch tests need root and losetup/sgdisk/mkfs.vfat/mkfs.ext4/mkswap/mount/lsblk/blkid/e2label')


def _run(*argv):
	result = subprocess.run([str(arg) for arg in argv], capture_output=True)
	if result.returncode:
		raise RuntimeError(f"{argv} rc={result.returncode}: {result.stderr!r}")


@contextlib.contextmanager
def _attached(image, read_only=False):
	argv = ['losetup', '--partscan', '--find', '--show'] + (['--read-only'] if read_only else []) + [str(image)]
	loop = subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()
	try:
		# The kernel (--partscan) and udev (re-read after a close-write) can both rescan the table, briefly
		# removing and re-adding the partition nodes: let udev go quiet, then wait for every GPT partition.
		if shutil.which('udevadm'):
			subprocess.run(['udevadm', 'settle'], capture_output=True)
		listing = subprocess.run(['sgdisk', '-p', loop], capture_output=True, text=True).stdout.splitlines()
		wanted = [line.split()[0] for line in listing if len(line.split()) > 3 and line.split()[0].isdigit()] or ['1']
		for _ in range(100):
			if all(os.path.exists(f'{loop}p{number}') for number in wanted):
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


def test_malformed_mountinfo_line_is_skipped_not_fatal(tmp_path, hpcp_mod):
	"""One bad line must not drop the mounts listed after it (the in-use check would fail open)."""
	path = tmp_path / 'mountinfo'
	path.write_text('\n'.join([
		'36 25 8:3 / / rw,relatime shared:1 - ext4 /dev/sda3 rw',
		'41 36 X:Y / /bad rw - ext4 /dev/sdz rw',
		'39 36 8:5 / /mnt/with\\040space ro,relatime - ext4 /dev/sda5 ro',
	]) + '\n')
	entries = hpcp_mod._mountinfo_entries(str(path))
	assert [entry['mount_point'] for entry in entries] == ['/', '/mnt/with space']
	assert hpcp_mod._live_mount((8, 5), entries) == ('/mnt/with space', True)


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


def test_section_volume_roots_drop_a_symlinked_section_directory(tmp_path, hpcp_mod):
	outside = tmp_path / 'outside'
	_write(outside / 'a.txt', 'hello')
	row = hpcp_mod.ChangeRow('A', 'a.txt', '-', 'f:5:1:100644:0:0:-:-')
	bad = tmp_path / 'bad'
	bad.mkdir()
	os.symlink(str(outside), str(bad / 'part1'))
	good = tmp_path / 'good'
	_write(good / 'part1' / 'a.txt', 'hello')
	roots = hpcp_mod._section_volume_roots([str(bad)], 'part1')
	assert roots == [] and hpcp_mod._payload_source(roots, row) is None
	roots = hpcp_mod._section_volume_roots([str(bad), str(good)], 'part1')
	assert roots == [str(good / 'part1')] and hpcp_mod._payload_source(roots, row) == str(good / 'part1' / 'a.txt')
	assert hpcp_mod._section_volume_roots([str(bad)], '.') == [str(bad)]


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


def _mounts_under(prefix):
	with open('/proc/self/mountinfo') as f:
		return [line for line in f if prefix in line.split()[4]]


def _read_part3_patch(hpcp_mod, out):
	with _session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(out, True, mount_points, loop_devices)
		meta = hpcp_mod.read_manifest(os.path.join(root, '.hpcp_diff', 'manifest'))
		rows = hpcp_mod.read_changes(os.path.join(root, '.hpcp_diff', 'part3.changes.tsv'))
	return meta, rows


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
@pytest.mark.parametrize('live_side', ['new', 'old'])
def test_live_partition_with_temp_dirs_on_the_live_filesystem(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, live_side):
	"""Binds of a live filesystem onto a directory of that same filesystem are real mounts and stay out of the patch."""
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_make_newer(old, new, label=None, raw=False, mbr=False, gpt_name=None)
	live_image = new if live_side == 'new' else old
	with _attached(live_image, read_only=False) as loop:
		with _mounted(loop + 'p3') as live_root:
			temp_dir = os.path.join(live_root, 'tmpdir')
			os.makedirs(temp_dir)
			monkeypatch.setattr(tempfile, 'tempdir', temp_dir)
			with _session(hpcp_mod) as (mount_points, loop_devices):
				args = ([loop], [str(old)]) if live_side == 'new' else ([str(new)], [loop])
				assert hpcp_mod.process_get_disk_diff_image(args[0], args[1], out, mount_points, loop_devices, max_workers=1) == out
			assert _mounts_under(temp_dir) == []
			monkeypatch.setattr(tempfile, 'tempdir', None)
	meta, part3 = _read_part3_patch(hpcp_mod, out)
	assert 'part3' in [section[0] for section in meta['sections']]
	assert 'part3' not in [raw[0] for raw in meta['raws']]
	assert ('M', 'etc/os-release') in {(row.op, row.path) for row in part3}
	# tmpdir/ itself exists on the live side only; nothing below it (this run's mount-point dirs) may appear.
	assert [row.path for row in part3 if row.path.startswith('tmpdir') and row.path != 'tmpdir/'] == []


def test_unmountable_filesystem_is_compared_as_raw_with_a_warning(hpcp_mod, monkeypatch, capsys):
	device = '/dev/none'
	parts = {1: _part(number=1, device=device, fs_type='ext4')}
	layout = _layout(parts)
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: path)
	monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk: layout)
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: [])
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda *a: None)
	monkeypatch.setattr(hpcp_mod, '_hash_area', lambda *a: 'h')
	monkeypatch.setattr(hpcp_mod, 'expand_paths', lambda paths: list(paths))
	assert hpcp_mod.process_get_disk_diff_image(['/x/new'], ['/x/old'], '/x/out', [], [], max_workers=1) is None
	assert 'partition 1 (ext4) could not be mounted on the new and old side; comparing it as a raw area' in capsys.readouterr().err


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
def test_disk_apply_round_trip_and_rerun(tmp_path, hpcp_mod, reset_hpcp_globals, capsys):
	"""Success criterion + Review Focus 5."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	summary = _apply(hpcp_mod, patch, target)
	assert summary['failed'] == 0 and summary['raw_written'] == 2 and summary['identity_changed'] == 2
	assert hpcp_mod.ERRORS == []
	out = capsys.readouterr().out
	assert 'Section part3:' in out and 'Disk patch summary:' in out and ', added ' in out
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


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_image_target_attached_but_unmounted_is_refused(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	before = _sha(target.read_bytes())
	with _attached(target) as loop:
		with pytest.raises(RuntimeError, match='Disk in use: .* is attached as ' + loop):
			_apply(hpcp_mod, patch, target)
	assert _sha(target.read_bytes()) == before


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_symlinked_block_device_target_resolves_to_the_device(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	with _attached(target) as loop:
		link = tmp_path / 'by-id-link'
		os.symlink(loop, str(link))
		with _session(hpcp_mod) as (mount_points, loop_devices):
			disk = hpcp_mod._attach_disk(str(link), False, loop_devices)
			assert disk == os.path.realpath(loop)
			assert sorted(hpcp_mod.read_disk_layout(disk)['parts']) == [1, 2, 3, 4]


def test_read_write_mount_of_xfs_uses_nouuid(hpcp_mod, reset_hpcp_globals, monkeypatch):
	calls = []
	monkeypatch.setattr(hpcp_mod, '_probe_fs_type', lambda device: 'xfs')
	monkeypatch.setattr(hpcp_mod, '_live_mount', lambda *a: (None, False))
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (7, 1))
	monkeypatch.setattr(hpcp_mod, '_is_mount_point', lambda path: True)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', lambda argv, **kw: calls.append(list(argv)))
	mount_points = []
	try:
		assert hpcp_mod._mount_partition('/dev/fake1', False, mount_points, []) is not None
	finally:
		for path in mount_points:
			os.rmdir(path)
	assert calls[0][:3] == ['mount', '-o', 'nouuid'] and calls[0][3] == '/dev/fake1'


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
