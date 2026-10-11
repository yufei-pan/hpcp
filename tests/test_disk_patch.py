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
