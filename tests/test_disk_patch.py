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
