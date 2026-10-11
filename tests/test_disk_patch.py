import contextlib
import errno
import os
import re
import shutil
import stat
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
	"""Review Focus 1: loop0p1 and loop0p11 must not be confused by name suffix.

	Only the disk's own sysfs children count: a partitioned md array built on the disk's
	partitions (md127p1, also partition 1) is a descendant lsblk lists, not a partition of the disk.
	"""
	names = {'loop0p1': '1', 'loop0p11': '11', 'loop0p2': '2'}
	for name, number in names.items():
		(tmp_path / 'loop0' / name).mkdir(parents=True)
		(tmp_path / 'loop0' / name / 'partition').write_text(number + '\n')
	(tmp_path / 'loop0' / 'holders' / 'md127').mkdir(parents=True)
	(tmp_path / 'loop0' / 'queue').mkdir()
	(tmp_path / 'md127' / 'md127p1').mkdir(parents=True)
	(tmp_path / 'md127' / 'md127p1' / 'partition').write_text('1\n')
	(tmp_path / 'md127p1').mkdir()
	(tmp_path / 'md127p1' / 'partition').write_text('1\n')
	# A symlinked child (bdi, subsystem, device) is never a partition of the disk.
	os.symlink(str(tmp_path / 'md127' / 'md127p1'), str(tmp_path / 'loop0' / 'md127p1'))
	monkeypatch.setattr(hpcp_mod, 'get_partitions', lambda disk: ['/dev/loop0p1', '/dev/md127p1', '/dev/loop0p11', '/dev/loop0p2'])
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


def _fake_blkid_pttype(monkeypatch, hpcp_mod, pttype_lines, calls=None):
	"""Answer `blkid -p ... PTTYPE` with pttype_lines (and record a failure like blkid's exit 2 when empty)."""
	def fake(argv, **kwargs):
		if calls is not None:
			calls.append(list(argv))
		assert argv[:2] == ['blkid', '-p'] and 'PTTYPE' in argv, argv
		if not pttype_lines:
			hpcp_mod.ERRORS.append('Task return code error: blkid exited 2')
		return list(pttype_lines)
	monkeypatch.setattr(hpcp_mod, 'run_command_in_multicmd_with_path_check', fake)


_NO_TABLE = ('blkid found no partition table on {0} (if it is a GPT disk, stale whole-disk signatures may hide it: '
	'check `wipefs -n {0}`)')


@pytest.mark.parametrize('lines, found', [(['dos'], 'dos'), ([], None), (['PMBR'], 'PMBR')])
def test_read_disk_layout_refuses_a_disk_without_gpt(hpcp_mod, reset_hpcp_globals, monkeypatch, lines, found):
	"""sgdisk reads an MBR disk as if it were GPT; a GPT identity change would then convert it.

	blkid also finds no partition table when its probe is ambivalent (stale whole-disk RAID or
	filesystem signatures next to a GPT): the message says what was found, not that there is no GPT."""
	_fake_blkid_pttype(monkeypatch, hpcp_mod, lines)
	def must_not_read(device):
		raise AssertionError('the GPT was read before the partition table type was checked')
	must_not_read.cache_clear = lambda: None
	monkeypatch.setattr(hpcp_mod, 'get_partition_infos', must_not_read)
	for kwargs, name in (({}, '/dev/x'), ({'name': '/images/new.img'}, '/images/new.img')):
		reason = _NO_TABLE.format(name) if found is None else f'{name} has no GPT partition table (found {found})'
		with pytest.raises(RuntimeError, match='^' + re.escape(f'Disk layout mismatch: {reason}; disk patches support GPT disks only') + '$'):
			hpcp_mod.read_disk_layout('/dev/x', **kwargs)
	assert hpcp_mod.ERRORS == []


def test_fs_params_text_ignores_device_paths(hpcp_mod):
	"""xfs_info names the device it read ('meta-data=/dev/loop3p3'); old and new always differ there."""
	def xfs(device):
		return {'meta-data': {'meta-data': device, 'isize': '512', 'agcount': '4'}, 'data': {'bsize': '4096', 'blocks': '10240'},
			'naming': {'version': '2', 'bsize': '4096'}}
	text = hpcp_mod._fs_params_text(xfs('/dev/loop3p3'))
	assert text == hpcp_mod._fs_params_text(xfs('/dev/loop9p3'))
	assert '/dev/' not in text and 'isize' in text and 'bsize' in text
	assert hpcp_mod._fs_params_text({'device': '/dev/sda1', 'block_size': 4096}) == 'block_size=4096'


def test_fs_params_text_ignores_ext_runtime_features(hpcp_mod):
	"""A live or dirty ext4 reports needs_recovery / orphan_present; they are state, not mkfs parameters."""
	clean = {'features': ['has_journal', 'ext_attr', 'extent', 'metadata_csum'], 'block_size': 4096, 'inode_size': 256}
	dirty = dict(clean, features=['has_journal', 'ext_attr', 'needs_recovery', 'extent', 'orphan_present', 'metadata_csum'])
	assert hpcp_mod._fs_params_text(dirty) == hpcp_mod._fs_params_text(clean)
	assert 'needs_recovery' not in hpcp_mod._fs_params_text(dirty)
	assert hpcp_mod._fs_params_text(dict(clean, features=['has_journal'])) != hpcp_mod._fs_params_text(clean)
	assert hpcp_mod._fs_params_text({}) == hpcp_mod._fs_params_text(None) == ''


def test_read_disk_layout_settles_udev_before_reading(hpcp_mod, monkeypatch):
	"""A node udev is still re-adding would otherwise probe its filesystem parameters as empty."""
	calls = []
	monkeypatch.setattr(hpcp_mod, '_partition_table_type', lambda device: 'gpt')
	monkeypatch.setattr(hpcp_mod, '_udev_settle', lambda: calls.append('settle'))
	def infos(device):
		calls.append('infos')
		return {'x': {'disk_identifier': 'G', 'sector_size': 512, 'size': 1 << 20}}
	infos.cache_clear = lambda: None
	monkeypatch.setattr(hpcp_mod, 'get_partition_infos', infos)
	monkeypatch.setattr(hpcp_mod, '_settle_partitions', lambda disk, numbers: calls.append('partitions') or {})
	assert hpcp_mod.read_disk_layout('/dev/x') == {'disk': {'guid': 'G', 'sector_size': 512, 'size': 1 << 20}, 'parts': {}}
	assert calls == ['settle', 'infos', 'partitions']


def test_partition_table_type_is_quiet(hpcp_mod, reset_hpcp_globals, monkeypatch):
	_fake_blkid_pttype(monkeypatch, hpcp_mod, ['gpt\n'])
	assert hpcp_mod._partition_table_type('/dev/x') == 'gpt'
	_fake_blkid_pttype(monkeypatch, hpcp_mod, [])
	assert hpcp_mod._partition_table_type('/dev/x') == ''
	assert hpcp_mod.ERRORS == []


def test_disk_apply_to_a_target_without_a_partition_table(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""No partition table at all is a wrong target (Invalid arguments); an MBR disk is a layout mismatch."""
	meta = {'layout': 'disk', 'disk': 'disk.tsv', 'sections': [], 'raws': []}
	monkeypatch.setattr(hpcp_mod, '_load_patch', lambda roots, names=None: (meta, [], list(roots)))
	monkeypatch.setattr(hpcp_mod, 'read_disk_tsv', lambda path: [])
	monkeypatch.setattr(hpcp_mod, '_refuse_if_image_in_use', lambda image: None)
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: '/dev/fake')
	target = tmp_path / 'fs.img'
	target.write_bytes(b'')
	_fake_blkid_pttype(monkeypatch, hpcp_mod, [])
	with pytest.raises(RuntimeError, match='^' + re.escape(f'Invalid arguments: {_NO_TABLE.format(target)}; '
			'a disk patch needs a whole GPT disk (block device or disk image)') + '$'):
		hpcp_mod.apply_disk_patch(['/vol'], str(target), [], [])
	_fake_blkid_pttype(monkeypatch, hpcp_mod, ['dos'])
	with pytest.raises(RuntimeError, match='^' + re.escape(f'Disk layout mismatch: {target} has no GPT partition table (found dos)')):
		hpcp_mod.apply_disk_patch(['/vol'], str(target), [], [])


# number: (kind, fs_type, size) of the golden layout _make_golden builds.
_GOLDEN_KINDS = {1: ('fs', 'vfat', 32 << 20), 2: ('raw', '', 1 << 20), 3: ('fs', 'ext4', 40 << 20), 4: ('swap', 'swap', 8 << 20)}


def _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch, sections=(), raws=(), kinds=None, open_target=False):
	"""A disk patch volume written by hand (a directory, no image) and the target layout it matches.

	sections are (name, [ChangeRow]); raws are manifest raw entries. Unless open_target, opening
	the target fails the test: these patches must be refused before the target is touched.
	"""
	kinds = kinds or _GOLDEN_KINDS
	volume = tmp_path / 'vol'
	diff_dir = volume / '.hpcp_diff'
	diff_dir.mkdir(parents=True)
	R = hpcp_mod.DiskRow
	disk_rows = [R('disk', '-', 'guid', 'G', 'G'), R('disk', '-', 'sector_size', '512', '512')]
	parts = {}
	for number, (kind, fs_type, size) in sorted(kinds.items()):
		values = {'kind': kind, 'type_guid': f'T{number}', 'unique_guid': f'U{number}', 'name': f'n{number}', 'attrs': '0',
			'size': str(size), 'fs_type': fs_type, 'fs_uuid': '', 'fs_label': '', 'fs_params': ''}
		disk_rows += [R('part', str(number), field, values[field], values[field]) for field in hpcp_mod._PART_FIELDS]
		parts[number] = dict(values, size=size, device=f'/dev/fake{number}')
	hpcp_mod.write_disk_tsv(str(diff_dir / 'disk.tsv'), disk_rows)
	for name, rows in sections:
		hpcp_mod.write_changes(str(diff_dir / f'{name}.changes.tsv'), rows)
	meta = {'format': 'hpcp-diff', 'format_version': 2, 'layout': 'disk', 'patch_id': 'p', 'volume': '1/1',
		'hash_algo': hpcp_mod._hash_algo_name(), 'hash_size': 65536, 'full_hash': 0, 'disk': 'disk.tsv',
		'sections': [(name, name, f'{name}.changes.tsv') for name, _ in sections], 'raws': list(raws)}
	hpcp_mod.write_manifest(str(diff_dir / 'manifest'), meta)
	if not open_target:
		def must_not_open(*args, **kwargs):
			raise AssertionError('the target was opened before the patch was checked')
		monkeypatch.setattr(hpcp_mod, '_attach_disk', must_not_open)
		monkeypatch.setattr(hpcp_mod, '_refuse_if_image_in_use', must_not_open)
	return str(volume), {'disk': {'guid': 'G', 'sector_size': 512, 'size': 96 << 20}, 'parts': parts}


def _one_row(hpcp_mod):
	return [hpcp_mod.ChangeRow('A', 'a.txt', '-', f'f:1:{T0}000000000:100644:0:0:-:-')]


@pytest.mark.parametrize('section, raw, message', [
	(None, ('mbr', 'raw/mbr-bootcode.bin', _C, _D, 512), 'raw area mbr is 512 bytes; the MBR boot code area is 440 bytes'),
	(None, ('part1', 'raw/part1.img', _A, _B, 32 << 20), 'raw area part1 is for partition 1, which the disk table records as fs, not a raw area'),
	(None, ('part4', 'raw/part4.img', _A, _B, 8 << 20), 'raw area part4 is for partition 4, which the disk table records as swap, not a raw area'),
	(None, ('part2', 'raw/part2.img', _A, _B, 2 << 20), 'raw area part2 is 2097152 bytes; the disk table records 1048576 for partition 2'),
	(None, ('part9', 'raw/part9.img', _A, _B, 1 << 20), 'raw area part9 is for partition 9, which the disk table does not list'),
	('part2', None, 'section part2 is for partition 2, which the disk table records as raw, not a filesystem'),
	('part4', None, 'section part4 is for partition 4, which the disk table records as swap, not a filesystem'),
	('part7', None, 'section part7 is for partition 7, which the disk table does not list'),
])
def test_disk_manifest_entries_must_match_the_disk_table(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, section, raw, message):
	"""Patch images are untrusted: a 512-byte mbr entry would write past the boot code into the partition table,
	and an entry for the wrong kind of partition was a KeyError traceback."""
	volume, _ = _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch, sections=[(section, _one_row(hpcp_mod))] if section else (),
		raws=[raw] if raw else ())
	with pytest.raises(RuntimeError, match='^Patch format error: ' + re.escape(message) + '$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])


def test_disk_table_kind_must_not_change(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	volume, _ = _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch)
	path = os.path.join(volume, '.hpcp_diff', 'disk.tsv')
	rows = [row._replace(old='raw') if (row.number, row.field) == ('3', 'kind') else row for row in hpcp_mod.read_disk_tsv(path)]
	hpcp_mod.write_disk_tsv(path, rows)
	with pytest.raises(RuntimeError, match=r'^Patch format error: the disk table changes the kind of partition 3 \(raw -> fs\)$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])


def test_in_use_raw_partition_is_refused_before_it_is_hashed(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""A live LVM PV (hundreds of GB) must not be read in full only to be refused."""
	raws = [('part2', 'raw/part2.img', _A, _B, 1 << 20), ('mbr', 'raw/mbr-bootcode.bin', _C, _D, 440)]
	volume, layout = _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch, raws=raws, open_target=True)
	payload = os.path.join(volume, 'raw', 'part2.img')
	os.makedirs(os.path.dirname(payload))
	with open(payload, 'wb') as f:
		f.truncate(1 << 20)
	monkeypatch.setattr(hpcp_mod, '_refuse_if_image_in_use', lambda image: None)
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: '/dev/fake')
	monkeypatch.setattr(hpcp_mod, '_partition_table_type', lambda device: 'gpt')
	monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk, name=None: layout)
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: [])
	monkeypatch.setattr(hpcp_mod, '_swap_devices', lambda: set())
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (7, 0))
	monkeypatch.setattr(hpcp_mod, '_partition_in_use', lambda device, devnum, entries, swaps: 'held by another device' if device == '/dev/fake2' else '')
	hashed = []
	# The disk's boot code is already new (skip); the area on the partition still holds the old bytes.
	answers = {'/dev/fake': _D, '/dev/fake2': _A, payload: _B}
	monkeypatch.setattr(hpcp_mod, '_hash_area', lambda path, offset=0, length=None: hashed.append(path) or answers[path])
	# Not "its raw area changes": unread, the area may already hold the new content (a live re-run).
	with pytest.raises(RuntimeError, match='^' + re.escape('Disk in use: partition 2 is held by another device; this patch carries its raw area, '
			'which cannot be checked or written while it is in use; apply the patch offline (an offline apply skips it if it is already done)') + '$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	assert '/dev/fake2' not in hashed and hashed == ['/dev/fake']


def test_several_in_use_raw_partitions_and_other_refusals_share_one_message(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	kinds = {1: ('fs', 'vfat', 32 << 20), 2: ('raw', '', 1 << 20), 5: ('raw', '', 1 << 20)}
	raws = [('part2', 'raw/part2.img', _A, _B, 1 << 20), ('part5', 'raw/part5.img', _A, _B, 1 << 20)]
	volume, layout = _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch, raws=raws, kinds=kinds, open_target=True)
	# The patch also renames partition 1, which is mounted.
	path = os.path.join(volume, '.hpcp_diff', 'disk.tsv')
	rows = [row._replace(new='ESP') if (row.number, row.field) == ('1', 'name') else row for row in hpcp_mod.read_disk_tsv(path)]
	hpcp_mod.write_disk_tsv(path, rows)
	monkeypatch.setattr(hpcp_mod, '_refuse_if_image_in_use', lambda image: None)
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: '/dev/fake')
	monkeypatch.setattr(hpcp_mod, '_partition_table_type', lambda device: 'gpt')
	monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk, name=None: layout)
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: [])
	monkeypatch.setattr(hpcp_mod, '_swap_devices', lambda: set())
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (7, 0))
	busy = {'/dev/fake1': 'mounted at /boot/efi', '/dev/fake2': 'held by another device', '/dev/fake5': 'active swap'}
	monkeypatch.setattr(hpcp_mod, '_partition_in_use', lambda device, devnum, entries, swaps: busy.get(device, ''))
	hashed = []
	monkeypatch.setattr(hpcp_mod, '_hash_area', lambda path, offset=0, length=None: hashed.append(path) or _A)
	with pytest.raises(RuntimeError, match='^' + re.escape('Disk in use: partition 1 is mounted at /boot/efi and its name changes; '
			'partition 2 is held by another device; this patch carries its raw area, which cannot be checked or written while it is in use; '
			'partition 5 is active swap; this patch carries its raw area, which cannot be checked or written while it is in use; '
			'apply the patch offline (an offline apply skips each one that is already done)') + '$'):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	assert hashed == []


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
def _mounted(device, fstype=None, options=None):
	mount_point = tempfile.mkdtemp()
	_run(*(['mount'] + (['-t', fstype] if fstype else []) + (['-o', options] if options else []) + [device, mount_point]))
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


def test_permissionless_filesystems_are_content_only(hpcp_mod):
	"""Two separate classes: ntfs3 / ntfs-3g keep no POSIX metadata but look names up by exact case;
	hfsplus keeps owner, group and mode on disk but looks names up case-insensitively."""
	content_only = ('vfat', 'fat', 'fat12', 'fat16', 'fat32', 'msdos', 'exfat', 'ntfs', 'ntfs3')
	assert [hpcp_mod._has_posix_metadata(t) for t in content_only] == [False] * len(content_only)
	assert [hpcp_mod._has_posix_metadata(t) for t in ('hfs', 'hfsplus', 'ext4', 'xfs', 'btrfs', '')] == [True] * 6
	folded = ('vfat', 'fat', 'fat12', 'fat16', 'fat32', 'msdos', 'exfat', 'hfs', 'hfsplus')
	assert [hpcp_mod._has_case_insensitive_names(t) for t in folded] == [True] * len(folded)
	assert [hpcp_mod._has_case_insensitive_names(t) for t in ('ntfs', 'ntfs3', 'ext4', 'xfs', 'btrfs', '')] == [False] * 6
	eq = hpcp_mod.states_equal
	base = 'f:10:1717000000000000000:100755:0:0:aaaa:1234'
	assert eq(base, 'f:10:1717000000000000000:100700:1000:1000:aaaa:5678', metadata=False)
	assert not eq(base, 'f:10:1717000000000000000:100700:0:0:aaaa:1234')
	assert not eq(base, 'f:11:1717000000000000000:100755:0:0:aaaa:1234', metadata=False)
	assert not eq(base, 'f:10:1717000000000000000:100755:0:0:bbbb:1234', metadata=False)
	assert not eq(base, 'f:10:1717000100000000000:100755:0:0:aaaa:1234', metadata=False)
	assert eq('d:40755:0:0:-', 'd:40700:1000:0:9999', metadata=False)
	assert not eq('d:40755:0:0:-', base, metadata=False) and not eq('?', '?', metadata=False)


def test_content_only_section_round_trip(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Mode-only differences make no rows; apply writes content and times but no mode; verify ignores modes."""
	hpcp_mod.HASH_SIZE = 65536
	new, base, volume, target = (tmp_path / name for name in ('new', 'base', 'vol', 'target'))
	for tree in (new, base):
		_write(tree / 'EFI' / 'BOOT' / 'same.efi', 'same')
		_write(tree / 'EFI' / 'BOOT' / 'grubx64.efi', 'old')
	_write(new / 'EFI' / 'BOOT' / 'grubx64.efi', 'new content')
	for name in ('a.efi', 'b.efi'):
		_write(new / 'EFI' / 'vendor' / name, name)
	for path in (new, new / 'EFI', new / 'EFI' / 'BOOT', new / 'EFI' / 'vendor'):
		os.chmod(str(path), 0o700)
	for path in (new / 'EFI' / 'BOOT' / 'same.efi', new / 'EFI' / 'vendor' / 'a.efi', new / 'EFI' / 'vendor' / 'b.efi'):
		os.chmod(str(path), 0o600)
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	assert len(hpcp_mod.compute_changes(new_map, base_map, max_workers=1)) == 8
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1, metadata=False)
	assert {(row.op, row.path) for row in rows} == {('M', 'EFI/BOOT/grubx64.efi'), ('A', 'EFI/vendor/'),
		('A', 'EFI/vendor/a.efi'), ('A', 'EFI/vendor/b.efi')}
	shutil.copytree(str(new), str(volume))
	shutil.copytree(str(base), str(target))
	os.chmod(str(target / 'EFI' / 'BOOT' / 'grubx64.efi'), 0o640)
	assert any(check.action == 'conflict' for check in hpcp_mod.check_patch_target(rows, str(target)))
	plan = hpcp_mod._plan_section('part1', rows, [str(volume)], str(target), True, False, 2, metadata=False)
	assert [check.action for check in plan.checks] == ['apply'] * 4
	summary = hpcp_mod._apply_section(plan, 2, False, False, True)
	assert summary['failed'] == 0 and summary['added'] == 3 and hpcp_mod.ERRORS == []
	for rel, data in (('EFI/BOOT/grubx64.efi', 'new content'), ('EFI/vendor/a.efi', 'a.efi'), ('EFI/vendor/b.efi', 'b.efi')):
		path = target / rel
		assert path.read_text() == data and os.stat(str(path)).st_mtime == T0
		# Modes are the target's own, not the payload's 0600 (or 0700 for the new directory).
		assert stat.S_IMODE(os.stat(str(path)).st_mode) != 0o600
	umask = os.umask(0o022)
	os.umask(umask)
	assert stat.S_IMODE(os.stat(str(target / 'EFI' / 'vendor')).st_mode) == 0o777 & ~umask
	assert hpcp_mod._verify_rows(rows, str(target), metadata=False) == 0
	assert hpcp_mod._verify_rows(rows, str(target)) > 0


def _case_changes(hpcp_mod, new, base, metadata=False, case_insensitive=True):
	"""Rows of a FAT-like section by default (content only, names paired case-insensitively)."""
	rows = hpcp_mod._partition_changes(str(new), str(base), [], [], None, False, 1, False, metadata=metadata,
		labels=('part1 of new.img', 'part1 of old.img'), case_insensitive=case_insensitive)
	return {(row.op, row.path) for row in rows}


def test_content_only_sections_pair_names_case_insensitively(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""A case-only rename on FAT must not become a D + A pair: the target's case-insensitive lookup finds
	the A rows already present (skip), then the D rows delete those very files."""
	hpcp_mod.HASH_SIZE = 65536
	new, base = tmp_path / 'new', tmp_path / 'base'
	_write(base / 'EFI' / 'BOOT' / 'bootx64.efi', 'shim')
	_write(base / 'EFI' / 'BOOT' / 'grubx64.efi', 'old')
	_write(base / 'EFI' / 'BOOT' / 'old.efi', 'gone soon')
	_write(base / 'efi.cfg', 'was a file')
	_write(new / 'EFI' / 'Boot' / 'BOOTX64.EFI', 'shim')
	_write(new / 'EFI' / 'Boot' / 'GRUBX64.efi', 'new content')
	_write(new / 'EFI.CFG' / 'now.cfg', 'now a directory')
	# Same content: no row. New content: an M row in the new spelling. Deleted below a renamed directory:
	# a D row in the new spelling. A type change keeps its D + A pair on one (new) spelling.
	assert _case_changes(hpcp_mod, new, base) == {('M', 'EFI/Boot/GRUBX64.efi'), ('D', 'EFI/Boot/old.efi'),
		('D', 'EFI.CFG'), ('A', 'EFI.CFG/'), ('A', 'EFI.CFG/now.cfg')}
	# Sections whose names are case-sensitive (POSIX, NTFS) are not paired.
	for metadata in (True, False):
		exact = _case_changes(hpcp_mod, new, base, metadata=metadata, case_insensitive=False)
		assert {('D', 'EFI/BOOT/'), ('A', 'EFI/Boot/'), ('D', 'efi.cfg')} <= exact
	# HFS+ (POSIX metadata, case-insensitive names) pairs them and still compares modes.
	os.chmod(str(new / 'EFI' / 'Boot' / 'BOOTX64.EFI'), 0o700)
	assert _case_changes(hpcp_mod, new, base) == {('M', 'EFI/Boot/GRUBX64.efi'), ('D', 'EFI/Boot/old.efi'),
		('D', 'EFI.CFG'), ('A', 'EFI.CFG/'), ('A', 'EFI.CFG/now.cfg')}
	assert _case_changes(hpcp_mod, new, base, metadata=True) == {('M', 'EFI/Boot/BOOTX64.EFI'), ('M', 'EFI/Boot/GRUBX64.efi'),
		('D', 'EFI/Boot/old.efi'), ('D', 'EFI.CFG'), ('A', 'EFI.CFG/'), ('A', 'EFI.CFG/now.cfg')}


@pytest.mark.parametrize('side', ['new', 'old'])
def test_names_differing_only_by_case_refuse_content_only_generation(tmp_path, hpcp_mod, reset_hpcp_globals, side):
	hpcp_mod.HASH_SIZE = 65536
	new, base = tmp_path / 'new', tmp_path / 'base'
	_write(new / 'EFI' / 'x.efi', 'x')
	_write(base / 'EFI' / 'x.efi', 'x')
	crowded = new if side == 'new' else base
	_write(crowded / 'EFI' / 'Boot' / 'a.efi', 'a')
	_write(crowded / 'EFI' / 'BOOT' / 'a.efi', 'A')
	with pytest.raises(RuntimeError, match=rf'^Diff image error: part1 of {side}\.img has names that differ only by case '
			r'\(EFI/BOOT/, EFI/Boot/\); a case-insensitive filesystem cannot hold both$'):
		_case_changes(hpcp_mod, new, base)
	with pytest.raises(RuntimeError, match='a case-insensitive filesystem cannot hold both'):
		_case_changes(hpcp_mod, new, base, metadata=True)
	# A section with case-sensitive names (POSIX, NTFS) may hold both.
	for metadata in (True, False):
		assert ('A' if side == 'new' else 'D', 'EFI/BOOT/a.efi') in _case_changes(hpcp_mod, new, base, metadata=metadata, case_insensitive=False)


def _patch_row(hpcp_mod, op, path):
	state = 'd:40755:0:0:-' if path.endswith('/') else f'f:1:{T0}000000000:100644:0:0:-:-'
	return hpcp_mod.ChangeRow(op, path, '-' if op == 'A' else state, '-' if op == 'D' else state)


@pytest.mark.parametrize('deleted, written, refused', [
	('EFI/BOOT/', 'EFI/Boot/', True),
	('EFI/BOOT/grubx64.efi', 'EFI/BOOT/GRUBX64.EFI', True),
	('EFI/BOOT/grubx64.efi', 'EFI/BOOT/GRUBX64.EFI M', True),
	('EFI/BOOT/', 'EFI/Boot/grubx64.efi', True),
	('EFI/BOOT/grubx64.efi', 'EFI/boot/', True),
	('x', 'x/', False),
	('x', 'x/child', False),
	('EFI/BOOT/old.efi', 'EFI/Boot/new.efi', False),
	('EFI/BOOT/old.efi', 'EFI/BOOT/new.efi', False),
])
def test_case_collision_finds_deletes_and_writes_that_differ_only_by_case(hpcp_mod, deleted, written, refused):
	op = 'M' if written.endswith(' M') else 'A'
	written = written[:-2] if op == 'M' else written
	rows = [_patch_row(hpcp_mod, 'D', deleted), _patch_row(hpcp_mod, op, written)]
	assert hpcp_mod._case_collision(rows) == ((deleted, written) if refused else None)
	assert hpcp_mod._case_collision(rows[1:]) is None


def test_directory_delete_matches_children_case_insensitively_on_case_insensitive_targets(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""A FAT target keeps its old spelling when a case-only rename made no row; a later patch that deletes
	the directory names its children in the new spelling, which must not count as unlisted entries."""
	hpcp_mod.HASH_SIZE = 65536
	target = tmp_path / 'target'
	_write(target / 'EFI' / 'old' / 'Grub.CFG', 'cfg')
	_write(target / 'EFI' / 'old' / 'sub' / 'X.efi', 'x')
	R = hpcp_mod.ChangeRow
	directory, file = 'd:40755:0:0:-', f'f:3:{T0}000000000:100644:0:0:-:-'
	rows = [R('D', 'EFI/old/', directory, '-'), R('D', 'EFI/old/grub.cfg', file, '-'),
		R('D', 'EFI/old/SUB/', directory, '-'), R('D', 'EFI/old/SUB/x.efi', file, '-')]
	def directory_check(**kwargs):
		(check,) = [check for check in hpcp_mod.check_patch_target(rows, str(target), metadata=False, **kwargs) if check.row.path == 'EFI/old/']
		return check.action, check.reason
	assert directory_check(case_insensitive=True) == ('apply', '')
	assert directory_check() == ('conflict', 'directory holds entries the patch does not delete')
	plan = hpcp_mod._plan_section('part1', rows, [str(tmp_path / 'vol')], str(target), True, False, 1, metadata=False, case_insensitive=True)
	assert [check.action for check in plan.checks if check.row.path == 'EFI/old/'] == ['apply']
	# An entry the patch does not delete is still one, whatever its case.
	_write(target / 'EFI' / 'old' / 'Extra.efi', 'e')
	assert directory_check(case_insensitive=True) == ('conflict', 'directory holds entries the patch does not delete')


def test_content_only_verify_covers_rows_skipped_as_done(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""A row skipped as already done that the apply then loses (a D row on a name that differs only by
	case, on FAT or HFS+) must be a verify failure in a section with case-insensitive names; other
	sections (POSIX, NTFS) verify what they apply."""
	hpcp_mod.HASH_SIZE = 65536
	new, base, volume = tmp_path / 'new', tmp_path / 'base', tmp_path / 'vol'
	_write(new / 'EFI' / 'keep.efi', 'keep')
	_write(new / 'EFI' / 'mod.efi', 'new')
	_write(base / 'EFI' / 'mod.efi', 'old')
	new_map, base_map = hpcp_mod.build_root_maps(str(new), str(base), max_workers=1)
	rows = hpcp_mod.compute_changes(new_map, base_map, max_workers=1, metadata=False)
	assert {(row.op, row.path) for row in rows} == {('A', 'EFI/keep.efi'), ('M', 'EFI/mod.efi')}
	shutil.copytree(str(new), str(volume))
	for metadata, case_insensitive in ((False, True), (False, False), (True, True), (True, False)):
		target = tmp_path / f'target-{metadata}-{case_insensitive}'
		shutil.copytree(str(base), str(target))
		_write(target / 'EFI' / 'keep.efi', 'keep')
		plan = hpcp_mod._plan_section('part1', rows, [str(volume)], str(target), True, False, 1, metadata=metadata,
			case_insensitive=case_insensitive)
		assert {check.row.path: check.action for check in plan.checks} == {'EFI/keep.efi': 'skip', 'EFI/mod.efi': 'apply'}
		os.unlink(str(target / 'EFI' / 'keep.efi'))
		hpcp_mod.ERRORS.clear()
		summary = hpcp_mod._apply_section(plan, 1, False, False, True)
		assert (target / 'EFI' / 'mod.efi').read_text() == 'new'
		if not case_insensitive:
			assert summary['failed'] == 0 and hpcp_mod.ERRORS == []
		else:
			assert summary['failed'] == 1 and summary['skipped'] == 1
			assert len(hpcp_mod.ERRORS) == 1 and hpcp_mod.ERRORS[0].startswith('Patch verify error: A EFI/keep.efi: expected f:')


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


def _read_part3_patch(hpcp_mod, out, section='part3'):
	with _session(hpcp_mod) as (mount_points, loop_devices):
		root = hpcp_mod._mount_single_fs_image(out, True, mount_points, loop_devices)
		meta = hpcp_mod.read_manifest(os.path.join(root, '.hpcp_diff', 'manifest'))
		rows = hpcp_mod.read_changes(os.path.join(root, '.hpcp_diff', f'{section}.changes.tsv'))
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


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_input_disk_image_on_a_live_partition_stays_out_of_the_patch(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""/root/golden-v1.img on the golden's live root must not become an A row installed on every node."""
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_make_newer(old, new, label=None, raw=False, mbr=False, gpt_name=None)
	with _attached(new) as loop:
		with _mounted(loop + 'p3') as live_root:
			inside = os.path.join(live_root, 'root', 'golden-v1.img')
			os.makedirs(os.path.dirname(inside))
			_run('cp', '--sparse=always', old, inside)
			with _session(hpcp_mod) as (mount_points, loop_devices):
				assert hpcp_mod.process_get_disk_diff_image([loop], [inside], out, mount_points, loop_devices, max_workers=1) == out
	_, part3 = _read_part3_patch(hpcp_mod, out)
	paths = {row.path for row in part3}
	assert 'root/golden-v1.img' not in paths
	assert 'root/' in paths and ('M', 'etc/os-release') in {(row.op, row.path) for row in part3}


def test_unmountable_filesystem_is_compared_as_raw_with_a_warning(hpcp_mod, monkeypatch, capsys):
	device = '/dev/none'
	parts = {1: _part(number=1, device=device, fs_type='ext4')}
	layout = _layout(parts)
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: path)
	monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk, name=None: layout)
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: [])
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda *a: None)
	monkeypatch.setattr(hpcp_mod, '_hash_area', lambda *a: 'h')
	monkeypatch.setattr(hpcp_mod, 'expand_paths', lambda paths: list(paths))
	monkeypatch.setattr(hpcp_mod, '_swap_devices', lambda: set())
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (7, 0))
	monkeypatch.setattr(hpcp_mod, '_partition_in_use', lambda device, devnum, entries, swaps: '')
	assert hpcp_mod.process_get_disk_diff_image(['/x/new'], ['/x/old'], '/x/out', [], [], max_workers=1) is None
	assert 'partition 1 (ext4) could not be mounted on the new and old side; comparing it as a raw area' in capsys.readouterr().err


def _fake_generation(hpcp_mod, monkeypatch, new_parts, old_parts, in_use=None):
	"""Stub the disk side of process_get_disk_diff_image for disks /x/new and /x/old.

	Raw areas hash to their device path (so old and new always differ); returns the list of
	hashed devices. in_use maps a partition device to _partition_in_use's answer.
	"""
	layouts = {'/x/new': _layout(new_parts), '/x/old': _layout(old_parts)}
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: path)
	monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk, name=None: layouts[disk])
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: [])
	monkeypatch.setattr(hpcp_mod, '_swap_devices', lambda: set())
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (7, 0))
	monkeypatch.setattr(hpcp_mod, '_partition_in_use', lambda device, devnum, entries, swaps: (in_use or {}).get(device, ''))
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda *a: None)
	monkeypatch.setattr(hpcp_mod, 'expand_paths', lambda paths: list(paths))
	def must_not_build(*args, **kwargs):
		raise AssertionError('a stubbed generation must not create a patch image')
	monkeypatch.setattr(hpcp_mod, '_claim_volume_names', must_not_build)
	monkeypatch.setattr(hpcp_mod, 'create_image', must_not_build)
	hashed = []
	monkeypatch.setattr(hpcp_mod, '_hash_area', lambda path, offset=0, length=None: hashed.append(path) or path)
	return hashed


@pytest.mark.parametrize('busy, side', [('/dev/new2', '/x/new'), ('/dev/old2', '/x/old')])
def test_in_use_raw_area_is_refused_before_anything_is_hashed(hpcp_mod, reset_hpcp_globals, monkeypatch, busy, side):
	"""An LVM PV holding the live root would be hashed and copied in full, then fail as changed."""
	new_parts = {2: _part(fs_type='', size=1 << 20, device='/dev/new2'), 4: _part(fs_type='swap', device='/dev/new4')}
	old_parts = {2: _part(fs_type='', size=1 << 20, device='/dev/old2'), 4: _part(fs_type='swap', device='/dev/old4')}
	hashed = _fake_generation(hpcp_mod, monkeypatch, new_parts, old_parts, in_use={busy: 'held by another device'})
	with pytest.raises(RuntimeError, match=rf'^Disk in use: partition 2 of {side} is held by another device; a raw area cannot be copied while it is in use'):
		hpcp_mod.process_get_disk_diff_image(['/x/new'], ['/x/old'], '/x/out', [], [], max_workers=1)
	assert hashed == []


def test_active_swap_and_unchanged_raw_areas_are_not_refused(hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Swap is identity-only, so an active swap partition never blocks generation."""
	new_parts = {4: _part(fs_type='swap', device='/dev/new4')}
	old_parts = {4: _part(fs_type='swap', device='/dev/old4')}
	_fake_generation(hpcp_mod, monkeypatch, new_parts, old_parts, in_use={'/dev/new4': 'active swap', '/dev/old4': 'active swap'})
	monkeypatch.setattr(hpcp_mod, '_hash_area', lambda *a, **k: 'same')
	assert hpcp_mod.process_get_disk_diff_image(['/x/new'], ['/x/old'], '/x/out', [], [], max_workers=1) is None


def test_unmountable_in_use_filesystem_is_refused_as_a_raw_area(hpcp_mod, reset_hpcp_globals, monkeypatch):
	new_parts = {1: _part(fs_type='ext4', device='/dev/new1')}
	old_parts = {1: _part(fs_type='ext4', device='/dev/old1')}
	hashed = _fake_generation(hpcp_mod, monkeypatch, new_parts, old_parts, in_use={'/dev/new1': 'held by another device'})
	with pytest.raises(RuntimeError, match=r'^Disk in use: partition 1 of /x/new is held by another device'):
		hpcp_mod.process_get_disk_diff_image(['/x/new'], ['/x/old'], '/x/out', [], [], max_workers=1)
	assert hashed == []


def test_raw_area_too_large_for_one_volume_suggests_a_dis_that_passes(hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	"""The suggested -dis is a whole number of MiB the -dis parser reads back, and it is large enough."""
	size = (1 << 30) + 12345 * 512
	hashed = _fake_generation(hpcp_mod, monkeypatch, {2: _part(fs_type='', size=size, device='/dev/new2')},
		{2: _part(fs_type='', size=size, device='/dev/old2')})
	with pytest.raises(RuntimeError, match=r'must fit in one patch volume') as error:
		hpcp_mod.process_get_disk_diff_image(['/x/new'], ['/x/old'], '/x/out', [], [], max_workers=1, dest_image_size=256 << 20)
	match = re.search(r'use -dis (\d+)MiB or larger$', str(error.value))
	assert match, str(error.value)
	suggested = hpcp_mod.format_bytes(f'{match.group(1)}MiB', to_int=True)
	assert suggested == int(match.group(1)) << 20
	count, volume_size = hpcp_mod._plan_dest_images(size + 440, suggested)
	assert count == 1 or size <= int(volume_size * 0.90) - (16 << 20)
	# One MiB less would not always do: the suggestion is rounded up, never down.
	assert int(hpcp_mod._align_up(suggested - (1 << 20)) * 0.90) - (16 << 20) < size
	# The planned sizes are printed before any volume is created.
	assert re.search(r'Disk changes: 0 filesystem section\(s\), 2 raw area\(s\), 0 identity value\(s\); 0 added, 0 modified, 0 deleted; '
		r'payload 0\.00 B, raw 1\.\d\d GiB', capsys.readouterr().out)
	assert '/dev/new2' in hashed


def test_raw_areas_are_stored_before_section_payloads_largest_first(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""Section payloads fill the first volume first; a large raw area stored after them could fail to fit late."""
	parts = lambda side: {1: _part(fs_type='vfat', device=f'/dev/{side}1'), 2: _part(fs_type='', size=1 << 20, device=f'/dev/{side}2'),
		5: _part(fs_type='', size=4 << 20, device=f'/dev/{side}5')}
	_fake_generation(hpcp_mod, monkeypatch, parts('new'), parts('old'))
	calls = []
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda device, read_only, mount_points, entries: str(tmp_path / device[5:]) + os.sep)
	monkeypatch.setattr(hpcp_mod, '_release_mount', lambda mount_point, mount_points: None)
	monkeypatch.setattr(hpcp_mod, '_live_mount', lambda *args: (None, False))
	monkeypatch.setattr(hpcp_mod, '_partition_changes', lambda *args, **kwargs: _one_row(hpcp_mod))
	monkeypatch.setattr(hpcp_mod, '_claim_volume_names', lambda output, count: [output])
	monkeypatch.setattr(hpcp_mod, 'create_image', lambda *args, **kwargs: [str(tmp_path / 'vol')])
	monkeypatch.setattr(hpcp_mod, '_build_payload', lambda rows, new_root, roots, *args: calls.append(('payload', new_root)))
	monkeypatch.setattr(hpcp_mod, '_store_raw', lambda roots, raw_file, device, size, new_sha: calls.append(('raw', raw_file)))
	monkeypatch.setattr(hpcp_mod, 'write_disk_patch_metadata', lambda *args: calls.append(('metadata',)))
	monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
	out = str(tmp_path / 'patch.img')
	assert hpcp_mod.process_get_disk_diff_image(['/x/new'], ['/x/old'], out, [], [], max_workers=1) == out
	assert calls == [('raw', 'raw/part5.img'), ('raw', 'raw/part2.img'), ('raw', 'raw/mbr-bootcode.bin'),
		('payload', str(tmp_path / 'new1') + os.sep), ('metadata',)]


@pytest.mark.parametrize('fs_type, expected', [
	# FAT: content only, names paired case-insensitively.
	('vfat', {('M', 'changed.txt')}),
	# NTFS: content only, but ntfs3 / ntfs-3g look names up by exact case: a case-only rename stays D + A.
	('ntfs', {('M', 'changed.txt'), ('D', 'Docs/'), ('D', 'Docs/readme.txt'), ('A', 'DOCS/'), ('A', 'DOCS/readme.txt')}),
	('ntfs3', {('M', 'changed.txt'), ('D', 'Docs/'), ('D', 'Docs/readme.txt'), ('A', 'DOCS/'), ('A', 'DOCS/readme.txt')}),
	# HFS+: names paired case-insensitively, and hfsplus keeps modes on disk, so a chmod is a change.
	('hfsplus', {('M', 'changed.txt'), ('M', 'tool.sh')}),
])
def test_generation_classifies_sections_by_filesystem_type(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, fs_type, expected):
	hpcp_mod.HASH_SIZE = 65536
	new, old = tmp_path / 'new1', tmp_path / 'old1'
	_write(old / 'Docs' / 'readme.txt', 'same')
	_write(new / 'DOCS' / 'readme.txt', 'same')
	_write(old / 'changed.txt', 'old')
	_write(new / 'changed.txt', 'new content')
	for tree, mode in ((old, 0o644), (new, 0o755)):
		os.chmod(_write(tree / 'tool.sh', 'run'), mode)
	parts = lambda side: {1: _part(fs_type=fs_type, device=f'/dev/{side}1')}
	_fake_generation(hpcp_mod, monkeypatch, parts('new'), parts('old'))
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda device, read_only, mount_points, entries: str(tmp_path / device[5:]) + os.sep)
	monkeypatch.setattr(hpcp_mod, '_release_mount', lambda mount_point, mount_points: None)
	monkeypatch.setattr(hpcp_mod, '_live_mount', lambda *args: (None, False))
	monkeypatch.setattr(hpcp_mod, '_claim_volume_names', lambda output, count: [output])
	monkeypatch.setattr(hpcp_mod, 'create_image', lambda *args, **kwargs: [str(tmp_path / 'vol')])
	sections = []
	monkeypatch.setattr(hpcp_mod, '_build_payload', lambda rows, new_root, roots, *args: sections.append({(row.op, row.path) for row in rows}))
	monkeypatch.setattr(hpcp_mod, '_store_raw', lambda *args: None)
	monkeypatch.setattr(hpcp_mod, 'write_disk_patch_metadata', lambda *args: None)
	monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
	out = str(tmp_path / 'patch.img')
	assert hpcp_mod.process_get_disk_diff_image(['/x/new'], ['/x/old'], out, [], [], max_workers=1) == out
	assert sections == [expected]


@pytest.mark.parametrize('fs_type, refused', [('vfat', True), ('exfat', True), ('hfsplus', True), ('hfs', True),
	('ntfs', False), ('ntfs3', False), ('ext4', False)])
def test_case_colliding_rows_are_refused_only_on_case_insensitive_filesystems(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, fs_type, refused):
	"""On NTFS (exact-case lookups) a delete of EFI/BOOT/x.efi and an add of EFI/Boot/x.efi are two names."""
	rows = [_patch_row(hpcp_mod, 'D', 'EFI/BOOT/x.efi'), _patch_row(hpcp_mod, 'A', 'EFI/Boot/x.efi')]
	volume, _ = _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch, sections=[('part1', rows)], kinds={1: ('fs', fs_type, 32 << 20)})
	if refused:
		with pytest.raises(RuntimeError, match=r'^Patch format error: section part1 deletes and writes EFI/BOOT/x\.efi and EFI/Boot/x\.efi'):
			hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	else:
		# Past the format checks: the next step opens the target.
		with pytest.raises(AssertionError, match='the target was opened before the patch was checked'):
			hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])


@pytest.mark.parametrize('fs_type, metadata, case_insensitive', [('vfat', False, True), ('exfat', False, True), ('ntfs', False, False),
	('hfsplus', True, True), ('ext4', True, False)])
def test_disk_apply_plans_each_section_by_filesystem_class(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, fs_type, metadata, case_insensitive):
	volume, layout = _hand_made_disk_patch(hpcp_mod, tmp_path, monkeypatch, sections=[('part1', _one_row(hpcp_mod))],
		kinds={1: ('fs', fs_type, 32 << 20)}, open_target=True)
	target_root = tmp_path / 'target'
	target_root.mkdir()
	monkeypatch.setattr(hpcp_mod, '_refuse_if_image_in_use', lambda image: None)
	monkeypatch.setattr(hpcp_mod, '_attach_disk', lambda path, ro, loops: '/dev/fake')
	monkeypatch.setattr(hpcp_mod, '_partition_table_type', lambda device: 'gpt')
	monkeypatch.setattr(hpcp_mod, 'read_disk_layout', lambda disk, name=None: layout)
	monkeypatch.setattr(hpcp_mod, '_mountinfo_entries', lambda: [])
	monkeypatch.setattr(hpcp_mod, '_swap_devices', lambda: set())
	monkeypatch.setattr(hpcp_mod, '_device_number', lambda device: (7, 0))
	monkeypatch.setattr(hpcp_mod, '_partition_in_use', lambda device, devnum, entries, swaps: '')
	monkeypatch.setattr(hpcp_mod, '_mount_partition', lambda device, read_only, mount_points, entries: str(target_root) + os.sep)
	planned = []
	class Planned(Exception):
		pass
	def stop(plans, force, extra_lines=()):
		planned.extend(plans)
		raise Planned()
	monkeypatch.setattr(hpcp_mod, '_resolve_section_conflicts', stop)
	with pytest.raises(Planned):
		hpcp_mod.apply_disk_patch([volume], str(tmp_path / 'node.img'), [], [])
	assert [(plan.name, plan.metadata, plan.case_insensitive) for plan in planned] == [('part1', metadata, case_insensitive)]


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


def _add_esp_file(image, rel, data):
	with _attached(image) as loop:
		with _mounted(loop + 'p1') as esp:
			_write(os.path.join(esp, rel), data)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_esp_live_mounted_with_umask_gives_content_only_rows(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""FAT modes come from mount options: the live golden's ESP (umask=0077) against an image ESP hpcp
	mounts with defaults must not make a mode-only row, and the patch must apply and verify offline."""
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_make_newer(old, new)
	with _attached(new) as loop:
		with _mounted(loop + 'p1', options='umask=0077') as esp:
			assert stat.S_IMODE(os.stat(os.path.join(esp, 'EFI', 'BOOT', 'grubx64.efi')).st_mode) == 0o700
			with _session(hpcp_mod) as (mount_points, loop_devices):
				assert hpcp_mod.process_get_disk_diff_image([loop], [str(old)], out, mount_points, loop_devices, max_workers=1) == out
	assert hpcp_mod.ERRORS == []
	_, part1 = _read_part3_patch(hpcp_mod, out, 'part1')
	assert {(row.op, row.path) for row in part1} == {('M', 'EFI/BOOT/grubx64.efi')}
	_, part3 = _read_part3_patch(hpcp_mod, out)
	assert ('M', 'etc/os-release') in {(row.op, row.path) for row in part3}
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	summary = _apply(hpcp_mod, out, target)
	assert summary['failed'] == 0 and hpcp_mod.ERRORS == []
	assert summary['sections']['part1']['modified'] == 1
	assert _no_differences(hpcp_mod, tmp_path, new, target)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_offline_patch_applies_to_an_esp_live_mounted_with_umask(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch):
	"""An offline-generated patch (ESP modes 0755) against a node whose ESP is live at umask=0077:
	no conflict for content the patch expects, no prompt without a tty, content applied and verified."""
	old, new, out = tmp_path / 'old.img', tmp_path / 'new.img', str(tmp_path / 'patch.img')
	_make_golden(old)
	_make_newer(old, new, label=None, raw=False, mbr=False, gpt_name=None)
	_add_esp_file(new, 'EFI/vendor/shim.efi', b'shim')
	with _session(hpcp_mod) as (mount_points, loop_devices):
		assert hpcp_mod.process_get_disk_diff_image([str(new)], [str(old)], out, mount_points, loop_devices, max_workers=1) == out
	_, part1 = _read_part3_patch(hpcp_mod, out, 'part1')
	assert {(row.op, row.path) for row in part1} == {('M', 'EFI/BOOT/grubx64.efi'), ('A', 'EFI/vendor/'), ('A', 'EFI/vendor/shim.efi')}
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	monkeypatch.setattr(sys, 'stdin', _FakeStdin(False))
	with _attached(target) as loop:
		with _mounted(loop + 'p1', options='umask=0077') as esp:
			summary = _apply(hpcp_mod, out, loop)
			assert summary['failed'] == 0 and summary['forced'] == 0 and hpcp_mod.ERRORS == []
			assert summary['sections']['part1']['modified'] == 1 and summary['sections']['part1']['added'] == 2
			efi = os.path.join(esp, 'EFI', 'BOOT', 'grubx64.efi')
			with open(efi, 'rb') as f:
				assert f.read() == b'new-efi-binary'
			assert os.stat(efi).st_mtime == T0
			with open(os.path.join(esp, 'EFI', 'vendor', 'shim.efi'), 'rb') as f:
				assert f.read() == b'shim'
			# A re-run finds everything already done.
			again = _apply(hpcp_mod, out, loop)
			assert again['failed'] == 0 and again['sections']['part1']['skipped'] == 3
	assert _no_differences(hpcp_mod, tmp_path, new, target)


def _rename_on_esp(image, renames):
	"""Rename ESP entries; vfat ignores a rename to a name that differs only by case, so go through a temporary name."""
	with _attached(image) as loop:
		with _mounted(loop + 'p1') as esp:
			for old_rel, new_rel in renames:
				temporary = os.path.join(esp, 'hpcp-rename.tmp')
				os.rename(os.path.join(esp, old_rel), temporary)
				os.rename(temporary, os.path.join(esp, new_rel))
			return sorted(os.path.relpath(os.path.join(top, name), esp) for top, dirs, files in os.walk(esp) for name in dirs + files)


def _esp_files(image):
	with _attached(image, read_only=True) as loop:
		with _mounted(loop + 'p1', options='ro') as esp:
			result = {}
			for top, _, files in os.walk(esp):
				for name in files:
					with open(os.path.join(top, name), 'rb') as f:
						result[os.path.relpath(os.path.join(top, name), esp)] = f.read()
			return result


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_esp_case_only_rename_keeps_the_files(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""Finding: EFI/BOOT -> EFI/Boot made D + A rows; the A rows were skipped as already present on the
	case-insensitive target and the D rows then deleted the boot files (exit 0, empty EFI/)."""
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
	assert {(row.op, row.path) for row in part1} == {('M', 'EFI/Boot/GRUBX64.efi')}
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	summary = _apply(hpcp_mod, out, target)
	assert summary['failed'] == 0 and hpcp_mod.ERRORS == []
	assert summary['sections']['part1'] == {'deleted': 0, 'added': 0, 'modified': 1, 'skipped': 0, 'forced': 0, 'failed': 0}
	files = {name.lower(): data for name, data in _esp_files(target).items()}
	assert files == {'efi/boot/bootx64.efi': b'shim-v1', 'efi/boot/grubx64.efi': b'new-efi-binary'}
	assert _no_differences(hpcp_mod, tmp_path, new, target)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_hand_made_case_colliding_delete_and_add_is_refused_before_anything_changes(tmp_path, hpcp_mod, reset_hpcp_globals):
	"""A patch from older code (or a crafted one) that deletes EFI/BOOT/grubx64.efi and adds EFI/Boot/grubx64.efi."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	with _attached(patch) as loop:
		with _mounted(loop) as root:
			changes = os.path.join(root, '.hpcp_diff', 'part1.changes.tsv')
			(row,) = hpcp_mod.read_changes(changes)
			assert (row.op, row.path) == ('M', 'EFI/BOOT/grubx64.efi')
			R = hpcp_mod.ChangeRow
			hpcp_mod.write_changes(changes, [R('D', row.path, row.old, '-'), R('A', 'EFI/Boot/grubx64.efi', '-', row.new)])
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	before = _sha(target.read_bytes())
	with pytest.raises(RuntimeError, match=r'^Patch format error: section part1 deletes and writes EFI/BOOT/grubx64\.efi and '
			r'EFI/Boot/grubx64\.efi differing only by case; regenerate the patch with this hpcp$'):
		_apply(hpcp_mod, patch, target)
	assert _sha(target.read_bytes()) == before


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_failed_section_stops_before_esp_raw_boot_code_and_identity(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	"""A root section that fails (ENOSPC, say) must not be followed by the ESP, raw areas, boot code or identity changes."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	real_apply_section = hpcp_mod._apply_section
	applied = []
	def failing(plan, *args):
		applied.append(plan.name)
		if plan.name == 'part3':
			hpcp_mod.eprint('Patch verify error: M etc/os-release: simulated failure')
			return {'deleted': 0, 'added': 0, 'modified': 1, 'skipped': 0, 'forced': 0, 'failed': 1}
		return real_apply_section(plan, *args)
	monkeypatch.setattr(hpcp_mod, '_apply_section', failing)
	capsys.readouterr()
	summary = _apply(hpcp_mod, patch, target)
	assert applied == ['part3']
	assert summary['failed'] == 1 and summary['raw_written'] == 0 and summary['identity_changed'] == 0
	assert list(summary['sections']) == ['part3']
	captured = capsys.readouterr()
	assert 'Section part3:' in captured.out and 'Section part1:' not in captured.out and 'Disk patch summary:' in captured.out
	stop = [error for error in hpcp_mod.ERRORS if 'section part3 failed' in error]
	assert len(stop) == 1 and stop[0].startswith('Patch verify error: ') and stop[0] in captured.err
	assert 'later sections, raw areas, boot code and identity changes were not applied' in stop[0]
	with open(str(target), 'rb') as f:
		assert f.read(14) == b'\xeb\x63OLD-BOOTCODE'
	with _attached(target, read_only=True) as loop:
		layout = hpcp_mod.read_disk_layout(loop)
		assert layout['parts'][1]['name'] == 'ESP' and layout['parts'][3]['fs_label'] == 'root'
		with open(loop + 'p2', 'rb') as f:
			assert f.read(8) == b'GRUBCORE'
		with _mounted(loop + 'p1', options='ro') as esp:
			with open(os.path.join(esp, 'EFI', 'BOOT', 'grubx64.efi'), 'rb') as f:
				assert f.read() == b'old-efi'
	# Once the cause is fixed, a re-run completes the patch.
	monkeypatch.setattr(hpcp_mod, '_apply_section', real_apply_section)
	hpcp_mod.ERRORS.clear()
	again = _apply(hpcp_mod, patch, target)
	assert again['failed'] == 0 and again['raw_written'] == 2 and again['identity_changed'] == 2 and hpcp_mod.ERRORS == []
	assert _no_differences(hpcp_mod, tmp_path, new, target)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
@pytest.mark.parametrize('failure', ['error', 'silent'])
def test_failed_raw_write_stops_before_boot_code_and_identity(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys, failure):
	"""A raw partition area that cannot be written (the BIOS boot core image) must not be followed by MBR boot
	code pointing at it, or by identity changes. Nor must one written without an error that does not hold the
	new content when read back (silent): each raw write is verified before the next step."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	real_write_area = hpcp_mod._write_area
	attempted = []
	def failing(device, payload, offset, length):
		attempted.append(device)
		# Partition 2 (/dev/loopNp2), not a whole disk that happens to be /dev/loop2.
		if re.search(r'\dp2$', device):
			if failure == 'silent':
				return None
			raise OSError(errno.EIO, 'simulated write failure')
		return real_write_area(device, payload, offset, length)
	monkeypatch.setattr(hpcp_mod, '_write_area', failing)
	capsys.readouterr()
	summary = _apply(hpcp_mod, patch, target)
	assert len(attempted) == 1 and re.search(r'\dp2$', attempted[0])
	assert summary['failed'] == 1 and summary['raw_written'] == 0 and summary['identity_changed'] == 0
	assert sorted(summary['sections']) == ['part1', 'part3'] and all(s['failed'] == 0 for s in summary['sections'].values())
	captured = capsys.readouterr()
	assert 'Section part3:' in captured.out and 'Disk patch summary:' in captured.out
	reason = '[Errno 5] simulated write failure' if failure == 'error' else 'does not hold the new content after writing'
	assert [error for error in hpcp_mod.ERRORS if error.startswith('Raw area error: ')] == [
		f'Raw area error: part2: {reason}',
		'Raw area error: stopping: raw area part2 could not be written; later raw areas, the boot code and identity changes '
		'were not applied; fix the cause and re-run']
	assert len(hpcp_mod.ERRORS) == 2
	with open(str(target), 'rb') as f:
		assert f.read(14) == b'\xeb\x63OLD-BOOTCODE'
	with _attached(target, read_only=True) as loop:
		layout = hpcp_mod.read_disk_layout(loop)
		assert layout['parts'][1]['name'] == 'ESP' and layout['parts'][3]['fs_label'] == 'root'
		with open(loop + 'p2', 'rb') as f:
			assert f.read(8) == b'GRUBCORE'
	# Once the cause is fixed, a re-run completes the patch.
	monkeypatch.setattr(hpcp_mod, '_write_area', real_write_area)
	hpcp_mod.ERRORS.clear()
	again = _apply(hpcp_mod, patch, target)
	assert again['failed'] == 0 and again['raw_written'] == 2 and again['identity_changed'] == 2 and hpcp_mod.ERRORS == []
	assert _no_differences(hpcp_mod, tmp_path, new, target)


@requires_root_and_disk_tools
@pytest.mark.root
@pytest.mark.loop
def test_boot_code_not_read_back_after_writing_stops_identity_changes(tmp_path, hpcp_mod, reset_hpcp_globals, monkeypatch, capsys):
	"""The MBR boot code is read back right after its write too: one that does not hold the new content stops the
	identity changes, as a failed write does."""
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	target = tmp_path / 'node.img'
	shutil.copyfile(str(old), str(target))
	real_write_area = hpcp_mod._write_area
	attempted = []
	def dropping_boot_code(device, payload, offset, length):
		attempted.append(length)
		if length == hpcp_mod._MBR_BOOTCODE_SIZE:
			return None
		return real_write_area(device, payload, offset, length)
	monkeypatch.setattr(hpcp_mod, '_write_area', dropping_boot_code)
	capsys.readouterr()
	summary = _apply(hpcp_mod, patch, target)
	assert attempted == [1 << 20, hpcp_mod._MBR_BOOTCODE_SIZE]
	assert summary['failed'] == 1 and summary['raw_written'] == 1 and summary['identity_changed'] == 0
	assert 'Disk patch summary:' in capsys.readouterr().out
	assert hpcp_mod.ERRORS == [
		'Raw area error: mbr: does not hold the new content after writing',
		'Raw area error: stopping: raw area mbr could not be written; the identity changes were not applied; fix the cause and re-run']
	with open(str(target), 'rb') as f:
		assert f.read(14) == b'\xeb\x63OLD-BOOTCODE'
	with _attached(target, read_only=True) as loop:
		layout = hpcp_mod.read_disk_layout(loop)
		assert layout['parts'][1]['name'] == 'ESP' and layout['parts'][3]['fs_label'] == 'root'
		with open(loop + 'p2', 'rb') as f:
			assert f.read(8) == b'GRUBCOR2'
	monkeypatch.setattr(hpcp_mod, '_write_area', real_write_area)
	hpcp_mod.ERRORS.clear()
	again = _apply(hpcp_mod, patch, target)
	assert again['failed'] == 0 and again['raw_written'] == 1 and again['raw_skipped'] == 1 and again['identity_changed'] == 2
	assert hpcp_mod.ERRORS == [] and _no_differences(hpcp_mod, tmp_path, new, target)


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
def test_disk_apply_refuses_mbr_and_partitionless_targets(tmp_path, hpcp_mod, reset_hpcp_globals):
	old, new, patch = _make_patch(hpcp_mod, tmp_path)
	fs_image = tmp_path / 'fs.img'
	with open(str(fs_image), 'wb') as f:
		f.truncate(16 << 20)
	_run('mkfs.ext4', '-q', '-F', fs_image)
	with pytest.raises(RuntimeError, match='^' + re.escape(f'Invalid arguments: {_NO_TABLE.format(fs_image)}; a disk patch needs a whole GPT disk')):
		_apply(hpcp_mod, patch, fs_image)
	# Generation names the disk image it was given, not the loop device it is read through.
	with _session(hpcp_mod) as (mount_points, loop_devices):
		with pytest.raises(RuntimeError, match='^' + re.escape(f'Disk layout mismatch: {_NO_TABLE.format(fs_image)}; disk patches support GPT disks only') + '$'):
			hpcp_mod.process_get_disk_diff_image([str(new)], [str(fs_image)], str(tmp_path / 'x.img'), mount_points, loop_devices, max_workers=1)
	# sgdisk would read this MBR disk as GPT, and the GPT name change would convert it.
	mbr_image = tmp_path / 'mbr.img'
	shutil.copyfile(str(old), str(mbr_image))
	_run('sgdisk', '--gpttombr=1:2:3:4', mbr_image)
	before = _sha(mbr_image.read_bytes())
	with pytest.raises(RuntimeError, match=r'Disk layout mismatch: .* has no GPT partition table \(found dos\)'):
		_apply(hpcp_mod, patch, mbr_image)
	assert _sha(mbr_image.read_bytes()) == before
	with _session(hpcp_mod) as (mount_points, loop_devices):
		with pytest.raises(RuntimeError, match='^' + re.escape(f'Disk layout mismatch: {mbr_image} has no GPT partition table (found dos)')):
			hpcp_mod.process_get_disk_diff_image([str(new)], [str(mbr_image)], str(tmp_path / 'x.img'), mount_points, loop_devices, max_workers=1)
	assert not (tmp_path / 'x.img').exists()


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
