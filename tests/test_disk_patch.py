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
